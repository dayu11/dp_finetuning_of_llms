import argparse, json, logging, math, os, random, sys
os.environ["WANDB_DISABLED"] = "true"

import datasets
import torch
import torch.nn as nn
from accelerate import Accelerator, DistributedType
from accelerate import FullyShardedDataParallelPlugin
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, SchedulerType, get_scheduler

import utils.lora_utils as lora_utils 
import utils.data_utils as data_utils
import utils.dp_utils as dp_utils



logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune LLMs with DP-SGD.")
    parser.add_argument("--dataset_name", type=str, default=None, help="The name of the dataset to use (via the datasets library).")
    parser.add_argument("--train_file", type=str, default=None, help='Specify the customized training file.')
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, required=True, help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="cosine", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--block_size", type=int, default=None, help="Max sequence length after tokenization. Sequences longer than this will be truncated.")
    parser.add_argument("--eval_freq", type=int, default=100, help="Freq of generation and loss logging.")
    parser.add_argument("--with_tracking", action="store_true", help="Whether to enable experiment trackers for logging.")    
    parser.add_argument("--access_token", type=str, default=None, help='Huggingface access token')

    # training hyperparameters
    parser.add_argument("--gradient_ckpt", action="store_true", help="Use gradient checkpointing or not. If true, will drop some forward activations and re-compute them during backward. This will save memory but slow down training.")    

    # lora hyperparameters
    parser.add_argument("--lora_r", type=int, default=16, help="Rank of LoRA fine-tuning.")
    parser.add_argument("--lora_alpha", type=float, default=16, help="Value of alpha for lora.")

    # dp hyperparameters
    parser.add_argument("--delta", type=float, default=1e-7, help="Privacy parameter delta.")
    parser.add_argument("--clip_norm", type=float, default=-1, help="Clip norm for DP-SGD. If negative, there will be no per-example gradient computation and clipping.")
    parser.add_argument("--noise_multiplier", type=float, default=-1, help="Noise multiplier for DP-SGD. If negative, no noise will be added.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    assert args.dataset_name is None or args.train_file is None, 'Only one of dataset_name and train_file should be specified. If you want to use built-in datasets, specify dataset_name. If you want to use customized datasets, specify train_file.'
    assert (args.clip_norm <= 0 and args.noise_multiplier <= 0) or (args.clip_norm > 0 and args.noise_multiplier > 0), 'Currently only support clip_norm and noise_multiplier both >0 or <= 0'


    # If passed along, set the training seed now.
    if args.seed is not None:
        from accelerate.utils import set_seed
        set_seed(args.seed)
    
    # Login to huggingface to get access to llama2
    access_token = args.access_token
    if access_token is not None:
        os.system('huggingface-cli login --token ' + access_token)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="right")


    # add padding token to the tokenizer
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))


    # replace 'q', 'k', 'v', 'o' in attention layers with lora linear layers
    lora_utils.make_lora(model, model_name=args.model_name_or_path, r=args.lora_r, alpha=args.lora_alpha, lora_weights=['q', 'k', 'v', 'o'])

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["project_dir"] = args.output_dir

    # Initialize accelerator, and do not use fsdp for lora parameters
    lora_params = [p[1] for p in model.named_parameters() if 'lora_branch' in p[0]]
    fsdp_plugin = FullyShardedDataParallelPlugin()
    fsdp_plugin.ignored_parameters = lora_params
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin, **accelerator_log_kwargs)


    num_trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f'Make llama model lora with r={args.lora_r}, alpha={args.lora_alpha}, num trainable params={num_trainable_p/1e6}M')

    logical_batch_size = accelerator.num_processes * args.per_device_train_batch_size * args.gradient_accumulation_steps
    dp_enabled = args.clip_norm > 0 and args.noise_multiplier > 0
    if dp_enabled:
        # register hooks for per-example gradient computation
        dp_utils.make_lora_model_dp(model)
        
        if args.dataset_name is not None: # use built-in datasets
            if args.dataset_name == 'pubmed_generation':
                num_training_points = 75329
            else:
                raise NotImplementedError
        else: # use customized datasets
            num_training_points = len(json.load(open(args.train_file, 'r'), strict=False)['instruction'])

        sampling_prob = logical_batch_size / num_training_points
        num_train_steps = int(args.num_train_epochs / sampling_prob) + 1
        eps_low = dp_utils.get_epsilon_prv(args.noise_multiplier, args.delta, num_train_steps, sampling_prob)
        
        accelerator.print(f'Sampling prob {sampling_prob:.4f}, epsilon {eps_low:.2f}, delta {args.delta}')


    if args.gradient_ckpt:
        accelerator.print('Use gradient checkpointing')
        model.gradient_checkpointing_enable()  
        model.enable_input_require_grads()

    model = accelerator.prepare(model)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.dataset_name is not None:
        # Simply pass a string to TokenizedSupervisedInstructDataset(). We will build a Dataset object in the function
        dataset_object_or_str = args.dataset_name
    else:
        # Load files and bulid a huggingface Dataset object, and pass it to TokenizedSupervisedInstructDataset()
        instruction_answer_dict = json.load(open(args.train_file, 'r'), strict=False)
        assert 'instruction' in instruction_answer_dict and 'answer' in instruction_answer_dict
        assert len(instruction_answer_dict['instruction']) == len(instruction_answer_dict['answer'])
        dataset_object_or_str = datasets.Dataset.from_dict(instruction_answer_dict)

    # Create tokenized training dataset
    train_dataset = data_utils.TokenizedSupervisedInstructDataset(dataset_object_or_str, tokenizer=tokenizer, split='train', max_length=args.block_size, truncation=True, num_proc=1)
    
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    data_collator = data_utils.DataCollatorForSupervisedDataset(tokenizer, padding='longest', return_tensors="pt", device=accelerator.device)

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

    # only update lora parameters
    require_grad_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(require_grad_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    # Prepare optimizer, train_dataloader with accelerator. It will handle distributed batch splitting, automated mixed precision scaling, ...
    optimizer, train_dataloader = accelerator.prepare(
        optimizer, train_dataloader
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    # get lr scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )



    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("train_clm", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    # Initialize variables for tracking training states
    accumulated_steps = 0
    recent_loss = torch.zeros(1, device=accelerator.device)
    total_loss = torch.zeros(1, device=accelerator.device)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # We shall sync gradients by ourself, instead of using accelerator.accummulate
            with accelerator.no_sync(model):
                
                outputs = model(**batch)
                loss = outputs.loss
                # Track of the loss of the most recent updates
                recent_loss += loss.detach()
                # Track of the loss of the entire epoch
                total_loss += loss.detach()
                    
                accelerator.backward(loss)
                
                if dp_enabled:
                    # Clip per-example gradients, and sum to .accumulated_grad
                    grad_norms = dp_utils.clip_and_accumulate_perexample_grads(require_grad_params, accumulated_steps, args.clip_norm, accelerator)

                accumulated_steps += 1
                if accumulated_steps == args.gradient_accumulation_steps:
                    # Sync gradients
                    if not dp_enabled:
                        # Undo mixed precision scaling. 
                        accelerator.unscale_gradients(optimizer=optimizer)
                        # Undo batch averaging
                        for p in require_grad_params:
                            p.grad = p.grad * args.per_device_train_batch_size
                        # When dp_enabled == True, the above two steps are done in `clip_and_accumulate_perexample_grads`

                        grads_to_sync = [p.grad for p in require_grad_params]
                    else:
                        grads_to_sync = [p.accumulated_grad for p in require_grad_params]
                    synced_grads = accelerator.gather(grads_to_sync)
                    synced_grads = [g.view(accelerator.num_processes, *p.shape) for g, p in zip(synced_grads, require_grad_params)]
                    synced_grads = [torch.sum(g, dim=0) for g in synced_grads]


                    if dp_enabled:
                        if accelerator.is_main_process:
                            noises = []
                            for g in synced_grads:
                                noises.append(torch.normal(0, args.clip_norm * args.noise_multiplier, size=g.shape, device=g.device, dtype=g.dtype))
                        else:
                            noises = [torch.zeros_like(g) for g in synced_grads]
                        # synchronize noise
                        noises = accelerator.reduce(noises, reduction='sum', scale=1.0)
                        # add noise to gradients
                        synced_grads = [g + n for g, n in zip(synced_grads, noises)]
                        
                    # Average over whole batch
                    synced_grads = [g / logical_batch_size for g in synced_grads]
                    # Assign synced grads to model
                    for g, p in zip(synced_grads, require_grad_params):
                        p.grad = g

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    performed_optimizer_step = True
                    accumulated_steps = 0
                else:
                    performed_optimizer_step = False

                if performed_optimizer_step:
                    
                    progress_bar.update(1)
                    completed_steps += 1
                    sys.stdout.flush()
                    if (completed_steps) % args.eval_freq == 0:
                        
                        current_loss = recent_loss / (args.eval_freq*args.gradient_accumulation_steps)
                        accelerator.print(f'epoch {epoch}, completed steps {completed_steps}, train loss {current_loss.item()}, current lr {lr_scheduler.get_last_lr()[0]}')
                        sys.stdout.flush()
                        recent_loss = 0

                    if completed_steps >= args.max_train_steps:
                        break
                    
        epoch_loss = total_loss.item() / len(train_dataloader)
        if args.with_tracking:
            accelerator.print(f'epoch {epoch}, train loss {epoch_loss}')
            accelerator.log(
                {
                    "train_loss": epoch_loss,
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )
        total_loss = 0

    if args.with_tracking:
        accelerator.end_training()
    
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        lora_state_dict = {}
        for p in model.module.named_parameters():
            if ('lora_branch' in p[0]):
                lora_state_dict[p[0]] = p[1]

        if accelerator.is_main_process:
            torch.save(lora_state_dict, os.path.join(args.output_dir, "lora_weights.bin"))
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"train_loss": epoch_loss}, f)

if __name__ == "__main__":
    main()
