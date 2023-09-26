import torch
import argparse
import datasets
from accelerate import Accelerator, DistributedType
from accelerate import FullyShardedDataParallelPlugin

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from datasets import Dataset

from torch.utils.data import DataLoader

from utils.data_utils import TokenizedSupervisedInstructDataset, DataCollatorForSupervisedDataset
from utils.lora_utils import make_lora, load_weights


import os
import sys

def generate_batch_with_input_ids(model, input_ids, attention_mask, accelerator, tokenizer, output_fname='generations', min_length=0, max_length=100, top_k=50, top_p=1, temperature=1, repetition_penalty=1):
    try:
        output = model.generate(input_ids, attention_mask=attention_mask, do_sample=True, min_length=min_length, max_length=max_length, top_k=top_k, top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty)
        generated = tokenizer.batch_decode(output, skip_special_tokens=True)
    except:
        generated = ['']
    
    generated = [g.strip() for g in generated]

    local_rank = accelerator.process_index
    output_str = u'Generated:\n' + u'\n'.join(generated)
    output_str = output_str.encode('utf-8').strip()
    with open(f'{output_fname}_gpu{local_rank}.txt', 'ab') as f:
        f.write(output_str)
        f.write(u'\n'.encode('utf-8'))

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a model for generation.")
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--lora_weights_path", default=None, type=str, help="Path to lora weights")
    parser.add_argument("--instruction_file", type=str, default=None, help="Dataset name.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--max_length", type=int, default=256, help="Max length of the generated sequence.")
    parser.add_argument("--output_dir", type=str, default='inference_generation', help="Output directory.")

    # generation config
    parser.add_argument("--top_k", type=int, default=50, help="top k words for generation")
    parser.add_argument("--top_p", type=float, default=1.0, help="top p probability")
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="The parameter for repetition penalty. 1.0 means no penalty.")

    parser.add_argument(
        "--access_token",
        type=str,
        default=None,
        help='Huggingface access token',
    )

    args = parser.parse_args()

    return args


## The prompt template that LLaMA 2 uses during its instrution tuning, according to https://huggingface.co/blog/llama2#using-transformers
## Not used here, but may be useful in the future
# SYSTEM_PROMPT = """
# [INST] <<SYS>>
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
# <</SYS>>

# {} [/INST] 
# """




def main():
    args = parse_args()

    assert args.instruction_file is not None, "Please specify the instruction file."
    assert args.batch_size == 1, "Currently only support generation with batchsize = 1."

    # huggingface access token
    access_token = args.access_token

    if access_token is not None:
        os.system('huggingface-cli login --token ' + access_token)


    accelerator = Accelerator()
    if accelerator.state.mixed_precision == 'fp16':
        load_dtype = torch.float16
    elif accelerator.state.mixed_precision == 'bf16':
        load_dtype = torch.bfloat16
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config, torch_dtype=load_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Add a [PAD] token to the tokenizer
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if args.lora_weights_path is not None:
        make_lora(model, model_name=args.model_name_or_path, r=16, alpha=16, lora_weights=['q', 'k', 'v', 'o'])
        load_weights(model, args.lora_weights_path)
    
    for p in model.parameters():
        p.requires_grad = False



    model = accelerator.prepare(model)

    # build a text dataset from the instruction file
    with open(args.instruction_file, 'r') as f:
        lines = f.readlines()
        instructions = []
        for line in lines:
            _instruction = line.strip()
            if len(_instruction) > 4:
                _instruction, _repeat = line.split('####')
                _instruction = _instruction.strip()
                _repeat = int(_repeat)
                instructions.extend([_instruction]*_repeat)
        if len(instructions) < 1:
            raise 'empty instruction file'
        answers = [""] * len(instructions)
        text_dataset = Dataset.from_dict({'instruction': instructions, 'answer': answers})

    dataset = TokenizedSupervisedInstructDataset(text_dataset, tokenizer=tokenizer, max_length=args.max_length, truncation=True, num_proc=1, tokenize_type='eval')

    data_collator = DataCollatorForSupervisedDataset(tokenizer, padding='longest', return_tensors="pt", device=accelerator.device)

    # DataLoaders creation:
    eval_data_loader = DataLoader(
        dataset, shuffle=False, collate_fn=data_collator, batch_size=args.batch_size
    )
    eval_data_loader = accelerator.prepare(eval_data_loader)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    model.eval()
    for batch in eval_data_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        generate_batch_with_input_ids(model, input_ids, attention_mask, accelerator, tokenizer, max_length=args.max_length, output_fname=args.output_dir+'/generations', top_k=args.top_k, top_p=args.top_p, temperature=args.temperature, repetition_penalty=args.repetition_penalty)
        sys.stdout.flush()


if __name__ == "__main__":
    main()