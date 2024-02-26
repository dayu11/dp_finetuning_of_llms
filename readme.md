# Differentially Private Fine-tuning of LLMs

## Intro


This repo intends to provide a lightweight code for fine-tuning large language models (>1B parameters) with DP.

To the best of my knowledge, using model parallelism is a non-trivial task in existing implementations of DP training. This makes DP training of LLMs less accessible: we need model parallelism because the model is often too large to fit in a single GPU. This code uses Huggingface [Accelerate](https://huggingface.co/docs/accelerate/index) to support Fully Sharded Data Parallel (FSDP) for the main branch parameters. In addition to that, this code uses [LoRA](https://arxiv.org/abs/2106.09685), which is an instance of parameter-efficient fine-tuning algorithms, to further reduce the computational cost. LoRA keeps the main branch weights frozen and only fine-tunes a small number of adapter parameters. Note that FSDP is only used for main branch parameters and not for LoRA parameters. The  primitives for DP training, i.e., per-example gradient computation, clipping, and noising are implemented in ```utils/dp_utils.py```.


## Prerequisite

Build the environment.

```
pip install -r requirements.txt
```

## Fine-tune Llama 2 to generate abstracts of medical papers

The example command shows how to fine-tune [Llama 2 7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) from Huggingface transformers. You should be able to use other models from Huggingface with some simple modifications to the code. The dataset ```data/pubmed_dataset_0801_0807.zip``` contains >75000 abstracts of medical papers that were published at [PubMed](https://pubmed.ncbi.nlm.nih.gov/) during 2023/08/01 - 2023/08/07. The pre-training data of Llama 2 has a cutoff date of September 2022.

For fine-tuning Llama 2 7B with a context length of 512 and LoRA rank=16:

**Minimum hardware** : 1 x 48GA6000 

**Recommended hardware**: 1 x 80GA100 or 4 x 48GA6000 or 16 x 32GV100

Below is an example command with 1 x 80GA100. If you are using a single A6000, add ```--gradient_ckpt``` to further reduce memory usage (at a cost of longer training time).


```
accelerate launch --config_file accelerate_configs/accelerate_config.cfg train_clm.py --dataset_name pubmed_generation --model_name_or_path meta-llama/Llama-2-7b-hf --per_device_train_batch_size 4 --output_dir private_pubmed_abstract_generation --block_size 512 --eval_freq 10 --gradient_accumulation_steps 512 --clip_norm 1 --noise_multiplier 1.2 --num_train_epochs 5 --learning_rate 3e-4 --num_warmup_steps 50 --weight_decay 0.01 --with_tracking --access_token {huggingface_access_token}
```

The fine-tuned LoRA weights will be saved at ```private_pubmed_abstract_generation/lora_weights.bin```.  You can change the number of GPUs by changing ```num_processes``` in ```accelerate_configs/accelerate_config.cfg```. You may also need to change ```--per_device_train_batch_size``` and ```--block_size``` to adapt to other hardware settings. The default mixed precision choice is ```bf16``` that is supported by A6000/A100. You need to change it to ```fp16``` in ```accelerate_config.cfg``` if you are using V100s.

Here are some important arguments of ```train_clm.py```.

1. ```--model_name_or_path```, pre-trained model name, 
2. ```--block_size```, max sequence length, 
3. ```--clip_norm```, **per-example gradient clipping (<0 means no clip),**
4. ```--noise_multiplier```, **noise multiplier (<0 means no noise),**
5. ```--access_token```, Huggingface access token. Required if you want to fine-tune Llama 2. You can request access to Llama 2 at [here](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).



**If you want to fine-tune without DP, you can use a smaller ```num_train_epochs``` and ```gradient_accumulation_steps```.** For instance, set ```--clip_norm -1 --noise_multiplier -1 --gradient_accumulation_steps 16 --num_train_epochs 3 --learning_rate 2e-5```

## Run inference to generate abstracts

Minimum hardware: 1 x 32GV100

Specify the path of LoRA weights with the ```--lora_weights_path``` argument. You can find the outputs in the ```generations``` folder.

```
accelerate launch  --config_file accelerate_configs/accelerate_config_nofsdp.cfg  inference_generation_dataset.py   --dataset_name pubmed_generation     --model_name_or_path meta-llama/Llama-2-7b-hf  --lora_weights_path {path_to_lora_weights}/lora_weights.bin --max_length 512 --output_dir generations --access_token {huggingface_access_token}
```


## Fine-tune with customized data

To fine-tune the model with a customized dataset, first create a Python dictionary that contains 'instruction' and 'answer'. If the target task is simply unconditional generation, the ```instruction``` could be a list of empty strings.
```
python_dict = {'instruction': [....], 'answer': [....]}
```

Save the python dict to a ```.json``` file and pass the path to the ```--train_file``` argument of ```train_clm.py```. Note that you need to remove the `--dataset_name` argument in the above command. E.g.,

```
accelerate launch --config_file accelerate_configs/accelerate_config.cfg train_clm.py --train_file <path to json file> --model_name_or_path meta-llama/Llama-2-7b-hf --per_device_train_batch_size 4 --output_dir {output_path} --block_size 512 --eval_freq 10 --gradient_accumulation_steps 16 --clip_norm -1 --noise_multiplier -1 --num_train_epochs 3 --learning_rate 2e-5 --num_warmup_steps 100 --weight_decay 0.01 --with_tracking --access_token {huggingface_access_token}
```

You can change ```--model_name_or_path``` to ```meta-llama/Llama-2-7b-chat-hf``` if you want to fine-tune the chat version of Llama 2.


## Run inference with customized instructions

Specify the instructions you need in a .txt file. I provide an example in ```scripts/example_instruction_file.txt```. The string before ```####``` is the instruction and the string after that is the number of repetitions.

Specify the path of LoRA weights with the ```--lora_weights_path```. If ```--lora_weights_path``` is not given, we simply use the official checkpoint from huggingface.

```
accelerate launch --config_file accelerate_configs/accelerate_config_nofsdp.cfg inference_generation_instruction.py --model_name_or_path meta-llama/Llama-2-7b-hf --lora_weights_path {path_to_lora_weights}/lora_weights.bin --instruction_file scripts/example_instruction_file.txt --max_length 512 --output_dir generation_test --access_token {huggingface_access_token}
``` 

## Some implementation details

Per-example gradient computation of LoRA layers is done with Pytorch forward/backward hooks (see the implementation of hooks in ```utils/dp_utils.py```). You can also find the code for gradient clipping/accumulation/noising in ```utils/dp_utils.py```. 

In ```utils/lora_utils.py```, you can find the implementation of wrapping attention linear layers into LoRA layers.

Gradient sync across GPUs is done with ```accelerator.gather```. See Line 390 in ```train_clm.py```.

## Reference

If you find this repo useful, please consider citing

```
@inproceedings{yu2022differentially,
  title={Differentially private fine-tuning of language models},
  author={Yu, Da and Naik, Saurabh and Backurs, Arturs and Gopi, Sivakanth and Inan, Huseyin A and Kamath, Gautam and Kulkarni, Janardhan and Lee, Yin Tat and Manoel, Andre and Wutschitz, Lukas and others},
  year = {2022},
  booktitle = {International Conference on Learning Representations (ICLR)}
}

@inproceedings{yu2023training,
  title={Training Private and Efficient Language Models with Synthetic Data from LLMs},
  author={Yu, Da and Backurs, Arturs and Gopi, Sivakanth and Inan, Huseyin and Kulkarni, Janardhan and Lin, Zinan and Xie, Chulin and Zhang, Huishuai and Zhang, Wanrong},
  booktitle={Socially Responsible Language Modelling Research},
  year={2023}
}
```

