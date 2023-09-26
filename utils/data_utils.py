import transformers
import torch
from datasets import Dataset, load_dataset, DatasetDict
import copy
import numpy as np

def clm_tokenize_function(examples, tokenizer, max_length=64, truncation=True, ignore_index=-100):
    # examples should from huggingface datasets that contains (instruction, answer) pairs
    # during training, the loss of instruction tokens will be ignored
    instructions = examples['instruction']
    answers = examples['answer']


    # concatenate instructions and answers
    texts = [f'{instructions[i]} {answers[i]}' for i in range(len(instructions))]
    tokenized_texts = tokenizer(texts, max_length=max_length-1, padding=False, truncation=truncation)
    # add <\s> token
    tokenized_texts['input_ids'] = [x + [tokenizer.eos_token_id] for x in tokenized_texts['input_ids']]
    tokenized_texts['attention_mask'] = [x + [1] for x in tokenized_texts['attention_mask']]
    tokenized_texts['labels'] = copy.deepcopy(tokenized_texts['input_ids'])

    # tokenize instructions to get the length of instruction tokens
    tokenized_instructions = tokenizer(instructions, max_length=max_length-1, padding=False, truncation=truncation)
    tokenized_instruction_lengths = [len(x) for x in tokenized_instructions['input_ids']]

    # zero out the loss of instruction tokens
    for i in range(len(tokenized_texts['labels'])):
        np_labels = np.array(tokenized_texts['labels'][i])
        np_labels[0:tokenized_instruction_lengths[i]] = ignore_index
        tokenized_texts['labels'][i] = np_labels.tolist()

    
    return tokenized_texts

def eval_tokenize_function(examples, tokenizer, max_length=64, truncation=True, ignore_index=-100):
    # examples should from huggingface datasets that contains (instruction, answer) pairs
    # during training, the loss of instruction tokens will be ignored
    instructions = examples['instruction']
    answers = examples['answer']


    # input_ids contain instruction only
    texts = instructions
    tokenized_texts = tokenizer(texts, max_length=max_length-1, padding=False, truncation=truncation)
    tokenized_texts['labels'] = tokenizer(answers, max_length=max_length-1, padding=False, truncation=truncation)['input_ids']

    
    return tokenized_texts


def _preprocess_text_dataset_pubmed_generation(text_dataset, prompt_template):
    """Preprocess the text dataset for pubmed abstracts."""

    if text_dataset is not None:
        answers = text_dataset['text']
        instructions = [prompt_template] * len(answers)
        # create a new huggingface dataset, with instruction and answer columns
        text_dataset = Dataset.from_dict({'instruction': instructions, 'answer': answers})
    else:
        instructions = [prompt_template] 
        answers = [''] 
        text_dataset = Dataset.from_dict({'instruction': instructions, 'answer': answers})
        # repeat the dataset after tokenization, used when all instructions are the same
        text_dataset.repeat = 2000000


    return text_dataset


def preprocess_text_dataset(dataset, dataset_name):
    if dataset_name == 'pubmed_generation':
        return _preprocess_text_dataset_pubmed_generation(dataset, prompt_template=get_prompt_template(dataset_name))
    else:
        raise f'Dataset {dataset_name} is not supported.'

def get_prompt_template(dataset_name):
    if dataset_name == 'pubmed_generation':
        # For generating pubmed abstracts, we do not need to provide any instruction. Simply use 'Abstract: ' here
        return 'Abstract: '
    else:
        raise f'Dataset {dataset_name} is not supported.'
    
class TokenizedSupervisedInstructDataset(Dataset):
    """Take a huggingface dataset that contains (insturction, answer) pairs and tokenize the concatenated text."""
    def __init__(self, dataset_or_name, tokenizer, split='train', max_length=64, truncation=True, num_proc=1, tokenize_type='clm'):

        # we shall build the text dataset from scratch
        # a processed text dataset should contain two columns: instruction and answer
        if isinstance(dataset_or_name, str):
            dataset_name = dataset_or_name
            if dataset_name == 'pubmed_generation':
                import os
                if not os.path.exists('data/pubmed_dataset_0801_0807'):
                    # unzip the file 'data/pubmed_dataset_0801_0807.zip', to data/pubmed_dataset_0801_0807
                    from zipfile import ZipFile
                    with ZipFile('data/pubmed_dataset_0801_0807.zip', 'r') as zipObj:
                        zipObj.extractall('data/')
                text_dataset = DatasetDict.load_from_disk('data/pubmed_dataset_0801_0807')
                text_dataset = text_dataset[split]
            else:
                raise ValueError(f"Dataset {dataset_name} is not supported.")
        # we alread have a text dataset, no need to build from scratch
        elif isinstance(dataset_or_name, Dataset):
            text_dataset = dataset_or_name
            dataset_name = None
        elif isinstance(dataset_or_name, DatasetDict):
            text_dataset = dataset_or_name[split]
            dataset_name = None


        self.text_dataset = text_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.num_proc = num_proc
        self.tokenize_type = tokenize_type
        self.dataset_name = dataset_name

        self.get_tokenized_dataset()


    def get_tokenized_dataset(self):
        if self.dataset_name is not None:
            processed_text_dataset = preprocess_text_dataset(self.text_dataset, self.dataset_name)
        else:
            processed_text_dataset = self.text_dataset
        # # print 10 examples
        # print(f'10 examples from text version of {self.dataset_name} dataset:')
        # for i in range(10):
        #     print(processed_text_dataset[i])
        # exit()

        # tokenize the text dataset
        if self.tokenize_type == 'clm':
            # this option concatenates the instruction and answer, and tokenize the concatenated text
            # the loss on instruction is ignored
            tokenize_func = clm_tokenize_function
        else:
            # this option only tokenizes the instruction
            # usually used during inference
            tokenize_func = eval_tokenize_function

        self.tokenized_text_dataset = processed_text_dataset.map(
            lambda x: tokenize_func(x, self.tokenizer, max_length=self.max_length, truncation=self.truncation),
            batched=True,
            num_proc=self.num_proc
        )
        if hasattr(processed_text_dataset, 'repeat'):
            # repeat input_ids, attention_mask, and labels for processed_text_dataset.repeat times
            input_ids = self.tokenized_text_dataset['input_ids'] * processed_text_dataset.repeat
            attention_mask = self.tokenized_text_dataset['attention_mask'] * processed_text_dataset.repeat
            labels = self.tokenized_text_dataset['labels'] * processed_text_dataset.repeat
            # build a huggerface dataset with the repeated input_ids, attention_mask, and labels. with the from_dict method
            self.tokenized_text_dataset = Dataset.from_dict({'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels})



    def __len__(self):
        return len(self.tokenized_text_dataset)

    def __getitem__(self, idx):
        # Return the tokenized text, attention mask, and labels
        if not isinstance(idx, list):
            idx = [idx]

        subset_dataset = self.tokenized_text_dataset[idx]
        input_ids = subset_dataset["input_ids"]
        attention_mask = subset_dataset["attention_mask"]
        labels = subset_dataset['labels']
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# Adapted from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
class DataCollatorForSupervisedDataset(object):
    IGNORE_INDEX = -100

    def __init__(self, tokenizer, padding='longest', return_tensors='pt', device='cuda', padding_side='right'):
        self.tokenizer = tokenizer
        self.padding = padding
        self.return_tensors = return_tensors
        self.device = device
        self.padding_side = padding_side

    def __call__(self, instances):
        if self.padding != 'longest':
            raise ValueError(f"Padding {self.padding} is not supported.")
        if self.return_tensors != 'pt':
            raise ValueError(f"return_tensors {self.return_tensors} is not supported.")
        

        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        if self.return_tensors == 'pt':
            input_ids = [torch.tensor(input_id).long() for input_id in input_ids]
            labels = [torch.tensor(label).long() for label in labels]
        
        if self.padding_side == 'left':
            # reverse each input_id in input_ids
            input_ids = [torch.flip(input_id, dims=[0]) for input_id in input_ids]
            labels = [torch.flip(label, dims=[0]) for label in labels]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.IGNORE_INDEX)

        if self.padding_side == 'left':
            # reverse each input_id in input_ids
            input_ids = torch.flip(input_ids, dims=[1])
            labels = torch.flip(labels, dims=[1])

        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )