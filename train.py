# Importing required libraries
import os
import transformers

# required imports from the transformers library

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    HfArgumentParser
)
from datasets import load_dataset
import torch

import bitsandbytes as bnb
from huggingface_hub import login, HfFolder

from trl import SFTTrainer

from utils import print_trainable_parameters, find_all_linear_names

from train_args import ScriptArguments

from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, prepare_model_for_kbit_training

# Setting up for command line argument parsing
parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

# the main training function
def training_function(args):

    # Authentication token for using Hugging Face services 
    login(token=args.hf_token)

    # Setting random seed 
    set_seed(args.seed)

    # Loading the dataset from the specified path
    data_path=args.data_path

    dataset = load_dataset(data_path)

    # Configuring bitsandbytes for 4-bit quantization, thats used to save memory and improve performance
    bnb_config = BitsAndBytesConfig(
        #  settings configuration for bitsandbytes
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # Loading the causal language model 
    model = AutoModelForCausalLM.from_pretrained(
        # Model loading setting
        args.model_name,
        use_cache=False,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    # Loading the tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # setting up the pad token
    tokenizer.pad_token=tokenizer.eos_token
    #  configuring padding
    tokenizer.padding_side='right'

    # Preparing  model for k-bit training 
    model=prepare_model_for_kbit_training(model)

    # Find all the names in the model for linear layers in targeted adaptation
    modules=find_all_linear_names(model)
    config = LoraConfig(

        # Configuration for LoRA adaptation
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=modules
    )

    # Wrapping the model with PEFT model configuration
    model=get_peft_model(model, config)

    output_dir = args.output_dir

    # Setting up training arguments based on user-provided values
    training_arguments = TrainingArguments(
        # Configuration for the process training
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        bf16=False,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=args.lr_scheduler_type,
        tf32=False,
        report_to="none",
        push_to_hub=False,
        max_steps = args.max_steps
    )

    # Initialising the trainer with the configured model and training arguments
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        dataset_text_field=args.text_field,
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_arguments
    )

    # Moving the normalization layers for stability in training
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32) # moving to float 32
    #printing the starting messade
    print('starting training')

    # Initiating the training
    trainer.train()

    # displaying completion message  
    print('LoRA training complete')

    # pushing the LoRA-adapted model to Hugging Face Hub
    lora_dir = args.lora_dir
    trainer.model.push_to_hub(lora_dir, safe_serialization=False)
    
    print("saved lora adapters")

    
#  script execution starts here
if __name__=='__main__':
    training_function(args)

