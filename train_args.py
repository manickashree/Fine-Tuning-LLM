## Importing the requiired libraries
from dataclasses import dataclass, field
import os
from typing import Optional

@dataclass
class ScriptArguments:
    # Hugging Face authentication token
    hf_token: str = field(metadata={"help": ""})

    # The model name that's used in hugging face.
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf", metadata={"help": ""}
    )
    # setting the seed for random number generator
    seed: Optional[int] = field(
        default=4761, metadata = {'help':''}
    )
    # Path to the data file
    data_path: Optional[str] = field(
        default="./data/forums_short.json", metadata={"help": ""}
    )
    # Output directory
    output_dir: Optional[str] = field(
        default="output", metadata={"help": ""}
    )
    # Batch size required per device during training   
    per_device_train_batch_size: Optional[int] = field(
        default = 2, metadata = {"help":""}
    )

    # The gradient steps required before backward pass
    gradient_accumulation_steps: Optional[int] = field(
        default = 1, metadata = {"help":""}
    )

    # training optimizer 
    optim: Optional[str] = field(
        default = "paged_adamw_32bit", metadata = {"help":""}
    )

    # saving frequency of the model checkpoint
    save_steps: Optional[int] = field(
        default = 25, metadata = {"help":""}
    )

    # logging training information frequency
    logging_steps: Optional[int] = field(
        default = 1, metadata = {"help":""}
    )

   # Initialising the learning rate for training purpose
    learning_rate: Optional[float] = field(
        default = 2e-4, metadata = {"help":""}
    )

    # Maximum gradient norm 
    max_grad_norm: Optional[float] = field (
        default = 0.3, metadata = {"help":""}
    )

    # number of training epochs to be performed in total
    num_train_epochs: Optional[int] = field (
        default = 1, metadata = {"help":""}
    ) 

    # Warmup ratio 
    warmup_ratio: Optional[float] = field (
        default = 0.03, metadata = {"help":""}
    )

    # learning rate scheduler type
    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata = {"help":""}
    ) 

    # Directory to store the LoRA adaptation layers
    lora_dir: Optional[str] = field(default = "./model/llm_hate_speech_lora", metadata = {"help":""})

    #  overriding epochs to train for set number of steps if its positive
    max_steps: Optional[int] = field(default=-1, metadata={"help": ""})

    # Field name to be used as input for the model
    text_field: Optional[str] = field(default='chat_sample', metadata={"help": ""})


