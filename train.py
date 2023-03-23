'''
org: https://github.com/tloen/alpaca-lora/blob/main/finetune.py

target_module:
https://github.com/huggingface/peft/blob/13e53fc7ee5d89d59b16523051006dddf0fb7a49/src/peft/mapping.py#L41
'''
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
import sys

# OUTPUT_DIR="/content/drive/MyDrive/models/bloom_lora_ja"
# SAVE_PRE_TRAINED_DIR = "/content/drive/MyDrive/models/bloom_lora_ja"
OUTPUT_DIR=sys.argv[-1]
SAVE_PRE_TRAINED_DIR=sys.argv[-1]

# optimized for RTX 4090. for larger GPUs, increase some of these?
MICRO_BATCH_SIZE = 4  # this could actually be 5 but i like powers of 2
# MICRO_BATCH_SIZE = 2  # this could actually be 5 but i like powers of 2

# BATCH_SIZE = 128
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
# EPOCHS = 3  # we don't always need 3 tbh
EPOCHS = 1  # we don't always need 3 tbh

LEARNING_RATE = 3e-4  # the Karpathy constant
CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 2000
# for llama or opt
# TARGET_MODULES = [
#     "q_proj",
#     "v_proj",
# ]
# for bloom
TARGET_MODULES = [
    "query_key_value",    
]

# DATA_PATH = "alpaca_data_cleaned.json"
DATA_PATH = 'japanese_alpaca_data.json'

device_map = "auto"
world_size = int(os.environ.get('WORLD_SIZE', 1))
ddp = world_size != 1
if ddp:
    device_map = {'':int(os.environ.get('LOCAL_RANK') or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

# model_name = "decapoda-research/llama-7b-hf"
# model = LlamaForCausalLM.from_pretrained(
#     model_name,
#     load_in_8bit=True,
#     device_map=device_map,
# )
# tokenizer = LlamaTokenizer.from_pretrained(
#     model_name,add_eos_token=True    
# )

model_name = "bigscience/bloom-1b1"
# model_name = "bigscience/bloom-560m"
# model_name = "facebook/opt-6.7b"
# model_name = "facebook/opt-350m"
# model_name = "facebook/opt-1.3b"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map=device_map,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,add_eos_token=True
)


model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
data = load_dataset("json", data_files=DATA_PATH)

train_val = data["train"].train_test_split(
    test_size=VAL_SET_SIZE, shuffle=True, seed=42
)
train_data = train_val["train"]
val_data = train_val["test"]

def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""以下は、タスクを説明する命令と、さらなるコンテキストを提供する入力の組み合わせです。要求を適切に満たすような応答を書きなさい。

### 命令:
{data_point["instruction"]}

### 入力:
{data_point["input"]}

### 応答:
{data_point["output"]}"""
    else:
        return f"""以下は、ある作業を記述した指示です。要求を適切に満たすような応答を書きなさい。

### 入力:
{data_point["instruction"]}

### 応答:
{data_point["output"]}"""


def tokenize(prompt):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }


def generate_and_tokenize_prompt(data_point):
    prompt = generate_prompt(data_point)
    return tokenize(prompt)



if VAL_SET_SIZE > 0:
    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
else:
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = None

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=20,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        output_dir=OUTPUT_DIR,
        save_total_limit=3,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

trainer.train()

model.save_pretrained(SAVE_PRE_TRAINED_DIR)

print("\n If there's a warning about missing keys above, please disregard :)")