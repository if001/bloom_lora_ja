#!/bin/sh

# model_name = "bigscience/bloom-1b1"
# model_name = "bigscience/bloom-560m"
# model_name = "facebook/opt-6.7b"
# model_name = "facebook/opt-350m"
# model_name = "facebook/opt-1.3b"

python3 train.py \
--model "bigscience/bloom-1b1" \
--output "/content/drive/MyDrive/models/bloom_lora_ja" \
--epoch 1