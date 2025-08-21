#!/bin/bash

MODEL="/tmp/base_model"
# MODEL="Qwen/Qwen2.5-7B-Instruct"
DISTRIBUTED_ARGS="--nproc_per_node $SM_NUM_GPUS --nnodes $NODE_NUMBER --node_rank $NODE_INDEX --master_addr $SM_MASTER_ADDR --master_port 12345"

# 22GB

swift sft \
    --model ${MODEL} \
    --model_type qwen2_5 \
    --template qwen2_5 \
    --train_type full \
    --dataset /opt/ml/input/data/train/example_data.json \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir /opt/ml/checkpoints \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
