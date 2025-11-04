#!/bin/bash

export CUDA_HOME=$CONDA_PREFIX
export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1

# model_path="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yusu/datasets/cache_hf/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
model_path="Qwen/Qwen3-8B"

# python3 -m sglang.launch_server \
#     --model-path ${model_path} \
#     --attention-backend triton \
#     --enable-deterministic-inference \
#     --disable-radix-cache \
#     --disable-cuda-graph \
#     --batch-invariant-rms-folder "11_03_cuda"

python3 -m sglang.launch_server \
    --model-path ${model_path} \
    --attention-backend triton \
    --enable-deterministic-inference \
    --disable-radix-cache \
    --batch-invariant-rms-folder "11_03_cuda"
