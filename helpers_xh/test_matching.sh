#!/bin/bash

export CUDA_HOME=$CONDA_PREFIX
export CUDA_VISIBLE_DEVICES=0

model_path="/workspace/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/"

python3 -m sglang.bench_offline_throughput_debug --model-path ${model_path} \
                                                 --dataset-name random \
                                                 --random-range-ratio 1 \
                                                 --random-input-len 32 \
                                                 --random-output-len 32 \
                                                 --num-prompts 1 \
                                                 --skip-warmup \
                                                 --attention-backend triton \
                                                 --disable-radix-cache \
                                                 --stats-folder-name "11_03/normal/r128p512d512"

# python3 -m sglang.bench_offline_throughput_debug --model-path ${model_path} \
#                                                  --dataset-name random \
#                                                  --random-range-ratio 1 \
#                                                  --random-input-len 512 \
#                                                  --random-output-len 512 \
#                                                  --num-prompts 128 \
#                                                  --skip-warmup \
#                                                  --attention-backend triton \
#                                                  --disable-radix-cache \
#                                                  --enable-deterministic-inference \
#                                                  --batch-invariant-rms-folder "sglang" \
#                                                  --stats-folder-name "11_03/sglang/r128p512d512"

# python3 -m sglang.bench_offline_throughput_debug --model-path ${model_path} \
#                                                  --dataset-name random \
#                                                  --random-range-ratio 1 \
#                                                  --random-input-len 512 \
#                                                  --random-output-len 512 \
#                                                  --num-prompts 128 \
#                                                  --skip-warmup \
#                                                  --attention-backend triton \
#                                                  --disable-radix-cache \
#                                                  --enable-deterministic-inference \
#                                                  --batch-invariant-rms-folder "11_03_cuda" \
#                                                  --stats-folder-name "11_03/our_rms_cuda/r128p512d512"
