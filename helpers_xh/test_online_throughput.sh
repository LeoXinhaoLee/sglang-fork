#!/bin/bash

export CUDA_HOME=$CONDA_PREFIX
export CUDA_VISIBLE_DEVICES=0

python3 -m sglang.bench_serving --dataset-name random \
                                --random-input-len 1024 \
                                --random-output-len 1024 \
                                --num-prompts 256 \
                                --random-range-ratio 1

# python3 -m sglang.bench_serving --dataset-name random \
#                                 --random-input-len 1024 \
#                                 --random-output-len 1024 \
#                                 --num-prompts 256 \
#                                 --random-range-ratio 1

