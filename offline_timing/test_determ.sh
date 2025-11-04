#!/bin/bash

export CUDA_HOME=$CONDA_PREFIX

# python3 -m sglang.test.test_deterministic --host '127.0.0.1' --port 30000 --n-trials 4 --test-mode single

# python3 -m sglang.test.test_deterministic --host '127.0.0.1' --port 30000 --n-trials 16 --test-mode mixed

python3 -m sglang.test.test_deterministic --host '127.0.0.1' --port 30000 --n-trials 50 --test-mode prefix
