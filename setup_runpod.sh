#!/bin/bash

WORK_PATH=$(pwd)
HOME_PATH="/root"
cd ${HOME_PATH}

if [[ ! -d ${HOME_PATH}/miniconda ]]; then
  wget https://repo.anaconda.com/miniconda/Miniconda3-py311_25.9.1-1-Linux-x86_64.sh &&
  bash Miniconda3-py311_25.9.1-1-Linux-x86_64.sh -b -p ${HOME_PATH}/miniconda
fi
source "${HOME_PATH}/miniconda/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

cd ${WORK_PATH}
conda create -y -n sglang_fork python=3.11
conda activate sglang_fork
pip install --upgrade pip
pip install -e "python"
conda install -y -c conda-forge "cuda-toolkit=12.9"
pip install flashinfer-python==0.4.1 flashinfer-cubin==0.4.1
pip install flashinfer-jit-cache==0.4.1 --index-url https://flashinfer.ai/whl/cu129
pip install nvidia-nvshmem-cu12==3.4.5

