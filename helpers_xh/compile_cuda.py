import os
import uuid
from torch.utils.cpp_extension import load_inline


def compile_cuda(name: str, code: str, compile_path: str):
    os.environ["FORCE_CUDA"] = "1"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"  # H100
    COMPILE_FLAGS = [
        "-O3",
        "--use_fast_math",
        "-gencode=arch=compute_90,code=sm_90",
    ]
    CUTLASS_PATH = os.path.expanduser('~/cutlass/include')

    load_inline(
        name=name,
        cpp_sources=[],
        cuda_sources=[code],
        extra_cuda_cflags=COMPILE_FLAGS,
        extra_include_paths=[CUTLASS_PATH],
        with_cuda=True,
        verbose=True,
        build_directory=compile_path,
    )


if __name__ == "__main__":
    from codetiming import Timer
    name = "batch_invariant_rmsnorm"
    # code_path = "/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/yusu/code/xh/sglang-fork/python/sglang/srt/batch_invariant_ops/rms_norm/11_03_cuda"
    # code_path = "/root/sglang-fork/python/sglang/srt/batch_invariant_ops/rms_norm/11_03_cuda"
    code_path = "/root/sglang-fork/python/sglang/srt/batch_invariant_ops/rms_norm/11_04_cuda"
    with open(code_path + '/kernel.cu', 'r') as f:
        code = f.read()
    unique_id = str(uuid.uuid4())[:8]
    ext_dir = code_path
    os.makedirs(ext_dir, exist_ok=True)
    print(ext_dir)
    with Timer(name="compile", logger=None) as timer:
        compile_cuda(
            name=name,
            code=code,
            compile_path=ext_dir,
        )
    print(f"Time: {timer.last}s")  # 45s
