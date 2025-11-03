# Adapted from https://github.com/thinking-machines-lab/batch_invariant_ops/blob/main/batch_invariant_ops/batch_invariant_ops.py

import contextlib
from collections import namedtuple
import os
import os.path as osp
import glob
import importlib.util

import torch

# if ENABLE_JIT_DEEPGEMM:
#     import deep_gemm

# _ENABLE_MM_DEEPGEMM = get_bool_env_var(
#     "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM", "1"
# )
# _ENABLE_MM_COMPARISON_TEST = get_bool_env_var(
#     "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_COMPARISON_TEST"
# )

# if not _ENABLE_MM_DEEPGEMM:
#     print("Disable DeepGEMM in batch invariant ops. Performance may be suboptimal.")

__all__ = [
    "set_batch_invariant_mode",
    "is_batch_invariant_mode_enabled",
    "disable_batch_invariant_mode",
    "enable_batch_invariant_mode",
]


try:
    folder_name = os.getenv("SGLANG_DETERMINISTIC_RMSNORM_PATH", "sglang")
    kernel_dir = osp.join("./python/sglang/srt/batch_invariant_ops", "rms_norm", folder_name)
    kernel_files = glob.glob(osp.join(kernel_dir, "*"))
    kernel_files = [f for f in kernel_files if f.endswith((".py", ".so"))]
    assert len(kernel_files) == 1
    kernel_file = kernel_files[0]
    spec = importlib.util.spec_from_file_location(f"batch_invariant_rmsnorm", kernel_file)
    _mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_mod)
    print(f"Importing deterministic RMSNorm from {kernel_file}")
    rms_norm_batch_invariant = _mod.fn
except Exception as e:
    raise ImportError(f"Failed to import backend '{kernel_file}': {e}") from e


_batch_invariant_MODE = False
_batch_invariant_LIB = None
_original_torch_bmm = None


def is_batch_invariant_mode_enabled():
    return _batch_invariant_MODE


def load_kernel(name: str, ops_dir, ops_folder):
    kernel_dir = osp.join(
        ops_dir,    # batch_invariant_ops/
        name,       # mm/
        ops_folder  # sglang/
    )
    pattern = osp.join(kernel_dir, "kernel*")
    kernel_files = glob.glob(pattern)
    assert len(kernel_files) == 1, "There should be only 1 kernel file under a dir"
    kernel_file = kernel_files[0]
    spec = importlib.util.spec_from_file_location(f"batch_invariant_{name}", kernel_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def enable_batch_invariant_mode(
    server_args, enable_bmm: bool = True,
):
    global _batch_invariant_MODE, _batch_invariant_LIB, _original_torch_bmm
    if _batch_invariant_MODE:
        return

    _batch_invariant_MODE = True
    osp_dir = server_args.batch_invariant_ops_dir
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")
    _batch_invariant_LIB.impl("aten::mm", load_kernel("mm", osp_dir, server_args.batch_invariant_mm_folder).fn, "CUDA")
    _batch_invariant_LIB.impl("aten::addmm", load_kernel("addmm", osp_dir, server_args.batch_invariant_addmm_folder).fn, "CUDA")
    _batch_invariant_LIB.impl("aten::_log_softmax", load_kernel("log_softmax", osp_dir, server_args.batch_invariant_log_softmax_folder).fn, "CUDA")
    _batch_invariant_LIB.impl("aten::mean.dim", load_kernel("mean", osp_dir, server_args.batch_invariant_mean_folder).fn, "CUDA")
    if enable_bmm:
        _batch_invariant_LIB.impl("aten::bmm", load_kernel("mm", osp_dir, server_args.batch_invariant_mm_folder).bmm_batch_invariant, "CUDA")
        # Also monkeypatch torch.bmm directly as a fallback
        _original_torch_bmm = torch.bmm
        torch.bmm = load_kernel("mm", osp_dir, server_args.batch_invariant_mm_folder).bmm_batch_invariant


def disable_batch_invariant_mode():
    global _batch_invariant_MODE, _batch_invariant_LIB, _original_torch_bmm
    if _batch_invariant_LIB is not None:
        _batch_invariant_LIB._destroy()
    if _original_torch_bmm is not None:
        torch.bmm = _original_torch_bmm
        _original_torch_bmm = None
    _batch_invariant_MODE = False
    _batch_invariant_LIB = None


@contextlib.contextmanager
def set_batch_invariant_mode(enabled: bool = True, server_args = None):
    global _batch_invariant_MODE, _batch_invariant_LIB
    old_data = (_batch_invariant_MODE, _batch_invariant_LIB)
    if enabled:
        enable_batch_invariant_mode(server_args)
    else:
        disable_batch_invariant_mode()
    yield
    if _batch_invariant_LIB is not None:
        _batch_invariant_LIB._destroy()
    _batch_invariant_MODE, _batch_invariant_LIB = old_data


AttentionBlockSize = namedtuple("AttentionBlockSize", ["block_m", "block_n"])


def get_batch_invariant_attention_block_size() -> AttentionBlockSize:
    return AttentionBlockSize(block_m=16, block_n=16)
