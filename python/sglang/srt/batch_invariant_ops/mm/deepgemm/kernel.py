import torch
import deep_gemm


def fn(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
):
    if (
        a.dtype == torch.bfloat16
        and (b.dtype == torch.bfloat16)
        and a.is_contiguous()
        and b.transpose(0, 1).is_contiguous()
    ):
        M, K = a.shape
        K, N = b.shape
        dtype = a.dtype
        out = torch.empty((M, N), device=a.device, dtype=dtype)

        deep_gemm.bf16_gemm_nn(a, b, out)

        # TODO can this be put in DeepGEMM's `c`?
        if bias is not None:
            out += bias
    else:
        print('Skipping DeepGEMM as input does not satisfy dtype and layout constraint')
        M, K = a.shape
        K, N = b.shape
        dtype = a.dtype
        out = torch.zeros((M, N), device=a.device, dtype=dtype)

    return out


def bmm_batch_invariant(a, b, *, out=None):
    # Batched matrix multiply: (B, M, K) x (B, K, N) -> (B, M, N)
    # Process each batch separately with our persistent kernel
    if a.ndim == 3 and b.ndim == 3:
        results = []
        for i in range(a.shape[0]):
            results.append(fn(a[i], b[i]))
        result = torch.stack(results, dim=0)

        if out is not None:
            out.copy_(result)
            return out
        return result
    else:
        raise ValueError(
            f"bmm_batch_invariant expects 3D tensors, "
            f"got shapes {a.shape} and {b.shape}"
        )
