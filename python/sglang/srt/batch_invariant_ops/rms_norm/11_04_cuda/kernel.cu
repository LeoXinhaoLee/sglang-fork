#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <assert.h>
#include <c10/cuda/CUDAGuard.h>     // correct header for CUDAGuard
#include <ATen/cuda/CUDAContext.h>  // for getCurrentCUDAStream()
#include <c10/cuda/CUDAStream.h>    // for CUDAStream and guard

#define WARP_SIZE 32

__global__ void batch_invariant_rmsnorm_kernel_128(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ y,
    float eps,
    int batch_size
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;
    int row_idx = blockIdx.x * warps_per_block + warp_id;
    if (row_idx >= batch_size) return;

    int row_start = row_idx * 128;

    float sum_sq = 0.0f;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int idx = row_start + lane_id + i * WARP_SIZE;
        __nv_bfloat16 val = x[idx];
        float fval = __bfloat162float(val);
        sum_sq += fval * fval;
    }

    // Warp reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset, WARP_SIZE);
    }

    float mean_sq = sum_sq / 128.0f;
    float rms = rsqrtf(mean_sq + eps);
    rms = __shfl_sync(0xFFFFFFFF, rms, 0, WARP_SIZE);

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int idx = row_start + lane_id + i * WARP_SIZE;
        __nv_bfloat16 x_val = x[idx];
        __nv_bfloat16 w_val = weight[lane_id + i * WARP_SIZE];

        float x_float = __bfloat162float(x_val);
        float w_float = __bfloat162float(w_val);

        float normalized = x_float * rms * w_float;
        y[idx] = __float2bfloat16(normalized);
    }
}

__global__ void batch_invariant_rmsnorm_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ y,
    float eps,
    int batch_size,
    int hidden_size
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;
    int row_idx = blockIdx.x * warps_per_block + warp_id;
    if (row_idx >= batch_size) return;

    int row_start = row_idx * hidden_size;
    int elements_per_warp = hidden_size / WARP_SIZE;

    float sum_sq = 0.0f;

    for (int i = 0; i < elements_per_warp; ++i) {
        int idx = row_start + lane_id + i * WARP_SIZE;
        __nv_bfloat16 val = x[idx];
        float fval = __bfloat162float(val);
        sum_sq += fval * fval;
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset, WARP_SIZE);
    }

    float mean_sq = sum_sq / hidden_size;
    float rms = rsqrtf(mean_sq + eps);
    rms = __shfl_sync(0xFFFFFFFF, rms, 0, WARP_SIZE);

    for (int i = 0; i < elements_per_warp; ++i) {
        int idx = row_start + lane_id + i * WARP_SIZE;
        __nv_bfloat16 x_val = x[idx];
        __nv_bfloat16 w_val = weight[lane_id + i * WARP_SIZE];

        float x_float = __bfloat162float(x_val);
        float w_float = __bfloat162float(w_val);

        float normalized = x_float * rms * w_float;
        y[idx] = __float2bfloat16(normalized);
    }
}

torch::Tensor fn(torch::Tensor x, torch::Tensor weight, float eps) {
    // Set the right device
    c10::cuda::CUDAGuard device_guard(x.device());
    // Use PyTorch's current stream for this device
    auto stream = at::cuda::getCurrentCUDAStream();
    // Make sure subsequent allocations/ops are "on" this stream
    c10::cuda::CUDAStreamGuard stream_guard(stream);

    x = x.contiguous();
    weight = weight.contiguous();

    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
    TORCH_CHECK(x.size(1) == weight.size(0), "hidden_size mismatch");

    int batch_size = x.size(0);
    int hidden_size = x.size(1);

    auto options = x.options().dtype(x.scalar_type()).device(x.device());
    torch::Tensor y = torch::empty_like(x);

    dim3 blockDim(128, 1, 1); 
    dim3 gridDim((batch_size + (blockDim.x / WARP_SIZE) - 1) / (blockDim.x / WARP_SIZE), 1, 1);

    if (hidden_size == 128) {
        batch_invariant_rmsnorm_kernel_128<<<gridDim, blockDim, 0, stream.stream()>>>(
            (const __nv_bfloat16*)x.data_ptr(),
            (const __nv_bfloat16*)weight.data_ptr(),
            (__nv_bfloat16*)y.data_ptr(),
            eps,
            batch_size
        );
    } else {
        batch_invariant_rmsnorm_kernel<<<gridDim, blockDim, 0, stream.stream()>>>(
            (const __nv_bfloat16*)x.data_ptr(),
            (const __nv_bfloat16*)weight.data_ptr(),
            (__nv_bfloat16*)y.data_ptr(),
            eps,
            batch_size,
            hidden_size
        );
    }

    // Check for launch/runtime errors without synchronizing the whole device
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fn", &fn, "Compute batch-invariant RMSNorm");
}