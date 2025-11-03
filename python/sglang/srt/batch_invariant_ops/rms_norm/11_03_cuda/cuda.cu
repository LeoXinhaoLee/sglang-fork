#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAGuard.h>     // correct header for CUDAGuard
#include <ATen/cuda/CUDAContext.h>  // for getCurrentCUDAStream()
#include <c10/cuda/CUDAStream.h>    // for CUDAStream and guard

__global__ void batch_invariant_rmsnorm_kernel(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    __nv_bfloat16* y,
    float eps,
    int batch_size,
    int hidden_size) {
    int warpSize = 32;
    int rows_per_block = blockDim.x / warpSize;

    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int row_in_block = warp_id;

    int row_global = blockIdx.x * rows_per_block + row_in_block;
    bool valid_row = (row_global < batch_size);
    // if (row_global >= batch_size) return;

    extern __shared__ float shared_sums[];

    int row_offset = row_global * hidden_size;

    float partial_sum = 0.0f;
    if (valid_row) {
        for (int elem = lane_id; elem < hidden_size; elem += warpSize) {
            int idx = row_offset + elem;
            __nv_bfloat16 val = x[idx];
            float val_f32 = __bfloat162float(val);
            partial_sum += val_f32 * val_f32;
        }

        // Warp-level reduction using shfl_xor
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            partial_sum += __shfl_xor_sync(0xFFFFFFFF, partial_sum, offset, warpSize);
        }
    }

    if (lane_id == 0) {
        shared_sums[row_in_block] = partial_sum;
    }
    __syncthreads();

    if (valid_row) {
        float sum_sq = shared_sums[row_in_block];
        float mean_sq = sum_sq / hidden_size;
        float inv_rms = 1.0f / sqrtf(mean_sq + eps);

        for (int elem = lane_id; elem < hidden_size; elem += warpSize) {
            int idx = row_offset + elem;
            __nv_bfloat16 val = x[idx];
            __nv_bfloat16 w = weight[elem];
            float val_f32 = __bfloat162float(val);
            float w_f32 = __bfloat162float(w);
            float normalized = val_f32 * inv_rms * w_f32;
            y[idx] = __float2bfloat16(normalized);
        }
    }
}

torch::Tensor fn(torch::Tensor x, torch::Tensor weight, float eps) {
    // Set the right device
    c10::cuda::CUDAGuard device_guard(x.device());
    // Use PyTorch's current stream for this device
    auto stream = at::cuda::getCurrentCUDAStream();
    // Make sure subsequent allocations/ops are "on" this stream
    c10::cuda::CUDAStreamGuard stream_guard(stream);

    torch::Tensor x_contig = x.contiguous();
    torch::Tensor weight_contig = weight.contiguous();
    int batch_size = x_contig.size(0);
    int hidden_size = x_contig.size(1);
    torch::Tensor y = torch::empty_like(x_contig);

    TORCH_CHECK(weight_contig.dim() == 1, "Weight must be 1-dimensional");
    TORCH_CHECK(weight_contig.size(0) == hidden_size, "Weight must match hidden size");

    int threads_per_block = 256;
    int warpSize = 32;
    int rows_per_block = threads_per_block / warpSize;
    int num_blocks = (batch_size + rows_per_block - 1) / rows_per_block;

    batch_invariant_rmsnorm_kernel<<<num_blocks, threads_per_block, rows_per_block * sizeof(float), stream.stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_contig.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(weight_contig.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        eps,
        batch_size,
        hidden_size
    );

    // Check for launch/runtime errors without synchronizing the whole device
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fn", &fn, "Batch-invariant RMSNorm (CUDA)");
}