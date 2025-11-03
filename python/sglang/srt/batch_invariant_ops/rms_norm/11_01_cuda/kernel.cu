#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

__global__ void batch_invariant_rmsnorm_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight, __nv_bfloat16* y, float eps, int batch_size, int hidden_size) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = 256;
    const int NUM_ELEM_PER_THREAD = 16;

    float val_f32[NUM_ELEM_PER_THREAD];
    float weight_f32[NUM_ELEM_PER_THREAD];
    float sum_sq = 0.0f;

    for (int e = 0; e < NUM_ELEM_PER_THREAD; e++) {
        int i = tid + e * num_threads;
        int idx = batch_idx * hidden_size + i;
        val_f32[e] = __bfloat162float(x[idx]);
        weight_f32[e] = __bfloat162float(weight[i]);
        sum_sq += val_f32[e] * val_f32[e];
    }

    extern __shared__ float shared_data[];
    shared_data[tid] = sum_sq;
    __syncthreads();

    for (int s = num_threads / 2; s >= 1; s /= 2) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    float mean_sq = shared_data[0] / (float) hidden_size;
    float rms = sqrtf(mean_sq + eps);
    float inv_rms = 1.0f / rms;

    for (int e = 0; e < NUM_ELEM_PER_THREAD; e++) {
        int i = tid + e * num_threads;
        int idx = batch_idx * hidden_size + i;
        float result = val_f32[e] * inv_rms * weight_f32[e];
        y[idx] = __float2bfloat16(result);
    }
}

torch::Tensor fn(torch::Tensor x, torch::Tensor weight, float eps) {
    x = x.contiguous();
    weight = weight.contiguous();
    int64_t batch_size = x.size(0);
    int64_t hidden_size = x.size(1);
    // TORCH_CHECK(hidden_size == 4096, "Only supports hidden_size=4096");

    torch::Tensor y = torch::empty_like(x);

    int threads_per_block = 256;
    int blocks_per_grid = batch_size;

    batch_invariant_rmsnorm_kernel<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(float)>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        eps,
        batch_size,
        hidden_size
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fn", &fn, "Compute batch-invariant RMSNorm (CUDA)");
}