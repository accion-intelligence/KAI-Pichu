#include <cuda_runtime.h>
#include "solution.cu"

// Fixed ABI shared by all workspaces. The adapter owns buffers and timing.
extern "C" int kai_launch(
    const float* x, const float* gamma, const float* beta,
    float* z, float* mu, float* rs,
    int rows, int cols, float eps, cudaStream_t stream) {
    launch_layernorm_forward(x, gamma, beta, z, mu, rs, rows, cols, eps, stream);
    return static_cast<int>(cudaGetLastError());
}
