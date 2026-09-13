/*
 * Unfused baseline: the three supplied kernels launched back to back.
 *
 * bias_add writes y to the workspace, gelu reads y and writes z to the
 * workspace, residual_add reads z and writes out. A fused implementation
 * computes out from x, bias and residual directly and may ignore the workspace.
 */
#include <cuda_runtime.h>

#include "kernels/bias_add.cu"
#include "kernels/gelu.cu"
#include "kernels/residual_add.cu"

// Returns 0 on success, -1 if the workspace cannot hold the two intermediates.
// Launch errors are read by the caller through cudaGetLastError.
int launch_epilogue(const float* x, const float* bias, const float* residual, float* out,
                    int rows, int cols, void* workspace, size_t workspace_bytes,
                    cudaStream_t stream) {
    const long long total = static_cast<long long>(rows) * cols;
    const size_t intermediate_bytes = static_cast<size_t>(total) * sizeof(float);
    if (workspace_bytes < 2 * intermediate_bytes) {
        return -1;
    }
    float* y = static_cast<float*>(workspace);
    float* z = y + total;
    int status = launch_bias_add(x, bias, y, rows, cols, stream);
    if (status == 0) {
        status = launch_gelu(y, z, total, stream);
    }
    if (status == 0) {
        status = launch_residual_add(z, residual, out, total, stream);
    }
    return status;
}
