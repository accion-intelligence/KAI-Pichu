#include <cuda_runtime.h>
#include "solution.cu"

// Fixed ABI shared by all workspaces. The adapter owns buffers, streams and
// timing. launch_epilogue returns 0 on success or a negative code for arguments
// the implementation does not support; CUDA launch errors are reported after it.
// workspace is scratch memory with undefined contents.
extern "C" int kai_launch(
    const float* x, const float* bias, const float* residual, float* out,
    int rows, int cols, void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    const int status = launch_epilogue(x, bias, residual, out, rows, cols, workspace, workspace_bytes, stream);
    if (status != 0) {
        return status;
    }
    return static_cast<int>(cudaGetLastError());
}
