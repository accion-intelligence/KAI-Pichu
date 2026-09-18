#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "solution.cu"

// Fixed ABI shared by all workspaces. The adapter owns buffers, streams and
// timing. launch_awq_linear returns 0 on success or a negative code for
// arguments the implementation does not support; CUDA launch errors are
// reported after it. workspace is scratch memory with undefined contents.
extern "C" int kai_launch(
    const __half* x, const int* qweight, const int* qzeros, const __half* scales, __half* y,
    int m, int k, int n, int group_size, void* workspace, size_t workspace_bytes,
    cudaStream_t stream) {
    const int status = launch_awq_linear(x, qweight, qzeros, scales, y, m, k, n, group_size,
                                         workspace, workspace_bytes, stream);
    if (status != 0) {
        return status;
    }
    return static_cast<int>(cudaGetLastError());
}
