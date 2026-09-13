#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "solution.cu"

// Fixed ABI shared by all workspaces. The adapter owns buffers, streams and
// timing. launch_attention_forward returns 0 on success or a negative code for
// arguments the implementation does not support; CUDA launch errors are
// reported after it. workspace is scratch memory with undefined contents.
extern "C" int kai_launch(
    const __half* q, const __half* k, const __half* v, __half* o,
    int batch, int heads, int seq, int head_dim, float scale, int causal,
    void* workspace, size_t workspace_bytes, cudaStream_t stream) {
    const int status = launch_attention_forward(q, k, v, o, batch, heads, seq, head_dim,
                                                scale, causal, workspace, workspace_bytes, stream);
    if (status != 0) {
        return status;
    }
    return static_cast<int>(cudaGetLastError());
}
