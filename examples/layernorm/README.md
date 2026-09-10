# LayerNorm forward

The original task, inputs, CUDA baseline and PyTorch oracle are preserved from
`tasks/class2/general/layernorm_fwd_1024`. FP32 [16384, 1024], epsilon=1e-5,
all three outputs checked at atol=rtol=1e-4. Both manifests run a
single-operation CUDA Graph on logical device 0 with 256 MiB L2 thrashing.

Requires a CUDA PyTorch installation and nvcc supporting SM120. Choose an idle
physical GPU through CUDA_VISIBLE_DEVICES, and follow the top-level optimize
command. No existing optimized candidate or experimental report is bundled.

The adapter uses the standalone `kai_light.benchmark` namespace.
`upstream/source.json` retains the original source hashes. Upstream headers retain
FlashAttention licensing; see ../../licenses and ../../THIRD_PARTY_NOTICES.md.

`benchmark-graph-events.yaml` is the default manifest. It scores
`operator_latency_ms`: two external CUDA event record
nodes captured around exactly one operator in the same graph. This excludes host
scheduling gaps between Python submissions. Input/output reset, 256 MiB cache
flushing and the original correctness requirements are identical for both arms.

`benchmark.yaml` keeps the original outer-event boundary: the SDK's `latency_ms`
measured around the graph replay from the host. That boundary includes submission
gaps, which are noise for a single operator and can prevent A/A calibration from
passing. Keep it for comparing timing boundaries, not as the optimization target.
