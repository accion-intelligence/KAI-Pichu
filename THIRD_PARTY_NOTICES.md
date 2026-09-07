# Source and license notices

## Optimizer

Original optimizer components are Copyright (c) 2025 Zijian Zhang and retain
[their MIT license](licenses/Optimizer-MIT.txt). Source revision and hashes are
recorded in [the source history](licenses/optimizer-source.json).

## KAI Benchmark SDK

`src/kai_light/benchmark` is a standalone copy of the KAI Benchmark SDK v1,
including the paired buffer crossover and CUDA event GC guard. Public imports
and CLI branding use `kai_light`. The parent KAI repository is not a runtime
dependency. Framework changes use the project's Apache 2.0 license.

## LayerNorm example

`examples/layernorm` contains the LayerNorm-1024 task definition and reference
snapshot from the CUDA-Hercules benchmark. The CUDA baseline is unchanged; the
adapter and graph measurement manifests were adapted for KAI-light.
`upstream/source.json` records the original source hashes.

The benchmark's [Apache 2.0 license](licenses/Benchmark-Apache-2.0.txt) is
included. The task identifies the kernel source as Dao-AILab/flash-attention
and BSD-3-Clause; the corresponding local FlashAttention
[license text](licenses/FlashAttention-LICENSE.txt) is retained for those headers.
Existing source notices remain intact. These notices do not relicense third-party
material under the framework license.
