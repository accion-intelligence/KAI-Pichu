# Source and license notices

This documentation bundle reflects the [source repository notices at commit 451a839](https://github.com/accion-intelligence/KAI-Core/blob/451a839fdea43da3513cb33c99075782477fa8ce/THIRD_PARTY_NOTICES.md). It contains documentation and license texts, not the executable source tree. Source-history references remain pinned to that repository; this page does not replace or change its original notices.

## Optimizer

Original optimizer components are Copyright (c) 2025 Zijian Zhang and retain [their MIT license](licenses/Optimizer-MIT.txt). Source revision and hashes are recorded in [the source history](https://github.com/accion-intelligence/KAI-Core/blob/451a839fdea43da3513cb33c99075782477fa8ce/licenses/optimizer-source.json).

## KAI Benchmark SDK

The source repository's `src/kai_core/benchmark` is a standalone copy of the KAI Benchmark SDK v1, including the paired buffer crossover and CUDA event GC guard. Public imports and CLI branding use `kai_core`. The parent KAI repository is not a runtime dependency. Framework changes use the project's [Apache 2.0 license](LICENSE).

## LayerNorm example

The source repository's `examples/layernorm` contains the LayerNorm-1024 task definition and reference snapshot from the CUDA-Hercules benchmark. The CUDA baseline is unchanged; the adapter and graph measurement manifests were adapted for the standalone agent. Its `upstream/source.json` records the original source hashes. The benchmark's [retained Apache 2.0 license](https://github.com/accion-intelligence/KAI-Core/blob/451a839fdea43da3513cb33c99075782477fa8ce/licenses/Benchmark-Apache-2.0.txt) is available in the source repository. The task identifies the kernel source as Dao-AILab/flash-attention and BSD-3-Clause; the corresponding [license text](licenses/FlashAttention-LICENSE.txt) is included here. Existing source notices remain intact. These notices do not relicense third-party material under the framework license.
