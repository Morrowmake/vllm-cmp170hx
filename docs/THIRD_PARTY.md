# Third-party sources and idea credits

The fork is based on [vLLM](https://github.com/vllm-project/vllm) and retains
its Apache-2.0 licence, source notices and upstream contributor history.
Morrowmake maintains the CMP 170HX changes; inherited and adapted code is not
claimed as independently authored. These credits supplement, rather than
replace, the original per-file and bundled licence notices. They do not imply
endorsement by the people or projects named here.

## Code and reference implementations

- **wtdcode/vllm-backport and lazymio:** the shared-expert aux-stream reorder in
  `vllm_sm80/layers/moe/{runner,shared_experts}.py` adapts
  [commit 9925c45](https://github.com/wtdcode/vllm-backport/commit/9925c45ab4c0740dfc8e77b4715f665e8b205447)
  (Apache-2.0). Morrowmake added model-specific gating and the fork's integration.
  The source's overlap measurement belongs to that source, not to this fork.
- **vLLM RecoverSSM contributors, including Yun Zhang:** the GLM KDA recovery
  path builds on [upstream commit 70afded](https://github.com/vllm-project/vllm/commit/70afdedc1081d28c3eaae53bece8292298484c86)
  and the inherited Kimi implementation (Apache-2.0).
  `vllm_sm80/kernels/kda/recover.py` adapts the state-recovery approach and
  reuses its commit-plan/conv helpers.
- **vLLM sparse-attention contributors:** `vllm_sm80/kernels/attention/triton_mla_sparse.py`
  adapts the inherited [XPU sparse-MLA implementation](https://github.com/vllm-project/vllm/blob/e55d076f89fd01a0538a3e496d8ff20bf7980100/vllm/v1/attention/ops/xpu_mla_sparse.py)
  for CUDA sm_80 (Apache-2.0).
- **Marlin, Elias Frantar and subsequent Neural Magic contributors:** the
  compiled Marlin work builds on the inherited Marlin implementation and
  [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin) (Apache-2.0).
  Original Marlin copyright and Neural Magic modification notices are retained
  in the inherited template and operation files.
- **Flash Linear Attention, Songlin Yang, Yu Zhang and contributors:**
  `vllm_sm80/models/glm5next/ops/kda_prefill.py` derives from the FLA
  KDA/chunk-state/triangular-solve implementations. Its original author notice is retained;
  the full MIT grant is reproduced in its source header.
- **DeepSeek / DeepGEMM:** `fp8_paged_mqa_logits_torch` in
  `vllm_sm80/kernels/attention/triton_mqa_logits.py` adapts the
  [DeepGEMM attention reference](https://github.com/vllm-project/DeepGEMM/blob/e1f418c2a4f20818221f6b0e578b4c2f634d4c3f/tests/test_attention.py)
  through upstream vLLM's `rocm_aiter_mla_sparse.py`. Morrowmake changed the
  page layout, per-row contexts and sm_80 conversion handling. The complete
  applicable MIT notice is reproduced below.

- **ExLlamaV3, Turboderp:** the EXL3 trellis packing, procedural codebooks,
  tensor-core lane ordering, Hadamard conventions and n-gram codec mapping in
  `csrc/exl3/` and `vllm_sm80/kernels/exl3/` adapt
  [ExLlamaV3 at d3739fd](https://github.com/turboderp-org/exllamav3/tree/d3739fd393337b1ff4d6c2a342b12f0c87a9592f)
  (MIT). The licence and the modification notice are in
  [licenses/exl3](../licenses/exl3/).
- **Curtis White (CurtisAccelerate):** the hidden-5120 lane shape of
  `csrc/exl3/moe_v2.cu` and `csrc/exl3/moe_v2_try.cu` adapts
  [DeepSeekv41-Flash-EXL3-TC-MoE at 88eb6e1](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE/tree/88eb6e1)
  (Apache-2.0); see [licenses/exl3/NOTICE](../licenses/exl3/NOTICE).
- **GNmac-lab:** `vllm_sm80/models/deepseek_v41/ops/sm80_wo_a_grouped.py`
  adapts the E4M3 decode and grouped tile loop of `patches/dsv41_wo_a.py` in
  [deepseek-v41-flash-1017-cmp170hx at bbaf367](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx/tree/bbaf367)
  (MIT). The full MIT notice is in the source header and reproduced below.

## Ideas informing the changes

The following are idea credits, not assertions that those projects' code was
copied or that their licences apply to our independent implementations.

- [MiaAI-Lab's GLM recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/tree/6278ecb01034cea9ef6de0f851d09fccafe3e835):
  kpool slot-mapping correctness, Mamba in-flight reservation, tool-call masking
  with `tool_choice="none"`, cached prompt-boundary lookup and scheduler back-off.
  Its startup canary, image layer budget and reasoning-effort setting also
  informed the companion recipe.
- [JJ48's CMP 170HX serving work](https://github.com/JJ48/glm53-flash-170hx-serving):
  published acceptance/step-cost modelling informed acceptance-adaptive draft
  depth and its concurrency limits.
- [kindlingai's GX10 recipe](https://github.com/kindlingai/glm-5.3-flash-gx10):
  the investigation lead for GLM KDA state recovery, unused-draft work and
  exact BF16 mHC weight storage. No code from this unlicensed recipe was used;
  the RecoverSSM code source is upstream vLLM, identified separately above.

- [GNmac-lab's DeepSeek-V4.1-Flash recipe](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx/tree/bbaf367):
  the greedy target argmax (`vllm_sm80/worker/target_argmax.py`), the mHC side
  stream (`vllm_sm80/models/deepseek_v41/ops/sm80_mhc_side.py`) and, with
  Curtis White's port, the per-layer rotated tensor split of the routed experts.
- [b12x (local-inference-lab)](https://github.com/local-inference-lab/b12x)
  (Apache-2.0): the copy-engine ring all-reduce in
  `vllm_sm80/distributed/ce_ring_all_reduce.py` follows the design of
  `b12x/comm/pcie/pcie_dma.py`. No code was copied.

Checkpoint licences are separate from engine licences. The companion recipe
identifies the GLM target and DFlash2 drafter, including the drafter's
non-commercial/no-derivatives restrictions and original method citations.

## DeepGEMM MIT notice

MIT License

Copyright (c) 2025 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## GNmac-lab MIT notice

MIT License

Copyright (c) 2026 GNmac-lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
