# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_80 KDA pieces called from shared engine code.

  ``recover``  RecoverSSM for KDA layers: the enable check, the GDN metadata
               builder hook, the per-token verify records and the commit
               kernel. The fork's GLM KDA layer, its GDN metadata builder
               (vllm_sm80/models/glm5next/sm80_kda.py) and the GLM model
               state call it; the KDA decode and prefill kernels themselves
               stay in vllm/models/glm5next/nvidia/ops/.

  ``commit_plan``  the RecoverSSM commit plan and align postprocess kernels.
               Upstream's Kimi-K3 commit context and RecoverSSMState launch
               them in place of their own (the upstream definitions stay,
               unused); ``recover`` launches the plan kernel and replicates
               it in its commit fold plan.
"""
