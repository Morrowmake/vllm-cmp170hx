# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# The modules here extend vllm.models.glm5next, and its package import loads
# modules that import some of them back. Load it first, as when these modules
# lived inside it.
import vllm.models.glm5next  # noqa: E402, F401
