# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup check that the fork's vLLM plugin is loaded.

vLLM logs and then ignores an entry-point plugin that fails to load, and
``VLLM_PLUGINS`` can leave a plugin out (``vllm.plugins.load_plugins_by_group``).
Anything the plugin registers would then quietly fall back to upstream
behaviour. ``VllmConfig.__post_init__`` calls :func:`check_plugin_loaded`, which
turns that case into an error at startup.
"""

PLUGIN_NAME = "sm80"
DISTRIBUTION = "vllm-sm80"


def check_plugin_loaded() -> None:
    """Raise if this process loaded vLLM's general plugins but not ``sm80``.

    A process that never loaded general plugins (unit tests that build configs
    directly, offline scripts) is not checked: there is nothing to compare with.
    """
    import vllm.plugins

    if not vllm.plugins.plugins_loaded:
        return

    from vllm_sm80 import plugin

    if plugin.is_registered():
        check_platform()
        return
    raise RuntimeError(
        f"vLLM loaded its general plugins, but not the fork's plugin "
        f"{PLUGIN_NAME!r}. Install the {DISTRIBUTION} distribution into the "
        f"environment that runs vLLM (pip install --no-deps plugins/sm80 from "
        f"the fork's source tree). If VLLM_PLUGINS is set, it must include "
        f"{PLUGIN_NAME!r}. If the distribution is installed, the log above has "
        f"'Failed to load plugin {PLUGIN_NAME}' with the reason."
    )


def check_platform() -> None:
    """Raise if the fork's plugin is loaded on CUDA without the fork's platform.

    The general plugin and the platform plugin come from the same
    distribution, but ``VLLM_PLUGINS`` filters both groups by name and a
    platform plugin that fails during detection is only logged at debug level
    (``vllm.platforms.resolve_current_platform_cls_qualname``). The platform
    then silently stays upstream's.
    """
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        return
    from vllm_sm80.platform import PLATFORM_CLASSES

    if isinstance(current_platform, PLATFORM_CLASSES):
        return
    raise RuntimeError(
        f"The fork's plugin {PLUGIN_NAME!r} is loaded, but the CUDA platform is "
        f"{type(current_platform).__module__}.{type(current_platform).__name__}, "
        f"not the fork's platform from the same distribution ({DISTRIBUTION}, "
        f"vllm.platform_plugins entry point {PLUGIN_NAME} = "
        f"vllm_sm80.plugin:platform_plugin). Reinstall "
        f"the distribution (pip install --no-deps plugins/sm80) so its "
        f"metadata lists both entry points; if VLLM_PLUGINS is set, it must "
        f"include {PLUGIN_NAME!r}."
    )
