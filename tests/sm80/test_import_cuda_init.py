# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Importing the fork's modules does not initialise CUDA.

Every module of ``vllm_sm80`` the engine process imports (all but the
launcher's, ``vllm_sm80/launch/``, which runs in a process of its own) and
every module the plugin registers is imported in a fresh interpreter that
reports as having a card (``torch.cuda.is_available`` and ``device_count``
forced, device properties of an sm_80 card, no card visible), so code that
would query the device at import reaches ``torch.cuda._lazy_init``; each
call is recorded with its stack and the import goes on.

The model and drafter classes the plugin registers extend upstream modules
whose import already initialises CUDA (the GLM model package through FLA's
platform check, the drafters through flashinfer). Those fork modules must
import completely and make no more CUDA initialisation calls than their
upstream module. For a model override that is the upstream registry entry
it replaces, loaded as the registry loads it (the module, then the class):
a package that defers its classes to first access (Qwen4Exp) initialises
CUDA only then.
"""
import json
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from vllm_sm80 import plugin

CHILD = r'''
import json, sys, traceback, types
import torch, torch.cuda
calls = []
_lazy_init = torch.cuda._lazy_init
def traced():
    calls.append("".join(traceback.format_stack(limit=60)[:-1]))
    try:
        return _lazy_init()
    except Exception as exc:  # no card is visible
        calls[-1] += "lazy_init raised: %r\n" % exc
torch.cuda._lazy_init = traced
torch.cuda.is_available = lambda: True
torch.cuda.device_count = lambda: 1
# torch.cuda defines _get_device_properties in a successful _lazy_init only;
# without a card, get_device_properties would raise NameError and stop the
# import before the fork module's own code runs.
torch.cuda._get_device_properties = lambda device=None: types.SimpleNamespace(
    name="sm_80", major=8, minor=0, multi_processor_count=70,
    total_memory=8 << 30, L2_cache_size=40 << 20, warp_size=32, gcnArchName="",
    max_threads_per_multi_processor=2048, uuid="0")
error = None
try:
    for name in sys.argv[1:]:
        if name == "register()":
            from vllm_sm80 import plugin
            plugin.register()
            plugin.platform_plugin()
        elif ":" in name:
            module, _, attr = name.partition(":")
            __import__(module)
            getattr(sys.modules[module], attr)
        else:
            __import__(name)
except Exception:
    error = traceback.format_exc()[-3000:]
print("RESULT " + json.dumps({"calls": calls, "error": error}))
'''


def _module(qualified: str) -> str:
    """Module of a "<module>:<class>" or "<module>.<class>" name."""
    if ":" in qualified:
        return qualified.partition(":")[0]
    return qualified.rpartition(".")[0]


def _sm80_modules() -> list[str]:
    """Modules of vllm_sm80 outside its model packages (vllm_sm80/models/,
    which extend upstream model packages; the registered ones are checked
    against their upstream module) and its launcher (vllm_sm80/launch/, a
    stdlib-only script run by path in a process of its own, whose modules
    import each other as top-level modules)."""
    root = pathlib.Path(plugin.__file__).parent
    names = set()
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if relative.parts[0] in ("models", "launch"):
            continue
        parts = ("vllm_sm80", *relative.with_suffix("").parts)
        names.add(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return sorted(names)


def _model_package(module: str) -> str:
    """The upstream model package a fork module of a model extends:
    vllm_sm80.models.<model>.<...> -> vllm.models.<model>."""
    parts = module.split(".")
    assert parts[:2] == ["vllm_sm80", "models"] and len(parts) > 3, module
    return f"vllm.models.{parts[2]}"


def _upstream_entry(architecture: str) -> str:
    """The upstream registry entry of ARCHITECTURE, as "<module>:<class>"."""
    from vllm.model_executor.models.registry import _VLLM_MODELS

    module, cls = _VLLM_MODELS[architecture]
    if not module.startswith("vllm."):
        module = f"vllm.model_executor.models.{module}"
    return f"{module}:{cls}"


def _upstream_based() -> dict[str, str]:
    """Registered fork module -> the upstream module (or registry entry) its
    classes extend."""
    based = {}
    for qualified in plugin.GLM_LAYER_OVERRIDES.values():
        module = _module(qualified)
        based[module] = _model_package(module)
    for architecture, qualified in plugin.MODEL_OVERRIDES.items():
        based.setdefault(_module(qualified), _upstream_entry(architecture))
    for qualified, upstream in plugin.DRAFTER_OVERRIDES.values():
        based[_module(qualified)] = upstream
    return based


def _registered() -> set[str]:
    names = {_module(q) for q in plugin.MODEL_OVERRIDES.values()}
    names |= {_module(q) for q, _ in plugin.DRAFTER_OVERRIDES.values()}
    names |= {_module(q) for q in plugin.GLM_LAYER_OVERRIDES.values()}
    names |= {_module(q) for q, _ in plugin.MOE_LAYERS.values()}
    names.add(_module(plugin.TRITON_MLA_SPARSE_BACKEND))
    return names


def _run(*names: str) -> dict:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path),
               CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run([sys.executable, "-c", CHILD, *names], env=env,
                          capture_output=True, text=True, timeout=600)
    lines = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
    assert lines, f"rc={proc.returncode}\n{proc.stderr[-3000:]}"
    return json.loads(lines[-1][len("RESULT "):])


def test_fork_modules_and_registration_do_not_initialise_cuda():
    based = _upstream_based()
    names = sorted((set(_sm80_modules()) | _registered()) - set(based))
    result = _run(*names, "register()")
    assert result["error"] is None, result["error"]
    assert not result["calls"], (
        f"{len(result['calls'])} CUDA lazy init call(s); first:\n"
        + result["calls"][0])


def test_registered_upstream_based_modules_add_no_cuda_initialisation():
    based = _upstream_based()
    assert based, "no registered model or drafter module"
    names = sorted(set(based) | set(based.values()))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = dict(zip(names, pool.map(_run, names)))
    for name, result in results.items():
        assert result["error"] is None, f"{name}:\n{result['error']}"
    for module, upstream in based.items():
        own, base = results[module]["calls"], results[upstream]["calls"]
        assert len(own) <= len(base), (
            f"{module} makes {len(own)} CUDA lazy init call(s) at import, "
            f"{upstream} {len(base)}; last call:\n" + own[-1])
