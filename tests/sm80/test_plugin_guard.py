# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fork's plugin entry point and the startup guard (vllm_sm80/guard.py)."""

import json
import os
import pathlib
import site
import subprocess
import sys
import tomllib

import pytest

import vllm.plugins
from vllm_sm80 import guard, plugin

REPO = pathlib.Path(__file__).resolve().parents[2]
PYPROJECT = REPO / "plugins" / "sm80" / "pyproject.toml"


def _entry_point(group: str = vllm.plugins.DEFAULT_PLUGINS_GROUP) -> str:
    data = tomllib.loads(PYPROJECT.read_text())
    return data["project"]["entry-points"][group][guard.PLUGIN_NAME]


def test_distribution_declares_the_register_entry_point():
    data = tomllib.loads(PYPROJECT.read_text())
    assert data["project"]["name"] == guard.DISTRIBUTION
    assert _entry_point() == "vllm_sm80.plugin:register"


def test_distribution_declares_the_platform_entry_point():
    assert (
        _entry_point(vllm.plugins.PLATFORM_PLUGINS_GROUP)
        == "vllm_sm80.plugin:platform_plugin"
    )


@pytest.fixture
def fork_platform(monkeypatch):
    """Make ``current_platform`` an instance of the fork's CUDA platform."""
    from vllm_sm80 import platform as sm80_platform

    monkeypatch.setattr(
        "vllm.platforms._current_platform", sm80_platform.Sm80CudaPlatform()
    )


def test_guard_ignores_a_process_without_plugins(monkeypatch):
    monkeypatch.setattr(vllm.plugins, "plugins_loaded", False)
    monkeypatch.setattr(plugin, "_registered", False)
    guard.check_plugin_loaded()


def test_guard_passes_when_the_plugin_ran(monkeypatch, fork_platform):
    monkeypatch.setattr(vllm.plugins, "plugins_loaded", True)
    monkeypatch.setattr(plugin, "_registered", False)
    plugin.register()
    plugin.register()
    guard.check_plugin_loaded()


def test_guard_raises_when_plugins_loaded_without_ours(monkeypatch):
    monkeypatch.setattr(vllm.plugins, "plugins_loaded", True)
    monkeypatch.setattr(plugin, "_registered", False)
    with pytest.raises(RuntimeError, match="not the fork's plugin 'sm80'"):
        guard.check_plugin_loaded()


def test_guard_raises_on_cuda_without_the_fork_platform(monkeypatch):
    from vllm.platforms.cuda import CudaPlatform

    monkeypatch.setattr(vllm.plugins, "plugins_loaded", True)
    monkeypatch.setattr(plugin, "_registered", True)
    monkeypatch.setattr("vllm.platforms._current_platform", CudaPlatform())
    with pytest.raises(RuntimeError, match="not the fork's platform"):
        guard.check_plugin_loaded()


def test_guard_ignores_the_platform_off_cuda(monkeypatch):
    from vllm.platforms.cpu import CpuPlatform

    monkeypatch.setattr(vllm.plugins, "plugins_loaded", True)
    monkeypatch.setattr(plugin, "_registered", True)
    monkeypatch.setattr("vllm.platforms._current_platform", CpuPlatform())
    guard.check_plugin_loaded()


def test_platform_plugin_follows_the_builtin_cuda_plugin(monkeypatch):
    import vllm.platforms

    monkeypatch.setattr(vllm.platforms, "cuda_platform_plugin", lambda: None)
    assert plugin.platform_plugin() is None
    builtin = "vllm.platforms.cuda.CudaPlatform"
    monkeypatch.setattr(vllm.platforms, "cuda_platform_plugin", lambda: builtin)
    assert plugin.platform_plugin() == plugin.PLATFORM_QUALNAME


def test_fork_platform_is_chosen_like_upstreams():
    """Same NVML choice as ``vllm.platforms.cuda.CudaPlatform``, and the fork
    class subclasses exactly the upstream class that would be chosen."""
    from vllm.platforms import cuda as cuda_platform
    from vllm_sm80 import platform as sm80_platform
    from vllm.utils.import_utils import resolve_obj_by_qualname

    cls = resolve_obj_by_qualname(plugin.PLATFORM_QUALNAME)
    assert cls is sm80_platform.Sm80CudaPlatform
    assert cls.__mro__[2] is cuda_platform.CudaPlatform
    assert cls in sm80_platform.PLATFORM_CLASSES


def _write_metadata(site: pathlib.Path, platform: bool = True) -> None:
    """The metadata pip writes for plugins/sm80, without running pip.

    ``platform=False`` writes the general entry point only, as an install
    made from an older plugins/sm80 would.
    """
    info = site / "vllm_sm80-0.1.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {guard.DISTRIBUTION}\nVersion: 0.1.0\n"
    )
    text = f"[{vllm.plugins.DEFAULT_PLUGINS_GROUP}]\n"
    text += f"{guard.PLUGIN_NAME} = {_entry_point()}\n"
    if platform:
        group = vllm.plugins.PLATFORM_PLUGINS_GROUP
        text += f"\n[{group}]\n{guard.PLUGIN_NAME} = {_entry_point(group)}\n"
    (info / "entry_points.txt").write_text(text)


CHILD = """
import vllm.plugins
from vllm.config import VllmConfig
vllm.plugins.load_general_plugins()
VllmConfig()
from vllm.platforms import current_platform
print("PLATFORM", type(current_platform).__name__)
print("STARTED")
"""


def _cuda_detected() -> bool:
    """Whether vLLM's builtin CUDA platform plugin activates on this host."""
    from vllm.platforms import cuda_platform_plugin

    return cuda_platform_plugin() is not None


def _has_metadata(path: str) -> bool:
    return any(pathlib.Path(path or ".").glob("vllm_sm80-*.dist-info"))


def _metadata_always_visible() -> bool:
    """Metadata in site-packages or next to the vllm package (a source tree
    on PYTHONPATH) cannot be hidden from the child process."""
    if any(map(_has_metadata, site.getsitepackages())):
        return True
    return any(_has_metadata(p) and _is_source_tree(p) for p in sys.path)


def _is_source_tree(path: str) -> bool:
    return (pathlib.Path(path or ".") / "vllm" / "__init__.py").exists()


def _start(
    tmp_path,
    with_metadata: bool,
    plugins: str | None = None,
    platform: bool = True,
    child: str = CHILD,
    args: tuple[str, ...] = (),
):
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    if with_metadata:
        _write_metadata(site_dir, platform=platform)
    env = {
        k: v for k, v in os.environ.items() if k not in ("VLLM_PLUGINS", "PYTHONPATH")
    }
    # The child sees the plugin metadata only when this test writes it, except
    # where the metadata sits next to the vllm package (a source tree with the
    # metadata installed into it): that entry must stay, or the child would
    # import another vllm.
    paths = [p for p in sys.path if not _has_metadata(p) or _is_source_tree(p)]
    env["PYTHONPATH"] = os.pathsep.join([str(site_dir), *paths])
    env["CUDA_VISIBLE_DEVICES"] = ""
    if plugins is not None:
        env["VLLM_PLUGINS"] = plugins
    return subprocess.run(
        [sys.executable, "-c", child, *args],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_config_starts_with_the_plugin_installed(tmp_path):
    proc = _start(tmp_path, with_metadata=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "STARTED" in proc.stdout
    if _cuda_detected():
        assert "PLATFORM Sm80" in proc.stdout, proc.stdout


def test_config_fails_loudly_without_the_platform_plugin(tmp_path):
    if not _cuda_detected():
        pytest.skip("no CUDA platform on this host, so no platform to check")
    if _metadata_always_visible():
        pytest.skip("the plugin metadata is installed where the child must look")
    proc = _start(tmp_path, with_metadata=True, platform=False)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "STARTED" not in proc.stdout
    assert "not the fork's platform" in out


@pytest.mark.parametrize(
    "with_metadata,plugins",
    [(False, None), (True, "some_other_plugin")],
    ids=["not-installed", "filtered-by-VLLM_PLUGINS"],
)
def test_config_fails_loudly_without_the_plugin(tmp_path, with_metadata, plugins):
    if not with_metadata and _metadata_always_visible():
        pytest.skip("the plugin metadata is installed where the child must look")
    proc = _start(tmp_path, with_metadata=with_metadata, plugins=plugins)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "STARTED" not in proc.stdout
    assert "not the fork's plugin 'sm80'" in out


# A text-only checkpoint config whose quantization method is EXL3. Model
# config verification needs no weights and no EXL3 extension.
EXL3_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 32,
    "vocab_size": 1024,
    "max_position_embeddings": 256,
    "torch_dtype": "bfloat16",
    "quantization_config": {"quant_method": "exl3"},
}

EXL3_CHILD = """
import sys
import vllm.plugins
from vllm.config import ModelConfig
vllm.plugins.load_general_plugins()
config = ModelConfig(
    model=sys.argv[1], skip_tokenizer_init=True, max_model_len=128, dtype="bfloat16"
)
print("QUANTIZATION", config.quantization)
"""


def _start_exl3(tmp_path, with_metadata: bool, plugins: str | None = None):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(EXL3_CONFIG))
    proc = _start(
        tmp_path, with_metadata, plugins, child=EXL3_CHILD, args=(str(checkpoint),)
    )
    return proc, proc.stdout + proc.stderr


def test_exl3_checkpoint_selects_the_registered_method(tmp_path):
    proc, out = _start_exl3(tmp_path, with_metadata=True)
    # EXL3 was selected: verification passes on CUDA hosts and refuses the
    # checkpoint where vLLM resolved another platform.
    assert "QUANTIZATION exl3" in proc.stdout or (
        "EXL3 execution requires the NVIDIA CUDA platform" in out
    ), out
    assert "Unknown quantization method" not in out


@pytest.mark.parametrize(
    "with_metadata,plugins",
    [(False, None), (True, "some_other_plugin")],
    ids=["not-installed", "filtered-by-VLLM_PLUGINS"],
)
def test_exl3_checkpoint_fails_at_startup_without_the_plugin(
    tmp_path, with_metadata, plugins
):
    if not with_metadata and _metadata_always_visible():
        pytest.skip("the plugin metadata is installed where the child must look")
    proc, out = _start_exl3(tmp_path, with_metadata, plugins)
    assert proc.returncode != 0, out
    # No fallback to an unquantized model: the method is unknown without the
    # plugin, so the model config itself fails.
    assert "QUANTIZATION" not in proc.stdout
    assert "Unknown quantization method: exl3" in out
