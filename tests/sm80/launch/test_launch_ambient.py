# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ambient environment: classes, policies and the exec'd child."""

from __future__ import annotations

import json
import os

import pytest

from helpers import World, promote, read_record, write

# names /bin/sh (the stub server) adds to its own environment
SHELL_ADDED = {"PWD", "SHLVL", "_", "OLDPWD"}


@pytest.fixture
def w(tmp_path):
    w = World(tmp_path)
    assert w.verify() == 0
    promote(w, ["tp2"])
    return w


def reasons(d):
    return d["validated"]["reasons"]


@pytest.mark.parametrize("policy", ["strict", "merge"])
def test_retired_name_refused(w, policy):
    msg = w.refused("--policy", policy, environ=w.env(VLLM_OLD_THIN_GEMM="1"))
    assert "renamed to VLLM_THIN_GEMM" in msg


@pytest.mark.parametrize("policy", ["strict", "merge"])
def test_unknown_name_refused(w, policy):
    assert "not registered" in w.refused("--policy", policy, environ=w.env(VLLM_FOO="1"))


@pytest.mark.parametrize("name,value", [("VLLM_KERNEL_X", "0"), ("NCCL_ALGO", "Ring"),
                                        ("PYTORCH_NO_CUDA_MEMORY_CACHING", "1"),
                                        ("CUBLAS_WORKSPACE_CONFIG", ":0:0"),
                                        ("PYTHONOPTIMIZE", "1"), ("PYTHONHASHSEED", "1"),
                                        ("MALLOC_ARENA_MAX", "2"), ("OMP_PROC_BIND", "true")])
def test_undeclared_functional(w, name, value):
    assert "not declared" in w.refused(environ=w.env(**{name: value}))
    d = w.resolve("--policy", "merge", environ=w.env(**{name: value}))
    assert d["validated"]["state"] == "NOT_VALIDATED"
    assert f"ambient {name}" in reasons(d)
    assert d["ambient_layer"] == [f"env.{name}"]
    assert d["child_env"][name]["value"] == value


@pytest.mark.parametrize("policy", ["strict", "merge"])
def test_equal_value_stays_validated(w, policy):
    environ = w.env(VLLM_THIN_GEMM="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                    CUDA_VISIBLE_DEVICES="0,1")
    d = w.resolve("--policy", policy, environ=environ)
    assert d["validated"]["state"] == "VALIDATED", reasons(d)
    assert set(d["ambient_equal"]) >= {"VLLM_THIN_GEMM", "CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES"}
    assert d["resolved"]["env.VLLM_THIN_GEMM"]["layer"] == "model"


def test_declared_different_value(w):
    assert "differs from the settings" in w.refused(environ=w.env(VLLM_THIN_GEMM="0"))
    d = w.resolve("--policy", "merge", environ=w.env(VLLM_THIN_GEMM="0"))
    assert d["resolved"]["env.VLLM_THIN_GEMM"]["layer"] == "ambient"
    assert "ambient VLLM_THIN_GEMM" in reasons(d)


def test_inert_passes_not_compared(w):
    d = w.resolve(environ=w.env(TERM="dumb", LANG="C"))
    assert d["child_env"]["TERM"] == {"value": "dumb", "class": "inert"}
    assert d["validated"]["state"] == "VALIDATED"


def test_fixed_key_from_ambient(w):
    model = open(os.path.join(w.settings, "model.toml")).read().replace(
        'fixed = ["args.', 'fixed = ["env.VLLM_FAST_PATH", "args.')
    write(os.path.join(w.settings, "model.toml"), model)
    msg = w.refused("--policy", "merge", environ=w.env(VLLM_FAST_PATH="0"))
    assert "fixed" in msg


def test_ownership_names(w):
    own = {"GPU_LOCK_TOKENS": "a b", "RAM_LEASE_TOKEN": "t", "RAM_JOB_MARKER": "m", "SESSION_JOB": "j"}
    d = w.resolve(environ=w.env(**own))
    for k, v in own.items():
        assert d["child_env"][k] == {"value": v, "class": "ownership"}
    assert d["validated"]["state"] == "VALIDATED"
    base = w.resolve()
    assert d["functional_sha256"] == base["functional_sha256"]
    p = w.run(environ=w.env(**own))
    assert p.returncode == 0, p.stderr
    _, env = read_record(w.venv)
    assert all(env[k] == v for k, v in own.items())


@pytest.mark.parametrize("policy", ["strict", "merge"])
@pytest.mark.parametrize("name", ["LD_PRELOAD", "LD_AUDIT", "PYTHONHOME", "PYTHONSTARTUP"])
def test_code_loading_always_refused(w, policy, name):
    assert "refused" in w.refused("--policy", policy, environ=w.env(**{name: "/x"}))


def test_pythonpath_other_tree(w):
    environ = w.env(PYTHONPATH="/elsewhere")
    assert "strict" in w.refused(environ=environ)
    d = w.resolve("--policy", "merge", environ=environ)
    assert "ambient PYTHONPATH" in reasons(d)


def test_path_ignored_and_rebuilt(w):
    d = w.resolve(environ=w.env(PATH="/evil/bin:/usr/bin"))
    assert "PATH" in d["ignored"]
    assert d["child_env"]["PATH"]["value"].startswith(os.path.join(w.venv, "bin") + ":")
    assert d["validated"]["state"] == "VALIDATED"


def test_cards_from_lock(w):
    assert "differs from the resolved cards" in w.refused(environ=w.env(CUDA_VISIBLE_DEVICES="1,0"))


def test_container_runtime_names(w):
    d = w.resolve(environ=w.env(NVIDIA_VISIBLE_DEVICES="all"))
    assert d["child_env"]["NVIDIA_VISIBLE_DEVICES"]["class"] == "container-runtime"
    assert d["validated"]["state"] == "VALIDATED"


def test_unclassified_dropped_from_child(w):
    environ = w.env(FOO_DEBUG="1", __NV_PRIME_RENDER_OFFLOAD="1")
    d = w.resolve(environ=environ)
    assert set(d["dropped"]) == {"FOO_DEBUG", "__NV_PRIME_RENDER_OFFLOAD"}
    assert d["validated"]["state"] == "VALIDATED"
    p = w.run(environ=environ)
    assert p.returncode == 0, p.stderr
    _, env = read_record(w.venv)
    assert "FOO_DEBUG" not in env and "__NV_PRIME_RENDER_OFFLOAD" not in env
    assert env["PYTHONNOUSERSITE"] == "1"
    assert set(env) - SHELL_ADDED == set(d["child_env"])


def test_isolated_interpreter_output_identical(w, tmp_path):
    shadow = tmp_path / "shadow"
    write(str(shadow / "tomllib.py"), "raise SystemExit('shadowed')\n")
    base = w.run("--print-config", "--json")
    hostile = w.run("--print-config", "--json",
                    environ=w.env(PYTHONOPTIMIZE="2", PYTHONPATH=str(shadow)))
    # the ambient names are refused (strict); the resolution itself is not altered
    assert base.returncode == 0
    assert hostile.returncode == 3 and "shadowed" not in hostile.stderr
    merged = w.run("--print-config", "--json", "--policy", "merge",
                   environ=w.env(PYTHONOPTIMIZE="2", PYTHONPATH=str(shadow)))
    d0, d1 = json.loads(base.stdout), json.loads(merged.stdout)
    assert d0["argv"] == d1["argv"] and d0["functional_sha256"] != d1["functional_sha256"]


def test_container_drops_ownership(tmp_path):
    w = World(tmp_path)
    d = w.resolve("--runtime", "container", "--tree", w.tree, environ={"RAM_LEASE_TOKEN": "t"})
    assert "RAM_LEASE_TOKEN" in d["dropped"]


# -- the model image's ENV --------------------------------------------------------------------

# the config Env of the published model images (identical in both), plus
# PYTHONDONTWRITEBYTECODE, which the model image sets
IMAGE_ENV = {
    "CUDA_HOME": "/usr/local/cuda", "CUDA_VERSION": "13.3.1",
    "LD_LIBRARY_PATH": "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64",
    "LIBRARY_PATH": "/usr/local/cuda/lib64/stubs", "NCCL_VERSION": "2.30.7-1", "NVARCH": "x86_64",
    "NVIDIA_DRIVER_CAPABILITIES": "compute,utility", "NVIDIA_PRODUCT_NAME": "CUDA",
    "NVIDIA_REQUIRE_CUDA": "cuda>=13.0", "NVIDIA_VISIBLE_DEVICES": "all",
    "NV_CUDA_CUDART_DEV_VERSION": "13.3.29-1", "NV_CUDA_CUDART_VERSION": "13.3.29-1",
    "NV_CUDA_LIB_VERSION": "13.3.1-1",
    "NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE": "cuda-nsight-compute-13-3=13.3.1-1",
    "NV_CUDA_NSIGHT_COMPUTE_VERSION": "13.3.1-1",
    "NV_LIBCUBLAS_DEV_PACKAGE": "libcublas-dev-13-3=13.6.0.2-1",
    "NV_LIBCUBLAS_DEV_PACKAGE_NAME": "libcublas-dev-13-3", "NV_LIBCUBLAS_DEV_VERSION": "13.6.0.2-1",
    "NV_LIBCUBLAS_PACKAGE": "libcublas-13-3=13.6.0.2-1", "NV_LIBCUBLAS_PACKAGE_NAME": "libcublas-13-3",
    "NV_LIBCUBLAS_VERSION": "13.6.0.2-1", "NV_LIBCUSPARSE_DEV_VERSION": "12.8.2.51-1",
    "NV_LIBCUSPARSE_VERSION": "12.8.2.51-1", "NV_LIBNCCL_DEV_PACKAGE": "libnccl-dev=2.30.7-1+cuda13.3",
    "NV_LIBNCCL_DEV_PACKAGE_NAME": "libnccl-dev", "NV_LIBNCCL_DEV_PACKAGE_VERSION": "2.30.7-1",
    "NV_LIBNCCL_PACKAGE": "libnccl2=2.30.7-1+cuda13.3", "NV_LIBNCCL_PACKAGE_NAME": "libnccl2",
    "NV_LIBNCCL_PACKAGE_VERSION": "2.30.7-1", "NV_LIBNPP_DEV_PACKAGE": "libnpp-dev-13-3=13.1.2.81-1",
    "NV_LIBNPP_DEV_VERSION": "13.1.2.81-1", "NV_LIBNPP_PACKAGE": "libnpp-13-3=13.1.2.81-1",
    "NV_LIBNPP_VERSION": "13.1.2.81-1", "NV_NVML_DEV_VERSION": "13.3.29-1",
    "NV_NVTX_VERSION": "13.3.29-1",
    "PATH": "/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:"
            "/usr/sbin:/usr/bin:/sbin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1", "VIRTUAL_ENV": "/opt/venv",
}
# the image names the launcher hands to the server as declared
IMAGE_TO_SERVER = sorted(set(IMAGE_ENV) - {"PATH", "VIRTUAL_ENV", "NVIDIA_VISIBLE_DEVICES",
                                           "NVIDIA_DRIVER_CAPABILITIES", "NVIDIA_REQUIRE_CUDA"})
CHANGED_IN_IMAGE = ["LD_LIBRARY_PATH", "LIBRARY_PATH", "CUDA_VERSION", "NCCL_VERSION",
                    "NV_LIBNCCL_PACKAGE_VERSION", "NVARCH", "NVIDIA_PRODUCT_NAME",
                    "PYTHONDONTWRITEBYTECODE"]


def _defaults():
    import main
    from settings_dir import load_toml
    return load_toml(os.path.join(os.path.dirname(main.__file__), "defaults.toml"), "launch-defaults")


def in_image(w, *extra, environ=None):
    """A container launch whose environment is the image ENV (the venv the
    test world provides stands in for the image's)."""
    env = w.env(**{**IMAGE_ENV, "VIRTUAL_ENV": w.venv}) if environ is None else environ
    return w.resolve("--runtime", "container", "--tree", w.tree, *extra, environ=env)


def test_image_env_is_declared_and_classified():
    from envclass import ClassTable
    defaults = _defaults()
    assert defaults["runtime"]["container"]["image_env"] == IMAGE_ENV
    table = ClassTable(defaults["classes"], [])
    assert {n: table.classify(n) for n in IMAGE_ENV if table.classify(n) == "unclassified"} == {}


def test_container_with_exactly_the_image_env_adds_no_reason(w):
    base = in_image(w, environ=w.env(VIRTUAL_ENV=w.venv))
    d = in_image(w)
    assert reasons(d) == reasons(base)
    assert not [r for r in reasons(d) if r.startswith("ambient")], reasons(d)
    assert not set(IMAGE_ENV) & set(d["dropped"])
    assert d["functional_sha256"] == base["functional_sha256"]
    for doc in (d, base):
        assert {n: doc["child_env"][n]["value"] for n in IMAGE_TO_SERVER} == \
            {n: IMAGE_ENV[n] for n in IMAGE_TO_SERVER}
    assert {"LD_LIBRARY_PATH", "LIBRARY_PATH", "CUDA_VERSION", "PYTHONDONTWRITEBYTECODE"} <= \
        set(d["ambient_equal"])
    in_image(w, "--policy", "strict")


@pytest.mark.parametrize("name", CHANGED_IN_IMAGE)
def test_container_with_a_changed_image_value_fails_closed(w, name):
    env = w.env(**{**IMAGE_ENV, name: "/changed", "VIRTUAL_ENV": w.venv})
    d = in_image(w, environ=env)
    assert f"ambient {name}" in reasons(d)
    assert d["child_env"][name]["value"] == "/changed"
    assert d["validated"]["state"] != "VALIDATED"
    assert name in w.refused("--runtime", "container", "--tree", w.tree, "--policy", "strict",
                             environ=env)


def test_settings_value_of_an_image_name_wins(w):
    """A value the launch declares for an image name reaches the server, not the image's."""
    d = in_image(w, "--set", "env.PYTHONDONTWRITEBYTECODE=0")
    assert d["child_env"]["PYTHONDONTWRITEBYTECODE"]["value"] == "0"
    assert d["validated"]["state"] != "VALIDATED"


@pytest.mark.parametrize("name, value", [("LD_PRELOAD", "/x.so"), ("FOO_DEBUG", "1"),
                                         ("HF_TOKEN", "t"), ("VLLM_PORT", "1"), ("HOME", "/root"),
                                         ("SESSION_JOB", "j"), ("CUDA_MODULE_LOADING", "EAGER"),
                                         ("NCCL_DEBUG", "INFO"), ("PYTHONHASHSEED", "0"),
                                         ("NV_NEW_PACKAGE", "1")])
def test_image_env_names_outside_the_image_classes_are_refused(name, value):
    from envclass import ClassTable
    from errors import Refusal
    from resolver import _image_env
    table = ClassTable(_defaults()["classes"], [])
    code, values = _image_env(IMAGE_ENV, table)
    assert code == {n: IMAGE_ENV[n] for n in ("CUDA_HOME", "LD_LIBRARY_PATH", "LIBRARY_PATH", "VIRTUAL_ENV")}
    with pytest.raises(Refusal, match=name):
        _image_env({**IMAGE_ENV, name: value}, table)


@pytest.mark.parametrize("runtime", ["native", "container"])
def test_only_the_image_nv_names_are_inert(w, runtime):
    """The image's NV_* names are inert by name; another NV_* name is
    unclassified, so it never reaches the server."""
    from envclass import ClassTable
    table = ClassTable(_defaults()["classes"], [])
    assert {table.classify(n) for n in IMAGE_ENV if n.startswith("NV_")} == {"inert"}
    assert table.classify("NV_NEW_PACKAGE") == "unclassified"
    d = w.resolve("--runtime", runtime, "--tree", w.tree, environ=w.env(NV_NEW_PACKAGE="1"))
    assert "NV_NEW_PACKAGE" in d["dropped"] and "NV_NEW_PACKAGE" not in d["child_env"]
    assert d["validated"]["state"] == "VALIDATED" or runtime == "container"
