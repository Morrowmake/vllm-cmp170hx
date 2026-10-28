# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The replay tests' inputs (tests/kernels/replay_inputs.py): a family's
records under VLLM_SM80_REPLAY_DIR and the VLLM_SM80_REPLAY_LIMIT cap. CPU
only."""

import pytest

import vllm.envs as envs
from tests.kernels import replay_inputs as replay


@pytest.fixture
def replay_dir(monkeypatch, tmp_path):
    for family, names in (("kda_pre", ["p0_L1_b64_1.pt", "p0_L0_b64_0.pt", "notes.txt"]),
                          ("moe_pre", ["a.pt"]), ("once", ["w.pt"])):
        (tmp_path / family).mkdir()
        for name in names:
            (tmp_path / family / name).write_bytes(b"")
    monkeypatch.setenv("VLLM_SM80_REPLAY_DIR", str(tmp_path))
    return tmp_path


def test_unset_means_no_records(monkeypatch):
    monkeypatch.delenv("VLLM_SM80_REPLAY_DIR", raising=False)
    assert envs.VLLM_SM80_REPLAY_DIR == ""
    assert replay.root() == "" and replay.files("kda_pre") == []


def test_files_are_the_family_subdirectory(replay_dir):
    assert replay.root() == str(replay_dir)
    assert replay.files("kda_pre") == [str(replay_dir / "kda_pre" / n)
                                       for n in ("p0_L0_b64_0.pt", "p0_L1_b64_1.pt")]
    assert replay.files("kda_pre", "p0_L1_*.pt") == [str(replay_dir / "kda_pre/p0_L1_b64_1.pt")]
    assert replay.files("moe_pre") == [str(replay_dir / "moe_pre/a.pt")]
    assert replay.files("smla_pre") == []


@pytest.mark.parametrize("setting,expected", [(None, 7), ("0", 0), ("3", 3)])
def test_limit(monkeypatch, setting, expected):
    if setting is None:
        monkeypatch.delenv("VLLM_SM80_REPLAY_LIMIT", raising=False)
    else:
        monkeypatch.setenv("VLLM_SM80_REPLAY_LIMIT", setting)
    assert replay.limit(7) == expected


@pytest.mark.parametrize("old", ["KDA_PRE_CAPTURE_DIR", "SMLA_PRE_CAPTURE_DIR",
                                 "VLLM_GLM5_IDX_GLUE_CAPTURE", "KDA_FOLD_REPLAY_DIR",
                                 "VLLM_PP_MARLIN_CAPTURE_DIR", "VLLM_TP4_MARLIN_CAPTURE_DIR"])
def test_retired_directories_are_not_read(monkeypatch, tmp_path, old):
    (tmp_path / "kda_pre").mkdir()
    (tmp_path / "kda_pre" / "p0_L0_b64_0.pt").write_bytes(b"")
    monkeypatch.delenv("VLLM_SM80_REPLAY_DIR", raising=False)
    monkeypatch.setenv(old, str(tmp_path))
    assert replay.files("kda_pre") == []
