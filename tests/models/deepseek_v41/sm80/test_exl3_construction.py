# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and load the sm_80 DeepSeek-V4.1 EXL3 target and DSpark drafter on
CPU, per TP/EP rank.

The tiny checkpoint of ``exl3_fixture`` has the real checkpoint's tensor
names, dtypes and kinds (checked against the real headers in
``fixtures/exl3_headers.json`` below). For every rank at TP1 and TP4/EP4 the
EXL3 loader runs end to end (``load_exl3_weights``: every assigned packed
slice, every plain tensor through ``model.load_weights``). Its ledger then
accounts for every checkpoint tensor exactly once and no runtime parameter
stays unloaded. Across ranks every routed expert and ``wo_a`` group has
exactly one owner, and every other tensor is consumed by the target or the
drafter on every rank, never both. The checkpoint holds no Engram table
(``engram.embed``): the sm_80 Engram storage reads it from the base
checkpoint, so it is the one source neither model expects here.
"""

import json
import re
from collections import defaultdict

import pytest

from tests.models.deepseek_v41.sm80 import construct
from tests.models.deepseek_v41.sm80 import exl3_fixture as fixture
from tests.models.test_native_checkpoint_config_meta import (
    meta_platform,  # noqa: F401
)
from vllm.config import set_current_vllm_config


def _one_rank(name: str) -> bool:
    """Routed experts (EP) and wo_a groups (TP) live on one rank each."""
    return ".experts." in name or ".wo_a.slice." in name


def _pattern(name: str) -> str:
    return re.sub(r"\.\d+(?=\.)", ".N", name)


def test_tiny_checkpoint_has_the_real_tensor_kinds():
    headers = json.loads((fixture.FIXTURES / "exl3_headers.json").read_text())
    real = {(_pattern(n), dt) for n, (dt, _) in headers["tensors"].items()}
    real |= {
        (_pattern(t.format(e=0)), dt) for t, (dt, _, _) in headers["experts"].items()
    }
    tiny = {
        (_pattern(n), dt)
        for n, (dt, _) in fixture.checkpoint_tensors(fixture.tiny_config()).items()
    }
    assert tiny == real


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny")
    fixture.write_checkpoint(directory, fixture.tiny_config())
    return directory


def _load(monkeypatch, model, config) -> dict[str, tuple[str, str]]:
    """Run the EXL3 loader on a CPU model; return its ledger."""
    from vllm_sm80.kernels.exl3 import loader, ops, utils

    monkeypatch.setattr(ops, "ensure_extension", lambda: None)
    monkeypatch.setattr(loader, "require_cuda_parameters", lambda model: None)
    entries: dict[str, tuple[str, str]] = {}
    finish = utils.Exl3LoadLedger.finish

    def record(self):
        entries.update(self.entries)
        finish(self)

    # Undone after this load, so the next load records into its own ledger.
    with monkeypatch.context() as patch, set_current_vllm_config(config):
        patch.setattr(utils.Exl3LoadLedger, "finish", record)
        loader.load_exl3_weights(model, config.model_config)
    return entries


@pytest.mark.parametrize("tp", [1, 4])
def test_every_checkpoint_tensor_has_one_owner(
    monkeypatch,
    meta_platform,  # noqa: F811
    checkpoint,
    tp,
):
    from vllm_sm80 import plugin

    plugin._register_exl3()
    ranks = []
    for rank in range(tp):
        construct.set_groups(monkeypatch, tp, rank)
        construct.stub_engram_tables(monkeypatch, tp)
        target_config, draft_config = construct.vllm_configs(str(checkpoint), tp)
        target = construct.build_model(target_config)
        draft = construct.build_model(draft_config)
        tensors = set(target_config.quant_config.manifest.tensors)
        ranks.append(
            {
                "target": _load(monkeypatch, target, target_config),
                "draft": _load(monkeypatch, draft, draft_config),
            }
        )
    assert not [t for t in tensors if ".engram.embed" in t]
    assert [t for t in tensors if ".engram.wkv." in t]
    owner = defaultdict(list)
    for rank, result in enumerate(ranks):
        for role in ("target", "draft"):
            # The ledger accounts for every source tensor exactly once.
            assert set(result[role]) == tensors
        for name, (target_use, _) in result["target"].items():
            draft_use = result["draft"][name][0]
            uses = [target_use, draft_use]
            if _one_rank(name):
                if uses != ["unused", "unused"]:
                    owner[name].append(rank)
                continue
            # Everything else: exactly one model consumes it, on every rank.
            assert uses.count("unused") == 1, (name, uses)
            assert (draft_use != "unused") == name.startswith("mtp."), name
    assert set(owner) == {t for t in tensors if _one_rank(t)}
    assert all(len(ranks_) == 1 for ranks_ in owner.values())
    per_rank = defaultdict(int)
    for ranks_ in owner.values():
        per_rank[ranks_[0]] += 1
    assert sorted(per_rank) == list(range(tp))
    assert len(set(per_rank.values())) == 1
