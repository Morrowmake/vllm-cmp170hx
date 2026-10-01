# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA/Mamba state indices: the debug range check
(``VLLM_GLM5_STATE_INDEX_CHECK``) and the block-table gather for masked rows.

CPU only (``CUDA_VISIBLE_DEVICES=""``); the gather kernel runs in the Triton
interpreter in a child process.
"""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.ampere_prefill import state_index_check as sic


def test_flag_declared_default_off(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv(sic.FLAG, raising=False)
    assert envs.environment_variables[sic.FLAG]() is False
    monkeypatch.setenv(sic.FLAG, "1")
    assert envs.environment_variables[sic.FLAG]() is True


def md(**kw):
    base = dict(num_prefills=0, num_decodes=0, num_spec_decodes=0, num_actual_tokens=0,
                has_initial_state=None, non_spec_state_indices_tensor=None,
                non_spec_query_start_loc=None, spec_state_indices_tensor=None,
                spec_query_start_loc=None, num_accepted_tokens=None)
    base.update(kw)
    return SimpleNamespace(**base)


def t(x):
    return torch.tensor(x, dtype=torch.int32)


def resumed_prefill(idx, cu, init):
    return md(num_prefills=1, num_decodes=len(cu) - 2, num_actual_tokens=cu[-1],
              has_initial_state=torch.tensor(init), non_spec_state_indices_tensor=t(idx),
              non_spec_query_start_loc=t(cu))


def test_resumed_prefill_in_range_passes():
    sic.check_kda_metadata("l", resumed_prefill([3, 7, 414], [0, 1, 2, 2306],
                                                [True, True, True]), 415, 2306)


@pytest.mark.parametrize("idx,needle", [([3, 7, 415], "outside [0, 415)"),
                                        ([3, -1, 9], "outside [0, 415)")])
def test_state_index_outside_pool_raises(idx, needle):
    with pytest.raises(sic.StateIndexError, match=needle.replace("[", r"\[").replace(")", r"\)")):
        sic.check_kda_metadata("l", resumed_prefill(idx, [0, 1, 2, 2306],
                                                    [True, True, True]), 415, 2306)


def test_short_initial_state_rows_and_token_overrun_raise():
    with pytest.raises(sic.StateIndexError, match="2 non-spec state rows for 3 sequences"):
        sic.check_kda_metadata("l", resumed_prefill([3, 7], [0, 1, 2, 2306], [True, True]),
                               415, 2306)
    with pytest.raises(sic.StateIndexError, match="> 2305 tokens"):
        sic.check_kda_metadata("l", resumed_prefill([3, 7, 9], [0, 1, 2, 2306],
                                                    [True, True, True]), 415, 2305)


def test_spec_rows_accepted_column_and_query_length():
    ok = md(num_spec_decodes=2, num_actual_tokens=16,
            spec_state_indices_tensor=t([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]]),
            num_accepted_tokens=t([8, 1]), spec_query_start_loc=t([0, 8, 16]))
    sic.check_kda_metadata("l", ok, 415, 16)
    bad = md(num_spec_decodes=2, num_actual_tokens=17,
             spec_state_indices_tensor=t([[1, 2, 3, 4], [9, 10, 11, 12]]),
             num_accepted_tokens=t([5, 1]), spec_query_start_loc=t([0, 8, 17]))
    with pytest.raises(sic.StateIndexError, match="num_accepted_tokens"):
        sic.check_kda_metadata("l", bad, 415, 17)


def _recover_md(state_cols, accepted, qsl, commit_state=None, request_indices=None,
                num_prefills=0, num_decodes=0):
    """GDN metadata as VLLM_GLM5_KDA_RECOVER builds it: num_accepted = 1 and
    only column 0 of the spec state table written (the rest is whatever the
    persistent torch.empty buffer holds)."""
    sp = t(state_cols)
    n = sp.shape[0]
    commit = SimpleNamespace(
        state_indices=sp[:, 0] if commit_state is None else t(commit_state),
        query_start_loc=t(qsl),
        request_indices=None if request_indices is None else t(request_indices),
        block_table=None, num_computed_tokens=None, block_size=None)
    return md(num_prefills=num_prefills, num_decodes=num_decodes, num_spec_decodes=n,
              num_actual_tokens=qsl[-1], spec_state_indices_tensor=sp,
              num_accepted_tokens=t(accepted), spec_query_start_loc=t(qsl),
              recover_commit=commit)


def test_recover_reads_only_column_zero():
    # Columns 1..7 hold uninitialised values (negative / beyond the pool): not
    # read under recover, so they must not fail the check.
    garbage = [-1765, 9_999_999, 0, 7, -3, 123456, 2]
    m = _recover_md([[5] + garbage, [9] + garbage], [1, 1], [0, 8, 16])
    sic.check_kda_metadata("l", m, 415, 16)
    # The same table without recover is a real violation.
    m.recover_commit = None
    with pytest.raises(sic.StateIndexError, match="spec state index"):
        sic.check_kda_metadata("l", m, 415, 16)


def test_recover_query_spans_token_columns_not_state_columns():
    # depth 7: 8-token verify per request against one state column
    sic.check_kda_metadata("l", _recover_md([[5] + [0] * 7], [1], [0, 8]), 415, 8)
    with pytest.raises(sic.StateIndexError, match="token columns"):
        sic.check_kda_metadata("l", _recover_md([[5] + [0] * 7], [1], [0, 9]), 415, 9)


def test_recover_column_zero_and_accepted_still_checked():
    with pytest.raises(sic.StateIndexError, match="spec state index"):
        sic.check_kda_metadata("l", _recover_md([[415] + [0] * 7], [1], [0, 8]), 415, 8)
    # the recover verify reads num_accepted = 1 only
    with pytest.raises(sic.StateIndexError, match="num_accepted_tokens"):
        sic.check_kda_metadata("l", _recover_md([[5] + [0] * 7], [3], [0, 8]), 415, 8)


def test_recover_commit_metadata_checked():
    with pytest.raises(sic.StateIndexError, match="recover commit state index"):
        sic.check_kda_metadata(
            "l", _recover_md([[5] + [0] * 7], [1], [0, 8], commit_state=[-1]), 415, 8)
    # mixed step: one prefill + one spec decode; batch row 2 does not exist
    m = _recover_md([[5] + [0] * 7], [1], [0, 8], request_indices=[2], num_prefills=1)
    m.non_spec_state_indices_tensor = t([3])
    m.non_spec_query_start_loc = t([0, 4])
    m.has_initial_state = torch.tensor([True])
    with pytest.raises(sic.StateIndexError, match="recover commit request index"):
        sic.check_kda_metadata("l", m, 415, 8)
    m.recover_commit.request_indices = t([1])
    sic.check_kda_metadata("l", m, 415, 8)


def test_recover_metadata_class_is_checked(monkeypatch):
    """Glm5KDARecoverMetadata is a GDNAttentionMetadata, so check_attn_metadata
    reaches it (and its recover_commit field) on every step."""
    from vllm.ampere_decode.kda_recover import Glm5KDARecoverMetadata
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    assert issubclass(Glm5KDARecoverMetadata, GDNAttentionMetadata)
    assert "recover_commit" in {f.name for f in
                                __import__("dataclasses").fields(Glm5KDARecoverMetadata)}


def test_align_copy_columns_and_block_ids():
    table = torch.zeros(4, 64, dtype=torch.int32)
    table[0, :3] = t([0, 12, 13])
    table[1, :2] = t([20, 414])
    idx = t([0, 1])
    # in range: row 0 copies col 1 -> col 2, row 1 col 0 -> col 1
    sic.check_align_copies("pre-copy", [table], 4, 415, idx, torch.tensor([1, 0]),
                           torch.tensor([2, 1]), torch.tensor([0, 0]),
                           torch.tensor([True, True]))
    with pytest.raises(sic.StateIndexError, match="dst column -1"):
        sic.check_align_copies("post-copy", [table], 4, 415, idx, torch.tensor([1, 0]),
                               torch.tensor([-1, 1]), torch.tensor([0, 0]),
                               torch.tensor([True, True]))
    with pytest.raises(sic.StateIndexError, match="src\\+bias column 64"):
        sic.check_align_copies("post-copy", [table], 4, 415, idx, torch.tensor([60, 0]),
                               torch.tensor([2, 1]), torch.tensor([4, 0]),
                               torch.tensor([True, True]))
    table[1, 1] = 415
    with pytest.raises(sic.StateIndexError, match="holds block 415"):
        sic.check_align_copies("pre-copy", [table], 4, 415, idx, torch.tensor([1, 0]),
                               torch.tensor([2, 1]), torch.tensor([0, 0]),
                               torch.tensor([True, True]))


def _post_kernel_ref(acc, st, new, bs):
    """postprocess_mamba_fused_kernel's decision (PRECOMPUTED_NEW_COMPUTED)."""
    running = new - acc + 1
    aligned = (new // bs) * bs
    if not aligned >= running:
        return None
    bias = aligned - running
    dst = aligned // bs - 1
    if st == dst and bias == 0:
        return None
    return st, dst, bias


def test_plans_match_kernel_decisions():
    g = torch.Generator().manual_seed(0)
    bs = 4608
    for _ in range(300):
        n = 6
        acc = torch.randint(1, 9, (8,), generator=g)
        new = torch.randint(1, 3 * bs, (8,), generator=g)
        new[:3] = torch.tensor([bs, 2 * bs, bs + 3])
        st = torch.randint(-1, 4, (8,), generator=g)
        idx = torch.tensor([0, 1, 2, -1, 5, 7], dtype=torch.int32)
        src, dst, bias, needs = sic.postcopy_plan(acc, st, new, idx, bs)
        for row in range(n):
            r = int(idx[row])
            want = None if r < 0 else _post_kernel_ref(int(acc[r]), int(st[r]), int(new[r]), bs)
            assert bool(needs[row]) == (want is not None)
            if want is not None:
                assert (int(src[row]), int(dst[row]), int(bias[row])) == want
        src_slots = torch.randint(-1, 4, (8,), generator=g)
        dst_slots = torch.randint(0, 4, (8,), generator=g)
        off = torch.randint(0, 7, (8,), generator=g)
        s, d, b, nd = sic.precopy_plan(src_slots, dst_slots, off, idx)
        for row in range(n):
            r = int(idx[row])
            want = r >= 0 and int(src_slots[r]) >= 0 and int(src_slots[r]) != int(dst_slots[r])
            assert bool(nd[row]) == want


def test_full_table_view_never_exceeds_storage():
    base = torch.zeros(4, 64, dtype=torch.int32)
    v = sic._full_table(base[:2], 8)
    assert v.shape == (4, 64)


# ---------------------------------------------------------------- gather kernel
_CHILD = r"""
import json, os, sys
os.environ["TRITON_INTERPRET"] = "1"
import torch
from vllm.v1.worker.gpu.block_table import _gather_block_tables_kernel
SENT = 123456789
rows, width = 4, 8
# the request-indexed table sits right after a sentinel row, as memory before a
# real table would: reading row -1 shows up as the sentinel
store = torch.full(((rows + 1) * width,), SENT, dtype=torch.int32)
src = store[width:].view(rows, width)
for r in range(rows):
    src[r] = torch.arange(r * 100, r * 100 + width, dtype=torch.int32)
num_blocks = torch.full((1, rows), 3, dtype=torch.int32)
nb_store = torch.full((rows + 1,), 5, dtype=torch.int32)
nb_store[1:] = num_blocks[0]
num_blocks = nb_store[1:].view(1, rows)
dst = torch.full((3, width), -7, dtype=torch.int32)
idx = torch.tensor([2, -1, 0], dtype=torch.int32)
ptr = lambda t: torch.tensor([t.data_ptr()], dtype=torch.uint64)
_gather_block_tables_kernel[(1, 3)](idx, ptr(src), ptr(dst), torch.tensor([width], dtype=torch.int64),
    num_blocks, num_blocks.stride(0), 3, BLOCK_SIZE=8)
print("RESULT " + json.dumps(dst.tolist()))
"""


def test_gather_masked_row_reads_nothing_before_the_table():
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-c", _CHILD], env=env, capture_output=True,
                       text=True, timeout=600)
    line = [x for x in r.stdout.splitlines() if x.startswith("RESULT ")]
    if r.returncode != 0 or not line:
        pytest.skip(f"Triton interpreter unavailable: {r.stderr[-600:]}")
    out = json.loads(line[-1][len("RESULT "):])
    assert out[0][:3] == [200, 201, 202]
    assert out[1] == [0] * 8          # masked row: null block, nothing read
    assert out[2][:3] == [0, 1, 2]
    assert 123456789 not in [x for row in out for x in row]


def test_gather_mapping_check():
    sic.check_gather_mapping(t([0, 3, 7]), 8)
    with pytest.raises(sic.StateIndexError, match="outside"):
        sic.check_gather_mapping(t([0, -1, 7]), 8)


def test_metadata_without_a_known_pool_fails(monkeypatch):
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    m = GDNAttentionMetadata(num_prefills=0, num_prefill_tokens=0, num_decodes=0,
                             num_decode_tokens=0, num_spec_decodes=0,
                             num_spec_decode_tokens=0, num_actual_tokens=0)
    cfg = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context={}))
    with pytest.raises(sic.StateIndexError, match="no state pool"):
        sic.check_attn_metadata({"model.layers.0.self_attn": m}, cfg)
    layer = SimpleNamespace(kv_cache=(torch.zeros(5, 2, 3), torch.zeros(5, 2, 2, 2)))
    cfg2 = SimpleNamespace(compilation_config=SimpleNamespace(
        static_forward_context={"model.layers.0.self_attn": layer}))
    sic.check_attn_metadata({"model.layers.0.self_attn": m}, cfg2)
