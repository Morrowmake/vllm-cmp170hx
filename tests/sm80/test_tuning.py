# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sm80 size cut-off table (vllm_sm80/tuning.py), on synthetic tables and
on the shipped table of the CMP 170HX. CPU only."""

import functools
import tomllib

import pytest

import vllm.envs as envs
from tests.sm80.tuning_rows import table_text, use_rows
from vllm_sm80 import tuning

CARD = "NVIDIA CMP 170HX"


class _Recorder:
    def __init__(self):
        self.lines = []

    def info(self, msg, *args):
        self.lines.append(msg % args)


@pytest.fixture
def fresh(monkeypatch):
    """Fresh caches for table() and the lookups, restored on teardown, and
    the INFO lines the module logs."""
    monkeypatch.setattr(tuning, "table", functools.cache(tuning.table.__wrapped__))
    monkeypatch.setattr(tuning, "_RESULTS", {})
    monkeypatch.delenv("VLLM_SM80_TUNING_FILE", raising=False)
    recorder = _Recorder()
    monkeypatch.setattr(tuning, "logger", recorder)
    return recorder


def test_card_file_name():
    assert tuning.card_file_name(CARD) == "nvidia-cmp-170hx.toml"
    assert tuning.card_file_name("NVIDIA A100-SXM4-80GB") == (
        "nvidia-a100-sxm4-80gb.toml"
    )


def test_tuning_file_flag_is_declared_and_off():
    assert "VLLM_SM80_TUNING_FILE" in envs.environment_variables
    assert envs.environment_variables["VLLM_SM80_TUNING_FILE"]() == ""


def test_most_specific_row_wins(monkeypatch, fresh):
    use_rows(
        monkeypatch,
        ("op_a", "p", {}, 1),
        ("op_a", "p", {"hidden": 4096}, 2),
        ("op_a", "p", {"hidden": 4096, "world_size": 4}, 3),
        ("op_a", "q", {"hidden": 4096}, 9),
    )
    assert tuning.lookup("op_a", "p", {"hidden": 4096, "world_size": 4}, 0) == 3
    assert tuning.lookup("op_a", "p", {"hidden": 4096, "world_size": 2}, 0) == 2
    assert tuning.lookup("op_a", "p", {"hidden": 5120, "world_size": 4}, 0) == 1
    assert tuning.lookup("op_a", "p", {}, 0) == 1
    assert tuning.lookup("op_a", "q", {"hidden": 4096, "top_k": 8}, 0) == 9


def test_every_key_field_must_match(monkeypatch, fresh):
    use_rows(monkeypatch, ("op_a", "p", {"hidden": 4096, "top_k": 8}, 5))
    assert tuning.lookup("op_a", "p", {"hidden": 4096}, 7) == 7
    assert tuning.lookup("op_a", "p", {"hidden": 4096, "top_k": 6}, 7) == 7
    assert tuning.lookup("op_a", "p", {"hidden": "4096", "top_k": 8}, 7) == 7
    assert tuning.lookup("op_a", "p", {"hidden": 4096, "top_k": 8}, 7) == 5
    assert tuning.lookup("op_b", "p", {"hidden": 4096, "top_k": 8}, 7) == 7


def test_string_key_fields(monkeypatch, fresh):
    use_rows(monkeypatch, ("op_a", "p", {"dtype": "bf16"}, 5))
    assert tuning.lookup("op_a", "p", {"dtype": "bf16"}, 0) == 5
    assert tuning.lookup("op_a", "p", {"dtype": "fp8"}, 0) == 0


def test_equally_specific_rows_are_an_error(monkeypatch, fresh):
    use_rows(
        monkeypatch,
        ("op_a", "p", {"hidden": 4096}, 1),
        ("op_a", "p", {"world_size": 4}, 2),
    )
    assert tuning.lookup("op_a", "p", {"hidden": 4096}, 0) == 1
    with pytest.raises(tuning.TuningTableError, match="matches two rows"):
        tuning.lookup("op_a", "p", {"hidden": 4096, "world_size": 4}, 0)


def test_default_logs_once_per_key(monkeypatch, fresh):
    use_rows(monkeypatch, ("op_a", "p", {"hidden": 4096}, 3))
    for _ in range(3):
        assert tuning.lookup("op_a", "p", {"hidden": 5120}, 11) == 11
        assert tuning.lookup("op_a", "p", {"hidden": 4096}, 11) == 3
    missing = [line for line in fresh.lines if "no row" in line]
    assert missing == [
        "sm80 tuning: no row for op_a/p at {hidden=5120} in <test rows>; "
        "using the default 11"
    ]
    found = [line for line in fresh.lines if "= 3" in line]
    assert len(found) == 1


def test_key_order_does_not_matter(monkeypatch, fresh):
    use_rows(monkeypatch, ("op_a", "p", {"a": 1, "b": 2}, 4))
    assert tuning.lookup("op_a", "p", {"b": 2, "a": 1}, 0) == 4


@pytest.mark.parametrize(
    "text,match",
    [
        ('schema = "sm80-tuning/2"\ncard = "x"\n', "schema"),
        ('schema = "sm80-tuning/1"\n', "card"),
        ('schema = "sm80-tuning/1"\ncard = "x"\nextra = 1\n', "unknown fields"),
        (
            'schema = "sm80-tuning/1"\ncard = "x"\n[[row]]\nop = "a"\n'
            'param = "p"\nkey = {}\nvalue = 1\n',
            "needs exactly",
        ),
        (
            'schema = "sm80-tuning/1"\ncard = "x"\n[[row]]\nop = "a"\n'
            'param = "p"\nkey = {}\nvalue = 1\nevidence = "e"\nnote = "n"\n',
            "needs exactly",
        ),
        (
            'schema = "sm80-tuning/1"\ncard = "x"\n[[row]]\nop = "a"\n'
            'param = "p"\nkey = {}\nvalue = true\nevidence = "e"\n',
            "value must be an integer",
        ),
        (
            'schema = "sm80-tuning/1"\ncard = "x"\n[[row]]\nop = "a"\n'
            'param = "p"\nkey = {}\nvalue = 1.5\nevidence = "e"\n',
            "value must be an integer",
        ),
        (
            'schema = "sm80-tuning/1"\ncard = "x"\n[[row]]\nop = "a"\n'
            'param = "p"\nkey = {hidden = 1.0}\nvalue = 1\nevidence = "e"\n',
            "integer or a string",
        ),
        (
            'schema = "sm80-tuning/1"\ncard = "x"\n[[row]]\nop = "a"\n'
            'param = "p"\nkey = {}\nvalue = 1\nevidence = ""\n',
            "evidence must be a non-empty string",
        ),
        ("schema = ", "<t>"),
    ],
)
def test_schema_faults_raise(text, match):
    with pytest.raises(tuning.TuningTableError, match=match):
        tuning.parse_table(text, "<t>")


def test_duplicate_rows_raise():
    text = table_text([("a", "p", {"hidden": 1}, 1), ("a", "p", {"hidden": 1}, 2)])
    with pytest.raises(tuning.TuningTableError, match="two rows"):
        tuning.parse_table(text, "<t>")


def test_no_cuda_device_means_no_rows(monkeypatch, fresh):
    monkeypatch.setattr(tuning, "_device_name", lambda: None)
    assert tuning.table().rows == {}
    assert tuning.lookup("dense_gemm", "thin_max_rows", {"gemm_n": 1}, 32) == 32


def test_card_without_table(monkeypatch, fresh, tmp_path):
    monkeypatch.setattr(tuning, "_device_name", lambda: "Some Card")
    monkeypatch.setattr(tuning, "TABLE_DIR", tmp_path)
    assert tuning.table().rows == {}
    assert any("no table for Some Card" in line for line in fresh.lines)


def test_card_table_must_name_its_card(monkeypatch, fresh, tmp_path):
    monkeypatch.setattr(tuning, "_device_name", lambda: "Some Card")
    monkeypatch.setattr(tuning, "TABLE_DIR", tmp_path)
    (tmp_path / "some-card.toml").write_text(table_text([], card="Other Card"))
    with pytest.raises(tuning.TuningTableError, match="Other Card"):
        tuning.table()


def test_device_table_is_loaded(monkeypatch, fresh):
    monkeypatch.setattr(tuning, "_device_name", lambda: CARD)
    loaded = tuning.table()
    assert loaded.card == CARD
    assert loaded.source.endswith("nvidia-cmp-170hx.toml")


def test_tuning_file_overrides_the_card_table(monkeypatch, fresh, tmp_path):
    path = tmp_path / "trial.toml"
    path.write_text(table_text([("op_a", "p", {}, 5)], card="Any Card"))
    monkeypatch.setattr(tuning, "_device_name", lambda: CARD)
    monkeypatch.setenv("VLLM_SM80_TUNING_FILE", str(path))
    assert tuning.lookup("op_a", "p", {"hidden": 1}, 0) == 5
    # Not the card's table: none of its rows are there.
    assert tuning.lookup("allreduce", "flags_max_bytes",
                         {"world_size": 4, "hidden": 5120}, 7) == 7
    banner = [line for line in fresh.lines if "VLLM_SM80_TUNING_FILE" in line]
    assert banner == [
        f"sm80 tuning table {path} (card Any Card, 1 rows) "
        "[VLLM_SM80_TUNING_FILE; unset to use the card's own table]"
    ]


def test_tuning_file_missing_raises(monkeypatch, fresh, tmp_path):
    monkeypatch.setenv("VLLM_SM80_TUNING_FILE", str(tmp_path / "none.toml"))
    with pytest.raises(FileNotFoundError):
        tuning.table()


def test_cache_clear_rereads(monkeypatch, fresh, tmp_path):
    path = tmp_path / "t.toml"
    path.write_text(table_text([("op_a", "p", {}, 1)]))
    monkeypatch.setenv("VLLM_SM80_TUNING_FILE", str(path))
    assert tuning.lookup("op_a", "p", {}, 0) == 1
    path.write_text(table_text([("op_a", "p", {}, 2)]))
    assert tuning.lookup("op_a", "p", {}, 0) == 1
    tuning.cache_clear()
    assert tuning.lookup("op_a", "p", {}, 0) == 2


# --- the shipped table -------------------------------------------------------


def _shipped():
    path = tuning.TABLE_DIR / tuning.card_file_name(CARD)
    return tuning.parse_table(path.read_text(), str(path))


def test_shipped_table_parses_for_its_card():
    path = tuning.TABLE_DIR / tuning.card_file_name(CARD)
    assert tomllib.loads(path.read_text())["schema"] == tuning.SCHEMA
    assert _shipped().card == CARD


# The value each served model ran with before the table, at the key its call
# site passes: the seeded rows must reproduce every one of them.
SERVED = [
    # GLM-5.3-Flash (TP4 and PP4)
    ("dense_gemm", "thin_max_rows", {"gemm_n": 4096, "gemm_k": 4096}, 32),
    ("mhc", "decode_v2_max_rows", {"hidden": 4096, "hc_mult": 4}, 32),
    ("mhc", "prefill_min_rows", {}, 384),
    ("moe_router", "route_v2_max_rows",
     {"n_experts": 288, "top_k": 8, "hidden": 4096}, 32),
    ("moe_router", "fused_route_max_rows", {"top_k": 8}, 8),
    ("attention_prefill", "sparse_mla_min_rows", {"head_dim": 512}, 384),
    ("attention_prefill", "sparse_mla_min_ctx",
     {"head_dim": 512, "index_width": 2176}, 4352),
    ("moe_prefill", "marlin_min_rows", {"gemm_n": 2048}, 384),
    ("moe_prefill", "marlin_min_rows", {"gemm_n": 512}, 384),
    ("moe_routed_gemm", "marlin_multi_max_rows", {"gemm_n": 2048}, 64),
    ("topk_select", "split_rows", {"top_k": 512, "indexer": "kpool"}, 8),
    ("allreduce", "flags_max_bytes", {"world_size": 4, "hidden": 4096}, 262144),
    ("allreduce", "host_max_bytes", {"world_size": 4, "hidden": 4096}, 524288),
    # Qwen3.8-Flash-Next (TP2)
    ("dense_gemm", "thin_max_rows", {"gemm_n": 512, "gemm_k": 2560}, 32),
    ("moe_routed_gemm", "exl3_chunk_rows", {"hidden": 2560}, 0),
    ("allreduce", "flags_max_bytes", {"world_size": 2, "hidden": 2560}, 262144),
    # DeepSeek-V4.1-Flash (TP4)
    ("moe_routed_gemm", "exl3_chunk_rows", {"hidden": 5120}, 2048),
    ("allreduce", "flags_max_bytes", {"world_size": 4, "hidden": 5120}, 524288),
    ("allreduce", "host_max_bytes", {"world_size": 4, "hidden": 5120}, 524288),
]


@pytest.mark.parametrize("op,param,key,value", SERVED)
def test_shipped_table_reproduces_the_served_values(
    monkeypatch, fresh, op, param, key, value
):
    monkeypatch.setattr(tuning, "_device_name", lambda: CARD)
    # A default nobody serves, so a missing row shows.
    assert tuning.lookup(op, param, key, -1) == value


def test_split_rows_of_one_indexer_never_reach_another(monkeypatch, fresh):
    """GLM's kpool and Qwen's QSA indexer both select 512 per row. The split
    tie fix was measured on GLM only: GLM's key gets its seeded 8 rows, Qwen's
    key no row, so the default 0 (one program per row, as Qwen always ran)."""
    monkeypatch.setattr(tuning, "_device_name", lambda: CARD)
    key = {"top_k": 512}
    lookup = tuning.lookup
    assert lookup("topk_select", "split_rows", dict(key, indexer="kpool"), 0) == 8
    assert lookup("topk_select", "split_rows", dict(key, indexer="qsa"), 0) == 0


def test_shipped_rows_have_evidence():
    for rows in _shipped().rows.values():
        for row in rows:
            assert row.evidence.strip()
