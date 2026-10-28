# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-step driver for the RecoverSSM commit fold tests
(tests/kernels/test_ampere_kda_fold.py).

A World is one card's KDA state (recurrent + conv state of every layer, the
record pool) driven step by step through the engine's own pieces: the align
preprocess and pre-copy kernels, BuilderRecover.wrap, RecoverSSMState
(record_step: the fold gather; commit_step: the commit plan, conv compaction,
commit and align postprocess), FoldState.reset_slot. Only the verify is
pluggable: `ModelVerify` mirrors the contract of kda_decode_v2's RECOVER and
FOLD variants in torch (CPU, Triton interpreter for the rest), the GPU tests
plug in the real kernel.

Two worlds fed the same schedule, one with the fold and one without, must
agree on everything a consumer reads: the state and conv window every verify
starts from, the verify's outputs and records, and every block-boundary
checkpoint (what a prefix-cache hit or a preempted request resumes from) as
soon as its block is full and at the end.
"""

import dataclasses
import random
import types

import torch

H, D, WS_T = 16, 128, 8
NUM_SPEC = 7
SQL = NUM_SPEC + 1           # spec query length (num_spec + 1)
CONV_K = 4
SLEN = CONV_K - 1 + NUM_SPEC  # conv state rows
MAX_SEQS = 8
PREFILL_SLOT = MAX_SEQS - 1   # a slot that only ever holds a prefill row
DECODE_SLOTS = MAX_SEQS - 1


def env(fold: bool):
    """Environment for kda_recover_enabled / kda_fold_enabled on a CPU box;
    returns an undo callable."""
    import os

    from vllm_sm80.kernels import _platform

    keys = ("VLLM_AMPERE_DECODE_KERNELS", "VLLM_AMPERE_KDA_FOLD",
            "VLLM_AMPERE_KDA_RECOVER_OFF")
    saved = {k: os.environ.get(k) for k in keys}
    sm80 = _platform._SM80_CACHE
    os.environ["VLLM_AMPERE_DECODE_KERNELS"] = "1"
    os.environ["VLLM_AMPERE_KDA_FOLD"] = "1" if fold else "0"
    os.environ["VLLM_AMPERE_KDA_RECOVER_OFF"] = "0"
    _platform._SM80_CACHE = True

    def undo():
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _platform._SM80_CACHE = sm80

    return undo


def config(mode="align", block_size=16, tp=4, pp=1, dp=1):
    text = types.SimpleNamespace(linear_num_heads=H * tp, linear_head_dim=D)
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(
                architectures=["Glm5NextForConditionalGeneration"]),
            hf_text_config=text, max_model_len=262_144, dtype=torch.bfloat16),
        parallel_config=types.SimpleNamespace(
            pipeline_parallel_size=pp, tensor_parallel_size=tp,
            data_parallel_size=dp),
        scheduler_config=types.SimpleNamespace(max_num_seqs=MAX_SEQS),
        cache_config=types.SimpleNamespace(
            mamba_cache_mode=mode, use_replayssm=False, mamba_block_size=block_size,
            use_kda_recoverssm=False),
        compilation_config=types.SimpleNamespace(static_forward_context={}),
        num_speculative_tokens=NUM_SPEC)


def fma32(a, b, c):
    """fp32 a * b + c as the Triton interpreter evaluates tl.fma (two
    roundings); on the GPU both sides use the hardware fma."""
    return a * b + c


# ----------------------------------------------------------------- inputs
def seed(*key):
    """A seed from `key` that does not depend on the process (str hashing
    is salted per process)."""
    import zlib

    return zlib.crc32(repr(key).encode())


def _gen(*key):
    return torch.Generator().manual_seed(seed(*key))


def token_inputs(req, pos, layer, conv_dim):
    """Deterministic per-token verify inputs (the model's view)."""
    g = _gen("tok", req, pos, layer)
    k = torch.randn(H, D, generator=g)
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    e = torch.exp(-5.0 * torch.sigmoid(torch.randn(H, D, generator=g)))
    v = torch.randn(H, D, generator=g)
    q = torch.randn(H, D, generator=g)
    beta = torch.sigmoid(torch.randn(H, generator=g))
    x = torch.randn(conv_dim, generator=g)
    return k, e, v, q, beta, x


def conv_weights(layer, conv_dim):
    g = _gen("convw", layer)
    return torch.randn(CONV_K, conv_dim, generator=g) * 0.25


# ----------------------------------------------------------------- verify
class ModelVerify:
    """kda_decode_v2's state / conv / record contract (RECOVER, and FOLD when
    the metadata carries fold buffers), in torch:

      initial state from column 0 of the spec state indices; FOLD: replay the
      owed tokens from the row's record row (fma(c, k, h*e)), store the state
      back if anything was owed; conv window read at the owed position
      (RECOVER: 0); per token the correction, key and gate go to the record
      row (FOLD: the request slot, else the verify row); the conv window is
      written back as [the 2 kept rows, the step's tokens].
    """

    def __call__(self, world, layer, meta, rows, log):
        rec, conv, pool = world.rec[layer], world.conv[layer], world.pool[layer]
        fold = world.records[layer].fold
        w = world.conv_w[layer]
        for i_n, (req, pos0, T) in enumerate(rows):
            s = int(meta.spec_state_indices_tensor[i_n, 0])
            if s <= 0 or T == 0:
                continue
            if fold is not None:
                n_f = int(fold[0][i_n])
                r_row = max(int(fold[1][i_n]), 0)
            else:
                n_f, r_row = 0, i_n
            h = rec[s].clone()
            if fold is not None and n_f > 0:
                for t in range(n_f):
                    c, k, e = pool[0, r_row, t], pool[1, r_row, t], pool[2, r_row, t]
                    h = fma32(c[:, :, None], k[:, None, :], h * e[:, None, :])
                rec[s] = h
            off = max(n_f, 1) - 1 if fold is not None else 0
            win = conv[s].clone()
            log.append(("init", layer, req, pos0, h.clone(), win[off:off + CONV_K - 1].clone()))
            xs = []
            outs = []
            for t in range(T):
                k, e, v, q, beta, x = token_inputs(req, pos0 + t, layer, world.conv_dim)
                xs.append(x)
                u = torch.zeros(world.conv_dim)
                for j in range(CONV_K):
                    src = win[off + t + j] if t + j < CONV_K - 1 else xs[t + j - (CONV_K - 1)]
                    u = u + w[j] * src.float()
                v = v + u.sum()
                he = h * e[:, None, :]
                c = (v - (he * k[:, None, :]).sum(-1)) * beta[:, None]
                h = fma32(c[:, :, None], k[:, None, :], he)
                pool[0, r_row, t], pool[1, r_row, t], pool[2, r_row, t] = c, k, e
                outs.append((h * q[:, None, :]).sum(-1))
            log.append(("out", layer, req, pos0, torch.stack(outs),
                        pool[:, r_row, :T].clone()))
            keep = CONV_K - 2
            new = [win[off + 1 + o] for o in range(keep)] + [x.to(conv.dtype) for x in xs]
            conv[s, :keep + T] = torch.stack(new)


# ----------------------------------------------------------------- world
@dataclasses.dataclass
class Req:
    rid: int
    slot: int
    nc: int
    end: int
    cols: dict             # group -> list of block ids per column


class World:
    def __init__(self, *, fold, block_size=16, mode="align", groups=2, layers_per_group=1,
                 nblocks=400, conv_dim=4, device="cpu", verify=None, conv_dtype=torch.float32):
        from vllm_sm80.kernels.kda import recover as kr
        from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState

        kr.FoldState._states.clear()
        self.kr = kr
        self.fold, self.B, self.mode, self.dev = fold, block_size, mode, torch.device(device)
        self.align = mode == "align"
        self.G, self.LPG = groups, layers_per_group
        self.L = groups * layers_per_group
        self.conv_dim = conv_dim
        self.verify = verify or ModelVerify()
        dev = self.dev
        self.rec = [torch.zeros(nblocks, H, D, D, device=dev) for _ in range(self.L)]
        # SD layout [blocks, rows, dim] (as the recover control-path test)
        self.conv = [torch.zeros(nblocks, SLEN, conv_dim, device=dev, dtype=conv_dtype)
                     for _ in range(self.L)]
        self.conv_w = [conv_weights(l, conv_dim).to(dev) for l in range(self.L)]
        self.pool = torch.zeros(self.L, 3, MAX_SEQS, WS_T, H, D, device=dev)
        # the layers' records (the verify reads the fold buffers from them)
        self.records = [kr.layer_records(self.pool, l) for l in range(self.L)]
        self.free = list(range(1, nblocks))
        undo = env(fold)
        try:
            cfg = config(mode, block_size)
            cfg.compilation_config.static_forward_context.update({
                f"g{l // layers_per_group}.l{l % layers_per_group}":
                    types.SimpleNamespace(_kda_recover_records=self.records[l])
                for l in range(self.L)})
            self.builders = []
            for g in range(groups):
                names = [f"g{g}.l{i}" for i in range(layers_per_group)]
                b = types.SimpleNamespace(
                    decode_cudagraph_max_bs=MAX_SEQS * SQL, device=dev, num_spec=NUM_SPEC,
                    vllm_config=cfg, layer_names=names,
                    kv_cache_spec=types.SimpleNamespace(mamba_cache_mode=mode,
                                                        block_size=block_size))
                br = kr.BuilderRecover(b)
                br.context = self._context(g)
                self.builders.append(br)
            # Glm5NextModelState.__init__
            self.hooks = kr.kda_fold_hooks(cfg, dev)
        finally:
            undo()
        assert (self.builders[0].fold is not None) == fold == (self.hooks is not None)
        self.state = RecoverSSMState()
        i32 = dict(dtype=torch.int32, device=dev)
        self.cols = torch.full((MAX_SEQS,), -1, **i32)       # mamba state column per slot
        self.accepted = torch.ones(MAX_SEQS, **i32)
        self.nc_slot = torch.zeros(MAX_SEQS, **i32)
        self.reqs: dict = {}
        self.next_rid = 0
        self.log: list = []
        self.snaps: dict = {}

    # -- layers / context
    def layer_of(self, g, i):
        return g * self.LPG + i

    def _context(self, g):
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

        layers = []
        for i in range(self.LPG):
            l = self.layer_of(g, i)
            conv = self.conv[l].transpose(-1, -2) if is_conv_state_dim_first() else self.conv[l]
            layers.append(types.SimpleNamespace(kv_cache=(conv, self.rec[l]),
                                                _kda_recover_pool=self.pool,
                                                _kda_recover_index=l))
        return self.kr.KDARecoverCommitContext.create(layers, spec_query_len=SQL,
                                                      max_num_reqs=MAX_SEQS)

    # -- requests
    def add(self, slot, nc0, length, rid=None, tag="prefill"):
        if rid is None:
            rid = self.next_rid
            self.next_rid += 1
        r = Req(rid, slot, nc0, nc0 + length, {g: [] for g in range(self.G)})
        self._alloc(r, nc0)
        col = (nc0 - 1) // self.B if self.align else 0
        g0 = _gen(tag, rid)
        for g in range(self.G):
            blk = r.cols[g][col]
            for i in range(self.LPG):
                l = self.layer_of(g, i)
                self.rec[l][blk] = (torch.randn(H, D, D, generator=g0) * 0.5).to(self.dev)
                self.conv[l][blk] = torch.randn(SLEN, self.conv_dim, generator=g0).to(
                    self.dev, self.conv[l].dtype)
        self.reqs[slot] = r
        # the worker's add_requests removes every new request id first (the
        # streaming-update removal; a no-op for an id it does not hold)
        self._model_state_remove(r)
        self._model_state_add(slot, r)
        return r

    def _model_state_add(self, slot, r):
        """MambaHybridModelState.add_request, then Glm5NextModelState's hook."""
        self.accepted[slot] = 1
        self.cols[slot] = (r.nc - 1) // self.B if self.align else 0
        if self.hooks is not None:
            self.hooks.add_request(slot, str(r.rid), r.nc)

    def _model_state_remove(self, r):
        """Glm5NextModelState.remove_request."""
        if self.hooks is not None:
            self.hooks.remove_request(str(r.rid))

    def finish(self, slot, front=False):
        """A finished (or preempted) request: removed, its blocks freed (to
        the front of the free list with `front`: the next allocation reuses
        them)."""
        r = self.reqs.pop(slot)
        self._model_state_remove(r)
        freed = [blk for g in range(self.G) for c, blk in enumerate(r.cols[g])
                 if (r.rid, g, c) not in self.snaps]
        self.free = freed + self.free if front else self.free + freed
        return r

    def stream_readd(self, slot):
        """The worker's streaming-input update: the paused session is removed
        and re-added under the same id in one add_requests (same slot, blocks
        and computed tokens kept); its next chunk's prefill then reads the
        state block and conv window, logged here as that consumer's read."""
        r = self.reqs[slot]
        self._model_state_remove(r)
        self._model_state_add(slot, r)
        self.read_states("stream", only=slot)

    def read_states(self, tag, only=None):
        """Live requests' current state block and conv window, logged (what
        any reader of a request's state sees between steps)."""
        for slot, r in sorted(self.reqs.items()):
            if only is not None and slot != only:
                continue
            col = (r.nc - 1) // self.B if self.align else 0
            for g in range(self.G):
                blk = r.cols[g][col]
                for l in range(g * self.LPG, (g + 1) * self.LPG):
                    self.log.append((tag, l, r.rid, r.nc, self.rec[l][blk].clone(),
                                     self.conv[l][blk, :CONV_K - 1].clone()))

    def _alloc(self, r, upto_tokens):
        ncols = (upto_tokens - 1) // self.B + 1 if self.align else 1
        for g in range(self.G):
            while len(r.cols[g]) < ncols:
                blk = self.free.pop(0)
                # a reused block holds whatever its last owner left; the
                # prefill (or the pre-copy) writes it before anyone reads it
                for l in range(g * self.LPG, (g + 1) * self.LPG):
                    self.rec[l][blk].zero_()
                    self.conv[l][blk].zero_()
                r.cols[g].append(blk)

    # -- one step
    def step(self, sched, mixed_pos=None):
        """sched: [(slot, qlen, num_sampled)] in batch order; mixed_pos: batch
        position of a prefill row (a mixed step), None for a pure verify step."""
        kr, dev = self.kr, self.dev
        i32 = dict(dtype=torch.int32, device=dev)
        batch = [("spec", s, q, n) for s, q, n in sched]
        if mixed_pos is not None:
            batch.insert(mixed_pos, ("prefill", PREFILL_SLOT, 3, 0))
        nb = len(batch)
        for kind, s, q, _ in batch:
            if kind == "spec":
                self._alloc(self.reqs[s], self.reqs[s].nc + q)
        idx_mapping = torch.tensor([s for _, s, _, _ in batch], **i32)
        qsl = torch.tensor([0] + list(_cumsum([q for _, _, q, _ in batch])), **i32)
        # the prefill row is a fresh request every time (add_request seeds -1)
        self.cols[PREFILL_SLOT] = -1
        self.nc_slot[PREFILL_SLOT] = 0
        for _, s, _, _ in batch:
            if s != PREFILL_SLOT:
                self.nc_slot[s] = self.reqs[s].nc
        width = max([len(r.cols[0]) for r in self.reqs.values()] + [1]) + 1
        # Batch-order tables (the attention metadata's) and the per-request-slot
        # tables the align state copies read (row = slot; free slots NULL).
        tables, slot_tables = [], []
        for g in range(self.G):
            t = torch.zeros(nb, width, **i32)
            for b, (kind, s, _, _) in enumerate(batch):
                if kind == "spec":
                    cg = self.reqs[s].cols[g]
                    t[b, :len(cg)] = torch.tensor(cg, **i32)
            tables.append(t)
            st = torch.zeros(MAX_SEQS, width, **i32)
            for s, r in self.reqs.items():
                st[s, :len(r.cols[g])] = torch.tensor(r.cols[g], **i32)
            slot_tables.append(st)
        if self.align:
            self._preprocess(idx_mapping, qsl, slot_tables, nb)
        spec = [(b, s, q, n) for b, (kind, s, q, n) in enumerate(batch) if kind == "spec"]
        n_spec = len(spec)
        mask = torch.tensor([kind == "spec" for kind, _, _, _ in batch])
        attn, groups = {}, []
        for g in range(self.G):
            src = []
            for b, s, q, _ in spec:
                col = int(self.cols[s]) if self.align else 0
                src.append(int(tables[g][b, col]))
            meta = self._meta(g, src, [q for _, _, q, _ in spec], mixed_pos is not None,
                              tables[g], [self.reqs[s].nc for _, s, _, _ in spec], mask, batch)
            names = [f"g{g}.l{i}" for i in range(self.LPG)]
            for name in names:
                attn[name] = meta
            groups.append([types.SimpleNamespace(layer_names=names)])
        # MambaHybridModelState.prepare_attn, then Glm5NextModelState's hook
        self.state.record_step(attn, groups, for_capture=False)
        if self.hooks is not None:
            self.hooks.prepare_step(attn, idx_mapping, False)
        rows = [(self.reqs[s].rid, self.reqs[s].nc, q) for _, s, q, _ in spec]
        for g in range(self.G):
            for i in range(self.LPG):
                self.verify(self, self.layer_of(g, i), attn[f"g{g}.l{i}"], rows, self.log)
        # postprocess_state: scatter num_accepted, then the RecoverSSM commit
        nsamp = torch.tensor([n for _, _, _, n in batch], **i32)
        for b, (_, s, _, n) in enumerate(batch):
            self.accepted[s] = max(n, 1)
        self.state.commit_step(nsamp, idx_mapping,
                               state_indices=self.cols if self.align else None,
                               num_accepted_tokens=self.accepted)
        for _, s, _, n in spec:
            self.reqs[s].nc += n
        self._snapshot()

    def _meta(self, g, src, qlens, mixed, table, ncs, mask, batch):
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

        dev = self.dev
        n = len(src)
        i32 = dict(dtype=torch.int32, device=dev)
        ssi = torch.zeros(n, SQL, **i32)
        ssi[:, 0] = torch.tensor(src, **i32)
        sq = torch.tensor([0] + list(_cumsum(qlens)), **i32)
        meta = GDNAttentionMetadata(
            num_prefills=1 if mixed else 0, num_prefill_tokens=3 if mixed else 0,
            num_decodes=0, num_decode_tokens=0, num_spec_decodes=n,
            num_spec_decode_tokens=int(sum(qlens)), num_actual_tokens=int(sum(qlens)),
            spec_query_start_loc=sq, spec_state_indices_tensor=ssi,
            num_accepted_tokens=torch.ones(n, **i32))
        nc_batch = torch.tensor(
            [self.reqs[s].nc if kind == "spec" else 0 for kind, s, _, _ in batch], **i32)
        m = types.SimpleNamespace(block_table_tensor=table,
                                  compute_num_computed_tokens=lambda: nc_batch)
        return self.builders[g].wrap(meta, m, mask if mixed else None)

    def _preprocess(self, idx_mapping, qsl, tables, nb):
        from vllm.v1.worker.mamba_utils import (
            precopy_mamba_align_fused_kernel,
            preprocess_mamba_align_fused_kernel,
        )

        dev = self.dev
        i32 = dict(dtype=torch.int32, device=dev)
        src_col = torch.full((MAX_SEQS,), -9, **i32)
        src_off = torch.full((MAX_SEQS,), -9, **i32)
        preprocess_mamba_align_fused_kernel[(1,)](
            idx_mapping, self.cols, self.nc_slot, qsl, self.accepted, src_col, src_off,
            nb, BLOCK_SIZE=32, MAMBA_BLOCK_SIZE=self.B)
        states, groups = [], []
        for l in range(self.L):
            states += [self.conv[l], self.rec[l]]
            groups += [l // self.LPG] * 2
        i64 = dict(dtype=torch.int64, device=dev)
        precopy_mamba_align_fused_kernel[(nb, len(states))](
            self.cols, src_col, src_off,
            torch.tensor([t.data_ptr() for t in tables], **i64), tables[0].stride(0),
            torch.tensor([s.data_ptr() for s in states], **i64),
            torch.tensor([s.stride(0) * s.element_size() for s in states], **i64),
            torch.tensor([s.element_size() for s in states], **i32),
            torch.tensor([s.shape[2] if i % 2 == 0 else s[0].numel()
                          for i, s in enumerate(states)], **i64),
            torch.tensor([s.shape[1] if i % 2 == 0 else 0 for i, s in enumerate(states)], **i32),
            torch.tensor(groups, **i32), torch.zeros(len(states), **i32),
            torch.zeros(len(states), **i64), idx_mapping, nb,
            COPY_BLOCK_SIZE=256, CONV_STATE_DIM_FIRST=False)

    def _snapshot(self):
        """A block is a checkpoint a prefix-cache hit (or a resumed request) can
        read once its last token is computed: record it then."""
        if not self.align:
            return
        for r in self.reqs.values():
            for g in range(self.G):
                for c in range(r.nc // self.B):
                    key = (r.rid, g, c)
                    if key in self.snaps or c >= len(r.cols[g]):
                        continue
                    blk = r.cols[g][c]
                    self.snaps[key] = [(self.rec[l][blk].clone(), self.conv[l][blk, :CONV_K - 1].clone())
                                       for l in range(g * self.LPG, (g + 1) * self.LPG)]


def _cumsum(xs):
    t = 0
    for x in xs:
        t += x
        yield t


# ----------------------------------------------------------------- schedules
def random_schedule(seed, steps, block_size, *, near_boundary=False, p_mixed=0.3,
                    p_skip=0.15, p_zero=0.03, slots=DECODE_SLOTS, max_len=None):
    """A deterministic list of step events:
    ("add", slot, nc0, length) / ("finish", slot) / ("step", sched, mixed_pos)."""
    rng = random.Random(seed)
    events = []
    live = {}
    for _ in range(steps):
        for s in range(slots):
            if s not in live and rng.random() < 0.5:
                if near_boundary:
                    nc0 = block_size * rng.randint(1, 3) - rng.randint(0, 24)
                else:
                    nc0 = rng.randint(1, 3 * block_size)
                nc0 = max(nc0, 1)
                length = rng.randint(8, max_len or (4 * block_size if not near_boundary else 60))
                live[s] = [nc0, nc0 + length]
                events.append(("add", s, nc0, length))
        sched = []
        for s in sorted(live):
            if rng.random() < p_skip:
                continue
            q = rng.randint(1, SQL)
            r = rng.random()
            if r < p_zero:
                n = 0
            elif r < 0.4:
                n = 1                    # every draft rejected
            else:
                n = rng.randint(1, q)
            sched.append((s, q, n))
        if not sched:
            continue
        rng.shuffle(sched)
        mixed = rng.randint(0, len(sched)) if rng.random() < p_mixed else None
        events.append(("step", sched, mixed))
        for s, q, n in sched:
            live[s][0] += n
        for s in [s for s, v in live.items() if v[0] >= v[1]]:
            del live[s]
            events.append(("finish", s))
    return events


def replay_schedule(cm_rows, block_size, steps):
    """A schedule from captured commits: per captured step the real batch
    size, query lengths, sampled counts and block offsets (cm_rows: list of
    steps, each [(qlen, num_sampled, nc_mod_block)])."""
    events = []
    live = {}
    for k, rows in enumerate(cm_rows[:steps]):
        rows = rows[:DECODE_SLOTS]
        for s in [s for s in live if s >= len(rows)]:
            del live[s]
            events.append(("finish", s))
        sched = []
        for s, (q, n, ncm) in enumerate(rows):
            if s not in live:
                nc0 = block_size + ncm if ncm > 0 else block_size
                live[s] = nc0
                events.append(("add", s, nc0, 10**9))
            q = max(1, min(q, SQL))
            sched.append((s, q, max(0, min(n, q))))
            live[s] += sched[-1][2]
        events.append(("step", sched, None))
    return events


def run(world, events):
    for ev in events:
        if ev[0] == "add":
            _, s, nc0, length = ev
            world.add(s, nc0, length)
        elif ev[0] == "finish":
            world.finish(ev[1])
        else:
            _, sched, mixed = ev
            world.step(sched, mixed)
    return world


def diff(a, b):
    """Byte differences between what the two worlds' consumers read."""
    def same(x, y):
        if x.dtype == torch.float32:
            return torch.equal(x.view(torch.int32), y.view(torch.int32))
        return torch.equal(x, y)

    problems = []
    if len(a.log) != len(b.log):
        problems.append(f"log length {len(a.log)} != {len(b.log)}")
    for i, (x, y) in enumerate(zip(a.log, b.log)):
        if x[:4] != y[:4]:
            problems.append(f"log {i}: {x[:4]} != {y[:4]}")
            continue
        for u, v in zip(x[4:], y[4:]):
            if not same(u.cpu(), v.cpu()):
                problems.append(f"log {i} {x[0]} layer {x[1]} req {x[2]} pos {x[3]}")
                break
    if a.snaps.keys() != b.snaps.keys():
        problems.append("checkpoint sets differ")
    for key in a.snaps.keys() & b.snaps.keys():
        for (ra, ca), (rb, cb) in zip(a.snaps[key], b.snaps[key]):
            if not (same(ra.cpu(), rb.cpu()) and same(ca.cpu(), cb.cpu())):
                problems.append(f"checkpoint {key}")
    return problems


def checkpoints_unchanged(world):
    """The checkpoints have not been written after their block was full."""
    bad = []
    by_rid = {}
    for r in world.reqs.values():
        by_rid[r.rid] = r
    for (rid, g, c), saved in world.snaps.items():
        r = by_rid.get(rid)
        if r is None:
            continue
        blk = r.cols[g][c]
        for i, (rs, cs) in enumerate(saved):
            l = g * world.LPG + i
            if not torch.equal(world.rec[l][blk], rs) or not torch.equal(
                    world.conv[l][blk, :CONV_K - 1], cs):
                bad.append((rid, g, c))
    return bad
