"""CPU tests for DSpark SPS autotuning."""

import asyncio
import contextlib
import itertools
import json
import math
import multiprocessing
import os
import random
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import msgspec
import numpy as np
import torch
import torch.distributed as dist

from sglang.srt import runtime_context as rc
from sglang.srt.environ import envs
from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.srt.managers.io_struct import (
    GetInternalStateReqOutput,
    SetInternalStateReqOutput,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.sampling.sampling_params import _SAMPLING_EPS
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dspark_components import dspark_sps_autotune as at
from sglang.srt.speculative.dspark_components.dspark_planner import (
    ContextCost,
    DSparkScheduleConfig,
    DSparkVerifyPlanner,
    _additive_step_time_tensor,
    _context_cumulative_seconds,
    compute_verify_token_budget,
)
from sglang.srt.speculative.dspark_components.dspark_sps import (
    SpsAdditiveCostTable,
    load_sps_table_from_path,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

WIDTH = 6
GAMMA = WIDTH - 1
# Captured verify tiers without sub-width tiers: capture bs 1-8, 10, 12, 14, 16.
TIERS = [WIDTH * b for b in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16)]
BATCH_SIZES = [1, 2, 3, 4, 6, 8, 12, 16]


def _tier_of(m, tiers=TIERS):
    return next(t for t in tiers if t >= m)


def _truth_ms(bs, tier):
    """A step cost with per-batch and per-tier terms (the planner's additive
    model), both increasing."""
    return 8.0 + 0.45 * bs + 0.18 * tier + (1.5 if tier > 48 else 0.0)


def _cells(noise_ms=0.0, rounds=1, seed=0):
    rng = np.random.default_rng(seed)
    cells = at.plan_cells(tiers=TIERS, batch_sizes=BATCH_SIZES, verify_width=WIDTH)
    return [
        (b, t, (_truth_ms(b, t) + rng.normal(0, noise_ms)) / 1e3)
        for b, t in cells
        for _ in range(rounds)
    ]


class TestCellDesign(CustomTestCase):
    def test_default_batch_sizes(self):
        self.assertEqual(at.default_batch_sizes(16), BATCH_SIZES)
        self.assertEqual(at.default_batch_sizes(1), [1])
        self.assertEqual(at.default_batch_sizes(10), [1, 2, 3, 4, 6, 8, 10])
        self.assertEqual(at.default_batch_sizes(48)[-4:], [16, 24, 32, 48])

    def test_cells_cover_every_reachable_tier(self):
        cells = at.plan_cells(tiers=TIERS, batch_sizes=BATCH_SIZES, verify_width=WIDTH)
        by_bs = {}
        for b, t in cells:
            by_bs.setdefault(b, []).append(t)
        self.assertEqual(by_bs[1], [6])
        self.assertEqual(by_bs[8], [12, 18, 24, 30, 36, 42, 48])
        self.assertEqual(by_bs[16], [18, 24, 30, 36, 42, 48, 60, 72, 84, 96])
        self.assertEqual(len(cells), 42)
        # Every M a batch can verify maps to a measured tier.
        for b in BATCH_SIZES:
            for m in range(b, b * WIDTH + 1):
                self.assertIn((b, _tier_of(m)), cells)
        sub = at.plan_cells(
            tiers=[1, 2, 3, 4, 5] + TIERS, batch_sizes=[1, 2], verify_width=WIDTH
        )
        self.assertEqual(
            sub,
            [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)]
            + [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 12)],
        )

    def test_forced_budget_frac_pins_the_tier(self):
        for b, t in at.plan_cells(
            tiers=TIERS, batch_sizes=list(range(1, 17)), verify_width=WIDTH
        ):
            frac = at.forced_budget_frac(bs=b, tier=t, gamma=GAMMA)
            self.assertTrue(0.0 < frac <= 1.0)
            # The planner's forced budget, then the tier its M rounds up to.
            budget = int(frac * b * GAMMA)
            self.assertEqual(_tier_of(b + budget), t, (b, t, frac))

    def test_cell_step_seconds(self):
        records = [
            {
                "forward_ct": i,
                "num_running_reqs": 4,
                "num_verify_tokens": 24,
                "step_cpu_ms": 10.0 + (i % 3),
            }
            for i in range(20)
        ]
        records += [  # other shapes and unmeasured steps are ignored
            {
                "forward_ct": 100,
                "num_running_reqs": 3,
                "num_verify_tokens": 24,
                "step_cpu_ms": 99.0,
            },
            {
                "forward_ct": 101,
                "num_running_reqs": 4,
                "num_verify_tokens": 18,
                "step_cpu_ms": 99.0,
            },
            {
                "forward_ct": 102,
                "num_running_reqs": 4,
                "num_verify_tokens": 24,
                "step_cpu_ms": None,
            },
        ]
        random.Random(0).shuffle(records)
        s = at.cell_step_seconds(records, bs=4, tier=24, warmup_steps=5, steady_steps=9)
        self.assertAlmostEqual(s, 0.011)
        self.assertIsNone(
            at.cell_step_seconds(
                records, bs=4, tier=24, warmup_steps=5, steady_steps=16
            )
        )


class TestFit(CustomTestCase):
    def test_recovers_an_additive_tier_model(self):
        table, stats = at.fit_step_cost_table(_cells())
        self.assertLess(stats.rms_ms, 1e-6)
        SpsAdditiveCostTable.from_json(table.to_json())
        for b in BATCH_SIZES:
            for m in range(b, b * WIDTH + 1):
                self.assertAlmostEqual(
                    table.step_time(num_reqs=b, budget=m - b) * 1e3,
                    _truth_ms(b, _tier_of(m)),
                    places=6,
                )
            vec = _additive_step_time_tensor(
                table=table, num_requests=b, num_budgets=b * GAMMA + 1
            )
            for budget in range(b * GAMMA + 1):
                self.assertAlmostEqual(
                    float(vec[budget]),
                    table.step_time(num_reqs=b, budget=budget),
                    places=12,
                )

    def test_planner_lands_on_tier_edges(self):
        table, _ = at.fit_step_cost_table(_cells())
        cfg = DSparkScheduleConfig(gamma=GAMMA)
        torch.manual_seed(0)
        for b in BATCH_SIZES:
            survival = torch.cumprod(
                torch.rand(b, GAMMA, dtype=torch.float64) * 0.3 + 0.65, dim=1
            )
            d = compute_verify_token_budget(
                history_survival_probs=survival, sps_table=table, cfg=cfg
            )
            self.assertIn(b + d.budget, TIERS + [b * WIDTH])

    def test_one_outlier_cell_barely_moves_shared_tiers(self):
        cells = _cells()
        idx = next(i for i, c in enumerate(cells) if c[:2] == (4, 24))
        cells[idx] = (4, 24, cells[idx][2] + 0.004)  # +4 ms on one cell
        table, _ = at.fit_step_cost_table(cells)
        for b, t in [(8, 24), (12, 24), (16, 24), (6, 24)]:
            got = table.step_time(num_reqs=b, budget=t - b) * 1e3
            self.assertLess(abs(got - _truth_ms(b, t)), 0.35, (b, t, got))

    def test_costs_never_decrease_with_tier_or_batch(self):
        cells = _cells(noise_ms=1.5, seed=3)
        table, _ = at.fit_step_cost_table(cells)
        self.assertEqual(table.theta_seconds, sorted(table.theta_seconds))
        self.assertEqual(table.alpha_seconds, sorted(table.alpha_seconds))
        self.assertGreater(table.bias_seconds, 0)

    def test_repeats_reduce_to_the_median(self):
        cells = [(b, t, s) for b, t, s in _cells()]
        noisy = cells + [(b, t, s + 0.01) for b, t, s in cells[:3]]
        noisy += [(b, t, s) for b, t, s in cells[:3]]
        table, stats = at.fit_step_cost_table(noisy)
        self.assertLess(stats.rms_ms, 1e-6)

    def test_rejects_unidentifiable_or_invalid_cells(self):
        with self.assertRaises(ValueError):  # bs 1 and bs 16 share no tier
            at.fit_step_cost_table([(1, 6, 0.01), (16, 96, 0.04)])
        with self.assertRaises(ValueError):
            at.fit_step_cost_table([(1, 6, 0.01)])
        with self.assertRaises(ValueError):
            at.fit_step_cost_table([(1, 6, 0.01), (2, 6, float("nan"))])
        with self.assertRaises(ValueError):
            at.fit_step_cost_table([(0, 6, 0.01), (2, 6, 0.01)])

    def test_pava(self):
        self.assertEqual(at._pava([1, 3, 2, 4], [1, 1, 1, 1]), [1, 2.5, 2.5, 4])
        self.assertEqual(at._pava([5, 1], [3, 1]), [4, 4])


class TestCacheKeyAndFile(CustomTestCase):
    def tearDown(self):
        rc.reset_context()

    def _material_key(
        self,
        *,
        power_limit_mw=250000,
        tiers=TIERS,
        compile_bs=(),
        config=None,
        commit=None,
        fused_draft=False,
        **args,
    ):
        from transformers import PretrainedConfig

        from sglang.srt.mem_cache.unified_draft_pool import UnifiedDraftKVPool

        hf_config = PretrainedConfig(**(config or {"hidden_size": 4096}))
        hf_config._commit_hash = commit
        graph = SimpleNamespace(
            ragged_verify_mode=True,
            capture_num_tokens=list(tiers),
            capture_bs=[1, 2, 4, 8, 16],
            compile_bs=list(compile_bs),
            max_bs=16,
        )
        target = SimpleNamespace(
            decode_cuda_graph_runner=graph,
            max_running_requests=16,
            max_total_num_tokens=1 << 20,
            model_config=SimpleNamespace(context_len=1 << 20, hf_config=hf_config),
        )
        draft_pool = object()
        if fused_draft:
            draft_pool = UnifiedDraftKVPool.__new__(UnifiedDraftKVPool)
            draft_pool.host_allocator = object()
        draft = SimpleNamespace(
            token_to_kv_pool=draft_pool, decode_cuda_graph_runner=None
        )
        rc.reset_context()
        rc.publish(ServerArgs(model_path="dummy"), role="test")
        if args:
            rc.get_context().override("test", **args)
        props = SimpleNamespace(
            name="GPU",
            major=12,
            minor=0,
            total_memory=1 << 36,
            multi_processor_count=188,
        )
        parallel = SimpleNamespace(
            tp_group=SimpleNamespace(world_size=4, rank_in_group=0)
        )
        with (
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("torch.cuda.device_count", return_value=4),
            mock.patch("torch.cuda.get_device_properties", return_value=props),
            mock.patch.object(
                at, "_nvml_material", return_value={"power_limit_mw": power_limit_mw}
            ),
            mock.patch.object(at, "package_versions", return_value={}),
            mock.patch.object(at, "source_digest", return_value="src"),
            mock.patch(
                "sglang.srt.runtime_context.get_parallel", return_value=parallel
            ),
        ):
            material, _, _ = at.build_startup_material(
                planner=SimpleNamespace(
                    schedules_verify_budget=True,
                    is_compact_mode=True,
                    mode_value="compact",
                ),
                target_model_runner=target,
                draft_model_runner=draft,
                gamma=GAMMA,
                verify_num_draft_tokens=WIDTH,
                simulate_acc_len=0.0,
            )
        return at.cache_key(material)

    def test_startup_material_tracks_the_configuration(self):
        base = self._material_key()
        self.assertEqual(self._material_key(), base)
        changes = [
            dict(power_limit_mw=300000),
            dict(tiers=TIERS[:-1]),
            dict(compile_bs=[1, 2]),
            dict(torch_compile_max_bs=8),
            # Same Hub id, different loaded config or resolved revision.
            dict(config={"hidden_size": 2048}),
            dict(commit="abc"),
            dict(tokenizer_path="/other"),
            dict(skip_tokenizer_init=True),
            dict(disable_radix_cache=True),
            dict(enable_w4a4_mxfp4_megamoe=True),
            dict(speculative_enable_w4a4_mxfp4_megamoe=True),
            dict(speculative_draft_model_quantization="fp8"),
            dict(speculative_moe_a2a_backend="deepep"),
            dict(boundary_reduction="ar"),
            dict(speculative_boundary_reduction="ar"),
            dict(speculative_use_block_verification=True),
            # The same pool capacity and graphs with a fused vs private draft.
            dict(fused_draft=True),
            dict(enable_unified_memory=True),
            dict(enable_page_major_kv_layout=True),
            dict(dcp_size=2),
        ]
        for change in changes:
            with self.subTest(change=change):
                self.assertNotEqual(self._material_key(**change), base)

    def test_key_is_order_independent(self):
        a = {"device": {"name": "GPU", "cc": [12, 0]}, "tiers": [6, 12]}
        b = {"tiers": [6, 12], "device": {"cc": [12, 0], "name": "GPU"}}
        self.assertEqual(at.cache_key(a), at.cache_key(b))

    def test_env_digest_ignores_per_run_and_location_variables(self):
        base = {"SGLANG_RAGGED_VERIFY_MODE": "compact", "NCCL_ALGO": "Ring"}
        digest = at.env_digest(base)
        for noise in (
            {"SGLANG_RUN_ID": "abc"},
            {"SGLANG_CACHE_DIR": "/x"},
            {"SGLANG_PORT": "30000"},
            {"SGLANG_HOST_IP": "10.0.0.1"},
            {"SGLANG_DSPARK_SPS_AUTOTUNE_CACHE_DIR": "/y"},
            {"HOME": "/home/x"},
        ):
            self.assertEqual(at.env_digest({**base, **noise}), digest, noise)
        for name, value in (
            ("SGLANG_ENABLE_SM120_MXFP8_SKINNY_GEMM", "0"),
            ("SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", "1"),
            ("SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT", "sharded"),
            ("NCCL_GRAPH_MIXING_SUPPORT", "0"),
        ):
            self.assertNotEqual(at.env_digest({**base, name: value}), digest, name)

    def test_source_digest_tracks_content(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "srt"
            root.mkdir()
            (root / "a.py").write_text("x = 1\n")
            (root / "notes.txt").write_text("ignored")
            first = at.source_digest([root, Path(d) / "missing"])
            (root / "notes.txt").write_text("still ignored")
            self.assertEqual(at.source_digest([root]), first)
            (root / "a.py").write_text("x = 2\n")
            self.assertNotEqual(at.source_digest([root]), first)

    def test_payload_round_trip_and_rejections(self):
        table, _ = at.fit_step_cost_table(_cells())
        key = at.cache_key({"k": 1})
        payload = at.encode_cache_payload(table=table, key=key, metadata={"x": 1})
        self.assertEqual(at.decode_cache_payload(payload, key=key), table)
        self.assertIsNone(at.decode_cache_payload(payload, key=at.cache_key({"k": 2})))
        self.assertIsNone(at.decode_cache_payload("not json", key=key))
        self.assertIsNone(at.decode_cache_payload("[]", key=key))
        stale = json.loads(payload)
        stale[at.METADATA_FIELD]["schema"] = 1
        self.assertIsNone(at.decode_cache_payload(json.dumps(stale), key=key))
        broken = json.loads(payload)
        broken["m_probes"][1] = broken["m_probes"][0]
        self.assertIsNone(at.decode_cache_payload(json.dumps(broken), key=key))

    def test_cache_file_loads_as_a_configured_table(self):
        table, _ = at.fit_step_cost_table(_cells())
        key = at.cache_key({"k": 1})
        with tempfile.TemporaryDirectory() as d:
            path = at.cache_path_for(d, key)
            at.write_cache_atomic(
                path, at.encode_cache_payload(table=table, key=key, metadata={})
            )
            self.assertEqual(load_sps_table_from_path(str(path)), table)

    def test_atomic_write_replaces_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sub" / "t.json"
            at.write_cache_atomic(path, "one")
            at.write_cache_atomic(path, "two")
            self.assertEqual(path.read_text(), "two")
            with mock.patch.object(at.os, "replace", side_effect=OSError("boom")):
                with self.assertRaises(OSError):
                    at.write_cache_atomic(path, "three")
            self.assertEqual(path.read_text(), "two")
            self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ["t.json"])


PARAMS = at.AutotuneParams(passes=2, warmup_steps=4, steady_steps=12)


def _prepare(material=None, cells=None, params=PARAMS):
    return lambda: (
        material
        or {
            "gpu": "test",
            "graphs": {"verify_tiers": TIERS},
            "dspark": {
                "gamma": GAMMA,
                "verify_num_draft_tokens": WIDTH,
                "max_new_tokens": 500,
            },
        },
        cells or at.plan_cells(tiers=TIERS, batch_sizes=[1, 2, 4], verify_width=WIDTH),
        params,
    )


def _pending(cache_dir):
    return at.agree_on_startup(
        mode=at.SpsAutotuneMode.AUTO,
        explicit_table_path=None,
        cache_dir=cache_dir,
        group=None,
        prepare=_prepare(),
    )[1]


def _install_payload(prepared, table=None, key=None):
    table = table or at.fit_step_cost_table(_cells())[0]
    return json.dumps(
        {
            "key": key or prepared.key,
            "table": json.loads(table.to_json()),
            "metadata": {"measure_seconds": 1.0},
        }
    )


class TestStartupAgreementSingleRank(CustomTestCase):
    def _agree(self, mode, cache_dir, explicit=None, prepare=None):
        return at.agree_on_startup(
            mode=mode,
            explicit_table_path=explicit,
            cache_dir=cache_dir,
            group=None,
            prepare=prepare or _prepare(),
        )

    def test_off_explicit_skip(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(
                self._agree(at.SpsAutotuneMode.OFF, d), (None, None, "off")
            )
            self.assertEqual(
                self._agree(at.SpsAutotuneMode.AUTO, d, explicit="/t.json"),
                (None, None, "explicit-table"),
            )

            def skip():
                raise at.SpsAutotuneSkip("static mode")

            self.assertEqual(
                self._agree(at.SpsAutotuneMode.AUTO, d, prepare=skip),
                (None, None, "skipped"),
            )

    def test_miss_is_pending_then_install_caches_and_hit_loads(self):
        with tempfile.TemporaryDirectory() as d:
            table, prepared, source = self._agree(at.SpsAutotuneMode.AUTO, d)
            self.assertEqual((table, source), (None, "pending"))
            status = prepared.status()
            self.assertEqual(status["gamma"], GAMMA)
            self.assertEqual(status["cells"][0], [1, 6])
            installed = []
            ok = at.install_measured_table(
                payload=_install_payload(prepared),
                prepared=prepared,
                group=None,
                install=lambda t, src: installed.append((t, src)),
            )
            self.assertTrue(ok)
            self.assertEqual(len(installed), 1)
            meta = json.loads(prepared.cache_path.read_text())[at.METADATA_FIELD]
            self.assertEqual(meta["measure_seconds"], 1.0)
            self.assertEqual(meta["material"]["gpu"], "test")
            table, prepared2, source = self._agree(at.SpsAutotuneMode.AUTO, d)
            self.assertEqual((source, prepared2), ("cache", None))
            self.assertEqual(table, installed[0][0])
            table, prepared3, source = self._agree(
                at.SpsAutotuneMode.FORCE, d, explicit="/t.json"
            )
            self.assertEqual((table, source), (None, "pending"))
            for corrupt in (b"{", b"\xff\xfe"):
                prepared.cache_path.write_bytes(corrupt)
                self.assertEqual(self._agree(at.SpsAutotuneMode.AUTO, d)[2], "pending")

    def test_install_rejects_bad_requests_and_survives_unwritable_cache(self):
        with tempfile.TemporaryDirectory() as d:
            _, prepared, _ = self._agree(at.SpsAutotuneMode.AUTO, d)
            installed = []
            for payload in (
                None,
                "{",
                _install_payload(prepared, key="other"),
                json.dumps({"key": prepared.key, "table": {"bias_seconds": 1}}),
            ):
                self.assertFalse(
                    at.install_measured_table(
                        payload=payload,
                        prepared=prepared,
                        group=None,
                        install=lambda t, s: installed.append(t),
                    )
                )
            self.assertEqual(installed, [])
            blocker = Path(d) / "file"
            blocker.write_text("")
            prepared.cache_path = at.cache_path_for(str(blocker), prepared.key)
            self.assertTrue(
                at.install_measured_table(
                    payload=_install_payload(prepared),
                    prepared=prepared,
                    group=None,
                    install=lambda t, s: installed.append(t),
                )
            )
            self.assertEqual(len(installed), 1)


class FakeServer:
    """A tokenizer-manager stand-in over a simulated DSpark scheduler: one
    decode step every ``truth(bs, tier)`` seconds of simulated time, the
    pinned budget deciding the tier, per-step records as the info dumper
    writes them."""

    def __init__(
        self,
        truth,
        *,
        pending=True,
        stall=(),
        gamma=GAMMA,
        context_ms_per_ktok=0.0,
        long_context=0,
        tokenizer=None,
        long_context_far=0,
        context_knee=float("inf"),
        context_ms_per_ktok_beyond=0.0,
        batch_sizes=BATCH_SIZES,
    ):
        self.truth, self.gamma, self.stall = truth, gamma, set(stall)
        # Extra ms per verify token per 1000 context tokens (attention/indexer).
        self.context_ms_per_ktok = context_ms_per_ktok
        self.long_context = long_context
        self.long_context_far = long_context_far
        # Past the knee each context token costs the "beyond" slope instead.
        self.context_knee = context_knee
        self.context_ms_per_ktok_beyond = context_ms_per_ktok_beyond
        if tokenizer is not None:
            self.tokenizer = tokenizer
        self.batch_sizes = batch_sizes
        self.ctx = 0.0
        self.steps_since_load = 0
        self.now = 0.0
        self.frac = None
        self.recording = False
        self.records_list = []
        self.forward_ct = 0
        self.bs = 0
        self.next_step = None
        self.prefill_steps = 0
        self.loads = []
        self.events = []
        self.table_payload = "unset"
        self.pending = pending
        self.kv_tokens = 0
        self.model_config = SimpleNamespace(vocab_size=129280)

    # simulated time ---------------------------------------------------
    def clock(self):
        return self.now

    async def sleep(self, dt):
        self._advance(self.now + dt)
        await asyncio.sleep(0)

    def _tier(self):
        budget = int(self.frac * self.bs * self.gamma) if self.frac else 0
        return _tier_of(self.bs + budget)

    def _step_ms(self, tier):
        # Context grows by one committed (bonus) token per step after the prompt.
        ctx = self.ctx + self.steps_since_load
        near = min(ctx, self.context_knee)
        far = max(ctx - self.context_knee, 0.0)
        per_token = (
            self.context_ms_per_ktok * near + self.context_ms_per_ktok_beyond * far
        ) / 1000.0
        return self.truth(self.bs, tier) + per_token * tier

    def _advance(self, until):
        while self.bs and self.next_step is not None and self.next_step <= until:
            tier = self._tier()
            if (self.bs, tier) in self.stall:
                self.next_step = None
                break
            step_ms = None if self.prefill_steps else self._step_ms(tier)
            self.prefill_steps = max(0, self.prefill_steps - 1)
            if self.recording:
                self.records_list.append(
                    {
                        "forward_ct": self.forward_ct,
                        "num_running_reqs": self.bs,
                        "num_verify_tokens": tier,
                        "step_cpu_ms": step_ms,
                    }
                )
            self.next_step += self._step_ms(tier) / 1e3
            self.forward_ct += 1
            self.steps_since_load += 1
        self.now = until

    # tokenizer manager API -------------------------------------------
    async def get_internal_state(self):
        return [self._internal_state()]

    async def set_internal_state(self, req):
        return [self._apply(req)]

    def _internal_state(self):
        state = {}
        if self.recording:
            state["dspark_info_record"] = {"records": list(self.records_list)}
        if self.pending:
            cells = at.plan_cells(
                tiers=TIERS, batch_sizes=self.batch_sizes, verify_width=WIDTH
            )
            state[at.STATUS_FIELD] = {
                "pending": True,
                "key": "k",
                "cells": [list(c) for c in cells],
                "verify_tiers": TIERS,
                "long_context": self.long_context,
                "long_context_far": self.long_context_far,
                "long_cells": (
                    [list(c) for c in at.plan_long_cells(cells)]
                    if self.long_context
                    else []
                ),
                "gamma": self.gamma,
                "verify_num_draft_tokens": WIDTH,
                "max_new_tokens": 777,
                "kv_tokens": self.kv_tokens,
                "params": {
                    **msgspec.to_builtins(PARAMS),
                    "batch_sizes": [],
                },
            }
        elif self.table_payload not in ("unset", None):
            state[at.STATUS_FIELD] = {"pending": False, "installed": "x"}
        return state

    def _apply(self, req):
        for key, value in req.server_args.items():
            self.events.append((key, value))
            if key == "dspark_force_budget_frac":
                self.frac = value
            elif key == "dspark_clear_info_records" and value:
                self.records_list = []
            elif key == at.RECORD_KEY:
                self.recording = value
                self.records_list = []
            elif key == at.TABLE_KEY:
                self.table_payload = value
                self.pending = False
                self.recording = False
        return True

    def generate_request(self, obj, request):
        self.loads.append(obj)
        self.bs = len(obj.input_ids)
        self.ctx = sum(len(ids) for ids in obj.input_ids) / self.bs
        self.steps_since_load = 0
        self.prefill_steps = 2
        self.next_step = self.now
        self.aborted = asyncio.Event()

        async def gen():
            await self.aborted.wait()
            yield {"meta_info": {"finish_reason": "abort"}}

        return gen()

    def abort_request(self, rid="", abort_all=False):
        self.bs = 0
        self.next_step = None
        self.aborted.set()

    async def flush_cache(self, timeout_s=None):
        return SimpleNamespace(success=True)


class CommServer(FakeServer):
    """FakeServer behind the tokenizer manager's internal-state calls and
    communicators; replies are withheld while ``hold_get`` / ``hold_set``
    return True."""

    get_internal_state = TokenizerControlMixin.get_internal_state
    set_internal_state = TokenizerControlMixin.set_internal_state

    def __init__(self):
        super().__init__(_truth_ms)
        self.hold_get = lambda server: False
        self.hold_set = lambda req: False
        self.held = []
        self.get_internal_state_communicator = FanOutCommunicator(self._get, 1)
        self.set_internal_state_communicator = FanOutCommunicator(self._set, 1)

    def auto_create_handle_loop(self):
        pass

    def _get(self, req):
        reply = GetInternalStateReqOutput(internal_state=self._internal_state())
        if self.hold_get(self):
            self.held.append(reply)
        else:
            self._reply(self.get_internal_state_communicator, reply)

    def _set(self, req):
        if not self.hold_set(req):
            reply = SetInternalStateReqOutput(updated=self._apply(req))
            self._reply(self.set_internal_state_communicator, reply)

    @staticmethod
    def _reply(communicator, reply):
        asyncio.get_running_loop().call_soon(communicator.handle_recv, reply)

    def release(self):
        held, self.held = self.held, []
        for reply in held:
            self.get_internal_state_communicator.handle_recv(reply)


def _status(server):
    return asyncio.run(server.get_internal_state())[0][at.STATUS_FIELD]


class TestMeasurement(CustomTestCase):
    def _run(self, server, status=None):
        status = status or _status(server)

        async def go():
            return await at.run_measurement(
                at.MeasureClient(server),
                status,
                vocab_size=129280,
                sleep=server.sleep,
                clock=server.clock,
            )

        return asyncio.run(go())

    def test_measures_every_cell_and_recovers_the_step_costs(self):
        server = FakeServer(_truth_ms)
        request = self._run(server)
        table = SpsAdditiveCostTable(**request["table"])
        for b in BATCH_SIZES:
            for m in range(b, b * WIDTH + 1):
                self.assertAlmostEqual(
                    table.step_time(num_reqs=b, budget=m - b) * 1e3,
                    _truth_ms(b, _tier_of(m)),
                    places=4,
                )
        cells = at.plan_cells(tiers=TIERS, batch_sizes=BATCH_SIZES, verify_width=WIDTH)
        self.assertEqual(len(server.loads), 1 + 2 * len(cells))
        self.assertEqual([len(l.input_ids) for l in server.loads[1:4]], [1, 2, 2])
        self.assertEqual(
            [len(l.input_ids) for l in server.loads[-3:]], [2, 2, 1]
        )  # second pass runs backwards
        params = server.loads[1].sampling_params
        self.assertTrue(params["ignore_eos"])
        self.assertEqual(params["max_new_tokens"], 777)
        self.assertEqual(params["temperature"], 1.0)
        self.assertEqual(server.events[0], (at.RECORD_KEY, True))
        self.assertIn(("dspark_force_budget_frac", None), server.events[-2:])
        self.assertEqual(server.events[-1], (at.RECORD_KEY, False))
        self.assertEqual(request["metadata"]["failed_cells"], [])
        self.assertEqual(len(request["metadata"]["step_ms_by_bs_tier"]), len(cells))

    def test_noisy_steps_still_fit_within_noise(self):
        rng = random.Random(1)
        server = FakeServer(lambda b, t: _truth_ms(b, t) + rng.gauss(0, 0.8))
        table = SpsAdditiveCostTable(**self._run(server)["table"])
        for b, t in [(1, 6), (8, 24), (8, 48), (16, 96)]:
            got = table.step_time(num_reqs=b, budget=t - b) * 1e3
            self.assertLess(abs(got - _truth_ms(b, t)), 0.6, (b, t, got))

    def test_a_stalled_cell_times_out_and_the_rest_still_fit(self):
        server = FakeServer(_truth_ms, stall={(8, 30)})
        request = self._run(server)
        self.assertEqual(request["metadata"]["failed_cells"], [[8, 30]])
        table = SpsAdditiveCostTable(**request["table"])
        self.assertAlmostEqual(
            table.step_time(num_reqs=8, budget=22) * 1e3, _truth_ms(8, 30), places=3
        )

    def test_failures_restore_the_scheduler_and_raise(self):
        server = FakeServer(_truth_ms)
        status = _status(server)
        status["params"] = {**status["params"], "max_seconds": 1.0}
        with self.assertRaises(TimeoutError):
            self._run(server, status)
        self.assertEqual(server.frac, None)
        self.assertFalse(server.recording)

    def test_requests_are_sized_to_the_kv_pool(self):
        server = FakeServer(_truth_ms)
        server.kv_tokens = 10000
        self._run(server)
        limits = {
            len(l.input_ids): l.sampling_params["max_new_tokens"] for l in server.loads
        }
        self.assertEqual(limits[1], 777)
        self.assertEqual(limits[16], 8000 // 16 - PARAMS.input_len)
        server = FakeServer(_truth_ms)
        server.kv_tokens = 400  # 12 and 16 requests cannot hold their steady steps
        with self.assertRaises(RuntimeError):
            self._run(server)

    def test_a_late_reply_answers_its_own_request(self):
        async def go():
            server = CommServer()
            server.hold_get = lambda s: True
            client = at.MeasureClient(server, rpc_timeout=0.05)
            with self.assertRaises(asyncio.TimeoutError):
                await client.status()  # held: still pending
            server.pending, server.table_payload = False, "{}"
            second = asyncio.ensure_future(client.status())
            await asyncio.sleep(0.01)
            server.release()  # the late reply
            await asyncio.sleep(0.01)
            server.release()
            return await second

        self.assertEqual(asyncio.run(go()), [{"pending": False, "installed": "x"}])

    def test_hung_rpcs_prevent_startup(self):
        spec = SimpleNamespace(
            speculative_dspark_sps_autotune="auto", speculative_algorithm="DSPARK"
        )
        with mock.patch("sglang.srt.runtime_context.get_spec", return_value=spec):
            # Status reads hang once recording starts: the abandon request is
            # still applied, but no terminal confirmation can arrive.
            server = CommServer()
            server.hold_get = lambda s: s.recording
            with self.assertRaises(RuntimeError):
                asyncio.run(at.run_startup_autotune(server, rpc_timeout=0.05))
            self.assertIsNone(server.table_payload)
            self.assertFalse(server.recording)
            server = CommServer()
            server.hold_set = lambda req: at.TABLE_KEY in req.server_args
            with self.assertRaises(RuntimeError):
                asyncio.run(
                    at.run_startup_autotune(
                        server, sleep=server.sleep, clock=server.clock, rpc_timeout=0.05
                    )
                )

    def test_startup_driver_installs_or_abandons(self):
        spec = SimpleNamespace(
            speculative_dspark_sps_autotune="auto", speculative_algorithm="DSPARK"
        )
        with mock.patch("sglang.srt.runtime_context.get_spec", return_value=spec):
            server = FakeServer(_truth_ms)
            asyncio.run(
                at.run_startup_autotune(server, sleep=server.sleep, clock=server.clock)
            )
            request = json.loads(server.table_payload)
            self.assertEqual(request["key"], "k")
            SpsAdditiveCostTable(**request["table"])

            server = FakeServer(
                _truth_ms,
                stall={
                    (c[0], c[1])
                    for c in at.plan_cells(
                        tiers=TIERS, batch_sizes=[1, 2], verify_width=WIDTH
                    )
                },
            )
            asyncio.run(
                at.run_startup_autotune(server, sleep=server.sleep, clock=server.clock)
            )
            self.assertIsNone(server.table_payload)  # abandoned: bs 1-2 unlinked

            server = FakeServer(_truth_ms, pending=False)
            asyncio.run(
                at.run_startup_autotune(server, sleep=server.sleep, clock=server.clock)
            )
            self.assertEqual(server.events, [])

        off = SimpleNamespace(
            speculative_dspark_sps_autotune="off", speculative_algorithm="DSPARK"
        )
        with mock.patch("sglang.srt.runtime_context.get_spec", return_value=off):
            server = FakeServer(_truth_ms)
            asyncio.run(at.run_startup_autotune(server))
            self.assertEqual(server.events, [])


class _CharTokenizer:
    """Deterministic stand-in: one token per character."""

    def encode(self, text, add_special_tokens=False):
        return [1000 + (ord(c) % 4000) for c in text]


def _context_cost(table, contexts, *, max_slots=None):
    runner = (
        None
        if max_slots is None
        else SimpleNamespace(ragged_verify_mode=True, max_bs=max_slots)
    )
    holder = SimpleNamespace(
        _budget_planner=SimpleNamespace(sps_table=table),
        model_runner=SimpleNamespace(decode_cuda_graph_runner=runner),
    )
    batch = SimpleNamespace(reqs=[SimpleNamespace(seqlen=c) for c in contexts])
    return DSparkVerifyPlanner.batch_context_cost(holder, batch)


def _base(table, bs):
    return _additive_step_time_tensor(
        table=table, num_requests=bs, num_budgets=bs * GAMMA + 1
    )


def _priced(table, contexts, *, survival=None, fill_tiers=True, max_slots=None):
    """The planner's step seconds for every budget, requests at ``contexts``."""
    cfg = DSparkScheduleConfig(gamma=GAMMA)
    bs = len(contexts)
    survival = torch.ones(bs, GAMMA) if survival is None else survival
    cost = _context_cost(table, contexts, max_slots=max_slots)
    return _additive_step_time_tensor(
        table=table,
        num_requests=bs,
        num_budgets=bs * GAMMA + 1,
        fill_tiers=fill_tiers,
        verify_width=cfg.resolved_max_verify_len(),
        context_cost=cost,
        context_cum=(
            None
            if cost is None
            else _context_cumulative_seconds(
                history_survival_probs=survival, context_cost=cost, cfg=cfg
            )
        ),
    )


class TestContextTerm(CustomTestCase):
    def _table(self, context_seconds=0.0, tiers=TIERS):
        table, _ = at.fit_step_cost_table(_cells())
        return msgspec.structs.replace(
            table, context_seconds=context_seconds, verify_tiers=list(tiers)
        )

    def test_two_segment_context_cost(self):
        table = msgspec.structs.replace(
            self._table(2e-9), context_knee_tokens=65536, context_seconds_beyond=1e-9
        )
        self.assertAlmostEqual(table.context_seconds_per_verify_token(30_000), 6e-5)
        self.assertAlmostEqual(
            table.context_seconds_per_verify_token(400_000),
            2e-9 * 65536 + 1e-9 * (400_000 - 65536),
        )
        context = _priced(table, [400_000] * 4) - _base(table, 4)
        self.assertAlmostEqual(
            float(context[20]), table.context_seconds_per_verify_token(400_000) * 24
        )
        # A zero near slope still prices context past the knee.
        flat_near = msgspec.structs.replace(table, context_seconds=0.0)
        self.assertTrue(flat_near.prices_context)
        context = _priced(flat_near, [100_000] * 4) - _base(flat_near, 4)
        self.assertAlmostEqual(float(context[20]), 1e-9 * (100_000 - 65536) * 24)
        cost = _context_cost(flat_near, [1000, 100_000])
        self.assertEqual(float(cost.token_seconds[0]), 0.0)
        with self.assertRaises(ValueError):
            msgspec.structs.replace(table, context_seconds_beyond=-1e-9)
        with self.assertRaises(ValueError):
            msgspec.structs.replace(table, context_knee_tokens=-1)

    def test_table_prices_context_and_old_tables_load(self):
        table = self._table(2e-9)
        context = _priced(table, [100_000] * 4) - _base(table, 4)
        self.assertAlmostEqual(float(context[20]), 2e-9 * 24 * 100_000)
        # Inside a tier the executed (filled) tier is priced, not the budget.
        self.assertAlmostEqual(float(context[6]), 2e-9 * 12 * 100_000)
        original, _ = at.fit_step_cost_table(_cells())
        old = json.loads(original.to_json())
        for field in (
            "context_seconds",
            "verify_tiers",
            "context_knee_tokens",
            "context_seconds_beyond",
            "context_step_seconds",
            "context_step_seconds_beyond",
        ):
            old.pop(field)
        loaded = SpsAdditiveCostTable.from_json(json.dumps(old))
        self.assertTrue(torch.equal(_priced(loaded, [100_000] * 4), _base(original, 4)))
        with self.assertRaises(ValueError):
            msgspec.structs.replace(table, context_seconds=-1e-9)
        with self.assertRaises(ValueError):
            msgspec.structs.replace(table, verify_tiers=[12, 6])
        # Two-token tiers: probes 1..8 are contiguous, the tiers stay explicit.
        narrow = SpsAdditiveCostTable(
            bias_seconds=0.004,
            bs_probes=[1],
            alpha_seconds=[0.0],
            m_probes=list(range(1, 9)),
            theta_seconds=[0.001 * (1 + (m - 1) // 2) for m in range(1, 9)],
            context_seconds=1e-9,
            verify_tiers=[2, 4, 6, 8],
        )
        context = _priced(narrow, [1e6]) - _base(narrow, 1)
        for m, executed in ((1, 2), (2, 2), (3, 4), (5, 6), (6, 6)):
            self.assertAlmostEqual(float(context[m - 1]), 1e-9 * executed * 1e6)

    def test_planner_uses_the_context_term(self):
        table = self._table(3e-9)
        survival = torch.full((4, GAMMA), 0.8).cumprod(dim=1)
        cfg = DSparkScheduleConfig(gamma=GAMMA)
        short = compute_verify_token_budget(
            history_survival_probs=survival, sps_table=table, cfg=cfg
        )
        long = compute_verify_token_budget(
            history_survival_probs=survival,
            sps_table=table,
            cfg=cfg,
            context_cost=_context_cost(table, [400_000.0] * 4),
        )
        # Context makes every verify token dearer, so the planner verifies fewer,
        # and it still lands on a tier's last token count (alignment fills it).
        self.assertLess(long.budget, short.budget)
        for ctx in (50_000.0, 400_000.0, 2_000_000.0):
            for p in (0.3, 0.6, 0.8, 0.95):
                d = compute_verify_token_budget(
                    history_survival_probs=torch.full((4, GAMMA), p).cumprod(dim=1),
                    sps_table=table,
                    cfg=cfg,
                    context_cost=_context_cost(table, [ctx] * 4),
                )
                self.assertIn(4 + d.budget, TIERS, (ctx, p))

    def test_fit_context_seconds(self):
        kappa = 1.3e-9
        short = {
            (b, t): _truth_ms(b, t)
            for b, t in at.plan_long_cells(
                at.plan_cells(tiers=TIERS, batch_sizes=BATCH_SIZES, verify_width=WIDTH)
            )
        }
        long = {k: v + kappa * 1e3 * k[1] * 30_000 for k, v in short.items()}
        fit = at.fit_context_seconds(short, long, context_delta=30_000)
        self.assertAlmostEqual(fit["context_seconds"], kappa, places=15)
        self.assertAlmostEqual(fit["context_step_seconds"], 0.0, places=15)
        # A per-step part is separated from the per-token part.
        both = {k: v + 0.25e-9 * 1e3 * 30_000 for k, v in long.items()}
        fit = at.fit_context_seconds(short, both, context_delta=30_000)
        self.assertAlmostEqual(fit["context_seconds"], kappa, places=14)
        self.assertAlmostEqual(fit["context_step_seconds"], 0.25e-9, places=14)
        # One wild cell barely moves the Huber-weighted fit.
        long[(1, 6)] += 5.0
        fit = at.fit_context_seconds(short, long, context_delta=30_000)
        self.assertLess(abs(fit["context_seconds"] - kappa), 0.02 * kappa)
        # Never negative; needs paired cells and a positive context delta.
        fit = at.fit_context_seconds(
            short, {k: v - 1.0 for k, v in short.items()}, context_delta=30_000
        )
        self.assertEqual(fit["context_seconds"], 0.0)
        with self.assertRaises(ValueError):
            at.fit_context_seconds(short, {(99, 6): 1.0}, context_delta=30_000)
        # One executed width cannot separate the two parts.
        with self.assertRaises(ValueError):
            at.fit_context_seconds(
                {(2, 12): 10.0, (4, 12): 12.0},
                {(2, 12): 11.0, (4, 12): 13.0},
                context_delta=30_000,
            )
        with self.assertRaises(ValueError):
            at.fit_context_seconds(short, long, context_delta=0)

    def test_long_cells_are_each_batch_sizes_extreme_tiers(self):
        cells = at.plan_cells(tiers=TIERS, batch_sizes=BATCH_SIZES, verify_width=WIDTH)
        long_cells = at.plan_long_cells(cells)
        self.assertEqual(long_cells[:3], [(1, 6), (2, 6), (2, 12)])
        self.assertEqual(long_cells[-2:], [(16, 18), (16, 96)])
        self.assertEqual(len(long_cells), 2 * len(BATCH_SIZES) - 1)


class TestContextMeasurement(CustomTestCase):
    _run = TestMeasurement._run

    def test_long_context_cells_fit_the_context_term(self):
        server = FakeServer(_truth_ms, context_ms_per_ktok=0.0013, long_context=8192)
        request = self._run(server)
        table = SpsAdditiveCostTable(**request["table"])
        self.assertAlmostEqual(table.context_seconds, 1.3e-9, places=13)
        cells = at.plan_cells(tiers=TIERS, batch_sizes=BATCH_SIZES, verify_width=WIDTH)
        long_cells = at.plan_long_cells(cells)
        self.assertEqual(len(server.loads), 2 + 2 * len(cells) + 2 * len(long_cells))
        long_loads = server.loads[-(1 + 2 * len(long_cells)) :]
        prefix = long_loads[0].input_ids[0][:8192]
        self.assertEqual(len(long_loads[0].input_ids), 1)
        for load in long_loads:
            for ids in load.input_ids:
                self.assertEqual(ids[:8192], prefix)
                self.assertEqual(len(ids), 8192 + at._LONG_SUFFIX_LEN)
        # The fitted table, re-based to context zero, reproduces the step costs
        # at both measured contexts (the timed steps' mean context).
        ref = at.reference_context_tokens(PARAMS)
        for b, t in long_cells:
            for ctx in (ref, ref - PARAMS.input_len + 8192 + at._LONG_SUFFIX_LEN):
                got = float(_priced(table, [ctx] * b)[t - b])
                want = _truth_ms(b, t) + 0.0013 * t * ctx / 1000
                # Within the few tokens the estimated window midpoint misses.
                self.assertLess(abs(got * 1e3 - want), 0.0013 * t * 4 / 1000, (b, t))
        meta = request["metadata"]
        self.assertEqual(meta["long_context"], 8192)
        self.assertEqual(len(meta["long_step_ms_by_bs_tier"]), len(long_cells))
        self.assertEqual(meta["prompts"], "random")

    def test_installed_tiers_cover_unsampled_batch_sizes(self):
        # Sampling stops at batch size 8 (tiers up to 48); a batch of 9 still
        # runs in captured tier 60, which alignment fills with 54 real tokens.
        from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
            schedule_verify_lens_topk_from_survival,
        )
        from sglang.srt.speculative.dspark_components.dspark_planner import (
            graph_tier_fill_budget,
        )

        server = FakeServer(
            _truth_ms,
            context_ms_per_ktok=0.0013,
            long_context=8192,
            batch_sizes=[1, 2, 3, 4, 6, 8],
        )
        table = SpsAdditiveCostTable(**self._run(server)["table"])
        cfg = DSparkScheduleConfig(gamma=GAMMA)
        budget = 40
        lens = schedule_verify_lens_topk_from_survival(
            survival_probs=torch.ones(9, GAMMA),
            budget=graph_tier_fill_budget(
                graph_num_tokens=_tier_of(9 + budget),
                bs=9,
                verify_num_draft_tokens=WIDTH,
                min_verify_len=cfg.min_verify_len,
            ),
            cfg=cfg,
        )
        context = _priced(table, [100_000] * 9) - _base(table, 9)
        self.assertAlmostEqual(
            float(context[budget]),
            float(lens.sum()) * table.context_seconds_per_verify_token(100_000)
            + table.context_step_seconds_at(100_000),
        )

    def test_far_context_fits_the_slope_beyond_the_knee(self):
        knee = 8192 + at._LONG_SUFFIX_LEN
        server = FakeServer(
            _truth_ms,
            context_ms_per_ktok=0.002,
            long_context=8192,
            long_context_far=32768,
            context_knee=knee,
            context_ms_per_ktok_beyond=0.0009,
        )
        request = self._run(server)
        table = SpsAdditiveCostTable(**request["table"])
        # Within 1%: the near cells' decoded tokens run a little past the knee.
        self.assertLess(abs(table.context_seconds - 2e-9), 2e-11)
        self.assertLess(abs(table.context_seconds_beyond - 0.9e-9), 0.9e-11)
        self.assertEqual(table.context_knee_tokens, knee)
        meta = request["metadata"]
        self.assertEqual(meta["long_context_far"], 32768)
        cells = at.plan_cells(tiers=TIERS, batch_sizes=BATCH_SIZES, verify_width=WIDTH)
        long_cells = at.plan_long_cells(cells)
        self.assertEqual(len(meta["far_step_ms_by_bs_tier"]), len(long_cells))
        self.assertEqual(
            len(server.loads), 1 + 2 * len(cells) + 2 * (1 + 2 * len(long_cells))
        )
        # The table reproduces step costs well past the knee.
        for b, t in long_cells:
            ctx = 400_000.0
            got = float(_priced(table, [ctx] * b)[t - b])
            want = _truth_ms(b, t) + (0.002 * knee + 0.0009 * (ctx - knee)) / 1000 * t
            # Within 1% of the context cost (the slopes are fitted to 1%).
            self.assertLess(
                abs(got * 1e3 - want), 0.01 * (want - _truth_ms(b, t)), (b, t)
            )

    def test_unpaired_far_cells_keep_the_near_fit(self):
        # The far cells all time out: the near fit stays, with no knee.
        server = FakeServer(
            _truth_ms,
            context_ms_per_ktok=0.002,
            long_context=8192,
            long_context_far=32768,
        )
        far_ctx = 32768 + at._LONG_SUFFIX_LEN
        orig = server.generate_request

        def gen(obj, request):
            out = orig(obj, request)
            if (
                len(obj.input_ids[0]) == far_ctx
                and obj.sampling_params["max_new_tokens"] > 1
            ):
                server.stall = set(
                    at.plan_long_cells(
                        at.plan_cells(
                            tiers=TIERS, batch_sizes=BATCH_SIZES, verify_width=WIDTH
                        )
                    )
                )
            else:
                server.stall = set()
            return out

        server.generate_request = gen
        status = _status(server)
        status["params"] = {**status["params"], "cell_timeout_seconds": 10.0}
        request = self._run(server, status)
        table = SpsAdditiveCostTable(**request["table"])
        self.assertLess(abs(table.context_seconds - 2e-9), 2e-11)
        self.assertEqual(table.context_knee_tokens, 0)
        self.assertNotIn("far", request["metadata"]["context_fit"])

    def test_unpaired_near_cells_keep_the_short_fit(self):
        # Short (2, 6) times out, and only (2, 6) succeeds at long context: no
        # pair, so no context term, while the short fit stays.
        server = FakeServer(_truth_ms, context_ms_per_ktok=0.002, long_context=8192)
        cells = at.plan_cells(tiers=TIERS, batch_sizes=BATCH_SIZES, verify_width=WIDTH)
        long_cells = set(at.plan_long_cells(cells))
        orig = server.generate_request

        def gen(obj, request):
            out = orig(obj, request)
            is_long = len(obj.input_ids[0]) > PARAMS.input_len
            server.stall = (long_cells - {(2, 6)}) if is_long else {(2, 6)}
            return out

        server.generate_request = gen
        status = _status(server)
        status["params"] = {**status["params"], "cell_timeout_seconds": 10.0}
        request = self._run(server, status)
        table = SpsAdditiveCostTable(**request["table"])
        self.assertFalse(table.prices_context)
        self.assertIsNone(request["metadata"]["context_fit"])

    def test_text_prompts_and_sampling(self):
        tokenizer = _CharTokenizer()
        server = FakeServer(_truth_ms, tokenizer=tokenizer)
        status = _status(server)
        status["params"] = {**status["params"], "top_p": 0.95}
        request = self._run(server, status)
        self.assertEqual(request["metadata"]["prompts"], "text")
        ids = at.source_text_token_ids(tokenizer, 64 * PARAMS.input_len)
        index = {}
        for i in range(len(ids) - 7):
            index.setdefault(tuple(ids[i : i + 8]), []).append(i)
        for load in server.loads:
            self.assertEqual(load.sampling_params["top_p"], 0.95)
            for window in load.input_ids:
                self.assertEqual(len(window), PARAMS.input_len)
                starts = index.get(tuple(window[:8]), [])
                self.assertTrue(any(ids[i : i + len(window)] == window for i in starts))
        server = FakeServer(_truth_ms)
        self._run(server)
        self.assertNotIn("top_p", server.loads[1].sampling_params)

    def test_text_prompts_required_when_promised(self):
        server = FakeServer(_truth_ms)  # no tokenizer
        status = {**_status(server), "prompt_source": "text"}
        with self.assertRaises(RuntimeError):
            self._run(server, status)
        self.assertEqual(server.frac, None)
        self.assertFalse(server.recording)

    def test_long_context_sizing(self):
        kw = dict(context_len=1 << 20, kv_tokens=0, max_bs=16, max_new_tokens=720)
        self.assertEqual(at.long_context_tokens(32768, prefix_cache=True, **kw), 32768)
        # No prefix cache: the requests could not share the prefix.
        self.assertEqual(at.long_context_tokens(32768, prefix_cache=False, **kw), 0)
        self.assertEqual(at.long_context_tokens(0, prefix_cache=True, **kw), 0)
        # Capped by the KV pool (one shared prefix plus each request's own).
        own = at._LONG_SUFFIX_LEN + 720
        kw["kv_tokens"] = 40000
        self.assertEqual(
            at.long_context_tokens(32768, prefix_cache=True, **kw),
            int(0.8 * 40000) - 16 * own,
        )
        kw["kv_tokens"] = 16000  # too little room left: skipped
        self.assertEqual(at.long_context_tokens(32768, prefix_cache=True, **kw), 0)
        # Capped by the context window.
        kw.update(kv_tokens=0, context_len=20000)
        self.assertEqual(
            at.long_context_tokens(32768, prefix_cache=True, **kw), 20000 - own - 1
        )
        # It must exceed the short cells' prompts by enough to fit the term.
        kw.update(context_len=1 << 20)
        self.assertEqual(
            at.long_context_tokens(8192, prefix_cache=True, input_len=8192, **kw), 0
        )
        self.assertEqual(
            at.long_context_tokens(16384, prefix_cache=True, input_len=8192, **kw),
            16384,
        )

    def test_unusable_long_context_keeps_the_short_fit(self):
        server = FakeServer(_truth_ms, long_context=8192)
        status = _status(server)
        status["params"] = {**status["params"], "input_len": 9000}
        request = self._run(server, status)
        self.assertIsNone(request["metadata"]["context_fit"])
        self.assertEqual(SpsAdditiveCostTable(**request["table"]).context_seconds, 0.0)

    def test_planner_reads_context_from_request_state(self):
        table = SpsAdditiveCostTable(
            bias_seconds=0.004,
            bs_probes=[1],
            alpha_seconds=[0.0],
            m_probes=[6],
            theta_seconds=[0.001],
            context_seconds=0.0,
            context_knee_tokens=32768,
            context_seconds_beyond=1e-9,
            verify_tiers=[6],
        )
        batch = SimpleNamespace(
            reqs=[SimpleNamespace(seqlen=1000), SimpleNamespace(seqlen=64000)]
        )
        holder = SimpleNamespace(
            _budget_planner=SimpleNamespace(sps_table=table),
            model_runner=SimpleNamespace(decode_cuda_graph_runner=None),
        )
        fn = DSparkVerifyPlanner.batch_context_cost
        # Each request is priced at its own context: the mean context (32.5K)
        # sits below the knee, but the 64K request still costs past it.
        got = fn(holder, batch)
        self.assertEqual(float(got.token_seconds[0]), 0.0)
        self.assertAlmostEqual(float(got.token_seconds[1]), 1e-9 * (64000 - 32768))
        unpriced = SimpleNamespace(
            _budget_planner=SimpleNamespace(
                sps_table=msgspec.structs.replace(table, context_seconds_beyond=0.0)
            )
        )
        self.assertIsNone(fn(unpriced, batch))  # no read without the term
        self.assertIsNone(fn(holder, SimpleNamespace(reqs=[])))
        self.assertIsNone(fn(SimpleNamespace(_budget_planner=None), batch))

    def test_context_order_follows_the_allocator_on_ties(self):
        # Fresh requests: every survival is 1, so the allocator breaks ties by
        # position, then request; the context of the k-th draft follows that.
        from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
            schedule_verify_lens_topk_from_survival,
        )

        table = msgspec.structs.replace(
            at.fit_step_cost_table(_cells())[0],
            context_seconds=2e-9,
            verify_tiers=list(TIERS),
        )
        lens = [2_000, 2_000, 400_000, 400_000]
        cost = ContextCost(
            step_seconds=0.0,
            token_seconds=torch.tensor(
                [table.context_seconds_per_verify_token(n) for n in lens],
                dtype=torch.float64,
            ),
        )
        survival = torch.ones(4, GAMMA)
        cfg = DSparkScheduleConfig(gamma=GAMMA)
        d = compute_verify_token_budget(
            history_survival_probs=survival, sps_table=table, cfg=cfg, context_cost=cost
        )
        executed = next(t for t in TIERS if t >= 4 + d.budget)
        lens_alloc = schedule_verify_lens_topk_from_survival(
            survival_probs=survival, budget=executed - 4, cfg=cfg
        ).to(torch.float64)
        expected_ctx = float((lens_alloc * cost.token_seconds).sum())
        self.assertAlmostEqual(
            d.predicted_step_seconds,
            table.step_time(num_reqs=4, budget=d.budget) + expected_ctx,
            places=12,
        )

    def test_drafts_under_the_cutoff_read_no_context(self):
        table = msgspec.structs.replace(
            at.fit_step_cost_table(_cells())[0],
            context_seconds=2e-9,
            verify_tiers=list(TIERS),
        )
        survival = torch.tensor([[0.9] * GAMMA, [0.6, 0.6, 0.01, 0.0001, 0.1]]).cumprod(
            dim=1
        )
        lens = [2_000, 400_000]
        cost = ContextCost(
            step_seconds=0.0,
            token_seconds=torch.tensor(
                [table.context_seconds_per_verify_token(n) for n in lens],
                dtype=torch.float64,
            ),
        )
        cfg = DSparkScheduleConfig(gamma=GAMMA, survival_eps=1e-3)
        d = compute_verify_token_budget(
            history_survival_probs=survival, sps_table=table, cfg=cfg, context_cost=cost
        )
        from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
            schedule_verify_lens_topk_from_survival,
        )

        executed = next(t for t in TIERS if t >= 2 + d.budget)
        alloc = schedule_verify_lens_topk_from_survival(
            survival_probs=survival, budget=executed - 2, cfg=cfg
        ).to(torch.float64)
        self.assertAlmostEqual(
            d.predicted_step_seconds,
            table.step_time(num_reqs=2, budget=d.budget)
            + float((alloc * cost.token_seconds).sum()),
            places=12,
        )

    def test_padding_is_priced_where_the_layout_puts_it(self):
        # No draft clears the cutoff: 2 real tokens run in a 6-token tier. Spare
        # captured slots take the padding; without them the last row does.
        from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

        table = msgspec.structs.replace(
            at.fit_step_cost_table(_cells())[0],
            context_seconds=2e-9,
            verify_tiers=list(TIERS),
        )
        batch = SimpleNamespace(
            reqs=[SimpleNamespace(seqlen=2_000), SimpleNamespace(seqlen=400_000)]
        )
        cfg = DSparkScheduleConfig(gamma=GAMMA, survival_eps=1e-3)
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[1, 1], device=torch.device("cpu"), grid=TIERS
        )
        for max_slots, fill_tiers in ((2, True), (2, False), (16, True)):
            runner = SimpleNamespace(ragged_verify_mode=True, max_bs=max_slots)
            holder = SimpleNamespace(
                _budget_planner=SimpleNamespace(sps_table=table),
                model_runner=SimpleNamespace(decode_cuda_graph_runner=runner),
            )
            cost = DSparkVerifyPlanner.batch_context_cost(holder, batch)
            d = compute_verify_token_budget(
                history_survival_probs=torch.full((2, GAMMA), 1e-4),
                sps_table=table,
                cfg=cfg,
                context_cost=cost,
                fill_tiers=fill_tiers,
            )
            # The graph runner's slot count for a tier: min(tier, max_bs).
            padded = layout.padded_to_bucket(
                padded_bs=min(layout.graph_num_tokens, max_slots)
            )
            read = padded.verify_lens[:2].to(torch.float64)
            self.assertAlmostEqual(
                d.predicted_step_seconds,
                table.step_time(num_reqs=2, budget=0)
                + float((read * cost.token_seconds).sum()),
                places=12,
                msg=(max_slots, fill_tiers),
            )

    def test_mixed_contexts_price_the_requests_that_get_tokens(self):
        # One long-context request with confident drafts and three short ones
        # with weak drafts: extra tokens go to the long request, so they cost
        # its context, not the batch mean.
        table = msgspec.structs.replace(
            at.fit_step_cost_table(_cells())[0],
            context_seconds=2e-9,
            verify_tiers=list(TIERS),
        )
        survival = torch.tensor(
            [[0.95] * GAMMA, [0.2] * GAMMA, [0.2] * GAMMA, [0.2] * GAMMA]
        ).cumprod(dim=1)
        lens = [400_000, 2_000, 2_000, 2_000]
        cost = ContextCost(
            step_seconds=0.0,
            token_seconds=torch.tensor(
                [table.context_seconds_per_verify_token(n) for n in lens],
                dtype=torch.float64,
            ),
        )
        cfg = DSparkScheduleConfig(gamma=GAMMA)
        d = compute_verify_token_budget(
            history_survival_probs=survival, sps_table=table, cfg=cfg, context_cost=cost
        )
        executed = next(t for t in TIERS if t >= 4 + d.budget)
        order = torch.argsort(survival.flatten(), descending=True)
        rows = torch.arange(4).repeat_interleave(GAMMA)[order][: executed - 4]
        expected_ctx = float(cost.token_seconds.sum() + cost.token_seconds[rows].sum())
        self.assertAlmostEqual(
            d.predicted_step_seconds,
            table.step_time(num_reqs=4, budget=d.budget) + expected_ctx,
            places=12,
        )

    def test_filled_tiers_are_capped_by_the_drafts(self):
        # 9 requests x width 6 fill a 60-token tier with at most 54 tokens.
        table = msgspec.structs.replace(
            at.fit_step_cost_table(_cells())[0],
            context_seconds=1e-9,
            verify_tiers=list(TIERS),
        )
        context = _priced(table, [100_000] * 9) - _base(table, 9)
        self.assertAlmostEqual(
            float(context[-1]), table.context_seconds_per_verify_token(100_000) * 54
        )
        short = {(9, 60): 30.0, (9, 12): 20.0}
        long = {
            (9, 60): 30.0 + 1e-9 * 1e3 * 54 * 50_000,
            (9, 12): 20.0 + 1e-9 * 1e3 * 12 * 50_000,
        }
        fit = at.fit_context_seconds(
            short, long, context_delta=50_000, verify_width=WIDTH
        )
        self.assertAlmostEqual(fit["context_seconds"], 1e-9, places=15)

    def test_rebased_table_reproduces_the_measured_cells(self):
        # (9, 60): 54 real verify tokens read context, 6 padding tokens take
        # spare slots.
        fit = {"context_seconds": 1e-9, "context_step_seconds": 2e-10}
        ref = at.reference_context_tokens(PARAMS)
        cells = at.plan_cells(
            tiers=TIERS, batch_sizes=[1, 2, 4, 9, 16], verify_width=WIDTH
        )
        measured = {
            (b, t): _truth_ms(b, t) / 1e3
            + ref
            * (
                fit["context_seconds"] * at.context_tokens(b, t, verify_width=WIDTH)
                + fit["context_step_seconds"]
            )
            for b, t in cells
        }
        table, _ = at.fit_step_cost_table(
            at.without_context(
                [(b, t, s) for (b, t), s in measured.items()],
                context_fit=fit,
                reference_context_tokens=ref,
                verify_width=WIDTH,
            )
        )
        table = msgspec.structs.replace(
            table,
            context_seconds=fit["context_seconds"],
            context_step_seconds=fit["context_step_seconds"],
            verify_tiers=list(TIERS),
        )
        self.assertAlmostEqual(
            float(_priced(table, [ref] * 9)[-1]), measured[(9, 60)], places=9
        )

    def test_context_counts_padding_only_when_tiers_are_filled(self):
        table = msgspec.structs.replace(
            at.fit_step_cost_table(_cells())[0],
            context_seconds=1e-9,
            verify_tiers=list(TIERS),
        )
        cost = table.context_seconds_per_verify_token(200_000)
        filled = _priced(table, [200_000] * 4)
        unfilled = _priced(table, [200_000] * 4, fill_tiers=False)
        # Budget 0 runs 4 real tokens in a 6-token tier.
        self.assertAlmostEqual(float(filled[0] - unfilled[0]), cost * 2)


@contextlib.contextmanager
def _context(
    mode,
    explicit=None,
    *,
    dp=False,
    dp_size=1,
    pp=1,
    elastic=None,
    tokenizers=1,
    disagg="null",
):
    spec = SimpleNamespace(
        speculative_dspark_sps_autotune=mode,
        speculative_dspark_sps_table_path=explicit,
    )
    parallel = SimpleNamespace(
        tp_group=SimpleNamespace(rank_in_group=0, world_size=1, cpu_group=None),
        attn_dp_enabled=dp,
        dp_size=dp_size,
        pp_size=pp,
    )
    values = {
        "get_spec": spec,
        "get_parallel": parallel,
        "get_exec": SimpleNamespace(moe=SimpleNamespace(elastic_ep_backend=elastic)),
        "get_serving": SimpleNamespace(tokenizer_worker_num=tokenizers),
        "get_disagg": SimpleNamespace(disaggregation_mode=disagg),
    }
    with contextlib.ExitStack() as stack:
        for name, value in values.items():
            stack.enter_context(
                mock.patch(f"sglang.srt.runtime_context.{name}", return_value=value)
            )
        yield


class TestSchedulerEntryPoint(CustomTestCase):
    def _run(self, planner, cache_dir, mode, explicit=None, **layout):
        from sglang.srt.speculative.dspark_components import dspark_planner as dp

        with (
            _context(mode, explicit, **layout),
            envs.SGLANG_DSPARK_SPS_AUTOTUNE_CACHE_DIR.override(cache_dir),
            mock.patch.object(
                at, "build_startup_material", side_effect=lambda **_: _prepare()()
            ),
            mock.patch.object(
                dp, "get_parallel", return_value=SimpleNamespace(tp_rank=0)
            ),
        ):
            return at.maybe_autotune_sps_table(
                planner=planner,
                target_model_runner=None,
                draft_model_runner=None,
                gamma=GAMMA,
                verify_num_draft_tokens=WIDTH,
                simulate_acc_len=0.0,
            )

    @staticmethod
    def _planner(table):
        from sglang.srt.speculative.dspark_components import dspark_planner as dp

        planner = dp.DSparkVerifyPlanner.__new__(dp.DSparkVerifyPlanner)
        planner._sps_measurement_verify_all = None
        planner._sps_measurement_survival_eps = None
        planner._budget_planner = SimpleNamespace(sps_table=table, last_decision=None)
        planner._is_verify_all = table is None
        planner._uniform_layout_cache = {}
        return planner

    @staticmethod
    def _cache(cache_dir):
        table = at.fit_step_cost_table(_cells())[0]
        prepared = _pending(cache_dir)
        at.install_measured_table(
            payload=_install_payload(prepared, table),
            prepared=prepared,
            group=None,
            install=lambda t, s: None,
        )
        return table

    def test_off_explicit_and_unsupported_layouts_keep_the_table(self):
        for args, layout in (
            (("off",), {}),
            (("auto", "/t.json"), {}),
            (("auto",), dict(dp=True)),
            (("auto",), dict(dp_size=2)),
            (("force",), dict(pp=2)),
            (("auto",), dict(elastic="nixl")),
            (("auto",), dict(tokenizers=2)),
            (("force",), dict(disagg="decode")),
        ):
            with self.subTest(args=args, layout=layout):
                configured = object()
                planner = self._planner(configured)
                with tempfile.TemporaryDirectory() as d:
                    self._cache(d)
                    self.assertIsNone(self._run(planner, d, *args, **layout))
                self.assertIs(planner._budget_planner.sps_table, configured)

    def test_cached_table_is_installed(self):
        planner = self._planner(None)
        with tempfile.TemporaryDirectory() as d:
            table = self._cache(d)
            self.assertIsNone(self._run(planner, d, "auto"))
        self.assertEqual(planner._budget_planner.sps_table, table)
        self.assertFalse(planner.is_verify_all)
        d = compute_verify_token_budget(
            history_survival_probs=torch.full((4, GAMMA), 0.3).cumprod(dim=1),
            sps_table=planner._budget_planner.sps_table,
            cfg=DSparkScheduleConfig(gamma=GAMMA),
        )
        self.assertLess(d.budget, 4 * GAMMA)

    def test_material_rejects_ineligible_configs(self):
        greedy = (0.0, 1e-8, math.nextafter(_SAMPLING_EPS, 0.0), math.nan, math.inf)
        for compact, simulate, env in (
            (False, 0.0, contextlib.nullcontext()),
            (True, 1.0, contextlib.nullcontext()),
            (True, 0.0, envs.SGLANG_DSPARK_SPS_AUTOTUNE_PASSES.override(0)),
            *(
                (True, 0.0, envs.SGLANG_DSPARK_SPS_AUTOTUNE_TEMPERATURE.override(t))
                for t in greedy
            ),
        ):
            with self.subTest(compact=compact, simulate=simulate):
                planner = mock.Mock(
                    schedules_verify_budget=True, is_compact_mode=compact
                )
                with env, self.assertRaises(at.SpsAutotuneSkip):
                    at.build_startup_material(
                        planner=planner,
                        target_model_runner=None,
                        draft_model_runner=None,
                        gamma=GAMMA,
                        verify_num_draft_tokens=WIDTH,
                        simulate_acc_len=simulate,
                    )

    def test_params_from_env(self):
        with (
            envs.SGLANG_DSPARK_SPS_AUTOTUNE_PASSES.override(3),
            envs.SGLANG_DSPARK_SPS_AUTOTUNE_BATCH_SIZES.override("16, 1,4"),
            envs.SGLANG_DSPARK_SPS_AUTOTUNE_TEMPERATURE.override(_SAMPLING_EPS),
        ):
            params = at.AutotuneParams.from_env()
        self.assertEqual((params.passes, params.batch_sizes), (3, (1, 4, 16)))


class TestWorkerAndPlannerHooks(CustomTestCase):
    def _worker(self, prepared):
        from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
            DSparkWorkerV2,
        )

        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._hosts_draft = True
        worker._forced_budget_frac = 0.5
        worker._sps_autotune = prepared
        worker._sps_autotune_installed = None
        worker._observers = mock.Mock()
        worker._verify_planner = mock.Mock()
        worker._verify_executor = SimpleNamespace(bonus_only=False)
        return worker

    def test_worker_status_recording_install_and_abandon(self):
        group = SimpleNamespace(world_size=1, cpu_group=None, rank_in_group=0)
        parallel = SimpleNamespace(tp_rank=0, tp_group=group)
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch(
                "sglang.srt.runtime_context.get_parallel", return_value=parallel
            ),
        ):
            prepared = _pending(d)
            worker = self._worker(prepared)
            self.assertTrue(worker.sps_autotune_status()["pending"])
            worker.set_sps_autotune_recording(True)
            worker._observers.set_sps_recording.assert_called_with(True)
            worker._verify_planner.set_sps_measurement.assert_called_with(True)
            # Measurement commits only the bonus token; the end restores it.
            self.assertTrue(worker._verify_executor.bonus_only)
            worker.set_sps_autotune_recording(True)
            self.assertTrue(worker._verify_executor.bonus_only)
            self.assertTrue(
                worker.install_sps_autotune_table(_install_payload(prepared))
            )
            worker._verify_planner.install_sps_table.assert_called_once()
            worker._verify_planner.set_sps_measurement.assert_called_with(False)
            self.assertFalse(worker._verify_executor.bonus_only)
            self.assertEqual(worker.sps_autotune_status()["pending"], False)
            self.assertTrue(prepared.cache_path.is_file())
            # Terminal: unpinned; later recording requests are no-ops.
            worker._verify_planner.set_forced_budget_frac.assert_called_with(None)
            self.assertIsNone(worker._forced_budget_frac)
            worker._observers.reset_mock()
            worker.set_sps_autotune_recording(True)
            worker._observers.set_sps_recording.assert_not_called()

            worker = self._worker(_pending(d + "/other"))
            self.assertFalse(worker.install_sps_autotune_table(None))
            worker._verify_planner.install_sps_table.assert_not_called()
            worker._verify_planner.set_forced_budget_frac.assert_called_with(None)
            worker._observers.set_sps_recording.assert_called_with(False)
            self.assertIsNone(worker.sps_autotune_status())

    def test_combined_terminal_update_through_the_real_controller(self):
        from sglang.srt.managers.io_struct import SetInternalStateReq
        from sglang.srt.managers.scheduler_components.internal_state import (
            SchedulerInternalStateController,
        )

        group = SimpleNamespace(world_size=1, cpu_group=None, rank_in_group=0)
        parallel = SimpleNamespace(tp_rank=0, tp_group=group)
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch(
                "sglang.srt.runtime_context.get_parallel", return_value=parallel
            ),
        ):
            worker = self._worker(_pending(d))
            scheduler = SimpleNamespace(
                spec_algorithm=SimpleNamespace(is_dspark=lambda: True),
                draft_worker=worker,
            )
            controller = SchedulerInternalStateController(scheduler, False)
            out = controller.set_internal_state(
                SetInternalStateReq(server_args={at.RECORD_KEY: True})
            )
            self.assertTrue(out.updated)
            self.assertTrue(worker._verify_executor.bonus_only)
            out = controller.set_internal_state(
                SetInternalStateReq(
                    server_args={at.TABLE_KEY: None, at.RECORD_KEY: False}
                )
            )
            self.assertTrue(out.updated)
            self.assertIsNone(worker.sps_autotune_status())
            self.assertFalse(worker._verify_executor.bonus_only)
            out = controller.set_internal_state(
                SetInternalStateReq(server_args={at.RECORD_KEY: True})
            )
            self.assertFalse(out.updated)  # nothing pending any more

    def _bonus_only_accept(
        self, sampling_info, correct_lens, accept_bonus, block_verification
    ):
        from sglang.srt.speculative.dspark_components import dspark_verify as dv

        bs, width = len(correct_lens), WIDTH
        candidates = torch.arange(bs * width, dtype=torch.int64).view(bs, width) + 100
        executor = dv.TargetVerifyExecutor.__new__(dv.TargetVerifyExecutor)
        executor._simulate_acc_len = 0.0
        executor._simulated_correct_len = None
        executor._target_is_dsv41 = False
        executor.gamma, executor.verify_num_draft_tokens = GAMMA, width
        executor._tp_sync = mock.Mock()
        executor._block_verification = block_verification
        executor.bonus_only = True
        accept_result = (
            torch.tensor(correct_lens, dtype=torch.int32),
            torch.tensor(accept_bonus, dtype=torch.int64),
            torch.zeros(bs, dtype=torch.int32),
        )
        passthrough = SimpleNamespace(
            execute=lambda **kw: SimpleNamespace(
                cap_trim_lens=kw.get("cap_trim_lens"),
                commit_lens=kw.get("correct_len"),
                new_seq_lens=None,
            )
        )
        resample = mock.Mock(side_effect=AssertionError("no extra sampling pass"))
        accept = mock.Mock(return_value=accept_result)
        with (
            mock.patch.object(dv, "accept_draft_tokens", accept),
            mock.patch.object(dv, "sample_simulated_bonus", resample),
            mock.patch.object(dv, "FinalizeAcceptLens", passthrough),
            mock.patch.object(
                dv, "BuildOutTokens", SimpleNamespace(execute=lambda **_: None)
            ),
        ):
            out = executor.accept_and_finalize(
                folded_accept=False,
                bs=bs,
                verify_ids_2d=candidates,
                target_logits=torch.zeros(bs * width, 11),
                draft_block=None,
                sampling_info=sampling_info,
                draft_input=None,
                layout=None,
                prefix_lens=None,
                draft_tokens=None,
            )
        self.assertIs(accept.call_args.kwargs["block_verification"], block_verification)
        return candidates, out

    def test_bonus_only_commits_the_row_zero_bonus(self):
        # No draft is committed; the bonus is the accept's token at row 0, with
        # the serving accept (block verification or not) deciding it.
        for sampling_info, block_verification in itertools.product(
            (None, SimpleNamespace(is_all_greedy=False)), (False, True)
        ):
            candidates, out = self._bonus_only_accept(
                sampling_info,
                correct_lens=[3, 0, 1],
                accept_bonus=[7, 8, 9],
                block_verification=block_verification,
            )
            self.assertTrue(
                torch.equal(out.correct_len, torch.zeros(3, dtype=torch.int32))
            )
            want = torch.tensor(
                [candidates[0, 1], 8, candidates[2, 1]], dtype=torch.int64
            )
            self.assertTrue(torch.equal(out.bonus, want))

    def test_planner_measurement_mode_and_install(self):
        from sglang.srt.speculative.dspark_components import dspark_planner as dp

        planner = dp.DSparkVerifyPlanner.__new__(dp.DSparkVerifyPlanner)
        planner._sps_measurement_verify_all = None
        planner._sps_measurement_survival_eps = None
        planner._budget_planner = SimpleNamespace(sps_table=None, last_decision=1)
        planner._is_verify_all = True
        planner._uniform_layout_cache = {(1, None): object()}
        planner._schedule_cfg = SimpleNamespace(survival_eps=1e-6)
        planner.set_sps_measurement(True)
        self.assertFalse(planner.is_verify_all)
        # Every draft can fill the pinned tier while measuring.
        self.assertEqual(planner._schedule_cfg.survival_eps, 0.0)
        planner.set_sps_measurement(True)
        planner.set_sps_measurement(False)
        self.assertTrue(planner.is_verify_all)
        self.assertEqual(planner._schedule_cfg.survival_eps, 1e-6)
        table, _ = at.fit_step_cost_table(_cells())
        with mock.patch.object(
            dp, "get_parallel", return_value=SimpleNamespace(tp_rank=0)
        ):
            planner.set_sps_measurement(True)
            planner.install_sps_table(table, source="test")
            planner.set_sps_measurement(False)
        self.assertIs(planner._budget_planner.sps_table, table)
        self.assertFalse(planner.is_verify_all)
        self.assertEqual(planner._uniform_layout_cache, {})
        planner._budget_planner = None
        with self.assertRaises(RuntimeError):
            planner.install_sps_table(table, source="test")

    def test_info_dumper_runtime_recording(self):
        from sglang.srt.speculative.dspark_components.dspark_observability import (
            DecodeStepObservation,
            DsparkInfoDumper,
            InfoComponent,
        )

        def obs(forward_ct, bs=2):
            t = torch.zeros(bs, dtype=torch.int64)
            return DecodeStepObservation(
                forward_ct=forward_ct,
                bs=bs,
                mode="compact",
                budget=0,
                lag_steps=0,
                num_verify_tokens=bs,
                verify_tokens_local=bs,
                verify_tokens_dp_synced=bs,
                verify_tokens_graph_key=bs,
                predicted_step_ms=None,
                predicted_theta=None,
                verify_lens=None,
                confidence=None,
                req_pool_indices=t,
                prefix_lens=t,
                draft_tokens=t,
                bonus_tokens=t,
                correct_len=t,
                cap_trim_lens=t,
                commit_lens=t,
                rids=None,
            )

        now = [0.0]
        dumper = DsparkInfoDumper(
            components={InfoComponent.CORE},
            gamma=GAMMA,
            verify_num_draft_tokens=WIDTH,
            attn_tp_rank=0,
            device="cpu",
            mode_value="compact",
            clock=lambda: now[0],
        )
        dumper.observe_decode_step(obs(0))
        dumper.set_runtime_components({InfoComponent.CORE, InfoComponent.STEP_CPU_TIME})
        for forward_ct, t in ((1, 1.0), (2, 1.025)):
            now[0] = t
            dumper.observe_decode_step(obs(forward_ct))
        records = dumper.dump()["records"]
        self.assertEqual([r["forward_ct"] for r in records], [1, 2])
        self.assertEqual([r.get("step_cpu_ms") for r in records], [None, 25.0])
        # Ending the runtime recording keeps the configured components.
        dumper.set_runtime_components(set())
        dump = dumper.dump()
        self.assertEqual((dump["components"], dump["records"]), (["core"], []))


class TestInternalStateHandlers(CustomTestCase):
    def _controller(self, status):
        from sglang.srt.managers.scheduler_components.internal_state import (
            SchedulerInternalStateController,
        )

        worker = mock.Mock()
        worker.sps_autotune_status.return_value = status
        scheduler = SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_dspark=lambda: True), draft_worker=worker
        )
        return SchedulerInternalStateController(scheduler, False), worker

    def _set(self, controller, **values):
        from sglang.srt.managers.io_struct import SetInternalStateReq

        return controller.set_internal_state(SetInternalStateReq(server_args=values))

    def test_handlers_reject_bad_values_and_no_pending_measurement(self):
        controller, worker = self._controller({"pending": True})
        self.assertFalse(self._set(controller, **{at.RECORD_KEY: "yes"}).updated)
        self.assertFalse(self._set(controller, **{at.TABLE_KEY: 3}).updated)
        controller, worker = self._controller({"pending": False, "installed": "x"})
        self.assertFalse(self._set(controller, **{at.TABLE_KEY: None}).updated)
        worker.install_sps_autotune_table.assert_not_called()


# --- Rank agreement over Gloo ----------------------------------------------


def _rank_worker(rank, rendezvous, cache_dir, scenario, out_dir):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        group = dist.group.WORLD
        material = {
            "rank": rank,
            "graphs": {"verify_tiers": TIERS},
            "dspark": {
                "gamma": GAMMA,
                "verify_num_draft_tokens": WIDTH,
                "max_new_tokens": 500,
            },
        }
        prepare = _prepare(material=material)
        if scenario == "skip_on_rank1" and rank == 1:

            def prepare():
                raise at.SpsAutotuneSkip("no graphs on this rank")

        if scenario == "params_mismatch" and rank == 1:
            prepare = _prepare(material=material, params=at.AutotuneParams(passes=3))
        table, prepared, source = at.agree_on_startup(
            mode=at.SpsAutotuneMode.AUTO,
            explicit_table_path=None,
            cache_dir=cache_dir if rank == 0 else cache_dir + "-rank1",
            group=group,
            prepare=prepare,
        )
        result = {
            "source": source,
            "table": None if table is None else table.to_json(),
            "key": None if prepared is None else prepared.key,
        }
        if scenario in ("install", "bad_install") and prepared is not None:
            payload = _install_payload(prepared)
            if scenario == "bad_install" and rank == 1:
                payload = "{"
            installed = []
            result["installed"] = at.install_measured_table(
                payload=payload,
                prepared=prepared,
                group=group,
                install=lambda t, s: installed.append(t.to_json()),
            )
            result["install_calls"] = len(installed)
        Path(out_dir, f"rank{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
class TestRankAgreement(CustomTestCase):
    def _spawn(self, scenario, cache_dir):
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d) / "out"
            out_dir.mkdir()
            processes = [
                context.Process(
                    target=_rank_worker,
                    args=(rank, str(Path(d) / "rv"), cache_dir, scenario, str(out_dir)),
                )
                for rank in range(2)
            ]
            try:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(timeout=120)
                self.assertEqual([p.exitcode for p in processes], [0, 0])
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
            return [
                json.loads((out_dir / f"rank{r}.json").read_text()) for r in range(2)
            ]

    def test_pending_install_then_cache_hit(self):
        with tempfile.TemporaryDirectory() as d:
            cache = str(Path(d) / "cache")
            first = self._spawn("install", cache)
            self.assertEqual([r["source"] for r in first], ["pending", "pending"])
            self.assertEqual(first[0]["key"], first[1]["key"])
            self.assertEqual([r["installed"] for r in first], [True, True])
            self.assertEqual([r["install_calls"] for r in first], [1, 1])
            self.assertEqual(len(os.listdir(cache)), 1)
            self.assertFalse(os.path.isdir(cache + "-rank1"))
            second = self._spawn("agree", cache)
            self.assertEqual([r["source"] for r in second], ["cache", "cache"])
            self.assertEqual(second[0]["table"], second[1]["table"])
            # An undecodable rank-0 cache is a miss on both ranks, not a hang.
            path = Path(cache) / os.listdir(cache)[0]
            path.write_bytes(b"\xff\xfe\x00")
            third = self._spawn("agree", cache)
            self.assertEqual([r["source"] for r in third], ["pending", "pending"])

    def test_one_rank_failing_falls_back_on_both(self):
        with tempfile.TemporaryDirectory() as d:
            cache = str(Path(d) / "cache")
            for scenario, source in (
                ("skip_on_rank1", "skipped"),
                ("params_mismatch", "skipped"),
            ):
                with self.subTest(scenario=scenario):
                    results = self._spawn(scenario, cache)
                    self.assertEqual([r["source"] for r in results], [source, source])
            results = self._spawn("bad_install", cache)
            self.assertEqual([r["installed"] for r in results], [False, False])
            self.assertEqual([r["install_calls"] for r in results], [0, 0])
            self.assertFalse(os.path.isdir(cache) and os.listdir(cache))


if __name__ == "__main__":
    unittest.main()
