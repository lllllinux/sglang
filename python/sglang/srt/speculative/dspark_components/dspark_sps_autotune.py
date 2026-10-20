"""Startup autotuning of the DSpark SPS cost table from real decode steps."""

from __future__ import annotations

import asyncio
import bisect
import contextlib
import enum
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import random
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Sequence

import msgspec
import numpy as np
import torch

from sglang.srt.speculative.dspark_components.dspark_sps import SpsAdditiveCostTable

logger = logging.getLogger(__name__)

# Bump when the measurement or the fit changes, so older cache files miss.
AUTOTUNE_SCHEMA = 3
METADATA_FIELD = "dspark_sps_autotune"
CACHE_FILE_PREFIX = "dspark_sps_"
STATUS_FIELD = "dspark_sps_autotune"
RECORD_KEY = "dspark_sps_autotune_record"
TABLE_KEY = "dspark_sps_autotune_table"

_ENV_PREFIXES = ("SGLANG_", "SGL_", "NCCL_", "TORCH_NCCL_")
_ENV_NAMES = ("CUDA_DEVICE_MAX_CONNECTIONS",)
# Per-run or location-only variables, by name suffix: hashing them would make
# every boot miss.
_ENV_EXCLUDE_SUFFIXES = (
    "_RUN_ID",
    "_CACHE_DIR",
    "_PORT",
    "_PORT_BASE",
    "_HOST",
    "_HOST_IP",
    "_ADDR",
    "_URL",
)
_OWN_ENV_PREFIX = "SGLANG_DSPARK_SPS_AUTOTUNE_"

# Resolved configuration fields that can change the cost of a decode step.
_SERVER_ARG_FIELDS = (
    "model_path",
    "revision",
    "tokenizer_path",
    "tokenizer_mode",
    "tokenizer_backend",
    "trust_remote_code",
    "skip_tokenizer_init",
    "disable_radix_cache",
    "dtype",
    "quantization",
    "kv_cache_dtype",
    "context_length",
    "json_model_override_args",
    "attention_backend",
    "decode_attention_backend",
    "dsv4_attn_backend",
    "sampling_backend",
    "moe_runner_backend",
    "moe_a2a_backend",
    "fp8_gemm_runner_backend",
    "enable_w4a4_mxfp4_megamoe",
    "boundary_reduction",
    "tp_size",
    "dp_size",
    "ep_size",
    "pp_size",
    "attn_cp_size",
    "enable_dp_attention",
    "enable_dp_lm_head",
    "page_size",
    "enable_unified_memory",
    "enable_page_major_kv_layout",
    "dcp_size",
    "max_running_requests",
    "cuda_graph_max_bs_decode",
    "cuda_graph_bs_decode",
    "disable_custom_all_reduce",
    "enable_symm_mem",
    "enable_deepseek_v4_fp4_indexer",
    "enable_torch_compile",
    "torch_compile_max_bs",
    "disable_overlap_schedule",
    "speculative_algorithm",
    "speculative_draft_model_path",
    "speculative_draft_model_revision",
    "speculative_draft_load_format",
    "speculative_num_draft_tokens",
    "speculative_dspark_block_size",
    "speculative_dspark_confidence_sts_path",
    "speculative_dspark_align_verify_tokens_to_graph_tier",
    "speculative_use_block_verification",
    "speculative_draft_attention_backend",
    "speculative_draft_kv_cache_dtype",
    "speculative_moe_runner_backend",
    "speculative_moe_a2a_backend",
    "speculative_enable_w4a4_mxfp4_megamoe",
    "speculative_draft_model_quantization",
    "speculative_boundary_reduction",
)

_PACKAGES = (
    "sglang",
    "torch",
    "triton",
    "flashinfer-python",
    "flashinfer-cubin",
    "flashinfer-jit-cache",
    "sgl-kernel",
    "sglang-kernel",
    "deep-gemm",
    "sgl-deep-gemm",
    "nvidia-nccl-cu12",
    "nvidia-nccl-cu13",
)

_SOURCE_SUFFIXES = (".py", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc")

# Extra decode steps a request may run before the steady window fills (ramp,
# polling latency).
_STEP_SLACK = 48
_RANDOM_TOKEN_LOW = 1000
_POLL_SECONDS = 0.25
# Each long-context request is the shared prefix plus this many own tokens.
_LONG_SUFFIX_LEN = 64
# Smallest shared context worth measuring for the context term.
_MIN_LONG_CONTEXT = 4096


class SpsAutotuneMode(str, enum.Enum):
    OFF = "off"
    AUTO = "auto"
    FORCE = "force"

    @classmethod
    def parse(cls, value: Optional[str]) -> SpsAutotuneMode:
        if value is None:
            return cls.OFF
        return cls(str(value).lower())


class SpsAutotuneSkip(Exception):
    """This rank cannot run the autotuner; every rank keeps its table."""


class AutotuneParams(msgspec.Struct, frozen=True):
    """Measurement settings; part of the cache key."""

    passes: int = 2
    warmup_steps: int = 8
    steady_steps: int = 64
    cell_timeout_seconds: float = 60.0
    max_seconds: float = 900.0
    input_len: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    long_context: int = 32768
    long_context_far: int = 131072
    batch_sizes: tuple[int, ...] = ()

    def validate(self) -> None:
        for name in ("passes", "steady_steps", "input_len"):
            if getattr(self, name) < 1:
                raise ValueError(f"autotune {name} must be >= 1.")
        if self.warmup_steps < 0:
            raise ValueError("autotune warmup_steps must be >= 0.")
        if self.cell_timeout_seconds <= 0 or self.max_seconds <= 0:
            raise ValueError("autotune timeouts must be > 0.")
        from sglang.srt.sampling.sampling_params import SamplingParams

        # Greedy serving can fold acceptance into the verify graph; the
        # bonus-only measurement cannot, so it would time another path.
        if not (math.isfinite(self.temperature) and self.temperature > 0) or (
            SamplingParams(temperature=self.temperature).top_k == 1
        ):
            raise ValueError(
                "autotune temperature must be finite and large enough to sample "
                f"(not greedy), got {self.temperature}."
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("autotune top_p must be in (0, 1].")
        if self.long_context < 0 or self.long_context_far < 0:
            raise ValueError("autotune long contexts must be >= 0.")
        if any(int(b) < 1 for b in self.batch_sizes):
            raise ValueError(f"autotune batch sizes must be >= 1: {self.batch_sizes}")

    @classmethod
    def from_env(cls) -> AutotuneParams:
        from sglang.srt.environ import envs

        raw_bs = envs.SGLANG_DSPARK_SPS_AUTOTUNE_BATCH_SIZES.get()
        params = cls(
            passes=int(envs.SGLANG_DSPARK_SPS_AUTOTUNE_PASSES.get()),
            warmup_steps=int(envs.SGLANG_DSPARK_SPS_AUTOTUNE_WARMUP_STEPS.get()),
            steady_steps=int(envs.SGLANG_DSPARK_SPS_AUTOTUNE_STEADY_STEPS.get()),
            cell_timeout_seconds=float(
                envs.SGLANG_DSPARK_SPS_AUTOTUNE_CELL_TIMEOUT_SECONDS.get()
            ),
            max_seconds=float(envs.SGLANG_DSPARK_SPS_AUTOTUNE_MAX_SECONDS.get()),
            input_len=int(envs.SGLANG_DSPARK_SPS_AUTOTUNE_INPUT_LEN.get()),
            temperature=float(envs.SGLANG_DSPARK_SPS_AUTOTUNE_TEMPERATURE.get()),
            top_p=float(envs.SGLANG_DSPARK_SPS_AUTOTUNE_TOP_P.get()),
            long_context=int(envs.SGLANG_DSPARK_SPS_AUTOTUNE_LONG_CONTEXT.get()),
            long_context_far=int(
                envs.SGLANG_DSPARK_SPS_AUTOTUNE_LONG_CONTEXT_FAR.get()
            ),
            batch_sizes=tuple(
                sorted({int(b) for b in raw_bs.replace(",", " ").split() if b})
            ),
        )
        params.validate()
        return params


# ---------------------------------------------------------------------------
# Cell design
# ---------------------------------------------------------------------------


def _pad_up(values: Sequence[int], x: int) -> int:
    idx = bisect.bisect_left(values, x)
    return values[min(idx, len(values) - 1)]


def default_batch_sizes(max_running: int) -> list[int]:
    """1, 2, 3, 4, 6, 8, 12, 16, 24, ... up to ``max_running``, plus it."""
    sizes, power = {1}, 1
    while power <= max_running:
        sizes.add(power)
        if power * 3 // 2 <= max_running:
            sizes.add(power * 3 // 2)
        power *= 2
    sizes.add(max_running)
    return sorted(b for b in sizes if 1 <= b <= max_running)


def plan_cells(
    *, tiers: Sequence[int], batch_sizes: Sequence[int], verify_width: int
) -> list[tuple[int, int]]:
    """Every (batch size, verify tier) the planner can choose: the tiers that
    ``M`` in ``[bs, bs * verify_width]`` rounds up to."""
    tiers = sorted({int(t) for t in tiers})
    cells = []
    for bs in sorted({int(b) for b in batch_sizes}):
        if bs > tiers[-1]:
            continue
        low, high = _pad_up(tiers, bs), _pad_up(tiers, bs * verify_width)
        cells.extend((bs, t) for t in tiers if low <= t <= high)
    return cells


def long_context_tokens(
    requested: int,
    *,
    context_len: int,
    kv_tokens: int,
    max_bs: int,
    max_new_tokens: int,
    prefix_cache: bool,
    input_len: int = 0,
) -> int:
    """Shared-prefix length of the long-context cells, or 0 to skip them. The
    prefix is held once (the prefix cache shares it, so it needs the cache);
    it and every request's own tokens must fit the context window and most
    of the KV pool."""
    if not prefix_cache or requested <= 0:
        return 0
    own = _LONG_SUFFIX_LEN + max_new_tokens
    length = min(requested, context_len - own - 1)
    if kv_tokens:
        length = min(length, int(0.8 * kv_tokens) - max_bs * own)
    # The context term is fitted from the difference to the short cells.
    return (
        length if length >= max(_MIN_LONG_CONTEXT, input_len + _MIN_LONG_CONTEXT) else 0
    )


def plan_long_cells(cells: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """The cells re-measured at a long shared context to fit the context term:
    each batch size's narrowest and widest tier (the term scales with the
    verify-token count, so the widest tier carries most of the signal)."""
    by_bs: dict[int, list[int]] = {}
    for bs, tier in cells:
        by_bs.setdefault(int(bs), []).append(int(tier))
    out = []
    for bs, tiers in sorted(by_bs.items()):
        out.extend((bs, t) for t in sorted({min(tiers), max(tiers)}))
    return out


def forced_budget_frac(*, bs: int, tier: int, gamma: int) -> float:
    """The ``dspark_force_budget_frac`` that makes ``bs`` requests verify
    ``tier`` tokens: the planner's forced budget is ``int(frac * bs * gamma)``
    tokens above the one-token floor per request."""
    full = bs * gamma
    budget = min(max(tier - bs, 0), full)
    if budget >= full:
        return 1.0
    # Half a token of margin keeps int() from rounding down; frac must be > 0.
    return (budget + 0.5) / full


def cell_step_seconds(
    records: Sequence[dict],
    *,
    bs: int,
    tier: int,
    warmup_steps: int,
    steady_steps: int,
) -> Optional[float]:
    """Median step period of the first ``steady_steps`` decode steps that ran at
    ``bs`` requests and verify tier ``tier``, after ``warmup_steps`` such steps;
    None until that many were recorded."""
    aligned = sorted(
        (int(r["forward_ct"]), float(r["step_cpu_ms"]))
        for r in records
        if r.get("step_cpu_ms") is not None
        and int(r.get("num_running_reqs", -1)) == bs
        and int(r.get("num_verify_tokens", -1)) == tier
    )
    window = aligned[warmup_steps : warmup_steps + steady_steps]
    if len(window) < steady_steps:
        return None
    return statistics.median(ms for _, ms in window) / 1000.0


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def _pava(values: Sequence[float], weights: Sequence[float]) -> list[float]:
    """Weighted least-squares non-decreasing fit (pool adjacent violators)."""
    blocks: list[list[float]] = []  # [mean, weight, count]
    for value, weight in zip(values, weights):
        blocks.append([float(value), max(float(weight), 1e-12), 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, c2 = blocks.pop()
            m1, w1, c1 = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    out: list[float] = []
    for mean, _, count in blocks:
        out.extend([mean] * count)
    return out


def _connected(cells: Sequence[tuple[int, int]]) -> bool:
    parent: dict = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for bs, tier in cells:
        parent[find(("b", bs))] = find(("t", tier))
    return len({find(node) for node in list(parent)}) == 1


class FitStats(msgspec.Struct, frozen=True):
    cells: int
    rms_ms: float
    max_abs_ms: float


def fit_step_cost_table(
    cells: Sequence[tuple[int, int, float]],
    *,
    huber_ms: float = 0.5,
    iterations: int = 200,
) -> tuple[SpsAdditiveCostTable, FitStats]:
    """Fit ``step(bs, M) = bias + alpha(bs) + theta(tier(M))`` to measured
    ``(bs, tier, seconds)`` cells.

    Repeated cells are reduced to their median. The fit is least squares with
    Huber weights (residuals beyond ``huber_ms`` count linearly), under the
    constraint that cost does not decrease with the tier or the batch size
    (alternating isotonic projections). ``theta`` is flat across each tier,
    with probes at its first and last token count; ``alpha`` is probed at the
    measured batch sizes and interpolated between them by the planner.
    """
    by_cell: dict[tuple[int, int], list[float]] = {}
    for bs, tier, seconds in cells:
        bs, tier, seconds = int(bs), int(tier), float(seconds)
        if bs < 1 or tier < 1 or not (0 < seconds < math.inf):
            raise ValueError(f"invalid cell (bs={bs}, tier={tier}, s={seconds}).")
        by_cell.setdefault((bs, tier), []).append(seconds * 1e3)
    if len(by_cell) < 2:
        raise ValueError(f"need at least 2 distinct cells, got {len(by_cell)}.")
    if not _connected(list(by_cell)):
        raise ValueError(
            "cells do not link every batch size and tier through shared "
            "measurements; the additive model is not identifiable."
        )
    keys = sorted(by_cell)
    y = np.array([statistics.median(by_cell[k]) for k in keys])
    bss = sorted({b for b, _ in keys})
    tiers = sorted({t for _, t in keys})
    b_idx = np.array([bss.index(b) for b, _ in keys])
    t_idx = np.array([tiers.index(t) for _, t in keys])

    # Huber-weighted least squares, alpha(bss[0]) = 0.
    cols = len(tiers) + len(bss) - 1
    design = np.zeros((len(keys), cols))
    design[np.arange(len(keys)), t_idx] = 1.0
    for row, bi in enumerate(b_idx):
        if bi > 0:
            design[row, len(tiers) + bi - 1] = 1.0
    weights = np.ones(len(keys))
    for _ in range(30):
        sw = np.sqrt(weights)
        x, *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
        resid = y - design @ x
        new = np.where(
            np.abs(resid) <= huber_ms, 1.0, huber_ms / np.maximum(np.abs(resid), 1e-12)
        )
        if np.allclose(new, weights, atol=1e-6):
            break
        weights = new
    theta = np.array(x[: len(tiers)])
    alpha = np.concatenate([[0.0], x[len(tiers) :]])

    # Monotone in tier and batch size: block-coordinate isotonic projections.
    for _ in range(iterations):
        prev = (theta.copy(), alpha.copy())
        theta = np.array(
            _pava(
                [
                    np.average(
                        y[t_idx == i] - alpha[b_idx[t_idx == i]],
                        weights=weights[t_idx == i],
                    )
                    for i in range(len(tiers))
                ],
                [weights[t_idx == i].sum() for i in range(len(tiers))],
            )
        )
        alpha = np.array(
            _pava(
                [
                    np.average(
                        y[b_idx == i] - theta[t_idx[b_idx == i]],
                        weights=weights[b_idx == i],
                    )
                    for i in range(len(bss))
                ],
                [weights[b_idx == i].sum() for i in range(len(bss))],
            )
        )
        if np.allclose(prev[0], theta, atol=1e-7) and np.allclose(
            prev[1], alpha, atol=1e-7
        ):
            break
    theta, alpha = theta + alpha[0], alpha - alpha[0]
    resid = y - theta[t_idx] - alpha[b_idx]

    base = 0.5 * float(theta.min())
    if not base > 0:
        raise ValueError(f"fitted tier costs must be positive, got {theta.tolist()}.")
    m_probes, theta_seconds, prev_tier = [], [], 0
    for tier, value in zip(tiers, theta):
        for m in [tier] if prev_tier + 1 == tier else [prev_tier + 1, tier]:
            m_probes.append(m)
            theta_seconds.append((float(value) - base) / 1e3)
        prev_tier = tier
    table = SpsAdditiveCostTable(
        bias_seconds=base / 1e3,
        bs_probes=bss,
        alpha_seconds=[float(a) / 1e3 for a in alpha],
        m_probes=m_probes,
        theta_seconds=theta_seconds,
    )
    stats = FitStats(
        cells=len(keys),
        rms_ms=float(np.sqrt(np.mean(resid**2))),
        max_abs_ms=float(np.max(np.abs(resid))),
    )
    return table, stats


def reference_context_tokens(params: AutotuneParams) -> float:
    """Mean context of a short cell's timed steps: the prompt plus the tokens
    generated until the middle of the steady window, one committed token per
    step (the measurement commits bonus tokens only)."""
    return float(params.input_len + params.warmup_steps + params.steady_steps / 2)


def context_tokens(bs: int, tier: int, *, verify_width: int, max_slots: int = 0) -> int:
    """Tokens of a pinned ``(bs, tier)`` measurement step that read request
    context: the drafts that exist, or the whole tier when every captured slot
    is taken (its padding then extends the last request's row)."""
    if not verify_width or (max_slots and bs >= max_slots):
        return tier
    return min(tier, bs * verify_width)


def without_context(
    cells: Sequence[tuple[int, int, float]],
    *,
    context_fit: dict,
    reference_context_tokens: float,
    verify_width: int,
    max_slots: int = 0,
) -> list[tuple[int, int, float]]:
    """``(bs, tier, seconds)`` cells measured at ``reference_context_tokens``
    with their context cost removed, so the baseline fits context zero."""
    return [
        (
            bs,
            tier,
            seconds
            - reference_context_tokens
            * (
                context_fit["context_seconds"]
                * context_tokens(
                    bs, tier, verify_width=verify_width, max_slots=max_slots
                )
                + context_fit["context_step_seconds"]
            ),
        )
        for bs, tier, seconds in cells
    ]


def fit_context_seconds(
    short_ms: dict[tuple[int, int], float],
    long_ms: dict[tuple[int, int], float],
    *,
    context_delta: float,
    verify_width: int = 0,
    max_slots: int = 0,
    huber_ns: float = 5.0,
) -> dict:
    """Context cost from cells measured at a short and at a long context. Per
    paired cell, the step-time difference per context token is modeled as
    ``step + token * executed`` (a per-step part and a per-verify-token part)
    and fitted by Huber-weighted least squares. Neither part is negative."""
    if context_delta <= 0:
        raise ValueError(f"context_delta must be > 0, got {context_delta}.")
    xs, ys, keys = [], [], []
    for (bs, tier), long_value in sorted(long_ms.items()):
        if (bs, tier) in short_ms:
            executed = context_tokens(
                bs, tier, verify_width=verify_width, max_slots=max_slots
            )
            xs.append(float(executed))
            ys.append((long_value - short_ms[(bs, tier)]) / 1e3 / context_delta)
            keys.append((bs, tier))
    if not keys:
        raise ValueError("no cell was measured at both contexts.")
    if len(set(xs)) < 2:
        # One executed width cannot separate the per-step from the per-token part.
        raise ValueError("the paired cells share one executed width.")
    x, y = np.array(xs), np.array(ys)
    w = np.ones_like(y)
    design = np.stack([np.ones_like(x), x], axis=1)
    for _ in range(30):
        sw = np.sqrt(w)
        (step, token), *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
        resid = np.abs(y - step - token * x) * 1e9  # ns per context token
        new = np.where(resid <= huber_ns, 1.0, huber_ns / np.maximum(resid, 1e-12))
        if np.allclose(new, w, atol=1e-6):
            break
        w = new
    if token < 0:
        token, step = 0.0, max(float(np.median(y)), 0.0)
    elif step < 0:
        step, token = 0.0, max(float(np.sum(w * x * y) / np.sum(w * x * x)), 0.0)
    return {
        "context_seconds": float(token),
        "context_step_seconds": float(step),
        "context_delta": float(context_delta),
        "per_cell_ns": {
            f"{b},{t}": round(v * 1e9 / xe, 4) for (b, t), v, xe in zip(keys, ys, xs)
        },
    }


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def cache_key(material: Any) -> str:
    return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()


def cache_path_for(cache_dir: str, key: str) -> Path:
    return Path(os.path.expanduser(cache_dir)) / f"{CACHE_FILE_PREFIX}{key[:32]}.json"


def encode_cache_payload(
    *, table: SpsAdditiveCostTable, key: str, metadata: dict
) -> str:
    """The table's own fields plus a metadata object, so the file also loads as
    a plain ``--speculative-dspark-sps-table-path`` table."""
    payload = msgspec.to_builtins(table)
    payload[METADATA_FIELD] = {**metadata, "schema": AUTOTUNE_SCHEMA, "key": key}
    return json.dumps(payload, indent=1, sort_keys=True, default=str)


def decode_cache_payload(data: str, *, key: str) -> Optional[SpsAdditiveCostTable]:
    """The table when ``data`` was written for ``key`` and validates, else None."""
    try:
        payload = json.loads(data)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    meta = payload.get(METADATA_FIELD)
    if (
        not isinstance(meta, dict)
        or meta.get("schema") != AUTOTUNE_SCHEMA
        or meta.get("key") != key
    ):
        return None
    try:
        return SpsAdditiveCostTable.from_json(data)
    except (msgspec.ValidationError, msgspec.DecodeError, TypeError, ValueError):
        return None


def read_cached_payload(path: Path, *, key: str) -> Optional[str]:
    try:
        data = path.read_text(encoding="utf-8")
    except (OSError, ValueError):  # ValueError covers undecodable bytes
        return None
    return data if decode_cache_payload(data, key=key) is not None else None


def write_cache_atomic(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------
# Rank agreement
# ---------------------------------------------------------------------------


class _Ranks:
    """Collectives over a CPU (gloo) group; a single rank needs no group."""

    def __init__(self, group) -> None:
        self.group = group
        if group is None:
            self.rank, self.world_size, self.src = 0, 1, 0
        else:
            import torch.distributed as dist

            self.rank = dist.get_rank(group=group)
            self.world_size = dist.get_world_size(group=group)
            self.src = dist.get_process_group_ranks(group)[0]

    def all_ok(self, ok: bool) -> bool:
        if self.world_size == 1:
            return ok
        import torch.distributed as dist

        flag = torch.tensor([1 if ok else 0], dtype=torch.int64)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.group)
        return bool(flag.item())

    def all_gather(self, obj: Any) -> list[Any]:
        if self.world_size == 1:
            return [obj]
        import torch.distributed as dist

        out: list[Any] = [None] * self.world_size
        dist.all_gather_object(out, obj, group=self.group)
        return out

    def broadcast(self, obj: Any) -> Any:
        if self.world_size == 1:
            return obj
        import torch.distributed as dist

        box = [obj if self.rank == 0 else None]
        dist.broadcast_object_list(box, src=self.src, group=self.group)
        return box[0]


def _agree_on_table(ranks: _Ranks, payload: Optional[str], *, key: str):
    """Decode rank 0's payload on every rank; a table only if all ranks can."""
    payload = ranks.broadcast(payload)
    table = decode_cache_payload(payload, key=key) if payload is not None else None
    if not ranks.all_ok(table is not None):
        return None
    return table


# ---------------------------------------------------------------------------
# Scheduler side
# ---------------------------------------------------------------------------


class AutotunePrepared(msgspec.Struct):
    """A pending measurement; every rank agreed on it at startup."""

    key: str
    cache_path: Path
    material: dict
    cells: tuple[tuple[int, int], ...]
    params: AutotuneParams

    def status(self) -> dict:
        dspark = self.material["dspark"]
        return {
            "pending": True,
            "key": self.key,
            "cells": [list(cell) for cell in self.cells],
            "gamma": int(dspark["gamma"]),
            "verify_num_draft_tokens": int(dspark["verify_num_draft_tokens"]),
            "max_new_tokens": int(dspark["max_new_tokens"]),
            "kv_tokens": int(dspark.get("kv_tokens") or 0),
            "max_slots": int(dspark.get("max_slots") or 0),
            "verify_tiers": list(self.material["graphs"]["verify_tiers"]),
            "long_context": int(dspark.get("long_context") or 0),
            "long_context_far": int(dspark.get("long_context_far") or 0),
            "prompt_source": dspark.get("prompt_source", "text"),
            "long_cells": (
                [list(c) for c in plan_long_cells(self.cells)]
                if dspark.get("long_context")
                else []
            ),
            "params": msgspec.to_builtins(self.params),
        }


def agree_on_startup(
    *,
    mode: SpsAutotuneMode,
    explicit_table_path: Optional[str],
    cache_dir: str,
    group,
    prepare: Callable[[], tuple[dict, list[tuple[int, int]], AutotuneParams]],
) -> tuple[Optional[SpsAdditiveCostTable], Optional[AutotunePrepared], str]:
    """Every rank of ``group`` must call this. Returns ``(table, prepared,
    source)``: ``table`` set means install it, ``prepared`` set means a
    measurement is pending, neither means keep the configured table."""
    if mode is SpsAutotuneMode.OFF:
        return None, None, "off"
    if mode is SpsAutotuneMode.AUTO and explicit_table_path:
        return None, None, "explicit-table"
    ranks = _Ranks(group)
    material, cells, params, reason = None, None, None, ""
    try:
        material, cells, params = prepare()
    except SpsAutotuneSkip as exc:
        reason = str(exc)
    except Exception as exc:  # local only: report and opt out together
        logger.exception("DSpark SPS autotune preparation failed.")
        reason = f"{type(exc).__name__}: {exc}"
    gathered = ranks.all_gather(
        {
            "material": material,
            "cells": cells,
            "params": None if params is None else msgspec.to_builtins(params),
            "reason": reason,
        }
    )
    reasons = [
        f"rank {i}: {g['reason'] or 'not prepared'}"
        for i, g in enumerate(gathered)
        if g["material"] is None
    ]
    if reasons:
        if ranks.rank == 0:
            logger.warning(
                "DSpark SPS autotune skipped; keeping the configured table (%s).",
                "; ".join(reasons),
            )
        return None, None, "skipped"
    for field in ("cells", "params"):
        if any(g[field] != gathered[0][field] for g in gathered):
            if ranks.rank == 0:
                logger.warning(
                    "DSpark SPS autotune skipped: ranks disagree on %s.", field
                )
            return None, None, "skipped"

    key = cache_key(
        {"schema": AUTOTUNE_SCHEMA, "ranks": [g["material"] for g in gathered]}
    )
    path = cache_path_for(cache_dir, key)
    if mode is SpsAutotuneMode.AUTO:
        payload = None
        if ranks.rank == 0:
            try:
                payload = read_cached_payload(path, key=key)
            except Exception:
                logger.exception("DSpark SPS autotune could not read %s.", path)
        table = _agree_on_table(ranks, payload, key=key)
        if table is not None:
            if ranks.rank == 0:
                logger.info(
                    "DSpark SPS autotune: loaded cached table %s (key %s).",
                    path,
                    key[:16],
                )
            return table, None, "cache"
    prepared = AutotunePrepared(
        key=key,
        cache_path=path,
        material=gathered[0]["material"],
        cells=tuple(tuple(c) for c in cells),
        params=params,
    )
    if ranks.rank == 0:
        logger.info(
            "DSpark SPS autotune: no cached table for key %s; the HTTP server "
            "measures %d cells before it is ready (other entrypoints keep the "
            "configured table).",
            key[:16],
            len(prepared.cells),
        )
    return None, prepared, "pending"


def decode_install_request(
    payload: Any, *, key: str
) -> tuple[Optional[SpsAdditiveCostTable], dict]:
    """Validate a measured-table request: ``{"key", "table", "metadata"}``."""
    if not isinstance(payload, str):
        return None, {}
    try:
        request = json.loads(payload)
    except ValueError:
        return None, {}
    if not isinstance(request, dict) or request.get("key") != key:
        return None, {}
    try:
        table = msgspec.convert(request.get("table"), SpsAdditiveCostTable)
    except (msgspec.ValidationError, TypeError, ValueError):
        return None, {}
    metadata = request.get("metadata")
    return table, metadata if isinstance(metadata, dict) else {}


def install_measured_table(
    *, payload: Any, prepared: AutotunePrepared, group, install: Callable
) -> bool:
    """Install a measured table on every rank or on none, then cache it on
    rank 0. Runs inside ``/set_internal_state``, which every TP rank processes
    in the same scheduler iteration, so the collective lines up."""
    ranks = _Ranks(group)
    table, metadata = decode_install_request(payload, key=prepared.key)
    if not ranks.all_ok(table is not None):
        if ranks.rank == 0:
            logger.warning("DSpark SPS autotune: measured table rejected.")
        return False
    install(table, f"autotune-measured:{prepared.cache_path}")
    if ranks.rank == 0:
        try:
            write_cache_atomic(
                prepared.cache_path,
                encode_cache_payload(
                    table=table,
                    key=prepared.key,
                    metadata={**metadata, "material": prepared.material},
                ),
            )
        except Exception as exc:
            logger.warning(
                "DSpark SPS autotune could not write %s (%s); the table is used "
                "for this run only.",
                prepared.cache_path,
                exc,
            )
    return True


# ---------------------------------------------------------------------------
# Cache-key material
# ---------------------------------------------------------------------------


def env_digest(environ: Optional[dict] = None) -> str:
    environ = os.environ if environ is None else environ
    items = sorted(
        (k, v)
        for k, v in environ.items()
        if (k.startswith(_ENV_PREFIXES) or k in _ENV_NAMES)
        and not k.startswith(_OWN_ENV_PREFIX)
        and not k.endswith(_ENV_EXCLUDE_SUFFIXES)
    )
    return hashlib.sha256(canonical_json(items).encode("utf-8")).hexdigest()


def source_digest(roots: Sequence[Path]) -> str:
    """Hash of the installed source tree, standing in for a commit id that
    patched builds do not change."""
    digest = hashlib.sha256()
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix not in _SOURCE_SUFFIXES or not path.is_file():
                continue
            digest.update(str(path.relative_to(root.parent)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def package_versions() -> dict:
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def model_config_material(model_runner) -> Optional[dict]:
    """Digest of the loaded model config and its resolved Hub revision."""
    hf_config = getattr(getattr(model_runner, "model_config", None), "hf_config", None)
    if hf_config is None:
        return None
    config = hf_config.to_dict() if hasattr(hf_config, "to_dict") else vars(hf_config)
    return {
        "config": cache_key(config),
        "commit": getattr(hf_config, "_commit_hash", None),
    }


def server_arg_material() -> dict:
    from sglang.srt.runtime_context import get_context

    context = get_context()
    return {name: context.config_leaf(name) for name in _SERVER_ARG_FIELDS}


def _nvml_material(device_index: int) -> Optional[dict]:
    """Power limit, driver and maximum SM clock of this CUDA device, or None
    when NVML is unavailable."""
    try:
        import pynvml
    except ImportError:
        return None
    try:
        props = torch.cuda.get_device_properties(device_index)
        bus = getattr(props, "pci_bus_id", None)
        dev = getattr(props, "pci_device_id", None)
        domain = getattr(props, "pci_domain_id", None)
        if bus is None or dev is None or domain is None:
            return None
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByPciBusId(
                f"{int(domain):08X}:{int(bus):02X}:{int(dev):02X}.0"
            )
            return {
                "power_limit_mw": int(pynvml.nvmlDeviceGetEnforcedPowerLimit(handle)),
                "max_sm_clock_mhz": int(
                    pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
                ),
                "driver": str(pynvml.nvmlSystemGetDriverVersion()),
            }
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def device_material(device_index: int) -> dict:
    props = torch.cuda.get_device_properties(device_index)
    return {
        "name": props.name,
        "capability": [int(props.major), int(props.minor)],
        "total_memory": int(props.total_memory),
        "multi_processor_count": int(props.multi_processor_count),
        "visible_device_count": int(torch.cuda.device_count()),
        "cuda": torch.version.cuda,
        "nvml": _nvml_material(device_index),
    }


def _hook_names(model_runner) -> list[str]:
    return [
        getattr(hook, "__qualname__", type(hook).__qualname__)
        for hook in getattr(model_runner, "capture_tail_hooks", ())
    ]


def _compile_bs(graph_runner) -> Optional[list[int]]:
    compile_bs = getattr(graph_runner, "compile_bs", None)
    return None if compile_bs is None else [int(b) for b in compile_bs]


def config_skip_reason() -> Optional[str]:
    """Unsupported layouts must opt out before TP collectives."""
    from sglang.srt.runtime_context import (
        get_disagg,
        get_exec,
        get_parallel,
        get_serving,
    )

    if get_disagg().disaggregation_mode != "null":
        return "disaggregated serving is not supported"
    parallel = get_parallel()
    if parallel.attn_dp_enabled or parallel.dp_size > 1:
        return "data parallelism is not supported"
    if parallel.pp_size > 1:
        return "pipeline parallelism is not supported"
    if get_exec().moe.elastic_ep_backend is not None:
        return "elastic expert parallelism is not supported"
    if get_serving().tokenizer_worker_num > 1:
        return "multiple tokenizer workers are not supported"
    return None


def _measurement_plan(
    *,
    params: AutotuneParams,
    target_model_runner,
    tiers: list[int],
    gamma: int,
    verify_num_draft_tokens: int,
    mode_value: str,
) -> tuple[dict, list[tuple[int, int]]]:
    """The ``dspark`` material (what the server side needs) and the cells."""
    from sglang.srt.runtime_context import get_memory, get_schedule, get_serving

    target_runner = target_model_runner.decode_cuda_graph_runner
    # The worker's effective limit (memory may cap the configured one).
    max_running = int(
        getattr(target_model_runner, "max_running_requests", None)
        or get_schedule().max_running_requests
        or target_runner.max_bs
    )
    max_running = min(max_running, int(target_runner.max_bs))
    # Long enough for warmup and steady steps, within the context window.
    steps = params.warmup_steps + params.steady_steps + _STEP_SLACK
    context_len = int(
        getattr(getattr(target_model_runner, "model_config", None), "context_len", 0)
        or 1 << 30
    )
    max_new_tokens = min(
        steps * int(verify_num_draft_tokens), context_len - params.input_len - 1
    )
    if max_new_tokens < steps:
        raise SpsAutotuneSkip(
            f"the context length {context_len} is too short for {steps} steps"
        )
    batch_sizes = [
        b
        for b in (params.batch_sizes or default_batch_sizes(max_running))
        if b <= max_running
    ]
    cells = plan_cells(
        tiers=tiers, batch_sizes=batch_sizes, verify_width=verify_num_draft_tokens
    )
    if not cells:
        raise SpsAutotuneSkip("no measurable (batch size, tier) cells")
    kv_tokens = int(getattr(target_model_runner, "max_total_num_tokens", 0) or 0)
    sizing = dict(
        context_len=context_len,
        kv_tokens=kv_tokens,
        max_bs=max(b for b, _ in cells),
        max_new_tokens=max_new_tokens,
        prefix_cache=not get_memory().disable_radix_cache,
    )
    long_context = long_context_tokens(
        params.long_context, input_len=params.input_len, **sizing
    )
    # A second, farther context fits the slope beyond the first.
    long_context_far = (
        long_context_tokens(params.long_context_far, input_len=long_context, **sizing)
        if long_context
        else 0
    )
    dspark = {
        "gamma": int(gamma),
        "verify_num_draft_tokens": int(verify_num_draft_tokens),
        "max_new_tokens": int(max_new_tokens),
        "kv_tokens": kv_tokens,
        "max_slots": int(target_runner.max_bs),
        "max_running_requests": max_running,
        "long_context": int(long_context),
        "long_context_far": int(long_context_far),
        "prompt_source": "random" if get_serving().skip_tokenizer_init else "text",
        "ragged_verify_mode": mode_value,
    }
    return dspark, cells


def _draft_kv_layout(draft_model_runner) -> Optional[str]:
    """Whether the draft's KV is fused into the target's unified pages or kept
    in a private pool; the configurator decides at boot."""
    if draft_model_runner is None:
        return None
    from sglang.srt.mem_cache.unified_draft_pool import fused_draft_host_allocator

    fused = fused_draft_host_allocator(draft_model_runner.token_to_kv_pool)
    return "private" if fused is None else "fused"


def _graph_material(target_model_runner, draft_model_runner, draft_sampler) -> dict:
    target_runner = target_model_runner.decode_cuda_graph_runner
    draft_runner = getattr(draft_model_runner, "decode_cuda_graph_runner", None)
    return {
        "verify_tiers": sorted(int(t) for t in target_runner.capture_num_tokens),
        "target_capture_bs": [int(b) for b in target_runner.capture_bs],
        "draft_capture_bs": (
            None if draft_runner is None else [int(b) for b in draft_runner.capture_bs]
        ),
        "target_compile_bs": _compile_bs(target_runner),
        "draft_compile_bs": _compile_bs(draft_runner),
        "target_tail_hooks": _hook_names(target_model_runner),
        "draft_tail_hooks": _hook_names(draft_model_runner),
        "draft_kv_layout": _draft_kv_layout(draft_model_runner),
        "draft_sampler": (
            None
            if draft_sampler is None
            else {
                "type": type(draft_sampler).__qualname__,
                "folded_sampling": getattr(draft_sampler, "folded_sampling", None),
            }
        ),
    }


def _software_material() -> dict:
    import sglang

    package_root = Path(sglang.__file__).resolve().parent
    return {
        "sglang_version": getattr(sglang, "__version__", None),
        "packages": package_versions(),
        "source": source_digest([package_root / "srt", package_root / "kernels"]),
        "env": env_digest(),
    }


def build_startup_material(
    *,
    planner,
    target_model_runner,
    draft_model_runner,
    gamma: int,
    verify_num_draft_tokens: int,
    simulate_acc_len: float,
    draft_sampler=None,
) -> tuple[dict, list[tuple[int, int]], AutotuneParams]:
    from sglang.srt.runtime_context import get_parallel

    if not planner.schedules_verify_budget or not planner.is_compact_mode:
        raise SpsAutotuneSkip("it requires SGLANG_RAGGED_VERIFY_MODE=compact")
    if simulate_acc_len > 0:
        raise SpsAutotuneSkip("SGLANG_SIMULATE_ACC_LEN is set (profiling mode)")
    try:
        params = AutotuneParams.from_env()
    except ValueError as exc:
        raise SpsAutotuneSkip(str(exc)) from exc
    target_runner = target_model_runner.decode_cuda_graph_runner
    if (
        target_runner is None
        or not target_runner.ragged_verify_mode
        or not target_runner.capture_num_tokens
    ):
        raise SpsAutotuneSkip("the target did not capture compact verify tiers")
    graphs = _graph_material(target_model_runner, draft_model_runner, draft_sampler)
    dspark, cells = _measurement_plan(
        params=params,
        target_model_runner=target_model_runner,
        tiers=graphs["verify_tiers"],
        gamma=gamma,
        verify_num_draft_tokens=verify_num_draft_tokens,
        mode_value=planner.mode_value,
    )
    tp_group = get_parallel().tp_group
    material = {
        "device": device_material(torch.cuda.current_device()),
        "parallel": {
            "world_size": int(tp_group.world_size),
            "tp_rank": int(tp_group.rank_in_group),
        },
        "server_args": server_arg_material(),
        "model_config": model_config_material(target_model_runner),
        "draft_config": model_config_material(draft_model_runner),
        "graphs": graphs,
        "dspark": dspark,
        "params": msgspec.to_builtins(params),
        "software": _software_material(),
    }
    return material, cells, params


def maybe_autotune_sps_table(
    *,
    planner,
    target_model_runner,
    draft_model_runner,
    gamma: int,
    verify_num_draft_tokens: int,
    simulate_acc_len: float,
    draft_sampler=None,
) -> Optional[AutotunePrepared]:
    """Startup half of ``--speculative-dspark-sps-autotune``: install a cached
    table, or return the pending measurement. Off by default. Failures are
    logged and leave every rank on the table it already had."""
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_parallel, get_spec

    spec = get_spec()
    mode = SpsAutotuneMode.parse(spec.speculative_dspark_sps_autotune)
    if mode is SpsAutotuneMode.OFF:
        return None
    group = get_parallel().tp_group
    is_rank0 = group.rank_in_group == 0
    explicit = spec.speculative_dspark_sps_table_path
    if mode is SpsAutotuneMode.AUTO and explicit:
        if is_rank0:
            logger.info(
                "DSpark SPS autotune (auto) not run: "
                "--speculative-dspark-sps-table-path %s is configured; use force "
                "to measure anyway.",
                explicit,
            )
        return None
    skip = config_skip_reason()
    if skip is not None:
        if is_rank0:
            logger.warning("DSpark SPS autotune skipped: %s.", skip)
        return None
    cpu_group = group.cpu_group if group.world_size > 1 else None
    try:
        table, prepared, source = agree_on_startup(
            mode=mode,
            explicit_table_path=explicit,
            cache_dir=envs.SGLANG_DSPARK_SPS_AUTOTUNE_CACHE_DIR.get(),
            group=cpu_group,
            prepare=lambda: build_startup_material(
                planner=planner,
                target_model_runner=target_model_runner,
                draft_model_runner=draft_model_runner,
                gamma=gamma,
                verify_num_draft_tokens=verify_num_draft_tokens,
                simulate_acc_len=simulate_acc_len,
                draft_sampler=draft_sampler,
            ),
        )
    except Exception:
        # Local steps are caught inside; what reaches here is a failed
        # collective, which fails on every rank.
        logger.exception("DSpark SPS autotune failed; keeping the configured table.")
        return None
    if table is not None:
        planner.install_sps_table(table, source=f"autotune-{source}")
    return prepared


# ---------------------------------------------------------------------------
# Server side: the measurement
# ---------------------------------------------------------------------------


class MeasureClient:
    """What the measurement needs from the server: control requests, per-step
    records and a synthetic load. Wraps a ``TokenizerManager``; every call is
    bounded by ``rpc_timeout`` seconds."""

    def __init__(self, tokenizer_manager, *, rpc_timeout: float = 60.0) -> None:
        self.tm = tokenizer_manager
        self.rpc_timeout = rpc_timeout

    async def _call(self, awaitable):
        # Never cancel the communicator's wait: its reply is not tagged, so a
        # late reply to a cancelled call would answer the next one. The
        # shielded call keeps its place (the communicator serializes calls),
        # and later calls queue behind it under their own deadlines.
        task = asyncio.ensure_future(awaitable)
        return await asyncio.wait_for(asyncio.shield(task), timeout=self.rpc_timeout)

    async def status(self) -> list[dict]:
        states = await self._call(self.tm.get_internal_state())
        return [s.get(STATUS_FIELD) for s in states if s.get(STATUS_FIELD)]

    async def set_state(self, **values) -> bool:
        from sglang.srt.managers.io_struct import SetInternalStateReq

        results = await self._call(
            self.tm.set_internal_state(SetInternalStateReq(server_args=values))
        )
        return bool(results) and all(results)

    async def records(self) -> list[dict]:
        states = await self._call(self.tm.get_internal_state())
        records: list[dict] = []
        for state in states:
            payload = state.get("dspark_info_record") or {}
            records.extend(payload.get("records", []))
        return records

    def start_load(self, *, input_ids, sampling_params) -> asyncio.Task:
        from sglang.srt.managers.io_struct import GenerateReqInput

        request = GenerateReqInput(input_ids=input_ids, sampling_params=sampling_params)

        async def drain():
            with contextlib.suppress(Exception):
                async for _ in self.tm.generate_request(request, None):
                    pass

        return asyncio.ensure_future(drain())

    async def stop_load(self, task: asyncio.Task) -> None:
        self.tm.abort_request(abort_all=True)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.shield(task), timeout=self.rpc_timeout)
        if not task.done():
            task.cancel()

    async def flush_cache(self) -> None:
        with contextlib.suppress(Exception):
            await self._call(self.tm.flush_cache())

    def text_token_ids(self, count: int) -> Optional[list[int]]:
        """``count`` token ids of real text (this package's Python source), or
        None without a tokenizer."""
        tokenizer = getattr(self.tm, "tokenizer", None)
        if tokenizer is None:
            return None
        return source_text_token_ids(tokenizer, count)


def source_text_token_ids(tokenizer, count: int) -> Optional[list[int]]:
    import sglang

    root = Path(sglang.__file__).resolve().parent / "srt"
    ids: list[int] = []
    for path in sorted(root.rglob("*.py")):
        if len(ids) >= count:
            break
        with contextlib.suppress(OSError, UnicodeDecodeError):
            text = f"# {path.relative_to(root)}\n{path.read_text()}\n"
            ids.extend(tokenizer.encode(text, add_special_tokens=False))
    return ids[:count] if len(ids) >= count else None


async def measure_cell(
    client: MeasureClient,
    *,
    bs: int,
    tier: int,
    gamma: int,
    max_new_tokens: int,
    params: AutotuneParams,
    prompts: Sequence[Sequence[int]],
    shared_prefix_len: int = 0,
    kv_tokens: int = 0,
    sleep: Callable[[float], Awaitable] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Optional[float]:
    """Median step period (seconds) of one cell, or None if it timed out.
    ``prompts`` holds ``bs`` token lists; the first ``shared_prefix_len``
    tokens of each are one cached prefix shared by all of them."""
    if not await client.set_state(
        dspark_force_budget_frac=forced_budget_frac(bs=bs, tier=tier, gamma=gamma),
        dspark_clear_info_records=True,
    ):
        raise RuntimeError("the scheduler rejected the forced verify budget")
    # The requests must fit the KV pool together; leave a fifth for the rest.
    if kv_tokens:
        own_prompt = max(len(p) for p in prompts) - shared_prefix_len
        max_new_tokens = min(
            max_new_tokens,
            (int(0.8 * kv_tokens) - shared_prefix_len) // bs - own_prompt,
        )
    if max_new_tokens < params.warmup_steps + params.steady_steps + 2:
        logger.warning(
            "DSpark SPS autotune: %d requests do not fit the KV pool; cell "
            "(%d, %d) skipped.",
            bs,
            bs,
            tier,
        )
        return None
    sampling_params = {
        "temperature": params.temperature,
        "max_new_tokens": max_new_tokens,
        "ignore_eos": True,
    }
    if params.top_p < 1.0:
        sampling_params["top_p"] = params.top_p
    task = client.start_load(
        input_ids=[list(p) for p in prompts[:bs]], sampling_params=sampling_params
    )
    deadline = clock() + params.cell_timeout_seconds
    seconds = None
    try:
        while clock() < deadline:
            await sleep(_POLL_SECONDS)
            seconds = cell_step_seconds(
                await client.records(),
                bs=bs,
                tier=tier,
                warmup_steps=params.warmup_steps,
                steady_steps=params.steady_steps,
            )
            if seconds is not None or task.done():
                break
    finally:
        await client.stop_load(task)
    return seconds


class _Prompts:
    """Text windows, or random token IDs without a tokenizer."""

    def __init__(
        self,
        *,
        text_ids: Optional[Sequence[int]],
        vocab_size: int,
        input_len: int,
        rng: random.Random,
    ) -> None:
        self.text_ids = list(text_ids) if text_ids else None
        self.vocab_size = vocab_size
        self.input_len = input_len
        self.rng = rng

    def window(self, length: int) -> list[int]:
        ids = self.text_ids
        if ids and len(ids) > length + 1:
            # Offsets >= 1: no window starts like the shared long prefix.
            start = self.rng.randrange(1, len(ids) - length)
            return ids[start : start + length]
        high = max(self.vocab_size - 1000, _RANDOM_TOKEN_LOW + 1)
        return [self.rng.randrange(_RANDOM_TOKEN_LOW, high) for _ in range(length)]

    def short(self, bs: int) -> list[list[int]]:
        return [self.window(self.input_len) for _ in range(bs)]

    def shared_prefix(self, length: int) -> list[int]:
        ids = self.text_ids
        if ids and len(ids) >= length:
            return ids[:length]
        return random.Random(1).choices(
            range(
                _RANDOM_TOKEN_LOW, max(self.vocab_size - 1000, _RANDOM_TOKEN_LOW + 1)
            ),
            k=length,
        )

    def long(self, bs: int, prefix: Sequence[int]) -> list[list[int]]:
        return [list(prefix) + self.window(_LONG_SUFFIX_LEN) for _ in range(bs)]


class _Samples(msgspec.Struct):
    """Step periods (seconds) per ``(bs, tier)`` cell at each context."""

    short: dict[tuple[int, int], list[float]]
    long: dict[tuple[int, int], list[float]]
    far: dict[tuple[int, int], list[float]]
    failed: list[tuple[int, int]] = []


def _measurement_prompts(
    client: MeasureClient,
    status: dict,
    params: AutotuneParams,
    *,
    longest_context: int,
    vocab_size: int,
) -> _Prompts:
    text_ids = None
    with contextlib.suppress(Exception):
        text_ids = client.text_token_ids(
            max(longest_context + 4 * _LONG_SUFFIX_LEN, 64 * params.input_len)
        )
    if status.get("prompt_source") == "text" and not text_ids:
        # The cache key promises text prompts; do not cache a random-id table.
        raise RuntimeError("the tokenizer did not produce the text prompts")
    return _Prompts(
        text_ids=text_ids,
        vocab_size=vocab_size,
        input_len=params.input_len,
        rng=random.Random(0),
    )


async def _collect_samples(
    client: MeasureClient,
    status: dict,
    params: AutotuneParams,
    prompts: _Prompts,
    *,
    cells: list[tuple[int, int]],
    long_cells: list[tuple[int, int]],
    long_context: int,
    long_context_far: int,
    started: float,
    sleep: Callable[[float], Awaitable],
    clock: Callable[[], float],
) -> _Samples:
    samples = _Samples(
        short={c: [] for c in cells},
        long={c: [] for c in long_cells},
        far={c: [] for c in long_cells},
    )

    async def one(cell, *, prefix=None):
        if clock() - started > params.max_seconds:
            raise TimeoutError(f"measurement exceeded {params.max_seconds:.0f} s")
        bs, tier = cell
        return await measure_cell(
            client,
            bs=bs,
            tier=tier,
            gamma=int(status["gamma"]),
            max_new_tokens=int(status["max_new_tokens"]),
            kv_tokens=int(status.get("kv_tokens") or 0),
            params=params,
            prompts=prompts.short(bs) if prefix is None else prompts.long(bs, prefix),
            shared_prefix_len=0 if prefix is None else len(prefix),
            sleep=sleep,
            clock=clock,
        )

    try:
        if not await client.set_state(**{RECORD_KEY: True}):
            raise RuntimeError("the scheduler rejected autotune recording")
        await one(cells[-1])  # warmup, not counted
        for pass_idx in range(params.passes):
            for cell in cells if pass_idx % 2 == 0 else cells[::-1]:
                seconds = await one(cell)
                if seconds is None:
                    samples.failed.append(cell)
                else:
                    samples.short[cell].append(seconds)
        for length, store in (
            (long_context, samples.long),
            (long_context_far, samples.far),
        ):
            if not (long_cells and length):
                continue
            prefix = prompts.shared_prefix(length)
            # Prefill the shared prefix once so every long cell hits the cache.
            await one(long_cells[0], prefix=prefix)
            for pass_idx in range(params.passes):
                for cell in long_cells if pass_idx % 2 == 0 else long_cells[::-1]:
                    seconds = await one(cell, prefix=prefix)
                    if seconds is not None:
                        store[cell].append(seconds)
    finally:
        # The terminal table request also unpins and stops recording on every
        # rank; this only restores the scheduler early.
        with contextlib.suppress(Exception):
            await client.set_state(dspark_force_budget_frac=None)
        with contextlib.suppress(Exception):
            await client.set_state(**{RECORD_KEY: False})
        await client.flush_cache()
    return samples


def _medians_ms(samples: dict) -> dict[tuple[int, int], float]:
    return {cell: statistics.median(v) * 1e3 for cell, v in samples.items() if v}


def _fit_context(
    medians: dict,
    long_medians: dict,
    far_medians: dict,
    *,
    params: AutotuneParams,
    long_context: int,
    long_context_far: int,
    verify_width: int,
    max_slots: int,
) -> Optional[dict]:
    """The context term from paired short, long and far cells, or None."""
    context_delta = long_context + _LONG_SUFFIX_LEN - params.input_len
    if long_medians and context_delta <= 0:
        logger.warning(
            "DSpark SPS autotune: the long context (%d) is not longer than the "
            "prompts (%d); context term not fitted.",
            long_context,
            params.input_len,
        )
        return None
    if not set(long_medians) & set(medians):
        if long_medians:
            logger.warning(
                "DSpark SPS autotune: no long-context cell pairs with a short "
                "one; context term not fitted."
            )
        return None
    try:
        context_fit = fit_context_seconds(
            medians,
            long_medians,
            context_delta=context_delta,
            verify_width=verify_width,
            max_slots=max_slots,
        )
    except ValueError as exc:
        logger.warning("DSpark SPS autotune: context term not fitted (%s).", exc)
        return None
    context_fit["reference_context_tokens"] = reference_context_tokens(params)
    if set(far_medians) & set(long_medians) and long_context_far > long_context:
        try:
            context_fit["far"] = fit_context_seconds(
                long_medians,
                far_medians,
                context_delta=long_context_far - long_context,
                verify_width=verify_width,
                max_slots=max_slots,
            )
            context_fit["knee_tokens"] = long_context + _LONG_SUFFIX_LEN
        except ValueError as exc:
            logger.warning(
                "DSpark SPS autotune: far context term not fitted (%s).", exc
            )
    return context_fit


def _fit_table(
    measured: list[tuple[int, int, float]],
    context_fit: Optional[dict],
    *,
    verify_tiers: Sequence[int],
    verify_width: int,
    max_slots: int,
) -> tuple[SpsAdditiveCostTable, FitStats]:
    if context_fit is None:
        return fit_step_cost_table(measured)
    # The short cells ran at a small but nonzero context; take its cost out of
    # the baseline, which the planner prices from context zero.
    table, stats = fit_step_cost_table(
        without_context(
            measured,
            context_fit=context_fit,
            reference_context_tokens=context_fit["reference_context_tokens"],
            verify_width=verify_width,
            max_slots=max_slots,
        )
    )
    far = context_fit.get("far")
    table = msgspec.structs.replace(
        table,
        context_seconds=context_fit["context_seconds"],
        context_step_seconds=context_fit["context_step_seconds"],
        verify_tiers=sorted(int(t) for t in verify_tiers),
        context_knee_tokens=int(context_fit.get("knee_tokens", 0)),
        context_seconds_beyond=0.0 if far is None else far["context_seconds"],
        context_step_seconds_beyond=(
            0.0 if far is None else far["context_step_seconds"]
        ),
    )
    return table, stats


def _describe_context(context_fit: Optional[dict]) -> str:
    if context_fit is None:
        return "not fitted"
    text = (
        f"{context_fit['context_seconds'] * 1e9:.3f} ns per verify token and "
        f"{context_fit['context_step_seconds'] * 1e9:.2f} ns per step, per "
        "context token"
    )
    if "far" in context_fit:
        text += (
            f" up to {context_fit['knee_tokens']} tokens, "
            f"{context_fit['far']['context_seconds'] * 1e9:.3f} and "
            f"{context_fit['far']['context_step_seconds'] * 1e9:.2f} ns beyond"
        )
    return text


async def run_measurement(
    client: MeasureClient,
    status: dict,
    *,
    vocab_size: int,
    sleep: Callable[[float], Awaitable] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Optional[dict]:
    """Measure every cell of ``status`` over the configured passes, then the
    long-context cells, and fit. Returns the install request, or None when the
    measurement failed."""
    params = msgspec.convert(status["params"], AutotuneParams)
    cells = [tuple(c) for c in status["cells"]]
    long_cells = [tuple(c) for c in status.get("long_cells") or []]
    long_context = int(status.get("long_context") or 0) if long_cells else 0
    long_context_far = int(status.get("long_context_far") or 0) if long_context else 0
    width = int(status["verify_num_draft_tokens"])
    max_slots = int(status.get("max_slots") or 0)
    prompts = _measurement_prompts(
        client,
        status,
        params,
        longest_context=max(long_context, long_context_far),
        vocab_size=vocab_size,
    )
    started = clock()
    samples = await _collect_samples(
        client,
        status,
        params,
        prompts,
        cells=cells,
        long_cells=long_cells,
        long_context=long_context,
        long_context_far=long_context_far,
        started=started,
        sleep=sleep,
        clock=clock,
    )
    if samples.failed:
        logger.warning(
            "DSpark SPS autotune: %d cell measurements timed out: %s",
            len(samples.failed),
            sorted(set(samples.failed)),
        )
    covered = {cell for cell, values in samples.short.items() if values}
    lost = sorted(
        {("batch size", b) for b, _ in cells} - {("batch size", b) for b, _ in covered}
        | {("tier", t) for _, t in cells} - {("tier", t) for _, t in covered}
    )
    if lost:
        raise RuntimeError(f"no measurement left for {lost}")
    medians = _medians_ms(samples.short)
    long_medians = _medians_ms(samples.long)
    far_medians = _medians_ms(samples.far)
    context_fit = _fit_context(
        medians,
        long_medians,
        far_medians,
        params=params,
        long_context=long_context,
        long_context_far=long_context_far,
        verify_width=width,
        max_slots=max_slots,
    )
    table, stats = _fit_table(
        [(b, t, s) for (b, t), values in samples.short.items() for s in values],
        context_fit,
        verify_tiers=status["verify_tiers"],
        verify_width=width,
        max_slots=max_slots,
    )
    elapsed = clock() - started
    prompt_kind = "text" if prompts.text_ids else "random"
    fmt = lambda d: {f"{b},{t}": round(v, 4) for (b, t), v in sorted(d.items())}
    logger.info(
        "DSpark SPS autotune: measured %d cells x %d passes in %.0f s (%s "
        "prompts, temperature %.2f, top_p %.2f); fit rms %.3f ms, max residual "
        "%.3f ms; context %s; step ms by (bs, tier) %s; at %d-token context %s.",
        len(medians),
        params.passes,
        elapsed,
        prompt_kind,
        params.temperature,
        params.top_p,
        stats.rms_ms,
        stats.max_abs_ms,
        _describe_context(context_fit),
        fmt(medians),
        long_context,
        fmt(long_medians),
    )
    return {
        "key": status["key"],
        "table": msgspec.to_builtins(table),
        "metadata": {
            "created_unix": int(time.time()),
            "measure_seconds": round(elapsed, 1),
            "prompts": prompt_kind,
            "step_ms_by_bs_tier": fmt(medians),
            "long_context": long_context,
            "long_step_ms_by_bs_tier": fmt(long_medians),
            "long_context_far": long_context_far,
            "far_step_ms_by_bs_tier": fmt(far_medians),
            "context_fit": context_fit,
            "failed_cells": [list(c) for c in sorted(set(samples.failed))],
            "fit": msgspec.to_builtins(stats),
        },
    }


async def run_startup_autotune(
    tokenizer_manager,
    *,
    sleep: Callable[[float], Awaitable] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
    rpc_timeout: float = 60.0,
) -> None:
    """Server half of ``--speculative-dspark-sps-autotune``: measure and install
    the table if the schedulers report a pending measurement.

    A failed measurement keeps the configured table. Every outcome ends with
    one table request (the table, or None to abandon) that unpins the budget
    and stops recording on every rank; if the schedulers do not confirm it,
    this raises so the server does not start with a pinned verify budget.
    """
    from sglang.srt.runtime_context import get_spec

    spec = get_spec()
    mode = SpsAutotuneMode.parse(spec.speculative_dspark_sps_autotune)
    if (
        mode is SpsAutotuneMode.OFF
        or "DSPARK" not in str(spec.speculative_algorithm).upper()
    ):
        return
    client = MeasureClient(tokenizer_manager, rpc_timeout=rpc_timeout)
    try:
        pending = [s for s in await client.status() if s.get("pending")]
    except Exception:
        logger.exception("DSpark SPS autotune could not read the scheduler state.")
        return
    if len(pending) != 1:
        if pending:
            logger.warning(
                "DSpark SPS autotune: %d schedulers report a pending measurement; "
                "skipped.",
                len(pending),
            )
        return
    status = pending[0]
    request = None
    try:
        vocab_size = int(tokenizer_manager.model_config.vocab_size)
        params = status["params"]
        logger.info(
            "DSpark SPS autotune: measuring %d (batch size, tier) cells before "
            "the server is ready (at most %.0f s).",
            len(status["cells"]),
            params["max_seconds"],
        )
        request = await asyncio.wait_for(
            run_measurement(
                client, status, vocab_size=vocab_size, sleep=sleep, clock=clock
            ),
            timeout=float(params["max_seconds"]) + 4 * rpc_timeout,
        )
    except Exception:
        logger.exception(
            "DSpark SPS autotune measurement failed; keeping the configured table."
        )
    finally:
        # Runs on cancellation too, and cannot itself be cancelled half-way.
        await asyncio.shield(_end_measurement(client, request))


async def _end_measurement(client: MeasureClient, request: Optional[dict]) -> None:
    """Send the terminal table request and confirm no measurement is pending."""
    payload = None if request is None else json.dumps(request)
    try:
        await client.set_state(**{TABLE_KEY: payload})
        states = await client.status()
    except Exception as exc:
        raise RuntimeError(
            "DSpark SPS autotune could not end its measurement on the schedulers"
        ) from exc
    if any(s.get("pending") for s in states):
        raise RuntimeError("DSpark SPS autotune measurement is still pending")
    if request is not None and not any(s.get("installed") for s in states):
        logger.warning("DSpark SPS autotune: the schedulers did not install it.")
