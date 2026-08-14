# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""b12x CE-DMA all-reduce adapter for PCIe-only SM120/SM121.

Default off. When enabled, ``GroupCoordinator`` constructs one communicator
per TP group, resolves eager/graph ``min_bytes`` from env, cache, or b12x
autotune, and dispatches after FlashInfer PCIe-IPC.
"""

from __future__ import annotations

import json
import logging
import os
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_SUPPORTED_WORLD_SIZES = (2, 4, 6, 8, 10)
_AUTOTUNE_HIDDEN = 4096
_AUTOTUNE_ELEM_SIZE = 2
_EXECUTION_MODES = ("eager", "graph")
ExecutionMode = Literal["eager", "graph"]
_OFF_VALUES = frozenset({"off", "disabled", "none"})
_BYTE_SUFFIXES = (
    ("kib", 1024),
    ("mib", 1024**2),
    ("gib", 1024**3),
    ("kb", 1000),
    ("mb", 1000**2),
    ("gb", 1000**3),
    ("k", 1024),
    ("m", 1024**2),
    ("g", 1024**3),
    ("b", 1),
)


def _parse_byte_size(value: str) -> int:
    raw = value.strip().lower().replace("_", "").replace(" ", "")
    if not raw:
        raise ValueError("empty byte size")
    for suffix, mul in _BYTE_SUFFIXES:
        if raw.endswith(suffix):
            number = raw[: -len(suffix)]
            parsed = int(float(number) * mul)
            if parsed < 0:
                raise ValueError(f"negative byte size: {value!r}")
            return parsed
    parsed = int(raw)
    if parsed < 0:
        raise ValueError(f"negative byte size: {value!r}")
    return parsed


def _cache_path() -> Path:
    configured = (envs.SGLANG_B12X_PCIE_DMA_CACHE.get() or "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(envs.SGLANG_CACHE_DIR.get()).expanduser() / "b12x_pcie_dma.json"


def _b12x_version() -> str:
    try:
        return metadata.version("b12x")
    except metadata.PackageNotFoundError:
        return "unknown"


def _pci_bus_id(device: torch.device) -> str:
    index = device.index if device.index is not None else torch.cuda.current_device()
    try:
        props = torch.cuda.get_device_properties(index)
        for attr in ("pci_bus_id", "uuid"):
            value = getattr(props, attr, None)
            if value:
                return str(value)
        return f"{props.name}:{index}"
    except Exception:
        return f"cuda:{index}"


def _local_gpu_record(device: torch.device) -> dict[str, str]:
    return {
        "gpu_name": torch.cuda.get_device_name(device),
        "pci_bus_id": _pci_bus_id(device),
    }


def _gather_gpu_inventory(
    group: ProcessGroup, device: torch.device
) -> list[dict[str, str]]:
    record = _local_gpu_record(device)
    gathered: list[Optional[dict[str, str]]] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, record, group=group)
    return [item if item is not None else {} for item in gathered]


def _cache_key(
    *,
    world_size: int,
    wire_mode: str,
    max_bytes: int,
    gpus: list[dict[str, str]],
    execution_mode: ExecutionMode,
) -> dict[str, Any]:
    try:
        nccl_version = str(torch.cuda.nccl.version())
    except Exception:
        nccl_version = "unknown"
    return {
        "world_size": world_size,
        "dtype": "bfloat16",
        "wire_mode": wire_mode,
        "max_bytes": max_bytes,
        "hidden_size": _AUTOTUNE_HIDDEN,
        "gpus": gpus,
        "b12x_version": _b12x_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "nccl_version": nccl_version,
        "comm_env": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith(("NCCL_", "B12X_PCIE_DMA_"))
        },
        "execution_mode": execution_mode,
    }


def _load_cached_min_bytes(key: dict[str, Any]) -> Optional[int]:
    path = _cache_path()
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("entries", [])
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("b12x PCIe DMA cache unreadable (%s); ignoring", exc)
        return None
    for entry in entries:
        min_bytes = entry.get("dma_min_bytes")
        cached_key = {k: entry.get(k) for k in key}
        if cached_key == key and isinstance(min_bytes, int):
            return min_bytes
    return None


def _store_cached_min_bytes(key: dict[str, Any], dma_min_bytes: int) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload.get("entries"), list):
                entries = payload["entries"]
        except (OSError, json.JSONDecodeError):
            entries = []
    kept = []
    for entry in entries:
        cached_key = {k: entry.get(k) for k in key}
        if cached_key != key:
            kept.append(entry)
    kept.append({**key, "dma_min_bytes": dma_min_bytes})
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"entries": kept}, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _pinned_min_bytes() -> tuple[str, Optional[int]]:
    raw = envs.SGLANG_B12X_PCIE_DMA_MIN_BYTES.get()
    if raw is None:
        return "auto", None
    text = str(raw).strip()
    if not text:
        return "auto", None
    if text.lower() in _OFF_VALUES:
        return "off", None
    try:
        return "pin", _parse_byte_size(text)
    except ValueError as exc:
        logger.warning(
            "invalid SGLANG_B12X_PCIE_DMA_MIN_BYTES=%r (%s); using autotune",
            raw,
            exc,
        )
        return "auto", None


class B12xPcieDmaAllReduce:
    """Thin wrapper around ``b12x.comm.pcie.DmaAllReduce``."""

    def __init__(self, dma: Any, min_bytes: dict[ExecutionMode, int]) -> None:
        self._dma = dma
        self._min_bytes = min_bytes
        self._disabled = False

    def try_all_reduce(self, input_: torch.Tensor) -> Optional[torch.Tensor]:
        if self._disabled or self._dma is None:
            return None
        try:
            execution_mode: ExecutionMode = (
                "graph" if torch.cuda.is_current_stream_capturing() else "eager"
            )
            min_bytes = self._min_bytes.get(execution_mode)
            if min_bytes is None:
                return None
            if not self._dma.should_allreduce(input_, min_bytes=min_bytes):
                return None
            return self._dma.all_reduce(input_, min_bytes=min_bytes)
        except Exception as exc:  # noqa: BLE001 - must never be fatal
            logger.warning(
                "b12x PCIe DMA all-reduce failed (%s); disabling for this group",
                exc,
            )
            self._disabled = True
            return None

    def close(self) -> None:
        dma = self._dma
        self._dma = None
        self._disabled = True
        if dma is not None:
            dma.close()


def _autotune_min_bytes(
    dma: Any, device_group: ProcessGroup, execution_mode: ExecutionMode
) -> int:
    from b12x.comm.pcie import autotune_dma_crossovers

    max_rows = max(1, dma.max_bytes // (_AUTOTUNE_HIDDEN * _AUTOTUNE_ELEM_SIZE))
    _oneshot_max, _dma_min = autotune_dma_crossovers(
        None,
        dma,
        device_group,
        hidden_size=_AUTOTUNE_HIDDEN,
        max_rows=max_rows,
        execution_mode=execution_mode,
    )
    return int(dma.min_bytes)


def _resolve_min_bytes(
    dma: Any,
    *,
    device_group: ProcessGroup,
    cpu_group: ProcessGroup,
    device: torch.device,
    policy: str,
    pinned: Optional[int],
    execution_mode: ExecutionMode,
    gpus: list[dict[str, str]],
) -> tuple[int, str]:
    if policy == "pin":
        assert pinned is not None
        return pinned, "env"

    key = _cache_key(
        world_size=dma.world_size,
        wire_mode=dma.wire_mode,
        max_bytes=dma.max_bytes,
        gpus=gpus,
        execution_mode=execution_mode,
    )
    use_cache = not envs.SGLANG_B12X_PCIE_DMA_FORCE_AUTOTUNE.get()
    cached: Optional[int] = None
    if use_cache and dist.get_rank(cpu_group) == 0:
        cached = _load_cached_min_bytes(key)
    payload = torch.tensor(
        [-1 if cached is None else cached],
        dtype=torch.int64,
        device="cpu",
    )
    src_rank = dist.get_process_group_ranks(cpu_group)[0]
    dist.broadcast(payload, src=src_rank, group=cpu_group)
    cached_value = int(payload.item())
    if use_cache and cached_value >= 0:
        return cached_value, "cache"

    min_bytes = _autotune_min_bytes(dma, device_group, execution_mode)
    if dist.get_rank(cpu_group) == 0:
        _store_cached_min_bytes(key, min_bytes)
    return min_bytes, "autotune"


def _required_execution_modes(cuda_graph_config: Any = None) -> tuple[ExecutionMode, ...]:
    if cuda_graph_config is None:
        try:
            from sglang.srt.runtime_context import get_exec

            cuda_graph_config = get_exec().graph.cuda_graph_config
        except Exception:
            return _EXECUTION_MODES

    graph_enabled = any(
        getattr(getattr(cuda_graph_config, phase), "backend", "disabled")
        != "disabled"
        for phase in ("decode", "prefill")
    )
    return _EXECUTION_MODES if graph_enabled else ("eager",)


def create_b12x_pcie_dma_all_reduce(
    *,
    device_group: ProcessGroup,
    cpu_group: ProcessGroup,
    device: torch.device,
) -> Optional[B12xPcieDmaAllReduce]:
    """Construct a DMA communicator or return None (fail closed)."""
    if not envs.SGLANG_OPT_USE_B12X_PCIE_DMA.get():
        return None
    policy, pinned = _pinned_min_bytes()
    if policy == "off":
        if dist.get_rank(device_group) == 0:
            logger.info(
                "b12x PCIe DMA disabled by SGLANG_B12X_PCIE_DMA_MIN_BYTES=off"
            )
        return None

    dma_cls = None
    skip_reason = None
    try:
        from b12x.comm.pcie import DmaAllReduce, is_supported

        dma_cls = DmaAllReduce
        if not is_supported(device):
            skip_reason = "not SM120/SM121"
        elif dist.get_world_size(device_group) not in _SUPPORTED_WORLD_SIZES:
            skip_reason = (
                f"world_size={dist.get_world_size(device_group)} "
                f"not in {_SUPPORTED_WORLD_SIZES}"
            )
    except Exception as exc:  # noqa: BLE001 - optional dependency
        skip_reason = f"import failed: {exc}"

    if _group_failed(device_group, device, skip_reason is not None):
        if dist.get_rank(device_group) == 0:
            logger.info("b12x PCIe DMA skipped: %s", skip_reason)
        return None

    assert dma_cls is not None
    max_bytes = int(envs.SGLANG_B12X_PCIE_DMA_MAX_BYTES.get())
    fp8 = envs.SGLANG_B12X_PCIE_DMA_FP8.get() or ""
    dma = None
    error: Optional[BaseException] = None
    try:
        dma = dma_cls(
            exchange_group=device_group,
            device=device,
            max_bytes=max_bytes,
            fp8=fp8,
        )
    except Exception as exc:  # noqa: BLE001 - fail closed
        error = exc

    if _group_failed(device_group, device, error is not None):
        if dma is not None:
            dma.close()
        logger.warning(
            "b12x PCIe DMA construction failed (rank %d error: %s); "
            "falling back to the default all-reduce path",
            dist.get_rank(device_group),
            error,
        )
        return None

    assert dma is not None
    min_bytes_by_mode: dict[ExecutionMode, int] = {}
    source_by_mode: dict[ExecutionMode, str] = {}
    modes = _EXECUTION_MODES if policy == "pin" else _required_execution_modes()
    gpus = _gather_gpu_inventory(cpu_group, device) if policy == "auto" else []
    for execution_mode in modes:
        mode_error: Optional[BaseException] = None
        try:
            min_bytes, source = _resolve_min_bytes(
                dma,
                device_group=device_group,
                cpu_group=cpu_group,
                device=device,
                policy=policy,
                pinned=pinned,
                execution_mode=execution_mode,
                gpus=gpus,
            )
            min_bytes_by_mode[execution_mode] = min_bytes
            source_by_mode[execution_mode] = source
        except Exception as exc:  # noqa: BLE001 - fail closed
            mode_error = exc
        if _group_failed(device_group, device, mode_error is not None):
            error = mode_error or RuntimeError(
                f"peer rank failed {execution_mode} b12x PCIe DMA autotune"
            )
            break

    if error is not None:
        dma.close()
        logger.warning(
            "b12x PCIe DMA autotune failed (rank %d error: %s); "
            "falling back to the default all-reduce path",
            dist.get_rank(device_group),
            error,
        )
        return None

    if dist.get_rank(device_group) == 0:
        logger.info(
            "b12x PCIe DMA all-reduce ready: min_bytes=%s max_bytes=%d "
            "wire=%s source=%s",
            min_bytes_by_mode,
            dma.max_bytes,
            dma.wire_mode,
            source_by_mode,
        )
    return B12xPcieDmaAllReduce(dma, min_bytes_by_mode)


def _group_failed(
    group: ProcessGroup, device: torch.device, local_failed: bool
) -> bool:
    flag = torch.tensor([int(local_failed)], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
    return int(flag.item()) != 0


def _run_cli() -> None:
    import os

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("b12x").setLevel(logging.DEBUG)
    logging.getLogger("b12x.comm.pcie").setLevel(logging.DEBUG)

    if not torch.cuda.is_available():
        raise SystemExit("b12x PCIe DMA CLI requires CUDA")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    cpu_group = dist.new_group(backend="gloo")

    with (
        envs.SGLANG_OPT_USE_B12X_PCIE_DMA.override(True),
        envs.SGLANG_B12X_PCIE_DMA_FORCE_AUTOTUNE.override(True),
    ):
        comm = create_b12x_pcie_dma_all_reduce(
            device_group=dist.group.WORLD,
            cpu_group=cpu_group,
            device=device,
        )

    if comm is None or comm._dma is None:
        raise SystemExit("b12x PCIe DMA communicator was not created")

    dma = comm._dma
    if dist.get_rank() == 0:
        print(f"dma_min_bytes={comm._min_bytes}")
        print(f"dma_max_bytes={dma.max_bytes}")
        print(f"wire_mode={dma.wire_mode}")
        print(f"cache={_cache_path()}")
        if all(value > dma.max_bytes for value in comm._min_bytes.values()):
            print("export SGLANG_B12X_PCIE_DMA_MIN_BYTES=off")
        else:
            print("export SGLANG_OPT_USE_B12X_PCIE_DMA=1")
    comm.close()
    dist.destroy_process_group(cpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    _run_cli()
