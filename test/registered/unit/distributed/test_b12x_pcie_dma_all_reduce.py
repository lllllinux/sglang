from types import SimpleNamespace

import torch

from sglang.srt.distributed.device_communicators import (
    b12x_pcie_dma_all_reduce as adapter,
)


def _cuda_graph_config(*, decode: str, prefill: str):
    return SimpleNamespace(
        decode=SimpleNamespace(backend=decode),
        prefill=SimpleNamespace(backend=prefill),
    )


def test_required_execution_modes_include_graph_only_when_enabled() -> None:
    assert adapter._required_execution_modes(
        _cuda_graph_config(decode="disabled", prefill="disabled")
    ) == ("eager",)
    assert adapter._required_execution_modes(
        _cuda_graph_config(decode="full", prefill="disabled")
    ) == ("eager", "graph")
    assert adapter._required_execution_modes(
        _cuda_graph_config(decode="disabled", prefill="breakable")
    ) == ("eager", "graph")


def test_cache_key_separates_execution_mode_and_nccl_env(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_b12x_version", lambda: "test")
    monkeypatch.setattr(adapter.torch.cuda.nccl, "version", lambda: (2, 28, 3))
    monkeypatch.setenv("NCCL_PROTO", "Simple")
    monkeypatch.setenv("B12X_PCIE_DMA_PIECES", "2")
    common = dict(
        world_size=4,
        wire_mode="bf16",
        max_bytes=64 * 1024 * 1024,
        gpus=[{"gpu_name": "RTX PRO 6000", "pci_bus_id": "0000:01:00.0"}],
    )

    eager = adapter._cache_key(**common, execution_mode="eager")
    graph = adapter._cache_key(**common, execution_mode="graph")

    assert eager["execution_mode"] == "eager"
    assert graph["execution_mode"] == "graph"
    assert eager != graph
    assert eager["comm_env"]["NCCL_PROTO"] == "Simple"
    assert eager["comm_env"]["B12X_PCIE_DMA_PIECES"] == "2"


def test_dispatch_uses_mode_specific_threshold_without_mutating_dma(monkeypatch) -> None:
    class FakeDma:
        def __init__(self):
            self.min_bytes = 123
            self.thresholds = []

        def should_allreduce(self, _input, *, min_bytes):
            self.thresholds.append(min_bytes)
            return True

        def all_reduce(self, input_, *, min_bytes):
            assert min_bytes == self.thresholds[-1]
            return input_

    dma = FakeDma()
    comm = adapter.B12xPcieDmaAllReduce(
        dma, {"eager": 16 * 1024 * 1024, "graph": 4 * 1024 * 1024}
    )
    input_ = object()

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert comm.try_all_reduce(input_) is input_
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert comm.try_all_reduce(input_) is input_

    assert dma.thresholds == [16 * 1024 * 1024, 4 * 1024 * 1024]
    assert dma.min_bytes == 123


def test_dispatch_falls_through_when_current_mode_was_not_tuned(monkeypatch) -> None:
    class FakeDma:
        def should_allreduce(self, *_args, **_kwargs):
            raise AssertionError("untuned graph mode must not call b12x")

    comm = adapter.B12xPcieDmaAllReduce(FakeDma(), {"eager": 16 * 1024 * 1024})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    assert comm.try_all_reduce(object()) is None
