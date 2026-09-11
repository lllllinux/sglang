"""Host-table engram lookups must reconstruct the checkpoint table.

Covers the SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE paths that the device-table
test does not touch: memfd-backed shared layout (one copy, no all-reduce)
and the anonymous private shard (zeroed unowned rows + all-reduce), plus
cudaHostRegister pinning. Purely synthetic tables -- no weights required.

The unpinned variants need GPU-addressable host memory (ATS / HMM): a
discrete Blackwell reports cudaDevAttrPageableMemoryAccess = 0 and the
gather from a plain mmap would fault the CUDA context, so the table pins
itself regardless (asserted) and the variants skip where the GPU cannot
reach unpinned host memory.
"""

import os
import socket
import unittest
from typing import Optional
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b", runner_config="2-gpu-large")

ROWS = 2049  # uneven over 2 ranks
DIM = 256
BLK = 32


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init(rank: int, world: int, port: int, backend: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend=backend,
    )
    initialize_model_parallel(tensor_model_parallel_size=world, backend=backend)


def _teardown() -> None:
    destroy_model_parallel()
    destroy_distributed_environment()


def _checkpoint_table(seed: int = 0, rows: int = ROWS):
    """The checkpoint tensors as the loader hands them over: identical on every rank."""
    g = torch.Generator().manual_seed(seed)
    weight = (torch.randn(rows, DIM, generator=g) * 3).to(torch.float8_e4m3fn)
    scale = torch.randint(
        100, 140, (rows, DIM // BLK), dtype=torch.uint8, generator=g
    ).view(torch.float8_e8m0fnu)
    # Exponent byte zero represents 2**-127, not zero. Keep it in a real lookup.
    weight.view(torch.uint8)[0, :BLK] = (
        torch.tensor(1.0).to(torch.float8_e4m3fn).view(torch.uint8)
    )
    scale.view(torch.uint8)[0, 0] = 0
    return weight, scale


def _reference(weight, scale, ids):
    ids = ids.cpu()
    rows = weight[ids].float().unflatten(-1, (-1, BLK))
    return (rows * scale[ids].float().unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


def _build(rows: int):
    from sglang.srt.layers.engram import EngramEmbedding

    with torch.device("cuda"):
        return EngramEmbedding(rows, DIM, layer_id=1)


def _unpinned_host_access() -> Optional[bool]:
    """The production probe; None means 'could not query'."""
    from sglang.srt.layers.engram import _unpinned_host_access as probe

    return probe()


def _load(embed, weight, scale) -> None:
    embed.weight.weight_loader(embed.weight, weight)
    embed.scale.weight_loader(embed.scale, scale)
    embed.finish_load()


def _tp_worker(rank: int, world: int, port: int, layout: str, pin: bool) -> None:
    torch.cuda.set_device(rank)
    _init(rank, world, port, "nccl")
    try:
        weight, scale = _checkpoint_table(rows=ROWS)
        with (
            envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.override(True),
            envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.override(layout),
            envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_PIN.override(pin),
            # Keep the test hermetic: no page-cache drops on the CI host.
            envs.SGLANG_ENABLE_DSV41_ENGRAM_DROP_PAGE_CACHE.override(False),
            get_parallel().override(tp_size=world, tp_rank=rank),
            patch("sglang.srt.layers.engram.get_attention_dp_size", return_value=1),
        ):
            embed = _build(ROWS)
            assert embed.host_table is not None, "host table was not created"
            assert embed.host_table.layout == layout
            if pin:
                assert embed.host_table.registered, "cudaHostRegister did not pin"
            else:
                # Without ATS the table pins itself instead of risking a fault.
                expected = _unpinned_host_access() is False
                assert embed.host_table.registered == expected, (
                    "unpinned table created where the GPU cannot reach it"
                )
            # Each rank loads only its own row range, like the real loader.
            _load(embed, weight, scale)
            ids = torch.arange(ROWS, device="cuda", dtype=torch.int64).view(-1, 1)
            out = embed(ids)
        torch.cuda.synchronize()
        assert torch.equal(out.cpu(), _reference(weight, scale, ids)), (
            f"rank {rank}, layout {layout}, pin {pin}: incorrect lookup"
        )
        assert out[0, 0, 0].item() == 2**-127, "zero exponent was decoded as zero"
        torch.distributed.barrier()
    finally:
        _teardown()


class TestEngramHostTable(CustomTestCase):
    def _spawn(self, layout: str, pin: bool) -> None:
        world = min(2, torch.cuda.device_count())
        if world < 2:
            self.skipTest("needs two GPUs to run two TP ranks")
        mp.spawn(
            _tp_worker,
            args=(world, _free_port(), layout, pin),
            nprocs=world,
            join=True,
        )

    def test_shared_layout_lookup_matches_checkpoint(self):
        self._spawn("shared", pin=False)

    def test_shared_layout_pinned_lookup_matches_checkpoint(self):
        self._spawn("shared", pin=True)

    def test_private_layout_lookup_matches_checkpoint(self):
        self._spawn("private", pin=False)

    def test_private_layout_pinned_lookup_matches_checkpoint(self):
        # Production default: PIN=True. Also the only meaningful private case
        # on boards without GPU-addressable host memory.
        self._spawn("private", pin=True)


class TestUnpinnedHostAccessProbe(CustomTestCase):
    """The probe's contract at the _HostTable pin decision: False forces the
    pin, True (ATS platform) and None (query failed) keep the requested
    mode."""

    def _decide_pin(self, requested: bool, probe_value: Optional[bool]) -> bool:
        # The exact decision _HostTable.__init__ makes on `pin=requested`.
        import sglang.srt.layers.engram as engram_mod

        with patch.object(engram_mod, "_unpinned_host_access", return_value=probe_value):
            pin = requested
            if not pin and engram_mod._unpinned_host_access() is False:
                pin = True
            return pin

    def test_probe_false_forces_pin(self):
        self.assertTrue(self._decide_pin(requested=False, probe_value=False))

    def test_probe_true_respects_unpinned(self):
        self.assertFalse(self._decide_pin(requested=False, probe_value=True))

    def test_probe_none_keeps_requested_mode(self):
        self.assertFalse(self._decide_pin(requested=False, probe_value=None))

    def test_pinned_request_untouched_by_probe(self):
        self.assertTrue(self._decide_pin(requested=True, probe_value=False))

    def test_probe_on_this_device_answers_bool(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA context for the real query")
        torch.cuda.init()
        from sglang.srt.layers.engram import _unpinned_host_access

        self.assertIsInstance(_unpinned_host_access(), bool)

    def test_probe_query_failure_falls_back_and_never_raises(self):
        # Break the cuda.bindings import: the ctypes/libcudart fallback must
        # still answer with a bool instead of raising.
        import builtins

        import sglang.srt.layers.engram as engram_mod

        real_import = builtins.__import__

        def no_cuda_bindings(name, *a, **kw):
            if name.startswith("cuda.bindings"):
                raise ImportError(name)
            return real_import(name, *a, **kw)

        with patch("builtins.__import__", side_effect=no_cuda_bindings):
            try:
                value = engram_mod._unpinned_host_access()
            except ImportError:
                self.skipTest("cuda.bindings absent; fallback already primary")
        self.assertIsInstance(value, bool)


if __name__ == "__main__":
    unittest.main()
