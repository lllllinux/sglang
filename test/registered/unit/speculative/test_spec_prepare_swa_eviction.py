"""dflash-family decode bookkeeping ownership: exactly one eviction + tick.

The dflash-family dispatcher branch (DFLASH, DSPARK) routes decode
preparation to DFlashDraftInputV2.prepare_for_decode (upstream #33676),
which owns both ``maybe_evict_swa`` and the ``decode_batch_idx`` tick --
mirroring eagle_prepare_for_decode. The dispatcher must NOT repeat either:
a second eviction in the same iteration would see ``decode_batch_idx >= 1``
and bypass the overlap first-round guard in maybe_evict_swa, freeing SWA
slots while the previous extend batch may still be running (pool sizing
assumes the eviction runs -- pool_configurator: trailing_tokens =
sliding_window + eviction_interval * draft_tokens + page_size).

These tests drive the REAL DFlashDraftInputV2.prepare_for_decode and the
REAL maybe_evict_swa; only allocator/tensor-buffer externals are stubbed.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_batch(spec_is_dflash_family: bool, spec_is_uno: bool = False):
    batch = MagicMock()
    batch.spec_algorithm.is_dflash_family.return_value = spec_is_dflash_family
    batch.spec_algorithm.is_uno.return_value = spec_is_uno
    req = SimpleNamespace(decode_batch_idx=0)
    batch.reqs = [req]
    return batch, req


def _make_real_dflash_batch():
    """A batch that runs the REAL ScheduleBatch.maybe_evict_swa and the REAL
    DFlashDraftInputV2.prepare_for_decode, with only allocator externals
    (alloc_for_spec_decode) stubbed. A wrapper around the real _evict_swa
    records the decode_batch_idx each eviction observed."""
    from sglang.srt.managers.schedule_batch import ReqKvInfo
    from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2

    batch = MagicMock()
    batch.spec_algorithm.is_dflash_family.return_value = True
    batch.spec_algorithm.is_uno.return_value = False
    batch.batch_size.return_value = 1
    batch.device = torch.device("cpu")
    # Route to the real eviction so the ownership under test is exercised.
    batch.maybe_evict_swa.side_effect = lambda: ScheduleBatch.maybe_evict_swa(
        batch
    )
    # The real maybe_evict_swa runs with a MagicMock self, so its
    # self._evict_swa resolves as a mock attribute -- patching the
    # ScheduleBatch class never reaches it. Wire the real eviction here.
    batch._evict_swa = (
        lambda req_, pre_len: ScheduleBatch._evict_swa(batch, req_, pre_len)
    )
    batch.tree_cache.supports_swa.return_value = True
    batch.tree_cache.sliding_window_size = 4
    batch.tree_cache.dec_swa_lock_only = MagicMock()
    batch.tree_cache.is_chunk_cache.return_value = False
    batch.tree_cache.swa_retain_floor = MagicMock(return_value=None)
    batch.req_to_token_pool = MagicMock()
    batch.req_to_token_pool.req_to_token = torch.zeros((1, 256), dtype=torch.int64)
    batch.forward_mode.is_decode.return_value = True
    batch.token_to_kv_pool_allocator.page_size = 1
    batch.tree_cache.page_size = 1
    batch.token_to_kv_pool_allocator.free_group_begin = MagicMock()
    batch.token_to_kv_pool_allocator.free_group_end = MagicMock()

    # Real ReqKvInfo so free_swa_out_of_window_slots runs for real: holds_kv
    # and swa_dead_lo derive from req_pool_idx; the eviction window is
    # [swa_dead_lo, seqlen-1-window) beyond swa_evicted_seqlen.
    kv = ReqKvInfo()
    kv.req_pool_idx = 0
    kv.cache_protected_len = 0
    kv.kv_committed_len = 32
    kv.kv_allocated_len = 32
    kv.swa_evict_floor = 0
    kv.swa_evicted_seqlen = 0

    req = SimpleNamespace(
        decode_batch_idx=0,
        # seqlen far enough past the window that seqlen-1-window (199) clears
        # swa_evicted_seqlen + SGLANG_SWA_EVICTION_INTERVAL (128) once the
        # decode_batch_idx >= 1 guard opens on the second iteration.
        seqlen=204,
        kv=kv,
        swa_prefix_lock_released=False,
        swa_uuid_for_lock=None,
        last_node=None,
        skip_lock_node_ids=None,
        # Read by the real DFlashDraftInputV2.prepare_for_decode:
        sampling_params=SimpleNamespace(top_k=1),
    )
    batch.reqs = [req]
    batch.req_pool_indices = torch.zeros((1,), dtype=torch.int64)
    batch.seq_lens = torch.tensor([32], dtype=torch.int64)

    batch.spec_info = DFlashDraftInputV2(
        topk_p=torch.empty((1, 1)),
        topk_index=torch.empty((1, 1), dtype=torch.int64),
        bonus_tokens=torch.empty((1,), dtype=torch.int64),
        new_seq_lens=torch.empty((1,), dtype=torch.int64),
        hidden_states=torch.empty((1, 1)),
    )
    return batch, req


class TestSpecPrepareSwaEviction(CustomTestCase):
    def setUp(self):
        # The real DFlashDraftInputV2.prepare_for_decode reads the spec bag
        # (speculative_num_draft_tokens); publish a minimal ServerArgs with
        # DSPARK configured, and restore the prior context afterwards.
        from sglang.srt.runtime_context import (
            get_context,
            reset_context,
        )
        from sglang.srt.server_args import ServerArgs

        self._reset_context = reset_context
        self._saved_server_args = get_context()._server_args
        get_context().set_server_args(
            ServerArgs(
                model_path="dummy",
                speculative_algorithm="DSPARK",
                speculative_num_draft_tokens=4,
            )
        )

    def tearDown(self):
        if self._saved_server_args is None:
            self._reset_context()
        else:
            from sglang.srt.runtime_context import get_context

            get_context().set_server_args(self._saved_server_args)

    def _run(self, batch):
        from sglang.srt.speculative import spec_utils

        with patch.object(
            spec_utils, "mamba_extra_buffer_lazy_enabled", return_value=False
        ):
            spec_utils.spec_prepare_for_decode(batch)

    def test_dflash_family_evicts_and_ticks(self):
        """The real chain ticks exactly once per iteration, and the eviction
        observes the PRE-tick clock. The first decode iteration is guarded
        (decode_batch_idx >= 1, the overlap first-round guard) so it evicts
        nothing; the second iteration's single eviction sees idx 1. If the
        dispatcher ever duplicates the bookkeeping again, the first iteration
        pre-ticks to 1 and evicts twice there, failing the assertions."""
        batch, req = _make_real_dflash_batch()
        seen_idx = []
        real_evict = ScheduleBatch._evict_swa

        def _record(req_, pre_len):
            seen_idx.append(req_.decode_batch_idx)
            return real_evict(batch, req_, pre_len)

        batch._evict_swa = _record
        with (
            patch("sglang.srt.mem_cache.allocation.alloc_for_spec_decode"),
            # The real draft input imports it by name; patch both bindings.
            patch("sglang.srt.speculative.dflash_info_v2.alloc_for_spec_decode"),
            patch(
                "sglang.srt.speculative.dflash_info_v2._get_overlap_plan_stream",
                return_value=(None, contextlib.nullcontext()),
            ),
        ):
            self._run(batch)
            self._run(batch)
        self.assertEqual(req.decode_batch_idx, 2)
        # One eviction in two iterations, at the second's pre-tick clock.
        self.assertEqual(seen_idx, [1])
        # The real eviction did run (swa_evicted_seqlen advanced).
        self.assertGreater(req.kv.swa_evicted_seqlen, 0)

    def test_tick_advances_every_iteration(self):
        """decode_batch_idx is a clock, not a flag: it must keep advancing so
        the SWA leaf-lock release gate (decode_batch_idx >= sliding_window_size)
        can fire -- but once per iteration, not twice. Two iterations through
        the real chain: two evictions observed, each at the pre-tick clock."""
        batch, req = _make_real_dflash_batch()
        seen_idx = []
        real_evict = ScheduleBatch.maybe_evict_swa

        def _record():
            seen_idx.append(req.decode_batch_idx)
            return real_evict(batch)

        batch.maybe_evict_swa.side_effect = _record
        with (
            patch("sglang.srt.mem_cache.allocation.alloc_for_spec_decode"),
            # The real draft input imports it by name; patch both bindings.
            patch("sglang.srt.speculative.dflash_info_v2.alloc_for_spec_decode"),
            patch(
                "sglang.srt.speculative.dflash_info_v2._get_overlap_plan_stream",
                return_value=(None, contextlib.nullcontext()),
            ),
        ):
            self._run(batch)
            self._run(batch)
        self.assertEqual(req.decode_batch_idx, 2)
        # maybe_evict_swa ran every iteration, each observing the pre-tick clock.
        self.assertEqual(seen_idx, [0, 1])

    def test_dflash_family_evicts_before_tick(self):
        """The overlap-scheduler gate (decode_batch_idx >= 1) must see the
        pre-tick value, exactly as in eagle_prepare_for_decode: at the second
        iteration the eviction observes 1, before the tick lands."""
        batch, req = _make_real_dflash_batch()
        seen_idx = []
        real_evict = ScheduleBatch._evict_swa

        def _record(req_, pre_len):
            seen_idx.append(req_.decode_batch_idx)
            return real_evict(batch, req_, pre_len)

        batch._evict_swa = _record
        with (
            patch("sglang.srt.mem_cache.allocation.alloc_for_spec_decode"),
            # The real draft input imports it by name; patch both bindings.
            patch("sglang.srt.speculative.dflash_info_v2.alloc_for_spec_decode"),
            patch(
                "sglang.srt.speculative.dflash_info_v2._get_overlap_plan_stream",
                return_value=(None, contextlib.nullcontext()),
            ),
        ):
            self._run(batch)
            self._run(batch)
        # The second iteration's eviction ran at the pre-tick clock 1.
        self.assertEqual(seen_idx, [1])
        self.assertEqual(req.decode_batch_idx, 2)

    def test_eagle_path_unchanged(self):
        # Since #37667 the dispatcher routes uno through its own draft-input
        # prep; a non-dflash, non-uno algorithm must still hit the eagle
        # helper. Pin is_uno=False explicitly -- a bare MagicMock would
        # truthy-match the uno branch and never reach the eagle path.
        batch, req = _make_batch(spec_is_dflash_family=False, spec_is_uno=False)
        with patch(
            "sglang.srt.speculative.eagle_utils.eagle_prepare_for_decode"
        ) as eagle_prep:
            self._run(batch)
        eagle_prep.assert_called_once_with(batch)
        # The dflash-branch tick must not run on the eagle path.
        self.assertEqual(req.decode_batch_idx, 0)
        batch.spec_info.prepare_for_decode.assert_not_called()

    def test_uno_path_routes_to_uno_draft_input(self):
        from sglang.srt.speculative.uno_info import UnoDraftInput

        batch, req = _make_batch(spec_is_dflash_family=False, spec_is_uno=True)
        batch.spec_info = MagicMock(spec=UnoDraftInput)
        self._run(batch)
        batch.spec_info.prepare_for_decode.assert_called_once_with(batch)
        self.assertEqual(req.decode_batch_idx, 0)
        batch.maybe_evict_swa.assert_not_called()


if __name__ == "__main__":
    unittest.main()
