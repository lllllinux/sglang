"""Regression tests for DeepSeek-V4 sparse MLA query-head selection."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention import flash_mla_sm120
from sglang.srt.environ import envs
from sglang.srt.models import deepseek_v4
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeepseekV4SparseMlaHeads(CustomTestCase):
    @staticmethod
    def _layer(*, local_heads: int = 16, total_heads: int = 64, tp_size: int = 4):
        layer = deepseek_v4.MqaAttentionBase.__new__(deepseek_v4.MqaAttentionBase)
        torch.nn.Module.__init__(layer)
        layer.n_local_heads = local_heads
        layer.n_heads = total_heads
        layer.attn_tp_size = tp_size
        layer.attn_tp_rank = 1
        layer.attn_sink = torch.nn.Parameter(
            torch.arange(total_heads, dtype=torch.float32)
        )
        layer._attn_sink_local = None
        return layer

    def test_tp4_sm120_flashinfer_decode_uses_exact_heads(self):
        """A supported TP4 decode must not execute 48 padded query heads."""
        layer = self._layer()
        with (
            patch.object(deepseek_v4, "is_sm120_supported", return_value=True),
            envs.SGLANG_SM120_FLASHMLA_BACKEND.override("flashinfer"),
            patch.object(
                deepseek_v4,
                "flashinfer_dsv4_decode_supports_num_heads",
                return_value=True,
            ),
        ):
            self.assertEqual(layer._kernel_num_heads(num_tokens=64), 16)

    def test_sm120_prefill_keeps_local_exact_heads(self):
        """SM120 prefill must retain the branch's existing unpadded shape."""
        layer = self._layer(local_heads=8, tp_size=8)
        with (
            patch.object(deepseek_v4, "is_sm120_supported", return_value=True),
            envs.SGLANG_SM120_FLASHMLA_BACKEND.override("flashinfer"),
        ):
            self.assertEqual(layer._kernel_num_heads(num_tokens=65), 8)

    def test_unsupported_decode_paths_keep_padding(self):
        """Fallback backends and unsupported FlashInfer shapes require padding."""
        layer = self._layer()
        with patch.object(deepseek_v4, "is_sm120_supported", return_value=True):
            for backend in ("triton", "torch"):
                with (
                    self.subTest(backend=backend),
                    envs.SGLANG_SM120_FLASHMLA_BACKEND.override(backend),
                ):
                    self.assertEqual(layer._kernel_num_heads(num_tokens=1), 64)

            with (
                envs.SGLANG_SM120_FLASHMLA_BACKEND.override("flashinfer"),
                patch.object(
                    deepseek_v4,
                    "flashinfer_dsv4_decode_supports_num_heads",
                    return_value=False,
                ),
            ):
                self.assertEqual(layer._kernel_num_heads(num_tokens=1), 64)

    def test_capability_check_rejects_prefill_token_count(self):
        """FlashInfer's external decode cutoff must gate exact-head dispatch."""
        with patch.object(
            flash_mla_sm120,
            "_flashinfer_dsv4_decode_capabilities",
            return_value=(64, frozenset({8, 16, 32, 64, 128})),
        ):
            self.assertTrue(
                flash_mla_sm120.flashinfer_dsv4_decode_supports_num_heads(16, 64)
            )
            self.assertFalse(
                flash_mla_sm120.flashinfer_dsv4_decode_supports_num_heads(16, 65)
            )

    def test_sink_views_share_fallback_allocation(self):
        """Exact and padded sink views must keep one CUDA-graph-stable pointer."""
        layer = self._layer()

        exact = layer._local_attn_sink(16)
        padded = layer._local_attn_sink(64)

        torch.testing.assert_close(exact, torch.arange(16, 32, dtype=torch.float32))
        torch.testing.assert_close(padded[16:], torch.zeros(48))
        self.assertEqual(exact.data_ptr(), padded.data_ptr())

    def test_sink_no_arg_preserves_dspark_contract(self):
        """DSpark's no-argument call must continue returning the padded sink."""
        sink = self._layer()._local_attn_sink()

        self.assertEqual(sink.shape, (64,))
        torch.testing.assert_close(sink[16:], torch.zeros(48))


if __name__ == "__main__":
    unittest.main()
