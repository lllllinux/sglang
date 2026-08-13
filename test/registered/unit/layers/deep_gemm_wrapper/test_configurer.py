import sys
import types
import unittest
from unittest.mock import patch

from sglang.srt.layers.deep_gemm_wrapper import configurer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepGemmConfigurer(unittest.TestCase):
    def _compute(self, *, sm, symbol_present=True, enabled=True):
        deep_gemm = types.ModuleType("deep_gemm")
        if symbol_present:
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous = object()

        with (
            patch.object(configurer, "_is_cuda", True),
            patch.object(configurer, "_is_musa", False),
            patch.object(configurer, "get_device_sm", return_value=sm),
            patch.object(
                configurer.envs.SGLANG_ENABLE_JIT_DEEPGEMM,
                "get",
                return_value=enabled,
            ),
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
        ):
            return configurer._compute_enable_deep_gemm()

    def test_sm120_requires_grouped_fp8_fp4_entrypoint(self):
        self.assertTrue(self._compute(sm=120, symbol_present=True))
        self.assertFalse(self._compute(sm=120, symbol_present=False))

    def test_sm120_preserves_environment_gate(self):
        self.assertFalse(self._compute(sm=120, symbol_present=True, enabled=False))

    def test_other_supported_cuda_arch_does_not_require_sm120_entrypoint(self):
        self.assertTrue(self._compute(sm=100, symbol_present=False))

    def test_enabled_sm120_uses_ue8m0_scales(self):
        with (
            patch.object(configurer, "ENABLE_JIT_DEEPGEMM", True),
            patch.object(configurer, "is_sm100_supported", return_value=False),
            patch.object(configurer, "get_device_sm", return_value=120),
        ):
            self.assertTrue(configurer._compute_scale_ue8m0())

    def test_disabled_sm120_does_not_use_ue8m0_scales(self):
        with (
            patch.object(configurer, "ENABLE_JIT_DEEPGEMM", False),
            patch.object(configurer, "is_sm100_supported", return_value=False),
            patch.object(configurer, "get_device_sm", return_value=120),
        ):
            self.assertFalse(configurer._compute_scale_ue8m0())


if __name__ == "__main__":
    unittest.main()
