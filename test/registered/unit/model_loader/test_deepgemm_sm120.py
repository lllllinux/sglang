import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.model_loader import utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepGemmSm120WeightRequant(unittest.TestCase):
    def test_sm120_keeps_dense_block_fp8_scales(self):
        with (
            patch.multiple(
                deep_gemm_wrapper,
                ENABLE_JIT_DEEPGEMM=True,
                DEEPGEMM_SCALE_UE8M0=True,
            ),
            patch.object(utils, "get_device_sm", return_value=120),
        ):
            self.assertFalse(
                utils.should_deepgemm_weight_requant_ue8m0(
                    [128, 128],
                    output_dtype=torch.bfloat16,
                    weight_shape=(64, 128),
                )
            )

    def test_other_supported_arch_preserves_requantization(self):
        with (
            patch.multiple(
                deep_gemm_wrapper,
                ENABLE_JIT_DEEPGEMM=True,
                DEEPGEMM_SCALE_UE8M0=True,
            ),
            patch.object(utils, "get_device_sm", return_value=100),
        ):
            self.assertTrue(
                utils.should_deepgemm_weight_requant_ue8m0(
                    [128, 128],
                    output_dtype=torch.bfloat16,
                    weight_shape=(64, 128),
                )
            )


if __name__ == "__main__":
    unittest.main()
