import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
    HostConfidenceBudgetPlanner,
)
from sglang.srt.speculative.dspark_components.dspark_sps import SpsCostTable
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _flat_table() -> SpsCostTable:
    return SpsCostTable(
        sample_batch_tokens=[1], sample_steps_per_sec=[1.0], max_batch_tokens=4096
    )


def _smoothing_planner(alpha: float, pool: int = 4) -> HostConfidenceBudgetPlanner:
    model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(pool, 8))
    )
    with mock.patch.dict(
        os.environ,
        {
            "SGLANG_DSPARK_BUDGET_SURVIVAL_EMA": str(alpha),
            "SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS": "1",
        },
    ):
        return HostConfidenceBudgetPlanner(
            sps_table=_flat_table(),
            cfg=DSparkScheduleConfig(gamma=2),
            model_runner=model_runner,
        )


class TestSmoothedBudgetSurvival(CustomTestCase):
    def _step(self, planner, rows, conf, gens, lagged=None):
        g = torch.tensor(gens, dtype=torch.int64)
        return planner._smoothed_survival(
            survival=torch.tensor(conf, dtype=torch.float32),
            lagged_generation=g if lagged is None else torch.tensor(lagged),
            current_generation=g,
            req_pool_indices_cpu=torch.tensor(rows),
        )

    def test_off_by_default_passes_survival_through(self):
        planner = _smoothing_planner(1.0)
        s = torch.tensor([[0.9, 0.2]])
        out = planner._smoothed_survival(
            survival=s,
            lagged_generation=torch.tensor([1]),
            current_generation=torch.tensor([1]),
            req_pool_indices_cpu=torch.tensor([0]),
        )
        self.assertIs(out, s)

    def test_running_average_of_a_lone_request(self):
        planner = _smoothing_planner(0.5)
        first = self._step(planner, [2], [[0.8, 0.4]], [1])
        self.assertTrue(torch.allclose(first, torch.tensor([[0.8, 0.4]])))
        second = self._step(planner, [2], [[0.4, 0.0]], [1])
        self.assertTrue(torch.allclose(second, torch.tensor([[0.6, 0.2]])))

    def test_output_does_not_alias_the_state(self):
        planner = _smoothing_planner(0.5)
        first = self._step(planner, [2], [[0.8, 0.4]], [1])
        first.zero_()
        second = self._step(planner, [2], [[0.4, 0.0]], [1])
        self.assertTrue(torch.allclose(second, torch.tensor([[0.6, 0.2]])))

    def test_batches_pass_through_and_restart_the_average(self):
        planner = _smoothing_planner(0.5)
        self._step(planner, [2], [[1.0, 1.0]], [1])
        both = torch.tensor([[0.2, 0.1], [0.3, 0.2]])
        out = planner._smoothed_survival(
            survival=both,
            lagged_generation=torch.tensor([1, 1]),
            current_generation=torch.tensor([1, 1]),
            req_pool_indices_cpu=torch.tensor([2, 3]),
        )
        self.assertIs(out, both)
        alone = self._step(planner, [2], [[0.2, 0.1]], [1])
        self.assertTrue(torch.allclose(alone, torch.tensor([[0.2, 0.1]])))

    def test_new_request_in_a_row_restarts(self):
        planner = _smoothing_planner(0.5)
        self._step(planner, [1], [[1.0, 1.0]], [3])
        out = self._step(planner, [1], [[0.2, 0.1]], [4])
        self.assertTrue(torch.allclose(out, torch.tensor([[0.2, 0.1]])))

    def test_stale_lagged_survival_passes_through_and_keeps_state(self):
        planner = _smoothing_planner(0.5)
        self._step(planner, [0], [[0.8, 0.4]], [1])
        # Lagged confidence from an older generation: survival is the planner's
        # placeholder (ones) and must neither be smoothed nor folded in.
        out = self._step(planner, [0], [[1.0, 1.0]], [1], lagged=[0])
        self.assertTrue(torch.equal(out, torch.ones(1, 2)))
        again = self._step(planner, [0], [[0.4, 0.0]], [1])
        self.assertTrue(torch.allclose(again, torch.tensor([[0.6, 0.2]])))

    def test_pool_clear_forgets_history(self):
        planner = _smoothing_planner(0.5)
        self._step(planner, [1], [[1.0, 1.0]], [1])
        planner.reset_request_history()
        out = self._step(planner, [1], [[0.2, 0.1]], [1])
        self.assertTrue(torch.allclose(out, torch.tensor([[0.2, 0.1]])))

    def test_request_pool_generations_survive_clear(self):
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

        pool = ReqToTokenPool(
            size=2, max_context_len=4, device="cpu", enable_memory_saver=False
        )
        rows = pool.alloc_rows(2)
        before = pool.req_generation[rows].clone()
        pool.clear()
        again = pool.alloc_rows(2)
        self.assertEqual(sorted(rows), sorted(again))
        self.assertTrue(bool((pool.req_generation[again] > before).all()))

    def test_rejects_out_of_range_weight(self):
        with self.assertRaises(ValueError):
            _smoothing_planner(0.0)


if __name__ == "__main__":
    unittest.main()
