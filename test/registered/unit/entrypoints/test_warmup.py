import asyncio
import unittest

from sglang.srt.entrypoints.warmup import decode_paths
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-c-test-cpu")


class _TokenizerManager:
    def __init__(self):
        self.requests = []
        self.drained = 0

    async def generate_request(self, request, _raw_request):
        self.requests.append(request)
        yield {"step": 1}
        yield {"step": 2}
        self.drained += 1


class TestDecodePathsWarmup(unittest.TestCase):
    def test_drains_text_greedy_and_input_id_sampled_requests(self):
        manager = _TokenizerManager()

        asyncio.run(decode_paths("null", manager))

        self.assertEqual(manager.drained, 2)
        self.assertEqual(len(manager.requests), 2)
        self.assertEqual(
            [request.sampling_params["temperature"] for request in manager.requests],
            [0.0, 1.0],
        )
        for request in manager.requests:
            self.assertEqual(request.sampling_params["max_new_tokens"], 16)
            self.assertTrue(request.sampling_params["ignore_eos"])
        self.assertEqual(manager.requests[0].text, "The capital city of France is")
        self.assertIsNone(manager.requests[0].input_ids)
        self.assertIsNone(manager.requests[1].text)
        self.assertEqual(len(manager.requests[1].input_ids), 256)


if __name__ == "__main__":
    unittest.main()
