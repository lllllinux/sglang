"""DsparkVerifyEpilogue.read_accept must not hand out views of its reused buffers.

The folded accept kernels write into fixed buffers that the next target-verify
graph replay overwrites. Under the overlap scheduler the result device-to-host
copy runs on a side stream that the forward stream never waits on, so a view
read late returns the NEXT step's accept lengths / tokens -- independently on
each TP rank, which made ranks commit different token counts (#42465).
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.dspark_components.dspark_verify import (
    DsparkVerifyEpilogue,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_FIELDS = (
    "correct_len",
    "bonus",
    "cap_trim_lens",
    "commit_lens",
    "new_seq_lens",
    "out_tokens",
)


def _epilogue(max_bs: int = 8, width: int = 6) -> DsparkVerifyEpilogue:
    ep = DsparkVerifyEpilogue.__new__(DsparkVerifyEpilogue)
    for name in _FIELDS:
        shape = (max_bs, width) if name == "out_tokens" else (max_bs,)
        setattr(
            ep, f"{name}_buf", torch.arange(int(torch.Size(shape).numel())).view(shape)
        )
    return ep


class TestDsparkReadAcceptNoAlias(unittest.TestCase):
    def test_outputs_do_not_share_storage_with_the_buffers(self):
        ep = _epilogue()
        outs = ep.read_accept(4)
        for name in _FIELDS:
            buf = getattr(ep, f"{name}_buf")
            out = getattr(outs, name)
            self.assertEqual(out.shape[0], 4, name)
            self.assertNotEqual(
                out.untyped_storage().data_ptr(),
                buf.untyped_storage().data_ptr(),
                name,
            )

    def test_outputs_keep_this_steps_values_after_the_next_replay(self):
        ep = _epilogue()
        outs = ep.read_accept(4)
        expected = {name: getattr(outs, name).clone() for name in _FIELDS}
        for name in _FIELDS:  # the next verify replay overwrites every buffer
            getattr(ep, f"{name}_buf").fill_(-1)
        for name in _FIELDS:
            self.assertTrue(torch.equal(getattr(outs, name), expected[name]), name)

    def test_in_graph_accept_is_what_the_next_read_returns(self):
        # _accept runs inside the target-verify graph; read_accept runs after the
        # replay. If _accept filled read_accept's copies instead of the buffers,
        # the read would return the buffers' initial zeros: commit_lens == 0 for
        # every live request, i.e. a negative correct-draft count downstream.
        bs, stride, vocab = 3, 5, 50
        torch.manual_seed(0)
        candidates = torch.randint(0, vocab, (bs, stride))
        target_predict = torch.randint(0, vocab, (bs, stride))
        target_predict[1, :2] = candidates[1, 1:3]
        target_predict[2, :-1] = candidates[2, 1:]
        logits = torch.randn(bs * stride, vocab)
        logits.scatter_(1, target_predict.view(-1, 1), 100.0)
        ep = DsparkVerifyEpilogue(
            max_bs=bs + 1,
            verify_num_draft_tokens=stride,
            device="cpu",
            tp_sync=SimpleNamespace(sync=lambda site, values: values),
        )
        commit_lens = ep._accept(
            candidates=candidates,
            logits=logits,
            draft_tokens=candidates[:, 1:].contiguous(),
            seq_lens=torch.tensor([7, 11, 13], dtype=torch.int64),
        )
        outs = ep.read_accept(bs)
        self.assertEqual(outs.correct_len.tolist(), [0, 2, stride - 1])
        self.assertEqual(outs.commit_lens.tolist(), [1, 3, stride])
        self.assertTrue(torch.equal(outs.commit_lens, commit_lens))
        self.assertEqual(outs.new_seq_lens.tolist(), [8, 14, 13 + stride])


if __name__ == "__main__":
    unittest.main()
