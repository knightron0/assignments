"""Verify the P4 norm/head transformation preserves predictions and rescales curvature."""

import unittest

import torch

from model_config import LMConfig
from modeling import AutoregressiveLM, initialize_model
from train import causal_lm_loss
from experiments.a3_optimization.hessian_measure import scaled
from experiments.a3_optimization.p4_core_runs import ScaledHeadLM


class RescalingTests(unittest.TestCase):
    def test_random_partition_matches_exact_diagonal_energy(self):
        from experiments.a3_optimization.hessian import random_block_energy
        diag = torch.arange(1, 13, dtype=torch.float32)
        blocks = {"small": [(0, 3)], "large": [(3, 12)]}
        keys, energy = random_block_energy(lambda v: diag * v, 12, blocks, "cpu", probes=2, seed=101)
        perm = torch.randperm(12, generator=torch.Generator().manual_seed(101))
        expected = torch.diag(torch.stack([diag[perm[:3]].square().sum(), diag[perm[3:]].square().sum()]))
        self.assertEqual(keys, list(blocks))
        for sample in energy:
            torch.testing.assert_close(torch.from_numpy(sample).float(), expected)

    def test_logits_and_head_curvature(self):
        config = LMConfig(name="tiny", vocab_size=11, context_length=6, hidden_size=8,
                          intermediate_size=12, num_hidden_layers=2, num_attention_heads=2,
                          num_key_value_heads=2, head_dim=4)
        torch.manual_seed(42)
        model = AutoregressiveLM(config).eval()
        initialize_model(model)
        tokens = torch.randint(0, 11, (2, 6))
        logits = model(input_ids=tokens).detach()
        head = model.lm_head.weight
        v = torch.randn_like(head)

        def hvp():
            loss = causal_lm_loss(model(input_ids=tokens), tokens)
            grad, = torch.autograd.grad(loss, head, create_graph=True)
            value, = torch.autograd.grad((grad * v).sum(), head)
            return value

        before = hvp()
        for factor in (0.5, 2.0, 4.0):
            with scaled(model, {"model.norm.weight": factor, "lm_head.weight": 1 / factor}):
                torch.testing.assert_close(model(input_ids=tokens), logits, atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(hvp(), before * factor**2, atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(model(input_ids=tokens), logits, atol=0, rtol=0)
        custom = ScaledHeadLM(config, norm_scale=4.0).eval()
        custom.load_state_dict(model.state_dict())
        custom.post_initialize()
        torch.testing.assert_close(custom(input_ids=tokens), logits, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
