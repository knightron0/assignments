import unittest

import torch

from model_config import LMConfig
from modeling import AutoregressiveLM
from optimizers import build_optimizer


class HeadDecayTests(unittest.TestCase):
    def model(self, tied=False):
        config = LMConfig(name="tiny", vocab_size=11, context_length=6, hidden_size=8,
                          intermediate_size=12, num_hidden_layers=1, num_attention_heads=2,
                          num_key_value_heads=2, head_dim=4)
        return AutoregressiveLM(config, tie_word_embeddings=tied)

    def optimizer(self, model, factor):
        return build_optimizer(model, "adamw", .001, .1, .9, .95,
                               head_weight_decay_multiplier=factor)

    def test_only_head_decay_changes_and_zero_gradient_step_matches(self):
        for factor in (1, 5, 20):
            model = self.model()
            baseline = self.optimizer(model, 1)
            default_decay = {id(p): g["weight_decay"] for g in baseline.param_groups for p in g["params"]}
            opt = self.optimizer(model, factor)
            params = [p for g in opt.param_groups for p in g["params"]]
            self.assertEqual(len(params), len(set(map(id, params))))
            self.assertEqual(set(map(id,params)),set(map(id,model.parameters())))
            before = {id(p): p.detach().clone() for p in params}
            for g in opt.param_groups:
                for p in g["params"]:
                    expected = default_decay[id(p)] * (factor if p is model.lm_head.weight else 1)
                    self.assertAlmostEqual(g["weight_decay"], expected)
                    p.grad = torch.zeros_like(p)
            opt.step()
            for g in opt.param_groups:
                for p in g["params"]:
                    torch.testing.assert_close(p, before[id(p)]*(1-.001*g["weight_decay"]))

    def test_tied_head_cannot_be_decayed_separately(self):
        with self.assertRaisesRegex(ValueError, "untied"):
            self.optimizer(self.model(tied=True), 5)


if __name__ == "__main__":
    unittest.main()
