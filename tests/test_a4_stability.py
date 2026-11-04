"""Synthetic CPU checks: equations, gradients, probes, and native checkpoints."""
import copy
from dataclasses import replace
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.nn import functional as F

from data import TokenDatasetConfig
from model_config import LMConfig
from model_io import load_model, save_model
from train import build_model, train, training_run_name
from experiments.a4_stability import _runtime, diagnostics
from experiments.a4_stability import p1_transformer_depth as p1
from experiments.a4_stability import p2_rnn_depth as p2
from experiments.a4_stability import p3_attention_residuals as p3
from experiments.a4_stability import p4_recurrent_updates as p4
from experiments.a4_stability.common import training_config
from experiments.a4_stability import sweeps
from train import TrainConfig, checked_train_config


TINY = LMConfig('tiny', 32, 8, 16, 32, 2, 2, 2, head_dim=8)


def config(cls, **kwargs):
    return training_config(cls, problem=1, model_config=TINY, precision='fp32',
                           torch_compile_mode=None, **kwargs)


def build(cls):
    with redirect_stdout(io.StringIO()):
        return build_model(config(cls), device='cpu')


class AlgorithmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_models_are_causal_differentiable_and_reloadable(self):
        classes = dict.fromkeys(c for p in (p1, p2, p3, p4) for c in p.MODELS.values())
        tokens = torch.arange(16).reshape(2, 8)
        changed = tokens.clone()
        changed[:, 5:] = (changed[:, 5:] + 7) % TINY.vocab_size
        with tempfile.TemporaryDirectory() as folder:
            for cls in classes:
                with self.subTest(model=cls.__name__):
                    model = build(cls)
                    logits = model(tokens)
                    torch.testing.assert_close(logits[:, :5], model(changed)[:, :5])
                    F.cross_entropy(logits.flatten(0, 1), tokens.flatten()).backward()
                    self.assertTrue(all(p.grad is not None and p.grad.isfinite().all()
                                        for p in model.parameters()))
                    cfg = config(cls)
                    save_model(model, folder, metadata=dict(model_builder=cfg.model_builder,
                                                           qk_norm=cfg.qk_norm))
                    restored = load_model(folder, precision='fp32')
                    self.assertIs(type(restored), cls)
                    torch.testing.assert_close(restored(tokens), logits, rtol=0, atol=0)

    def test_transformer_baseline_matches_native(self):
        cfg = config(p1.Transformer)
        with redirect_stdout(io.StringIO()):
            native = build_model(replace(cfg, model_builder=None), device='cpu')
        tokens = torch.arange(16).reshape(2, 8)
        torch.testing.assert_close(build(p1.Transformer)(tokens), native(tokens), rtol=0, atol=0)

    def test_sweep_has_five_unique_rates_per_variant_and_native_recipe(self):
        configs = sweeps.grid()
        self.assertEqual(len(sweeps.PROVIDED_VARIANTS), 21)
        self.assertEqual(len(configs), 105)
        self.assertEqual(len({training_run_name(c) for c in configs}), 105)
        self.assertTrue(all(sweeps.VARIANTS[name].problem in (1, 2)
                            for name in sweeps.PROVIDED_VARIANTS))
        self.assertLessEqual({'gru', 'tanh'}, set(p4.MODELS))
        for name, variant in sweeps.VARIANTS.items():
            rates = sweeps.learning_rates(name)
            self.assertEqual(len(set(rates)), 5)
            self.assertEqual(rates[2], variant.center)
            for lr in rates:
                cfg = checked_train_config(sweeps.config(name, lr))
                defaults = TrainConfig()
                for key in ('metric_loggers', 'lr_schedule', 'weight_decay', 'num_evals',
                            'warmup_percent', 'grad_norm', 'optimizer_name', 'precision',
                            'batch_size', 'model_seed', 'data_seed'):
                    self.assertEqual(getattr(cfg, key), getattr(defaults, key), key)
                self.assertEqual(training_run_name(cfg), sweeps.run_name(name, lr))
                self.assertEqual(cfg.num_train_sequences, 600_000)
                self.assertEqual(cfg.num_micro_batches, 1)
        self.assertEqual(sweeps.config('d8').run_name, 'a4-d8-lr0.003')
        for invalid in ('', '..', '../run', '/run', 'run/name', 'run\nname'):
            with self.assertRaises(ValueError):
                training_run_name(TrainConfig(run_name=invalid))

    def test_fused_tanh_matches_explicit_states_and_gradients(self):
        sequence = build(p2.PlainRNN).stack[0]
        eager = copy.deepcopy(sequence)
        eager.record_states = True
        x = torch.randn(2, 8, 16, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        a, b = sequence(x), eager(y)
        torch.testing.assert_close(a, b)
        a.square().sum().backward()
        b.square().sum().backward()
        torch.testing.assert_close(x.grad, y.grad)
        for p, q in zip(sequence.parameters(), eager.parameters()):
            torch.testing.assert_close(p.grad, q.grad)
        self.assertFalse(any('_fused' in k for k in sequence.state_dict()))

    def test_gru_handout_equation(self):
        x, h = torch.randn(2, 16), torch.randn(2, 16)
        cell = p4.GRUCell(TINY)
        r = torch.sigmoid(cell.reset.input(x) + cell.reset.hidden(h))
        z = torch.sigmoid(cell.update.input(x) + cell.update.hidden(h))
        candidate = torch.tanh(cell.input(x) + F.linear(r*h, cell.hidden.weight, cell.hidden.bias))
        torch.testing.assert_close(cell.step(cell.project(x), h), (1-z)*h + z*candidate)

    def test_chunk_boundaries_preserve_bptt(self):
        sequence = p2.Recurrent(TINY, p4.GRUCell)
        reference = copy.deepcopy(sequence)
        reference.record_states = True
        x = torch.randn(2, 8, 16, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        with patch.object(_runtime, 'CHUNK_SIZE', 3):
            a = sequence(x)
        b = reference(y)
        torch.testing.assert_close(a, b)
        a[:, -1].sum().backward()
        b[:, -1].sum().backward()
        self.assertGreater(x.grad[:, 0].abs().sum().item(), 0)
        torch.testing.assert_close(x.grad, y.grad)
        for p, q in zip(sequence.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, q.grad)

    def test_editable_cells_compile_with_the_same_gradients(self):
        # Check the editable cell path without compiling a CUDA job.
        for cls in (p4.GRUCell, p4.NoResetCell, p4.NoUpdateCell, p4.FixedUpdateCell,
                    p4.InputGatesCell, p4.NormGRUCell):
            with self.subTest(cell=cls.__name__):
                sequence = p2.Recurrent(TINY, cls)
                other = copy.deepcopy(sequence)
                x = torch.randn(2, 3, 16, requires_grad=True)
                y = x.detach().clone().requires_grad_(True)
                h = torch.zeros(2, 16)
                eager = sequence.cell_steps(sequence.cell.project(x), h)[0]
                compiled = torch.compile(other.cell_steps, backend='aot_eager', fullgraph=True)
                actual = compiled(other.cell.project(y), h)[0]
                torch.testing.assert_close(actual, eager)
                eager.sum().backward()
                actual.sum().backward()
                torch.testing.assert_close(y.grad, x.grad)
                for p, q in zip(sequence.parameters(), other.parameters()):
                    torch.testing.assert_close(p.grad, q.grad)

    def test_ema_keeps_branch_injections_at_full_strength(self):
        # Two constant branch injections (2, then 3) isolate the carry decay.
        for classes, value, gradient in (
                ((p3.EMA01Transformer, p3.EMA01RNN), 3.2, .01),
                ((p3.EMA09Transformer, p3.EMA09RNN), 4.8, .81)):
            for cls in classes:
                with self.subTest(model=cls.__name__):
                    model = build(cls)
                    x = torch.zeros(2, 8, 16, requires_grad=True)
                    branches = (lambda x: torch.full_like(x, 2.),
                                lambda x: torch.full_like(x, 3.))
                    with patch.object(model, 'branches', return_value=branches):
                        result = model.run_depth(x)
                    torch.testing.assert_close(result, torch.full_like(x, value))
                    result.sum().backward()
                    torch.testing.assert_close(x.grad, torch.full_like(x, gradient))

    def test_half_mixes_two_previous_stream_states_and_keeps_gradients(self):
        for cls in (p3.HalfTransformer, p3.HalfRNN):
            model = build(cls)
            branches = (lambda x: 2*x, lambda x: 3*x, lambda x: 5*x)
            # From x0=1: x1=3, x2=.5*3+.5*1+9=11, x3=5.5+1.5+55=62.
            for count, expected in enumerate((3., 11., 62.), 1):
                with self.subTest(model=cls.__name__, steps=count):
                    x = torch.ones(2, 8, 16, requires_grad=True)
                    with patch.object(model, 'branches', return_value=branches[:count]):
                        result = model.run_depth(x)
                    torch.testing.assert_close(result, torch.full_like(x, expected))
                    result.sum().backward()
                    torch.testing.assert_close(x.grad, torch.full_like(x, expected))

    def test_zero_query_attention_equals_uniform_raw_history(self):
        for attention, uniform in ((p3.AttentionTransformer, p3.UniformTransformer),
                                   (p3.AttentionRNN, p3.UniformRNN)):
            tokens = torch.arange(16).reshape(2, 8)
            torch.testing.assert_close(build(attention)(tokens), build(uniform)(tokens))

    def test_blocked_attention_includes_partial_block_in_final_read(self):
        for cls in (p3.Block2Transformer, p3.Block4Transformer, p3.Block2RNN, p3.Block4RNN):
            with self.subTest(model=cls.__name__):
                # Six sublayers leave a partial block for block_size=4.
                with redirect_stdout(io.StringIO()):
                    model = build_model(replace(config(cls), model_config=replace(TINY, num_hidden_layers=3)), 'cpu')
                x = torch.randn(2, 8, 16)
                context = {}
                if isinstance(model, p1.Transformer):
                    context['position_embeddings'] = model.model.rotary_emb(x, torch.arange(8)[None])
                raw = []
                for branch in model.branches():
                    blocks = [sum(raw[i:i+model.block_size]) for i in range(0, len(raw), model.block_size)]
                    raw.append(branch(torch.stack([x] + blocks).mean(0), **context))
                blocks = [sum(raw[i:i+model.block_size]) for i in range(0, len(raw), model.block_size)]
                expected = torch.stack([x] + blocks).mean(0)
                torch.testing.assert_close(model.run_depth(x, **context), expected)

    def test_probes_measure_real_hidden_gradients_and_restore_model(self):
        tokens = torch.arange(16).reshape(2, 8)
        for cls in (p1.NoQKNormTransformer, p2.ModernRNN, p2.GatedOutputRNN,
                    p3.AttentionRNN, p4.GRU):
            with self.subTest(model=cls.__name__):
                model = build(cls)
                measured = diagnostics.profile(model, tokens)
                self.assertTrue(model.training)
                self.assertEqual(len(measured['last']['depth']), len(model.branches())+1)
                if isinstance(model, p2.PlainRNN):
                    self.assertGreater(measured['last']['hidden_gradient_rms'][0][0], 0)
                    for m in model.modules():
                        if isinstance(m, p2.Recurrent):
                            self.assertFalse(m.record_states)
                            self.assertEqual(m.states, [])
                self.assertTrue(all(p.grad is None for p in model.parameters()))
        q = torch.zeros(1, 2, 8, 8)
        attention = diagnostics.attention_summary(q, q)
        for head in attention['first_mass']:
            torch.testing.assert_close(torch.tensor(head), torch.tensor(attention['uniform_first_self']))

    def test_context_controls_preserve_positions_and_reset_state(self):
        tokens = torch.arange(16).reshape(2, 8)
        for cls in (p1.Transformer, p2.RoPERNN, p4.GRU):
            model = build(cls)
            for keep, first in ((7, True), (3, True), (3, False)):
                positions = torch.arange(7-keep, 7)
                if first and keep < 7:
                    positions = torch.cat((torch.tensor([0]), positions))
                expected = F.cross_entropy(model(tokens[:, positions], position_ids=positions[None])[:, -1],
                                           tokens[:, -1], reduction='none')
                actual = diagnostics.context_loss(model, tokens, keep, first)
                torch.testing.assert_close(actual, expected)
                self.assertTrue(model.training)

    def test_shared_checkpoint_measurement_example_locally(self):
        from experiments.a4_stability.measure_default import measure
        with tempfile.TemporaryDirectory() as folder:
            cfg = sweeps.config('d8', model_config=replace(TINY, context_length=512, num_hidden_layers=1))
            with redirect_stdout(io.StringIO()):
                model = build_model(cfg, 'cpu')
            save_model(model, Path(folder)/cfg.run_name, metadata=dict(model_builder=cfg.model_builder))
            validation = {'input_ids': (np.arange(512)[None] % 32).astype('uint16')}
            class Rows:
                def __getitem__(self, key):
                    return validation
            with patch('data.load_token_dataset', return_value=Rows()):
                measured = measure(cfg.run_name, folder, 'cpu')
            self.assertEqual(len(measured['context_loss_increase']), 6)
            self.assertIn('depth', measured['last'])

    def test_native_training_logs_rms_and_writes_custom_checkpoints(self):
        for cls in (p3.Block2Transformer, p4.GRU):
            with self.subTest(model=cls.__name__), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                for split in ('train', 'val'):
                    path = root / split
                    path.mkdir()
                    np.random.default_rng(0).integers(0, 32, size=(8, 8), dtype=np.uint16).tofile(path/'tokens.bin')
                    (path/'metadata.json').write_text(json.dumps(dict(dtype='uint16', num_sequences=8, seq_len=8)))
                cfg = config(cls, train_dataset=TokenDatasetConfig('train', 'test', 8, path=str(root/'train')),
                             val_dataset=TokenDatasetConfig('val', 'test', 8, path=str(root/'val')),
                             num_train_sequences=8, num_epochs=2., batch_size=4, num_evals=4, warmup_steps=0,
                             data_seed=None, model_dir=str(root/'models'), wandb_online=True,
                             keep_checkpoint_steps=(2,))
                wandb = Mock()
                wandb.init.return_value = SimpleNamespace(id='test', url='test')
                with patch.dict(sys.modules, {'wandb': wandb}), patch('torch.cuda.device_count', return_value=0):
                    model = train(cfg)
                run = root/'models'/training_run_name(cfg)
                self.assertRegex(wandb.init.call_args.kwargs['id'], r'^[0-9a-f]{8}$')
                self.assertFalse((run/'diagnostics').exists())
                self.assertEqual(torch.load(run/'step_2.pt', weights_only=False)['completed_steps'], 2)
                rows = [call.args[0] for call in wandb.log.call_args_list]
                rms = next(row for row in rows if 'logging/rms/global/activation' in row)
                for name in ('activation', 'parameter', 'gradient'):
                    self.assertGreater(rms[f'logging/rms/global/{name}'], 0.)
                restored = load_model(run, precision='fp32')
                self.assertIs(type(restored), cls)
                for name, tensor in model.state_dict().items():
                    torch.testing.assert_close(tensor, restored.state_dict()[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
