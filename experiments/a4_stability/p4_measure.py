"""P4 checkpoint probes on Modal: P2-style depth/time diagnostics plus gate statistics for the GRUs.

    uv run python -m experiments.a4_stability.p4_measure

Runs on CPU containers: training holds the workspace's two GPU slots. Each run is probed at init,
step 100 and final with diagnostics.profile on the same 32 held-out sequences as P1-P3, and its final
checkpoint is scored on VAL_SEQUENCES validation sequences (several runs never synced to W&B).
Gate statistics: mean update gate z and reset gate r per layer and per position, and the fraction of
z below 0.1 (a coordinate that mostly keeps its old value). Writes results/a4_stability/p4/measurements/.
"""
from pathlib import Path

OUTPUT = Path('results/a4_stability/p4/measurements')
USER_ROOT = Path('/root/data/ckpts')
VAL_SEQUENCES = 128
# name: (variant, lr, checkpoint directory). None = the supplied tanh baseline on the shared volume.
RUNS = {
    'rnn': ('rnn', .002, None),
    'gru': ('gru', .006, 'a4-gru-lr0.006'),
    'gru-no-reset': ('gru-no-reset', .0015, 'a4-gru-no-reset-lr0.0015'),
    'gru-no-update@0.003': ('gru-no-update', .003, 'a4-gru-no-update-lr0.003'),
    'gru-no-update@0.006': ('gru-no-update', .006, 'a4-gru-no-update-lr0.006'),
    'gru-fixed-update@0.003': ('gru-fixed-update', .003, 'a4-gru-fixed-update-lr0.003'),
    'gru-fixed-update@0.006': ('gru-fixed-update', .006, 'a4-gru-fixed-update-lr0.006'),
    'gru-input-gates': ('gru-input-gates', .003, 'a4-gru-input-gates-lr0.003'),
    'gru-const-update': ('gru-const-update', .003, 'a4-gru-const-update-lr0.003'),
    'gru-norm': ('gru-norm', .003, 'a4-gru-norm-lr0.003'),
    'gru-no-candidate-h': ('gru-no-candidate-h', .003, 'a4-gru-no-candidate-h-lr0.003'),
    'gru-min': ('gru-min', .003, 'a4-gru-min-lr0.003'),
}
ONLY = ('gru-input-gates', 'gru-const-update')  # runs with a final checkpoint not yet probed
CHECKPOINTS = ('init', 'step 100', 'final')
DEVICE = 'cpu'


def load(name, checkpoint):
    from model_io import load_model
    from . import p1a_depth
    variant, lr, directory = RUNS[name]
    if directory is None or checkpoint == 'init':
        return p1a_depth.load(checkpoint, DEVICE, variant, lr)
    path = USER_ROOT / directory
    return load_model(path / 'step_100.pt' if checkpoint == 'step 100' else path, device=DEVICE, precision='fp32')


def val_loss(model):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from data import dclm_val_dataset, load_token_dataset
    validation = load_token_dataset(dclm_val_dataset(), role='val', data_seed=42)
    tokens = torch.tensor(np.asarray(validation[:VAL_SEQUENCES]['input_ids']).astype('int64'), device=DEVICE)
    total, count = 0., 0
    with torch.no_grad():
        for batch in tokens.split(32):
            logits = model(batch[:, :-1])
            total += F.cross_entropy(logits.flatten(0, 1).float(), batch[:, 1:].flatten(), reduction='sum').item()
            count += batch[:, 1:].numel()
    return total / count


def gates(model, tokens):
    """Mean gate value per layer and per position; z = fraction of new information written."""
    import torch
    from .p2_rnn_depth import Recurrent
    from .p4_recurrent_updates import Gate
    seen = {}
    handles = []
    for layer, sequence in enumerate(m for m in model.modules() if isinstance(m, Recurrent)):
        for label in ('update', 'reset'):
            gate = getattr(sequence.cell, label, None)
            if isinstance(gate, Gate):
                seen[(layer, label)] = []
                handles.append(gate.register_forward_hook(
                    lambda m, a, out, key=(layer, label): seen[key].append(out.detach())))
    with torch.no_grad():
        model.eval()(tokens[:, :-1])
    for handle in handles:
        handle.remove()
    result = {}
    for (layer, label), steps in seen.items():
        values = torch.stack(steps, 1)  # batch, time, channel
        result.setdefault(label, []).append(dict(
            mean=values.mean().item(), by_position=values.mean((0, 2)).tolist(),
            below_01=(values < .1).float().mean().item(), above_09=(values > .9).float().mean().item()))
    return result


def run(name):
    import torch
    from .diagnostics import profile
    from .p1a_depth import held_out_tokens
    torch.set_num_threads(16)
    variant, lr, _ = RUNS[name]
    tokens = held_out_tokens(DEVICE)
    result = dict(name=name, variant=variant, lr=lr)
    for checkpoint in CHECKPOINTS:
        model = load(name, checkpoint)
        stats = profile(model, tokens)
        result[checkpoint] = {k: stats[k] for k in ('full', 'last', 'temporal_cosine') if k in stats}
        if checkpoint == 'final':
            result['gates'] = gates(model, tokens)
            result['val_loss'] = val_loss(model.eval())
    return result


def probe(name):
    try:
        return name, run(name), None
    except Exception as error:  # one failed probe should not discard the others
        import traceback
        return name, None, traceback.format_exc()


if __name__ == '__main__':
    import modal
    from modal_utils import app, build_image, VOLUME_MOUNTS, MODAL_ENVIRONMENT
    from .diagnostics import save

    remote = app.function(image=build_image(), volumes=VOLUME_MOUNTS, cpu=16, memory=32768, timeout=3600)(probe)
    todo = [name for name in (ONLY or RUNS) if not (OUTPUT / f'{name}.json').exists()]  # rerun only what is missing
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT):
        for name, result, error in remote.map(todo, order_outputs=False):  # save each as it finishes
            if error:
                print('FAILED', name, error)
                continue
            save(OUTPUT / f'{name}.json', result)
            print(OUTPUT / f'{name}.json', result['val_loss'])
