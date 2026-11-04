"""P3 checkpoint probes on Modal: depth statistics and routing weights for every variant.

    uv run python -m experiments.a4_stability.p3_measure

Each variant is probed at its best LR (W&B final val loss over the two trained rates) at init,
step 100 and final, with diagnostics.profile on the same 32 held-out sequences as P1/P2:
activation RMS, gradient RMS for L and L_T, branch/input RMS ratio, adjacent-depth cosine and
the mean routing weights of every attention-residual router. Baselines are re-probed the same way.
Also records validation loss on VAL_SEQUENCES sequences (d8-block2 at 0.0045 has no W&B record).
Writes results/a4_stability/p3/measurements/<name>.json.
"""
from pathlib import Path

OUTPUT = Path('results/a4_stability/p3/measurements')
USER_ROOT = Path('/root/data/ckpts')
VAL_SEQUENCES = 512
# name: (variant, lr, checkpoint directory). None = the supplied baseline on the shared volume.
RUNS = {
    'd8': ('d8', .003, None), 'rnn': ('rnn', .002, None),
    'd8-half': ('d8-half', .003, 'a4-d8-half-lr0.003'),
    'd8-ema01': ('d8-ema01', .003, 'a4-d8-ema01-lr0.003'),  # collapsed at both rates
    'd8-ema09': ('d8-ema09', .003, 'a4-d8-ema09-lr0.003'),
    'd8-uniform': ('d8-uniform', .003, 'a4-d8-uniform-lr0.003'),
    'd8-attention': ('d8-attention', .003, 'a4-d8-attention-h200-lr0.003'),
    'd8-block2': ('d8-block2', .0045, 'a4-d8-block2-lr0.0045'),
    'd8-block2@0.009': ('d8-block2', .009, 'a4-d8-block2-lr0.009'),
    'd8-block4': ('d8-block4', .003, 'a4-d8-block4-lr0.003'),
    'rnn-half': ('rnn-half', .001, 'a4-rnn-half-lr0.001'),
    'rnn-ema01': ('rnn-ema01', .001, 'a4-rnn-ema01-lr0.001'),
    'rnn-ema09': ('rnn-ema09', .001, 'a4-rnn-ema09-lr0.001'),
    'rnn-uniform': ('rnn-uniform', .002, 'a4-rnn-uniform-lr0.002'),
    'rnn-attention': ('rnn-attention', .002, 'a4-rnn-attention-lr0.002'),
    'rnn-block2': ('rnn-block2', .001, 'a4-rnn-block2-lr0.001'),
    'rnn-block4': ('rnn-block4', .001, 'a4-rnn-block4-lr0.001'),
}
CHECKPOINTS = ('init', 'step 100', 'final')


def load(name, checkpoint):
    from model_io import load_model
    from . import p1a_depth
    variant, lr, directory = RUNS[name]
    if directory is None or checkpoint == 'init':
        return p1a_depth.load(checkpoint, 'cuda', variant, lr)
    path = USER_ROOT / directory
    return load_model(path / 'step_100.pt' if checkpoint == 'step 100' else path, device='cuda', precision='fp32')


def val_loss(model):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from data import dclm_val_dataset, load_token_dataset
    validation = load_token_dataset(dclm_val_dataset(), role='val', data_seed=42)
    tokens = torch.tensor(np.asarray(validation[:VAL_SEQUENCES]['input_ids']).astype('int64'), device='cuda')
    total, count = 0., 0
    with torch.no_grad():
        for batch in tokens.split(32):
            logits = model(batch[:, :-1])
            total += F.cross_entropy(logits.flatten(0, 1).float(), batch[:, 1:].flatten(), reduction='sum').item()
            count += batch[:, 1:].numel()
    return total / count


def run(name):
    from .diagnostics import profile
    from .p1a_depth import held_out_tokens
    variant, lr, _ = RUNS[name]
    tokens = held_out_tokens('cuda')
    result = dict(name=name, variant=variant, lr=lr)
    for checkpoint in CHECKPOINTS:
        model = load(name, checkpoint)
        stats = profile(model, tokens)
        result[checkpoint] = {k: stats[k] for k in ('full', 'last', 'routing') if k in stats}
        for objective in ('full', 'last'):  # per-time recurrent states are P2's business
            for key in ('hidden_rms', 'hidden_gradient_rms', 'fraction_abs_gt_095'):
                result[checkpoint][objective].pop(key, None)
        if checkpoint == 'final':
            result['val_loss'] = val_loss(model.eval())
    return result


def probe(name):
    try:
        return name, run(name), None
    except Exception as error:  # one failed probe should not discard the others
        return name, None, repr(error)


if __name__ == '__main__':
    import modal
    from modal_utils import app, build_image, VOLUME_MOUNTS, MODAL_ENVIRONMENT
    from .diagnostics import save

    remote = app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu='H100', timeout=3600)(probe)
    todo = [name for name in RUNS if not (OUTPUT / f'{name}.json').exists()]  # rerun only what is missing
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT):
        for name, result, error in remote.map(todo, order_outputs=False):  # save each as it finishes
            if error:
                print('FAILED', name, error)
                continue
            save(OUTPUT / f'{name}.json', result)
            print(OUTPUT / f'{name}.json', result['val_loss'])
