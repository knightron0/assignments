"""P1(a)(i): activation and activation-gradient RMS across depth for the default d8.

    uv run python -m experiments.a4_stability.p1a_depth

Probes three checkpoints on Modal with the same held-out sequences: initialization
(rebuilt with model seed 42, exactly as train.py builds it), step 100 (early/), and final.
L and L_T are backpropagated separately by diagnostics.profile (FP32, eval mode).
Writes results/a4_stability/p1a_i/{measurements.json, depth.png, ratios.md}.
"""
import json
from pathlib import Path

import numpy as np

VARIANT, LR = 'd8', .003
RUN = f'a4-{VARIANT}-lr{LR:g}'
REFERENCE_ROOT = Path('/root/shared_data/a4_stability')
OUTPUT = Path('results/a4_stability/p1a_i')
SEQUENCES = 32  # the first 32 held-out validation sequences, fixed for every probe
CHECKPOINTS = ('init', 'step 100', 'final')


def held_out_tokens(device):
    import torch
    from data import dclm_val_dataset, load_token_dataset
    validation = load_token_dataset(dclm_val_dataset(), role='val', data_seed=42)
    return torch.tensor(np.asarray(validation[:SEQUENCES]['input_ids']).astype('int64'), device=device)


def load(checkpoint, device, variant=VARIANT, lr=LR):
    from model_io import load_model
    if checkpoint == 'init':
        from train import build_model
        from .sweeps import config
        return build_model(config(variant, lr), device)  # 'mp' keeps FP32 weights, as in training
    path = REFERENCE_ROOT / f'a4-{variant}-lr{lr:g}' / ('early' if checkpoint == 'step 100' else '')
    return load_model(path, device=device, precision='fp32')


def measure(device='cuda', variant=VARIANT, lr=LR):
    from .diagnostics import profile
    tokens = held_out_tokens(device)
    results = {}
    for checkpoint in CHECKPOINTS:
        model = load(checkpoint, device, variant, lr)
        stats = profile(model, tokens)
        results[checkpoint] = {objective: dict(loss=stats[objective]['loss'],
                                               activation_rms=[r['activation_rms'] for r in stats[objective]['depth']],
                                               gradient_rms=[r['gradient_rms'] for r in stats[objective]['depth']])
                               for objective in ('full', 'last')}
    results['parameters'] = sum(p.numel() for p in model.parameters())
    return results


def point_labels(count):
    return ['emb'] + [f'{i}{part}' for i in range((count - 1)//2) for part in ('A', 'M')]


def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .p1_plots import INK, MUTED, save

    shades = dict(zip(CHECKPOINTS, ('#86b6ef', '#2a78d6', '#104281')))  # light → dark: init → final
    count = len(results['final']['full']['activation_rms'])
    x, labels = np.arange(count), point_labels(count)
    panels = (('activation_rms', 'full', 'Activation RMS of the residual stream', 'RMS(x)'),
              ('gradient_rms', 'full', 'Gradient RMS for L (mean loss over all positions)', 'RMS(∂L/∂x)'),
              ('gradient_rms', 'last', 'Gradient RMS for L_T (final-token loss)', 'RMS(∂L_T/∂x)'))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, (key, objective, title, ylabel) in zip(axes, panels):
        for checkpoint in CHECKPOINTS:
            ax.plot(x, results[checkpoint][objective][key], color=shades[checkpoint], marker='o', ms=4,
                    label=checkpoint)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    axes[2].sharey(axes[1])  # L and L_T gradients on one scale
    from .p1b_plots import DEPTH_XLABEL, label_depth
    for ax in axes:
        ax.set_yscale('log')
        label_depth(ax, count)
        ax.set_xlabel(DEPTH_XLABEL)
    caption = '   '.join(f"{c}: L={results[c]['full']['loss']:.2f}, L_T={results[c]['last']['loss']:.2f}"
                         for c in CHECKPOINTS)
    fig.suptitle(f'P1(a)(i) {RUN}: depth profile on {SEQUENCES} held-out sequences (FP32, eval mode)\n{caption}',
                 fontweight='bold', color=INK, fontsize=10)
    save(fig, 'depth.png', OUTPUT)


def ratio_table(results):
    lines = ['| loss | bottom point | ' + ' | '.join(CHECKPOINTS) + ' |', '|---|---|' + '---|'*len(CHECKPOINTS)]
    for objective, name in (('full', 'L'), ('last', 'L_T')):
        for bottom, label in ((0, '0 (emb)'), (2, '2 (after 0M)')):
            cells = []
            for checkpoint in CHECKPOINTS:
                grads = results[checkpoint][objective]['gradient_rms']
                cells.append(f'{grads[bottom]/grads[-1]:.2f}')
            lines.append(f'| {name} | {label} | ' + ' | '.join(cells) + ' |')
    lines.append('')
    lines.append('Ratio = gradient RMS at the bottom point ÷ gradient RMS at point 16 (after 7M).')
    lines.append('Point 0 is dominated by the small embedding scale (the first RMSNorm makes the gradient')
    lines.append('scale like 1/RMS(x)); point 2 measures propagation through the remaining 14 updates.')
    lines.append(f"Parameters: {results['parameters']:,}")
    (OUTPUT/'ratios.md').write_text('\n'.join(lines)+'\n')
    print(OUTPUT/'ratios.md')


if __name__ == '__main__':
    import modal
    from modal_utils import app, build_image, VOLUME_MOUNTS, MODAL_ENVIRONMENT
    from .diagnostics import save as save_json
    probe = app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu='H100', timeout=1800)(measure)
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT):
        results = probe.remote()
    save_json(OUTPUT/'measurements.json', results)
    plot(results)
    ratio_table(results)
