"""P3 plots: loss and depth statistics for the residual variants, plus routing weights.

    uv run python -m experiments.a4_stability.p3_plots

Needs p3_measure output. Writes results/a4_stability/p3/*.png and losses.md.
"""
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .p1_fetch import RESULTS
from .p1_plots import BASE, INK, MUTED, save
from .p1b_plots import CATEGORICAL, DEPTH_XLABEL, label_depth
from .p3_measure import OUTPUT as MEASUREMENTS

OUTPUT = RESULTS / 'p3'
FAMILIES = {'d8': 'd8 transformer', 'rnn': 'modern RNN'}
LABELS = {'': 'additive (baseline)', 'half': 'half-and-half', 'ema01': 'EMA β = 0.1', 'ema09': 'EMA β = 0.9',
          'uniform': 'uniform routing', 'attention': 'full attention residuals', 'block2': 'blocked, B = 2',
          'block4': 'blocked, B = 4'}
PARTS = {'a': ('', 'half', 'ema01', 'ema09', 'uniform'), 'b': ('', 'attention'), 'c': ('', 'attention', 'block2', 'block4')}
COLORS = {'': BASE, 'half': CATEGORICAL[0], 'ema01': CATEGORICAL[1], 'ema09': CATEGORICAL[2], 'uniform': CATEGORICAL[3],
          'attention': CATEGORICAL[6], 'block2': CATEGORICAL[4], 'block4': CATEGORICAL[5]}
# W&B final validation loss for both trained rates; d8-block2 at 0.0045 never reached W&B, so its value is the 512-sequence probe loss.
WANDB = {'d8': {.003: 2.9265}, 'rnn': {.002: 3.516},
         'd8-half': {.0015: 2.9549, .003: 2.9354}, 'd8-ema01': {.0015: 6.8747, .003: 6.8747},
         'd8-ema09': {.0015: 2.9578, .003: 2.9497}, 'd8-uniform': {.0015: 2.9467, .003: 2.9253},
         'd8-attention': {.0015: 2.9190, .003: 2.8986}, 'd8-block2': {.0045: 2.9131, .009: 3.5842},
         'd8-block4': {.0015: 2.9365, .003: 2.9097},
         'rnn-half': {.001: 3.5037, .002: 3.5600}, 'rnn-ema01': {.001: 3.5726, .002: 3.5773},
         'rnn-ema09': {.001: 3.5066, .002: 3.5199}, 'rnn-uniform': {.001: 3.5717, .002: 3.5631},
         'rnn-attention': {.001: 3.5491, .002: 3.5369}, 'rnn-block2': {.001: 3.5159, .002: 3.5896},
         'rnn-block4': {.001: 3.5173, .002: 3.5802}}
TITLES = {}  # filename -> takeaway, filled in after looking at the data


def measurements():
    return {p.stem: json.loads(p.read_text()) for p in MEASUREMENTS.glob('*.json')}


def name(family, variant):
    return family if not variant else f'{family}-{variant}'


def title(fig, filename, fallback):
    fig.suptitle(TITLES.get(filename, fallback), fontweight='bold', color=INK, fontsize=11)


def depth_rows(run, checkpoint, objective='full'):
    return run[checkpoint][objective]['depth']


def column(rows, key):
    return np.array([r.get(key, np.nan) for r in rows], float)


STATS = (('activation_rms', 'full', 'activation RMS', True),
         ('gradient_rms', 'full', 'gradient RMS for L', True),
         ('gradient_rms', 'last', 'gradient RMS for L_T', True),
         ('raw_branch_input_ratio', 'full', 'branch output RMS ÷ input RMS', True),
         ('adjacent_cosine', 'full', 'cosine with the previous depth point', False))


def depth_grid(data, family, part, filename):
    """Final-checkpoint depth statistics for one family and part, plus the gradient ratio over training."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2))
    variants = [v for v in PARTS[part] if name(family, v) in data]
    for ax, (key, objective, ylabel, log) in zip(axes.flat, STATS):
        for v in variants:
            rows = depth_rows(data[name(family, v)], 'final', objective)
            values = column(rows, key)
            ax.plot(np.arange(len(values)), values, color=COLORS[v], lw=1.8, label=LABELS[v])
        if log:
            ax.set_yscale('log')
        ax.set_ylabel(ylabel)
        label_depth(ax, len(values))
        ax.set_xlabel(DEPTH_XLABEL)
    # Bottom-to-top gradient ratio at each checkpoint (point 1 = input to the second sublayer).
    ax = axes.flat[5]
    width = .8 / len(variants)
    for i, v in enumerate(variants):
        ratios = []
        for checkpoint in ('init', 'step 100', 'final'):
            g = column(depth_rows(data[name(family, v)], checkpoint), 'gradient_rms')
            ratios.append(g[2] / g[-1] if g[-1] > 0 else np.nan)
        ax.bar(np.arange(3) + (i - (len(variants) - 1) / 2) * width, ratios, width=width, color=COLORS[v])
    ax.axhline(1, color=INK, lw=.8, ls='--')
    ax.set_yscale('log')
    ax.set_xticks(range(3), ['init', 'step 100', 'final'])
    ax.set_ylabel('gradient RMS: bottom ÷ top (L)')
    ax.grid(axis='x', visible=False)
    axes.flat[0].legend(fontsize=8)
    title(fig, filename, f'{FAMILIES[family]}, P3({part}): depth statistics at the final checkpoint')
    save(fig, filename, OUTPUT)


def routers(run, checkpoint='final'):
    routes = run[checkpoint].get('routing', {})
    return [routes[k] for k in sorted(routes, key=lambda k: int(re.search(r'routers\.(\d+)', k).group(1)))]


def routing_heatmap(ax, weights, heading, block_size=1):
    width = max(len(w) for w in weights)
    grid = np.full((len(weights), width), np.nan)
    for j, w in enumerate(weights):
        grid[j, :len(w)] = w
    image = ax.imshow(grid, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    count = len(weights)
    ax.set_yticks(range(count), [f'sublayer {j}' if j < count - 1 else 'output read' for j in range(count)], fontsize=7)
    if block_size == 1:
        ax.set_xticks(range(width), ['emb'] + [str(i) for i in range(1, width)], fontsize=7)
        ax.set_xlabel('source (emb, then sublayer outputs, oldest → newest)', fontsize=8)
    else:
        ax.set_xticks(range(width), ['emb'] + [f'block {i}' for i in range(1, width)], fontsize=7, rotation=45)
        ax.set_xlabel('source (emb, completed blocks, current partial block)', fontsize=8)
    ax.set_title(heading, fontsize=10)
    ax.grid(False)
    return image


def routing_full(data, filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    for ax, family in zip(axes, FAMILIES):
        image = routing_heatmap(ax, routers(data[name(family, 'attention')]), FAMILIES[family])
    fig.colorbar(image, ax=axes, shrink=.8, label='mean routing weight')
    title(fig, filename, 'P3(b) full attention residuals: mean routing weight per sublayer (final)')
    fig.savefig(OUTPUT / filename, bbox_inches='tight')
    plt.close(fig)
    print(OUTPUT / filename)


def routing_over_training(data, filename):
    """How the routing weights move from init (uniform, zero queries) through training."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, family in zip(axes, FAMILIES):
        for ax, checkpoint in zip(row, ('init', 'step 100', 'final')):
            image = routing_heatmap(ax, routers(data[name(family, 'attention')], checkpoint),
                                    f'{FAMILIES[family]}, {checkpoint}')
    fig.colorbar(image, ax=axes, shrink=.6, label='mean routing weight')
    title(fig, filename, 'P3(b) full attention residuals: routing weights over training')
    fig.savefig(OUTPUT / filename, bbox_inches='tight')
    plt.close(fig)
    print(OUTPUT / filename)


def routing_blocked(data, filename):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row, family in zip(axes, FAMILIES):
        for ax, (variant, size) in zip(row, (('block2', 2), ('block4', 4))):
            image = routing_heatmap(ax, routers(data[name(family, variant)]), f'{FAMILIES[family]}, {LABELS[variant]}', size)
    fig.colorbar(image, ax=axes, shrink=.6, label='mean routing weight')
    title(fig, filename, 'P3(c) blocked attention residuals: mean routing weight per sublayer (final)')
    fig.savefig(OUTPUT / filename, bbox_inches='tight')
    plt.close(fig)
    print(OUTPUT / filename)


def loss_bars(data, filename):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, family in zip(axes, FAMILIES):
        base = data[family]['val_loss']
        variants = [v for v in LABELS if v and name(family, v) in data]
        y = np.arange(len(variants))[::-1]
        deltas = [data[name(family, v)]['val_loss'] - base for v in variants]
        shown = np.clip(deltas, -.1, .2)
        ax.barh(y, shown, height=.6, color=[COLORS[v] for v in variants])
        for yi, d, s in zip(y, deltas, shown):
            ax.text(s + (.004 if s >= 0 else -.004), yi, f'{d:+.3f}', va='center', fontsize=8,
                    ha='left' if s >= 0 else 'right')
        ax.axvline(0, color=INK, lw=.8)
        ax.set_yticks(y, [LABELS[v] for v in variants], fontsize=8)
        ax.set_xlim(-.1, .26)
        ax.set_xlabel(f'val loss − additive baseline ({base:.3f}); bars clipped at +0.2')
        ax.set_title(FAMILIES[family])
        ax.grid(axis='y', visible=False)
    title(fig, filename, 'P3: loss change vs the additive baseline, each variant at its best trained LR')
    save(fig, filename, OUTPUT)


EXTENSIONS = {'d8-half': {.006: 2.9393}, 'd8-ema09': {.006: 2.9712}, 'd8-uniform': {.006: 2.9409}}


def loss_wandb(filename):
    """Final W&B val loss at every trained LR (lines), each family against its additive baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, family in zip(axes, FAMILIES):
        base = list(WANDB[family].values())[0]
        ax.axhline(base, color=BASE, lw=1.4, ls='--', label=f'additive baseline: {base:.3f}')
        for v in [v for v in LABELS if v]:
            runs = {**WANDB[name(family, v)], **EXTENSIONS.get(name(family, v), {})}
            runs = {lr: loss for lr, loss in sorted(runs.items()) if loss is not None}
            best = min(runs.values())
            ax.plot(list(runs), list(runs.values()), color=COLORS[v], marker='o', ms=5,
                    label=f'{LABELS[v]}: {best:.3f} ({best - base:+.3f})')
        ax.set_xscale('log')
        ax.set_ylim(base - .06, base + .1)
        ax.set_xlabel('learning rate')
        ax.set_ylabel('final val loss (W&B)')
        ax.set_title(FAMILIES[family])
        ax.legend(fontsize=7, loc='upper left')
    title(fig, filename, 'P3: final val loss at each trained LR (legend: best, and change vs additive)')
    save(fig, filename, OUTPUT)


def loss_table(data):
    lines = ['| model | variant | LRs trained (W&B final val loss) | best LR | val loss (512 seqs) | Δ vs additive |',
             '|---|---|---|---|---|---|']
    for family in FAMILIES:
        base = data[family]['val_loss']
        for v in LABELS:
            key = name(family, v)
            if key not in data:
                continue
            trained = ', '.join(f'{lr:g}: ' + ('—' if loss is None else f'{loss:.4f}') for lr, loss in WANDB[key].items())
            run = data[key]
            lines.append(f"| {FAMILIES[family]} | {LABELS[v]} | {trained} | {run['lr']:g} | {run['val_loss']:.4f} | "
                         f"{run['val_loss'] - base:+.4f} |")
    if 'd8-block2@0.009' in data:
        lines += ['', f"d8 blocked B = 2 at LR 0.009: val loss (512 seqs) {data['d8-block2@0.009']['val_loss']:.4f}."]
    (OUTPUT / 'losses.md').write_text('\n'.join(lines) + '\n')
    print(OUTPUT / 'losses.md')


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = measurements()
    loss_wandb('1_loss.png')
    if all(f in data for f in FAMILIES):
        loss_table(data)
    for family in FAMILIES:
        for part in PARTS:
            if family in data and sum(name(family, v) in data for v in PARTS[part]) > 1:
                depth_grid(data, family, part, f'{part}_{family}_depth.png')
    if all(name(f, 'attention') in data for f in FAMILIES):
        routing_full(data, 'b_routing.png')
        routing_over_training(data, 'b_routing_training.png')
    if all(name(f, v) in data for f in FAMILIES for v in ('block2', 'block4')):
        routing_blocked(data, 'c_routing.png')


if __name__ == '__main__':
    main()
