"""P4 plots from p4_measure results (run p4_measure first).

    uv run python -m experiments.a4_stability.p4_plots

Writes results/a4_stability/p4/{a_loss, a_time, a_gates}.png and a_table.md.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .p1_plots import INK, save
from .p2bf_plots import FLOOR, LAYER_RAMP, floor, layer_lines, reach, time_summary, title
from .p4_measure import OUTPUT as MEASUREMENTS

OUTPUT = MEASUREMENTS.parent
NAMES = {'rnn': 'tanh RNN (supplied)', 'gru': 'GRU', 'gru-no-reset': 'no reset (r = 1)',
         'gru-no-update': 'no update gate (z = 1)', 'gru-fixed-update': 'fixed update (z = 1/2)',
         'gru-const-update': 'learned constant z = σ(b)', 'gru-input-gates': 'gates read only x_t'}
# W&B final losses for runs that synced; the rest come from the probe's validation loss.
# Losses are the probe's 128-sequence validation loss for every model, so they share one test set
# (about 0.055 below the W&B 1,000-sequence loss for every run where both exist).
WANDB = {}


def load():
    runs = {}
    for path in sorted(MEASUREMENTS.glob('*.json')):
        result = json.loads(path.read_text())
        runs[result['variant'], result['lr']] = result
    best = {}
    for (variant, lr), result in runs.items():
        loss = WANDB.get((variant, lr), result['val_loss'])
        if variant not in best or loss < best[variant][0]:
            best[variant] = (loss, lr, result)
    return runs, {v: best[v] for v in NAMES if v in best}


def plot_loss(best):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    names = list(best)
    losses = [best[v][0] for v in names]
    colors = ['#8a8a8a' if v == 'rnn' else '#2a78d6' if v == 'gru' else '#eb6834' for v in names]
    ax.barh(range(len(names)), losses, color=colors, height=.6)
    for i, v in enumerate(names):
        ax.text(losses[i] + .005, i, f'{losses[i]:.3f}  (lr {best[v][1]:g})', va='center', fontsize=9, color=INK)
    ax.set_yticks(range(len(names)), [NAMES[v] for v in names])
    ax.invert_yaxis()
    ax.set_xlim(3.1, max(losses) + .12)
    ax.set_xlabel('best final validation loss (lower is better)')
    title(fig, 'P4(a) Removing or freezing the update gate loses most of the GRU\'s gain over tanh')
    save(fig, 'a_loss.png', OUTPUT)


def plot_time(best):
    """Last-token gradient reach across time, one panel per model (saturation and RMS are in the table)."""
    names = list(best)
    cols = 3
    rows = -(-len(names) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4*rows), squeeze=False, sharex=True, sharey=True)
    for ax, v in zip(axes.flat, names):
        last = best[v][2]['final']['last']['hidden_gradient_rms']
        layer_lines(ax, last, x=np.arange(1, len(last[0])), transform=lambda g: floor(g[::-1][1:]))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(FLOOR / 3, 1e-1)
        ax.set_title(f'{NAMES[v]}  ({best[v][0]:.3f})', fontsize=10)
    for ax in axes.flat[len(names):]:
        ax.axis('off')
    for ax in axes[-1]:
        ax.set_xlabel('tokens before the final token')
    for ax in axes[:, 0]:
        ax.set_ylabel('RMS(∂L_T/∂h_t)')
    handles = [plt.Line2D([], [], color=c, lw=1.5, label=f'layer {i}') for i, c in enumerate(LAYER_RAMP)]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(.5, -.02))
    title(fig, 'P4(a) Only the GRU and the models with learnable z keep the last token\'s gradient\n'
               'alive across the whole sequence (final checkpoints; light → dark = bottom → top layer)')
    save(fig, 'a_time.png', OUTPUT)


def plot_gates(best):
    names = [v for v in best if best[v][2].get('gates')]
    fig, axes = plt.subplots(1, len(names), figsize=(4.2*len(names), 3.8), squeeze=False, sharey=True)
    for ax, v in zip(axes[0], names):
        gates = best[v][2]['gates']
        for label, style in (('update', '-'), ('reset', '--')):
            for i, layer in enumerate(gates.get(label, [])):
                ax.plot(layer['by_position'], color=LAYER_RAMP[i], ls=style, lw=1.3)
        ax.set_title(NAMES[v], fontsize=10)
        ax.set_xlabel('time index t')
        ax.set_ylim(0, 1)
    axes[0, 0].set_ylabel('mean gate value')
    handles = [plt.Line2D([], [], color=INK, ls='-', label='update z (share of new info)'),
               plt.Line2D([], [], color=INK, ls='--', label='reset r')]
    handles += [plt.Line2D([], [], color=c, lw=1.5, label=f'layer {i}') for i, c in enumerate(LAYER_RAMP)]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=8, bbox_to_anchor=(.5, -.06))
    title(fig, 'P4(a) Mean gate values by position (final checkpoints)')
    save(fig, 'a_gates.png', OUTPUT)


def table(runs, best):
    lines = ['| model | LR: loss | saturation (L0-L3) | hidden RMS | L_T reach (tokens) | mean z | z < 0.1 | mean r |',
             '|---|---|---|---|---|---|---|---|']
    for v, (loss, lr, result) in best.items():
        rates = ', '.join(f'{r:g}: {WANDB.get((v, r), runs[v, r]["val_loss"]):.3f}'
                          for (name, r) in sorted(runs) if name == v)
        s = time_summary({v: result}, v)
        gates = result.get('gates', {})
        def gate(label, key):
            return ' / '.join(f'{g[key]:.2f}' for g in gates.get(label, [])) or '-'
        lines.append(f"| {NAMES[v]} | {rates} | {' / '.join(f'{x:.0%}' for x in s['saturation'])} "
                     f"| {' / '.join(f'{x:.2f}' for x in s['hidden_rms'])} | {' / '.join(map(str, s['reach']))} "
                     f"| {gate('update', 'mean')} | {gate('update', 'below_01')} | {gate('reset', 'mean')} |")
    (OUTPUT / 'a_table.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    runs, best = load()
    plot_loss(best)
    plot_time(best)
    plot_gates(best)
    table(runs, best)
