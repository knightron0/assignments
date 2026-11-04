"""P1 plots from the cached W&B histories (run p1_fetch first).

    uv run python -m experiments.a4_stability.p1_plots

Writes PNGs and an outcomes table to results/a4_stability/p1_plots/.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .p1_fetch import OUTPUT as RUNS
from .sweeps import learning_rates

PLOTS = Path('results/a4_stability/p1_plots')
VARIANTS = ['d8', 'd8-no-residual', 'd8-no-prenorm', 'd8-no-qknorm', 'd8-no-rope',
            'd8-no-final-norm', 'd8-postnorm', 'd8-gated-residual', 'd8-gated-attention', 'd8-gated-norm']
LAYERS = 8
COLLAPSE = 6.  # Final loss near the unigram entropy (~6.87): trained, but learned nothing past token frequency.

INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BASE, SERIES, BAD = '#9b9a95', '#2a78d6', '#d03b3b'
LR_RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']  # ordinal: .25x ... 4x center
TIME_RAMP = ['#86b6ef', '#2a78d6', '#104281']

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'axes.titlecolor': INK, 'axes.titlesize': 10,
    'axes.titleweight': 'bold', 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'font.size': 9, 'lines.linewidth': 2, 'legend.frameon': False,
    'figure.dpi': 150})


def load(root=RUNS):
    runs = {}
    for path in root.glob('*.json'):
        run = json.loads(path.read_text())
        runs[run['variant'], run['lr']] = run
    return runs


def outcome(run):
    loss = run['final_val_loss']
    if run['state'] != 'finished' or loss is None or not math.isfinite(loss):
        return 'diverged'
    return 'collapsed' if loss > COLLAPSE else 'ok'


def best(runs, variant):
    candidates = [(runs[variant, lr]['final_val_loss'], lr) for lr in learning_rates(variant)
                  if outcome(runs[variant, lr]) == 'ok']
    candidates = candidates or [(runs[variant, lr]['final_val_loss'], lr) for lr in learning_rates(variant)
                                if outcome(runs[variant, lr]) == 'collapsed']
    return min(candidates) if candidates else (None, None)


def series(run, key):
    values = run['series'].get(key, [])
    return np.array([s for s, _ in values], float), np.array([v if v is not None else np.nan for _, v in values], float)


def at_step(run, key, step):
    steps, values = series(run, key)
    if not len(steps):
        return np.nan
    return values[np.argmin(np.abs(steps - step))]


def stream_keys():
    """Residual-stream states: embedding output, then each sublayer's output (17 states)."""
    keys = ['model.embed_tokens']
    for i in range(LAYERS):
        keys += [f'model.layers.{i}.attention', f'model.layers.{i}.mlp']
    return [f'logging/rms/{k}/activation' for k in keys]


def branch_keys():
    return [f'logging/rms/model.layers.{i}.{part}.operation/activation'
            for i in range(LAYERS) for part in ('attention', 'mlp')]


def depth_labels():
    return ['emb'] + [f'{i}{p}' for i in range(LAYERS) for p in ('A', 'M')]


def grid(n, cols=5, width=3.1, height=2.4):
    rows = math.ceil(n/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*width, rows*height), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


def save(fig, name, root=PLOTS):
    fig.tight_layout()
    fig.savefig(root/name, bbox_inches='tight')
    plt.close(fig)
    print(root/name)


def outcomes_table(runs):
    lines = ['| Variant | ' + ' | '.join(f'{f:g}x' for f in (.25, .5, 1, 2, 4)) + ' | Best | Best LR | At edge? |',
             '|---|' + '---|'*8]
    for variant in VARIANTS:
        rates = learning_rates(variant)
        cells = []
        for lr in rates:
            run, kind = runs[variant, lr], outcome(runs[variant, lr])
            cell = f"{run['final_val_loss']:.4f}" if kind == 'ok' else (
                f"collapsed ({run['final_val_loss']:.2f})" if kind == 'collapsed' else
                f"diverged (step {int(series(run, 'train_loss')[0][-1])})")
            cells.append(f'{lr:g}: {cell}')
        loss, lr = best(runs, variant)
        losses = [runs[variant, r]['final_val_loss'] for r in rates if outcome(runs[variant, r]) != 'diverged']
        flat = max(losses) - min(losses) < 1e-3
        edge = 'flat' if flat else ('low edge' if lr == rates[0] else 'high edge' if lr == rates[-1] else 'no')
        lines.append(f'| {variant} | ' + ' | '.join(cells) + f' | {loss:.4f} | {lr:g} | {edge} |')
    (PLOTS/'p1_outcomes.md').write_text('\n'.join(lines)+'\n')
    print(PLOTS/'p1_outcomes.md')


def plot_best_loss(runs):
    rows = sorted(((best(runs, v)[0], v) for v in VARIANTS), reverse=True)
    base = best(runs, 'd8')[0]
    fig, (ax, inset) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw=dict(width_ratios=(3, 1.1)))
    shown = [(loss, v) for loss, v in rows if loss < 4]
    y = np.arange(len(shown))
    ax.barh(y, [loss - 2.85 for loss, _ in shown], left=2.85, height=.6,
            color=[BASE if v == 'd8' else SERIES for _, v in shown])
    ax.axvline(base, color=MUTED, lw=1, ls='--')
    ax.set_yticks(y, [v for _, v in shown])
    for yi, (loss, v) in zip(y, shown):
        ax.text(loss + .004, yi, f'{loss:.3f}  ({loss-base:+.3f})', va='center', color=INK, fontsize=8)
    ax.set_xlim(2.85, max(l for l, _ in shown) + .07)
    ax.set_xlabel('best final validation loss over the 5 supplied LRs (axis starts at 2.85)')
    ax.set_title('Best validation loss per architecture (dashed: baseline d8)')
    ax.grid(axis='y', visible=False)
    # Off-scale variants get their own panel so they don't flatten the comparison.
    off = [(loss, v) for loss, v in rows if loss >= 4]
    inset.barh(range(len(off)), [l for l, _ in off], height=.6, color=BAD)
    inset.set_yticks(range(len(off)), [v for _, v in off])
    for i, (loss, _) in enumerate(off):
        inset.text(loss/2, i, f'{loss:.2f}', va='center', ha='center', color='white', fontsize=8)
    inset.set_title('Off scale (all 5 LRs fail)')
    inset.set_xlabel('validation loss')
    inset.grid(axis='y', visible=False)
    save(fig, 'p1_best_loss.png')


def plot_lr_sweeps(runs):
    fig, axes = grid(len(VARIANTS) - 1)
    base_rates = learning_rates('d8')
    base_loss = [runs['d8', lr]['final_val_loss'] for lr in base_rates]
    for ax, variant in zip(axes, VARIANTS[1:]):
        ax.plot(base_rates, base_loss, color=BASE, marker='o', ms=4, label='d8 baseline')
        rates = learning_rates(variant)
        ok = [(lr, runs[variant, lr]['final_val_loss']) for lr in rates if outcome(runs[variant, lr]) == 'ok']
        if ok:
            ax.plot(*zip(*ok), color=SERIES, marker='o', ms=5, label=variant)
        top = 3.45
        for lr in rates:
            run, kind = runs[variant, lr], outcome(runs[variant, lr])
            loss = run['final_val_loss']
            if kind != 'ok' or loss > top:
                text = 'div.' if kind == 'diverged' else f'{loss:.2f}'
                ax.scatter([lr], [top], marker='^' if kind == 'ok' else 'x', s=36, color=BAD, zorder=3)
                ax.annotate(text, (lr, top), textcoords='offset points', xytext=(0, -12),
                            ha='center', fontsize=7, color=BAD)
        b_loss, b_lr = best(runs, variant)
        if b_lr:
            ax.scatter([b_lr], [b_loss], s=70, facecolor='none', edgecolor=INK, lw=1.2, zorder=4)
        ax.set_xscale('log')
        ax.set_ylim(2.88, top + .03)
        ax.set_title(variant)
        ax.set_xlabel('learning rate')
        ax.set_ylabel('final val loss')
    fig.legend(handles=[plt.Line2D([], [], color=BASE, marker='o', label='d8 baseline'),
                            plt.Line2D([], [], color=SERIES, marker='o', label='variant'),
                            plt.Line2D([], [], color=BAD, marker='x', ls='', label='diverged / collapsed'),
                            plt.Line2D([], [], color=BAD, marker='^', ls='', label='off scale (finite)'),
                            plt.Line2D([], [], color=INK, marker='o', mfc='none', ls='', label='best LR')],
               fontsize=8, loc='lower right', bbox_to_anchor=(.97, .15))
    fig.suptitle('Final validation loss vs LR: each variant against the d8 baseline', fontweight='bold', color=INK)
    save(fig, 'p1_lr_sweeps.png')


def plot_val_curves(runs):
    fig, axes = grid(len(VARIANTS))
    for ax, variant in zip(axes, VARIANTS):
        for color, lr in zip(LR_RAMP, learning_rates(variant)):
            steps, loss = series(runs[variant, lr], 'val_loss')
            ax.plot(steps, loss, color=color, lw=1.5, label=f'{lr:g}')
            if outcome(runs[variant, lr]) == 'diverged' and len(steps):
                ax.scatter(steps[-1:], [min(loss[-1], 4.4)], marker='x', color=BAD, s=30, zorder=3)
        if variant != 'd8-no-residual':  # its losses sit near 6.9; let it autoscale
            ax.set_ylim(2.85, 4.5)
        ax.set_title(variant)
        ax.set_xlabel('step')
        ax.set_ylabel('val loss')
        ax.legend(title='LR (light→dark)', fontsize=6, title_fontsize=6, loc='upper right')
    fig.suptitle('Validation loss during training, all five LRs (× = run stopped)', fontweight='bold', color=INK)
    save(fig, 'p1_val_curves.png')


def depth_profile(run, keys, step):
    return np.array([at_step(run, k, step) for k in keys])


def plot_stream_rms(runs):
    fig, axes = grid(len(VARIANTS))
    x, labels = np.arange(2*LAYERS + 1), depth_labels()
    base_run = runs['d8', best(runs, 'd8')[1]]
    last = series(base_run, stream_keys()[0])[0][-1]
    for ax, variant in zip(axes, VARIANTS):
        loss, lr = best(runs, variant)
        lr = lr or learning_rates(variant)[2]
        run = runs[variant, lr]
        if variant != 'd8':
            ax.plot(x, depth_profile(base_run, stream_keys(), last), color=BASE, lw=1.5, label='d8, end')
        for color, step, name in zip(TIME_RAMP, (100, last//2, last), ('step 100', 'mid', 'end')):
            ax.plot(x, depth_profile(run, stream_keys(), step), color=color, label=name)
        ax.set_yscale('log')
        ax.set_xticks(x[::2], labels[::2], fontsize=6)
        ax.set_title(f'{variant} (lr {lr:g})')
        ax.set_xlabel('stream state (A: after attn, M: after MLP)', fontsize=7)
        ax.set_ylabel('activation RMS')
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.suptitle('Residual-stream activation RMS across depth (training-time logger, best LR)',
                 fontweight='bold', color=INK)
    save(fig, 'p1_stream_rms_depth.png')


def plot_branch_ratio(runs):
    fig, axes = grid(len(VARIANTS))
    x = np.arange(2*LAYERS)
    labels = depth_labels()[1:]
    for ax, variant in zip(axes, VARIANTS):
        loss, lr = best(runs, variant)
        lr = lr or learning_rates(variant)[2]
        run = runs[variant, lr]
        last = series(run, stream_keys()[0])[0][-1]
        for color, step, name in zip(TIME_RAMP, (100, last//2, last), ('step 100', 'mid', 'end')):
            inputs = depth_profile(run, stream_keys()[:-1], step)
            ax.plot(x, depth_profile(run, branch_keys(), step)/inputs, color=color, label=name)
        ax.axhline(1, color=MUTED, lw=.8, ls=':')
        ax.set_yscale('log')
        ax.set_xticks(x[::2], labels[::2], fontsize=6)
        ax.set_title(f'{variant} (lr {lr:g})')
        ax.set_xlabel('sublayer')
        ax.set_ylabel('RMS(F) / RMS(stream in)')
    axes[0].legend(fontsize=7)
    fig.suptitle('Branch update size relative to its input stream (≪1: small edits; ≈1: rewrites)',
                 fontweight='bold', color=INK)
    save(fig, 'p1_branch_ratio_depth.png')


def sublayer_param_grad(run, i, part, step):
    prefix = f'logging/rms/model.layers.{i}.{part}.operation.'
    keys = [k for k in run['series'] if k.startswith(prefix) and k.endswith('_proj/gradient')]
    return np.sqrt(np.nanmean([at_step(run, k, step)**2 for k in keys]))


def plot_param_grads(runs):
    fig, axes = grid(len(VARIANTS))
    x = np.arange(2*LAYERS)
    labels = depth_labels()[1:]
    for ax, variant in zip(axes, VARIANTS):
        loss, lr = best(runs, variant)
        lr = lr or learning_rates(variant)[2]
        run = runs[variant, lr]
        last = series(run, stream_keys()[0])[0][-1]
        for color, step, name in zip(TIME_RAMP, (100, last//2, last), ('step 100', 'mid', 'end')):
            ax.plot(x, [sublayer_param_grad(run, i, p, step) for i in range(LAYERS) for p in ('attention', 'mlp')],
                    color=color, label=name)
        ax.set_yscale('log')
        ax.set_xticks(x[::2], labels[::2], fontsize=6)
        ax.set_title(f'{variant} (lr {lr:g})')
        ax.set_xlabel('sublayer')
        ax.set_ylabel('param-grad RMS')
    axes[0].legend(fontsize=7)
    fig.suptitle('Parameter-gradient RMS per sublayer (pre-clip; a proxy; activation gradients need checkpoints)',
                 fontweight='bold', color=INK)
    save(fig, 'p1_param_grad_depth.png')


def plot_instability(runs):
    cases = [('d8-no-prenorm', (.001, .002, .004)), ('d8-postnorm', (.002, .004)),
             ('d8-no-qknorm', (.003, .006, .012)), ('d8-no-residual', (.003,))]
    fig, axes = plt.subplots(3, len(cases), figsize=(4*len(cases), 7.5), sharex='col', squeeze=False)
    metrics = [('train_loss', 'train loss', False),
               ('max_stream', 'max stream RMS over depth', True),
               ('logging/rms/global/gradient', 'global grad RMS (pre-clip)', True)]
    for col, (variant, rates) in enumerate(cases):
        ramp = LR_RAMP[-len(rates):] if len(rates) > 1 else [SERIES]
        for color, lr in zip(ramp, rates):
            run = runs[variant, lr]
            for row, (key, label, log) in enumerate(metrics):
                ax = axes[row, col]
                if key == 'max_stream':
                    stacked = [series(run, k) for k in stream_keys()]
                    steps = stacked[0][0]
                    values = np.nanmax(np.stack([v[:len(steps)] for _, v in stacked]), 0)
                else:
                    steps, values = series(run, key)
                    if key == 'train_loss' and len(values) > 25:  # smooth the noisy per-step loss
                        values = np.convolve(values, np.ones(25)/25, mode='valid')
                        steps = steps[12:12+len(values)]
                ax.plot(steps, values, color=color, lw=1.4, label=f"lr {lr:g} ({ {'ok': 'finished'}.get(outcome(run), outcome(run)) }, val {run['final_val_loss'] or float('nan'):.2f})")
                ax.set_yscale('log' if log else 'linear')
                ax.set_ylabel(label)
        axes[0, col].set_title(variant)
        axes[0, col].legend(fontsize=7)
        axes[-1, col].set_xlabel('step')
    fig.suptitle('Anatomy of the failing runs: loss, stream size, gradient size over training',
                 fontweight='bold', color=INK)
    save(fig, 'p1_instability.png')


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    runs = load()
    outcomes_table(runs)
    plot_best_loss(runs)
    plot_lr_sweeps(runs)
    plot_val_curves(runs)
    plot_stream_rms(runs)
    plot_branch_ratio(runs)
    plot_param_grads(runs)
    plot_instability(runs)


if __name__ == '__main__':
    main()
