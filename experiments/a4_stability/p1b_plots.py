"""P1(b) plots: each variant against the d8 baseline (run p1_fetch and p1b_measure first).

    uv run python -m experiments.a4_stability.p1b_plots

1 = loss (W&B sweep), 2 = depth (repeat of (a)(i)), 3 = sequence (repeat of (a)(ii), raw shares
at late queries), 4 = failed runs (W&B training logs), 5 = summary table.
Writes results/a4_stability/p1b/*.png and summary.md.
"""
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .p1_fetch import RESULTS
from .p1_plots import BAD, BASE, INK, LR_RAMP, MUTED, SERIES, best, load, outcome, save, series, stream_keys
from .p1a_attention import LATE
from .p1b_measure import BEST_LR, OUTPUT as MEASUREMENTS
from .sweeps import learning_rates

OUTPUT = MEASUREMENTS.parent
VARIANTS = [v for v in BEST_LR if v != 'd8']
NAMES = {'d8-no-residual': '(i) no residual', 'd8-no-prenorm': '(ii) no pre-norm', 'd8-no-qknorm': '(iii) no QK-norm',
         'd8-no-rope': '(iv) no RoPE', 'd8-no-final-norm': '(v) no final norm', 'd8-postnorm': '(vi) post-norm',
         'd8-gated-residual': '(vii) gated residual', 'd8-gated-attention': '(viii) gated attention',
         'd8-gated-norm': '(ix) gated norm', 'd8': 'd8 baseline (P1(a))'}


def measurements():
    return {v: json.loads((MEASUREMENTS / f'{v}.json').read_text()) for v in BEST_LR}


def title(fig, text):
    fig.suptitle(text, fontweight='bold', color=INK, fontsize=11)


# --------------------------------------------------------------------------- 1. loss

def plot_best_loss(runs):
    """Validation loss during training at each variant's best LR, against the P1(a) d8 baseline."""
    base = best(runs, 'd8')[0]
    base_steps, base_curve = series(runs['d8', BEST_LR['d8']], 'val_loss')
    fig, axes = plt.subplots(3, 3, figsize=(13, 9.5), sharex=True)
    for ax, v in zip(axes.flat, VARIANTS):
        loss = best(runs, v)[0]
        ax.plot(base_steps, base_curve, color=BASE, lw=1.4, label=f'd8 baseline (P1(a), lr 0.003): {base:.3f}')
        ax.plot(*series(runs[v, BEST_LR[v]], 'val_loss'), color=SERIES, lw=1.6, label='variant at its best LR')
        if v != 'd8-no-residual':  # its curve sits near 6.9: let it autoscale
            ax.set_ylim(2.85, 3.8)
        ax.set_title(f"{NAMES[v]}, lr {BEST_LR[v]:g}: {loss:.3f} ({loss - base:+.3f})", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel('training step')
    for ax in axes[:, 0]:
        ax.set_ylabel('val loss')
    fig.legend(handles=axes.flat[0].get_lines(), loc='lower center', ncol=2, fontsize=9, bbox_to_anchor=(.5, -.02))
    title(fig, '1a. Validation loss during training, each variant at its best LR (gray = d8 baseline from P1(a))\n'
               'panel titles: best final val loss (difference from d8); only gated residual and gated attention beat d8')
    save(fig, '1a_best_loss.png', OUTPUT)


CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']


def plot_loss_overview(runs):
    """All variants' validation curves on one chart (no residual is off scale and left out)."""
    shown = [v for v in VARIANTS if v != 'd8-no-residual']
    colors = dict(zip(shown, CATEGORICAL))  # fixed by variant, not by rank
    base = best(runs, 'd8')[0]
    fig, (full, zoom) = plt.subplots(1, 2, figsize=(18, 5.6), gridspec_kw=dict(width_ratios=(1.25, 1)))
    order = sorted(shown + ['d8'], key=lambda v: best(runs, v)[0])
    for ax in (full, zoom):
        for v in order:
            steps, loss = series(runs[v, BEST_LR[v]], 'val_loss')
            final = best(runs, v)[0]
            ax.plot(steps, loss, color=INK if v == 'd8' else colors[v], lw=2.2 if v == 'd8' else 1.6,
                    ls='--' if v == 'd8' else '-',
                    label=f"{NAMES[v]} (lr {BEST_LR[v]:g}): {final:.3f}" + ('' if v == 'd8' else f' ({final - base:+.3f})'))
        ax.set_xlabel('training step')
        ax.set_ylabel('val loss')
    full.set_ylim(2.85, 3.8)
    full.set_title('Whole run', fontsize=10)
    zoom.set_xlim(6000, steps[-1] + 100)
    zoom.set_ylim(2.88, 3.2)
    zoom.set_title('Last third of training (zoomed)', fontsize=10)
    zoom.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1), title='best final val loss (vs d8), sorted',
                title_fontsize=8)
    title(fig, '1c. Validation loss overview: every variant at its best LR (dashed = d8 baseline from P1(a))\n'
               f"(i) no residual is off scale: flat at {best(runs, 'd8-no-residual')[0]:.2f} for the whole run")
    save(fig, '1c_loss_overview.png', OUTPUT)


def plot_lr_sweeps(runs):
    fig, axes = plt.subplots(3, 3, figsize=(13, 9.5), sharey=True)
    base_rates = learning_rates('d8')
    base_loss = [runs['d8', lr]['final_val_loss'] for lr in base_rates]
    top = 3.45
    for ax, v in zip(axes.flat, VARIANTS):
        ax.plot(base_rates, base_loss, color=BASE, marker='o', ms=4, lw=1.6, label='d8 baseline')
        rates = learning_rates(v)
        ok = [(lr, runs[v, lr]['final_val_loss']) for lr in rates if outcome(runs[v, lr]) == 'ok']
        if ok:
            ax.plot(*zip(*[(lr, min(loss, top)) for lr, loss in ok]), color=SERIES, marker='o', ms=5, lw=1.8,
                    label='variant')
        for lr in rates:
            run, kind = runs[v, lr], outcome(runs[v, lr])
            loss = run['final_val_loss']
            if kind != 'ok' or loss > top:
                ax.scatter([lr], [top], marker='x', s=40, color=BAD, zorder=3)
                ax.annotate('diverged' if kind == 'diverged' else f'{loss:.2f}', (lr, top), textcoords='offset points',
                            xytext=(0, -13), ha='center', fontsize=7, color=BAD)
        ax.scatter([BEST_LR[v]], [min(best(runs, v)[0], top)], s=80, facecolor='none', edgecolor=INK, lw=1.2, zorder=4)
        ax.set_xscale('log')
        ax.set_ylim(2.88, top + .04)
        ax.set_title(NAMES[v], fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel('learning rate')
    for ax in axes[:, 0]:
        ax.set_ylabel('final val loss')
    handles = [plt.Line2D([], [], color=BASE, marker='o', label='d8 baseline'),
               plt.Line2D([], [], color=SERIES, marker='o', label='variant'),
               plt.Line2D([], [], color=BAD, marker='x', ls='', label='diverged / off scale'),
               plt.Line2D([], [], color=INK, marker='o', mfc='none', ls='', label='best LR')]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(.5, -.02))
    title(fig, '1b. Final validation loss at each supplied learning rate (gray = d8 baseline)')
    save(fig, '1b_lr_sweeps.png', OUTPUT)


# --------------------------------------------------------------------------- 2. depth

def label_depth(ax, count):
    """Major ticks at the end of each layer (after its MLP); attention updates sit halfway between."""
    ax.set_xticks(range(0, count, 2), ['emb'] + [f'L{i}' for i in range((count - 1)//2)], fontsize=8)
    ax.set_xticks(range(1, count, 2), minor=True)


DEPTH_XLABEL = 'layer'


def plot_depth(data, objective, key, ylabel, heading, filename):
    fig, axes = plt.subplots(3, 3, figsize=(13, 9.5), sharex=True)
    base = data['d8']['depth']['final'][objective][key]
    x = np.arange(len(base))
    for ax, v in zip(axes.flat, VARIANTS):
        ax.plot(x, base, color=BASE, lw=1.6, label='d8 baseline')
        ax.plot(x, data[v]['depth']['final'][objective][key], color=SERIES, lw=1.8, label='variant')
        ax.set_yscale('log')
        ax.set_title(NAMES[v], fontsize=10)
        label_depth(ax, len(x))
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    for ax in axes[-1]:
        ax.set_xlabel(DEPTH_XLABEL)
    fig.legend(handles=axes.flat[0].get_lines(), loc='lower center', ncol=2, fontsize=9, bbox_to_anchor=(.5, -.02))
    title(fig, f'{heading} (final checkpoints, same 32 held-out sequences; gray = d8 baseline)')
    save(fig, filename, OUTPUT)


def plot_gradients(data):
    """Gradient RMS for L and L_T in one panel per variant, against d8 (dashed)."""
    losses = (('full', 'L', SERIES), ('last', 'L_T', '#eb6834'))
    fig, axes = plt.subplots(3, 3, figsize=(13, 9.5), sharex=True)
    x = np.arange(len(data['d8']['depth']['final']['full']['gradient_rms']))
    for ax, v in zip(axes.flat, VARIANTS):
        for objective, name, color in losses:
            ax.plot(x, data['d8']['depth']['final'][objective]['gradient_rms'], color=color, lw=1.3, ls='--',
                    alpha=.7, label=f'{name}, d8 baseline (P1(a))')
            ax.plot(x, data[v]['depth']['final'][objective]['gradient_rms'], color=color, lw=1.8,
                    label=f'{name}, variant')
        ax.set_yscale('log')
        ax.set_title(NAMES[v], fontsize=10)
        label_depth(ax, len(x))
    for ax in axes[:, 0]:
        ax.set_ylabel('RMS(∂Loss/∂x)')
    for ax in axes[-1]:
        ax.set_xlabel(DEPTH_XLABEL)
    fig.legend(handles=axes.flat[0].get_lines(), loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(.5, -.02))
    title(fig, '2b. Gradient RMS across depth for L (all tokens) and L_T (last token)\n'
               '(final checkpoints, same 32 held-out sequences; dashed = d8 baseline from P1(a))')
    save(fig, '2b_gradients.png', OUTPUT)


def depth_table(data):
    lines = ['| variant | activation growth, pt 2 → 16 (final) | grad ratio L: init / step 100 / final '
             '| grad ratio L_T: init / step 100 / final |', '|---|---|---|---|']
    for v in BEST_LR:
        depth = data[v]['depth']
        act = depth['final']['full']['activation_rms']
        cells = []
        for objective in ('full', 'last'):
            ratios = [np.array(depth[c][objective]['gradient_rms']) for c in ('init', 'step 100', 'final')]
            cells.append(' / '.join(f'{g[2]/g[-1]:.2f}' for g in ratios))
        lines.append(f'| {NAMES[v]} | {act[-1]/act[2]:.1f}× | {cells[0]} | {cells[1]} |')
    lines += ['', 'Gradient ratio = gradient RMS at point 2 (after layer 0 MLP) ÷ at point 16 (top). '
              '≈1 balanced, ≪1 vanishing toward the bottom, ≫1 growing toward the bottom.']
    (OUTPUT / '2c_depth_table.md').write_text('\n'.join(lines) + '\n')
    print(OUTPUT / '2c_depth_table.md')


# --------------------------------------------------------------------------- 3. sequence

def late_share(values):
    return float(np.mean(np.mean(values, 0)[LATE:]))


SEQUENCE = (('local', 'share on the previous 1–8 tokens',
             lambda layer: float(np.mean(layer['mass_by_lag'], 0)[1:9].sum())),
            ('self', 'share on the current token', lambda layer: late_share(layer['self_mass'])),
            ('sink', 'share on position 0 (BOS)', lambda layer: late_share(layer['first_mass'])))


def plot_sequence(data, key, label, measure, letter):
    order = ['d8'] + VARIANTS
    values = np.array([[measure(layer) for layer in data[v]['attention']['final']['layers']] for v in order])
    fig, ax = plt.subplots(figsize=(10, 5.6))
    image = ax.imshow(values, cmap='Blues', vmin=0, vmax=max(values.max(), .01), aspect='auto')
    for (r, c), value in np.ndenumerate(values):
        ax.text(c, r, f'{value:.0%}', ha='center', va='center', fontsize=8,
                color='white' if value > .6 * values.max() else INK)
    ax.axhline(.5, color=INK, lw=1.5)  # separate the baseline row
    ax.set_xticks(range(values.shape[1]), [f'layer {i}' for i in range(values.shape[1])])
    ax.set_yticks(range(len(order)), [NAMES[v] for v in order])
    ax.grid(False)
    plt.colorbar(image, ax=ax, fraction=.03, pad=.02, format=plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    title(fig, f'3{letter}. Attention {label}, per layer (head average, queries {LATE}–1022, final checkpoints)')
    save(fig, f'3{letter}_{key}.png', OUTPUT)


# --------------------------------------------------------------------------- 4. failed runs

FAILURES = (('d8-no-prenorm', (.001, .002, .004)), ('d8-postnorm', (.002, .004)),
            ('d8-no-qknorm', (.003, .006, .012)), ('d8-no-residual', (.00075, .012)))


def plot_failures(runs):
    fig, axes = plt.subplots(3, len(FAILURES), figsize=(16, 8.5), sharex='col')
    for col, (v, rates) in enumerate(FAILURES):
        for color, lr in zip(LR_RAMP[-len(rates):], rates):
            run = runs[v, lr]
            loss = run['final_val_loss']
            label = f"lr {lr:g}: " + ('diverged' if outcome(run) == 'diverged' else f'val {loss:.2f}')
            steps, train = series(run, 'train_loss')
            if len(train) > 25:
                train = np.convolve(train, np.ones(25)/25, mode='valid')
                steps = steps[12:12 + len(train)]
            axes[0, col].plot(steps, train, color=color, lw=1.4, label=label)
            stacked = [series(run, k) for k in stream_keys()]
            steps = stacked[0][0]
            axes[1, col].plot(steps, np.nanmax(np.stack([s[:len(steps)] for _, s in stacked]), 0), color=color, lw=1.4)
            axes[2, col].plot(*series(run, 'logging/rms/global/gradient'), color=color, lw=1.4)
        axes[0, col].set_title(NAMES[v], fontsize=10)
        axes[0, col].legend(fontsize=7)
        axes[1, col].set_yscale('log')
        axes[2, col].set_yscale('log')
        axes[-1, col].set_xlabel('training step')
    axes[0, 0].set_ylabel('train loss (smoothed)')
    axes[1, 0].set_ylabel('largest stream RMS over depth')
    axes[2, 0].set_ylabel('gradient RMS (pre-clip)')
    title(fig, '4. Runs that diverged or failed to learn: loss, stream size and gradient size during training (W&B)')
    save(fig, '4_failed_runs.png', OUTPUT)


# --------------------------------------------------------------------------- 5. summary

def summary(runs, data):
    base = best(runs, 'd8')[0]
    lines = ['| variant | best LR | Δ best val loss vs d8 | diverged / collapsed LRs | grad ratio L (final) '
             '| top stream RMS | sink share, layers 3–7 | local share, layer 1 |', '|---|---|---|---|---|---|---|---|']
    for v in BEST_LR:
        failed = [f'{lr:g}' for lr in learning_rates(v) if outcome(runs[v, lr]) != 'ok']
        depth = data[v]['depth']['final']['full']
        grads = depth['gradient_rms']
        layers = data[v]['attention']['final']['layers']
        sink = np.mean([late_share(layer['first_mass']) for layer in layers[3:]])
        local = float(np.mean(layers[1]['mass_by_lag'], 0)[1:9].sum())
        lines.append(f"| {NAMES[v]} | {BEST_LR[v]:g} | {best(runs, v)[0] - base:+.3f} | {', '.join(failed) or '—'} "
                     f"| {grads[2]/grads[-1]:.2f} | {depth['activation_rms'][-1]:.3g} | {sink:.0%} | {local:.0%} |")
    (OUTPUT / '5_summary.md').write_text('\n'.join(lines) + '\n')
    print(OUTPUT / '5_summary.md')


def main():
    runs, data = load(RESULTS / 'p1_runs'), measurements()
    plot_best_loss(runs)
    plot_lr_sweeps(runs)
    plot_loss_overview(runs)
    plot_depth(data, 'full', 'activation_rms', 'activation RMS', '2a. Activation RMS across depth', '2a_activation_rms.png')
    plot_gradients(data)
    depth_table(data)
    for letter, (key, label, measure) in zip('abc', SEQUENCE):
        plot_sequence(data, key, label, measure, letter)
    plot_failures(runs)
    summary(runs, data)


if __name__ == '__main__':
    main()
