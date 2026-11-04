"""P2 plots from the cached W&B histories (run p1_fetch first; it fetches P1 and P2).

    uv run python -m experiments.a4_stability.p2_plots

Writes PNGs and an outcomes table to results/a4_stability/p2_plots/. Time-index
diagnostics (hidden gradients per t, saturation) need checkpoints, not W&B.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .p1_fetch import RESULTS
from .p1_plots import (BAD, BASE, INK, LR_RAMP, MUTED, SERIES, TIME_RAMP, at_step, best, grid,
                       load, outcome, series)
from . import p1_plots
from .sweeps import learning_rates

PLOTS = RESULTS / 'p2_plots'
PLAIN, MODERN = ['rnn1', 'rnn4'], 'rnn'
ABLATIONS = ['rnn-no-residual', 'rnn-no-prenorm', 'rnn-rope', 'rnn-final-norm', 'rnn-postnorm',
             'rnn-gated-residual', 'rnn-gated-output', 'rnn-gated-norm']
VARIANTS = PLAIN + [MODERN] + ABLATIONS
LAYERS = 4
CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a']  # fixed slot order: blue, orange, aqua

# Each RNN intervention and its transformer counterpart. 'remove' interventions take a
# component away from the baseline; 'add' interventions put one in.
TRANSFER = [('depth residual', 'rnn-no-residual', 'remove', 'd8-no-residual', 'remove'),
            ('pre-norm', 'rnn-no-prenorm', 'remove', 'd8-no-prenorm', 'remove'),
            ('position rotation', 'rnn-rope', 'add', 'd8-no-rope', 'remove'),
            ('final output norm', 'rnn-final-norm', 'add', 'd8-no-final-norm', 'remove'),
            ('post-norm (vs pre-norm)', 'rnn-postnorm', 'add', 'd8-postnorm', 'add'),
            ('gated residual', 'rnn-gated-residual', 'add', 'd8-gated-residual', 'add'),
            ('gated branch output', 'rnn-gated-output', 'add', 'd8-gated-attention', 'add'),
            ('gated norm', 'rnn-gated-norm', 'add', 'd8-gated-norm', 'add')]


def save(fig, name):
    p1_plots.save(fig, name, PLOTS)


def modern(variant):
    return variant not in PLAIN


def stream_keys(variant):
    """Depth-stream states: embedding, then each sublayer output (plain: each RNN layer output)."""
    keys = ['embedding']
    for i in range(1 if variant == 'rnn1' else LAYERS):
        keys += [f'stack.{i}.recurrent', f'stack.{i}.mlp'] if modern(variant) else [f'stack.{i}']
    return [f'logging/rms/{k}/activation' for k in keys]


def depth_labels(variant):
    if not modern(variant):
        return ['emb'] + [f'h{i}' for i in range(1 if variant == 'rnn1' else LAYERS)]
    return ['emb'] + [f'{i}{p}' for i in range(LAYERS) for p in ('R', 'M')]


def best_run(runs, variant):
    loss, lr = best(runs, variant)
    lr = lr or learning_rates(variant)[2]
    return runs[variant, lr], lr


def last_step(run):
    return series(run, 'logging/rms/embedding/activation')[0][-1]


def snapshots(run):
    last = last_step(run)
    return zip(TIME_RAMP, (100, last//2, last), ('step 100', 'mid', 'end'))


def cell(run, kind):
    loss = run['final_val_loss']
    return {'ok': f'{loss:.4f}', 'collapsed': f'collapsed ({loss:.2f})'}.get(
        kind, f"diverged (step {int(series(run, 'train_loss')[0][-1])})")


def outcomes_table(runs):
    lines = ['| Variant | ' + ' | '.join(f'{f:g}x' for f in (.25, .5, 1, 2, 4)) + ' | Best | Best LR | At edge? |',
             '|---|' + '---|'*8]
    for variant in VARIANTS:
        rates = learning_rates(variant)
        cells = [f'{lr:g}: {cell(runs[variant, lr], outcome(runs[variant, lr]))}' for lr in rates]
        loss, lr = best(runs, variant)
        edge = 'low edge' if lr == rates[0] else 'high edge' if lr == rates[-1] else 'no'
        lines.append(f'| {variant} | ' + ' | '.join(cells) + f' | {loss:.4f} | {lr:g} | {edge} |')
    (PLOTS/'p2_outcomes.md').write_text('\n'.join(lines)+'\n')
    print(PLOTS/'p2_outcomes.md')


def plot_plain_and_modern(runs):
    """(a) and (c): rnn1 vs rnn4 sweeps, with the modern RNN for reference."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), gridspec_kw=dict(width_ratios=(1.3, 1, 1, 1)))
    ax = axes[0]
    top = 4.5
    for color, variant in zip(CATEGORICAL, PLAIN + [MODERN]):
        rates = learning_rates(variant)
        ok = [(lr, runs[variant, lr]['final_val_loss']) for lr in rates if outcome(runs[variant, lr]) == 'ok']
        ax.plot(*zip(*ok), color=color, marker='o', ms=5, label=variant)
        for lr in rates:
            if outcome(runs[variant, lr]) != 'ok':
                ax.scatter([lr], [top], marker='x', s=36, color=BAD, zorder=3)
                ax.annotate(f"{runs[variant, lr]['final_val_loss']:.2f}", (lr, top), textcoords='offset points',
                            xytext=(0, -12), ha='center', fontsize=7, color=BAD)
        loss, lr = best(runs, variant)
        ax.scatter([lr], [loss], s=70, facecolor='none', edgecolor=INK, lw=1.2, zorder=4)
        ax.annotate(f'{loss:.3f}', (lr, loss), textcoords='offset points', xytext=(8, -10), fontsize=7, color=INK)
    ax.set_xscale('log')
    ax.set_ylim(3.45, top + .05)
    ax.set_xlabel('learning rate')
    ax.set_ylabel('final val loss')
    ax.set_title('Final val loss vs LR (○ best; × collapsed)')
    ax.legend(fontsize=7, loc='lower left')
    for ax, variant in zip(axes[1:], PLAIN + [MODERN]):
        for color, lr in zip(LR_RAMP, learning_rates(variant)):
            steps, loss = series(runs[variant, lr], 'val_loss')
            ax.plot(steps, loss, color=color, lw=1.5, label=f'{lr:g}')
        ax.set_ylim(3.4, 5.5)
        ax.set_title(f'{variant}: val loss during training')
        ax.set_xlabel('step')
        ax.legend(title='LR (light→dark)', fontsize=6, title_fontsize=6, loc='upper right')
    save(fig, 'p2_plain_vs_modern.png')


def plot_best_loss(runs):
    rows = sorted(((best(runs, v)[0], v) for v in VARIANTS), reverse=True)
    base = best(runs, MODERN)[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(rows))
    left = 3.4
    colors = [BASE if v == MODERN else CATEGORICAL[1] if v in PLAIN else SERIES for _, v in rows]
    ax.barh(y, [loss - left for loss, _ in rows], left=left, height=.6, color=colors)
    ax.axvline(base, color=MUTED, lw=1, ls='--')
    ax.set_yticks(y, [v for _, v in rows])
    for yi, (loss, v) in zip(y, rows):
        ax.text(loss + .005, yi, f'{loss:.3f}  ({loss-base:+.3f})', va='center', color=INK, fontsize=8)
    ax.set_xlim(left, max(l for l, _ in rows) + .12)
    ax.set_xlabel(f'best final validation loss over the 5 supplied LRs (axis starts at {left})')
    ax.set_title('Best validation loss per RNN (gray: modern baseline; orange: plain RNNs)')
    ax.grid(axis='y', visible=False)
    save(fig, 'p2_best_loss.png')


def plot_lr_sweeps(runs):
    fig, axes = grid(len(ABLATIONS) + 1, cols=3, width=3.6)
    base_rates = learning_rates(MODERN)
    base_loss = [runs[MODERN, lr]['final_val_loss'] for lr in base_rates]
    top = 3.95
    for ax, variant in zip(axes, ABLATIONS):
        ax.plot(base_rates, base_loss, color=BASE, marker='o', ms=4)
        rates = learning_rates(variant)
        ok = [(lr, runs[variant, lr]['final_val_loss']) for lr in rates if outcome(runs[variant, lr]) == 'ok']
        ax.plot(*zip(*ok), color=SERIES, marker='o', ms=5)
        for lr in rates:
            run, kind = runs[variant, lr], outcome(runs[variant, lr])
            loss = run['final_val_loss']
            if kind != 'ok' or loss > top:
                ax.scatter([lr], [top], marker='^' if kind == 'ok' else 'x', s=36, color=BAD, zorder=3)
                ax.annotate('div.' if kind == 'diverged' else f'{loss:.2f}', (lr, top), textcoords='offset points',
                            xytext=(0, -12), ha='center', fontsize=7, color=BAD)
        loss, lr = best(runs, variant)
        ax.scatter([lr], [loss], s=70, facecolor='none', edgecolor=INK, lw=1.2, zorder=4)
        ax.set_xscale('log')
        ax.set_ylim(3.45, top + .03)
        ax.set_title(variant)
        ax.set_xlabel('learning rate')
        ax.set_ylabel('final val loss')
    legend_ax = axes[len(ABLATIONS)]
    legend_ax.set_visible(True)
    legend_ax.axis('off')
    legend_ax.legend(handles=[plt.Line2D([], [], color=BASE, marker='o', label='modern rnn baseline'),
                              plt.Line2D([], [], color=SERIES, marker='o', label='variant'),
                              plt.Line2D([], [], color=BAD, marker='x', ls='', label='diverged / collapsed'),
                              plt.Line2D([], [], color=BAD, marker='^', ls='', label='off scale (finite)'),
                              plt.Line2D([], [], color=INK, marker='o', mfc='none', ls='', label='best LR')],
                     loc='center', fontsize=9)
    fig.suptitle('RNN ablations: final validation loss vs LR against the modern RNN', fontweight='bold', color=INK)
    save(fig, 'p2_lr_sweeps.png')


def plot_val_curves(runs):
    fig, axes = grid(len(ABLATIONS) + 1, cols=3, width=3.6)
    for ax, variant in zip(axes, [MODERN] + ABLATIONS):
        for color, lr in zip(LR_RAMP, learning_rates(variant)):
            steps, loss = series(runs[variant, lr], 'val_loss')
            ax.plot(steps, loss, color=color, lw=1.5, label=f'{lr:g}')
            if outcome(runs[variant, lr]) == 'diverged' and len(steps):
                ax.scatter(steps[-1:], [min(loss[-1], 5.4)], marker='x', color=BAD, s=30, zorder=3)
        ax.set_ylim(3.4, 5.5)
        ax.set_title(variant)
        ax.set_xlabel('step')
        ax.set_ylabel('val loss')
        ax.legend(title='LR (light→dark)', fontsize=6, title_fontsize=6, loc='upper right')
    fig.suptitle('RNN validation loss during training, all five LRs (× = run stopped)', fontweight='bold', color=INK)
    save(fig, 'p2_val_curves.png')


def benefit(runs, variant, direction, baseline):
    """Loss without the component minus loss with it: positive means the component helps."""
    delta = best(runs, variant)[0] - best(runs, baseline)[0]
    return delta if direction == 'remove' else -delta


def plot_transfer(rnn_runs, transformer_runs):
    """(d): does each tool matter more for RNNs or transformers?"""
    rows = [(label, benefit(rnn_runs, rv, rd, MODERN), benefit(transformer_runs, tv, td, 'd8'))
            for label, rv, rd, tv, td in TRANSFER]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(rows))
    clip = .25
    for offset, color, index, name in ((-.2, CATEGORICAL[0], 2, 'transformer (d8)'), (.2, CATEGORICAL[1], 1, 'RNN (modern)')):
        values = np.array([row[index] for row in rows])
        ax.barh(y + offset, np.clip(values, -clip, clip), height=.36, color=color, label=name)
        for yi, value in zip(y, values):
            x = np.clip(value, -clip, clip)
            text = f'{value:+.3f}' + (' →' if abs(value) > clip else '')
            ax.text(x + (.006 if x >= 0 else -.006), yi + offset, text, va='center',
                    ha='left' if x >= 0 else 'right', fontsize=7, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, [row[0] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(-clip - .08, clip + .1)
    ax.set_xlabel('value of the component: best loss without it − best loss with it  (>0: it helps; bars clipped at ±0.25)')
    ax.set_title('Same tool, two architectures: loss effect vs each architecture\'s own tuned baseline')
    ax.grid(axis='y', visible=False)
    ax.legend(loc='lower right', fontsize=8)
    save(fig, 'p2_transfer_vs_transformer.png')
    lines = ['| Component | RNN variant | Transformer variant | Value for RNN | Value for transformer |', '|---|---|---|---|---|']
    for (label, rv, _, tv, _), (_, r, t) in zip(TRANSFER, rows):
        lines.append(f'| {label} | {rv} | {tv} | {r:+.4f} | {t:+.4f} |')
    (PLOTS/'p2_transfer.md').write_text('\n'.join(lines)+'\n')
    print(PLOTS/'p2_transfer.md')


def depth_plot(runs, name, title, ylabel, profile, labels, ref=None):
    fig, axes = grid(len(VARIANTS), cols=4, width=3.3)
    for ax, variant in zip(axes, VARIANTS):
        run, lr = best_run(runs, variant)
        x_labels = labels(variant)
        x = np.arange(len(x_labels))
        if ref and modern(variant) and variant != MODERN:
            base = best_run(runs, MODERN)[0]
            ax.plot(x, profile(base, MODERN, last_step(base)), color=BASE, lw=1.5, label='rnn, end')
        for color, step, label in snapshots(run):
            ax.plot(x, profile(run, variant, step), color=color, marker='o' if len(x) < 4 else None, ms=4, label=label)
        if ref == 'one':
            ax.axhline(1, color=MUTED, lw=.8, ls=':')
        ax.set_yscale('log')
        ax.set_xticks(x, x_labels, fontsize=7)
        ax.set_title(f'{variant} (lr {lr:g})')
        ax.set_ylabel(ylabel)
    axes[0].legend(fontsize=7)
    axes[3].legend(fontsize=7)
    fig.suptitle(title, fontweight='bold', color=INK)
    save(fig, name)


def stream_profile(run, variant, step):
    return np.array([at_step(run, k, step) for k in stream_keys(variant)])


def ratio_profile(run, variant, step):
    if not modern(variant):  # plain layers replace the stream: compare each output with its input
        states = stream_profile(run, variant, step)
        return states[1:] / states[:-1]
    keys = [f'logging/rms/stack.{i}.{p}.operation/activation' for i in range(LAYERS) for p in ('recurrent', 'mlp')]
    return np.array([at_step(run, k, step) for k in keys]) / stream_profile(run, variant, step)[:-1]


def recurrent_grad_profile(run, variant, matrix):
    prefix = 'stack.{i}.cell' if not modern(variant) else 'stack.{i}.recurrent.operation.cell'
    keys = [f'logging/rms/{prefix.format(i=i)}.{matrix}/gradient' for i in range(1 if variant == 'rnn1' else LAYERS)]
    return lambda run, variant, step: np.array([at_step(run, k, step) for k in keys])


def plot_recurrent_grads(runs):
    fig, axes = grid(len(VARIANTS), cols=4, width=3.3)
    for ax, variant in zip(axes, VARIANTS):
        run, lr = best_run(runs, variant)
        layers = 1 if variant == 'rnn1' else LAYERS
        x = np.arange(layers)
        for matrix, style in (('hidden', '-'), ('input', ':')):
            profile = recurrent_grad_profile(run, variant, matrix)
            for color, step, label in snapshots(run):
                ax.plot(x, profile(run, variant, step), color=color, ls=style, marker='o', ms=4,
                        label=f'{"U (h→h)" if matrix == "hidden" else "W (x→h)"}, {label}')
        ax.set_yscale('log')
        ax.set_xticks(x, [f'layer {i}' for i in x], fontsize=7)
        ax.set_title(f'{variant} (lr {lr:g})')
        ax.set_ylabel('param-grad RMS')
    axes[0].legend(fontsize=6)
    fig.suptitle('Recurrent weight gradients per layer (solid: temporal U; dotted: input W; pre-clip)',
                 fontweight='bold', color=INK)
    save(fig, 'p2_recurrent_weight_grads.png')


def plot_instability(runs):
    cases = [('rnn4', (.001, .002, .004)), ('rnn-postnorm', (.001, .002, .004)),
             ('rnn-no-prenorm', (.002, .004)), ('rnn-no-residual', (.001, .002, .004))]
    fig, axes = plt.subplots(3, len(cases), figsize=(4*len(cases), 7.5), sharex='col', squeeze=False)
    for col, (variant, rates) in enumerate(cases):
        for color, lr in zip(LR_RAMP[-len(rates):], rates):
            run = runs[variant, lr]
            label = f"lr {lr:g} ({ {'ok': 'finished'}.get(outcome(run), outcome(run)) }, val {run['final_val_loss'] or float('nan'):.2f})"
            steps, loss = series(run, 'train_loss')
            if len(loss) > 25:
                loss = np.convolve(loss, np.ones(25)/25, mode='valid')
                steps = steps[12:12+len(loss)]
            axes[0, col].plot(steps, loss, color=color, lw=1.4, label=label)
            stacked = [series(run, k) for k in stream_keys(variant)]
            steps = stacked[0][0]
            axes[1, col].plot(steps, np.nanmax(np.stack([v[:len(steps)] for _, v in stacked]), 0), color=color, lw=1.4)
            axes[2, col].plot(*series(run, 'logging/rms/global/gradient'), color=color, lw=1.4)
        axes[0, col].set_title(variant)
        axes[0, col].legend(fontsize=7)
        axes[0, col].set_ylabel('train loss')
        axes[1, col].set_ylabel('max depth-stream RMS')
        axes[2, col].set_ylabel('global grad RMS (pre-clip)')
        axes[1, col].set_yscale('log')
        axes[2, col].set_yscale('log')
        axes[-1, col].set_xlabel('step')
    fig.suptitle('Anatomy of the failing RNN runs: loss, stream size, gradient size over training',
                 fontweight='bold', color=INK)
    save(fig, 'p2_instability.png')


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    runs = load(RESULTS / 'p2_runs')
    outcomes_table(runs)
    plot_plain_and_modern(runs)
    plot_best_loss(runs)
    plot_lr_sweeps(runs)
    plot_val_curves(runs)
    plot_transfer(runs, load(RESULTS / 'p1_runs'))
    depth_plot(runs, 'p2_stream_rms_depth.png', 'Depth-stream activation RMS (R: after recurrent branch, M: after MLP; best LR)',
               'activation RMS', stream_profile, depth_labels, ref='baseline')
    depth_plot(runs, 'p2_branch_ratio_depth.png',
               'Branch output relative to its input stream (plain RNNs: layer output / layer input)',
               'RMS(out) / RMS(in)', ratio_profile, lambda v: depth_labels(v)[1:], ref='one')
    plot_recurrent_grads(runs)
    plot_instability(runs)


if __name__ == '__main__':
    main()
