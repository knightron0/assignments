"""P2(b)-(f) plots from p2_measure results and the W&B cache (run p1_fetch and p2_measure first).

    uv run python -m experiments.a4_stability.p2bf_plots

Writes results/a4_stability/p2/{b,c,d,e,f}*.png and tables. P2(a) is p2a_plots.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .p1_fetch import RESULTS
from .p1_plots import BAD, BASE, INK, LR_RAMP, SERIES, best, load, outcome, save, series
from .p2_measure import BEST_LR, OUTPUT as MEASUREMENTS
from .sweeps import learning_rates

OUTPUT = MEASUREMENTS.parent
FLOOR = 1e-30  # gradients that underflow to exactly 0 are drawn at this floor
ORANGE = '#eb6834'
LAYER_RAMP = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']
CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
ABLATIONS = ['rnn-no-residual', 'rnn-no-prenorm', 'rnn-rope', 'rnn-final-norm', 'rnn-postnorm',
             'rnn-gated-residual', 'rnn-gated-output', 'rnn-gated-norm']
PLAIN = ('rnn1', 'rnn4', 'rnn4-collapsed')
NAMES = {'rnn1': '1-layer plain RNN', 'rnn4': '4-layer plain RNN', 'rnn4-collapsed': '4-layer plain, lr 0.002 (collapsed)',
         'rnn': 'modern RNN', 'rnn-no-residual': '(i) no depth residuals', 'rnn-no-prenorm': '(ii) no pre-norm',
         'rnn-rope': '(iii) + position rotation', 'rnn-final-norm': '(iv) + final RMSNorm', 'rnn-postnorm': '(v) post-norm',
         'rnn-gated-residual': '(vi) gated depth residuals', 'rnn-gated-output': '(vii) gated recurrent output',
         'rnn-gated-norm': '(viii) gated norm', 'd8': 'd8 transformer'}
# Each RNN intervention, its transformer counterpart, and whether each one adds or removes the component.
TRANSFER = [('depth residual', 'rnn-no-residual', 'remove', 'd8-no-residual', 'remove'),
            ('pre-norm', 'rnn-no-prenorm', 'remove', 'd8-no-prenorm', 'remove'),
            ('position rotation', 'rnn-rope', 'add', 'd8-no-rope', 'remove'),
            ('final output norm', 'rnn-final-norm', 'add', 'd8-no-final-norm', 'remove'),
            ('post-norm (instead of pre-norm)', 'rnn-postnorm', 'add', 'd8-postnorm', 'add'),
            ('gated residual', 'rnn-gated-residual', 'add', 'd8-gated-residual', 'add'),
            ('gated branch output', 'rnn-gated-output', 'add', 'd8-gated-attention', 'add'),
            ('gated norm', 'rnn-gated-norm', 'add', 'd8-gated-norm', 'add')]


def read(kind):
    return {p.stem: json.loads(p.read_text()) for p in (MEASUREMENTS / kind).glob('*.json')}


def title(fig, text):
    fig.suptitle(text, fontweight='bold', color=INK, fontsize=11)


def floor(values):
    return np.maximum(np.asarray(values, float), FLOOR)


def layer_lines(ax, per_layer, x=None, transform=lambda v: v):
    for i, values in enumerate(per_layer):
        values = transform(np.asarray(values, float))
        color = LAYER_RAMP[i] if len(per_layer) > 1 else SERIES
        ax.plot(np.arange(len(values)) if x is None else x, values, color=color, lw=1.3, label=f'layer {i}')


def depth_labels(ax, count):
    """Modern RNN depth stream: two points per layer (after the recurrent branch, after the MLP)."""
    ax.set_xticks(range(0, count, 2), ['emb'] + [f'L{i}' for i in range((count - 1)//2)])
    ax.set_xticks(range(1, count, 2), minor=True)
    ax.set_xlabel('layer')


def reach(gradients, threshold=1e-3):
    """Tokens back from the final token until the gradient falls below threshold × its value there."""
    values = np.asarray(gradients, float)[::-1]
    below = np.nonzero(values < threshold * values[0])[0]
    return int(below[0]) if len(below) else len(values) - 1


def legend_panel(ax, handles):
    ax.axis('off')
    ax.legend(handles=handles, loc='center', fontsize=10)


# --------------------------------------------------------------------------- time and depth figures

def time_figure(diag, names, filename, heading):
    """Four rows: L_T gradient reach, L gradient, hidden RMS, saturation; one column per model."""
    fig, axes = plt.subplots(4, len(names), figsize=(5.2*len(names), 13), squeeze=False)
    for col, name in enumerate(names):
        final = diag[name]['final']
        ax = axes[0, col]
        last = final['last']['hidden_gradient_rms']
        layer_lines(ax, last, x=np.arange(1, len(last[0])), transform=lambda v: floor(v[::-1][1:]))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(FLOOR / 3, None)
        ax.set_xlabel('tokens before the final token')
        ax.set_ylabel('RMS(∂L_T/∂h_t)  (0 drawn at 1e-30)')
        ax.set_title(NAMES[name], fontsize=10)
        ax = axes[1, col]
        layer_lines(ax, final['full']['hidden_gradient_rms'], transform=floor)
        ax.set_yscale('log')
        ax.set_ylabel('RMS(∂L/∂h_t)')
        ax = axes[2, col]
        layer_lines(ax, final['full']['hidden_rms'])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('RMS(h_t)')
        ax = axes[3, col]
        layer_lines(ax, final['full']['fraction_abs_gt_095'])
        ax.set_ylim(0, 1.02)
        ax.set_ylabel('share of |h_t| > 0.95')
        for ax in axes[1:, col]:
            ax.set_xlabel('time index t')
    handles = [plt.Line2D([], [], color=c, lw=1.5, label=f'layer {i}') for i, c in enumerate(LAYER_RAMP)]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(.5, -.01))
    title(fig, f'{heading}\n(final checkpoints, same 32 held-out sequences; light → dark = bottom → top layer)')
    save(fig, filename, OUTPUT)


def depth_figure(diag, names, filename, heading):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for name, color in zip(names, CATEGORICAL):
        final = diag[name]['final']
        for ax, (objective, key) in zip(axes, (('full', 'activation_rms'), ('full', 'gradient_rms'), ('last', 'gradient_rms'))):
            values = [r[key] for r in final[objective]['depth']]
            ax.plot(np.linspace(0, 1, len(values)), values, color=color, marker='o', ms=4, lw=1.6, label=NAMES[name])
    for ax, sub, ylabel in zip(axes, ('Activation RMS of the depth stream', 'Gradient RMS for L', 'Gradient RMS for L_T'),
                               ('activation RMS', 'RMS(∂L/∂x)', 'RMS(∂L_T/∂x)')):
        ax.set_yscale('log')
        ax.set_title(sub, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks([0, 1], ['embedding', 'top'])
        ax.set_xlabel('depth (relative: models have different layer counts)')
    axes[0].legend(fontsize=8)
    title(fig, f'{heading} (final checkpoints)')
    save(fig, filename, OUTPUT)


# --------------------------------------------------------------------------- loss (c, d)

def plot_modern_sweep(runs):
    fig, (ax, curves) = plt.subplots(1, 2, figsize=(13, 4.4))
    for v, color in (('rnn4', BASE), ('rnn', SERIES)):
        ok = [(lr, runs[v, lr]['final_val_loss']) for lr in learning_rates(v) if outcome(runs[v, lr]) == 'ok']
        loss, lr = best(runs, v)
        ax.plot(*zip(*ok), color=color, marker='o', ms=5, lw=1.8, label=f'{NAMES[v]}: best {loss:.3f} at lr {lr:g}')
        ax.scatter([lr], [loss], s=90, facecolor='none', edgecolor=INK, lw=1.2, zorder=4)
    ax.set_xscale('log')
    ax.set_xlabel('learning rate')
    ax.set_ylabel('final validation loss')
    ax.set_title('Final val loss at each supplied LR (4-layer plain collapses above 0.001)', fontsize=10)
    ax.legend(fontsize=8)
    for color, lr in zip(LR_RAMP, learning_rates('rnn')):
        run = runs['rnn', lr]
        curves.plot(*series(run, 'val_loss'), color=color, lw=1.5, label=f"lr {lr:g}: {run['final_val_loss']:.3f}")
    curves.set_ylim(3.4, 5)
    curves.set_xlabel('training step')
    curves.set_ylabel('validation loss')
    curves.set_title('Modern RNN, all five LRs (light → dark = low → high)', fontsize=10)
    curves.legend(fontsize=8)
    title(fig, f"(c) Modern 4-layer RNN: best {best(runs, 'rnn')[0]:.3f}, "
               f"{best(runs, 'rnn4')[0] - best(runs, 'rnn')[0]:.2f} better than the 4-layer plain RNN")
    save(fig, 'c1_sweep.png', OUTPUT)


def plot_ablation_curves(runs):
    base = best(runs, 'rnn')[0]
    base_steps, base_curve = series(runs['rnn', BEST_LR['rnn']], 'val_loss')
    fig, axes = plt.subplots(3, 3, figsize=(13, 9.5), sharex=True)
    for ax, v in zip(axes.flat, ABLATIONS):
        loss = best(runs, v)[0]
        ax.plot(base_steps, base_curve, color=BASE, lw=1.4)
        ax.plot(*series(runs[v, BEST_LR[v]], 'val_loss'), color=SERIES, lw=1.6)
        ax.set_ylim(3.4, 4.3)
        ax.set_title(f'{NAMES[v]}, lr {BEST_LR[v]:g}: {loss:.3f} ({loss - base:+.3f})', fontsize=9)
    legend_panel(axes.flat[-1], [plt.Line2D([], [], color=BASE, lw=1.4, label=f'modern RNN baseline (lr 0.002): {base:.3f}'),
                                 plt.Line2D([], [], color=SERIES, lw=1.6, label='variant at its best LR')])
    for ax in axes[-1]:
        ax.set_xlabel('training step')
    for ax in axes[:, 0]:
        ax.set_ylabel('val loss')
    title(fig, '(d) Validation loss during training, each ablation at its best LR (gray = modern RNN)\n'
               'panel titles: best final val loss (difference from the modern RNN)')
    save(fig, 'd1_loss_curves.png', OUTPUT)


def plot_ablation_sweeps(runs):
    base_rates = learning_rates('rnn')
    base_loss = [runs['rnn', lr]['final_val_loss'] for lr in base_rates]
    top = 4.0
    fig, axes = plt.subplots(3, 3, figsize=(13, 9.5), sharey=True)
    for ax, v in zip(axes.flat, ABLATIONS):
        ax.plot(base_rates, base_loss, color=BASE, marker='o', ms=4, lw=1.6)
        rates = learning_rates(v)
        ok = [(lr, min(runs[v, lr]['final_val_loss'], top)) for lr in rates if outcome(runs[v, lr]) == 'ok']
        ax.plot(*zip(*ok), color=SERIES, marker='o', ms=5, lw=1.8)
        for lr in rates:
            run, kind = runs[v, lr], outcome(runs[v, lr])
            if kind != 'ok' or run['final_val_loss'] > top:
                ax.scatter([lr], [top], marker='x', s=40, color=BAD, zorder=3)
                ax.annotate('diverged' if kind == 'diverged' else f"{run['final_val_loss']:.2f}", (lr, top),
                            textcoords='offset points', xytext=(0, -13), ha='center', fontsize=7, color=BAD)
        ax.scatter([BEST_LR[v]], [best(runs, v)[0]], s=80, facecolor='none', edgecolor=INK, lw=1.2, zorder=4)
        ax.set_xscale('log')
        ax.set_ylim(3.45, top + .04)
        ax.set_title(NAMES[v], fontsize=10)
    legend_panel(axes.flat[-1], [plt.Line2D([], [], color=BASE, marker='o', label='modern RNN baseline'),
                                 plt.Line2D([], [], color=SERIES, marker='o', label='variant'),
                                 plt.Line2D([], [], color=BAD, marker='x', ls='', label='diverged / off scale'),
                                 plt.Line2D([], [], color=INK, marker='o', mfc='none', ls='', label='best LR')])
    for ax in axes[-1]:
        ax.set_xlabel('learning rate')
    for ax in axes[:, 0]:
        ax.set_ylabel('final val loss')
    title(fig, '(d) Final validation loss at each supplied learning rate (gray = modern RNN)')
    save(fig, 'd2_lr_sweeps.png', OUTPUT)


def plot_ablation_overview(runs):
    base = best(runs, 'rnn')[0]
    colors = dict(zip(ABLATIONS, CATEGORICAL))
    fig, (full, zoom) = plt.subplots(1, 2, figsize=(18, 5.6), gridspec_kw=dict(width_ratios=(1.25, 1)))
    order = sorted(ABLATIONS + ['rnn'], key=lambda v: best(runs, v)[0])
    for ax in (full, zoom):
        for v in order:
            final = best(runs, v)[0]
            steps, loss = series(runs[v, BEST_LR[v]], 'val_loss')
            ax.plot(steps, loss, color=INK if v == 'rnn' else colors[v], lw=2.2 if v == 'rnn' else 1.6,
                    ls='--' if v == 'rnn' else '-',
                    label=f'{NAMES[v]} (lr {BEST_LR[v]:g}): {final:.3f}' + ('' if v == 'rnn' else f' ({final - base:+.3f})'))
        ax.set_xlabel('training step')
        ax.set_ylabel('val loss')
    full.set_ylim(3.4, 4.3)
    full.set_title('Whole run', fontsize=10)
    zoom.set_xlim(6000, steps[-1] + 100)
    zoom.set_ylim(3.45, 3.75)
    zoom.set_title('Last third of training (zoomed)', fontsize=10)
    zoom.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1), title='best final val loss (vs modern RNN), sorted',
                title_fontsize=8)
    title(fig, '(d) Validation loss overview: every ablation at its best LR (dashed = modern RNN baseline)')
    save(fig, 'd3_loss_overview.png', OUTPUT)


def benefit(runs, variant, direction, baseline):
    """Loss without the component minus loss with it: positive means the component helps."""
    delta = best(runs, variant)[0] - best(runs, baseline)[0]
    return delta if direction == 'remove' else -delta


def plot_transfer(rnn_runs, transformer_runs):
    rows = [(label, benefit(rnn_runs, rv, rd, 'rnn'), benefit(transformer_runs, tv, td, 'd8'))
            for label, rv, rd, tv, td in TRANSFER]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    y = np.arange(len(rows))
    clip = .25
    for offset, color, index, name in ((-.2, SERIES, 2, 'transformer (vs d8)'), (.2, ORANGE, 1, 'RNN (vs modern RNN)')):
        values = np.array([row[index] for row in rows])
        ax.barh(y + offset, np.clip(values, -clip, clip), height=.36, color=color, label=name)
        for yi, value in zip(y, values):
            x = np.clip(value, -clip, clip)
            ax.text(x + (.006 if x >= 0 else -.006), yi + offset, f'{value:+.3f}' + (' →' if abs(value) > clip else ''),
                    va='center', ha='left' if x >= 0 else 'right', fontsize=7, color=INK)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y, [row[0] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(-clip - .08, clip + .1)
    ax.set_xlabel('value of the component = best loss without it − best loss with it   (> 0: it helps; clipped at ±0.25)')
    ax.grid(axis='y', visible=False)
    ax.legend(loc='lower right', fontsize=8)
    title(fig, "(d) Which tools matter more for RNNs than transformers? Loss effect vs each architecture's tuned baseline")
    save(fig, 'd8_transfer_vs_transformer.png', OUTPUT)
    lines = ['| component | RNN variant | value for RNN | transformer variant | value for transformer |', '|---|---|---|---|---|']
    for (label, rv, _, tv, _), (_, r, t) in zip(TRANSFER, rows):
        lines.append(f'| {label} | {rv} | {r:+.4f} | {tv} | {t:+.4f} |')
    (OUTPUT / 'd8_transfer.md').write_text('\n'.join(lines) + '\n')
    print(OUTPUT / 'd8_transfer.md')


# --------------------------------------------------------------------------- ablation diagnostics (d)

def plot_ablation_depth(diag):
    base = diag['rnn']['final']
    count = len(base['full']['depth'])
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, v in zip(axes.flat, ABLATIONS):
        ax.plot([r['activation_rms'] for r in base['full']['depth']], color=BASE, lw=1.6)
        ax.plot([r['activation_rms'] for r in diag[v]['final']['full']['depth']], color=SERIES, lw=1.8)
        ax.set_yscale('log')
        ax.set_title(NAMES[v], fontsize=10)
        depth_labels(ax, count)
    legend_panel(axes.flat[-1], [plt.Line2D([], [], color=BASE, lw=1.6, label='modern RNN baseline'),
                                 plt.Line2D([], [], color=SERIES, lw=1.8, label='variant')])
    for ax in axes[:, 0]:
        ax.set_ylabel('activation RMS')
    title(fig, '(d) Activation RMS across depth (final checkpoints; gray = modern RNN)')
    save(fig, 'd4_depth_activation.png', OUTPUT)

    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, v in zip(axes.flat, ABLATIONS):
        for objective, color in (('full', SERIES), ('last', ORANGE)):
            ax.plot([r['gradient_rms'] for r in base[objective]['depth']], color=color, lw=1.3, ls='--', alpha=.7)
            ax.plot([r['gradient_rms'] for r in diag[v]['final'][objective]['depth']], color=color, lw=1.8)
        ax.set_yscale('log')
        ax.set_title(NAMES[v], fontsize=10)
        depth_labels(ax, count)
    legend_panel(axes.flat[-1], [plt.Line2D([], [], color=SERIES, lw=1.8, label='L, variant'),
                                 plt.Line2D([], [], color=SERIES, lw=1.3, ls='--', label='L, modern RNN'),
                                 plt.Line2D([], [], color=ORANGE, lw=1.8, label='L_T, variant'),
                                 plt.Line2D([], [], color=ORANGE, lw=1.3, ls='--', label='L_T, modern RNN')])
    for ax in axes[:, 0]:
        ax.set_ylabel('RMS(∂Loss/∂x)')
    title(fig, '(d) Gradient RMS across depth for L and L_T (final checkpoints; dashed = modern RNN)')
    save(fig, 'd5_depth_gradients.png', OUTPUT)


def time_summary(diag, name):
    """Per layer: mean saturation and hidden RMS over t >= 16, and L_T gradient reach in tokens."""
    final = diag[name]['final']
    return dict(saturation=[float(np.mean(s[16:])) for s in final['full']['fraction_abs_gt_095']],
                hidden_rms=[float(np.mean(h[16:])) for h in final['full']['hidden_rms']],
                reach=[reach(g) for g in final['last']['hidden_gradient_rms']])


def plot_time_heatmaps(diag):
    order = ['rnn4', 'rnn'] + ABLATIONS
    table = {n: time_summary(diag, n) for n in order}
    panels = (('saturation', 'tanh saturation: share of |h_t| > 0.95', '{:.0%}'),
              ('hidden_rms', 'hidden-state RMS', '{:.2f}'),
              ('reach', 'last-token gradient reach: tokens back until\n∂L_T/∂h_t falls below 0.1% of its value at the end', '{:.0f}'))
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    for ax, (key, label, fmt) in zip(axes, panels):
        values = np.array([table[n][key] for n in order], float)
        shown = np.maximum(values, 1) if key == 'reach' else values
        norm = (matplotlib.colors.LogNorm(1, max(values.max(), 2)) if key == 'reach'
                else matplotlib.colors.Normalize(0, 1))
        image = ax.imshow(shown, cmap='Blues', norm=norm, aspect='auto')
        for (r, c), v in np.ndenumerate(values):
            ax.text(c, r, fmt.format(v), ha='center', va='center', fontsize=8,
                    color='white' if norm(shown[r, c]) > .6 else INK)
        ax.axhline(1.5, color=INK, lw=1.5)  # plain 4-layer and modern baseline above, ablations below
        ax.set_xticks(range(4), [f'layer {i}' for i in range(4)])
        ax.set_yticks(range(len(order)), [NAMES[n] for n in order] if ax is axes[0] else [], fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.grid(False)
        plt.colorbar(image, ax=ax, fraction=.046, pad=.03)
    title(fig, '(d) Time diagnostics per recurrent layer (final checkpoints; saturation and RMS averaged over t ≥ 16)')
    save(fig, 'd6_time_summary.png', OUTPUT)
    lines = ['| model | saturation by layer | hidden RMS by layer | L_T gradient reach by layer (tokens) |', '|---|---|---|---|']
    for n in ['rnn1', 'rnn4-collapsed'] + order:
        row = time_summary(diag, n)
        lines.append(f"| {NAMES[n]} | {' / '.join(f'{v:.0%}' for v in row['saturation'])} "
                     f"| {' / '.join(f'{v:.2f}' for v in row['hidden_rms'])} | {' / '.join(str(v) for v in row['reach'])} |")
    (OUTPUT / 'd6_time_summary.md').write_text('\n'.join(lines) + '\n')
    print(OUTPUT / 'd6_time_summary.md')


def plot_ablation_reach(diag):
    """L_T gradient vs tokens back, top recurrent layer, each ablation against the modern RNN."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 9.5), sharex=True, sharey=True)
    for ax, v in zip(axes.flat, ABLATIONS):
        for name, color in (('rnn', BASE), (v, SERIES)):
            g = np.asarray(diag[name]['final']['last']['hidden_gradient_rms'][-1], float)[::-1]
            ax.plot(np.arange(1, len(g)), floor(g[1:] / max(g[0], FLOOR)), color=color, lw=1.6)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-12, 3)
        ax.set_title(NAMES[v], fontsize=10)
    legend_panel(axes.flat[-1], [plt.Line2D([], [], color=BASE, lw=1.6, label='modern RNN baseline'),
                                 plt.Line2D([], [], color=SERIES, lw=1.6, label='variant')])
    for ax in axes[-1]:
        ax.set_xlabel('tokens before the final token')
    for ax in axes[:, 0]:
        ax.set_ylabel('∂L_T/∂h_t ÷ value at the final token')
    title(fig, '(d) How far back the last-token gradient reaches in the top recurrent layer (gray = modern RNN)')
    save(fig, 'd7_gradient_reach.png', OUTPUT)


# --------------------------------------------------------------------------- (e)

COMPARE_COLORS = {'rnn': ORANGE, 'd8': SERIES}


def plot_similarity(compare):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for name, data in compare.items():
        color, n = COMPARE_COLORS[name], len(data['activation_rms'])
        axes[0].plot(np.linspace(0, 1, n), data['activation_rms'], color=color, marker='o', ms=4, label=NAMES[name])
        axes[1].plot(np.linspace(0, 1, n)[1:], data['adjacent_cosine'], color=color, marker='o', ms=4, label=NAMES[name])
        lags = sorted(data['lag_cosine'], key=int)
        for depth, style, where in ((-1, '-', 'top of the stack'), (n // 2, '--', 'middle')):
            axes[2].plot([int(l) for l in lags], [data['lag_cosine'][l][depth] for l in lags], color=color, marker='o',
                         ms=4, ls=style, label=f'{NAMES[name]}, {where}')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('activation RMS')
    axes[0].set_title('Activation RMS across depth', fontsize=10)
    axes[1].set_ylabel('cosine between consecutive depth steps')
    axes[1].set_title('Similarity between adjacent depth steps', fontsize=10)
    for ax in axes[:2]:
        ax.set_xticks([0, 1], ['embedding', 'top'])
        ax.set_xlabel('depth (relative)')
    axes[2].set_xscale('log')
    axes[2].set_xlabel('lag between positions (tokens)')
    axes[2].set_ylabel('cosine(x_t, x_t−lag)')
    axes[2].set_title('Similarity between positions at several lags', fontsize=10)
    for ax in axes:
        ax.legend(fontsize=7)
    title(fig, '(e) Modern RNN vs d8 transformer: activation scale and similarity across layers and positions '
               '(final checkpoints)')
    save(fig, 'e1_similarity.png', OUTPUT)


def plot_removal(compare):
    """Does high similarity mean something can be removed? Layers (adjacent cosine) and tokens (lag cosine)."""
    fig, (layers, tokens) = plt.subplots(1, 2, figsize=(14, 5))
    for name, data in compare.items():
        color = COMPARE_COLORS[name]
        cos = np.array(data['adjacent_cosine'])[:len(data['layer_removal'])]  # input vs output of each sublayer
        layers.scatter(cos, np.maximum(data['layer_removal'], 1e-3), color=color, s=40, label=NAMES[name], zorder=3)
        lags = sorted(data['token_removal'], key=int)
        tokens.plot([data['last_position_cosine'][l] for l in lags], [data['token_removal'][l] for l in lags],
                    color=color, marker='o', ms=6, label=NAMES[name])
        for l in ('1', '2', '256'):
            tokens.annotate(f'lag {l}', (data['last_position_cosine'][l], data['token_removal'][l]), fontsize=7,
                            textcoords='offset points', xytext=(4, 4), color=color)
    layers.set_yscale('log')
    layers.set_xlabel('cosine between the stream before and after the sublayer (high = small change)')
    layers.set_ylabel('loss increase when the sublayer is removed')
    layers.set_title('Layers: one point per sublayer', fontsize=10)
    tokens.axhline(0, color=INK, lw=.8, ls='--')
    tokens.set_xlabel('cosine between the final position and the token `lag` back (top of the stack)')
    tokens.set_ylabel('change in last-token loss when that token is deleted')
    tokens.set_title('Tokens: delete one input token at a given lag', fontsize=10)
    for ax in (layers, tokens):
        ax.legend(fontsize=8)
    title(fig, '(e) Does high similarity mean a layer or token can be removed? (final checkpoints, 32 sequences)')
    save(fig, 'e2_removal.png', OUTPUT)


# --------------------------------------------------------------------------- (f)

def plot_context(context):
    colors = {'d8': SERIES, 'rnn': ORANGE, 'rnn4': '#1baf7a', 'rnn1': '#eda100'}
    order = [n for n in ('d8', 'rnn', 'rnn4', 'rnn1') if n in context]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for name in order:
        data, color = context[name], colors[name]
        keeps = data['keeps']
        axes[0].plot(keeps, np.maximum([np.mean(data['increase']['bos_True'][str(k)]) for k in keeps], 1e-3),
                     color=color, marker='o', ms=4, label=NAMES[name])
        axes[1].plot(keeps, [np.mean(np.array(data['increase']['bos_False'][str(k)]) > 1) for k in keeps],
                     color=color, marker='o', ms=4, label=NAMES[name])
        g = np.asarray(data['gradient_reach'], float)[::-1]  # last input position first
        axes[2].plot(np.arange(1, len(g) + 1), floor(g / g[0]), color=color, lw=1.5, label=NAMES[name])
    axes[0].set_yscale('log')
    axes[0].set_ylabel('mean loss increase vs full context (nats)')
    axes[0].set_title('Loss increase with a shorter prefix (BOS token kept)', fontsize=10)
    axes[1].set_ylabel('share of targets whose loss rises > 1 nat')
    axes[1].set_title('Without the BOS token: how often prediction breaks', fontsize=10)
    for ax in axes[:2]:
        ax.set_xscale('log')
        ax.set_xlabel('tokens of context kept')
        ax.legend(fontsize=8)
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].set_ylim(1e-12, 3)
    axes[2].set_xlabel('tokens before the final token')
    axes[2].set_ylabel('RMS(∂L_T/∂embedding) ÷ value at the last token')
    axes[2].set_title('Last-token gradient reach at the input', fontsize=10)
    axes[2].legend(fontsize=8)
    title(fig, f"(f) How much prior context do the models use? (final checkpoints; {len(context['d8']['full_loss'])} "
               "sequences; RNN states reset, transformer positions preserved)")
    save(fig, 'f_context.png', OUTPUT)
    lines = ['| model | full-context last-token loss | useful context: smallest prefix within 0.01 nats (BOS kept) '
             '| share broken at 512 tokens without BOS | gradient reach: tokens back to < 0.1% of the last-token gradient |',
             '|---|---|---|---|---|']
    for name in order:
        data = context[name]
        useful = next((k for k in data['keeps'] if np.mean(data['increase']['bos_True'][str(k)]) < .01), '> 512')
        broken = np.mean(np.array(data['increase']['bos_False']['512']) > 1)
        lines.append(f"| {NAMES[name]} | {np.mean(data['full_loss']):.3f} | {useful} | {broken:.0%} "
                     f"| {reach(data['gradient_reach'])} |")
    (OUTPUT / 'f_context.md').write_text('\n'.join(lines) + '\n')
    print(OUTPUT / 'f_context.md')


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    diag, compare, context = read('diagnose'), read('compare'), read('context')
    rnn_runs, transformer_runs = load(RESULTS / 'p2_runs'), load(RESULTS / 'p1_runs')
    time_figure(diag, list(PLAIN), 'b1_time.png', '(b) Plain RNNs across time: gradient reach, hidden-state size, saturation')
    depth_figure(diag, list(PLAIN), 'b2_depth.png', '(b) Plain RNNs across depth')
    plot_modern_sweep(rnn_runs)
    time_figure(diag, ['rnn4', 'rnn'], 'c2_time.png', '(c) Modern RNN vs 4-layer plain RNN across time')
    depth_figure(diag, ['rnn4', 'rnn'], 'c3_depth.png', '(c) Modern RNN vs 4-layer plain RNN across depth')
    plot_ablation_curves(rnn_runs)
    plot_ablation_sweeps(rnn_runs)
    plot_ablation_overview(rnn_runs)
    plot_ablation_depth(diag)
    plot_time_heatmaps(diag)
    plot_ablation_reach(diag)
    plot_transfer(rnn_runs, transformer_runs)
    plot_similarity(compare)
    plot_removal(compare)
    plot_context(context)


if __name__ == '__main__':
    main()
