"""P1(a) plots from the checkpoint measurements (run p1a_checkpoint first).

    uv run python -m experiments.a4_stability.p1a_plots

(a)(i): p1a_i_depth.png. (a)(ii): p1a_ii_*.png. All in results/a4_stability/p1a/.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, PowerNorm

from .p1_plots import BASE, INK, MUTED, SERIES, save
from .p1a_checkpoint import LATE, OUTPUT, RUN

ORANGE = '#eb6834'
HEAD = '#c3c2b7'
UNIFORM = INK
LAYER_RAMP = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#104281']
HEATMAP = 'Blues'


def load(checkpoint):
    return json.loads((OUTPUT / f'{RUN}-{checkpoint}.json').read_text())


def depth_labels(count):
    return ['emb'] + [f'{i}{p}' for i in range((count - 1)//2) for p in ('A', 'M')]


# ---------------------------------------------------------------- (a)(i)

def plot_depth(final, early):
    count = len(final['full']['depth'])
    x, labels = np.arange(count), depth_labels(count)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    for stats, color, name in ((early, '#86b6ef', 'step 100'), (final, SERIES, 'final')):
        ax.plot(x, [r['activation_rms'] for r in stats['full']['depth']], color=color, marker='o', ms=4, label=name)
    ax.set_yscale('log')
    ax.set_title('Activation RMS of the residual stream')
    ax.set_ylabel('RMS(x)')
    ax.legend(fontsize=8)

    ax = axes[1]
    for objective, color, name in (('full', SERIES, 'L (all tokens)'), ('last', ORANGE, 'L_T (last token)')):
        for stats, style, when in ((final, '-', 'final'), (early, ':', 'step 100')):
            ax.plot(x, [r['gradient_rms'] for r in stats[objective]['depth']], color=color, ls=style,
                    marker='o' if style == '-' else None, ms=4, label=f'{name}, {when}')
    ax.set_yscale('log')
    ax.set_title('Activation-gradient RMS, dLoss/dx')
    ax.set_ylabel('RMS(dLoss/dx)')
    ax.legend(fontsize=7)

    ax = axes[2]
    for objective, color, name in (('full', SERIES, 'L'), ('last', ORANGE, 'L_T')):
        for stats, style, when in ((final, '-', 'final'), (early, ':', 'step 100')):
            grads = np.array([r['gradient_rms'] for r in stats[objective]['depth']])
            ax.plot(x, grads / grads[-1], color=color, ls=style, marker='o' if style == '-' else None, ms=4,
                    label=f'{name}, {when}')
    ax.axhline(1, color=MUTED, lw=.8, ls='--')
    ax.set_yscale('log')
    ax.set_title('Gradient RMS relative to the top of the stack')
    ax.set_ylabel('RMS(dLoss/dx) / value at 7M')
    ax.legend(fontsize=7)

    for ax in axes:
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_xlabel('stream state (emb, then after each Attention / MLP sublayer)')
    fig.suptitle(f"(a)(i) d8 (lr 0.003): depth profile on {final['sequences']} held-out sequences, FP32  "
                 f"[final L={final['full']['loss']:.3f}, L_T={final['last']['loss']:.3f}]",
                 fontweight='bold', color=INK)
    save(fig, 'p1a_i_depth.png', OUTPUT)


# ---------------------------------------------------------------- (a)(ii)

def log_bins(values, start=1):
    """Average a 1-D profile indexed from `start` over power-of-two bins. Returns centers, means."""
    values = np.asarray(values)
    edges = [start]
    while edges[-1] < start + len(values):
        edges.append(edges[-1]*2)
    centers, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        chunk = values[lo-start:hi-start]
        if len(chunk):
            centers.append(np.sqrt(lo*(hi-1)) if hi-1 > lo else lo)
            means.append(chunk.mean(-1) if chunk.ndim == 1 else chunk.mean(0))
    return np.array(centers), np.array(means)


def layer_grid(title):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4), sharex=True, sharey=True)
    fig.suptitle(title, fontweight='bold', color=INK)
    return fig, axes.flat


def head_lines(ax, x, per_head, uniform, ylabel):
    for row in per_head:
        ax.plot(x, row, color=HEAD, lw=1)
    ax.plot(x, np.mean(per_head, 0), color=SERIES, lw=2.2)
    ax.plot(x, uniform, color=UNIFORM, lw=1.2, ls='--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(ylabel)


def legend_handles():
    return [plt.Line2D([], [], color=HEAD, lw=1, label='individual heads (8)'),
            plt.Line2D([], [], color=SERIES, lw=2.2, label='head average'),
            plt.Line2D([], [], color=UNIFORM, lw=1.2, ls='--', label='uniform over causal prefix')]


def plot_distance(attn):
    fig, axes = layer_grid(f'(a)(ii) Attention mass vs distance (queries {LATE}–1022; mass per key at that lag)')
    for i, (ax, layer) in enumerate(zip(axes, attn)):
        lags = np.array(layer['mass_by_lag'])[:, 1:]  # lag 0 is the current token: see the self plot
        x, binned = log_bins(lags.T, start=1)
        head_lines(ax, x, binned.T, np.full(len(x), layer['uniform_mass_per_lag']), 'attention per key')
        ax.set_title(f'layer {i}')
    for ax in list(axes)[4:]:
        ax.set_xlabel('lag (tokens back)')
    fig.legend(handles=legend_handles(), loc='lower center', ncol=3, fontsize=8, bbox_to_anchor=(.5, -.03))
    save(fig, 'p1a_ii_distance.png', OUTPUT)


def plot_by_query(attn, key, name, filename):
    fig, axes = layer_grid(f'(a)(ii) {name} vs query position')
    for i, (ax, layer) in enumerate(zip(axes, attn)):
        values = np.array(layer[key])[:, 1:]  # q = 0 can only attend to itself
        x, binned = log_bins(values.T, start=1)
        _, uniform = log_bins(np.array(layer['uniform_by_query'])[1:], start=1)
        head_lines(ax, x, binned.T, uniform, name.split(' (')[0])
        ax.set_title(f'layer {i}')
    for ax in list(axes)[4:]:
        ax.set_xlabel('query position q')
    fig.legend(handles=legend_handles(), loc='lower center', ncol=3, fontsize=8, bbox_to_anchor=(.5, -.03))
    save(fig, filename, OUTPUT)


def late_ratio(layer, key):
    """Mean over late queries of mass / uniform 1/(q+1). (H,)"""
    values = np.array(layer[key])[:, LATE:]
    return (values * (np.arange(LATE, LATE + values.shape[1]) + 1)).mean(1)


def heatmap(ax, values, title, norm, fmt):
    image = ax.imshow(values, cmap=HEATMAP, norm=norm, aspect='auto')
    for (r, c), v in np.ndenumerate(values):
        level = norm(v) if np.isfinite(v) else 0
        ax.text(c, r, fmt(v), ha='center', va='center', fontsize=6.5, color='white' if level > .6 else INK)
    ax.set_xticks(range(values.shape[1]), [f'h{j}' for j in range(values.shape[1])], fontsize=7)
    ax.set_yticks(range(values.shape[0]), [f'L{i}' for i in range(values.shape[0])], fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.grid(False)
    plt.colorbar(image, ax=ax, fraction=.046, pad=.03)


def plot_heads(attn, early_attn):
    bos = np.array([layer['bos_ratio'] for layer in attn])
    eos = np.array([layer['eos_ratio'] for layer in attn])
    bos_early = np.array([layer['bos_ratio'] for layer in early_attn])
    self_ratio = np.array([late_ratio(layer, 'self_mass') for layer in attn])
    local = np.array([np.array(layer['mass_by_lag'])[:, 1:9].sum(1) for layer in attn])
    ratio_norm = LogNorm(vmin=.1, vmax=400)
    clip = lambda m: np.clip(m, .1, None)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.4))
    heatmap(axes[0], clip(bos), 'BOS (pos 0): mass ÷ uniform, final', ratio_norm, lambda v: f'{v:.0f}')
    heatmap(axes[1], clip(bos_early), 'BOS (pos 0): mass ÷ uniform, step 100', ratio_norm, lambda v: f'{v:.0f}')
    heatmap(axes[2], clip(eos), f"EOS tokens: mass ÷ uniform, final (n={attn[0]['eos_count']})", ratio_norm,
            lambda v: f'{v:.0f}' if v >= 10 else f'{v:.1f}')
    heatmap(axes[3], clip(self_ratio), 'Current token: mass ÷ uniform, final', ratio_norm,
            lambda v: f'{v:.0f}' if v >= 10 else f'{v:.1f}')
    heatmap(axes[4], local, 'Mass on lags 1–8 (local), final', PowerNorm(.5, vmin=0, vmax=1), lambda v: f'{v:.2f}')
    fig.suptitle(f'(a)(ii) Every head at a glance (queries {LATE}–1022; ratio 1 = uniform attention)',
                 fontweight='bold', color=INK)
    save(fig, 'p1a_ii_heads.png', OUTPUT)
    return bos, self_ratio, local


def plot_positions(attn):
    fig, (ax, bars) = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw=dict(width_ratios=(1.6, 1)))
    for i, (color, layer) in enumerate(zip(LAYER_RAMP, attn)):
        received = np.mean(layer['received_ratio'], 0)
        ax.plot(np.arange(len(received)), received, color=color, lw=1.2, label=f'layer {i}')
    ax.axhline(1, color=UNIFORM, lw=1, ls='--')
    ax.set_yscale('log')
    ax.set_xlim(-5, LATE)
    ax.set_xlabel('key position')
    ax.set_ylabel('mass received ÷ uniform (head avg)')
    ax.set_title(f'Which key positions do late queries ({LATE}–1022) attend to?')
    ax.annotate('position 0 (BOS)', (0, np.mean(attn[5]['received_ratio'], 0)[0]), xytext=(40, -8),
                textcoords='offset points', fontsize=8, color=INK, arrowprops=dict(arrowstyle='-', color=MUTED))
    ax.legend(fontsize=7, ncol=2, title='light→dark = shallow→deep', title_fontsize=7)
    x = np.arange(len(attn))
    groups = (('BOS (pos 0)', 'bos_ratio', SERIES), ('EOS tokens', 'eos_ratio', ORANGE), ('all other tokens', 'other_ratio', BASE))
    for k, (name, key, color) in enumerate(groups):
        bars.bar(x + (k-1)*.27, [np.mean(layer[key]) for layer in attn], width=.25, color=color, label=name)
    bars.axhline(1, color=UNIFORM, lw=1, ls='--')
    bars.set_yscale('log')
    bars.set_xticks(x, [f'L{i}' for i in x])
    bars.set_ylabel('mass ÷ uniform (head avg)')
    bars.set_title('BOS and EOS vs ordinary tokens, by layer')
    bars.legend(fontsize=8)
    bars.grid(axis='x', visible=False)
    fig.suptitle('(a)(ii) Positions that attract unusual attention (dashed: uniform)', fontweight='bold', color=INK)
    save(fig, 'p1a_ii_positions.png', OUTPUT)


def plot_examples(attn, bos, self_ratio, local):
    picks = []
    for name, scores in (('strongest BOS sink', bos), ('most local (lags 1–8)', local),
                         ('most self-attending', self_ratio), ('broad (little BOS or local mass late)', -np.log(bos) - 5*local)):
        order = np.argsort(scores, axis=None)[::-1]
        layer, head = next(divmod(int(k), scores.shape[1]) for k in order if divmod(int(k), scores.shape[1]) not in
                           [(l, h) for _, l, h in picks])
        picks.append((name, layer, head))
    fig, axes = plt.subplots(1, len(picks), figsize=(16, 4.3))
    for ax, (name, layer, head) in zip(axes, picks):
        image = np.array(attn[layer]['example_maps'][head])
        image = np.where(np.tril(np.ones_like(image)) > 0, image, np.nan)
        shown = ax.imshow(image, cmap=HEATMAP, norm=PowerNorm(.4, vmin=0, vmax=1), interpolation='nearest')
        ax.set_title(f'{name}\nlayer {layer}, head {head}', fontsize=9)
        ax.set_xlabel('key position')
        ax.set_ylabel('query position')
        ax.grid(False)
    plt.colorbar(shown, ax=axes, fraction=.015, pad=.01, label='attention weight')
    fig.suptitle('(a)(ii) Example heads: first 64 positions of one validation sequence', fontweight='bold', color=INK)
    fig.savefig(OUTPUT/'p1a_ii_examples.png', bbox_inches='tight')
    plt.close(fig)
    print(OUTPUT/'p1a_ii_examples.png')


def main():
    final, early = load('final'), load('early')
    plot_depth(final, early)
    attn = final['attention_detail']
    plot_distance(attn)
    plot_by_query(attn, 'self_mass', 'Attention on the current token', 'p1a_ii_self.png')
    plot_by_query(attn, 'first_mass', 'Attention on position 0 (BOS)', 'p1a_ii_first.png')
    bos, self_ratio, local = plot_heads(attn, early['attention_detail'])
    plot_positions(attn)
    plot_examples(attn, bos, self_ratio, local)


if __name__ == '__main__':
    main()
