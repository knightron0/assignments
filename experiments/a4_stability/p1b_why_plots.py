"""P1(b): one figure per variant with the statistic or intervention that explains its loss difference.

    uv run python -m experiments.a4_stability.p1b_why_plots

Needs p1_fetch, p1b_measure and p1b_interventions output. Writes results/a4_stability/p1b/why/*.png
and table.md.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .p1_fetch import RESULTS
from .p1_plots import BASE, INK, LR_RAMP, MUTED, SERIES, best, load, outcome, save, series
from .p1b_interventions import OUTPUT as INTERVENTIONS
from .p1b_measure import BEST_LR
from .p1b_plots import CATEGORICAL, DEPTH_XLABEL, NAMES, label_depth, measurements
from .sweeps import learning_rates

OUTPUT = RESULTS / 'p1b' / 'why'
CHECKPOINT_SHADES = {'init': '#b7d3f6', 'step 100': '#5598e7', 'final': '#104281'}
BASE_SHADES = {'init': '#dcdbd6', 'step 100': '#b3b2ac', 'final': '#6f6e69'}


def interventions():
    return {p.stem: json.loads(p.read_text()) for p in INTERVENTIONS.glob('*.json')}


def title(fig, text):
    fig.suptitle(text, fontweight='bold', color=INK, fontsize=11)


def sublayer_ticks(ax, count=16):
    ax.set_xticks(range(0, count, 2), [f'L{i}' for i in range(count // 2)], fontsize=8)
    ax.set_xticks(range(1, count, 2), minor=True)
    ax.set_xlabel('sublayer (major tick = attention, minor = MLP)')


def no_residual(depth):
    """Gradient RMS across depth over training: d8 vs no residual."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, v in zip(axes, ('d8', 'd8-no-residual')):
        for checkpoint, color in CHECKPOINT_SHADES.items():
            values = depth[v]['depth'][checkpoint]['full']['gradient_rms']
            ax.plot(np.maximum(values, 1e-30), color=color, label=checkpoint)
        ax.set_yscale('log')
        ax.set_title(NAMES[v])
        label_depth(ax, 17)
        ax.set_xlabel(DEPTH_XLABEL)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('activation-gradient RMS for L')
    title(fig, '(i) Without residuals the gradient reaching the lower layers collapses during training (to ~1e-20)')
    save(fig, '1_no_residual.png', OUTPUT)


def removal_and_updates(data, v, heading, filename):
    """Left: loss increase when each sublayer is removed. Right: update size relative to the stream."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, color in (('d8', BASE), (v, SERIES)):
        axes[0].plot(data[name]['layer_removal'], color=color, marker='o', ms=4, label=NAMES[name])
        axes[1].plot(data[name]['update_size'], color=color, marker='o', ms=4, label=NAMES[name])
    axes[0].set_yscale('symlog', linthresh=.01)
    axes[0].set_ylabel('loss increase when removed (nats)')
    axes[0].set_title('how much each sublayer is used')
    axes[1].set_yscale('log')
    axes[1].set_ylabel('update RMS ÷ stream RMS')
    axes[1].set_title('how large each update is')
    for ax in axes:
        sublayer_ticks(ax)
    axes[0].legend(fontsize=8)
    title(fig, heading)
    save(fig, filename, OUTPUT)


def no_prenorm(data):
    removal_and_updates(data, 'd8-no-prenorm',
                        '(ii) Without pre-norm the lower layers barely contribute: removing any of L0–L2 costs < 0.03',
                        '2_no_prenorm.png')


def no_qknorm(data, runs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for v, shades in (('d8', ('#c3c2b7', '#8f8e88', '#52514e')), ('d8-no-qknorm', LR_RAMP[1:4])):
        for lr, color in zip((.003, .006, .012), shades):
            key = v if lr == BEST_LR[v] else f'{v}@{lr:g}'
            stats = data[key]['attention']
            loss = runs[v, min(learning_rates(v), key=lambda r: abs(r - lr))]['final_val_loss']
            label = f"{'d8' if v == 'd8' else 'no QK-norm'}, lr {lr:g} (val {loss:.2f})"
            axes[0].plot([s['max_logit'] for s in stats], color=color, marker='o', ms=4, label=label)
            axes[1].plot([s['spiky'] * 100 for s in stats], color=color, marker='o', ms=4, label=label)
    axes[0].set_yscale('log')
    axes[0].set_ylabel('largest attention logit |q·k|/√d')
    axes[1].set_ylabel('% of late queries with one key > 90%')
    for ax in axes:
        ax.set_xticks(range(8), [f'L{i}' for i in range(8)])
        ax.set_xlabel(DEPTH_XLABEL)
    axes[1].legend(fontsize=7)
    title(fig, '(iii) Without QK-norm, attention logits grow with LR until attention becomes one-hot, '
               'and loss degrades')
    save(fig, '3_no_qknorm.png', OUTPUT)


def bars(ax, rows):
    """rows: (label, value, spread-or-None, color)."""
    y = np.arange(len(rows))[::-1]
    for yi, (label, value, spread, color) in zip(y, rows):
        ax.barh(yi, value, height=.6, color=color)
        if spread is not None:
            ax.errorbar(value, yi, xerr=[[value - spread[0]], [spread[1] - value]], color=INK, lw=1, capsize=3)
        ax.text(max(value, spread[1] if spread else value) + .005, yi, f'{value:+.3f}', va='center', fontsize=8)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.grid(axis='y', visible=False)


def random_row(label, values, color):
    return (label, float(np.mean(values)), (min(values), max(values)), color)


def no_rope(data):
    d8, rope = data['d8'], data['d8-no-rope']
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw=dict(width_ratios=[1, 1.5]))
    bars(axes[0], [('d8', d8['mask_bos'], None, BASE), ('no RoPE', rope['mask_bos'], None, SERIES)])
    axes[0].set_xlabel('loss increase when attention to BOS is blocked')
    axes[0].set_title('no RoPE uses BOS to find positions')
    rows = [(f"d8: its {len(d8['early_local_heads']['heads'])} local heads in L1–L2",
             d8['early_local_heads']['loss'], None, SERIES),
            random_row('d8: same number of other heads, same layers', d8['early_random_heads'], BASE),
            (f"no RoPE: its {len(rope['local_heads']['heads'])} local heads (L3–L6)",
             rope['local_heads']['loss'], None, CATEGORICAL[1]),
            random_row('no RoPE: same number of other heads, same layers', rope['random_heads'], BASE)]
    bars(axes[1], rows)
    axes[1].set_xlabel('loss increase when the heads are zeroed (gray: 3 draws, min–max)')
    axes[1].set_title('what the local heads are worth')
    title(fig, '(iv) Without RoPE, position has to come from BOS, and local heads only form from layer 3 up')
    save(fig, '4_no_rope.png', OUTPUT)


def no_prenorm_activations(depth):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, checkpoint in zip(axes, ('init', 'final')):
        for v, color in (('d8', BASE), ('d8-no-prenorm', SERIES)):
            ax.plot(depth[v]['depth'][checkpoint]['full']['activation_rms'], color=color, label=NAMES[v])
        ax.set_yscale('log')
        ax.set_title(checkpoint)
        label_depth(ax, 17)
        ax.set_xlabel(DEPTH_XLABEL)
    axes[0].set_ylabel('activation RMS')
    axes[0].legend(fontsize=8)
    title(fig, '(ii) Without pre-norm the stream stays tiny through the lower layers; d8 rescales it in layer 0')
    save(fig, '2b_no_prenorm_activation.png', OUTPUT)


def no_final_norm(depth, data):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for v, color in (('d8', BASE), ('d8-no-final-norm', SERIES)):
        ax.plot(depth[v]['depth']['final']['full']['activation_rms'], color=color,
                label=f"{NAMES[v]} (logit spread {data[v]['logit_rms']:.1f})")
    ax.set_yscale('log')
    ax.set_ylabel('activation RMS')
    label_depth(ax, 17)
    ax.set_xlabel(DEPTH_XLABEL)
    ax.legend(fontsize=8)
    title(fig, '(v) Without a final norm the logits read the stream directly, so the whole stream is kept small')
    save(fig, '5_no_final_norm.png', OUTPUT)


def postnorm(depth, data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for v, color in (('d8', BASE), ('d8-postnorm', SERIES)):
        grads = np.asarray(depth[v]['depth']['final']['full']['gradient_rms'])
        axes[0].plot(np.arange(2, 17), grads[2:] / grads[16], color=color, label=NAMES[v])
        axes[1].plot(data[v]['layer_removal'], color=color, marker='o', ms=4, label=NAMES[v])
    axes[0].axhline(1, color=INK, lw=.8, ls='--')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('gradient RMS ÷ gradient at top')
    label_depth(axes[0], 17)
    axes[0].set_xlabel(DEPTH_XLABEL)
    axes[0].set_title('gradient reaching each depth')
    axes[1].set_yscale('symlog', linthresh=.01)
    axes[1].set_ylabel('loss increase when the sublayer is removed')
    sublayer_ticks(axes[1])
    axes[1].set_title('how much each sublayer is used')
    axes[1].legend(fontsize=8)
    title(fig, '(vi) Post-norm starves the lower layers of gradient, and they end up barely used')
    save(fig, '6_postnorm.png', OUTPUT)


GATED = ('d8-gated-residual', 'd8-gated-attention', 'd8-gated-norm')


def gates(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[1, 1.4]))
    bars(axes[0], [(NAMES[v], data[v]['gates_mean']['loss'], None, CATEGORICAL[i]) for i, v in enumerate(GATED)])
    axes[0].set_xlabel('loss increase when each gate is fixed at its mean')
    axes[0].set_title('does the gate need to depend on the input?')
    for i, v in enumerate(GATED):
        means = list(data[v]['gates_mean']['mean_gate'].values())
        axes[1].plot(means, color=CATEGORICAL[i], marker='o', ms=4, label=NAMES[v])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('mean gate value')
    axes[1].set_xlabel('gate (in depth order)')
    axes[1].set_title('how open the gates are on average')
    axes[1].legend(fontsize=8)
    title(fig, '(vii)–(ix) Fixing the gates at their mean breaks every gated model: the gating is input-dependent')
    save(fig, '7_gates.png', OUTPUT)


def sink_mask(data):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    order = list(BEST_LR)
    y = np.arange(len(order))[::-1]
    for offset, key, color, label in ((.18, 'mask_bos', SERIES, 'block attention to BOS'),
                                      (-.18, 'mask_boundary', CATEGORICAL[1], 'block BOS and EOS')):
        values = [data[v][key] for v in order]
        ax.barh(y + offset, values, height=.34, color=color, label=label)
    ax.set_yticks(y, [NAMES[v] for v in order], fontsize=8)
    ax.set_xscale('symlog', linthresh=.01)
    ax.set_xlabel('validation loss increase (nats)')
    ax.grid(axis='y', visible=False)
    ax.legend(fontsize=8)
    title(fig, '(viii) Blocking BOS costs gated attention less than d8 (0.10 vs 0.16); no RoPE and no final norm '
               'depend on it most')
    save(fig, '8_sink_mask.png', OUTPUT)


def gated_norm_lr(runs):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for v, color in (('d8', BASE), ('d8-gated-norm', SERIES)):
        rates = learning_rates(v)
        ax.plot(rates, [runs[v, lr]['final_val_loss'] for lr in rates], color=color, marker='o', ms=5, label=NAMES[v])
    ax.set_xscale('log')
    ax.set_xlabel('learning rate')
    ax.set_ylabel('final val loss')
    ax.legend(fontsize=8)
    title(fig, '(ix) Gated norm is best at its lowest supplied LR: its optimum may lie below the range')
    save(fig, '9_gated_norm_lr.png', OUTPUT)


def table(data, depth, runs):
    base = best(runs, 'd8')[0]
    lines = ['| variant | Δ loss vs d8 | update/stream (L0 attn) | grad ratio | mask BOS | mask BOS+EOS | '
             'local heads off | random heads off | gates → mean |', '|---|' + '---|' * 8]
    for v in BEST_LR:
        r = data[v]
        g = np.asarray(depth[v]['depth']['final']['full']['gradient_rms'])
        lines.append(f"| {NAMES[v]} | {best(runs, v)[0] - base:+.3f} | {r['update_size'][0]:.2f} | {g[2] / g[16]:.2f} | "
                     f"{r['mask_bos']:+.3f} | {r['mask_boundary']:+.3f} | {r['local_heads']['loss']:+.3f} | "
                     f"{np.mean(r['random_heads']) if r['random_heads'] else 0:+.3f} | "
                     f"{r['gates_mean']['loss'] if r['gates_mean'] else float('nan'):+.3f} |")
    (OUTPUT / 'table.md').write_text('\n'.join(lines) + '\n')
    print(OUTPUT / 'table.md')


LOSS_FILES = {'d8-no-residual': '1', 'd8-no-prenorm': '2', 'd8-no-qknorm': '3', 'd8-no-rope': '4',
              'd8-no-final-norm': '5', 'd8-postnorm': '6', 'd8-gated-residual': '7a', 'd8-gated-attention': '7b',
              'd8-gated-norm': '9'}


def loss_curves(runs):
    """One validation-loss plot per variant: its best-LR run against d8."""
    base_loss = best(runs, 'd8')[0]
    for v, prefix in LOSS_FILES.items():
        loss = best(runs, v)[0]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(*series(runs['d8', BEST_LR['d8']], 'val_loss'), color=BASE, label=f"d8, lr {BEST_LR['d8']:g}: {base_loss:.3f}")
        ax.plot(*series(runs[v, BEST_LR[v]], 'val_loss'), color=SERIES, label=f"{NAMES[v]}, lr {BEST_LR[v]:g}: {loss:.3f}")
        ax.set_ylim(2.85, 7.2 if v == 'd8-no-residual' else 3.4)
        ax.set_xlabel('training step')
        ax.set_ylabel('validation loss')
        ax.legend(fontsize=8)
        title(fig, f'{NAMES[v]}: {loss - base_loss:+.3f} vs d8 at the best supplied LR')
        save(fig, f'{prefix}_loss.png', OUTPUT)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runs, depth, data = load(RESULTS / 'p1_runs'), measurements(), interventions()
    loss_curves(runs)
    no_residual(depth)
    no_prenorm(data)
    no_prenorm_activations(depth)
    no_qknorm(data, runs)
    no_rope(data)
    no_final_norm(depth, data)
    postnorm(depth, data)
    gates(data)
    sink_mask(data)
    gated_norm_lr(runs)
    table(data, depth, runs)


if __name__ == '__main__':
    main()
