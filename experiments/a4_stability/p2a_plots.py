"""P2(a): supplied LR sweeps for the plain 1-layer and 4-layer RNNs (run p1_fetch first).

    uv run python -m experiments.a4_stability.p2a_plots

Writes results/a4_stability/p2a/{sweeps.png, losses.md}.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .p1_fetch import RESULTS
from .p1_plots import BAD, INK, LR_RAMP, best, load, outcome, save, series
from .sweeps import learning_rates

OUTPUT = RESULTS / 'p2a'
MODELS = {'rnn1': ('1-layer plain RNN', '#2a78d6'), 'rnn4': ('4-layer plain RNN', '#eb6834')}
PARAMETERS = {'rnn1': '4.72M (0.53M non-embedding)', 'rnn4': '6.30M (2.11M non-embedding)'}
COLLAPSE_TOP = 4.5  # collapsed runs (~6.87) are marked at the top of the LR panel instead


def plot_sweeps(runs):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    for v, (name, color) in MODELS.items():
        rates = learning_rates(v)
        ok = [(lr, runs[v, lr]['final_val_loss']) for lr in rates if outcome(runs[v, lr]) == 'ok']
        loss, lr = best(runs, v)
        ax.plot(*zip(*ok), color=color, marker='o', ms=5, lw=1.8, label=f'{name}: best {loss:.3f} at lr {lr:g}')
        ax.scatter([lr], [loss], s=90, facecolor='none', edgecolor=INK, lw=1.2, zorder=4)
        for lr in rates:
            run = runs[v, lr]
            if outcome(run) != 'ok':
                ax.scatter([lr], [COLLAPSE_TOP], marker='x', s=45, color=BAD, zorder=4)
                ax.annotate(f"collapsed\n{run['final_val_loss']:.2f}", (lr, COLLAPSE_TOP), textcoords='offset points',
                            xytext=(0, -22), ha='center', fontsize=7, color=BAD)
    ax.set_xscale('log')
    ax.set_ylim(3.95, COLLAPSE_TOP + .05)
    ax.set_xlabel('learning rate')
    ax.set_ylabel('final validation loss')
    ax.set_title('Final val loss at each supplied LR (○ = best)', fontsize=10)
    ax.legend(fontsize=8, loc='lower left')

    for ax, v in zip(axes[1:], MODELS):
        for color, lr in zip(LR_RAMP, learning_rates(v)):
            run = runs[v, lr]
            ax.plot(*series(run, 'val_loss'), color=color, lw=1.5,
                    label=f"lr {lr:g}: {run['final_val_loss']:.3f}" + (' (collapsed)' if outcome(run) != 'ok' else ''))
        ax.set_ylim(3.9, 7.3 if v == 'rnn4' else 5.5)
        ax.set_xlabel('training step')
        ax.set_ylabel('validation loss')
        ax.set_title(f'{MODELS[v][0]}: all five LRs (light → dark = low → high)', fontsize=10)
        ax.legend(fontsize=8, loc='center right')
    fig.suptitle('P2(a) Plain RNNs: the 1-layer model beats the 4-layer one, and the 4-layer one '
                 'collapses at LR ≥ 0.002', fontweight='bold', color=INK, fontsize=11)
    save(fig, 'sweeps.png', OUTPUT)


def loss_table(runs):
    lines = ['| model | parameters | ' + ' | '.join(f'{f:g}× center' for f in (.25, .5, 1, 2, 4)) + ' | best |',
             '|---|---|' + '---|' * 6]
    for v, (name, _) in MODELS.items():
        cells = []
        for lr in learning_rates(v):
            run = runs[v, lr]
            cells.append(f"lr {lr:g}: {run['final_val_loss']:.4f}" + (' (collapsed)' if outcome(run) != 'ok' else ''))
        loss, lr = best(runs, v)
        lines.append(f'| {name} | {PARAMETERS[v]} | ' + ' | '.join(cells) + f' | **{loss:.4f}** (lr {lr:g}) |')
    lines += ['', 'Collapsed = finished training but stuck near 6.87, roughly the loss of predicting token frequencies.']
    (OUTPUT / 'losses.md').write_text('\n'.join(lines) + '\n')
    print(OUTPUT / 'losses.md')


if __name__ == '__main__':
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runs = load(RESULTS / 'p2_runs')
    plot_sweeps(runs)
    loss_table(runs)
