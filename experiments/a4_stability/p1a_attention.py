"""P1(a)(ii): attention patterns across sequence positions for the default d8.

    uv run python -m experiments.a4_stability.p1a_attention

Same checkpoints (init, step 100, final) and held-out sequences as p1a_depth.
Attention weights are recomputed exactly from each layer's normalized, rotated q/k.
Every summary is compared with uniform attention over the causal prefix, 1/(q+1).
Writes results/a4_stability/p1a_ii/{measurements.json, *.png}.
"""
import json
import math
from pathlib import Path

import numpy as np

from .p1a_depth import CHECKPOINTS, LR, RUN, SEQUENCES, VARIANT, held_out_tokens, load

OUTPUT = Path('results/a4_stability/p1a_ii')
LATE = 512        # late queries (q >= LATE): uniform attention is spread thin, so sinks stand out
MIN_GAP = 8       # for received mass, only count queries more than 8 tokens after the key
BOS, EOS = 1, 2   # every sequence starts with BOS; EOS separates packed documents
MAP_SIZE = 64


def capture_qk(model):
    from .p1_transformer_depth import Attention
    store, handles = {}, []
    for i, module in enumerate(m for m in model.modules() if isinstance(m, Attention)):
        for label, norm in (('q', module.q_norm), ('k', module.k_norm)):
            def hook(module, args, output, i=i, label=label):
                store.setdefault(i, {})[label] = output.detach()
            handles.append(norm.register_forward_hook(hook))
    return store, handles


def attention_probs(q, k):
    """The model's causal softmax(q k^T / sqrt(d)) in FP32. q: (H, T, d), k: (Hkv, T, d)."""
    import torch
    k = k.repeat_interleave(q.shape[0] // k.shape[0], 0)
    scores = q.float() @ k.float().transpose(-1, -2) / math.sqrt(q.shape[-1])
    mask = torch.ones(q.shape[1], q.shape[1], dtype=torch.bool, device=q.device).triu(1)
    return scores.masked_fill(mask, -torch.inf).softmax(-1)


def summarize(store, tokens):
    import torch
    length, device = tokens.shape[1], tokens.device
    queries = torch.arange(length, device=device)
    late = queries[LATE:]
    lag_index = late[:, None] - torch.arange(LATE, device=device)[None]
    uniform_weight = (queries + 1).float()[:, None]  # multiply p(q, k) by (q+1): 1 means uniform
    far = queries[:, None] > queries[None, :] + MIN_GAP  # (q, k) pairs with q - k > MIN_GAP
    layers = []
    for i in sorted(store):
        acc = dict(lag=0., self=0., first=0., bos=0., bos_received=0., eos=[], other=[])
        for b in range(tokens.shape[0]):
            p = attention_probs(store[i]['q'][b], store[i]['k'][b])  # (H, T, T)
            acc['lag'] += p[:, LATE:].gather(-1, lag_index.expand(p.shape[0], -1, -1)).mean(1)
            acc['self'] += p.diagonal(dim1=-2, dim2=-1)
            acc['first'] += p[..., 0]
            acc['bos'] += (p[:, LATE:, 0] * uniform_weight[LATE:, 0]).mean(-1)
            # Mass each key receives from queries more than MIN_GAP later, relative to uniform.
            ratio = p * uniform_weight * far
            counts = far.sum(0)
            received = ratio.sum(1)[:, counts > 0] / counts[counts > 0]
            keys = torch.arange(length, device=device)[counts > 0]
            kinds = tokens[b, keys]
            if (kinds == EOS).any():
                acc['eos'].append(received[:, kinds == EOS])
            acc['other'].append(received[:, (kinds != EOS) & (keys > 0)])
            acc['bos_received'] += received[:, 0]  # key 0 under the same rule as EOS/other
            if b == 0:
                maps = p[:, :MAP_SIZE, :MAP_SIZE]
        n = tokens.shape[0]
        layers.append(dict(
            mass_by_lag=(acc['lag']/n).tolist(),
            self_mass=(acc['self']/n).tolist(),
            first_mass=(acc['first']/n).tolist(),
            bos_ratio=(acc['bos']/n).tolist(),
            bos_received=(acc['bos_received']/n).tolist(),
            eos_ratio=torch.cat(acc['eos'], 1).mean(1).tolist() if acc['eos'] else None,
            eos_count=sum(e.shape[1] for e in acc['eos']),
            other_ratio=torch.cat(acc['other'], 1).mean(1).tolist(),
            example_map=maps.tolist()))
    return dict(layers=layers, uniform_by_query=(1/(queries.float()+1)).tolist(),
                uniform_per_lag=(1/(late.float()+1)).mean().item())


def measure(device='cuda', variant=VARIANT, lr=LR):
    import torch
    tokens = held_out_tokens(device)[:, :-1]  # the same model inputs profile() uses
    results = {}
    for checkpoint in CHECKPOINTS:
        model = load(checkpoint, device, variant, lr).eval()
        store, handles = capture_qk(model)
        try:
            with torch.no_grad():
                model(tokens)
        finally:
            for handle in handles:
                handle.remove()
        results[checkpoint] = summarize(store, tokens)
    results['sequences'] = tokens.shape[0]
    return results


# ------------------------------------------------------------------------- plots
# One figure per question in the handout. "÷ uniform" = attention divided by 1/(q+1), the weight
# a head spreading evenly over its causal prefix would give: 1 means no preference.

LAYER_RAMP = ['#b7d3f6', '#9ec5f4', '#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281', '#0d366b']
CHECKPOINT_SHADES = dict(zip(CHECKPOINTS, ('#b7d3f6', '#5598e7', '#104281')))
HEAD_DOT = '#52514e'
HEAD_LINE = '#c3c2b7'


def late_ratio(values):
    """(H, T) mass per query -> (H,) mean over late queries of mass / uniform."""
    values = np.asarray(values)[:, LATE:]
    return (values * (np.arange(LATE, LATE + values.shape[1]) + 1)).mean(1)


def bars_with_heads(ax, per_layer_heads, color, label=None):
    """Bar = head average for each layer; dots = the 8 individual heads."""
    x = np.arange(len(per_layer_heads))
    ax.bar(x, [np.mean(h) for h in per_layer_heads], width=.6, color=color, label=label)
    for i, heads in enumerate(per_layer_heads):
        ax.scatter(np.full(len(heads), i) + np.linspace(-.2, .2, len(heads)), heads, s=10,
                   color=HEAD_DOT, zorder=3, label='individual heads' if i == 0 else None)
    ax.set_xticks(x, [f'layer {i}' for i in x])
    ax.grid(axis='x', visible=False)


def finish(fig, ax, title, filename, even_line=True):
    from .p1_plots import INK, save
    if even_line:
        ax.axhline(1, color=INK, lw=1, ls='--', label='even share (no preference)')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    fig.suptitle(title, fontweight='bold', color=INK, fontsize=11)
    save(fig, filename, OUTPUT)


def layer_lines(x, per_layer, xlabel, ylabel, title, filename, even_share=None, even_label='even share = 1/(q+1)'):
    """One plot, one head-averaged line per layer (light → dark = layer 0 → 7).

    even_share: None draws the ratio reference at 1; an array draws the raw even share 1/(q+1)."""
    import matplotlib.pyplot as plt
    from .p1_plots import INK
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (color, y) in enumerate(zip(LAYER_RAMP, per_layer)):
        ax.plot(x, y, color=color, lw=1.6, label=f'layer {i}')
    if even_share is not None:
        ax.plot(x, even_share, color=INK, lw=1, ls='--', label=even_label)
    ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    finish(fig, ax, title, filename, even_line=even_share is None)


def plot_distance(results):
    """Q1: how does attention mass vary with distance? Raw weight per key at every lag 1..LATE-1."""
    final = results['final']
    per_layer = [np.mean(layer['mass_by_lag'], 0)[1:] for layer in final['layers']]
    lags = np.arange(1, LATE)
    layer_lines(lags, per_layer, 'distance back (tokens)', 'attention on the token at that distance (head average)',
                f'1. Attention vs distance: nearby tokens get far more than an even share\n'
                f'(final d8, queries {LATE}–1022; even share ≈ {final["uniform_per_lag"]:.2%} per token; '
                f'light → dark = layer 0 → 7)',
                '1_distance.png', even_share=np.full(len(lags), final['uniform_per_lag']), even_label='even share')


def plot_self(results):
    """Q2: how much goes to the current token? Head average per layer, every query position."""
    q = np.arange(1, len(results['final']['uniform_by_query']))  # q = 0 has only itself to attend to
    per_layer = [np.mean(layer['self_mass'], 0)[1:] for layer in results['final']['layers']]
    layer_lines(q, per_layer, 'query position q', 'share of attention on itself (head average)',
                '2. Attention on the current token: each layer keeps a roughly fixed share\n'
                '(final d8; layer 0 ≈ 31%, others 1–6%, while an even share falls to 0.1%; light → dark = layer 0 → 7)',
                '2_self.png', even_share=1/(q + 1))


def plot_sink(results):
    """Q3: does position 0 attract an unusual amount? Raw share per layer vs the even share."""
    q = np.arange(1, len(results['final']['uniform_by_query']))
    per_layer = [np.mean(layer['first_mass'], 0)[1:] for layer in results['final']['layers']]
    layer_lines(q, per_layer, 'query position q', 'share of attention on position 0 (head average)',
                '3. Attention on position 0 (BOS): layers 3 and 5–7 still give it 13–20% at late queries — a sink\n'
                '(final d8; the even share falls 1000× over the sequence, the sink only ~5×; light → dark = layer 0 → 7)',
                '3_sink.png', even_share=1/(q + 1))


def plot_sink_over_training(results):
    """Q3: is the sink learned, or an effect of the causal mask? Compare init, step 100, final."""
    import matplotlib.pyplot as plt
    from .p1_plots import INK
    x = np.arange(len(results['final']['layers']))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for k, name in enumerate(CHECKPOINTS):
        ax.bar(x + (k-1)*.27, [np.mean(l['bos_ratio']) for l in results[name]['layers']], width=.25,
               color=CHECKPOINT_SHADES[name], label=name)
    ax.set_xticks(x, [f'layer {i}' for i in x])
    ax.set_ylabel('attention on position 0 ÷ even share\n(head average)')
    ax.grid(axis='x', visible=False)
    finish(fig, ax, f'3b. The BOS sink is learned: absent at init and step 100, 10–160× by the end\n'
                    f'(d8, queries {LATE}–1022)', '3b_sink_over_training.png')


def plot_positions(results):
    """Q3, continued: are other positions sinks too?"""
    import matplotlib.pyplot as plt
    from .p1_plots import BASE, SERIES
    layers = results['final']['layers']
    x = np.arange(len(layers))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for k, (name, key, color) in enumerate((('BOS (position 0)', 'bos_received', SERIES),
                                            (f"EOS tokens (n={layers[0]['eos_count']})", 'eos_ratio', '#eb6834'),
                                            ('all other tokens', 'other_ratio', BASE))):
        ax.bar(x + (k-1)*.27, [np.mean(layer[key]) for layer in layers], width=.25, color=color, label=name)
    ax.set_xticks(x, [f'layer {i}' for i in x])
    ax.grid(axis='x', visible=False)
    ax.set_ylabel('attention received ÷ uniform (head avg)')
    finish(fig, ax, f'4. Document boundaries are sinks too: EOS ≈ BOS, ordinary tokens ≈ uniform\n'
                    f'(final d8; queries more than {MIN_GAP} tokens after the key)', '4_positions.png')


def head_tables(results):
    final = results['final']['layers']
    return dict(bos=np.array([layer['bos_ratio'] for layer in final]),
                self=np.array([late_ratio(layer['self_mass']) for layer in final]),
                local=np.array([np.array(layer['mass_by_lag'])[:, 1:9].sum(1) for layer in final]))


def plot_heads(results):
    """Individual heads: one number per head for each question."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from .p1_plots import INK, save
    tables = head_tables(results)
    panels = ((tables['bos'], 'Sink: attention on position 0 ÷ uniform', LogNorm(.1, 500), '{:.0f}'),
              (tables['self'], 'Attention on the current token ÷ uniform', LogNorm(.1, 500), '{:.0f}'),
              (tables['local'], 'Share of attention on the previous 1–8 tokens', Normalize(0, 1), '{:.2f}'))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (values, title, norm, fmt) in zip(axes, panels):
        shown = np.clip(values, norm.vmin, None)
        image = ax.imshow(shown, cmap='Blues', norm=norm, aspect='auto')
        for (r, c), v in np.ndenumerate(values):
            ax.text(c, r, fmt.format(v), ha='center', va='center', fontsize=8,
                    color='white' if norm(shown[r, c]) > .6 else INK)
        ax.set_xticks(range(values.shape[1]), [f'head {j}' for j in range(values.shape[1])], fontsize=7)
        ax.set_yticks(range(values.shape[0]), [f'layer {i}' for i in range(values.shape[0])], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(False)
        plt.colorbar(image, ax=ax, fraction=.046, pad=.03)
    fig.suptitle(f'5. Individual heads specialize (final d8, queries {LATE}–1022): one cell per head',
                 fontweight='bold', color=INK, fontsize=11)
    save(fig, '5_heads.png', OUTPUT)


def plot_examples(results):
    """What the three head types look like."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from .p1_plots import INK
    tables = head_tables(results)
    picks, used = [], set()
    for name, scores in (('sink head', tables['bos']), ('previous-token head', tables['local']),
                         ('current-token head', tables['self'])):
        for flat in np.argsort(scores, axis=None)[::-1]:
            layer, head = divmod(int(flat), scores.shape[1])
            if (layer, head) not in used:
                used.add((layer, head))
                picks.append((name, layer, head))
                break
    fig, axes = plt.subplots(1, len(picks), figsize=(13, 4.4))
    for ax, (name, layer, head) in zip(axes, picks):
        image = np.array(results['final']['layers'][layer]['example_map'][head])
        image = np.where(np.tril(np.ones_like(image)) > 0, image, np.nan)
        shown = ax.imshow(image, cmap='Blues', norm=PowerNorm(.4, vmin=0, vmax=1), interpolation='nearest')
        ax.set_title(f'{name} (layer {layer}, head {head})', fontsize=10)
        ax.set_xlabel('key position k (attended to)')
        ax.set_ylabel('query position q (attending)')
        ax.grid(False)
    fig.colorbar(shown, ax=list(axes), fraction=.015, pad=.01, label='attention weight')
    fig.suptitle(f'6. Example heads: first {MAP_SIZE} positions of one sequence '
                 '(dark first column = sink; dark band beside the diagonal = previous token; diagonal = self)',
                 fontweight='bold', color=INK, fontsize=10)
    fig.savefig(OUTPUT/'6_examples.png', bbox_inches='tight')
    plt.close(fig)
    print(OUTPUT/'6_examples.png')


def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    for old in OUTPUT.glob('*.png'):
        old.unlink()  # figures are regenerated from measurements.json
    plot_distance(results)
    plot_self(results)
    plot_sink(results)
    plot_sink_over_training(results)
    plot_positions(results)
    plot_heads(results)
    plot_examples(results)


if __name__ == '__main__':
    import modal
    from modal_utils import app, build_image, VOLUME_MOUNTS, MODAL_ENVIRONMENT
    from .diagnostics import save as save_json
    probe = app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu='H100', timeout=1800)(measure)
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT):
        results = probe.remote()
    save_json(OUTPUT/'measurements.json', results)
    plot(results)
