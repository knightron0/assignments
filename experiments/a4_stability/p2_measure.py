"""P2 checkpoint probes on Modal: depth and time diagnostics for every RNN, plus the (e)/(f)
comparisons between the modern RNN and the d8 transformer.

    uv run python -m experiments.a4_stability.p2_measure

Best LR = lowest final validation loss among the five supplied rates (W&B sweep).
Same held-out sequences as P1. Writes results/a4_stability/p2/measurements/*.json.
"""
from pathlib import Path

OUTPUT = Path('results/a4_stability/p2/measurements')
BEST_LR = {
    'rnn1': .003, 'rnn4': .001, 'rnn': .002,
    'rnn-no-residual': .001, 'rnn-no-prenorm': .002, 'rnn-rope': .002, 'rnn-final-norm': .001,
    'rnn-postnorm': .001, 'rnn-gated-residual': .001, 'rnn-gated-output': .001, 'rnn-gated-norm': .001,
}
EXTRA = {'rnn4-collapsed': ('rnn4', .002)}  # a run that trained but learned nothing: what saturation looks like
COMPARE = {'rnn': ('rnn', .002), 'd8': ('d8', .003)}  # (e): the reference RNN and transformer
CONTEXT = {'rnn1': ('rnn1', .003), 'rnn4': ('rnn4', .001), **COMPARE}  # (f)
CONTEXT_SEQUENCES = 256
KEEPS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
LAGS = (1, 2, 4, 8, 16, 64, 256)


def diagnose(name):
    """(b)-(d): profile() at init, step 100 and final. Keeps depth rows and per-time recurrent stats."""
    from .diagnostics import profile
    from .p1a_depth import CHECKPOINTS, held_out_tokens, load
    variant, lr = EXTRA.get(name, (name, BEST_LR.get(name)))
    tokens = held_out_tokens('cuda')
    result = dict(name=name, variant=variant, lr=lr)
    for checkpoint in CHECKPOINTS:
        stats = profile(load(checkpoint, 'cuda', variant, lr), tokens)
        result[checkpoint] = {k: stats[k] for k in ('full', 'last', 'temporal_cosine') if k in stats}
    return result


def capture_stream(model):
    """Record every depth-stream state (branch inputs, then the final stream)."""
    states, handles = [], []
    for branch in model.branches():
        handles.append(branch.register_forward_pre_hook(lambda m, a: states.append(a[0].detach())))
    handles.append(model.output_norm.register_forward_pre_hook(lambda m, a: states.append(a[0].detach())))
    return states, handles


def last_loss(model, tokens, position_ids=None):
    import torch.nn.functional as F
    logits = model(tokens[:, :-1], position_ids=position_ids)
    return F.cross_entropy(logits[:, -1].float(), tokens[:, -1])


def compare(name):
    """(e): similarity across depth and positions, and whether similar layers/tokens can be removed."""
    import torch
    import torch.nn.functional as F
    from .p1a_depth import held_out_tokens, load
    variant, lr = COMPARE[name]
    model = load('final', 'cuda', variant, lr).eval()
    tokens = held_out_tokens('cuda')
    with torch.no_grad():
        states, handles = capture_stream(model)
        full = F.cross_entropy(model(tokens[:, :-1]).flatten(0, 1).float(), tokens[:, 1:].flatten()).item()
        for handle in handles:
            handle.remove()
        rms = [s.double().square().mean().sqrt().item() for s in states]
        adjacent = [F.cosine_similarity(a.float(), b.float(), dim=-1).mean().item() for a, b in zip(states, states[1:])]
        lag_cosine = {str(lag): [F.cosine_similarity(s[:, lag:].float(), s[:, :-lag].float(), dim=-1).mean().item()
                                 for s in states] for lag in LAGS}
        # Layer removal: zero one branch's update (its skip connection stays), measure the full loss.
        removal = []
        for branch in model.branches():
            target = getattr(branch, 'operation', branch)
            handle = target.register_forward_hook(lambda m, a, out: torch.zeros_like(out))
            removal.append(F.cross_entropy(model(tokens[:, :-1]).flatten(0, 1).float(),
                                           tokens[:, 1:].flatten()).item() - full)
            handle.remove()
        # Token removal: delete one input token `lag` positions before the last query, keep the
        # original positions of the rest, and measure the change in the last-token loss.
        length = tokens.shape[1] - 1
        base = last_loss(model, tokens).item()
        token_removal = {}
        for lag in LAGS:
            keep = torch.tensor([i for i in range(tokens.shape[1]) if i != length - 1 - lag], device=tokens.device)
            positions = keep[:-1][None]
            token_removal[str(lag)] = last_loss(model, tokens[:, keep], positions).item() - base
        last_cosine = {str(lag): F.cosine_similarity(states[-1][:, length - 1].float(),
                                                    states[-1][:, length - 1 - lag].float(), dim=-1).mean().item()
                       for lag in LAGS}
    return dict(name=name, variant=variant, lr=lr, full_loss=full, last_loss=base, activation_rms=rms,
                adjacent_cosine=adjacent, lag_cosine=lag_cosine, layer_removal=removal,
                token_removal=token_removal, last_position_cosine=last_cosine)


def context_use(name):
    """(f): loss increase on the same final targets with shorter prefixes, with/without the BOS token,
    plus the last-token gradient reach at the input embedding."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from data import dclm_val_dataset, load_token_dataset
    from .diagnostics import context_loss
    from .p1a_depth import held_out_tokens, load
    variant, lr = CONTEXT[name]
    model = load('final', 'cuda', variant, lr).eval()
    validation = load_token_dataset(dclm_val_dataset(), role='val', data_seed=42)
    tokens = torch.tensor(np.asarray(validation[:CONTEXT_SEQUENCES]['input_ids']).astype('int64'), device='cuda')
    length = tokens.shape[1] - 1
    batches = tokens.split(32)
    full = torch.cat([context_loss(model, b, length) for b in batches])
    increase = {}
    for preserve_first in (False, True):
        increase[f'bos_{preserve_first}'] = {
            str(keep): (torch.cat([context_loss(model, b, keep, preserve_first) for b in batches]) - full).tolist()
            for keep in KEEPS}
    # Gradient reach: RMS of dL_T / d(embedding) at each input position (L_T = last-token loss).
    sample = held_out_tokens('cuda')
    embedded = []
    embedding = model.get_input_embeddings()
    handle = embedding.register_forward_hook(lambda m, a, out: (out.retain_grad(), embedded.append(out))[1])
    with torch.enable_grad(), torch.autocast('cuda', enabled=False):
        model.zero_grad(set_to_none=True)
        logits = model(sample[:, :-1])
        F.cross_entropy(logits[:, -1].float(), sample[:, -1]).backward()
    handle.remove()
    grad = embedded[0].grad.double()
    reach = grad.square().mean((0, 2)).sqrt().tolist()  # one value per input position
    model.zero_grad(set_to_none=True)
    return dict(name=name, variant=variant, lr=lr, keeps=list(KEEPS), full_loss=full.tolist(),
                increase=increase, gradient_reach=reach)


def run(job):
    kind, name = job
    return kind, name, dict(diagnose=diagnose, compare=compare, context=context_use)[kind](name)


if __name__ == '__main__':
    import modal
    from modal_utils import app, build_image, VOLUME_MOUNTS, MODAL_ENVIRONMENT
    from .diagnostics import save
    jobs = ([('diagnose', name) for name in [*BEST_LR, *EXTRA]] + [('compare', name) for name in COMPARE]
            + [('context', name) for name in CONTEXT])
    probe = app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu='H100', timeout=3600)(run)
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT):
        for kind, name, result in probe.map(jobs):
            path = OUTPUT / kind / f'{name}.json'
            save(path, result)
            print(path)
