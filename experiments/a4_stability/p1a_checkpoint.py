"""P1(a): depth and attention diagnostics on the supplied d8 checkpoints, on Modal.

    uv run python -m experiments.a4_stability.p1a_checkpoint

Reads the final and step-100 checkpoints and validation data from the shared volume
(like measure_default.py) and writes the returned measurements to results/a4_stability/p1a/.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .diagnostics import profile, save
from .p1_transformer_depth import Attention

RUN = 'a4-d8-lr0.003'
REFERENCE_ROOT = Path('/root/shared_data/a4_stability')
OUTPUT = Path('results/a4_stability/p1a')
SEQUENCES = 16  # the first 16 validation sequences, plus 16 more that contain an EOS in [1, LATE)
LATE = 512  # "late" queries: q >= LATE, where uniform attention is spread thin
BOS, EOS = 1, 2
MAP_SIZE = 64


def validation_tokens(count=SEQUENCES, device='cpu'):
    from data import dclm_val_dataset, load_token_dataset
    validation = load_token_dataset(dclm_val_dataset(), role='val', data_seed=42)
    tokens = np.asarray(validation[:len(validation)]['input_ids'])
    with_eos = [i for i in range(count, len(tokens)) if (tokens[i, 1:LATE] == EOS).any()][:count]
    return torch.tensor(tokens[list(range(count)) + with_eos].astype('int64'), device=device)


def capture_qk(model):
    """Hook each attention layer's normalized (post-RoPE) queries and keys."""
    store, handles = {}, []
    layers = [m for m in model.modules() if isinstance(m, Attention)]
    for i, module in enumerate(layers):
        for label, norm in (('q', module.q_norm), ('k', module.k_norm)):
            def hook(module, args, output, i=i, label=label):
                store.setdefault(i, {})[label] = output.detach()
            handles.append(norm.register_forward_hook(hook))
    return store, handles


def attention_probs(q, k):
    """Exactly the model's causal softmax(qk^T/sqrt(d)), in FP32. q: (H, T, d), k: (Hkv, T, d)."""
    k = k.repeat_interleave(q.shape[0] // k.shape[0], 0)
    length = q.shape[1]
    scores = q.float() @ k.float().transpose(-1, -2) / math.sqrt(q.shape[-1])
    mask = torch.ones(length, length, dtype=torch.bool, device=q.device).triu(1)
    return scores.masked_fill(mask, -torch.inf).softmax(-1)


def attention_summary(store, tokens):
    """Per layer and head: mass vs lag, self/first mass vs query, received mass vs key."""
    length = tokens.shape[1]
    queries = torch.arange(length, device=tokens.device)
    late = queries[LATE:]
    lags = torch.arange(LATE, device=tokens.device)
    gather = (late[:, None] - lags[None, :])  # key index for each (late query, lag)
    result = []
    for i in sorted(store):
        acc = dict(lag=0., self=0., first=0., received=0., bos=0., eos=[], other=[])
        for b in range(tokens.shape[0]):
            p = attention_probs(store[i]['q'][b], store[i]['k'][b])  # (H, T, T)
            heads = p.shape[0]
            p_late = p[:, LATE:]
            acc['lag'] += p_late.gather(-1, gather.expand(heads, -1, -1)).mean(1)
            acc['self'] += p.diagonal(dim1=-2, dim2=-1)
            acc['first'] += p[..., 0]
            # Mass a key receives from late queries, divided by what uniform attention gives it.
            ratio = (p_late * (late[:, None] + 1).float()).mean(1)  # (H, T)
            acc['received'] += ratio[:, :LATE]
            acc['bos'] += ratio[:, 0]
            is_eos = (tokens[b, :LATE] == EOS)
            if is_eos.any():
                acc['eos'].append(ratio[:, :LATE][:, is_eos])
            acc['other'].append(ratio[:, 1:LATE][:, ~is_eos[1:]])
            if b == 0:
                maps = p[:, :MAP_SIZE, :MAP_SIZE]
        n = tokens.shape[0]
        uniform_lag = (1/(late.float()+1)).mean().item()  # uniform mass per lag (lag < LATE always allowed)
        result.append(dict(
            mass_by_lag=(acc['lag']/n).tolist(), uniform_mass_per_lag=uniform_lag,
            self_mass=(acc['self']/n).tolist(), first_mass=(acc['first']/n).tolist(),
            uniform_by_query=(1/(queries.float()+1)).tolist(),
            received_ratio=(acc['received']/n).tolist(),
            bos_ratio=(acc['bos']/n).tolist(),
            eos_ratio=torch.cat(acc['eos'], 1).mean(1).tolist() if acc['eos'] else None,
            eos_count=int(sum(e.shape[1] for e in acc['eos'])),
            other_ratio=torch.cat(acc['other'], 1).mean(1).tolist(),
            example_maps=maps.tolist()))
    return result


def measure(run=RUN, checkpoint='final', root=REFERENCE_ROOT, device='cuda'):
    from model_io import load_model
    path = Path(root) / run / ('early' if checkpoint == 'early' else '')
    model = load_model(path, device=device, precision='fp32')
    tokens = validation_tokens(device=device)
    stats = profile(model, tokens)  # (i): activation and activation-gradient RMS, losses L and L_T
    store, handles = capture_qk(model)
    try:
        with torch.no_grad():
            model.eval()
            model(tokens[:, :-1])
    finally:
        for handle in handles:
            handle.remove()
    stats['attention_detail'] = attention_summary(store, tokens[:, :-1])
    stats['sequences'] = tokens.shape[0]
    stats['run'], stats['checkpoint'] = run, checkpoint
    return stats


def measure_both(run=RUN):
    return {checkpoint: measure(run, checkpoint) for checkpoint in ('final', 'early')}


if __name__ == '__main__':
    import modal
    from modal_utils import app, build_image, VOLUME_MOUNTS, MODAL_ENVIRONMENT
    probe = app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu='H100', timeout=1800)(measure_both)
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT):
        results = probe.remote()
    for checkpoint, stats in results.items():
        path = OUTPUT / f'{RUN}-{checkpoint}.json'
        save(path, stats)
        print(path)
