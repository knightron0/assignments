"""P1(b) evaluation-time interventions and measurements on the final best-LR checkpoints (no training).

    uv run python -m experiments.a4_stability.p1b_interventions

Interventions (loss change on held-out data when one thing is broken inside the trained model):
  mask_bos / mask_boundary  no query may attend to position 0 (and to EOS tokens), renormalised
  local_heads / random_heads  zero the outputs of the model's local heads (>=50% of attention on the
                             previous 8 tokens), or of the same number of random other heads
  gates_mean                every learned gate replaced by its mean value (input dependence removed)
  layer_removal             zero one sublayer's update at a time
Measurements: update size |F|/|x| per sublayer, attention logit scale and entropy, logit RMS.
Writes results/a4_stability/p1b/interventions/<name>.json.
"""
import json
import types
from pathlib import Path

OUTPUT = Path('results/a4_stability/p1b/interventions')
SEQUENCES = 128  # loss-based interventions
ATTENTION_SEQUENCES = 8  # logit/entropy statistics
LATE = 512
EOS = 2
LOCAL_SHARE = .5
RANDOM_DRAWS = 3
EXTRA = {'d8-no-qknorm@0.006': ('d8-no-qknorm', .006), 'd8-no-qknorm@0.012': ('d8-no-qknorm', .012),
         'd8@0.006': ('d8', .006), 'd8@0.012': ('d8', .012)}  # attention-logit comparison only


def local_heads(variant):
    """Heads whose final attention puts >= LOCAL_SHARE on lags 1-8 (from p1b_measure output)."""
    import numpy as np
    layers = json.loads(Path(f'results/a4_stability/p1b/measurements/{variant}.json').read_text())
    layers = layers['attention']['final']['layers']
    return [(i, h) for i, layer in enumerate(layers)
            for h, row in enumerate(np.asarray(layer['mass_by_lag'])) if row[1:9].sum() >= LOCAL_SHARE]


def patch_attention(model):
    """Swap each attention forward for one that honours module.extra_mask and module.head_scale."""
    import torch
    import torch.nn.functional as F
    from modeling import repeat_kv
    from .p1_transformer_depth import Attention

    def forward(self, x, *, position_embeddings, attention_mask=None, **context):
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotate(q, k, position_embeddings)
        q, k = self.q_norm(q), self.k_norm(k)
        k, v = repeat_kv(k, self.num_key_value_groups), repeat_kv(v, self.num_key_value_groups)
        if self.extra_mask is None:
            heads = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            heads = F.scaled_dot_product_attention(q, k, v, attn_mask=self.extra_mask)
        heads = self.gate_heads(heads.transpose(1, 2), x)
        if self.head_scale is not None:
            heads = heads * self.head_scale[:, None]
        return self.o_proj(heads.reshape(batch, length, -1))

    modules = [m for m in model.modules() if isinstance(m, Attention)]
    for module in modules:
        module.extra_mask, module.head_scale = None, None
        module.forward = types.MethodType(forward, module)
    return modules


def blocked_mask(tokens, boundary):
    """(B, 1, T, T) boolean 'may attend' mask: causal, minus key 0 (and EOS keys); self always allowed."""
    import torch
    length = tokens.shape[1]
    position = torch.arange(length, device=tokens.device)
    allowed = position[None, :] <= position[:, None]
    blocked = (position == 0)[None].expand(tokens.shape[0], -1)
    if boundary:
        blocked = blocked | (tokens == EOS)
    diagonal = torch.eye(length, dtype=torch.bool, device=tokens.device)
    return (allowed[None] & (~blocked[:, None, :] | diagonal[None]))[:, None]


def mean_loss(model, batches, attentions=None, boundary=None):
    import torch
    import torch.nn.functional as F
    total, count = 0., 0
    with torch.no_grad():
        for tokens in batches:
            inputs = tokens[:, :-1]
            if attentions is not None and boundary is not None:
                mask = blocked_mask(inputs, boundary)
                for module in attentions:
                    module.extra_mask = mask
            logits = model(inputs)
            loss = F.cross_entropy(logits.flatten(0, 1).float(), tokens[:, 1:].flatten(), reduction='sum')
            total, count = total + loss.item(), count + tokens[:, 1:].numel()
    if attentions is not None:
        for module in attentions:
            module.extra_mask = None
    return total / count


def gate_modules(model):
    """Every learned sigmoid gate: residual (.gate), attention (.operation.gate) and norm (.norm.gate)."""
    import torch.nn as nn
    return {name: m for name, m in model.named_modules() if name.endswith('.gate') and isinstance(m, nn.Linear)}


def gates_mean(model, batches):
    """Record each gate's mean sigmoid value per channel, then rerun with the gate fixed at that mean."""
    import torch
    gates = gate_modules(model)
    if not gates:
        return None
    sums = {name: 0. for name in gates}
    counts = {name: 0 for name in gates}
    handles = []
    for name, module in gates.items():
        def record(m, a, out, name=name):
            sums[name] = sums[name] + out.float().sigmoid().sum((0, 1))
            counts[name] += out.shape[0] * out.shape[1]
        handles.append(module.register_forward_hook(record))
    base = mean_loss(model, batches)
    for handle in handles:
        handle.remove()
    means = {name: (sums[name] / counts[name]).clamp(1e-6, 1 - 1e-6) for name in gates}
    handles = [module.register_forward_hook(
        lambda m, a, out, mean=means[name]: torch.logit(mean).to(out.dtype).expand_as(out).clone())
        for name, module in gates.items()]
    fixed = mean_loss(model, batches)
    for handle in handles:
        handle.remove()
    return dict(loss=fixed - base,
                mean_gate={name: means[name].mean().item() for name in gates},
                gate_spread={name: means[name].std().item() for name in gates})


def head_ablation(model, attentions, batches, heads, base):
    import torch
    for module in attentions:
        module.head_scale = torch.ones(module.num_heads, device=batches[0].device)
    for layer, head in heads:
        attentions[layer].head_scale[head] = 0.
    loss = mean_loss(model, batches) - base
    for module in attentions:
        module.head_scale = None
    return loss


def update_sizes(model, batches):
    """RMS of each sublayer's update F(.) divided by the RMS of its input stream x."""
    import torch
    branches = model.branches()
    inputs, outputs, handles = {}, {}, []
    for i, branch in enumerate(branches):
        handles.append(branch.register_forward_pre_hook(
            lambda m, a, i=i: inputs.__setitem__(i, a[0].double().square().mean().sqrt().item())))
        handles.append(branch.operation.register_forward_hook(
            lambda m, a, out, i=i: outputs.__setitem__(i, out.double().square().mean().sqrt().item())))
    with torch.no_grad():
        model(batches[0][:, :-1])
    for handle in handles:
        handle.remove()
    return [outputs[i] / inputs[i] for i in range(len(branches))]


def attention_stats(model, tokens):
    """Per layer: largest |q.k|/sqrt(d), mean attention entropy (nats) and share of late queries
    whose top key gets > 90% of the attention."""
    import math
    import torch
    from .p1a_attention import capture_qk
    store, handles = capture_qk(model)
    with torch.no_grad():
        logits = model(tokens[:, :-1])
    for handle in handles:
        handle.remove()
    stats = []
    for i in sorted(store):
        peak, entropy, spiky = 0., 0., 0.
        for b in range(tokens.shape[0]):
            q, k = store[i]['q'][b].float(), store[i]['k'][b].float()
            k = k.repeat_interleave(q.shape[0] // k.shape[0], 0)
            scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
            mask = torch.ones(scores.shape[-1], scores.shape[-1], dtype=torch.bool, device=q.device).triu(1)
            scores = scores.masked_fill(mask, -torch.inf)
            peak = max(peak, scores[:, LATE:].masked_fill(mask[LATE:], 0).abs().max().item())
            p = scores[:, LATE:].softmax(-1)
            entropy += -(p * p.clamp_min(1e-30).log()).sum(-1).mean().item()
            spiky += (p.max(-1).values > .9).float().mean().item()
        n = tokens.shape[0]
        stats.append(dict(max_logit=peak, entropy=entropy / n, spiky=spiky / n))
    logit_rms = logits.float().std(-1).mean().item()
    return stats, logit_rms


def run(job):
    import numpy as np
    import torch
    from data import dclm_val_dataset, load_token_dataset
    from .p1a_depth import load
    name, variant, lr, heads = job
    model = load('final', 'cuda', variant, lr).eval()
    validation = load_token_dataset(dclm_val_dataset(), role='val', data_seed=42)
    tokens = torch.tensor(np.asarray(validation[:SEQUENCES]['input_ids']).astype('int64'), device='cuda')
    attention, logit_rms = attention_stats(model, tokens[:ATTENTION_SEQUENCES])
    result = dict(name=name, variant=variant, lr=lr, attention=attention, logit_rms=logit_rms)
    if name in EXTRA:
        return result
    batches = list(tokens.split(32))
    attentions = patch_attention(model)
    base = mean_loss(model, batches)
    result.update(loss=base,
                  mask_bos=mean_loss(model, batches, attentions, boundary=False) - base,
                  mask_boundary=mean_loss(model, batches, attentions, boundary=True) - base,
                  update_size=update_sizes(model, batches),
                  gates_mean=gates_mean(model, batches))
    # Local vs random heads, matched in count *and layer* (layers differ hugely in importance).
    heads = [tuple(h) for h in heads]
    rng = np.random.default_rng(0)

    def matched(chosen):
        picked = []
        for layer in sorted({i for i, _ in chosen}):
            count = sum(i == layer for i, _ in chosen)
            pool = [h for h in range(attentions[layer].num_heads) if (layer, h) not in heads]
            if len(pool) < count:  # nearly every head in this layer is local: fill from the rest
                pool += [h for h in range(attentions[layer].num_heads) if (layer, h) in heads and h not in pool]
            picked += [(layer, h) for h in rng.choice(pool, count, replace=False)]
        return picked

    def compare(chosen):
        return (head_ablation(model, attentions, batches, chosen, base),
                [head_ablation(model, attentions, batches, matched(chosen), base) for _ in range(RANDOM_DRAWS)])

    if heads:
        loss, random = compare(heads)
        result['local_heads'], result['random_heads'] = dict(heads=heads, loss=loss), random
    else:
        result['local_heads'], result['random_heads'] = dict(heads=[], loss=0.), []
    if variant == 'd8':  # what early (layer 0-2) locality is worth, to set against no-RoPE's gap
        early = [h for h in heads if h[0] <= 2]
        loss, random = compare(early)
        result['early_local_heads'], result['early_random_heads'] = dict(heads=early, loss=loss), random
    # Layer removal: zero one sublayer's update (skip connection kept).
    removal = []
    for branch in model.branches():
        handle = branch.operation.register_forward_hook(lambda m, a, out: torch.zeros_like(out))
        removal.append(mean_loss(model, batches) - base)
        handle.remove()
    result['layer_removal'] = removal
    return result


if __name__ == '__main__':
    import modal
    from modal_utils import app, build_image, VOLUME_MOUNTS, MODAL_ENVIRONMENT
    from .diagnostics import save
    from .p1b_measure import BEST_LR
    jobs = [(v, v, lr, local_heads(v)) for v, lr in BEST_LR.items()]
    jobs += [(name, v, lr, []) for name, (v, lr) in EXTRA.items()]
    probe = app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu='H100', timeout=3600)(run)
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT):
        for result in probe.map(jobs):
            path = OUTPUT / f"{result['name']}.json"
            save(path, result)
            print(path, result.get('loss'), result.get('mask_bos'))
