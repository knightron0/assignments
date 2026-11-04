"""Execution only: cuDNN tanh, compiled 16-token chunks, optional CUDA replay.

Algorithms live in the problem files. Every editable path calls cell.project()
and cell.step(); state is never detached between chunks. Probes run eagerly.
"""
import torch
from torch import nn
from module_rms_logging import log_module_rms

CHUNK_SIZE = 16
# Each layer's compiled cell_steps is guarded per module and mode; 4 layers x train/eval
# exceeds the default limit of 8 and raises mid-run.
torch._dynamo.config.cache_size_limit = 64


@torch.compiler.disable
def fused_tanh(sequence, x):
    if not hasattr(sequence, '_fused'):
        with torch.random.fork_rng(devices=[]):
            view = nn.RNN(x.shape[-1], x.shape[-1], batch_first=True)
        cell = sequence.cell
        view.weight_ih_l0, view.weight_hh_l0 = cell.input.weight, cell.hidden.weight
        view.bias_ih_l0, view.bias_hh_l0 = cell.input.bias, cell.hidden.bias
        # This is a view of existing parameters, not another registered layer.
        object.__setattr__(sequence, '_fused', view)
    view = sequence._fused
    view.train(sequence.training or torch.is_grad_enabled())
    return view(x.contiguous())[0]


def steps(cell, projected, h, states=None):
    outputs = []
    for t in range(projected[0].shape[1]):
        h = cell.step(tuple(p[:, t] for p in projected), h)
        if states is not None:
            h.retain_grad()
            states.append(h)
        outputs.append(h)
    return torch.stack(outputs, 1), h


def recurrence(sequence, x):
    projected = sequence.cell.project(x)
    h = x.new_zeros(x.shape[0], sequence.config.hidden_size)
    if sequence.record_states:
        sequence.states = []
        return steps(sequence.cell, projected, h, sequence.states)[0]
    # Sampling uses eager cell calls so the native module hooks see each step.
    compiled = x.is_cuda and not (log_module_rms.capture and sequence.training)
    if compiled and not hasattr(sequence, '_compiled'):
        sequence._compiled = torch.compile(sequence.cell_steps, mode='default', fullgraph=True)
    runner = sequence._compiled if compiled else sequence.cell_steps
    outputs = []
    for start in range(0, x.shape[1], CHUNK_SIZE):
        chunk = tuple(p[:, start:start+CHUNK_SIZE] for p in projected)
        out, h = runner(chunk, h)
        if compiled:
            # Own outputs before the next compiled CUDA call reuses buffers.
            out, h = out.clone(), h.clone()
        outputs.append(out)
    return torch.cat(outputs, 1)


@torch.compiler.disable
def run_stack(stack, inputs):
    """Capture the complete GRU stack, including its depth norms and MLPs."""
    if (inputs.is_cuda and stack.training and torch.is_grad_enabled()
            and not hasattr(stack, '_captured')):
        _capture(stack, inputs)
    return stack(inputs)


def _capture(stack, inputs):
    original = stack.forward
    enabled = torch.is_autocast_enabled('cuda')
    dtype = torch.get_autocast_dtype('cuda')
    with torch.autocast('cuda', enabled=enabled, dtype=dtype, cache_enabled=False):
        sample = inputs.detach().requires_grad_(True)
        torch.cuda.synchronize()
        # Capture before the first backward. Warmups must not enter RMS samples;
        # PyTorch also requires the root hook to be absent during capture.
        sampling, log_module_rms.capture = log_module_rms.capture, False
        hooks, stack._forward_hooks = stack._forward_hooks, {}
        try:
            torch.cuda.make_graphed_callables(stack, (sample,), num_warmup_iters=3, allow_unused_input=True)
        finally:
            stack._forward_hooks = hooks
            log_module_rms.capture = sampling
        replay = stack.forward
        torch.cuda.synchronize()
    # Captured backward buffers must not become accumulated parameter gradients.
    stack._grad_hooks = [p.register_hook(lambda g: g.clone()) for p in stack.parameters() if p.requires_grad]
    signature = (sample.shape, sample.dtype, sample.device, sample.stride())
    stack._inflight = False

    def backward_started(gradient):
        stack._inflight = False
        return gradient

    @torch.compiler.disable
    def forward(x, **context):
        matches = (x.shape, x.dtype, x.device, x.stride()) == signature
        if (stack.training and torch.is_grad_enabled() and matches and x.requires_grad
                and not stack._inflight and not context
                and not log_module_rms.capture
                and torch.is_autocast_enabled('cuda') == enabled
                and (not enabled or torch.get_autocast_dtype('cuda') == dtype)):
            stack._inflight = True
            x.register_hook(backward_started)
            return replay(x)
        return original(x, **context)

    stack.forward, stack._captured = forward, True
