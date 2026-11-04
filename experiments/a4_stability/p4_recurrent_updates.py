"""Problem 4: edit the temporal update, keeping the modern RNN's depth stack.

    uv run python -m experiments.a4_stability.p4_recurrent_updates

z is the fraction of NEW information, following the handout. Reset is applied
BEFORE the candidate's recurrent projection; torch.nn.GRU uses a different form.
"""
import torch
from torch import nn
from modeling import LlamaRMSNorm
from .p2_rnn_depth import TanhCell, ModernRNN


class Gate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input = nn.Linear(config.hidden_size, config.hidden_size)
        self.hidden = nn.Linear(config.hidden_size, config.hidden_size)

    def project(self, x):
        return self.input(x)

    def forward(self, projected, h):
        return (projected + self.hidden(h)).sigmoid()


class GRUCell(TanhCell):
    def __init__(self, config):
        super().__init__(config)
        self.reset = self.make_reset(config)
        self.update = self.make_update(config)

    def make_reset(self, config):
        return Gate(config)

    def make_update(self, config):
        return Gate(config)

    def project(self, x):
        return self.input(x), self.reset.project(x), self.update.project(x)

    def step(self, projected, h):
        candidate_input, reset_input, update_input = projected
        r, z = self.reset(reset_input, h), self.update(update_input, h)
        candidate = torch.tanh(candidate_input + self.hidden(r*h))
        return self.combine(h, candidate, z)

    def combine(self, h, candidate, z):
        return (1-z)*h + z*candidate


class GRU(ModernRNN):
    cell_type = GRUCell


# (a) Ablations: each removes or freezes one GRU component.

class InputGate(Gate):
    """sigma(W x + b): the gate sees only the current input, not h."""
    def __init__(self, config):
        nn.Module.__init__(self)
        self.input = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, projected, h):
        return projected.sigmoid()


class NoResetCell(GRUCell):
    """r = 1: the candidate sees the whole old state."""
    def make_reset(self, config):
        return None

    def project(self, x):
        return self.input(x), self.update.project(x)

    def step(self, projected, h):
        candidate_input, update_input = projected
        z = self.update(update_input, h)
        return self.combine(h, torch.tanh(candidate_input + self.hidden(h)), z)


class NoUpdateCell(GRUCell):
    """z = 1: h = candidate, i.e. a tanh RNN with a reset gate (no interpolation path)."""
    def make_update(self, config):
        return None

    def project(self, x):
        return self.input(x), self.reset.project(x)

    def step(self, projected, h):
        candidate_input, reset_input = projected
        r = self.reset(reset_input, h)
        return self.combine(h, torch.tanh(candidate_input + self.hidden(r*h)), 1.)


class FixedUpdateCell(NoUpdateCell):
    """z = 1/2: keep the interpolation path but remove its data dependence."""
    def combine(self, h, candidate, z):
        return .5*h + .5*candidate


class InputGatesCell(GRUCell):
    """Gates read only x; h enters only through the candidate."""
    def make_reset(self, config):
        return InputGate(config)

    def make_update(self, config):
        return InputGate(config)


class ConstUpdateCell(NoUpdateCell):
    """z = sigmoid(b): a learned per-channel rate, independent of x and h (starts at 1/2)."""
    def __init__(self, config):
        super().__init__(config)
        self.update_bias = nn.Parameter(torch.zeros(config.hidden_size))

    def combine(self, h, candidate, z):
        z = self.update_bias.sigmoid()
        return (1-z)*h + z*candidate


class NoCandidateHiddenCell(NoResetCell):
    """candidate = tanh(W x): the old state enters only through the update gate and interpolation."""
    def __init__(self, config):
        super().__init__(config)
        self.hidden = None  # U_h is unused; a parameter with no gradient breaks the captured backward

    def step(self, projected, h):
        candidate_input, update_input = projected
        z = self.update(update_input, h)
        return self.combine(h, torch.tanh(candidate_input), z)


class MinCell(NoCandidateHiddenCell):
    """minGRU-style: z and the candidate read only x, so h_t is linear in h_{t-1}."""
    def make_update(self, config):
        return InputGate(config)


# (b) Proposal: pre-norm across time. Gates and candidate read RMSNorm(h_{t-1});
# the interpolation still carries the raw h, so the state is unchanged in kind.

class NormGRUCell(GRUCell):
    def __init__(self, config):
        super().__init__(config)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=1e-6)

    def step(self, projected, h):
        candidate_input, reset_input, update_input = projected
        n = self.norm(h)
        r, z = self.reset(reset_input, n), self.update(update_input, n)
        candidate = torch.tanh(candidate_input + self.hidden(r*n))
        return self.combine(h, candidate, z)


class GRUNoReset(ModernRNN):
    cell_type = NoResetCell


class GRUNoUpdate(ModernRNN):
    cell_type = NoUpdateCell


class GRUFixedUpdate(ModernRNN):
    cell_type = FixedUpdateCell


class GRUInputGates(ModernRNN):
    cell_type = InputGatesCell


class NormGRU(ModernRNN):
    cell_type = NormGRUCell


class GRUConstUpdate(ModernRNN):
    cell_type = ConstUpdateCell


class GRUNoCandidateHidden(ModernRNN):
    cell_type = NoCandidateHiddenCell


class GRUMin(ModernRNN):
    cell_type = MinCell


MODELS = dict(gru=GRU, tanh=ModernRNN, no_reset=GRUNoReset, no_update=GRUNoUpdate,
              fixed_update=GRUFixedUpdate, input_gates=GRUInputGates, norm=NormGRU,
              const_update=GRUConstUpdate, no_candidate_h=GRUNoCandidateHidden, min=GRUMin)


if __name__ == "__main__":
    from .sweeps import launch_one
    launch_one('gru')  # Edit the variant and LR here.
