"""P3: launch the fixed/full/blocked residual variants on Modal, two LRs each.

    uv run python -m experiments.a4_stability.p3_launch

Baselines (d8, rnn) reuse the supplied P1/P2 runs. The supplied d8 baseline is best at
.003 with flat neighbours and the rnn baseline at .001-.002, so the first pass brackets
each variant's center on that side: d8 at (.5, 1)x center, rnn at (1, 2)x center.
Extend with launch_p3(names, factors=...) toward whichever edge wins.
"""
from modal_train import launch_training_jobs

from .sweeps import VARIANTS, config

FIXED = ('half', 'ema01', 'ema09', 'uniform')  # (a)
FULL = ('attention',)  # (b)
BLOCKED = ('block2', 'block4')  # (c)
P3_VARIANTS = tuple(f'{family}-{v}' for family in ('d8', 'rnn') for v in FIXED + FULL + BLOCKED)
FACTORS = {'d8': (.5, 1.), 'rnn': (1., 2.)}
PARALLEL = 8  # sweeps.launch caps Modal at 2 containers; ~12-18 min per run on an H100.


def rates(name, factors=None):
    factors = FACTORS[name.split('-')[0]] if factors is None else factors
    return tuple(float(f'{VARIANTS[name].center*f:.8g}') for f in factors)


# Third rate toward whichever edge won the first pass. d8-ema01 collapsed at both rates.
EXTEND = {'d8-half': 2., 'd8-ema09': 2., 'd8-uniform': 2., 'd8-attention': 2., 'd8-block2': .25,
          'd8-block4': 2., 'rnn-half': .5, 'rnn-ema01': .5, 'rnn-ema09': .5, 'rnn-uniform': 4.,
          'rnn-attention': 4., 'rnn-block2': .5, 'rnn-block4': .5}


def launch_p3(names=P3_VARIANTS, *, factors=None, gpu='H100', parallel=PARALLEL, dry_run=False):
    configs = [config(name, lr, wandb_tags=('a4', 'p3', name))
               for name in names for lr in rates(name, factors)]
    for cfg in configs:
        print(cfg.run_name, cfg.learning_rate)
    print(len(configs), 'runs')
    if not dry_run:
        return launch_training_jobs(configs, gpu=gpu, max_parallel_runs=min(parallel, len(configs)))


def launch_extensions(*, gpu='H200', dry_run=False):
    """H200: full-attention d8 runs out of memory on an H100 at batch 64."""
    configs = [config(name, lr, wandb_tags=('a4', 'p3', name))
               for name, f in EXTEND.items() for lr in rates(name, (f,))]
    for cfg in configs:
        print(cfg.run_name, cfg.learning_rate)
    print(len(configs), 'runs')
    if not dry_run:
        return launch_training_jobs(configs, gpu=gpu, max_parallel_runs=min(PARALLEL, len(configs)))


if __name__ == '__main__':
    launch_p3(dry_run=True)  # Remove dry_run to launch.
