"""P4(a)/(b): GRU ablations and the norm-GRU proposal on Modal, in priority order.

    uv run python -m experiments.a4_stability.p4_launch

Already trained: gru at .0015/.003/.006 (best .006, 3.183) and gru-no-reset at .0015/.003.
GRU runs take 60-100 min on an H100 (Python time loop) and the workspace runs two at a time,
so the queue is ordered: every variant at .003 first, then .006, then the edge checks.
"""
from modal_train import launch_training_jobs

from .sweeps import config

ABLATIONS = ('gru-no-update', 'gru-fixed-update', 'gru-input-gates', 'gru-const-update',
             'gru-no-candidate-h', 'gru-min', 'gru-norm')
DONE = ('gru-no-update', 'gru-fixed-update')
# Deadline cut: only the .003 runs still missing a final checkpoint.
QUEUE = [(name, .003) for name in ABLATIONS if name not in DONE]


def launch(queue=QUEUE, *, gpu='H100', parallel=2, dry_run=False):
    configs = [config(name, lr, wandb_tags=('a4', 'p4', name)) for name, lr in queue]
    for cfg in configs:
        print(cfg.run_name, cfg.learning_rate)
    print(len(configs), 'runs')
    if not dry_run:
        return launch_training_jobs(configs, gpu=gpu, max_parallel_runs=parallel)


if __name__ == '__main__':
    launch()
