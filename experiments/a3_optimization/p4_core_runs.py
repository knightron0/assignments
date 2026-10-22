"""Focused P4 follow-ups; train jobs measure their own final model automatically.

    uv run python -m experiments.a3_optimization.p4_core_runs
"""

import json
from pathlib import Path

import torch

from experiments.a3_optimization.hessian_jobs import EXPERIMENT_KEY, launch
from experiments.a3_optimization.p4_inside_the_hessian import START
from modeling import AutoregressiveLM
from train import TrainConfig, training_run_name


class ScaledHeadLM(AutoregressiveLM):
    """Same initial function, different norm/head parameter coordinates."""

    def __init__(self, config, *, norm_scale=1.0, **kwargs):
        super().__init__(config, **kwargs)
        self.norm_scale = norm_scale

    @torch.no_grad()
    def post_initialize(self):
        if self.tie_word_embeddings:
            raise ValueError("Head rescaling requires untied embeddings")
        self.model.norm.weight.mul_(self.norm_scale)
        self.lm_head.weight.div_(self.norm_scale)


TENSORS = ["model.norm.weight", "lm_head.weight"]
MEASURE = {"extremes": True, "families": True, "tensors": TENSORS}
RESCALE_JOBS = [
    {"name": f"b-core-rescale-{scale:g}", "measure": {
        "run": "adamw", "step": 9375, **MEASURE,
        "check_scale_invariance": True,
        "scale": {"model.norm.weight": scale, "lm_head.weight": 1 / scale}}}
    for scale in (1.0, 0.5, 2.0, 4.0)
]

TRAIN_SETTINGS = [
    {"model_builder": "experiments.a3_optimization.p4_core_runs:ScaledHeadLM",
     "model_builder_kwargs": {"norm_scale": scale},
     "run_name_suffix": f"a3-p4-normscale{scale:g}"}
    for scale in (0.5, 2.0, 4.0)
] + [{"weight_decay": 0.3, "run_name_suffix": "a3-p4-wd0.3"}]

TRAIN_JOBS = [
    {"name": f"b-core-final-{s['run_name_suffix']}", "train_measure": s,
     "measure_options": MEASURE}
    for s in TRAIN_SETTINGS
]

CONTINUE_JOBS = [
    {"name": f"c-core-{arm}-lr0.25", "continue": {
        **START, "transform": arm, "lr_scale": 0.25}}
    for arm in ("full", "top", "removed")
] + [
    {"name": f"c-core-{arm}-refresh10", "continue": {
        **START, "transform": arm, "refresh": 10}}
    for arm in ("top", "removed")
]


def train_and_measure(settings, options):
    """Run both dependent stages on one GPU so measurements cannot start early."""
    from experiments.a3_optimization.hessian_measure import measure
    from train import train, evaluate, load_train_and_val_datasets
    from utils import create_batches

    config = TrainConfig(**settings, wandb_tags=(EXPERIMENT_KEY,))
    model = train(config)
    # Use the normal full validation protocol for the assignment's loss target.
    _, val = load_train_and_val_datasets(config)
    final_val = float(evaluate(model, create_batches(val["val"], config.batch_size), config, "cuda"))
    model.float().eval()
    result = measure(model, **options)
    return {"run": training_run_name(config), "step": 9375,
            "tokens": 614400000, "training_settings": settings,
            "full_val_loss": final_val, **result}


def main():
    jobs = RESCALE_JOBS + TRAIN_JOBS + CONTINUE_JOBS
    calls = launch(jobs, max_parallel=2)
    out = Path(__file__).with_name("results") / EXPERIMENT_KEY / "jobs-core.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"jobs": jobs, "calls": calls}, indent=2) + "\n")
    print(f"Saved job manifest to {out}")


if __name__ == "__main__":
    main()
