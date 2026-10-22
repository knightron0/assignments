"""P4(c): isolate subspace size, then compare earlier/later training stages.

    uv run python -m experiments.a3_optimization.p4_subspace_sweep

Checkpoint comparisons retain the shipped schedule, so starting learning rates
differ by training stage. No model or optimizer hyperparameters are retuned.
"""
import json

from experiments.a3_optimization.hessian_checkpoints import STEPS
from experiments.a3_optimization.hessian_jobs import launch, RESULTS_DIR
from experiments.a3_optimization.p4_inside_the_hessian import START

ARMS = ("full", "top", "removed")
# A larger subspace requires a larger Lanczos basis to resolve all 50 pairs.
# Keep the same budget across all three arms of each comparison and log residuals.
K_JOBS = [
    {"name": f"c-k{k}-{arm}-100M", "continue": {
        **START, "k": k, "transform": arm,
        "lanczos_steps": 128 if k == 50 else 48}}
    for k in (1, 50)
    for arm in ARMS
]
STAGE_JOBS = [
    {"name": f"c-stage{step}-{arm}-k10", "continue": {
        **START, "step": step, "transform": arm}}
    for step in (153, 7630)  # approximately 10M and 500M tokens
    for arm in ARMS
]
JOBS = K_JOBS + STAGE_JOBS


def main():
    assert len({j["name"] for j in JOBS}) == len(JOBS)
    for job in JOBS:
        spec = job["continue"]
        assert spec["step"] in STEPS
        assert spec["step"] + spec["steps"] <= STEPS[-1]
        assert spec.get("lanczos_steps", 48) >= 2 * spec["k"] + 10
    calls = launch(JOBS, max_parallel=2)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "jobs-subspace-sweep.json"
    output.write_text(json.dumps({"jobs": JOBS, "calls": calls}, indent=2) + "\n")
    print(f"Saved job manifest to {output}")


if __name__ == "__main__":
    main()
