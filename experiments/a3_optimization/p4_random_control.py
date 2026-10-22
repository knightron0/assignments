"""P4(a): random partitions matched to family/layer block sizes at all checkpoints."""
import json

from experiments.a3_optimization.hessian_checkpoints import STEPS
from experiments.a3_optimization.hessian_jobs import launch, RESULTS_DIR

JOBS = [
    {"name": f"a-random-adamw-{step}", "measure": {
        "run": "adamw", "step": step, "extremes": False, "families": False,
        "random_energy": ("family", "layer"), "random_partitions": 2, "energy_probes": 4}}
    for step in STEPS
]


def main():
    calls = launch(JOBS, max_parallel=2)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "jobs-random-control.json"
    output.write_text(json.dumps({"jobs": JOBS, "calls": calls}, indent=2) + "\n")
    print(f"Saved job manifest to {output}")


if __name__ == "__main__":
    main()
