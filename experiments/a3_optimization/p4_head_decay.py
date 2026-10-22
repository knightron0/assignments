"""P4(b): change head decay during training, with ordinary initialization.

    uv run python -m experiments.a3_optimization.p4_head_decay
"""
import argparse
import json

from experiments.a3_optimization.hessian_jobs import launch, RESULTS_DIR
from experiments.a3_optimization.p4_core_runs import MEASURE

def jobs_for(factors):
    return [
    {"name": f"b-headwdx{factor:g}-final", "train_measure": {
        "head_weight_decay_multiplier": float(factor),
        "run_name_suffix": f"a3-p4-headwdx{factor:g}"},
     "measure_options": MEASURE}
    for factor in factors
]

JOBS = jobs_for((1, 5, 20))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", nargs="+", type=float, default=[1, 5, 20])
    args = parser.parse_args()
    jobs = jobs_for(args.factors)
    calls = launch(jobs, max_parallel=2)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.factors == [1, 5, 20] else "-" + "-".join(f"{f:g}" for f in args.factors)
    output = RESULTS_DIR / f"jobs-head-decay{suffix}.json"
    output.write_text(json.dumps({"jobs": jobs, "calls": calls}, indent=2) + "\n")
    print(f"Saved job manifest to {output}")


if __name__ == "__main__":
    main()
