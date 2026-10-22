"""Plot P2(c) cached W&B measurements: python -m experiments.a3_optimization.plot_p2_rewarm."""

import csv
import json
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.a3_optimization.p2_river_valley import (
    EXPERIMENT_KEY, RESULTS_DIR, REWARM_RUNS, SCHEDULE_RUNS,
)
from train import training_run_name
from utils import WANDB_ENTITY, WANDB_PROJECT


def main():
    cache = RESULTS_DIR / "data" / "rewarm-history.json"
    control = next(c for c in SCHEDULE_RUNS
                   if c.lr_schedule == "wsd0.2" and c.learning_rate == 0.003)
    configs = [control, *REWARM_RUNS]
    if cache.exists():
        histories = json.loads(cache.read_text())
    else:
        import wandb
        wanted = {training_run_name(c) for c in configs}
        latest = {}
        for run in wandb.Api().runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}",
                                    filters={"tags": EXPERIMENT_KEY}):
            if run.name in wanted and run.state == "finished" and run.name not in latest:
                latest[run.name] = run
        if wanted - latest.keys():
            raise RuntimeError(f"Missing finished runs: {wanted - latest.keys()}")

        def read(name):
            run = latest[name]
            series = {}
            for key in ("val_loss", "learning_rate"):
                series[key] = [[row["optimizer_step"] + 1, row[key]] for row in
                               run.scan_history(keys=["optimizer_step", key], page_size=10000)]
            return name, {"url": run.url, **series}

        histories = dict(ThreadPoolExecutor(max_workers=4).map(read, wanted))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(histories, indent=2))

    for h in histories.values():
        for key in ("val_loss", "learning_rate"):
            h[key] = [[step, value] for step, value in h[key] if value is not None]
    ref = histories[training_run_name(control)]
    ref_vals = dict(ref["val_loss"])
    total = ref["val_loss"][-1][0]
    data = []
    for config in REWARM_RUNS:
        h = histories[training_run_name(config)]
        common = [(step, val - ref_vals[step]) for step, val in h["val_loss"] if step in ref_vals]
        if not common or common[-1][0] != total:
            raise RuntimeError("Missing matching final evaluation")
        data.append((h, np.array(common)))

    out = RESULTS_DIR / "c-rewarm"
    out.mkdir(parents=True, exist_ok=True)
    groups = [([1, 0, 2], ["To 30% LR", "To 10% LR", "To zero LR"], "Vary dip depth"),
              ([3, 0, 4], ["4% duration", "10% duration", "20% duration"], "Vary dip duration")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    colors = ["#0072B2", "#D55E00", "#009E73"]
    def draw_group(row_axes, group):
        indices, labels, title = group
        lr_ax, regret_ax, final_ax = row_axes
        control_lr = np.array(ref["learning_rate"])
        lr_ax.plot(control_lr[:, 0] / total * 100, control_lr[:, 1] * 1000,
                   "--", color="0.25", label="Control", linewidth=1.5)
        for i, label, color in zip(indices, labels, colors):
            h, regret = data[i]
            lr = np.array(h["learning_rate"])
            lr_ax.plot(lr[:, 0] / total * 100, lr[:, 1] * 1000,
                       color=color, label=label, linewidth=1.8)
            for ax in (regret_ax, final_ax):
                ax.plot(regret[:, 0] / total * 100, regret[:, 1], color=color,
                        label=f"{label} (final {regret[-1, 1]:+.4f})", linewidth=1.8)
        lr_ax.set(title=f"{title}: learning rate", ylabel="Learning rate (x 0.001)", ylim=(-0.08, 3.2))
        regret_ax.set(title="Regret over training", ylabel="Validation loss minus control",
                      ylim=(-0.16, 0.035))
        final_ax.set(title="Final cooldown: close-up", xlim=(80, 100), ylim=(-0.003, 0.024))
        for ax in row_axes:
            ax.axvspan(80, 100, color="0.5", alpha=0.1)
            ax.set_xlabel("Training completed (%)")
            ax.grid(alpha=0.2)
            if ax is not lr_ax:
                ax.axhline(0, color="0.3", linestyle="--", linewidth=1)
        lr_ax.legend(fontsize=8)
        regret_ax.legend(fontsize=8, loc="lower left")
    for row, group in enumerate(groups):
        draw_group(axes[row], group)
    fig.suptitle("P2(c): temporary LR dips and rewarmup\n"
                 "Negative regret is better; shaded region is the identical final cooldown", fontsize=14)
    fig.savefig(out / "rewarm-comparison.png", dpi=180, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(out / "rewarm-comparison.svg", bbox_inches="tight", pad_inches=0.2)
    # Render report figures independently so adjacent rows cannot leak into crops.
    for name, group in zip(("depth", "duration"), groups):
        report_fig, report_axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
        draw_group(report_axes, group)
        report_fig.suptitle("P2(c): " + group[2] + "\nNegative regret is better; shaded region is the shared final cooldown",
                            fontsize=13)
        report_fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight", pad_inches=0.2)
        report_fig.savefig(out / f"{name}.svg", bbox_inches="tight", pad_inches=0.2)
        plt.close(report_fig)
    with (out / "summary.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["dip", "min_regret", "min_at_fraction", "final_val_loss", "final_regret"])
        for c, (h, r) in zip(REWARM_RUNS, data):
            j = np.argmin(r[:, 1])
            result = [c.lr_dip, r[j, 1], r[j, 0] / total, h["val_loss"][-1][1], r[-1, 1]]
            writer.writerow(result)
            print(result)
    print(f"Plots saved to {out}")


if __name__ == "__main__":
    main()
