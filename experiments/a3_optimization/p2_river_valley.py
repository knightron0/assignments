"""Problem 2: schedules, parameter averaging, rewarming, and horizon-free schedules.

    uv run python -m experiments.a3_optimization.p2_river_valley launch --part a
    uv run python -m experiments.a3_optimization.p2_river_valley plot

`plot` fetches the finished runs from W&B once (cached in results/; pass --refresh
to refetch) and writes the part (a) and (b) figures and tables.
"""

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from train import TrainConfig, training_run_name


EXPERIMENT_KEY = "a3-river-valley"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / EXPERIMENT_KEY
TOKENS_PER_STEP = TrainConfig.batch_size * 1024
# Each run logs val_ema{decay}_loss next to val_loss at every evaluation.
EMA_DECAYS = (0.99, 0.999)

SCHEDULES = (
    "wsd0.0",  # constant after warmup
    "linear",
    "cos",
    "wsd0.2",  # plateau, then decay over the last 20% of training
)
# Part (a): tune the peak LR per schedule so the comparison is not an artifact of 3e-3.
PEAK_LRS = (1.5e-3, 3e-3, 6e-3)

# Parts (a) and (b): one run per schedule and peak LR, with evaluation-only EMAs.
SCHEDULE_RUNS = [
    TrainConfig(lr_schedule=lr_schedule, learning_rate=lr, ema_decays=EMA_DECAYS,
                wandb_tags=(EXPERIMENT_KEY,))
    for lr_schedule in SCHEDULES
    for lr in PEAK_LRS
]

# Part (b): a wider spread of EMA timescales (~100 to ~5k updates) at the default peak LR.
EMA_SWEEP_DECAYS = (0.99, 0.995, 0.998, 0.999, 0.9995, 0.9998)
EMA_SWEEP_RUNS = [
    TrainConfig(lr_schedule=lr_schedule, ema_decays=EMA_SWEEP_DECAYS, wandb_tags=(EXPERIMENT_KEY,))
    for lr_schedule in SCHEDULES
]

# Part (c): the LR falls to 0.1x between 40% and 45% of training and is back at
# full LR by 50%, before the same 20% cooldown as the wsd0.2 control above.
# The other dips vary its depth (scale) and duration around that default.
LR_DIPS = (
    (0.4, 0.45, 0.5, 0.1),  # default
    (0.4, 0.45, 0.5, 0.3),  # shallower
    (0.4, 0.45, 0.5, 0.0),  # all the way to zero
    (0.4, 0.42, 0.44, 0.1),  # shorter
    (0.4, 0.5, 0.6, 0.1),  # longer
)
REWARM_RUNS = [
    TrainConfig(lr_schedule="wsd0.2", lr_dip=dip, wandb_tags=(EXPERIMENT_KEY,)) for dip in LR_DIPS
]

# Part (d): horizon-free schedules. warmup_steps fixes the warmup in updates,
# so it does not depend on the budget; sgdrT restarts every T updates.
HORIZON_FREE_RUNS = [
    TrainConfig(lr_schedule="invsqrt", warmup_steps=94, wandb_tags=(EXPERIMENT_KEY,)),
    TrainConfig(lr_schedule="sgdr2000", warmup_steps=94, min_lr_ratio=0.1,
                wandb_tags=(EXPERIMENT_KEY,)),
]
# Keep the full dataset and its ordering; shorten the schedule horizon rather
# than shrinking the dataset (which would change the shuffle).
# Full-budget linear/LR comparisons are already available from part (a).
BUDGET_FRACTIONS = (0.25, 0.4, 0.5, 0.75)
BUDGET_RUNS = [
    TrainConfig(num_epochs=fraction, learning_rate=lr,
                wandb_tags=(EXPERIMENT_KEY,))
    for fraction in BUDGET_FRACTIONS
    for lr in PEAK_LRS
]
FULL_BUDGET_UPDATES = TrainConfig.num_train_sequences // TrainConfig.batch_size
# Stop with no added cooldown. The full horizon and all schedule parameters
# remain unchanged, and the trainer evaluates at the exact stopping update.
HORIZON_FREE_STOP_RUNS = [
    replace(config, stop_at_step=int(fraction * FULL_BUDGET_UPDATES))
    for config in HORIZON_FREE_RUNS
    for fraction in BUDGET_FRACTIONS
]

RUNS_BY_PART = {"a": SCHEDULE_RUNS, "b": EMA_SWEEP_RUNS, "c": REWARM_RUNS,
               "d": HORIZON_FREE_RUNS + HORIZON_FREE_STOP_RUNS + BUDGET_RUNS}


SCHEDULE_LABELS = {"wsd0.0": "constant", "linear": "linear", "cos": "cosine", "wsd0.2": "wsd0.2"}
SCHEDULE_COLORS = {"wsd0.0": "tab:red", "linear": "tab:blue", "cos": "tab:green", "wsd0.2": "tab:purple"}
# Fractions of training at which the part (b) tables read off regret.
READ_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def ema_key(decay):
    return f"val_ema{decay:g}_loss"


def load_histories(refresh=False):
    """{run name: {key: (steps, values)}} for every part (a)/(b) run, cached as JSON."""
    cache = RESULTS_DIR / "data" / "history.json"
    if cache.exists() and not refresh:
        return {name: {k: tuple(np.array(v) for v in sv) for k, sv in h.items()}
                for name, h in json.loads(cache.read_text()).items()}
    import wandb

    from utils import WANDB_ENTITY, WANDB_PROJECT

    wanted = {training_run_name(c) for c in SCHEDULE_RUNS + EMA_SWEEP_RUNS}
    keys = ("train_loss", "val_loss", "learning_rate", *map(ema_key, EMA_SWEEP_DECAYS))
    latest = {}
    for run in wandb.Api().runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}",
                                filters={"tags": {"$in": [EXPERIMENT_KEY]}}):
        if run.name in wanted and run.state == "finished" and (
                run.name not in latest or run.created_at > latest[run.name].created_at):
            latest[run.name] = run
    histories = {}
    for name, run in latest.items():
        # One pass over the history; part (a) runs log only some of the EMA keys.
        rows = list(run.scan_history())
        histories[name] = {}
        for key in keys:
            logged = [(row["optimizer_step"] + 1, row[key]) for row in rows
                      if row.get(key) is not None]
            if logged:
                histories[name][key] = tuple(np.array(column, dtype=float) for column in zip(*logged))
    missing = wanted - histories.keys()
    if missing:
        raise SystemExit(f"Not finished on W&B yet: {sorted(missing)}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({name: {k: [s.tolist(), v.tolist()] for k, (s, v) in h.items()}
                                 for name, h in histories.items()}))
    return histories


def output_path(folder, name):
    (RESULTS_DIR / folder).mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / folder / name


def smooth(values, window=100):
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid"), window // 2


def regret(history, ref, key="val_loss"):
    """Steps and L(run) - L(ref) at the eval steps both runs share."""
    steps, values = history[key]
    ref_steps, ref_values = ref["val_loss"]
    common, i, j = np.intersect1d(steps, ref_steps, return_indices=True)
    return common, values[i] - ref_values[j]


def value_at(steps, values, fraction):
    return values[np.searchsorted(steps, fraction * steps[-1])]


def plot_schedules(histories, plt):
    """Part (a): LR, training loss, and validation loss for each schedule at its best peak LR."""
    by_config = {(c.lr_schedule, c.learning_rate): histories[training_run_name(c)]
                 for c in SCHEDULE_RUNS}
    final = {key: h["val_loss"][1][-1] for key, h in by_config.items()}
    best_lr = {s: min(PEAK_LRS, key=lambda lr: final[s, lr]) for s in SCHEDULES}

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for schedule in SCHEDULES:
        h = by_config[schedule, best_lr[schedule]]
        style = dict(color=SCHEDULE_COLORS[schedule],
                     label=f"{SCHEDULE_LABELS[schedule]} (lr {best_lr[schedule]:g})")
        steps, lr = h["learning_rate"]
        axes[0].plot(steps * TOKENS_PER_STEP, lr, **style)
        steps, loss = h["train_loss"]
        smoothed, offset = smooth(loss)
        axes[1].plot(steps[offset:offset + len(smoothed)] * TOKENS_PER_STEP, smoothed, **style)
        steps, loss = h["val_loss"]
        axes[2].plot(steps * TOKENS_PER_STEP, loss, **style)
    axes[0].set(title="Learning rate", ylabel="learning rate")
    axes[1].set(title="Training loss (100-step moving average)", ylabel="train loss", ylim=(2.85, 3.6))
    axes[2].set(title="Validation loss", ylabel="val loss", ylim=(2.9, 3.6))
    for ax in axes:
        ax.set_xlabel("tokens")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.savefig(output_path("a-schedules", "schedules.png"), dpi=150, bbox_inches="tight")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    labels = {"wsd0.0": "Constant", "linear": "Linear decay",
              "cos": "Cosine decay", "wsd0.2": "Final-20% cooldown"}
    for schedule in SCHEDULES:
        lr = best_lr[schedule]
        h = by_config[schedule, lr]
        label = f"{labels[schedule]} · peak LR {lr:g}"
        for ax, key in zip(axes, ("learning_rate", "val_loss")):
            steps, values = h[key]
            ax.plot(steps * TOKENS_PER_STEP / 1e6, values,
                    color=SCHEDULE_COLORS[schedule], label=label, linewidth=2)
    axes[0].set(title="Learning rate", ylabel="Learning rate")
    axes[1].set(title="Validation loss", ylabel="Validation loss (lower is better)",
                ylim=(2.9, 3.6))
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for ax in axes:
        ax.set_xlabel("Training tokens (millions)")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=9)
    fig.suptitle("P2(a): best tested peak LR for each schedule", fontsize=14)
    fig.savefig(output_path("a-schedules", "best-lr-and-validation.png"), dpi=180)
    fig.savefig(output_path("a-schedules", "best-lr-and-validation.svg"))

    # Peak-LR sweep: validation curves per peak LR, then final loss against peak LR.
    fig, axes = plt.subplots(1, len(PEAK_LRS) + 1, figsize=(5.5 * (len(PEAK_LRS) + 1), 4.5))
    for ax, lr in zip(axes, PEAK_LRS):
        for schedule in SCHEDULES:
            steps, loss = by_config[schedule, lr]["val_loss"]
            ax.plot(steps * TOKENS_PER_STEP, loss, color=SCHEDULE_COLORS[schedule],
                    label=SCHEDULE_LABELS[schedule])
        ax.set(title=f"Validation loss, peak lr {lr:g}", xlabel="tokens", ylim=(2.9, 3.6))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("val loss")
    axes[0].legend()
    for schedule in SCHEDULES:
        axes[-1].plot(PEAK_LRS, [final[schedule, lr] for lr in PEAK_LRS], "o-",
                      color=SCHEDULE_COLORS[schedule], label=SCHEDULE_LABELS[schedule])
    axes[-1].set(title="Final validation loss", xlabel="peak learning rate", xscale="log")
    axes[-1].set_xticks(PEAK_LRS, [f"{lr:g}" for lr in PEAK_LRS])
    axes[-1].grid(alpha=0.3)
    fig.savefig(output_path("a-schedules", "lr-sweep.png"), dpi=150, bbox_inches="tight")

    with open(output_path("data", "lr-sweep.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["schedule", "peak_lr", "final_val_loss", *map(ema_key, EMA_DECAYS)])
        for (schedule, lr), h in by_config.items():
            writer.writerow([schedule, lr, final[schedule, lr],
                             *(h[ema_key(d)][1][-1] for d in EMA_DECAYS)])
    reference = min(final, key=final.get)
    print(f"Best peak LR per schedule: {best_lr}")
    print(f"Reference run (lowest final val loss): {reference} at {final[reference]:.4f}")
    return by_config[reference], reference


def plot_ema(histories, ref, ref_label, plt):
    """Part (b): raw vs EMA validation loss and regret, from the EMA-sweep runs."""
    sweep = {c.lr_schedule: histories[training_run_name(c)] for c in EMA_SWEEP_RUNS}
    cmap = plt.get_cmap("viridis")
    ema_colors = {d: cmap(i / (len(EMA_SWEEP_DECAYS) - 1)) for i, d in enumerate(EMA_SWEEP_DECAYS)}

    def ema_label(decay):
        return f"EMA {decay:g} (~{round(1 / (1 - decay)):,} steps)"

    fig, axes = plt.subplots(2, len(SCHEDULES), figsize=(5.5 * len(SCHEDULES), 9), sharey="row")
    for col, schedule in enumerate(SCHEDULES):
        h = sweep[schedule]
        top, bottom = axes[0, col], axes[1, col]
        steps, loss = h["val_loss"]
        top.plot(steps * TOKENS_PER_STEP, loss, "k", lw=2, label="raw")
        common, r = regret(h, ref)
        bottom.plot(common * TOKENS_PER_STEP, r, "k", lw=2, label="raw")
        for decay in EMA_SWEEP_DECAYS:
            steps, loss = h[ema_key(decay)]
            top.plot(steps * TOKENS_PER_STEP, loss, color=ema_colors[decay], label=ema_label(decay))
            common, r = regret(h, ref, ema_key(decay))
            bottom.plot(common * TOKENS_PER_STEP, r, color=ema_colors[decay])
        top.set(title=f"{SCHEDULE_LABELS[schedule]}: validation loss", ylim=(2.88, 3.4))
        bottom.set(title=f"{SCHEDULE_LABELS[schedule]}: regret vs {ref_label}",
                   xlabel="tokens", ylim=(-0.12, 0.25))
        bottom.axhline(0, color="gray", lw=1)
        for ax in (top, bottom):
            ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("val loss")
    axes[1, 0].set_ylabel("regret R(t) = L(θ_t) − L(θ_ref_t)")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(output_path("b-ema", "ema.png"), dpi=150, bbox_inches="tight")

    # All schedules on one axis: raw regret (solid) vs regret of the best EMA at each read-off.
    fig, ax = plt.subplots(figsize=(8, 5))
    for schedule in SCHEDULES:
        h = sweep[schedule]
        common, r = regret(h, ref)
        ax.plot(common * TOKENS_PER_STEP, r, color=SCHEDULE_COLORS[schedule],
                label=f"{SCHEDULE_LABELS[schedule]} raw")
        best = np.min([regret(h, ref, ema_key(d))[1] for d in EMA_SWEEP_DECAYS], axis=0)
        ax.plot(common * TOKENS_PER_STEP, best, "--", color=SCHEDULE_COLORS[schedule],
                label=f"{SCHEDULE_LABELS[schedule]} best EMA")
    ax.axhline(0, color="gray", lw=1)
    ax.set(title=f"Regret vs {ref_label}: raw (solid) and best EMA timescale at each step (dashed)",
           xlabel="tokens", ylabel="regret", ylim=(-0.12, 0.25))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(output_path("b-ema", "regret.png"), dpi=150, bbox_inches="tight")

    # EMA benefit against timescale, read off at several points in training.
    timescales = [1 / (1 - d) for d in EMA_SWEEP_DECAYS]
    fig, axes = plt.subplots(1, len(READ_FRACTIONS), figsize=(5 * len(READ_FRACTIONS), 4.2),
                             sharey=True)
    rows = []
    for ax, fraction in zip(axes, READ_FRACTIONS):
        for schedule in SCHEDULES:
            h = sweep[schedule]
            raw = value_at(*h["val_loss"], fraction)
            ema = [value_at(*h[ema_key(d)], fraction) for d in EMA_SWEEP_DECAYS]
            ax.plot(timescales, np.array(ema) - raw, "o-", color=SCHEDULE_COLORS[schedule],
                    label=SCHEDULE_LABELS[schedule])
            raw_r = value_at(*regret(h, ref), fraction)
            ema_r = [value_at(*regret(h, ref, ema_key(d)), fraction) for d in EMA_SWEEP_DECAYS]
            best = int(np.argmin(ema_r))
            rows.append([schedule, fraction, raw_r, EMA_SWEEP_DECAYS[best], ema_r[best],
                         raw - ema[best], *ema_r])
        ax.axhline(0, color="gray", lw=1)
        ax.set(title=f"{fraction:.0%} of training", xlabel="EMA timescale 1/(1−β) [steps]",
               xscale="log", ylim=(-0.25, 0.1))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("EMA val loss − raw val loss")
    axes[0].legend()
    fig.savefig(output_path("b-ema", "timescale.png"), dpi=150, bbox_inches="tight")

    with open(output_path("data", "regret.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["schedule", "fraction", "raw_regret", "best_ema_decay", "best_ema_regret",
                         "ema_improvement", *(f"regret_ema{d:g}" for d in EMA_SWEEP_DECAYS)])
        writer.writerows(rows)
    print("\nschedule  frac  raw regret  best EMA  EMA regret")
    for schedule, fraction, raw_r, decay, ema_r, *_ in rows:
        print(f"{SCHEDULE_LABELS[schedule]:9s} {fraction:4.0%}  {raw_r:+.4f}     "
              f"{decay:<8g}  {ema_r:+.4f}")


def plot(refresh=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    histories = load_histories(refresh)
    ref, (schedule, lr) = plot_schedules(histories, plt)
    plot_ema(histories, ref, f"{SCHEDULE_LABELS[schedule]} lr {lr:g}", plt)
    plt.close("all")
    print(f"Wrote plots to {RESULTS_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("launch", "plot"), nargs="?", default="launch")
    parser.add_argument("--part", choices=(*RUNS_BY_PART, "all"), default="all")
    parser.add_argument("--refresh", action="store_true", help="refetch the W&B histories")
    args = parser.parse_args()
    if args.stage == "plot":
        plot(args.refresh)
        return
    from modal_train import launch_training_jobs

    parts = RUNS_BY_PART if args.part == "all" else (args.part,)
    configs = [run for part in parts for run in RUNS_BY_PART[part]]
    jobs = launch_training_jobs(configs, max_parallel_runs=2)
    for job, config in zip(jobs, configs):
        job["run_name"] = training_run_name(config)
    output = output_path("data", f"jobs-{args.part}.json")
    output.write_text(json.dumps(jobs, indent=2) + "\n")
    print(f"Saved job manifest to {output}")


if __name__ == "__main__":
    main()
