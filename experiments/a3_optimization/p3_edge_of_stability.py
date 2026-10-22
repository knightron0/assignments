"""Problem 3: sharpness and the edge of stability.

    uv run python -m experiments.a3_optimization.p3_edge_of_stability

Both helpers attach sharpness_logging.SharpnessLogger, which logs to W&B under
logging/sharpness/*. Read the results back with experiments.a3_optimization.sharpness.
"""

from pathlib import Path

from metric_logging import AFTER_BACKWARD, MetricLogger
from modal_train import launch_training_jobs
from sharpness_logging import SharpnessLogger
from train import TrainConfig, training_run_name


EXPERIMENT_KEY = "a3-edge-of-stability"
FRACTION = 0.32  # of an epoch: 3,000 updates, 197M tokens at batch size 64


def full_batch(num_sequences, updates, every=25, **changes):
    """Full-batch gradient descent on the first `num_sequences` training sequences.

    Every update uses all of them, so one epoch is one update. lambda_max is
    measured on the same sequences every `every` updates. Keyword arguments
    override any setting, e.g. learning_rate=0.02 or optimizer_name="adamw".
    """
    optimizer_name = changes.get("optimizer_name", "sgd")
    settings = dict(
        num_train_sequences=num_sequences,
        batch_size=num_sequences,
        num_epochs=float(updates),
        optimizer_name=optimizer_name,
        # plain gradient descent for sgd (its default beta1 is momentum); AdamW keeps 0.9
        beta1=0.0 if optimizer_name == "sgd" else TrainConfig.beta1,
        lr_schedule="constant",
        weight_decay=0.0,
        grad_norm=None,
        embedding_init_scale=16.0,
        num_evals=10,
        metric_loggers=TrainConfig.metric_loggers
        + (MetricLogger(AFTER_BACKWARD, SharpnessLogger(every=every)),),
        wandb_tags=(EXPERIMENT_KEY,),
    )
    settings.update(changes)
    return TrainConfig(**settings)


def measurement_steps(batch_size):
    """Every 16 updates to update 208, then every 100 to 1,500, then every 300, at
    batch size 64; other batch sizes measure at the same token counts."""
    at_64 = [*range(0, 209, 16), *range(300, 1501, 100), *range(1800, 3001, 300)]
    total = int(FRACTION * (600_000 // batch_size))
    return sorted({round(u * 64 / batch_size) for u in at_64} & set(range(total)))


def minibatch(batch_size=64, **changes):
    """The course recipe with a constant learning rate after warmup, cut at 0.32
    epochs, measuring the preconditioned sharpness on 64 held-out sequences.

    Warmup is 3.125% of the run: 93 updates at batch size 64, and the same
    6M tokens at every batch size.
    """
    settings = dict(
        num_epochs=FRACTION,
        warmup_percent=0.03125,
        lr_schedule="wsd0.0",
        batch_size=batch_size,
        num_micro_batches=max(1, batch_size // 64),
        metric_loggers=TrainConfig.metric_loggers
        + (MetricLogger(AFTER_BACKWARD, SharpnessLogger(
            steps=measurement_steps(batch_size), probe="heldout", kinds=("pre",))),),
        wandb_tags=(EXPERIMENT_KEY,),
    )
    settings.update(changes)
    return TrainConfig(**settings)


# Part (a): full-batch training.
FULL_BATCH_RUNS = [
    full_batch(32, 1500, learning_rate=0.01),  # (a) i: the clean case
    full_batch(32, 1500, learning_rate=0.01, lr_schedule="step750x0.5"),
    full_batch(32, 1500, learning_rate=0.01, lr_schedule="step750x2"),
]

# Part (b): minibatch training.
MINIBATCH_RUNS = [
    minibatch(),  # (b): the base recipe
]

EXPLORATION_RUNS = [
    minibatch(batch_size=batch_size, learning_rate=lr, beta1=beta1)
    for lr in (0.0015, 0.003, 0.006)
    for batch_size in (32, 128, 256)
    for beta1 in (0.9, 0.95)
]

# Matched to the part (b) constant-rate control: same peak LR, warmup,
# batch size, seed, token budget, and sharpness measurement steps.
SCHEDULE_RUNS = [
    minibatch(lr_schedule=schedule)
    for schedule in ("linear", "cos", "wsd0.2")
]

RUNS = FULL_BATCH_RUNS + MINIBATCH_RUNS + EXPLORATION_RUNS + SCHEDULE_RUNS


RESULTS_DIR = Path(__file__).resolve().parent / "results" / EXPERIMENT_KEY
FULL_BATCH_LABELS = ("constant LR", "LR × 0.5 at update 750", "LR × 2 at update 750")


def load_full_batch(refresh=False):
    """Per-run sharpness rows and per-update training losses, cached as JSON."""
    import json

    cache = RESULTS_DIR / "data" / "full-batch.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    from experiments.a3_optimization.sharpness import history, summarize

    runs = []
    for config in FULL_BATCH_RUNS:
        rows = history(config)
        runs.append(dict(rows=list(rows), losses=rows.losses, summary=summarize(rows)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(runs))
    return runs


def plot_full_batch(refresh=False):
    """Part (a): sharpness against 2 / eta, eta * lambda against 2, and the training loss."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    runs = load_full_batch(refresh)
    out_dir = RESULTS_DIR / "a-full-batch"
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = ("tab:blue", "tab:green", "tab:red")

    def columns(run):
        rows = run["rows"]
        return (np.array([r["step"] for r in rows]), np.array([r["lambda_max"] for r in rows]),
                np.array([r["lr"] for r in rows]), np.array([r["train_loss"] for r in rows]))

    # (a) i: the clean case, sharpness against the update and against the training loss.
    steps, lam, lr, loss = columns(runs[0])
    crossing = runs[0]["summary"]["crossing"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    axes[0].plot(steps, lam, "o-", ms=3, color=colors[0], label="λ_max")
    axes[0].plot(steps, 2 / lr, "k--", lw=1.2, label="2 / η = 200")
    axes[0].axvline(crossing, color="gray", ls=":", lw=1, label=f"crosses 2/η (update {crossing})")
    axes[0].set(xlabel="update", ylabel="sharpness λ_max", title="λ_max against the update")
    axes[1].plot(loss, lam, "o-", ms=3, color=colors[0], label="λ_max")
    axes[1].axhline(2 / lr[0], color="k", ls="--", lw=1.2, label="2 / η = 200")
    axes[1].set(xlabel="training loss (decreasing →)", ylabel="sharpness λ_max",
                title="λ_max against the training loss")
    axes[1].invert_xaxis()
    loss_steps, losses = zip(*runs[0]["losses"])
    axes[2].plot(loss_steps, losses, color=colors[0], lw=0.8)
    axes[2].axvline(crossing, color="gray", ls=":", lw=1, label=f"crosses 2/η (update {crossing})")
    axes[2].set(xlabel="update", ylabel="training loss (every update)", yscale="log",
                title="Training loss")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "i-clean-case.png", dpi=150)

    # (a) ii: halving vs. doubling eta at update 750, with the constant run for reference.
    ref_steps, ref_lam, _, _ = columns(runs[0])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    for col, run in enumerate(runs[1:], start=1):
        steps, lam, lr, _ = columns(run)
        top, bottom = axes[0, col - 1], axes[1, col - 1]
        top.plot(ref_steps, ref_lam, color="gray", alpha=0.5, lw=1, label="constant LR (reference)")
        top.plot(steps, lam, "o-", ms=3, color=colors[col], label="λ_max")
        top.plot(steps, 2 / lr, "k--", lw=1.2, label="2 / η")
        top.axvline(750, color="gray", ls=":", lw=1)
        top.set(title=FULL_BATCH_LABELS[col], ylim=(0, 1.15 * max(lam.max(), (2 / lr).max())))
        loss_steps, losses = zip(*run["losses"])
        bottom.plot(loss_steps, losses, color=colors[col], lw=0.8)
        bottom.axvline(750, color="gray", ls=":", lw=1)
        bottom.set(xlabel="update", yscale="log")
        for ax in (top, bottom):
            ax.grid(alpha=0.3)
        top.legend(fontsize=8)
    axes[0, 0].set_ylabel("sharpness λ_max")
    axes[1, 0].set_ylabel("training loss (every update)")
    fig.tight_layout()
    fig.savefig(out_dir / "ii-lr-change.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run, label, color in zip(runs, FULL_BATCH_LABELS, colors):
        ax.plot([r["step"] for r in run["rows"]], [r["eta_lambda"] for r in run["rows"]], "o-", ms=3,
                color=color, label=label)
    ax.axhline(2, color="k", ls="--", lw=1.2, label="stability threshold η·λ = 2")
    ax.axvline(750, color="gray", ls=":", lw=1)
    ax.set(xlabel="update", ylabel="η · λ_max", title="Normalized sharpness under full-batch GD")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "eta-lambda.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    print("run                     crossing  plateau η·λ  exit   loss-up before/after")
    for run, label in zip(runs, FULL_BATCH_LABELS):
        m = run["summary"]
        print(f"{label:24s} {m['crossing']!s:8s}  {m['plateau']:.2f}         {m['exit']!s:6s} "
              f"{m['loss_up_before']:.2f} / {m['loss_up_after']:.2f}")
    print(f"Wrote plots to {out_dir}")


# Part (c): equal-token factorial sweep of learning rate, batch size, and momentum.
# The frozen-preconditioner stability threshold is 2 * (1 + beta1) / (1 - beta1).
EXPLORE_LRS = (0.0015, 0.003, 0.006)
EXPLORE_BATCH_SIZES = (32, 128, 256)
EXPLORE_BETA1S = (0.9, 0.95)
# Plateau window in training tokens: well after warmup (~6M), before the end (197M).
PLATEAU_TOKENS = (100e6, 180e6)
# Warmup is 3.125% of the 197M-token run. Before it ends, Adam's second-moment estimate is
# still filling in, which inflates the preconditioned sharpness, so entry is read after it.
WARMUP_TOKENS = 0.03125 * FRACTION * 600_000 * 1024


def plot_minibatch(refresh=False):
    """Part (b): eta * lambda_pre on the held-out probe against training tokens, after warmup."""
    import json
    import statistics

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = MINIBATCH_RUNS[0]
    cache = RESULTS_DIR / "data" / "minibatch.json"
    if cache.exists() and not refresh:
        rows = json.loads(cache.read_text())
    else:
        from experiments.a3_optimization.sharpness import history

        rows = [{k: r.get(k) for k in ("step", "eta_lambda_pre", "lambda_max_pre", "lr", "train_loss")}
                for r in history(config)]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows))
    per_update = config.batch_size * config.train_dataset.context_length
    rows = [r for r in rows if r["eta_lambda_pre"] is not None and r["step"] * per_update >= WARMUP_TOKENS]
    tokens = [r["step"] * per_update / 1e6 for r in rows]
    values = [r["eta_lambda_pre"] for r in rows]
    threshold = adam_threshold(config.beta1)
    plateau = statistics.median(v for t, v in zip(tokens, values)
                                if PLATEAU_TOKENS[0] / 1e6 <= t < PLATEAU_TOKENS[1] / 1e6)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(tokens, values, "o-", ms=3, label="η·λ_max (preconditioned, held-out probe)")
    ax.axhline(threshold, color="k", ls="--", lw=1.2, label=f"stability threshold {threshold:.0f}")
    ax.axhline(plateau, color="tab:blue", ls=":", lw=1.2,
               label=f"median over {PLATEAU_TOKENS[0] / 1e6:.0f}–{PLATEAU_TOKENS[1] / 1e6:.0f}M tokens: {plateau:.1f}")
    ax.set(xlabel="training tokens (M), warmup excluded", ylabel="η · λ_max (preconditioned)",
           title=f"Minibatch AdamW (batch {config.batch_size}, lr {config.learning_rate:g}, β₁ = {config.beta1})",
           ylim=(0, 1.3 * max(threshold, max(values))))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    out_dir = RESULTS_DIR / "b-minibatch"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "eta-lambda-pre.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"threshold {threshold:.1f}, plateau {plateau:.1f} ({plateau / threshold:.2f}x), "
          f"max after warmup {max(values):.1f}, first after warmup {values[0]:.1f} at {tokens[0]:.1f}M")
    print("trajectory:", [(round(t), round(v, 1)) for t, v in zip(tokens, values)][::3])
    print(f"Wrote {out_dir / 'eta-lambda-pre.png'}")


def adam_threshold(beta1):
    return 2 * (1 + beta1) / (1 - beta1)


def load_exploration(refresh=False):
    """Sharpness rows and the final validation loss for every exploration run, cached as JSON."""
    import json

    import wandb

    from train import checked_train_config

    cache = RESULTS_DIR / "data" / "explore.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    from experiments.a3_optimization.sharpness import history

    runs = []
    for config in EXPLORATION_RUNS:
        rows = history(config)
        name = training_run_name(checked_train_config(config))
        run = max(wandb.Api().runs(f"{config.wandb_entity}/{config.wandb_project}",
                                   filters={"display_name": name}), key=lambda r: r.created_at)
        runs.append(dict(lr=config.learning_rate, batch_size=config.batch_size, beta1=config.beta1,
                         rows=[{k: r.get(k) for k in ("step", "eta_lambda_pre", "lambda_max_pre", "lr",
                                                      "train_loss")} for r in rows],
                         val_loss=run.summary.get("val_loss")))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(runs))
    return runs


def exploration_metrics(run):
    """Plateau eta * lambda_pre (window median), its ratio to the threshold, and when the run
    first reaches the threshold, in training tokens."""
    import statistics

    per_update = run["batch_size"] * TrainConfig().train_dataset.context_length
    threshold = adam_threshold(run["beta1"])
    m = [(r["step"] * per_update, r["eta_lambda_pre"]) for r in run["rows"]
         if r["eta_lambda_pre"] is not None]
    window = [v for t, v in m if PLATEAU_TOKENS[0] <= t < PLATEAU_TOKENS[1]]
    plateau = statistics.median(window)
    entry = next((t for t, v in m if t >= WARMUP_TOKENS and v >= threshold), None)
    after_warmup = [v for t, v in m if t >= WARMUP_TOKENS]
    at_edge = sum(v >= threshold for v in after_warmup) / len(after_warmup)
    return dict(threshold=threshold, plateau=plateau, ratio=plateau / threshold, entry_tokens=entry,
                at_edge=at_edge)


def plot_exploration(refresh=False):
    """Part (c): eta * lambda_pre over training for every run, and plateau / threshold."""
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    runs = load_exploration(refresh)
    out_dir = RESULTS_DIR / "c-exploration"
    out_dir.mkdir(parents=True, exist_ok=True)
    by_key = {(r["lr"], r["batch_size"], r["beta1"]): r for r in runs}
    lr_colors = dict(zip(EXPLORE_LRS, ("tab:blue", "tab:orange", "tab:red")))

    fig, axes = plt.subplots(len(EXPLORE_BETA1S), len(EXPLORE_BATCH_SIZES),
                             figsize=(5.5 * len(EXPLORE_BATCH_SIZES), 4.2 * len(EXPLORE_BETA1S)),
                             sharex=True, sharey="row")
    for i, beta1 in enumerate(EXPLORE_BETA1S):
        for j, batch_size in enumerate(EXPLORE_BATCH_SIZES):
            ax = axes[i, j]
            for lr in EXPLORE_LRS:
                run = by_key[lr, batch_size, beta1]
                per_update = batch_size * TrainConfig().train_dataset.context_length
                rows = [r for r in run["rows"] if r["eta_lambda_pre"] is not None]
                rows = [r for r in rows if r["step"] * per_update >= WARMUP_TOKENS]
                ax.plot([r["step"] * per_update / 1e6 for r in rows], [r["eta_lambda_pre"] for r in rows],
                        "o-", ms=2.5, lw=1, color=lr_colors[lr], label=f"lr {lr:g}")
            threshold = adam_threshold(beta1)
            ax.set_ylim(0, 2.2 * threshold)
            ax.axhline(threshold, color="k", ls="--", lw=1.2, label=f"threshold {threshold:.0f}")
            ax.axvspan(*(t / 1e6 for t in PLATEAU_TOKENS), color="gray", alpha=0.1)
            ax.set(title=f"batch {batch_size}, β₁ = {beta1} (after warmup)")
            ax.grid(alpha=0.3)
            if i == len(EXPLORE_BETA1S) - 1:
                ax.set_xlabel("training tokens (M)")
            if j == 0:
                ax.set_ylabel("η · λ_max (preconditioned)")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=150)

    # Summary: plateau / threshold and final val loss, one heatmap per beta1.
    metrics = {key: exploration_metrics(run) for key, run in by_key.items()}
    fig, axes = plt.subplots(2, len(EXPLORE_BETA1S), figsize=(7 * len(EXPLORE_BETA1S), 9.5))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    panels = (("ratio", "plateau η·λ_pre / threshold", "coolwarm", (0, 2), "{:.2f}"),
              ("val_loss", "final validation loss", "viridis_r", None, "{:.3f}"))
    for row, (key, title, cmap, limits, fmt) in enumerate(panels):
        values = {b: np.array([[metrics[lr, bs, b][key] if key != "val_loss" else by_key[lr, bs, b]["val_loss"]
                                for bs in EXPLORE_BATCH_SIZES] for lr in EXPLORE_LRS]) for b in EXPLORE_BETA1S}
        vmin, vmax = limits or (min(v.min() for v in values.values()), max(v.max() for v in values.values()))
        for col, beta1 in enumerate(EXPLORE_BETA1S):
            ax = axes[row, col]
            im = ax.imshow(values[beta1], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
            for a in range(len(EXPLORE_LRS)):
                for b in range(len(EXPLORE_BATCH_SIZES)):
                    ax.text(b, a, fmt.format(values[beta1][a, b]), ha="center", va="center", fontsize=9)
            ax.set_xticks(range(len(EXPLORE_BATCH_SIZES)), EXPLORE_BATCH_SIZES)
            ax.set_yticks(range(len(EXPLORE_LRS)), [f"{lr:g}" for lr in EXPLORE_LRS])
            ax.set(xlabel="batch size", ylabel="peak learning rate" if col == 0 else "",
                   title=f"{title}\nβ₁ = {beta1}")
        fig.colorbar(im, ax=axes[row])
    fig.savefig(out_dir / "summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # Reaching the edge: plateau / threshold against batch size, one line per LR and beta1.
    styles = {0.9: "-", 0.95: "--"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for key, ax, ylabel, title in (
            ("ratio", axes[0], "plateau η·λ_pre / threshold", "Does the run reach the edge?"),
            ("at_edge", axes[1], "fraction of measurements ≥ threshold",
             "Does it stay there? (after warmup)")):
        for beta1 in EXPLORE_BETA1S:
            for lr in EXPLORE_LRS:
                ax.plot(EXPLORE_BATCH_SIZES, [metrics[lr, bs, beta1][key] for bs in EXPLORE_BATCH_SIZES],
                        styles[beta1], marker="o", color=lr_colors[lr],
                        label=f"lr {lr:g}, β₁ = {beta1}")
        ax.set(xscale="log", xlabel="batch size", ylabel=ylabel, title=title)
        ax.set_xticks(EXPLORE_BATCH_SIZES, EXPLORE_BATCH_SIZES)
        ax.minorticks_off()
        ax.grid(alpha=0.3)
    axes[0].axhline(1, color="k", lw=1.2)
    axes[0].text(EXPLORE_BATCH_SIZES[0], 1.03, "at the edge", fontsize=8)
    axes[1].set_ylim(-0.03, 1.03)
    # Two legends: color = learning rate, line style = beta1.
    from matplotlib.lines import Line2D

    lr_handles = [Line2D([], [], color=lr_colors[lr], marker="o", label=f"lr {lr:g}") for lr in EXPLORE_LRS]
    beta_handles = [Line2D([], [], color="gray", ls=styles[b], label=f"β₁ = {b}") for b in EXPLORE_BETA1S]
    axes[0].add_artist(axes[0].legend(handles=lr_handles, title="color", fontsize=8, loc="upper left"))
    axes[0].legend(handles=beta_handles, title="line style", fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(out_dir / "edge-vs-batch.png", dpi=150)
    plt.close("all")

    with open(RESULTS_DIR / "data" / "explore-summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lr", "batch_size", "beta1", "threshold", "plateau", "ratio", "entry_tokens",
                         "fraction_at_edge", "val_loss"])
        for (lr, bs, b), m in sorted(metrics.items()):
            writer.writerow([lr, bs, b, m["threshold"], m["plateau"], m["ratio"], m["entry_tokens"],
                             m["at_edge"], by_key[lr, bs, b]["val_loss"]])
    print("lr      bs   β1    plateau  /thr   enters at (M tokens)  at-edge  val")
    for (lr, bs, b), m in sorted(metrics.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
        entry = "never" if m["entry_tokens"] is None else f"{m['entry_tokens'] / 1e6:.0f}"
        print(f"{lr:<7g} {bs:<4d} {b:<5g} {m['plateau']:6.1f}  {m['ratio']:.2f}   {entry:>6s}              "
              f"{m['at_edge']:.2f}     {by_key[lr, bs, b]['val_loss']:.3f}")
    print(f"Wrote plots to {out_dir}")


def main():
    import argparse
    import json
    from pathlib import Path
    from train import checked_train_config, training_run_name

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="all",
                        choices=("all", "baseline", "explore", "schedules", "plot-a", "plot-b", "plot-c"))
    parser.add_argument("--refresh", action="store_true", help="refetch W&B histories for plotting")
    args = parser.parse_args()
    if args.stage == "plot-a":
        plot_full_batch(args.refresh)
        return
    if args.stage == "plot-b":
        plot_minibatch(args.refresh)
        return
    if args.stage == "plot-c":
        plot_exploration(args.refresh)
        return
    configs = {"all": RUNS, "baseline": FULL_BATCH_RUNS + MINIBATCH_RUNS,
               "explore": EXPLORATION_RUNS, "schedules": SCHEDULE_RUNS}[args.stage]
    jobs = launch_training_jobs(configs, max_parallel_runs=2)
    for job, config in zip(jobs, configs):
        job["run_name"] = training_run_name(checked_train_config(config))
    output = Path(__file__).with_name("results") / EXPERIMENT_KEY / f"jobs-{args.stage}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jobs, indent=2) + "\n")
    print(f"Saved job manifest to {output}")


if __name__ == "__main__":
    main()
