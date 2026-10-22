"""Problem 1: forked runs, weight averaging, and ensembles.

Run the stages in order, waiting for each to finish:

    uv run python -m experiments.a3_optimization.p1_mode_connectivity prefix
    uv run python -m experiments.a3_optimization.p1_mode_connectivity branches
    uv run python -m experiments.a3_optimization.p1_mode_connectivity measure
    uv run python -m experiments.a3_optimization.p1_mode_connectivity plot

Every stage takes --variant (default: baseline; "all" for every variant). After
measuring several variants, `compare` plots their barrier grids side by side.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from train import TrainConfig, training_run_name


EXPERIMENT_KEY = "a3-mode-connectivity"
PREFIX_STEPS = (0, 50, 100, 250, 500, 1000)  # x
BRANCH_STEPS = (25, 50, 100, 250, 500)  # y
BRANCH_ORDER_SEEDS = (1, 2)
RESULTS_DIR = Path(__file__).resolve().parent / "results" / EXPERIMENT_KEY
# Baseline figures go in one folder per handout part; every other variant's figures
# go under d-training-knobs/<variant>/. Measured cells and CSVs live in data/.
FIGURE_DIRS = {"ensemble": "a-ensemble", "ensemble-gain": "a-ensemble",
               "barrier": "b-weight-averaging", "interpolation": "c-interpolation"}
KNOBS_DIR = "d-training-knobs"

# Constant LR after warmup. The schedule (and so the warmup) is set by the full
# 600M-token run length; stop_at_step only ends training early.
BASE = TrainConfig(lr_schedule="wsd0.0", wandb_tags=(EXPERIMENT_KEY,))
# The baseline's warmup: 1% of its 9,375 updates. Variants that change the run
# length pin it, so every grid shares the same LR over its first updates.
BASE_WARMUP_STEPS = 93
# Linear decay whose horizon is the longest branch's end (x + y = 1500 updates).
# num_epochs, not num_train_sequences, shortens it: branches still permute
# batches drawn from the full 600,000-row subset, as in the baseline.
DECAY_EPOCHS = (max(PREFIX_STEPS) + max(BRANCH_STEPS)) / 9375

# Part (d): one knob changed from BASE at a time.
VARIANTS = {
    "baseline": BASE,
    "lr1e-3": replace(BASE, learning_rate=1e-3),
    "lr1e-2": replace(BASE, learning_rate=1e-2),
    "bs32": replace(BASE, batch_size=32, warmup_steps=BASE_WARMUP_STEPS),
    "bs128": replace(BASE, batch_size=128, warmup_steps=BASE_WARMUP_STEPS),
    "linear1500": replace(
        BASE, lr_schedule="linear", num_epochs=DECAY_EPOCHS, warmup_steps=BASE_WARMUP_STEPS
    ),
    "invsqrt": replace(BASE, lr_schedule="invsqrt"),
    # Longer warmups keep the LR low over the early updates where basins split.
    "warmup300": replace(BASE, warmup_steps=300),
    "warmup1000": replace(BASE, warmup_steps=1000),
    # Branches at 0.3x the LR, forked from the baseline prefix (see VARIANT_PREFIXES).
    "branchlr0.3x": replace(BASE, learning_rate=0.3 * BASE.learning_rate),
    # Weight decay around the baseline's 0.1: off, and 3x stronger.
    "wd0": replace(BASE, weight_decay=0.0),
    "wd0.3": replace(BASE, weight_decay=0.3),
}
# Variants whose branches fork from another variant's prefix instead of their own.
VARIANT_PREFIXES = {"branchlr0.3x": "baseline"}
# Knob -> variants in display order, for the comparison figures.
KNOBS = {
    "learning rate": ("lr1e-3", "baseline", "lr1e-2"),
    "batch size": ("bs32", "baseline", "bs128"),
    "schedule": ("baseline", "linear1500", "invsqrt"),
    "weight decay": ("wd0", "baseline", "wd0.3"),
}
# Follow-ups testing whether early, post-fork LR is what splits basins.
FOLLOWUP_KNOBS = {
    "warmup": ("baseline", "warmup300", "warmup1000"),
    "branch LR": ("baseline", "branchlr0.3x"),
}


def prefix_for(base):
    """The shared prefix run, keeping model and optimizer state at every x > 0."""
    return replace(
        base,
        keep_checkpoint_steps=tuple(x for x in PREFIX_STEPS if x > 0),
        stop_at_step=max(PREFIX_STEPS),
    )


PREFIX = prefix_for(BASE)


def branch(x, order_seed, base=BASE, prefix_base=None):
    """One branch: x shared updates, then max(BRANCH_STEPS) updates in its own data order.

    x = 0 starts from the same initialization (same model_seed) without a fork. The
    prefix is prefix_for(prefix_base), defaulting to base's own.
    """
    fork = {}
    if x > 0:
        prefix = prefix_for(base if prefix_base is None else prefix_base)
        fork = dict(fork_from_run=training_run_name(prefix), fork_from_step=x)
    return replace(
        base,
        batch_order_seed=order_seed,
        stop_at_step=x + max(BRANCH_STEPS),
        keep_checkpoint_steps=tuple(x + y for y in BRANCH_STEPS),
        **fork,
    )


def branches_for(base, prefix_base=None):
    return {x: [branch(x, seed, base=base, prefix_base=prefix_base) for seed in BRANCH_ORDER_SEEDS]
            for x in PREFIX_STEPS}


def variant_prefix(variant):
    """The prefix config a variant's branches fork from."""
    return VARIANTS[VARIANT_PREFIXES.get(variant, variant)]


def variant_branches(variant):
    return branches_for(VARIANTS[variant], prefix_base=variant_prefix(variant))


BRANCHES = branches_for(BASE)


def pairs(variant="baseline"):
    """((run_a, step), (run_b, step)) for every (x, y) cell."""
    cells = []
    for x, (a, b) in variant_branches(variant).items():
        for y in BRANCH_STEPS:
            cells.append(((training_run_name(a), x + y), (training_run_name(b), x + y)))
    return cells


def results_path(variant="baseline"):
    return RESULTS_DIR / "data" / f"{variant}.json"


def figure_path(variant, name, kind=None):
    """kind picks the baseline folder; it defaults to name."""
    folder = FIGURE_DIRS[kind or name] if variant == "baseline" else f"{KNOBS_DIR}/{variant}"
    (RESULTS_DIR / folder).mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / folder / f"{name}.png"


def check_prefix_checkpoints(prefix=PREFIX):
    """Fail before launching if the prefix's kept checkpoints are not on the volume yet."""
    from pathlib import PurePosixPath

    from modal.exception import NotFoundError

    from modal_train import _model_volume_path
    from modal_utils import user_volume

    try:
        entries = user_volume.listdir(_model_volume_path(training_run_name(prefix)))
    except NotFoundError:
        entries = []
    names = {PurePosixPath(entry.path).name for entry in entries}
    missing = [step for step in prefix.keep_checkpoint_steps if f"step_{step}.pt" not in names]
    if missing:
        raise SystemExit(
            f"Prefix {training_run_name(prefix)!r} has no checkpoints for steps {missing}. "
            "Run the prefix stage and wait for it to finish."
        )


BARRIER_THRESHOLD = 0.01
# (x, y) cells for the full interpolation curves: one per regime of the barrier heatmap.
INTERPOLATION_CELLS = ((0, 500), (100, 500), (250, 100), (1000, 25), (1000, 500))


def grid_from_results(rows, variant="baseline"):
    """{name: (len(PREFIX_STEPS), len(BRANCH_STEPS)) array} from measure rows, in pairs() order."""
    import numpy as np

    by_cell = {(row["run_a"], row["step_a"]): row for row in rows}
    shape = (len(PREFIX_STEPS), len(BRANCH_STEPS))
    grid = {k: np.full(shape, np.nan) for k in ("loss_a", "loss_b", "ensemble", "average", "barrier")}
    for i, (x, (a, _)) in enumerate(variant_branches(variant).items()):
        for j, y in enumerate(BRANCH_STEPS):
            row = by_cell[(training_run_name(a), x + y)]
            losses = dict(zip(row["alphas"], row["interpolation"]))
            grid["loss_a"][i, j], grid["loss_b"][i, j] = losses[0.0], losses[1.0]
            grid["ensemble"][i, j] = row["ensemble"]
            grid["average"][i, j] = losses[0.5]
    # Barrier: midpoint loss above the mean endpoint loss.
    grid["barrier"] = grid["average"] - 0.5 * (grid["loss_a"] + grid["loss_b"])
    return grid


def load_grid(variant="baseline"):
    return grid_from_results(json.loads(results_path(variant).read_text()), variant)


def _heatmap(ax, values, title, *, vmin=None, vmax=None, cmap="viridis", fmt="{:.3f}", boxed=None,
             ylabel=True):
    """Annotated (x, y) heatmap; cells where boxed is True get a red outline."""
    import matplotlib.pyplot as plt

    im = ax.imshow(values, cmap=cmap, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            r, g, b, _ = im.cmap(im.norm(values[i, j]))
            light = 0.299 * r + 0.587 * g + 0.114 * b > 0.5
            ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center", fontsize=7,
                    color="black" if light else "white")
            if boxed is not None and boxed[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="red", lw=2))
    ax.set_xticks(range(len(BRANCH_STEPS)), BRANCH_STEPS)
    ax.set_yticks(range(len(PREFIX_STEPS)), PREFIX_STEPS)
    ax.set_xlabel("y (branch steps)")
    if ylabel:
        ax.set_ylabel("x (shared prefix steps)")
    ax.set_title(title)
    return im


def plot_results(variant="baseline"):
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path = results_path(variant)
    grid = load_grid(variant)

    # Part (a): both endpoint losses and the ensemble loss on a shared scale.
    loss_range = dict(vmin=min(grid[k].min() for k in ("loss_a", "loss_b", "ensemble")),
                      vmax=max(grid[k].max() for k in ("loss_a", "loss_b", "ensemble")))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    im = _heatmap(axes[0], grid["loss_a"], "Endpoint L(θ₁)", **loss_range)
    _heatmap(axes[1], grid["loss_b"], "Endpoint L(θ₂)", ylabel=False, **loss_range)
    _heatmap(axes[2], grid["ensemble"], "Ensemble loss L(½p₁ + ½p₂)", ylabel=False, **loss_range)
    fig.colorbar(im, ax=axes, label="val loss")
    fig.savefig(figure_path(variant, "ensemble"), dpi=150, bbox_inches="tight")

    # Ensemble gain over the mean endpoint loss, too small to see on the shared loss scale.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    gain = 0.5 * (grid["loss_a"] + grid["loss_b"]) - grid["ensemble"]
    im = _heatmap(ax, gain, "Ensemble gain ½[L(θ₁) + L(θ₂)] − L(ensemble)", cmap="magma")
    fig.colorbar(im, ax=ax, label="loss reduction")
    fig.savefig(figure_path(variant, "ensemble-gain"), dpi=150, bbox_inches="tight")

    # Weight averaging: both endpoints and L(θ_avg) on a shared scale, then the barrier.
    loss_range = dict(vmin=min(grid[k].min() for k in ("loss_a", "loss_b", "average")),
                      vmax=max(grid[k].max() for k in ("loss_a", "loss_b", "average")))
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.5))
    im = _heatmap(axes[0], grid["loss_a"], "Endpoint L(θ₁)", **loss_range)
    _heatmap(axes[1], grid["loss_b"], "Endpoint L(θ₂)", ylabel=False, **loss_range)
    _heatmap(axes[2], grid["average"], "Weight average L(½θ₁ + ½θ₂)", ylabel=False, **loss_range)
    fig.colorbar(im, ax=axes[:3], label="val loss")
    barrier = grid["barrier"]
    im = _heatmap(axes[3], barrier, f"Barrier B(x, y); red box = B ≤ {BARRIER_THRESHOLD}",
                  cmap="coolwarm", vmin=-abs(barrier).max(), vmax=abs(barrier).max(),
                  boxed=barrier <= BARRIER_THRESHOLD, ylabel=False)
    fig.colorbar(im, ax=axes[3], label="L(θ_avg) − endpoint loss")
    fig.savefig(figure_path(variant, "barrier"), dpi=150, bbox_inches="tight")

    # Weight averaging vs. ensembling: the same pair of models combined in weight space or in
    # prediction space. Negative = the single averaged model beats the two-model ensemble.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    diff = grid["average"] - grid["ensemble"]
    im = _heatmap(axes[0], diff, "L(θ_avg) − L(ensemble); blue = averaging wins", cmap="coolwarm",
                  vmin=-abs(diff).max(), vmax=abs(diff).max(), boxed=barrier <= BARRIER_THRESHOLD)
    fig.colorbar(im, ax=axes[0], label=f"loss difference; red box = B ≤ {BARRIER_THRESHOLD}")
    mean_endpoint = 0.5 * (grid["loss_a"] + grid["loss_b"])
    ensemble_gain, average_gain = mean_endpoint - grid["ensemble"], mean_endpoint - grid["average"]
    ax = axes[1]
    for i, x in enumerate(PREFIX_STEPS):
        ax.scatter(ensemble_gain[i], average_gain[i], s=[20 + y / 5 for y in BRANCH_STEPS],
                   color=plt.get_cmap("viridis")(i / (len(PREFIX_STEPS) - 1)), label=f"x = {x}")
    lim = (min(average_gain.min(), 0) - 0.02, ensemble_gain.max() + 0.02)
    ax.plot([0, lim[1]], [0, lim[1]], "k--", lw=1, label="averaging = ensemble")
    ax.axhline(0, color="gray", lw=1)
    ax.set(xlabel="ensemble gain  ½[L(θ₁)+L(θ₂)] − L(ensemble)",
           ylabel="averaging gain  ½[L(θ₁)+L(θ₂)] − L(θ_avg)  (= −B)",
           title="Gain over the endpoints (marker size ∝ y)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(figure_path(variant, "averaging-vs-ensemble", kind="barrier"), dpi=150,
                bbox_inches="tight")

    # Full interpolation curves for representative cells, against the endpoint chord and ensemble.
    rows = {(row["run_a"], row["step_a"]): row for row in json.loads(path.read_text())}
    branches = variant_branches(variant)

    def plot_interpolation(ax, x, y):
        row = rows[(training_run_name(branches[x][0]), x + y)]
        alphas, losses = row["alphas"], row["interpolation"]
        i, j = PREFIX_STEPS.index(x), BRANCH_STEPS.index(y)
        ax.plot(alphas, losses, "o-", label="L(θ(α))")
        ax.plot([0, 1], [losses[0], losses[-1]], "k--", lw=1, label="endpoint chord")
        ax.axhline(row["ensemble"], color="tab:green", ls=":", label="ensemble")
        ax.set_title(f"x = {x}, y = {y}\nB = {grid['barrier'][i, j]:+.3f}")
        ax.set_xlabel("α")
        ax.set_xticks(alphas)
        ax.set_ylabel("val loss")

    fig, axes = plt.subplots(1, len(INTERPOLATION_CELLS), figsize=(4 * len(INTERPOLATION_CELLS), 3.8))
    for ax, (x, y) in zip(axes, INTERPOLATION_CELLS):
        plot_interpolation(ax, x, y)
    for ax in axes[1:]:
        ax.set_ylabel("")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path(variant, "interpolation"), dpi=150)
    # One figure per cell, each on its own loss scale.
    for x, y in INTERPOLATION_CELLS:
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_interpolation(ax, x, y)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_path(variant, f"interpolation-x{x}-y{y}", kind="interpolation"), dpi=150)

    # Where the chosen cells sit on the barrier grid.
    fig, ax = plt.subplots(figsize=(7, 5))
    im = _heatmap(ax, barrier, "Barrier B(x, y); black = interpolation cells", cmap="coolwarm",
                  vmin=-abs(barrier).max(), vmax=abs(barrier).max(), boxed=barrier <= BARRIER_THRESHOLD)
    for x, y in INTERPOLATION_CELLS:
        i, j = PREFIX_STEPS.index(x), BRANCH_STEPS.index(y)
        ax.add_patch(plt.Rectangle((j - 0.38, i - 0.38), 0.76, 0.76, fill=False, ec="black", lw=2.5))
    fig.colorbar(im, ax=ax, label=f"barrier; red box = B ≤ {BARRIER_THRESHOLD}")
    fig.savefig(figure_path(variant, "interpolation-cells", kind="interpolation"), dpi=150,
                bbox_inches="tight")

    # Every cell's interpolation curve, laid out like the heatmap (x up, y across), on a shared
    # scale after subtracting the endpoint chord so the shape is comparable across cells.
    fig, axes = plt.subplots(len(PREFIX_STEPS), len(BRANCH_STEPS), figsize=(3 * len(BRANCH_STEPS),
                             2.2 * len(PREFIX_STEPS)), sharex=True, sharey=True)
    for i, x in enumerate(PREFIX_STEPS):
        for j, y in enumerate(BRANCH_STEPS):
            ax = axes[len(PREFIX_STEPS) - 1 - i, j]
            row = rows[(training_run_name(branches[x][0]), x + y)]
            alphas, losses = np.array(row["alphas"]), np.array(row["interpolation"])
            excess = losses - ((1 - alphas) * losses[0] + alphas * losses[-1])
            ax.plot(alphas, excess, "o-", ms=3, color="tab:red" if excess.max() > BARRIER_THRESHOLD
                    else "tab:blue")
            ax.axhline(0, color="k", ls="--", lw=0.8)
            ax.set_title(f"x={x}, y={y}", fontsize=8)
            ax.set_yscale("symlog", linthresh=0.05)
            ax.set_xticks(alphas[::2])
    fig.supxlabel("α")
    fig.supylabel("L(θ(α)) − endpoint chord")
    fig.tight_layout()
    fig.savefig(figure_path(variant, "interpolation-all", kind="interpolation"), dpi=130)

    with open(path.with_suffix(".csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "loss_theta1", "loss_theta2", "loss_ensemble", "loss_avg", "barrier"])
        for i, x in enumerate(PREFIX_STEPS):
            for j, y in enumerate(BRANCH_STEPS):
                writer.writerow([x, y] + [f"{grid[k][i, j]:.4f}" for k in
                                          ("loss_a", "loss_b", "ensemble", "average", "barrier")])
    plt.close("all")
    return grid


def plot_comparison(knobs=KNOBS, name="compare-barrier"):
    """Barrier grids for every knob setting: one row per knob.

    The color scale is shared across every measured variant, so separate comparison
    figures stay comparable.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grids = {v: load_grid(v) for v in VARIANTS if results_path(v).is_file()}
    limit = max(abs(g["barrier"]).max() for g in grids.values())
    ncols = max(len(variants) for variants in knobs.values())
    fig, axes = plt.subplots(len(knobs), ncols, figsize=(5.5 * ncols, 4.3 * len(knobs)), squeeze=False)
    for row, (knob, variants) in enumerate(knobs.items()):
        for col, variant in enumerate(variants):
            ax = axes[row, col]
            if variant not in grids:
                ax.set_axis_off()
                ax.set_title(f"{knob}: {variant} (not measured)")
                continue
            barrier = grids[variant]["barrier"]
            im = _heatmap(ax, barrier, f"{knob}: {variant}", cmap="coolwarm", vmin=-limit, vmax=limit,
                          boxed=barrier <= BARRIER_THRESHOLD, ylabel=col == 0)
    for row, variants in enumerate(knobs.values()):
        for col in range(len(variants), ncols):
            axes[row, col].set_axis_off()
    fig.subplots_adjust(hspace=0.35)
    fig.colorbar(im, ax=axes, label=f"barrier B(x, y); red box = B ≤ {BARRIER_THRESHOLD}")
    out = RESULTS_DIR / KNOBS_DIR / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prefix", "branches", "measure", "plot", "compare"))
    parser.add_argument("--variant", default="baseline", choices=(*VARIANTS, "all"))
    args = parser.parse_args()
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    if args.stage == "prefix":
        from modal_train import launch_training_jobs

        launch_training_jobs([prefix_for(VARIANTS[v]) for v in variants if v not in VARIANT_PREFIXES])
    elif args.stage == "branches":
        from modal_train import launch_training_jobs

        for v in variants:
            check_prefix_checkpoints(prefix_for(variant_prefix(v)))
        launch_training_jobs(
            [run for v in variants for runs in variant_branches(v).values() for run in runs]
        )
    elif args.stage == "plot":
        for v in variants:
            plot_results(v)
        print(f"Wrote plots to {RESULTS_DIR}")
    elif args.stage == "compare":
        print(f"Wrote {plot_comparison()}")
        print(f"Wrote {plot_comparison(FOLLOWUP_KNOBS, 'compare-barrier-followups')}")
    else:
        from experiments.a3_optimization.connectivity import measure_pairs_on_modal

        for v in variants:
            rows = measure_pairs_on_modal(pairs(v))
            path = results_path(v)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rows, indent=2) + "\n")
            print(f"Wrote {len(rows)} cells to {path}")


if __name__ == "__main__":
    main()
