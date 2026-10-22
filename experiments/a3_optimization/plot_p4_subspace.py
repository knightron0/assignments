"""Plot the completed P4(c) baseline continuations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).with_name("results") / "a3-inside-the-hessian"
    out = root / "c-plots"
    out.mkdir(exist_ok=True)
    runs = {}
    for arm in ("full", "top", "removed"):
        rows = [json.loads(line) for line in (root / f"c-{arm}.jsonl").read_text().splitlines()]
        runs[arm] = {"start": rows[0], "steps": [r for r in rows if r["kind"] == "step"],
                     "refresh": [r for r in rows if r["kind"] == "refresh"]}
    colors = {"full": "#0072B2", "top": "#D55E00", "removed": "#009E73"}
    labels = {"full": "Full gradient", "top": "Top-10 only", "removed": "Top-10 removed"}

    def finish(fig, axes, name, title):
        for ax in axes:
            ax.set_xlabel("Updates since SGDM checkpoint at 100M tokens")
            ax.grid(alpha=.2)
        fig.suptitle(title, fontsize=14)
        for ext in ("png", "svg"):
            fig.savefig(out / f"{name}.{ext}", dpi=180, bbox_inches="tight", pad_inches=.2)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    for arm, run in runs.items():
        start = run["start"]
        steps = run["steps"]
        x = np.array([r["step"]-start["step"] for r in steps])
        loss = np.array([r["loss"] for r in steps])
        smooth = np.convolve(loss, np.ones(10)/10, mode="valid")
        axes[0].plot(x[9:], smooth, color=colors[arm], label=labels[arm])
        vals = [(0, start["val_loss"])] + [(r["step"]-start["step"],r["val_loss"])
                                             for r in steps if "val_loss" in r]
        vx, vy = np.array(vals).T
        axes[1].plot(vx, vy, ".-", color=colors[arm], label=labels[arm])
        if arm != "top":
            axes[2].plot(vx, vy, ".-", color=colors[arm],
                         label=f"{labels[arm]} (final {vy[-1]:.4f})")
    axes[0].set(title="Training loss (10-update moving average)", ylabel="Training loss")
    axes[1].set(title="Validation loss: all three arms", ylabel="Validation loss (64 held-out rows)")
    axes[2].set(title="Validation loss: full vs. removed (zoom)", ylabel="Validation loss (64 held-out rows)")
    for ax in axes: ax.legend(fontsize=8)
    finish(fig, axes, "losses", "P4(c): zero momentum, 300 updates, top-10 basis refreshed every 25 updates")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    for arm, run in runs.items():
        start = run["start"]["step"]
        steps, refresh = run["steps"], run["refresh"]
        style = {"color": colors[arm], "label": labels[arm]}
        axes[0].plot([r["step"]-start for r in refresh],
                     [r["probe_gradient_fraction"] for r in refresh], ".-", **style)
        axes[1].plot([r["step"]-start for r in steps],
                     [r["top_fraction"] for r in steps], linewidth=1, alpha=.8, **style)
        axes[2].plot([r["step"]-start for r in refresh],
                     [r["lambda_max"]/r["heavy_ball_threshold"] for r in refresh], ".-", **style)
    axes[0].set(title="Probe gradient in freshly computed top-10", ylabel=r"$\|U^T g_{probe}\|^2/\|g_{probe}\|^2$", ylim=(0,1))
    axes[1].set(title="Minibatch gradient in current top-10 basis", ylabel=r"$\|U^T g_{batch}\|^2/\|g_{batch}\|^2$", ylim=(0,1))
    axes[2].axhline(1, color="black", linestyle="--", label="Frozen-quadratic stability boundary")
    axes[2].set(title="Sharpness relative to heavy-ball threshold",
                ylabel=r"$\lambda_{max}/[2(1+\beta)/\eta_t]$", yscale="log")
    for ax in axes: ax.legend(fontsize=8)
    finish(fig, axes, "diagnostics", "P4(c): gradient fractions are squared-norm shares before projection")
    print(f"Saved plots to {out}")


if __name__ == "__main__":
    main()
