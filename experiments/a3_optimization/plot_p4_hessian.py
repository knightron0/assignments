"""Plot the saved P4(a) measurements without launching GPUs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from experiments.a3_optimization.hessian import density, partition, layer_of, block_size


def main():
    root = Path(__file__).with_name("results") / "a3-inside-the-hessian"
    rows = sorted((json.loads(p.read_text()) for p in root.glob("a-adamw-*.json")), key=lambda r: r["step"])
    out = root / "a-plots"
    out.mkdir(exist_ok=True)
    tokens = np.array([r["tokens"] / 1e6 for r in rows])

    def save(fig, name):
        for ext in ("png", "svg"):
            fig.savefig(out / f"{name}.{ext}", dpi=180, bbox_inches="tight", pad_inches=.2)
        plt.close(fig)

    def token_axis(ax):
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(0, tokens[-1] * 1.05)
        ax.set_xlabel("Training tokens (millions; linear below 1M, log above)")
        ax.grid(alpha=.2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    for fam in rows[0]["families"]:
        axes[0].plot(tokens, [r["families"][fam]["lambda_max"] for r in rows], ".-", label=fam)
    axes[0].plot(tokens, [r["lambda_max"] for r in rows], "k--", label="full Hessian", linewidth=2)
    axes[0].set(yscale="log", title="Top eigenvalue by parameter family", ylabel="Largest eigenvalue")
    axes[0].legend(fontsize=8, ncol=2)
    means = [np.mean(r["hutchinson"]["squared_per_param"]) for r in rows]
    errors = [np.std(r["hutchinson"]["squared_per_param"], ddof=1) / np.sqrt(8) for r in rows]
    axes[1].errorbar(tokens, means, yerr=errors, fmt=".-", capsize=3)
    axes[1].set(yscale="log", title="Mean squared curvature (8 probes; +/- 1 SE)",
                ylabel=r"$\mathrm{tr}(H^2)/d$")
    for ax in axes: token_axis(ax)
    save(fig, "sharpness")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, step in zip(axes.flat, (0, 153, 1526, 9375)):
        r = next(r for r in rows if r["step"] == step)
        runs = r["density_runs"]
        # Smooth in signed-log coordinates to show the bulk and distant outliers
        # together; this is explicitly a density in transformed coordinates.
        scale = .001
        transform = lambda x: np.sign(x) * np.log10(1 + np.abs(x) / scale)
        inv = lambda x: np.sign(x) * scale * (10**np.abs(x) - 1)
        transformed = [{"ritz": transform(np.array(v["ritz"])), "weights": v["weights"]} for v in runs]
        lo, hi = transform(np.array([r["lambda_min"], r["lambda_max"]]))
        grid = np.linspace(lo - .3, hi + .3, 2500)
        y = density(transformed, grid, width=.06)
        ax.plot(inv(grid), y)
        ax.axvline(0, color="0.5", linewidth=.7)
        ax.axvline(r["lambda_min"], color="tab:red", linestyle=":", label=r"$\lambda_{\min}$")
        ax.axvline(r["lambda_max"], color="tab:green", linestyle=":", label=r"$\lambda_{\max}$")
        ax.set(xscale="symlog", yscale="log", title=f"{r['tokens']/1e6:g}M tokens (step {step})",
               xlabel="Eigenvalue (symmetric log scale)", ylabel="Density per signed-log coordinate")
        ax.set_xscale("symlog", linthresh=scale)
        ax.set_ylim(bottom=max(y.max()*1e-9, 1e-12), top=y.max()*3)
        ax.grid(alpha=.2)
        ax.text(.04, .95, f"min = {r['lambda_min']:.3g}; max = {r['lambda_max']:.3g}",
                transform=ax.transAxes, va="top", fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("Eigenvalue spectrum: four 96-step Lanczos probes; smoothing width 0.06 in signed-log coordinates")
    save(fig, "spectrum")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    random_count = 0
    for ax, part in zip(axes, ("family", "layer")):
        energies = [np.array(r["energy"][part]["per_probe"]).mean(0) for r in rows]
        within = np.array([np.trace(e) / e.sum() for e in energies])
        ax.plot(tokens, within, ".-", label="Within blocks")
        ax.plot(tokens, 1-within, ".-", label="Between blocks")
        random_x, random_y = [], []
        for r in rows:
            path = root / f"a-random-adamw-{r['step']}.json"
            if path.exists():
                control = json.loads(path.read_text())["random_energy"][part]
                E = np.concatenate([np.array(p["per_probe"]) for p in control["partitions"]]).mean(0)
                random_x.append(r["tokens"] / 1e6)
                random_y.append(np.trace(E) / E.sum())
        if random_x:
            ax.plot(random_x, random_y, "k.--", label="Within random blocks (matched sizes)")
        random_count = len(random_x)
        ax.set(title=f"Partition by {part}", ylabel=r"Fraction of $\|H\|_F^2$", ylim=(0, 1))
        token_axis(ax)
        ax.legend()
    fig.suptitle(f"Block structure over training (random controls: {random_count}/{len(rows)} checkpoints)")
    save(fig, "block-structure")

    layers = [f"L{i}" for i in range(8)]
    values = []
    for r in rows:
        e = np.array(r["energy"]["layer"]["per_probe"]).mean(0)
        keys = r["energy"]["layer"]["blocks"]
        blocks = partition(r["names"], r["sizes"], layer_of)
        values.append([e[keys.index(k), keys.index(k)] / block_size(blocks[k]) for k in layers])
    a = np.array(values).T
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    im = ax.imshow(a, aspect="auto", origin="lower", norm=LogNorm(vmin=a.min(), vmax=a.max()))
    ax.set_yticks(range(8), layers)
    ax.set_xticks(range(len(rows)), [f"{r['tokens']/1e6:.0f}" if r['tokens'] >= 1e6
                                     else f"{r['tokens']/1e6:.2g}" for r in rows], rotation=40)
    ax.set(xlabel="Training tokens (millions; checkpoint columns equally spaced)", ylabel="Transformer block",
           title="Within-layer squared curvature per parameter")
    fig.colorbar(im, ax=ax, label=r"$\|H_{bb}\|_F^2/d_b$")
    save(fig, "depth")
    print(f"Saved four figures to {out}")


if __name__ == "__main__":
    main()
