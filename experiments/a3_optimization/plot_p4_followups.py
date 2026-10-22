"""Report figures for completed P4(b) and P4(c) follow-ups."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).with_name("results") / "a3-inside-the-hessian"


def save(fig, folder, name):
    out = ROOT / folder
    out.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=180, bbox_inches="tight", pad_inches=.2)
    plt.close(fig)


def main():
    scales = (.5, 1, 2, 4)
    rows = [json.loads((ROOT / f"b-core-rescale-{c:g}.json").read_text()) for c in scales]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    axes[0].plot(scales, [r["lambda_max"] for r in rows], "ko-", label="Full Hessian")
    for name in ("lm_head.weight", "model.norm.weight"):
        axes[0].plot(scales, [r["tensors"][name]["lambda_max"] for r in rows], ".-", label=name)
    axes[0].set(yscale="log", title="Sharpness changes with parameter scale", ylabel="Largest eigenvalue")
    axes[0].legend(fontsize=8)
    loss = [r["probe_loss"] for r in rows]
    axes[1].plot(scales, loss, "o-")
    axes[1].set(title="Probe loss stays unchanged", ylabel="Loss on the same 64 sequences",
                ylim=(loss[0]-.002, loss[0]+.002))
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[2].plot(scales, [r["scale_max_logit_error"] for r in rows], "o-")
    axes[2].set(title="Prediction-invariance check (1 sequence)", ylabel="Maximum absolute logit difference", ylim=(-1e-6,1e-6))
    for ax in axes:
        ax.set(xlabel="Final RMSNorm gain multiplier c", xscale="log")
        ax.set_xticks(scales, [str(c) for c in scales])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(alpha=.2)
    fig.suptitle("P4(b), manual rescaling: final norm gain x c; output head / c")
    save(fig, "b-plots", "manual-rescaling")

    files = ["normscale0.5", "normscale2", "normscale4", "wd0.3"]
    labels = ["Init scale 0.5", "Init scale 2", "Init scale 4", "Global WD 0.3"]
    rows = [json.loads((ROOT / f"b-core-final-a3-p4-{s}.json").read_text()) for s in files]
    ref = json.loads((ROOT / "a-adamw-9375.json").read_text())["lambda_max"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    x = np.arange(4)
    axes[0].bar(x, [r["lambda_max"]/ref for r in rows], color=["#999999"]*3+["#0072B2"])
    axes[0].axhline(2, color="black", linestyle="--", label="2x supplied reference")
    axes[0].set(title="Final sharpness", ylabel="Sharpness / supplied reference")
    axes[0].legend(fontsize=8)
    axes[1].axhspan(2.925,2.931,color="green",alpha=.15,label="Required loss interval")
    axes[1].scatter(x, [r["full_val_loss"] for r in rows], color="black")
    axes[1].set(title="Final full validation loss", ylabel="Validation loss")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x,labels,rotation=15)
        ax.grid(axis="y",alpha=.2)
    fig.suptitle("Exploratory training variants: initialization rescaling does not settle the training-recipe requirement")
    save(fig, "b-plots", "training-exploration")

    settings = [("Original LR; refresh 25", {a:f"c-{a}" for a in ("full","top","removed")}),
                ("Quarter LR; refresh 25", {a:f"c-core-{a}-lr0.25" for a in ("full","top","removed")}),
                ("Original LR; refresh 10", {a:f"c-core-{a}-refresh10" for a in ("top","removed")})]
    colors = ["#0072B2", "#D55E00", "#009E73"]
    fig, axes = plt.subplots(2,3,figsize=(15,8),layout="constrained")
    summary=[]
    for col,(title,names) in enumerate(settings):
        for arm,name in names.items():
            records=[json.loads(l) for l in (ROOT/f"{name}.jsonl").read_text().splitlines()]
            start=records[0]
            vals=[(0,start["val_loss"])]+[(r["step"]-start["step"],r["val_loss"]) for r in records if r["kind"]=="step" and "val_loss" in r]
            refresh=[r for r in records if r["kind"]=="refresh"]
            color=colors[("full","top","removed").index(arm)]
            label={"full":"Full", "top":"Top-10 only", "removed":"Top-10 removed"}[arm]
            vx,vy=np.array(vals).T
            axes[0,col].plot(vx,vy,".-",color=color,label=f"{label}: {vy[-1]:.4f}")
            axes[1,col].plot([r["step"]-start["step"] for r in refresh],
                             [r["lambda_max"]/r["heavy_ball_threshold"] for r in refresh],".-",color=color,label=label)
            summary.append({"run":name,"final_val_loss":vy[-1],"seconds":records[-1]["seconds"]})
        axes[0,col].set(title=title,yscale="log",ylabel="Validation loss (64 rows)")
        axes[1,col].axhline(1,color="black",linestyle="--")
        axes[1,col].set(yscale="log",ylabel="Sharpness / heavy-ball threshold")
        for ax in axes[:,col]:
            ax.set_xlabel("Continuation updates")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle("P4(c): lower LR stabilizes top-only training; more frequent refreshes do not rescue it")
    save(fig,"c-plots","lr-refresh-comparison")
    (ROOT/"c-plots"/"followup-summary.json").write_text(json.dumps(summary,indent=2))
    print("Saved P4(b) plots and P4(c) follow-up comparison")


if __name__ == "__main__":
    main()
