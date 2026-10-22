"""Final P4 report plots for head decay and k/checkpoint sweeps."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).with_name("results") / "a3-inside-the-hessian"
ARMS = ("full", "top", "removed")
COLORS = dict(zip(ARMS, ("#0072B2", "#D55E00", "#009E73")))
LABELS = {"full": "Full", "top": "Top only", "removed": "Top removed"}


def save(fig, folder, name):
    target = ROOT / folder
    target.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(target / f"{name}.{ext}", dpi=180, bbox_inches="tight", pad_inches=.2)
    plt.close(fig)


def main():
    factors = (1, 5, 20)
    rows = [json.loads((ROOT / f"b-headwdx{f}-final.json").read_text()) for f in factors]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    axes[0].plot(x, [r["lambda_max"] for r in rows], "o-", label="Full Hessian")
    axes[0].axhline(2*rows[0]["lambda_max"],color="0.4",linestyle="--",label="2x control sharpness")
    axes[0].set(title="Final sharpness", ylabel=r"$\lambda_{max}$")
    axes[0].legend(fontsize=8)
    axes[1].axhspan(2.925,2.931,color="green",alpha=.15,label="Required loss interval")
    axes[1].plot(x,[r["full_val_loss"] for r in rows],"o-",color="#D55E00")
    for i,r in enumerate(rows):
        axes[1].annotate(f"{r['full_val_loss']:.5f}",(i,r["full_val_loss"]),
                         xytext=(0,8),textcoords="offset points",ha="center",fontsize=9)
    axes[1].set(title="Final full validation loss",ylabel="Validation loss",ylim=(2.922,2.955))
    axes[1].legend(fontsize=8,loc="upper left")
    for name,label in (("lm_head.weight","Output-head RMS"),("model.norm.weight","Final-norm gain RMS")):
        values=[r["tensors"][name]["rms"] for r in rows]
        axes[2].plot(x,np.array(values)/values[0],"o-",label=label)
    axes[2].set(title="Parameter-scale compensation",ylabel="RMS / control RMS",yscale="log")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x,[f"{f}x" for f in factors])
        ax.set_xlabel("Head weight-decay multiplier")
        ax.grid(alpha=.2)
    fig.suptitle("P4(b): standard initialization; only output-head decay changes during training")
    save(fig,"b-plots","head-decay")

    def sweep(groups, name, title):
        fig, axes = plt.subplots(2,3,figsize=(15,7.5),layout="constrained")
        for col,(label,names) in enumerate(groups):
            residual=0
            for arm in ARMS:
                records=[json.loads(l) for l in (ROOT/f"{names[arm]}.jsonl").read_text().splitlines()]
                start=records[0]
                vals=[(0,start["val_loss"])]+[(r["step"]-start["step"],r["val_loss"])
                    for r in records if r["kind"]=="step" and "val_loss" in r]
                vx,vy=np.array(vals).T
                ref=[r for r in records if r["kind"]=="refresh"]
                residual=max(residual,max(max(r["residuals"]) for r in ref))
                axes[0,col].plot(vx,vy,".-",color=COLORS[arm],label=f"{LABELS[arm]}: {vy[-1]:.4f}")
                axes[1,col].plot([r["step"]-start["step"] for r in ref],
                    [r["probe_gradient_fraction"] for r in ref],".-",color=COLORS[arm],label=LABELS[arm])
            axes[0,col].set(title=label,ylabel="Validation loss (64 rows)",yscale="log")
            axes[0,col].legend(fontsize=8)
            axes[0,col].text(.03,.95,f"Largest eigenpair residual: {residual:.2g}",
                transform=axes[0,col].transAxes,va="top",fontsize=8)
            axes[1,col].set(ylabel="Probe gradient squared-norm share",ylim=(0,1))
            for ax in axes[:,col]:
                ax.set_xlabel("Continuation updates")
                ax.grid(alpha=.2)
        fig.suptitle(title+"\nLoss axes use separate log ranges; gradient fractions use the same 0-1 range",fontsize=13)
        save(fig,"c-plots",name)

    sweep([(f"k = {k}",{a:(f"c-k{k}-{a}-100M" if k!=10 else f"c-{a}") for a in ARMS})
           for k in (1,10,50)],"k-sweep","P4(c): subspace size at the same 100M-token checkpoint")
    sweep([(label,{a:(f"c-stage{step}-{a}-k10" if step!=1526 else f"c-{a}") for a in ARMS})
           for step,label in ((153,"Start at 10M tokens"),(1526,"Start at 100M tokens"),(7630,"Start at 500M tokens"))],
          "checkpoint-sweep","P4(c): training stage with k=10 (each checkpoint retains its scheduled LR)")
    print("Saved head-decay, k-sweep and checkpoint-sweep figures")


if __name__ == "__main__":
    main()
