"""exp04 gSDE vs baseline — 거리×밴드 요약 (밴드별 거리 추세, 5 seed 평균±std).

eval_results.pkl만 읽음(순수 배열, env import 불필요). 밴드=앞/옆/뒤, x=거리.
angle_success(각도 상세)를 밴드로 압축해 gSDE 우위·옆 격차를 한눈에.

  python analysis/exp04_gsde/plot_summary.py
"""
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
COLORS = {"gSDE": "#0072B2", "baseline": "#E69F00", "heuristic": "#009E73"}
BANDS = [("front |angle|<=30", 0, 30), ("side 60-120", 60, 120), ("back |angle|>=150", 150, 180)]


def load(path=None):
    with open(path or (HERE / "eval_results.pkl"), "rb") as f:
        return pickle.load(f)


def band_mean_std(d, label, lo, hi, di):
    """그룹 label, 밴드[lo,hi], 거리 di → 모델seed 간 (평균, std) %."""
    idx = [i for i, a in enumerate(d["angles"]) if lo <= abs(a) <= hi]
    per_model = [100.0 * d["results"][n][:, di, :][:, idx].mean() for n in d["group_names"][label]]
    return np.mean(per_model), np.std(per_model)


def plot(d, out=None):
    dists = np.array(d["dists"])
    labels = list(d["group_names"])
    fig, axes = plt.subplots(1, len(BANDS), figsize=(4.8 * len(BANDS), 4.4), sharey=True)
    for ax, (bname, lo, hi) in zip(axes, BANDS):
        for label in labels:
            c = COLORS.get(label, "#555")
            ms = np.array([band_mean_std(d, label, lo, hi, di) for di in range(len(dists))])
            mean, std = ms[:, 0], ms[:, 1]
            ax.fill_between(dists, mean - std, mean + std, color=c, alpha=0.18, lw=0)
            ax.plot(dists, mean, color=c, lw=2.4, marker="o", ms=6, label=label)
        ax.set_title(bname, fontsize=11)
        ax.set_xlabel("goal distance (m)")
        ax.set_xticks(dists)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_ylim(-3, 103)
    axes[0].set_ylabel("reach rate (%)")
    axes[1].legend(loc="upper left", framealpha=0.9, fontsize=10)
    fig.suptitle("Reach rate by band and distance — heuristic vs gSDE vs baseline "
                 "(learned: 5-seed mean ± std)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = out or (HERE / "band_summary.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("→", plot(load()))
