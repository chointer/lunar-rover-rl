"""exp04 gSDE vs baseline — 각도별 도달률 선그래프 (거리별 facet).

eval_results.pkl(policy_eval.py 산출)만 읽는다. 순수 배열이라 env·SB3 import 불필요.
  results[name]: bool [평가seed, 거리, 각도]  /  group_names, dists, angles, seeds

  python analysis/exp04_gsde/plot_eval.py         # 같은 폴더의 eval_results.pkl 사용
"""
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent

# Okabe-Ito CVD-안전 색 (그룹=정체성). 전 스크립트 공통 라벨: gSDE=파랑, baseline=주황.
COLORS = {"gSDE": "#0072B2", "baseline": "#E69F00", "heuristic": "#009E73"}
SIDE = (60, 120)   # 옆 밴드(|각도|) — 핵심 구간 음영


def load(path=None):
    path = path or (HERE / "eval_results.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def group_curves(d, label, di):
    """그룹 label, 거리 인덱스 di → (각도별 개별모델 곡선 [n_model, n_angle], 평균 [n_angle]) %."""
    names = d["group_names"][label]
    per_model = np.stack([100.0 * d["results"][n][:, di, :].mean(axis=0) for n in names])  # 평가seed 평균
    return per_model, per_model.mean(axis=0)


def plot(d, out=None):
    dists, angles = d["dists"], np.array(d["angles"])
    labels = list(d["group_names"])
    n = len(dists)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.4 * nrow),
                             squeeze=False, sharex=True, sharey=True)

    for di, dist in enumerate(dists):
        ax = axes[di // ncol][di % ncol]
        # 옆 밴드 음영 (양쪽 부호)
        for lo, hi in [SIDE, (-SIDE[1], -SIDE[0])]:
            ax.axvspan(lo, hi, color="0.6", alpha=0.10, lw=0)
        for label in labels:
            c = COLORS.get(label, "#555555")
            per_model, mean = group_curves(d, label, di)
            for curve in per_model:                              # 개별 5 seed 옅게
                ax.plot(angles, curve, color=c, lw=0.8, alpha=0.25)
            ax.plot(angles, mean, color=c, lw=2.4, label=label,  # 그룹 평균 굵게
                    marker="o", ms=3.5)
        ax.set_title(f"goal {dist:.0f} m", fontsize=11)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_xticks(range(-180, 181, 90))
        ax.set_ylim(-3, 103)
        if di % ncol == 0:
            ax.set_ylabel("reach rate (%)")
        if di // ncol == nrow - 1:
            ax.set_xlabel("goal angle (deg, + = left)")

    # 범례(2계열) — 한 번만
    axes[0][0].legend(loc="upper right", framealpha=0.9, fontsize=9)
    fig.suptitle("Reach rate by goal angle — heuristic vs gSDE vs baseline\n"
                 "learned (gSDE/baseline): 5-seed mean, faint = seeds · heuristic = hand-designed expert · shaded = side band",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = out or (HERE / "angle_success.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


if __name__ == "__main__":
    d = load()
    print("정책:", list(d["results"]))
    print("거리:", d["dists"], "| 각도수:", len(d["angles"]), "| 평가seed:", d["seeds"])
    print("→", plot(d))
