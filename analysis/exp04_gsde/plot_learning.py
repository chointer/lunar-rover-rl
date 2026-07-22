"""exp04 gSDE vs baseline — 학습커브 (TensorBoard 스칼라, 5 seed 평균±std).

각 run의 tb_1/ 이벤트에서 스칼라를 뽑아 공통 step grid에 보간 후 그룹 평균±표준편차.
baseline=exp02 seed1~5(3M), gSDE=exp04 seed1~5(3M) — 동일 예산(3M) 비교.

  python analysis/exp04_gsde/plot_learning.py
"""
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
COLORS = {"gSDE": "#0072B2", "baseline": "#E69F00"}

# 그룹별 run 디렉토리 (tb_1 포함). exp04-00은 2M·seed0라 제외.
GROUPS = {
    "baseline": sorted(glob(str(ROOT / "experiments/exp02-seed-variance/*/tb_1"))),
    "gSDE":     sorted(glob(str(ROOT / "experiments/exp04-explore/exp04-0[1-5]-*/tb_1"))),
}
# (y축 라벨, tb 태그, y스케일 배수, 지표 설명) — x축은 항상 training step. goal_rate는 0~1이라 ×100로 %
PANELS = [("goal reach rate (%)", "outcome/goal_rate", 100.0,
           "episodes that reached the goal, rolling mean (last 100 eps)"),
          ("episode return", "rollout/ep_rew_mean", 1.0,
           "cumulative shaped reward per episode, rolling mean (last 100 eps)")]


def load_scalar(tb_dir, tag):
    ea = EventAccumulator(tb_dir); ea.Reload()
    s = ea.Scalars(tag)
    return np.array([x.step for x in s], float), np.array([x.value for x in s], float)


def group_band(tb_dirs, tag, scale, n=200):
    """여러 seed → 공통 step grid 보간 후 (grid, 평균, 표준편차)."""
    curves = [load_scalar(d, tag) for d in tb_dirs]
    xmax = min(x[-1] for x, _ in curves)          # 가장 짧은 run 기준(외삽 방지)
    grid = np.linspace(0, xmax, n)
    ys = np.stack([np.interp(grid, x, y) * scale for x, y in curves])
    return grid, ys.mean(axis=0), ys.std(axis=0)


def plot(out=None):
    fig, axes = plt.subplots(1, len(PANELS), figsize=(6.4 * len(PANELS), 4.6))
    for ax, (ylabel, tag, scale, desc) in zip(axes, PANELS):
        for label, dirs in GROUPS.items():
            c = COLORS[label]
            grid, mean, std = group_band(dirs, tag, scale)
            gx = grid / 1e6
            ax.fill_between(gx, mean - std, mean + std, color=c, alpha=0.18, lw=0)
            ax.plot(gx, mean, color=c, lw=2.4, label=label)
        ax.set_title(desc, fontsize=9, color="0.35", pad=6)   # 지표 설명(suptitle보다 작게·뮤트)
        ax.set_xlabel("training steps (M)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, lw=0.6)
    axes[0].legend(loc="upper left", framealpha=0.9, fontsize=10)
    fig.suptitle("Training curves — gSDE vs baseline (5 seeds each, mean ± std)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = out or (HERE / "learning_curve.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for label, dirs in GROUPS.items():
        print(f"{label}: {len(dirs)} runs")
    print("→", plot())
