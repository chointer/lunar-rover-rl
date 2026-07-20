"""학습 진화 스냅샷(pkl)을 그림으로 (#23).

한 그림에 스냅샷을 다 겹치면 굵은 det 선이 stoch fan을 가려 안 보인다.
→ **step 구간으로 3~4개 그룹으로 나눠** 여러 장 그린다 (그룹당 스냅샷 2~3개만 겹침).
σ는 겹침을 피해 **별도 그림**으로 뺀다.

  python viz/plot_snapshots.py --snapshots experiments/<run>/traj_snapshots.pkl --groups 4

산출: <base>_g1.png … <base>_gK.png (구간별 궤적), <base>_sigma.png (탐험 σ)
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def grid_pos(angle_deg):
    """goal 방향에 맞춰 3x3 격자 위치 (row, col). 예: 좌상단 goal(135°) → (0,0). 중앙은 빔."""
    a = np.radians(angle_deg)
    cx, sy = np.cos(a), np.sin(a)
    col = 1 if abs(cx) < 0.1 else (2 if cx > 0 else 0)
    row = 1 if abs(sy) < 0.1 else (0 if sy > 0 else 2)
    return row, col


def plot_group(snaps, all_snaps, conditions, title, out):
    """한 step 구간의 스냅샷들을 조건별 subplot에 겹쳐 그린다 (goal 방향대로 3x3 배치).

    snaps=이 구간, all_snaps=전체(중앙 σ 곡선 배경용).
    """
    fig, axgrid = plt.subplots(3, 3, figsize=(15, 15))
    colors = plt.cm.tab10.colors
    used = set()

    for angle, dist in conditions:
        r, c = grid_pos(angle)
        used.add((r, c))
        ax = axgrid[r][c]
        for k, s in enumerate(snaps):
            color = colors[k % len(colors)]
            entry = s["trajs"][(angle, dist)]
            if "stoch" in entry:                       # 탐험 fan (얇게)
                for pth in entry["stoch"]:
                    ax.plot(pth[:, 0], pth[:, 1], color=color, alpha=0.35, lw=0.9)
            if "det" in entry:                         # 의도된 경로 (굵게)
                d = entry["det"]
                ax.plot(d[:, 0], d[:, 1], color=color, lw=2.4)
        gx, gy = dist * np.cos(np.radians(angle)), dist * np.sin(np.radians(angle))
        ax.plot(0, 0, "ko", ms=6); ax.plot(gx, gy, "k*", ms=14)     # 출발·goal
        lim = dist + 1.5
        ax.set_title(f"goal {angle}deg, {dist}m"); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    for r in range(3):                                  # 조건이 안 붙은 칸은 끔 (중앙은 σ용으로 사용)
        for cc in range(3):
            if (r, cc) not in used and (r, cc) != (1, 1):
                axgrid[r][cc].axis("off")

    # 중앙 칸: 전체 σ 곡선 + 이 구간 스냅샷을 궤적과 같은 색으로 표시
    # → 색 ↔ step ↔ σ 대응이 그림 안에서 바로 읽혀 별도 범례가 필요 없다
    axc = axgrid[1][1]
    all_steps = [s["step"] for s in all_snaps]
    all_sig   = np.array([s["sigma"] for s in all_snaps])
    axc.plot(all_steps, all_sig[:, 1], "-",  color="0.55", lw=1.6, label="sigma steer")
    axc.plot(all_steps, all_sig[:, 0], "--", color="0.75", lw=1.3, label="sigma throttle")
    for k, s in enumerate(snaps):
        axc.plot(s["step"], s["sigma"][1], "o", color=colors[k % len(colors)], ms=13, zorder=5)
        axc.annotate(f"{s['step']/1000:.0f}k", (s["step"], s["sigma"][1]),
                     textcoords="offset points", xytext=(7, 7), fontsize=10)
    axc.set_xlabel("training step"); axc.set_ylabel("sigma = exp(log_std)")
    axc.set_title("exploration sigma"); axc.legend(fontsize=8); axc.grid(alpha=0.3)

    fig.suptitle(f"{title}   thin=stochastic(explore)  thick=deterministic(intent)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=110); plt.close(fig)
    return out


def plot_sigma(snaps, out):
    steps  = [s["step"] for s in snaps]
    sigmas = np.array([s["sigma"] for s in snaps])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, sigmas[:, 0], "-o", ms=5, label="sigma throttle")
    ax.plot(steps, sigmas[:, 1], "-o", ms=5, label="sigma steer")
    ax.set_xlabel("training step"); ax.set_ylabel("sigma = exp(log_std)")
    ax.set_title("Exploration sigma over training")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshots", type=str, required=True)
    p.add_argument("--groups", type=int, default=4, help="step 구간 분할 개수")
    args = p.parse_args()

    data = pickle.loads(Path(args.snapshots).read_bytes())
    conditions, snaps = data["conditions"], data["snapshots"]
    if not snaps:
        print("스냅샷이 비어 있음"); return

    base = Path(args.snapshots).with_suffix("")
    for i, grp in enumerate(np.array_split(np.arange(len(snaps)), args.groups), start=1):
        if len(grp) == 0:
            continue
        sel = [snaps[k] for k in grp]
        title = f"steps {sel[0]['step']/1000:.0f}k ~ {sel[-1]['step']/1000:.0f}k"
        print("→", plot_group(sel, snaps, conditions, title, f"{base}_g{i}.png"))
    print("→", plot_sigma(snaps, f"{base}_sigma.png"))


if __name__ == "__main__":
    main()
