"""BC 정책 평가 — head-goal 각도별 성공률(3·4·5m 독립) + 궤적 시각화.

- 성공률: max_steer 0.5, 15° 간격, N/각도. 기존 정책·휴리스틱 히스토그램과 같은 조건 → 직접 비교.
- 궤적: 각도 밴드(정면/왼/오/후방)별로 나눠 그려 클러터 방지. 성공=초록, 실패=빨강.
- 거리 3·4·5m 독립 평가 (BC는 2~4m로 학습 → 4·5m는 transfer 테스트).

goal은 env.reset(options=...)로 heading 기준 각도·거리에 배치 (env가 소유, private 안 건드림).

  python bc/eval_bc.py --bc bc/bc_policy.pt
"""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # 헤드리스 렌더
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from envs.config import EnvConfig
from envs.lunar_rover_env import LunarRoverEnv
from bc.bc_model import make_bc_action_fn

ANGLES = list(range(-180, 180, 15))
BANDS = [                                    # (제목, 각도필터)
    ("front |theta|<=30", lambda a: abs(a) <= 30),
    ("left 45~135",       lambda a: 45 <= a <= 135),
    ("right -135~-45",    lambda a: -135 <= a <= -45),
    ("back |theta|>=150", lambda a: abs(a) >= 150),
]


def run_ep(env, action_fn, angle_deg, dist, seed):
    """(도달여부, 궤적 xy(N,2), goal xy) 반환."""
    obs, _ = env.reset(seed=seed, options={"goal_angle": angle_deg, "goal_dist": dist})
    goal = env._goal.copy()
    path = [env.data.qpos[:2].copy()]
    for _ in range(env.cfg.max_steps + 5):
        obs, _, term, trunc, info = env.step(action_fn(obs))
        path.append(env.data.qpos[:2].copy())
        if term or trunc:
            return bool(info.get("reached")), np.asarray(path), goal
    return False, np.asarray(path), goal


def bar(r, w=20):
    n = int(round(r*w)); return "█"*n + "·"*(w-n)


def survey_dist(env, action_fn, dist, n):
    """한 거리에서 각도별 성공률 + 궤적 수집."""
    res, trajs = {}, {}                      # trajs[a] = [(reached, path, goal), ...]
    for a in ANGLES:
        runs = [run_ep(env, action_fn, a, dist, 1000+i) for i in range(n)]
        res[a] = sum(r[0] for r in runs) / n
        trajs[a] = runs
    return res, trajs


def print_hist(res, dist, n):
    print(f"\nBC 각도별 성공률 (goal {dist}m, N={n})")
    for a in sorted(ANGLES):
        mark = "  ← 정면" if a == 0 else ("  ← 정후방" if abs(a) == 180 else "")
        print(f"  {a:+4d}°  {bar(res[a])} {res[a]*100:3.0f}%{mark}")
    front = np.mean([res[a] for a in ANGLES if abs(a) <= 30])*100
    diag  = np.mean([res[a] for a in ANGLES if 60 <= abs(a) <= 120])*100
    back  = np.mean([res[a] for a in ANGLES if abs(a) >= 150])*100
    print(f"  → 앞(≤30°) {front:.0f}%  |  옆(60~120°) {diag:.0f}%  |  뒤(≥150°) {back:.0f}%")


def plot_trajs(trajs, dist, out_dir, k=4):
    """거리별 그림 1장, 밴드 4개 subplot. 각 각도에서 최대 k개 궤적 오버레이.

    색 = goal 각도(밴드 내 그라데이션), 선 스타일 = 성공(실선)/실패(점선).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    lim = dist + 1.5
    cmap = plt.cm.turbo
    for ax, (title, filt) in zip(axes.ravel(), BANDS):
        band = sorted(a for a in ANGLES if filt(a))
        handles = []
        for i, a in enumerate(band):
            color = cmap(i / max(len(band) - 1, 1))         # 밴드 내 각도별 그라데이션
            for reached, path, goal in trajs[a][:k]:
                ax.plot(path[:, 0], path[:, 1], color=color,
                        ls=("-" if reached else "--"), alpha=0.7, lw=1.1)
                ax.plot(*goal, marker="*", color=color, ms=8, mec="k", mew=0.3)
            handles.append(Line2D([0], [0], color=color, lw=2.5, label=f"{a:+d}°"))
        ax.plot(0, 0, marker="o", color="k", ms=8)          # 출발점
        ax.set_title(title); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.6)
    fig.suptitle(f"BC trajectories (goal {dist}m)  color=goal angle  "
                 f"solid=reach dashed=fail  *=goal  o=start", fontsize=12)
    fig.tight_layout()
    out = Path(out_dir) / f"bc_traj_{int(dist)}m.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bc",    type=str, default="bc/bc_policy.pt")
    p.add_argument("--dists", type=float, nargs="+", default=[3.0, 4.0, 5.0])
    p.add_argument("--n",     type=int,   default=20)
    p.add_argument("--k",     type=int,   default=4, help="밴드 그림에 각도당 궤적 개수")
    p.add_argument("--out",   type=str,   default="bc/eval_plots")
    args = p.parse_args()

    action_fn = make_bc_action_fn(args.bc)
    cfg = replace(EnvConfig(), goal_dist_min=2.0, goal_dist_max=4.0, max_steer=0.5)
    env = LunarRoverEnv(cfg=cfg)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    for dist in args.dists:
        res, trajs = survey_dist(env, action_fn, dist, args.n)
        print_hist(res, dist, args.n)
        out = plot_trajs(trajs, dist, args.out, args.k)
        print(f"  궤적 그림 → {out}")


if __name__ == "__main__":
    main()
