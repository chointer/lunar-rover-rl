"""exp04 gSDE vs baseline — 옆목표 궤적 대비 (머니샷).

동일 지형에서 정옆(±90°, 3m) 목표에 baseline과 gSDE의 deterministic 궤적을 겹쳐 그린다.
baseline은 옆으로 못 돌아 실패, gSDE는 큰 호로 도달 — "왜 됐나"를 직관적으로.
궤적 롤아웃이 필요해 env·정책을 로드한다(다른 plot 스크립트와 달리 self-contained 아님).

  python analysis/exp04_gsde/plot_moneyshot.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from policy_eval import load_ppo_action_fn          # noqa: E402
from envs.config import EnvConfig                    # noqa: E402
from envs.lunar_rover_env import LunarRoverEnv       # noqa: E402

COLORS = {"gSDE": "#0072B2", "baseline": "#E69F00"}
# 대표 seed 1개씩(첫 seed, cherry-pick 아님) @3M
POLICIES = {
    "baseline": ROOT / "experiments/exp02-seed-variance/exp02-00-seed1/ckpt/exp02-00-seed1_3000000_steps.zip",
    "gSDE":     ROOT / "experiments/exp04-explore/exp04-01-seed1/ckpt/exp04-01-seed1_3000000_steps.zip",
}
TERRAIN_SEEDS = [100, 101, 102, 103, 104]   # 여러 지형 겹쳐 메커니즘의 견고함을 보임(cherry-pick 아님)
ANGLES, DIST = [90, -90], 3.0


def rollout_path(env, action_fn, seed, dist, angle):
    obs, _ = env.reset(seed=seed, options={"goal_angle": angle, "goal_dist": dist})
    goal = env._goal.copy()
    path, reached = [env.data.qpos[:2].copy()], False
    for _ in range(env.cfg.max_steps):
        obs, _, term, trunc, info = env.step(action_fn(obs))
        path.append(env.data.qpos[:2].copy())
        if term or trunc:
            reached = bool(info.get("reached", False)); break
    return np.array(path), goal, reached


def plot(out=None):
    env = LunarRoverEnv(cfg=EnvConfig())
    fns = {lab: load_ppo_action_fn(str(p)) for lab, p in POLICIES.items()}

    fig, axes = plt.subplots(1, len(ANGLES), figsize=(5.4 * len(ANGLES), 5.4), squeeze=False)
    for j, ang in enumerate(ANGLES):
        ax = axes[0][j]
        goal = None
        n_reach = {lab: 0 for lab in fns}
        for lab, fn in fns.items():
            for s in TERRAIN_SEEDS:
                path, goal, reached = rollout_path(env, fn, s, DIST, ang)
                n_reach[lab] += reached
                ax.plot(path[:, 0], path[:, 1], color=COLORS[lab], lw=1.8, alpha=0.8,
                        ls="-" if reached else "--")
        # 범례: 정책별 도달 수
        handles = [plt.Line2D([], [], color=COLORS[lab], lw=2.4,
                              label=f"{lab} ({n_reach[lab]}/{len(TERRAIN_SEEDS)} reach)") for lab in fns]
        ax.plot(0, 0, "ko", ms=7)                                 # 출발
        ax.plot(goal[0], goal[1], "k*", ms=18, zorder=5)          # 목표(각 지형 goal 거의 동일 위치)
        ax.set_title(f"side goal {ang:+d}°, {DIST:.0f} m")
        ax.set_aspect("equal"); ax.grid(alpha=0.25, lw=0.6)
        ax.legend(handles=handles, loc="best", fontsize=9, framealpha=0.9)
    env.close()

    fig.suptitle(f"Side goal across {len(TERRAIN_SEEDS)} terrains — baseline stalls in place, "
                 "gSDE sustains an arc\n(solid = reached, dashed = failed)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = out or (HERE / "moneyshot_side.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("→", plot())
