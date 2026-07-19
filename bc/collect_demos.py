"""휴리스틱 expert 시연 수집 (BC 학습용).

BC는 expert가 가진 기술만 복제한다. 휴리스틱은 앞/어깨(±45~60°)·뒤는 잘하지만
정옆(±90°)은 못한다 → 그 구멍은 사람이 직접 모은 3점 회전 시연으로 따로 보강한다(별도 수집).

covariate shift 대비 (DART식):
  실행 action에는 노이즈를 섞어 궤적 주변 상태까지 방문하되,
  저장하는 라벨은 그 상태에서의 휴리스틱 정답(노이즈 없는 값)으로 둔다.
  → BC가 "정상 궤적에서 살짝 벗어난 상태에서도 되돌아오는 법"을 배운다.
휴리스틱이 함수(obs→action)라 어떤 상태든 공짜로 라벨할 수 있어 가능.

성공 에피소드의 transition만 저장한다(실패=맴돌기 행동을 복제하지 않도록).

  python bc/collect_demos.py --episodes 600 --out bc/demos_heuristic.npz
"""
import argparse
from pathlib import Path

import numpy as np

from envs.config import EnvConfig
from envs.lunar_rover_env import LunarRoverEnv


def heuristic(obs):
    """goal_rel(obs 마지막 2개, rover 로컬)을 향해 조향하며 전진 (노이즈 없는 정답)."""
    g = obs[-2:]
    steer = float(np.clip(np.arctan2(g[1], g[0]), -1.0, 1.0))
    return np.array([1.0, steer], dtype=np.float32)


def collect(episodes, noise_std, success_only, goal_range, seed):
    cfg = EnvConfig(goal_dist_min=goal_range[0], goal_dist_max=goal_range[1])
    env = LunarRoverEnv(cfg=cfg)

    obs_all, act_all, ep_id_all = [], [], []
    n_success = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_obs, ep_act = [], []
        reached = False
        for _ in range(cfg.max_steps):
            a = heuristic(obs)                       # 라벨(정답)
            ep_obs.append(obs.astype(np.float32))
            ep_act.append(a)
            a_exec = a.copy()
            if noise_std > 0:                        # 실행만 노이즈(steer) → 상태 커버리지
                a_exec[1] = np.clip(a[1] + np.random.normal(0, noise_std), -1.0, 1.0)
            obs, _, term, trunc, info = env.step(a_exec)
            if term or trunc:
                reached = bool(info.get("reached"))
                break
        if reached:
            n_success += 1
        if reached or not success_only:              # 성공만(기본) 저장
            obs_all.extend(ep_obs)
            act_all.extend(ep_act)
            ep_id_all.extend([ep] * len(ep_obs))     # 에피소드 ID → train/val 누수 없는 분할용

    return (np.asarray(obs_all, np.float32), np.asarray(act_all, np.float32),
            np.asarray(ep_id_all, np.int32), n_success)


def main():
    p = argparse.ArgumentParser(description="휴리스틱 시연 수집 (BC용)")
    p.add_argument("--episodes",   type=int,   default=600)
    p.add_argument("--noise-std",  type=float, default=0.15, help="실행 steer 노이즈 std (라벨엔 미적용)")
    p.add_argument("--goal-range", type=float, nargs=2, default=[2.0, 4.0], metavar=("MIN", "MAX"))
    p.add_argument("--all",        action="store_true", help="실패 에피소드도 포함(기본은 성공만)")
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--out",        type=str,   default="bc/demos_heuristic.npz")
    args = p.parse_args()

    obs, act, ep_id, n_success = collect(
        args.episodes, args.noise_std, not args.all, args.goal_range, args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, obs=obs, act=act, ep_id=ep_id)

    print(f"에피소드 {args.episodes} 중 성공 {n_success} ({n_success/args.episodes*100:.0f}%)")
    print(f"저장 transition: {len(obs)}개  (obs {obs.shape}, act {act.shape})")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
