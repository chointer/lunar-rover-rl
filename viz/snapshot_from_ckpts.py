"""저장된 체크포인트들로 학습-진화 스냅샷을 사후 생성 (#23, 콜백 대안).

TrajSnapshotCallback과 산출물이 동일하다(고정 조건 re-evaluation). 차이는 시점뿐:
콜백은 학습 중 기록, 이건 이미 저장된 ckpt를 나중에 로드해 재현.
σ = exp(log_std)는 정책 파라미터라 ckpt 안에 있으므로 사후에도 그대로 얻는다.

40개처럼 많은 ckpt는 균등 subsample(기본 8개)해서 그림이 빽빽해지지 않게 한다.

  python viz/snapshot_from_ckpts.py --run experiments/exp01-easy-goal/exp01-00-baseline
  python viz/plot_snapshots.py --snapshots experiments/.../traj_snapshots.pkl
"""
import argparse
import pickle
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO

from envs.config import EnvConfig
from envs.lunar_rover_env import LunarRoverEnv

# 고정 평가 조건 (heading 기준 각도°, 거리 m) — 45° 간격 8방향.
# 좌우 부호를 모두 넣어야 좌우 비대칭(이 정책의 핵심 결함)이 그림에 드러난다.
DEFAULT_CONDITIONS = [(a, 3.0) for a in (135, 90, 45, 180, 0, -135, -90, -45)]


def find_ckpts(ckpt_dir, run):
    """(step, zip, vecnorm) 목록을 step 오름차순으로. 숫자 스텝 체크포인트만(‘last’ 제외)."""
    out = []
    for z in ckpt_dir.glob(f"{run}_*_steps.zip"):
        m = re.fullmatch(rf"{re.escape(run)}_(\d+)_steps", z.stem)
        if not m:
            continue
        step = int(m.group(1))
        vn = z.with_name(f"{run}_vecnormalize_{step}_steps.pkl")
        if vn.exists():
            out.append((step, z, vn))
    return sorted(out)


def subsample(items, k):
    """양끝 포함 균등하게 k개 선택."""
    if len(items) <= k:
        return items
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in sorted(set(idx))]


def normalizer(vn_path):
    with open(vn_path, "rb") as f:
        vn = pickle.load(f)
    rms, clip, eps = vn.obs_rms, vn.clip_obs, vn.epsilon
    return lambda o: np.clip((o - rms.mean) / np.sqrt(rms.var + eps), -clip, clip).astype(np.float32)


def rollout(env, model, norm, angle, dist, seed, deterministic):
    obs, _ = env.reset(seed=seed, options={"goal_angle": angle, "goal_dist": dist})
    path = [env.data.qpos[:2].copy()]
    for _ in range(env.cfg.max_steps):
        act = model.predict(norm(obs), deterministic=deterministic)[0]
        obs, _, term, trunc, _ = env.step(act)
        path.append(env.data.qpos[:2].copy())
        if term or trunc:
            break
    return np.asarray(path, np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run",  type=str, required=True, help="run 디렉토리 (config.yaml, ckpt/ 포함)")
    p.add_argument("--n",    type=int, default=8, help="subsample할 스냅샷 개수")
    p.add_argument("--mode", type=str, default="both", choices=["stochastic", "deterministic", "both"])
    p.add_argument("--n-stochastic", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out",  type=str, default=None)
    args = p.parse_args()

    run_dir = Path(args.run); run = run_dir.name
    cfg = replace(EnvConfig(), **(yaml.safe_load((run_dir / "config.yaml").read_text()).get("env") or {}))
    env = LunarRoverEnv(cfg=cfg)

    ckpts = subsample(find_ckpts(run_dir / "ckpt", run), args.n)
    print(f"{run}: 체크포인트 {len(ckpts)}개 사용 (steps {[s for s,_,_ in ckpts]})")

    snapshots = []
    for step, zip_path, vn_path in ckpts:
        model = PPO.load(zip_path)
        norm  = normalizer(vn_path)
        sigma = np.exp(model.policy.log_std.detach().cpu().numpy())
        snap = {"step": step, "sigma": sigma, "trajs": {}}
        for angle, dist in DEFAULT_CONDITIONS:
            entry = {}
            if args.mode in ("deterministic", "both"):
                entry["det"] = rollout(env, model, norm, angle, dist, args.seed, True)
            if args.mode in ("stochastic", "both"):
                # 지형 seed 고정, action 샘플링만 다르게 → fan 폭이 순수하게 탐험(σ)을 반영
                # (seed를 바꾸면 지형 차이가 섞여, σ가 줄어도 fan이 안 좁아 보인다)
                entry["stoch"] = [rollout(env, model, norm, angle, dist, args.seed, False)
                                  for _ in range(args.n_stochastic)]
            snap["trajs"][(angle, dist)] = entry
        snapshots.append(snap)
        print(f"  step {step:>7d}  σ={np.round(sigma, 3)}", flush=True)

    out = args.out or str(run_dir / "traj_snapshots.pkl")
    with open(out, "wb") as f:
        pickle.dump({"conditions": DEFAULT_CONDITIONS, "snapshots": snapshots}, f)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
