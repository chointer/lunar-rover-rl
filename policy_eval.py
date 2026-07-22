"""정책 성능 평가 (재사용 모듈).

여러 정책을 '동일한 (평가seed × 거리 × 각도) 격자'에서 돌려 목표 도달률을 측정한다.
action_fn(obs->action)만 있으면 PPO·BC 등 무엇이든 평가 가능. 
결과는 집계 메서드를 가진 EvalResult로 반환.

평가 seed: 한 정책을 여러 지형에서 시행 (evaluate(seeds=...)). 
지형은 seed로 재현 가능하고, 모든 정책이 같은 seed 집합을 보므로 같은 지형에서 비교된다.

CLI:
  python policy_eval.py --dists 2 3 4 5 --angle-step 15 --seeds 10 \
      baseline=a.zip,b.zip gsde=c.zip,d.zip
"""
import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from envs.config import EnvConfig
from envs.lunar_rover_env import LunarRoverEnv


# ===== 정책 로더 (obs->action 함수) =====

def load_ppo_action_fn(policy_path, deterministic=True):
    """PPO zip(+짝 vecnormalize pkl)을 로드해 obs->action 함수를 반환.

    학습 때 쓴 VecNormalize 통계로 obs를 동일 정규화한다(빠뜨리면 정책이 엉뚱하게 움직임).
    """
    from stable_baselines3 import PPO
    p = Path(policy_path)
    model = PPO.load(policy_path)
    prefix, ident = p.stem.removesuffix("_steps").rsplit("_", 1)
    vn = p.with_name(f"{prefix}_vecnormalize_{ident}_steps.pkl")
    if vn.exists():
        with open(vn, "rb") as f:
            v = pickle.load(f)
        rms, clip, eps = v.obs_rms, v.clip_obs, v.epsilon
        norm = lambda o: np.clip((o - rms.mean) / np.sqrt(rms.var + eps), -clip, clip).astype(np.float32)
    else:
        norm = lambda o: o
    return lambda obs: model.predict(norm(obs), deterministic=deterministic)[0]


# ===== 평가 결과 (정책 1개) =====

@dataclass
class EvalResult:
    """도달 여부 [평가seed, 거리, 각도] 불리언 배열 + 축. 집계 메서드 제공."""
    reached: np.ndarray          # bool [n_seeds, n_dists, n_angles]
    seeds: list
    dists: list
    angles: list

    def rate(self):
        """전체 도달률 %."""
        return 100.0 * self.reached.mean()

    def per_angle(self):
        """각도별 도달률 % (seed·거리 평균) -> {angle: %}."""
        return dict(zip(self.angles, 100.0 * self.reached.mean(axis=(0, 1))))

    def per_dist(self):
        """거리별 도달률 % (seed·각도 평균) -> {dist: %}."""
        return dict(zip(self.dists, 100.0 * self.reached.mean(axis=(0, 2))))

    def _band_idx(self, lo, hi):
        return [i for i, a in enumerate(self.angles) if lo <= abs(a) <= hi]

    def band(self, lo, hi, dist=None):
        """|각도|∈[lo,hi] 밴드 도달률 % (전 seed·거리 평균). dist 지정 시 그 거리만."""
        sub = self.reached[:, :, self._band_idx(lo, hi)]
        if dist is not None:
            sub = sub[:, [self.dists.index(dist)], :]
        return 100.0 * sub.mean()

    def band_per_seed(self, lo, hi, dist=None):
        """밴드 도달률을 평가 seed별로 -> [n_seeds]. (지형 분산 확인용)"""
        sub = self.reached[:, :, self._band_idx(lo, hi)]
        if dist is not None:
            sub = sub[:, [self.dists.index(dist)], :]
        return 100.0 * sub.mean(axis=(1, 2))


# ===== 평가기 (여러 정책을 동일 격자에서) =====

class PolicyEvaluator:
    def __init__(self, dists=(2, 3, 4, 5), angle_step=15, n_angles=None, env_cfg=None):
        """dists: 목표 거리 리스트. 각도는 angle_step(도) 간격 OR n_angles개 균등 분할."""
        self.dists = list(dists)
        if n_angles is not None:
            self.angles = [int(round(a)) for a in np.linspace(-180, 180, n_angles, endpoint=False)]
        else:
            self.angles = list(range(-180 + angle_step, 181, angle_step))   # 예: 15° → -165..180 (24개)
        self.env = LunarRoverEnv(cfg=env_cfg or EnvConfig())

    def _rollout(self, action_fn, seed, dist, angle):
        """한 조건(seed 지형, 거리, 각도)에서 deterministic 1회 실행 → 도달 여부(bool)."""
        obs, _ = self.env.reset(seed=seed, options={"goal_angle": angle, "goal_dist": dist})
        info = {}
        for _ in range(self.env.cfg.max_steps):
            obs, _, term, trunc, info = self.env.step(action_fn(obs))
            if term or trunc:
                break
        return bool(info.get("reached", False))

    def evaluate(self, policies, seeds, verbose=True):
        """여러 정책을 동일 (평가seed × 거리 × 각도) 격자에서 평가 → {이름: EvalResult}.

        policies: {이름: 경로 or action_fn}. 경로(str/Path)면 PPO로 자동 로드.
        seeds: 정수(단일 지형) 또는 지형 seed 리스트. 모든 정책이 같은 seed 집합을 본다(공정 비교).
        지형은 seed로 재현되며, 한 seed 내에선 모든 거리·각도가 같은 지형을 공유(지형 통제).
        """
        seeds = [seeds] if isinstance(seeds, int) else list(seeds)
        shape = (len(seeds), len(self.dists), len(self.angles))
        out = {}
        for name, pol in policies.items():
            fn = pol if callable(pol) else load_ppo_action_fn(pol)
            reached = np.zeros(shape, dtype=bool)
            for si, s in enumerate(seeds):
                for di, dist in enumerate(self.dists):
                    for ai, ang in enumerate(self.angles):
                        reached[si, di, ai] = self._rollout(fn, s, dist, ang)
            out[name] = EvalResult(reached, seeds, self.dists, self.angles)
            if verbose:
                print(f"  {name}: 전체 {out[name].rate():.0f}%", flush=True)
        return out

    def close(self):
        self.env.close()


# ===== 모델 seed 간 집계 =====

def summarize_models(results, lo, hi, dist=None):
    """모델 여러 개(EvalResult 리스트)의 밴드 도달률 → (평균, 표준편차) across 모델.

    각 모델을 자기 평가 seed 평균 밴드값 1개로 요약한 뒤, 모델 간 mean±std.
    """
    vals = [r.band(lo, hi, dist=dist) for r in results]
    return float(np.mean(vals)), float(np.std(vals))


# ===== 저장 / 로드 =====

def save_results(path, results, group_names, dists, angles, seeds):
    """평가 결과를 pkl로 저장. 커스텀 클래스 없이 순수 배열·리스트만 담아 어디서든 로드 가능."""
    with open(path, "wb") as f:
        pickle.dump({"results": {n: r.reached for n, r in results.items()},
                     "group_names": group_names,
                     "dists": dists, "angles": angles, "seeds": seeds}, f)


def load_results(path):
    """저장된 pkl 로드 → results를 EvalResult로 복원한 dict 반환 (그래프·재집계용)."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    d["results"] = {n: EvalResult(reached, d["seeds"], d["dists"], d["angles"])
                    for n, reached in d["results"].items()}
    return d


# ===== CLI =====

BANDS = [("앞", 0, 30), ("옆", 60, 120), ("뒤", 150, 180)]


def _parse_group(arg):
    """'label=p1.zip,p2.zip' -> (label, [p1, p2])."""
    label, paths = arg.split("=", 1)
    return label, paths.split(",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+", type=_parse_group,
                    help="label=path1,path2,... (그룹별 모델 정책들)")
    ap.add_argument("--dists", nargs="+", type=float, default=[2, 3, 4, 5])
    ap.add_argument("--angle-step", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=10, help="평가 지형 seed 개수")
    ap.add_argument("--seed-start", type=int, default=0, help="평가 지형 seed 시작값 (seed_start..+N-1)")
    ap.add_argument("--out", type=str, default="eval_results.pkl",
                    help="결과 저장 경로(pkl). 나중에 그래프용으로 원자료 전체를 담는다")
    args = ap.parse_args()

    ev = PolicyEvaluator(dists=args.dists, angle_step=args.angle_step)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    # 모든 그룹의 정책을 하나의 dict로 모아 동일 격자에서 한 번에 평가
    policies, group_names = {}, {}
    for label, paths in args.groups:
        group_names[label] = []
        for pth in paths:
            name = f"{label}#{Path(pth).parent.parent.name}"
            policies[name] = pth
            group_names[label].append(name)

    print(f"평가: dists={args.dists}, angle_step={args.angle_step}°, 지형 seed {args.seeds}개, deterministic\n")
    results = ev.evaluate(policies, seeds)

    # 거리 × 밴드, 그룹별 모델 간 평균±표준편차
    print("\n=== 거리 × 밴드 도달률 (모델 seed 간 평균±표준편차 %) ===")
    header = "거리 | 밴드       | " + " | ".join(f"{lab:>13}" for lab in group_names)
    print(header); print("-" * len(header))
    for dist in args.dists:
        for bname, lo, hi in BANDS:
            cells = []
            for lab in group_names:
                grp = [results[n] for n in group_names[lab]]
                m, sd = summarize_models(grp, lo, hi, dist=dist)
                cells.append(f"{m:>4.0f} ± {sd:>4.1f}")
            print(f"{dist:>4.0f} | {bname}({lo:>3}-{hi:>3}) | " + " | ".join(f"{c:>13}" for c in cells))
    ev.close()

    # 원자료 전체 저장 (나중에 load_results로 EvalResult 복원 → per_angle()/band() 등으로 그래프)
    save_results(args.out, results, group_names, ev.dists, ev.angles, seeds)
    print(f"\n→ 저장: {args.out}  (로드: policy_eval.load_results('{args.out}'))")


if __name__ == "__main__":
    main()
