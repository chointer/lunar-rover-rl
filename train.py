"""LunarRoverEnv PPO 학습 스크립트.

1차 목표는 sanity 학습 — env·reward·PPO로 이어지는 핵심 루프가 실제로 학습되는지 검증한다.
성능 목표가 아니다. return이 오르고 도달률이 무작위(0%)보다 높아지면 통과.
(참고: 휴리스틱 도달률 ~73%, 무작위 0% → 학습 신호는 존재)

  python train.py                           # 기본 30만 스텝, 8 env 병렬
  python train.py --timesteps 1000000       # 본 학습
  tensorboard --logdir runs/                # 학습 곡선 확인
  python sim_viewer.py --policy models/ppo  # 학습된 정책 눈으로 확인

VecNormalize 통계를 모델과 함께 저장한다(models/*_vecnorm.pkl). 이게 없으면 정책이
학습 때와 다른 스케일의 obs를 보게 되어 엉뚱하게 움직인다 — 정책과 항상 세트로 다룰 것.
"""
import argparse
from collections import defaultdict, deque
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from envs.lunar_rover_env import LunarRoverEnv


def make_env(seed, rank):
    def _init():
        env = LunarRoverEnv()
        env.reset(seed=seed + rank)
        # Monitor는 VecNormalize 안쪽에 둔다 → ep_rew_mean이 정규화 전 원본 return으로 기록됨
        return Monitor(env)
    return _init


class RolloutLogger(BaseCallback):
    """reward 항목별 기여도·원시 측정값·에피소드 종료 원인을 TensorBoard에 기록.

    항목별 기여도(r_*)를 봐야 가중치를 단계적으로 켤 때 각 항목이 실제로 얼마나
    기여하는지 알 수 있다. 원시값(energy·slip·speed·collided)은 가중치가 0이어도 크기를 보여준다.
    info를 이름으로 읽으므로 obs 레이아웃이 바뀌어도 조용히 깨지지 않는다.
    """
    KEYS = ("r_progress", "r_time", "r_goal", "r_flip", "r_energy", "r_collision",
            "energy", "slip", "speed", "collided")

    def __init__(self, window=100):
        super().__init__()
        self._sum     = defaultdict(float)
        self._steps   = 0
        self._outcome = deque(maxlen=window)   # 최근 window개 에피소드의 종료 원인

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            for k in self.KEYS:
                if k in info:
                    self._sum[k] += float(info[k])
            self._steps += 1

            if done:   # 종료 원인 구분 (도달 / 전복 / 타임아웃)
                if   info.get("reached"): self._outcome.append("goal")
                elif info.get("flipped"): self._outcome.append("flip")
                else:                     self._outcome.append("timeout")
        return True

    def _on_rollout_end(self) -> None:
        if self._steps:
            for k, v in self._sum.items():
                self.logger.record(f"detail/{k}", v / self._steps)   # 스텝당 평균
        if self._outcome:
            n = len(self._outcome)
            for name in ("goal", "flip", "timeout"):
                self.logger.record(f"outcome/{name}_rate", self._outcome.count(name) / n)
        self._sum.clear()
        self._steps = 0


def main():
    p = argparse.ArgumentParser(description="LunarRoverEnv PPO 학습")
    p.add_argument("--timesteps", type=int, default=300_000, help="총 학습 스텝")
    p.add_argument("--n-envs",    type=int, default=8,       help="병렬 env 개수")
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--run-name",  type=str, default="ppo",   help="모델·로그 이름")
    args = p.parse_args()

    Path("models").mkdir(exist_ok=True)

    venv_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    venv = venv_cls([make_env(args.seed, i) for i in range(args.n_envs)])

    # obs: 중심화+스케일링. goal_rel의 std가 height_scan의 20배라 그대로 넣으면 지형을 무시하게 됨
    # reward: 스케일만 (평균을 빼면 에피소드 길이에 유불리가 생겨 최적 정책이 바뀜)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO("MlpPolicy", venv, seed=args.seed, verbose=1, tensorboard_log="runs/")

    ckpt = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),   # save_freq는 env 하나당 스텝 기준
        save_path="models/ckpt", name_prefix=args.run_name,
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=[RolloutLogger(), ckpt],
        tb_log_name=args.run_name,
        progress_bar=True,
    )

    model.save(f"models/{args.run_name}")
    venv.save(f"models/{args.run_name}_vecnorm.pkl")   # obs 정규화 통계 — 정책과 항상 세트
    venv.close()

    print(f"\n저장: models/{args.run_name}.zip + models/{args.run_name}_vecnorm.pkl")
    print(f"확인: python sim_viewer.py --policy models/{args.run_name}")


if __name__ == "__main__":
    main()
