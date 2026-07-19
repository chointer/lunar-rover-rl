"""LunarRoverEnv PPO 학습 스크립트.

1차 목표는 sanity 학습 — env·reward·PPO로 이어지는 핵심 루프가 실제로 학습되는지 검증한다.
성능 목표가 아니다. return이 오르고 도달률이 무작위(0%)보다 높아지면 통과.
(참고: 휴리스틱 도달률 ~73%, 무작위 0% → 학습 신호는 존재)

  python train.py --config configs/baseline.yaml
  python train.py --config configs/w_time_low.yaml --timesteps 1000000   # CLI가 yaml을 덮어씀
  tensorboard --logdir experiments            # 여러 실험 곡선 비교
  python sim_viewer.py --policy experiments/<run>/ckpt/<run>_last_steps.zip

설정은 yaml로 관리하고, 해석된 전체 설정을 experiments/<run>/config.yaml에 기록한다
(덮어쓴 값만이 아니라 EnvConfig 전 필드 → 결과와 설정이 폴더 안에서 완결).

VecNormalize 통계를 정책과 같은 이름 규칙으로 저장한다. 이게 없으면 정책이 학습 때와
다른 스케일의 obs를 보게 되어 엉뚱하게 움직인다 — 정책과 항상 세트로 다룰 것.
"""
import argparse
from collections import defaultdict, deque
from dataclasses import asdict, replace
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from envs.config import EnvConfig
from envs.lunar_rover_env import LunarRoverEnv


def make_env(cfg, seed, rank):
    def _init():
        env = LunarRoverEnv(cfg=cfg)
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
    p.add_argument("--config", type=str, required=True, help="실험 설정 yaml (configs/*.yaml)")
    # 아래는 yaml 값을 덮어쓰는 선택 인자 (자주 바꾸는 것만). 미지정이면 yaml을 따른다
    p.add_argument("--run-name",  type=str, default=None)
    p.add_argument("--timesteps", type=str, default=None, help="총 학습 스텝 (예: 1e6)")
    p.add_argument("--seed",      type=int, default=None, help="난수 시드 (seed sweep용)")
    p.add_argument("--group",     type=str, default=None, help="실험 묶음 (config 재사용해 다른 그룹으로)")
    args = p.parse_args()

    conf = yaml.safe_load(Path(args.config).read_text()) or {}
    run_name  = args.run_name or conf["run_name"]
    timesteps = int(float(args.timesteps)) if args.timesteps else int(conf["timesteps"])
    seed      = args.seed if args.seed is not None else int(conf["seed"])
    group     = args.group or conf.get("group")                    # 실험 묶음 (없으면 experiments/ 바로 밑)
    n_envs    = int(conf["n_envs"])

    env_cfg    = replace(EnvConfig(), **(conf.get("env") or {}))   # 미지정 필드는 EnvConfig 기본값
    ppo_kwargs = conf.get("ppo") or {}                             # 비우면 SB3 기본값

    # 산출물을 한 폴더에 모은다: experiments/[<group>/]<run>/{config.yaml, ckpt, tb_N}
    base_dir = Path("experiments") / group if group else Path("experiments")
    exp_dir  = base_dir / run_name
    ckpt_dir = exp_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 해석된 전체 설정을 기록 (덮어쓴 값만이 아니라 EnvConfig 전 필드 → 결과와 설정이 폴더 안에서 완결)
    (exp_dir / "config.yaml").write_text(yaml.safe_dump(
        {"group": group, "run_name": run_name, "timesteps": timesteps, "n_envs": n_envs,
         "seed": seed, "env": asdict(env_cfg), "ppo": ppo_kwargs},
        sort_keys=False, allow_unicode=True))

    venv_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    venv = venv_cls([make_env(env_cfg, seed, i) for i in range(n_envs)])

    # obs: 중심화+스케일링. goal_rel의 std가 height_scan의 20배라 그대로 넣으면 지형을 무시하게 됨
    # reward: 스케일만 (평균을 빼면 에피소드 길이에 유불리가 생겨 최적 정책이 바뀜)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO("MlpPolicy", venv, seed=seed, verbose=1,
                tensorboard_log=str(exp_dir), **ppo_kwargs)

    ckpt = CheckpointCallback(
        # save_freq는 venv.step() 호출 횟수 기준 (1회 = 총 n_envs 스텝) → 총 스텝으로 환산해 나눔
        save_freq=max(50_000 // n_envs, 1),
        save_path=str(ckpt_dir), name_prefix=run_name,
        save_vecnormalize=True,   # 통계 없이 저장하면 그 체크포인트는 평가에 못 씀 (기본값이 False라 명시 필요)
    )
    model.learn(
        total_timesteps=timesteps,
        callback=[RolloutLogger(), ckpt],
        tb_log_name="tb",         # experiments/<run>/tb_N/ (재실행마다 N 증가 → 곡선 분리 보존)
        progress_bar=True,
    )

    # 최종 모델도 체크포인트와 동일한 이름 규칙으로 저장 (ID 자리에 "last")
    #   정책:  <run>_last_steps.zip   통계: <run>_vecnormalize_last_steps.pkl
    # → 중간 ckpt와 최종을 sim_viewer가 같은 규칙 하나로 짝지어 찾는다
    model.save(str(ckpt_dir / f"{run_name}_last_steps.zip"))
    venv.save(str(ckpt_dir / f"{run_name}_vecnormalize_last_steps.pkl"))
    venv.close()

    print(f"\n저장: {ckpt_dir}/{run_name}_last_steps.zip (+ vecnormalize 짝)")
    print(f"설정: {exp_dir}/config.yaml")
    print(f"확인: python sim_viewer.py --policy {ckpt_dir}/{run_name}_last_steps.zip")
    print(f"곡선: tensorboard --logdir {exp_dir}   (여러 실험 비교는 --logdir experiments)")


if __name__ == "__main__":
    main()
