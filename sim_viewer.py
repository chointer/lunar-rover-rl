"""LunarRoverEnv를 실시간 뷰어로 돌리는 통합 도구.

세 가지 action 소스를 지원한다 (전부 onscreen viewer라 GPU 없이 동작):
  python sim_viewer.py                          # 휴리스틱 (목표 방향 자동, env 검증)
  python sim_viewer.py --manual                 # 키보드 수동 조작
  python sim_viewer.py --policy models/ppo.zip  # 학습된 정책 (결과 확인)

목표는 빨간 구, obs height-scan 관측 지점은 파란 점으로 표시하고, 콘솔에 거리·reward·소비일률을 출력한다.
에피소드가 끝나면 자동 reset한다. (녹화는 학습 스크립트에서 별도로 처리)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
from envs.lunar_rover_env import LunarRoverEnv

SEED = 0


# ===== action 소스 =====
def heuristic_action(obs):
    """goal_rel(obs 마지막 2개, rover 로컬)을 향해 조향하며 전진."""
    goal_rel = obs[-2:]
    steer = np.clip(np.arctan2(goal_rel[1], goal_rel[0]), -1.0, 1.0)
    return np.array([1.0, steer], dtype=np.float32)


def make_keyboard_action():
    """키보드 상태로 action을 만드는 함수 반환 (↑↓=전후진, ←→=조향)."""
    from keyboard_input import KeyboardInput, KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT
    kb = KeyboardInput()

    def action_fn(_obs):
        speed = 1.0 if kb.is_held(KEY_UP)   else (-1.0 if kb.is_held(KEY_DOWN)  else 0.0)
        steer = 1.0 if kb.is_held(KEY_LEFT) else (-1.0 if kb.is_held(KEY_RIGHT) else 0.0)
        return np.array([speed, steer], dtype=np.float32)
    return action_fn


def _vecnorm_for(policy_path):
    """정책 zip 경로 → 짝이 되는 VecNormalize 통계 pkl 경로.

    train.py·SB3 CheckpointCallback의 이름 규칙을 그대로 뒤집는다:
      <prefix>_<ID>_steps.zip  ↔  <prefix>_vecnormalize_<ID>_steps.pkl
    (ID는 중간 ckpt면 스텝 수, 최종이면 "last"). 규칙 하나로 둘 다 짝을 찾는다.
    """
    p = Path(policy_path)
    stem = p.stem                                    # <prefix>_<ID>_steps
    if stem.endswith("_steps"):
        prefix, ident = stem.removesuffix("_steps").rsplit("_", 1)
        return p.with_name(f"{prefix}_vecnormalize_{ident}_steps.pkl")
    return p.with_name(p.stem + "_vecnormalize.pkl")   # 규칙 밖 이름이면 단순 폴백


def make_policy_action(policy_path):
    """학습된 정책(.zip)을 로드해 action을 만드는 함수 반환 (PPO는 이 모드에서만 import).

    학습 때 쓴 VecNormalize 통계를 같은 폴더에서 찾아 obs를 동일하게 정규화한다.
    이걸 빠뜨리면 정책이 학습 때와 다른 스케일의 obs를 보게 되어 엉뚱하게 움직인다.
    """
    import pickle
    from stable_baselines3 import PPO
    model = PPO.load(policy_path)

    vn_path = _vecnorm_for(policy_path)
    if vn_path.exists():
        with open(vn_path, "rb") as f:
            vecnorm = pickle.load(f)
        rms, clip, eps = vecnorm.obs_rms, vecnorm.clip_obs, vecnorm.epsilon
        print(f"obs 정규화 통계 로드: {vn_path}")

        def normalize(obs):
            return np.clip((obs - rms.mean) / np.sqrt(rms.var + eps), -clip, clip).astype(np.float32)
    else:
        print(f"경고: {vn_path} 없음 → obs 정규화 없이 실행 "
              f"(학습에 VecNormalize를 썼다면 정책이 이상하게 움직입니다)")
        normalize = lambda obs: obs

    def action_fn(obs):
        return model.predict(normalize(obs), deterministic=True)[0]
    return action_fn


# ===== 공통 뷰어 루프 =====
def draw_markers(viewer, env, obs):
    """목표·obs 관측 지점을 뷰어에 표시 (env._draw_markers 공용 헬퍼 사용, render와 동일)."""
    viewer.user_scn.ngeom = 0                                    # 매 프레임 새로 채움
    env._draw_markers(viewer.user_scn, obs[:env._scan_offsets.shape[1]])


def run(action_fn, seed=SEED):
    env = LunarRoverEnv()
    obs, _ = env.reset(seed=seed)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance  = 8.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth   = 135

        ep, ep_ret, ep_len = 0, 0.0, 0
        while viewer.is_running():
            action = action_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            ep_len += 1

            power = float(np.sum(np.abs(env.data.actuator_force * env.data.actuator_velocity)))

            draw_markers(viewer, env, obs)
            viewer.cam.lookat[:] = env.data.qpos[:3]      # 카메라가 rover를 따라감
            viewer.sync()
            time.sleep(env.cfg.frame_skip * env.model.opt.timestep)  # 실시간 재생

            dist = float(np.linalg.norm(env._goal - env.data.qpos[:2]))
            print(f"\rep{ep} step{ep_len:3d}  dist={dist:5.2f}m  "
                  f"reward={reward:+.2f}  power={power:5.1f}W  return={ep_ret:+.2f}", end="")

            if terminated or truncated:
                if dist < env.cfg.goal_radius:
                    reason = "도달 ✅"
                elif terminated:
                    reason = "전복 💥"
                else:
                    reason = "타임아웃 ⏱"
                print(f"   → {reason}  (return={ep_ret:+.2f}, len={ep_len})")
                ep, ep_ret, ep_len = ep + 1, 0.0, 0
                obs, _ = env.reset()
                viewer.update_hfield(0)   # 새 지형을 GPU에 재업로드


def main():
    parser = argparse.ArgumentParser(description="LunarRoverEnv 실시간 뷰어")
    # action 소스는 하나만: --manual과 --policy는 동시 사용 불가 (없으면 휴리스틱)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--manual", action="store_true", help="키보드 수동 조작")
    src.add_argument("--policy", type=str, default=None, help="학습된 정책(.zip) 경로")
    args = parser.parse_args()

    if args.policy:
        action_fn = make_policy_action(args.policy)
        print(f"정책 모드: {args.policy}")
    elif args.manual:
        action_fn = make_keyboard_action()
        print("수동 조작 모드 | 터미널 포커스 유지 | ↑↓ 전후진  ←→ 조향  Ctrl+C 종료")
    else:
        action_fn = heuristic_action
        print("휴리스틱 모드 (목표 방향 자동 주행)")

    run(action_fn)


if __name__ == "__main__":
    main()
