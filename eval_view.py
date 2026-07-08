"""env.step() 시각 검증용 뷰어.

휴리스틱(목표 방향으로 조향)으로 rover를 몰아 env.step()이 정상 동작하는지
눈으로 확인한다. 목표 지점은 빨간 구 마커로 표시하고, 콘솔에 거리·reward·소비
일률·종료 사유를 출력한다. 에피소드가 끝나면 자동으로 reset한다.

  python eval_view.py
"""
import time
import numpy as np
import mujoco
import mujoco.viewer
from envs.lunar_rover_env import LunarRoverEnv

SEED = 0


def heuristic_action(obs):
    """goal_rel(obs 마지막 2개, rover 로컬 좌표)을 향해 조향하며 전진."""
    goal_rel = obs[-2:]                                   # x=전방, y=좌
    steer = np.clip(np.arctan2(goal_rel[1], goal_rel[0]), -1.0, 1.0)
    return np.array([1.0, steer], dtype=np.float32)       # [전진 최대, 목표 쪽 조향]


def draw_goal(viewer, env):
    """목표 지점을 빨간 구로 표시 (뷰어 런타임 마커)."""
    gx, gy = env._goal
    gz = env._terrain_height_at(gx, gy) + 0.3
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([0.3, 0.0, 0.0]),
        pos=np.array([gx, gy, gz]),
        mat=np.eye(3).flatten(),
        rgba=np.array([1.0, 0.2, 0.2, 0.6]),
    )
    viewer.user_scn.ngeom = 1


def main():
    env = LunarRoverEnv()
    obs, _ = env.reset(seed=SEED)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance  = 8.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth   = 135

        ep, ep_ret, ep_len = 0, 0.0, 0
        while viewer.is_running():
            action = heuristic_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            ep_len += 1

            # 참고: 순간 소비 일률 (에너지 reward 감 잡기용, W)
            power = float(np.sum(np.abs(env.data.actuator_force * env.data.actuator_velocity)))

            draw_goal(viewer, env)
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
                viewer.update_hfield(0)   # 새 지형(hfield)을 GPU에 재업로드 (안 하면 첫 지형이 계속 보임)


if __name__ == "__main__":
    main()
