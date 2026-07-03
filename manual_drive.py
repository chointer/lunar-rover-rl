import time
import mujoco
import mujoco.viewer
import numpy as np
from scipy.ndimage import gaussian_filter
from keyboard_input import KeyboardInput, KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT

MAX_SPEED = 5.0  # rad/s — 전진/후진 최대 바퀴 속도
MAX_STEER = 0.5  # rad   — 최대 조향각 (±28°)

NROW, NCOL = 64, 64
TERRAIN_HALF_X = 10   # rover.xml size[0]
TERRAIN_HALF_Y = 10   # rover.xml size[1]
TERRAIN_MAX_H  = 2.0  # rover.xml size[2]

def generate_heightmap(spawn_row=None, spawn_col=None, flat_radius=12):
    def octave(sigma):
        return gaussian_filter(np.random.uniform(0, 1, (NROW, NCOL)), sigma=sigma)

    combined = 0.6 * octave(12) + 0.3 * octave(5) + 0.1 * octave(2)
    combined = (combined - combined.min()) / (combined.max() - combined.min())

    # spawn 주변을 부드럽게 평탄화
    if spawn_row is None: spawn_row = NROW // 2
    if spawn_col is None: spawn_col = NCOL // 2
    r, c = np.mgrid[0:NROW, 0:NCOL]
    dist = np.sqrt((r - spawn_row)**2 + (c - spawn_col)**2)
    flat_mask = np.clip(1.0 - dist / flat_radius, 0, 1)
    spawn_val = combined[spawn_row, spawn_col]
    combined = combined * (1 - flat_mask) + spawn_val * flat_mask

    return combined.astype(np.float32)

def terrain_height_at(heightmap, x, y):
    """world (x, y) 좌표의 실제 지형 높이 반환"""
    col = int((x + TERRAIN_HALF_X) / (2 * TERRAIN_HALF_X) * NCOL)
    row = int((y + TERRAIN_HALF_Y) / (2 * TERRAIN_HALF_Y) * NROW)
    col = np.clip(col, 0, NCOL - 1)
    row = np.clip(row, 0, NROW - 1)
    return float(heightmap[row, col]) * TERRAIN_MAX_H

# ===== 키보드 상태 추적 =====
# 터미널 창에 포커스를 두고 화살표 키 입력, 뷰어 창은 시뮬 확인용
kb = KeyboardInput()

# ===== 조작 방법 안내 =====
print("터미널 창 포커스 유지 | ↑/↓: 전진/후진 | ←/→: 좌/우 조향 | Ctrl+C: 종료")

model = mujoco.MjModel.from_xml_path("envs/assets/rover.xml")
data  = mujoco.MjData(model)

heightmap = generate_heightmap()
model.hfield_data[:] = heightmap.flatten()

# spawn 위치 (x=0, y=0) 지형 위에 rover 배치
spawn_x, spawn_y = 0.0, 0.0
ground_z = terrain_height_at(heightmap, spawn_x, spawn_y)
data.qpos[0] = spawn_x
data.qpos[1] = spawn_y
data.qpos[2] = ground_z + 0.25   # + 바퀴 반지름(0.15) + 바퀴-chassis 오프셋(0.1)
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [spawn_x, spawn_y, ground_z + 1.0]
    viewer.cam.distance  = 8.0
    viewer.cam.elevation = -20
    viewer.cam.azimuth   = 135

    while viewer.is_running():
        speed = MAX_SPEED if kb.is_held(KEY_UP) else (-MAX_SPEED if kb.is_held(KEY_DOWN) else 0.0)
        steer = MAX_STEER if kb.is_held(KEY_LEFT) else (-MAX_STEER if kb.is_held(KEY_RIGHT) else 0.0)

        # Ackermann: ctrl[0~1] 조향 position, ctrl[2~5] 구동 velocity
        data.ctrl[0] = steer  # act_steer_fl
        data.ctrl[1] = steer  # act_steer_fr
        if speed != 0.0:
            data.ctrl[2] = speed  # act_fl
            data.ctrl[3] = speed  # act_fr
            data.ctrl[4] = speed  # act_rl
            data.ctrl[5] = speed  # act_rr
        else:
            # 키를 놓으면 현재 속도 유지 → 관성으로 자연감속 (토크=0)
            data.ctrl[2] = data.sensor('vel_fl').data[0]
            data.ctrl[3] = data.sensor('vel_fr').data[0]
            data.ctrl[4] = data.sensor('vel_rl').data[0]
            data.ctrl[5] = data.sensor('vel_rr').data[0]

        step_start = time.perf_counter()
        mujoco.mj_step(model, data)
        viewer.sync()
        elapsed = time.perf_counter() - step_start
        time.sleep(max(0, model.opt.timestep - elapsed))

