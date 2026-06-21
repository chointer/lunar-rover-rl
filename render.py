import mujoco
import mujoco.viewer
import numpy as np
from scipy.ndimage import gaussian_filter

NROW, NCOL = 64, 64
TERRAIN_HALF_X = 10   # rover.xml size[0]
TERRAIN_HALF_Y = 10   # rover.xml size[1]
TERRAIN_MAX_H  = 2.0  # rover.xml size[2]

def generate_heightmap(spawn_row=None, spawn_col=None, flat_radius=3):
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
        mujoco.mj_step(model, data)
        viewer.sync()
