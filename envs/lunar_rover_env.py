import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from scipy.ndimage import gaussian_filter, map_coordinates
from envs.config import EnvConfig

GRID_ROWS = 7
GRID_COLS = 7
CELL_SIZE = 0.5   # m/cell
assert GRID_ROWS % 2 == 1 and GRID_COLS % 2 == 1, "GRID_ROWS, GRID_COLS must be odd"

NROW, NCOL     = 64, 64
TERRAIN_HALF_X = 10.0   # rover.xml hfield size[0]
TERRAIN_HALF_Y = 10.0   # rover.xml hfield size[1]
TERRAIN_MAX_H  = 2.0    # rover.xml hfield size[2]

class LunarRoverEnv(gym.Env):

    def __init__(self, cfg: EnvConfig = None):
        super().__init__()
        # cfg 미지정 시 기본값. None 기본 인자로 mutable default 공유 문제 회피
        self.cfg = cfg if cfg is not None else EnvConfig()

        self.model = mujoco.MjModel.from_xml_path("envs/assets/rover.xml")
        self.data  = mujoco.MjData(self.model)

        # obs: height_grid(49) + imu(8) + wheel_vel(4) + goal_rel(2) = 63
        obs_dim = GRID_ROWS * GRID_COLS + 8 + 4 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # action: [전진속도, 조향] 각각 -1 ~ 1
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # rover 위치·속도 등 모든 시뮬레이션 상태를 초기값으로 리셋
        mujoco.mj_resetData(self.model, self.data)

        self._heightmap = self._generate_heightmap()
        self.model.hfield_data[:] = self._heightmap.flatten()     # 새로 생성한 지형을 MuJoCo 모델에 반영

        # 달 레골리스 마찰 랜덤화 (에피소드마다 전체 지형에 단일값 적용)
        # sliding 낮음 → 큰 토크 시 바퀴가 지면을 못 잡고 헛돎 (모래 위 wheel spin)
        #   참고: 콘크리트~0.8, 젖은 흙~0.5, 모래~0.3, 느슨한 레골리스~0.2
        # torsional 높음 → 회전 시 바퀴가 모래를 옆으로 밀어내는 저항 발생
        #   참고: 딱딱한 지면~0.005, 모래~0.05
        # rolling 높음 → 전진할수록 모래가 쌓이며 저항 증가하는 느낌 (sinkage 근사)
        #   참고: 딱딱한 지면~0.01, 모래~0.08~0.15
        # 세 값의 조합이 "단단한 바닥과 달리 모래에서 힘을 써도 잘 안 나가는" 거동을 만들어냄
        sliding   = self.np_random.uniform(0.2, 0.5)
        torsional = self.np_random.uniform(0.02, 0.08)
        rolling   = self.np_random.uniform(0.05, 0.12)
        self.model.geom('terrain').friction[:] = [sliding, torsional, rolling]

        # spawn 위치 설정, 지형 위에 rover 배치
        spawn_x, spawn_y = 0.0, 0.0                 # TODO: 랜덤 혹은 지정할 수 있게 수정
        ground_z = self._terrain_height_at(spawn_x, spawn_y)
        self.data.qpos[0] = spawn_x
        self.data.qpos[1] = spawn_y
        self.data.qpos[2] = ground_z + 0.25   # + 바퀴 반지름(0.15) + 바퀴-chassis 오프셋(0.1)

        # 목표 지점: spawn 기준 반경 3~8m 사이 랜덤, 지형 경계(절대 좌표) 안쪽으로 제한
        while True:
            angle,r         = self.np_random.uniform(0, 2 * np.pi), self.np_random.uniform(3.0, 8.0)
            goal_x, goal_y  = spawn_x + r * np.cos(angle), spawn_y + r * np.sin(angle)
            if abs(goal_x) < TERRAIN_HALF_X - 1 and abs(goal_y) < TERRAIN_HALF_Y - 1:
                self._goal = np.array([goal_x, goal_y], dtype=np.float32)
                break

        mujoco.mj_forward(self.model, self.data)    # qpos 변경 후 물리 상태 업데이트

        for _ in range(100):                        # 제어 입력 없이 100스텝 진행하여 rover가 지면에 안착하도록 대기
            mujoco.mj_step(self.model, self.data)

        # 에피소드 상태 초기화 (step에서 사용)
        self._step_count = 0                        # 타임아웃 카운터
        self._prev_dist  = float(np.linalg.norm(self._goal - self.data.qpos[:2]))  # potential reward 기준 거리

        return self._get_obs(), {}

    def step(self, action):
        # 1. action[-1,1] → ctrl (조향 2 + 구동 4)
        speed = float(action[0]) * self.cfg.max_speed
        steer = float(action[1]) * self.cfg.max_steer
        self.data.ctrl[0]   = steer   # act_steer_fl
        self.data.ctrl[1]   = steer   # act_steer_fr
        self.data.ctrl[2:6] = speed   # 구동 4륜

        # 2. frame_skip만큼 물리 진행
        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        # 3. obs
        obs = self._get_obs()

        # 4. 종료 판정
        cur_dist    = float(np.linalg.norm(self._goal - self.data.qpos[:2]))
        roll, pitch = obs[GRID_ROWS * GRID_COLS], obs[GRID_ROWS * GRID_COLS + 1]  # imu 앞 2개
        reached = cur_dist < self.cfg.goal_radius
        flipped = (abs(roll)  > np.deg2rad(self.cfg.flip_threshold_deg) or
                   abs(pitch) > np.deg2rad(self.cfg.flip_threshold_deg))
        terminated = reached or flipped
        truncated  = self._step_count >= self.cfg.max_steps

        # 5. reward (항목별 가중합)
        reward = self._compute_reward(cur_dist, reached, flipped)
        self._prev_dist = cur_dist   # progress 계산 후 갱신

        return obs, reward, terminated, truncated, {}

    def _compute_reward(self, cur_dist, reached, flipped):
        r  = self.cfg.w_progress * (self._prev_dist - cur_dist)  # 목표 접근 (직전 대비 거리 감소)
        r -= self.cfg.w_time                                      # 시간 페널티 (매 스텝)
        if reached: r += self.cfg.w_goal                          # 도달 보너스
        if flipped: r -= self.cfg.w_flip                          # 전복 페널티
        return float(r)

    def _get_obs(self):
        height_grid = self._get_height_grid()                          # 49
        imu         = self._get_imu()                                  # 8
        wheel_vel   = self.data.sensordata[6:10].astype(np.float32)   # 4
        goal_rel    = self._get_goal_rel()                             # 2
        return np.concatenate([height_grid, imu, wheel_vel, goal_rel])

    def _get_height_grid(self):
        # TODO: 전방/후방 비율 조정 검토 (현재 대칭: 전방 3칸, 후방 3칸)
        rx, ry, rz = self.data.qpos[:3]
        yaw = self._get_yaw()
        half_r, half_c = GRID_ROWS // 2, GRID_COLS // 2

        # 7×7 로컬 오프셋 그리드 생성
        ii, jj = np.meshgrid(np.arange(GRID_ROWS), np.arange(GRID_COLS), indexing='ij')
        dx_local = (ii - half_r) * CELL_SIZE   # 전방(+) / 후방(-)
        dy_local = (jj - half_c) * CELL_SIZE   # 좌(+) / 우(-)

        # 로컬 오프셋 → world 좌표 (local → world 회전)
        R = np.array([[ np.cos(yaw), -np.sin(yaw)],
                      [ np.sin(yaw),  np.cos(yaw)]])
        local_offsets = np.stack([dx_local.flatten(), dy_local.flatten()])  # (2, 49)
        world_offsets = R @ local_offsets                                    # (2, 49)
        wx = rx + world_offsets[0].reshape(GRID_ROWS, GRID_COLS)
        wy = ry + world_offsets[1].reshape(GRID_ROWS, GRID_COLS)

        # world 좌표 → heightmap 실수 인덱스 (bilinear interpolation)
        cols = (wx + TERRAIN_HALF_X) / (2 * TERRAIN_HALF_X) * NCOL
        rows = (wy + TERRAIN_HALF_Y) / (2 * TERRAIN_HALF_Y) * NROW
        heights = map_coordinates(self._heightmap, [rows.flatten(), cols.flatten()], order=1, mode='nearest')
        grid = heights.reshape(GRID_ROWS, GRID_COLS) * TERRAIN_MAX_H - rz

        return grid.flatten().astype(np.float32)

    def _get_imu(self):
        # quaternion → roll, pitch
        qw, qx, qy, qz = self.data.qpos[3:7]
        roll  = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1.0, 1.0))

        gyro = self.data.sensordata[0:3].astype(np.float32)   # 각속도 xyz
        vel  = self.data.sensordata[3:6].astype(np.float32)   # 선속도 xyz

        return np.array([roll, pitch, vel[0], vel[1], vel[2], gyro[0], gyro[1], gyro[2]], dtype=np.float32)

    def _get_goal_rel(self):
        yaw = self._get_yaw()
        # yaw만큼 역회전해서 world 좌표 → rover 로컬 좌표로 변환
        R   = np.array([[np.cos(yaw),  np.sin(yaw)],
                        [-np.sin(yaw), np.cos(yaw)]])
        rel = R @ (self._goal - self.data.qpos[:2])
        return rel.astype(np.float32)

    def _get_yaw(self):
        qw, qx, qy, qz = self.data.qpos[3:7]
        return float(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2)))

    def _generate_heightmap(self, spawn_row=None, spawn_col=None, flat_radius=12):
        def octave(sigma):
            return gaussian_filter(self.np_random.uniform(0, 1, (NROW, NCOL)), sigma=sigma)

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

    def _terrain_height_at(self, x, y):
        """world (x, y) 좌표의 실제 지형 높이 반환"""
        col = int((x + TERRAIN_HALF_X) / (2 * TERRAIN_HALF_X) * NCOL)
        row = int((y + TERRAIN_HALF_Y) / (2 * TERRAIN_HALF_Y) * NROW)
        col = np.clip(col, 0, NCOL - 1)
        row = np.clip(row, 0, NROW - 1)
        return float(self._heightmap[row, col]) * TERRAIN_MAX_H