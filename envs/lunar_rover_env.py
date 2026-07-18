import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from scipy.ndimage import gaussian_filter, map_coordinates
from envs.config import EnvConfig

SCAN_ROWS = 7
SCAN_COLS = 7
SCAN_CELL = 0.5   # m/cell
assert SCAN_ROWS % 2 == 1 and SCAN_COLS % 2 == 1, "SCAN_ROWS, SCAN_COLS must be odd"

NROW, NCOL     = 64, 64
TERRAIN_HALF_X = 10.0   # rover.xml hfield size[0]
TERRAIN_HALF_Y = 10.0   # rover.xml hfield size[1]
TERRAIN_MAX_H  = 2.0    # rover.xml hfield size[2]

class LunarRoverEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, cfg: EnvConfig = None, render_mode=None):
        super().__init__()
        # cfg 미지정 시 기본값. None 기본 인자로 mutable default 공유 문제 회피
        self.cfg = cfg if cfg is not None else EnvConfig()
        self.render_mode = render_mode
        self._renderer   = None                       # 첫 render() 호출 시 생성 (지연 초기화)

        self.model = mujoco.MjModel.from_xml_path("envs/assets/rover.xml")
        self.data  = mujoco.MjData(self.model)

        # obs: height_scan(49) + imu(8) + wheel_vel(4) + steer(2) + goal_rel(2) = 65
        obs_dim = SCAN_ROWS * SCAN_COLS + 8 + 4 + 2 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # action: [전진속도, 조향] 각각 -1 ~ 1
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # obs height scan의 로컬 오프셋(중심 기준, 고정) — 매 스텝 재생성하지 않도록 미리 계산
        hr, hc = SCAN_ROWS // 2, SCAN_COLS // 2
        ii, jj = np.meshgrid(np.arange(SCAN_ROWS), np.arange(SCAN_COLS), indexing='ij')
        dx_local = (ii - hr) * SCAN_CELL   # 전방(+) / 후방(-)
        dy_local = (jj - hc) * SCAN_CELL   # 좌(+) / 우(-)
        self._scan_offsets = np.stack([dx_local.flatten(), dy_local.flatten()])  # (2, 49)

        self._chassis_gid = self.model.geom('chassis_geom').id   # 충돌 판정용 (정상 주행은 바퀴만 닿음)

        # 조향각 qpos 인덱스 (obs용). 이름으로 조회해 xml 변경에 안전하게
        self._steer_qadr = [self.model.joint('steer_fl_joint').qposadr[0],
                            self.model.joint('steer_fr_joint').qposadr[0]]

        self._wheel_radius = float(self.model.geom('wheel_fl_geom').size[0])   # slip 계산용 (cylinder 반지름)

        # 조향 물리 한계를 cfg.max_steer에 맞춘다. xml 기본은 ±0.5라, cfg만 키우면 여기서 포화돼
        # 무효가 됨 → joint range·actuator ctrlrange를 함께 ±max_steer로 덮어써야 실제로 반영된다.
        for jn in ("steer_fl_joint", "steer_fr_joint"):
            self.model.joint(jn).range[:] = [-self.cfg.max_steer, self.cfg.max_steer]
        for an in ("act_steer_fl", "act_steer_fr"):
            self.model.actuator(an).ctrlrange[:] = [-self.cfg.max_steer, self.cfg.max_steer]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # rover 위치·속도 등 모든 시뮬레이션 상태를 초기값으로 리셋
        mujoco.mj_resetData(self.model, self.data)

        self._heightmap = self._generate_heightmap()
        self.model.hfield_data[:] = self._heightmap.flatten()     # 새로 생성한 지형을 MuJoCo 모델에 반영

        # 달 레골리스 마찰 랜덤화 (에피소드마다 전체 지형에 단일값 적용, 범위는 EnvConfig)
        # sliding 낮음 → 큰 토크 시 바퀴가 지면을 못 잡고 헛돎 (모래 위 wheel spin)
        # torsional 높음 → 회전 시 바퀴가 모래를 옆으로 밀어내는 저항 발생
        # rolling 높음 → 전진할수록 모래가 쌓이며 저항 증가하는 느낌 (sinkage 근사)
        # 세 값의 조합이 "단단한 바닥과 달리 모래에서 힘을 써도 잘 안 나가는" 거동을 만들어냄
        sliding   = self.np_random.uniform(*self.cfg.friction_sliding)
        torsional = self.np_random.uniform(*self.cfg.friction_torsional)
        rolling   = self.np_random.uniform(*self.cfg.friction_rolling)
        self.model.geom('terrain').friction[:] = [sliding, torsional, rolling]

        # spawn 위치 설정, 지형 위에 rover 배치
        spawn_x, spawn_y = 0.0, 0.0                 # TODO: 랜덤 혹은 지정할 수 있게 수정
        ground_z = self._terrain_height_at(spawn_x, spawn_y)
        self.data.qpos[0] = spawn_x
        self.data.qpos[1] = spawn_y
        self.data.qpos[2] = ground_z + 0.25   # + 바퀴 반지름(0.15) + 바퀴-chassis 오프셋(0.1)

        # 목표 지점: spawn 기준 반경 [goal_dist_min, goal_dist_max] 랜덤, 지형 경계(절대 좌표) 안쪽으로 제한
        while True:
            angle,r         = self.np_random.uniform(0, 2 * np.pi), self.np_random.uniform(self.cfg.goal_dist_min, self.cfg.goal_dist_max)
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
        #    에너지는 서브스텝마다 적산 (루프 후 값만 읽으면 그 사이 변동을 놓침).
        #    회전 액추에이터라 일률 = τ·ω = force × velocity (N·m × rad/s = W), × timestep → J
        energy = 0.0
        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)
            power   = float(np.sum(np.abs(self.data.actuator_force * self.data.actuator_velocity)))
            energy += power * self.model.opt.timestep
        self._step_count += 1

        # 3. obs
        obs = self._get_obs()

        # 4. 종료 판정
        cur_dist    = float(np.linalg.norm(self._goal - self.data.qpos[:2]))
        roll, pitch = obs[SCAN_ROWS * SCAN_COLS], obs[SCAN_ROWS * SCAN_COLS + 1]  # imu 앞 2개
        # roll/pitch가 numpy 스칼라라 비교 결과도 np.bool_ → Gymnasium 규약대로 Python bool로 변환
        reached = bool(cur_dist < self.cfg.goal_radius)
        flipped = bool(abs(roll)  > np.deg2rad(self.cfg.flip_threshold_deg) or
                       abs(pitch) > np.deg2rad(self.cfg.flip_threshold_deg))
        terminated = reached or flipped
        truncated  = bool(self._step_count >= self.cfg.max_steps)

        # 5. reward (항목별 가중합)
        collided = any(c.geom1 == self._chassis_gid or c.geom2 == self._chassis_gid
                       for c in self.data.contact[:self.data.ncon])
        reward, reward_info = self._compute_reward(cur_dist, reached, flipped, energy, collided)
        self._prev_dist = cur_dist   # progress 계산 후 갱신

        # 원시 측정값도 함께 (가중치가 0이어도 크기·발생 빈도를 볼 수 있게)
        # reached/flipped는 학습 로그의 성공률·전복률 집계에 쓴다 (종료 원인 구분)
        info = {**reward_info, "energy": energy, "collided": collided,
                "reached": reached, "flipped": flipped,
                "slip": self._get_slip(), "speed": float(self.data.sensordata[3])}

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, cur_dist, reached, flipped, energy, collided):
        """reward(가중합)와 항목별 기여도(r_*)를 반환. 기여도의 합이 reward."""
        cfg = self.cfg
        reward_info = {
            "r_progress":  cfg.w_progress * (self._prev_dist - cur_dist),  # 목표 접근 (직전 대비 거리 감소)
            "r_time":      -cfg.w_time,                                    # 시간 페널티 (매 스텝)
            "r_goal":      cfg.w_goal if reached else 0.0,                 # 도달 보너스
            "r_flip":      -cfg.w_flip if flipped else 0.0,                # 전복 페널티
            # 고도화 (기본 가중치 0 → 비활성, 학습 중 단계적으로 켠다)
            "r_energy":    -cfg.w_energy * energy,                         # 에너지 페널티
            "r_collision": -cfg.w_collision if collided else 0.0,          # 충돌 페널티
        }
        reward = float(sum(reward_info.values()))
        return reward, reward_info

    def render(self):
        """rgb_array 모드: mujoco.Renderer로 track 카메라 시점을 이미지로 반환."""
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="track")
        self._draw_markers(self._renderer.scene, self._get_height_scan())
        return self._renderer.render()

    def _draw_markers(self, scene, height_scan):
        """목표(빨간 구)와 obs 관측 지점(파란 점 49개)을 주어진 MjvScene에 이어 붙인다.

        scene.ngeom부터 append하므로 뷰어의 user_scn·render의 renderer.scene 양쪽에서 공용.
        height_scan: obs 앞 49개(rover 기준 상대 높이). 각 점 z = 상대높이 + rover z로 복원해
        지형 재샘플링 없이 그린다 (뷰어는 obs에서, render는 _get_height_scan()로 넘김).
        """
        def add(pos, size, rgba):
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([size, 0.0, 0.0]),
                pos=np.asarray(pos, dtype=float),
                mat=np.eye(3).flatten(),
                rgba=np.asarray(rgba, dtype=float),
            )
            scene.ngeom += 1

        # 목표: 빨간 구
        gx, gy = self._goal
        add([gx, gy, self._terrain_height_at(gx, gy) + 0.3], 0.3, [1.0, 0.2, 0.2, 0.6])

        # obs 관측 지점: (x,y)는 _scan_world_xy, z는 상대높이 + rover z
        wx, wy = self._scan_world_xy()
        wz = height_scan + self.data.qpos[2]
        for x, y, z in zip(wx, wy, wz):
            add([x, y, z], 0.04, [0.2, 0.5, 1.0, 0.9])

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _get_obs(self):
        height_scan = self._get_height_scan()                          # 49
        imu         = self._get_imu()                                  # 8
        wheel_vel   = self.data.sensordata[6:10].astype(np.float32)   # 4
        steer       = self._get_steer()                                # 2
        goal_rel    = self._get_goal_rel()                             # 2  (뷰어가 obs[-2:]로 읽으므로 맨 뒤 유지)
        return np.concatenate([height_scan, imu, wheel_vel, steer, goal_rel])

    def _get_slip(self):
        """접지면 미끄러짐 속도 (m/s) — 바퀴가 구르는 속도와 실제 전진 속도의 차이. 모니터링 전용.

        > 0 : 바퀴가 헛돎 (모래에서 힘만 쓰고 안 나감)   < 0 : 몸이 바퀴보다 빠름 (미끄러져 내려감/제동)
        비율이 아닌 속도차라 정지 상태에서도 특이점이 없다 (0 − 0 = 0).
        reward에는 넣지 않는다 — 회복을 위한 역주행까지 벌하게 되기 때문.
        """
        v_wheel = float(np.mean(self.data.sensordata[6:10])) * self._wheel_radius  # 바퀴 평균 각속도 → 선속도
        v_body  = float(self.data.sensordata[3])                                    # 몸체 전진속도 (body frame vx)
        return v_wheel - v_body

    def _get_steer(self):
        """앞바퀴 조향각 (rad). position 액추에이터라 명령을 5~8스텝에 걸쳐 따라가므로,
        현재 각도를 관측하지 않으면 정책이 바퀴가 어디를 향하는지 모른 채 명령하게 된다.
        """
        return self.data.qpos[self._steer_qadr].astype(np.float32)

    def _get_height_scan(self):
        rz     = self.data.qpos[2]
        wx, wy = self._scan_world_xy()   # 관측 지점 world 좌표 (49,)

        # world 좌표 → heightmap 실수 인덱스 (bilinear interpolation)
        # 꼭짓점 NCOL개가 폭 전체에 NCOL-1칸 간격으로 놓이므로 스케일은 (NCOL-1) (MuJoCo hfield 정렬)
        cols = (wx + TERRAIN_HALF_X) / (2 * TERRAIN_HALF_X) * (NCOL - 1)
        rows = (wy + TERRAIN_HALF_Y) / (2 * TERRAIN_HALF_Y) * (NROW - 1)
        heights = map_coordinates(self._heightmap, [rows, cols], order=1, mode='nearest')
        scan = heights * TERRAIN_MAX_H - rz   # rover 기준 상대 높이

        return scan.astype(np.float32)

    def _scan_world_xy(self):
        """obs height scan 각 셀의 world (x, y) 좌표를 각각 (49,)로 반환.

        _get_height_scan(관측)와 뷰어(관측 지점 시각화)가 공유한다. 각 점의 z(높이)는
        obs의 height_scan에 rover z를 더하면 복원되므로 여기선 x, y만 계산한다.
        회전만 하는 값싼 연산이라, 학습에는 얹지 않고 필요한 쪽이 그때그때 호출한다.
        """
        # TODO: 전방/후방 비율 조정 검토 (현재 대칭: 전방 3칸, 후방 3칸)
        rx, ry = self.data.qpos[:2]
        yaw = self._get_yaw()

        # 로컬 오프셋(고정) → world 좌표 (local → world 회전)
        R = np.array([[ np.cos(yaw), -np.sin(yaw)],
                      [ np.sin(yaw),  np.cos(yaw)]])
        world = R @ self._scan_offsets          # (2, 49)
        return rx + world[0], ry + world[1]

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
        """world (x, y) 좌표의 지형 높이 반환 (obs와 동일한 bilinear 보간)"""
        col = (x + TERRAIN_HALF_X) / (2 * TERRAIN_HALF_X) * (NCOL - 1)
        row = (y + TERRAIN_HALF_Y) / (2 * TERRAIN_HALF_Y) * (NROW - 1)
        h = map_coordinates(self._heightmap, [[row], [col]], order=1, mode='nearest')
        return float(h[0]) * TERRAIN_MAX_H