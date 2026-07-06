from dataclasses import dataclass


@dataclass
class EnvConfig:
    """LunarRoverEnv의 에피소드·제어·reward 하이퍼파라미터.

    실험마다 바꿔 튜닝하는 값들. env·train·eval이 이 config를 공유한다.
    예: EnvConfig(w_goal=20, max_steps=800)
    """
    # 제어 / 에피소드
    max_speed:  float = 5.0    # rad/s — 바퀴 구동 최대 (action=±1 → ±max_speed), manual_drive와 동일
    max_steer:  float = 0.5    # rad   — 조향 최대 (action=±1 → ±max_steer)
    frame_skip: int   = 10     # policy 1스텝당 물리 스텝 수 (0.005×10=0.05s → 20Hz)
    max_steps:  int   = 500    # 타임아웃 (=25초)

    # 종료 판정
    goal_radius:        float = 0.5   # m   — 목표 도달 판정 반경
    flip_threshold_deg: float = 60.0  # deg — |roll| 또는 |pitch| 이 이상이면 전복

    # reward 가중치 (1단계: 목표접근·도달·전복·시간)
    w_progress: float = 1.0    # 목표 접근 (직전 대비 거리 감소량)
    w_goal:     float = 10.0   # 도달 보너스
    w_flip:     float = 10.0   # 전복 페널티 (reward에서 음수로 적용)
    w_time:     float = 0.01   # 시간 페널티 (매 스텝 −w_time)
