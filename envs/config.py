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

    # 마찰 도메인 랜덤화 범위 (reset마다 지형 전체에 단일값을 min~max에서 uniform 추출)
    #   sliding   낮을수록 잘 미끄러짐 (레골리스 저마찰) — 참고: 콘크리트~0.8, 젖은 흙~0.5, 모래~0.3, 느슨한 레골리스~0.2
    #   torsional 높을수록 제자리 회전 저항 (모래)      — 참고: 딱딱한 지면~0.005, 모래~0.05
    #   rolling   높을수록 구름 저항 (sinkage 근사)     — 참고: 딱딱한 지면~0.01, 모래~0.08~0.15
    friction_sliding:   tuple = (0.2, 0.5)
    friction_torsional: tuple = (0.02, 0.08)
    friction_rolling:   tuple = (0.05, 0.12)
