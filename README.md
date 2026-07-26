# 달 레골리스 지형 자율주행 강화학습

MuJoCo + Gymnasium + Stable-Baselines3(PPO)로, **Ackermann 조향 4륜 로버**가 미끄러운 달 레골리스 지형에서 목표 지점까지 자율주행하도록 학습한다.
핵심은 성능 수치보다 **비자명한 RL 실패(옆 방향 주행)를 통제 실험으로 진단하고, 탐험 기법(gSDE)으로 정조준 개선한 과정**이다.

## 환경

- **로버** — 4륜구동 + 앞바퀴 Ackermann 조향 (제자리 회전 불가, 최소 회전반경 R≈1.28m — 자전거 모델 근사)
- **관측 65D** — 지형 높이 스캔 49 · IMU 8(자세·속도·각속도) · 바퀴속도 4 · 조향각 2 · 목표 상대위치 2
- **행동 2D** — 스로틀, 조향 · **reward** — 목표 접근·도달·시간·전복 (에너지·충돌은 확장용, 기본 가중치 0)
- 레골리스 마찰·지형을 에피소드마다 랜덤화

## 핵심 서사 — 옆 방향 실패의 진단과 개선

1. **증상** — PPO가 정면·후방은 도달하나 **옆 방향 목표에서 실패(≈4%)**.
2. **진단** — 하이퍼파라미터(exp01)·물리 마찰(exp03)·seed(exp02) 가설을 **통제 실험으로 소거**. 원인은 **탐험 붕괴**: i.i.d. 가우시안 노이즈가 관성·조향지연 계의 heading에서 **상쇄**되어 "한 방향 조향 지속"이 탐험되지 않음.
3. **처방** — 시간상관·상태의존 탐험 **gSDE**(SB3 native) 도입.
4. **결과** — 옆 도달률 **4% → 34~48%**(거리별), **5 seed 멀티시드로 재현 확증**(baseline 통제 비교). 다만 휴리스틱 expert는 **넘지 못함** — 순수 RL의 탐험 병목만 해소.

![각도별 도달률](analysis/exp04_gsde/angle_success.png)

> baseline은 정면·후방 축에서만 성공하는 "W자"(옆에서 0으로 추락). gSDE가 그 옆을 메운다. (heuristic = 무학습 expert 상한)

## 실험·분석 문서

| 단계 | 문서 |
|---|---|
| 조향 병목 진단 | [steering_bottleneck](docs/2026-07-18_analysis__steering_bottleneck.md) · [exp00 flat_tuning](docs/2026-07-16_experiment__exp00_flat_tuning.md) |
| 하이퍼파라미터 소거 | [exp01 easy_goal](docs/2026-07-20_experiment__exp01_easy_goal.md) |
| 물리(마찰) 소거 | [exp03 low_slip](docs/2026-07-20_experiment__exp03_low_slip.md) |
| 학습 궤적·탐험 분석 | [training_trajectory](docs/2026-07-20_analysis__training_trajectory.md) |
| **탐험 개선(gSDE)** | [**exploration_improvement**](docs/2026-07-21_experiment__exploration_improvement.md) |
| 모방학습(BC) | [bc_imitation](docs/2026-07-20_experiment__bc_imitation.md) — 옆 도달률 6% → 32~80% 회복 |

## 실행

```bash
python train.py      --config configs/exp04-explore/exp04-00-gsde.yaml     # 학습
python policy_eval.py baseline=<ckpt>,... gSDE=<ckpt>,... --dists 2 3 4 5   # 거리×각도 격자 평가
python sim_viewer.py  --policy <ckpt>                                       # 시각화
```

분석 그림·재현 스크립트는 [analysis/exp04_gsde/](analysis/exp04_gsde/) (`run_eval.py`로 평가 재현).

## 한계 · 다음

- **expert 미달** — gSDE는 baseline은 크게 앞서나 휴리스틱 expert(전체 66%)엔 못 미침(~40%).
- **직진 축 소폭 후퇴** — 0°·180°에서 baseline보다 낮음(호 편향 정책으로 수렴한 부작용).
- **미평가** — 근거리 정옆(회전원 안쪽 <2.56m).
- **다음** — 직진 축을 학습시키기 위한 **보상 조정**(에너지 페널티 `w_energy` 활성 등으로 불필요한 조향 억제), 각도 커리큘럼, offline RL(expert 초월 방향).
