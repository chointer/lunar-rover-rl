# Lunar Rover RL Project — Context Document

### 프로젝트 한 줄 요약
> 통신이 차단된 달 용암 튜브 환경에서, rover가 자율적으로 미탐사 지역을 최대한 넓게 탐사하고 시작점으로 귀환하는 Hierarchical RL 시스템 — MuJoCo + PPO 기반

---

### 배경 및 Motivation

- 달 용암 튜브는 통신이 차단되어 지구에서 실시간 조종 불가
- 지구-달 통신 딜레이 약 1.3초 (왕복 2.6초) → 온보드 자율 판단 필요
- Spirit rover 사례: 레골리스에 바퀴가 빠져 미션 실패 → 사전 대응 불가능했던 실제 사례
- 달 레골리스는 공극률 83%, 미세하고 날카로운 입자 → slip/sinkage 발생
- 달 표면 RMS 경사각 16~25도 + 저중력(1/6) → rule-based 제어 한계
- RL이 필요한 이유: 매번 다른 지형, 불확실한 마찰, 실시간 적응 필요

---

### 타겟 기업
- **UEL (무인탐사연구소)** — 서울 성수동, 달/화성 탐사 로버 전주기 개발 스타트업
- ISRU(현지자원이용기술) 고도화 목표
- 이 프로젝트의 narrative가 UEL 미션과 직접 연결됨

---

### 환경 설정 (MuJoCo)

| 항목 | 설정 |
|------|------|
| 시뮬레이터 | MuJoCo + Gymnasium |
| 지형 | Height field 기반 불규칙 지형 |
| 경사 | RMS 15~25도 참고, 매 에피소드 randomization |
| 중력 | 1.62 m/s² (지구의 1/6) |
| 마찰 계수 | 낮게 설정 + 구역별 랜덤화 |
| Rover 크기 | 1m, 4륜 |
| 동굴 경계 | 오르지 못할 급경사로 표현 |
| 셀 크기 | 0.5m (rover 크기의 절반) |

---

### 전체 시스템 구조 (Hierarchical)

```
[High-level Policy]
입력: 전체 탐사 map + rover 현재 위치
출력: 다음 목표 지점 좌표
        ↓
[Low-level Policy]
입력: 로컬 height scan + 내부 센서 + 목표 지점
출력: 주행 action (전진 속도 + 조향각)
        ↓
[귀환 — Rule-based]
에너지 threshold 도달 시 시작점으로 귀환
```

---

### Low-level Policy

#### 목표
주어진 목표 지점까지 달 표면 환경에서 안전하게 도달

#### Policy가 풀어야 할 문제들

1. **Slip 회피** — 레골리스 저마찰 구역에서 바퀴 과회전 시 전진 불가. 속도 조절로 slip 최소화
2. **급경사 대응** — RMS 16~25도 경사 + 저중력에서 접지력 상실 방지. 경사 전 속도 줄이기
3. **불균일 마찰 적응** — 위치마다 마찰 계수 달라 갑자기 미끄러지는 상황. 바퀴 피드백으로 실시간 적응
4. **목표 도달 vs 안전 tradeoff** — 위험 구역 근처 목표일 때 우회 경로 선택
5. **실패 인식** — stuck/타임아웃 감지 후 에피소드 종료, high-level에 실패 신호 전달

#### Observation Space (총 63개)
- 로컬 height scan 7×7 (전방 3.5m, 좌우 ±1.5m) — bilinear interpolation 적용
- IMU 8개: roll, pitch, 선속도(xyz), 각속도(xyz)
- 바퀴 회전속도 4개 (slip ratio 계산용)
- 목표 지점 로컬 상대 좌표 (rel_x, rel_y)
- visible_mask: 초기 구현 제외, Week 3 이후 추가 검토


#### Action Space
- 전진 속도
- 조향각

#### Reward
| 항목 | 방향 |
|------|------|
| 목표 방향 전진 거리 | positive (메인, 설계 검토 중) |
| 목표 반경 도달 | large positive + 에피소드 종료 |
| 충돌 (chassis contact) | negative |
| Slip ratio 높을 때 | negative |
| Stuck / 타임아웃 | large negative + 에피소드 종료 |

> 충돌 패널티: 바퀴 geom 제외, chassis body contact 감지.

#### 실패 조건
- **타임아웃**: 정해진 스텝 초과
- **Stuck**: 일정 시간 동안 이동거리 threshold 이하

---

### High-level Policy

#### 목표
미탐사 지역 최대화 + 효율적 탐사 순서 결정

#### Policy가 풀어야 할 문제들

1. **효율적 탐사 순서** — 미탐사 구역 여러 곳 중 어떤 순서로 가야 최대 면적 탐사 가능한지
2. **접근 불가 구역 학습** — Low-level 실패 누적 구역 피하기
3. **귀환 타이밍** — 에너지 threshold rule-based로 처리
4. **미탐사 면적 vs 접근성 tradeoff** — 넓은 미탐사 구역이어도 접근 경로 위험하면 포기

#### Observation Space
- 전체 탐사 map (탐사됨 / 미탐사 / 접근 불가)
- Rover 현재 위치

#### 출력
- 다음 목표 지점 좌표

#### Map 상태
- ⬜ 미탐사
- ✅ 탐사 성공
- ❌ 접근 불가 (Low-level 실패 누적)

---

### 안전 위협 요소 (sinkage 없이)

| 위협 | 원인 |
|------|------|
| 급경사 미끄러짐 | 저중력 + 저마찰 조합 |
| 뒤집힘 | 급경사 + 저중력 |
| 저마찰 구역 slip | 레골리스 마찰 불균일 |
| 저중력 접지력 상실 | 바퀴가 지면에서 뜨는 현상 |
| 막다른 길 | 들어갔다가 나오지 못하는 지형 |
| 과도한 탐사 | 귀환 불가 거리까지 진출 |

> Sinkage는 MuJoCo height field로 정확한 구현 어려워 Limitation으로 명시. 저마찰 + slip으로 근사.

---

### PID Baseline 비교 계획

- **평탄한 구간**: PID도 잘 됨 → 기준점
- **급경사 + 저마찰**: RL이 더 잘 됨
- **복잡한 지형 (막다른 길 등)**: RL만 가능

→ 실험으로 RL 필요성 증명

---

### Low-level 개발 플랜 (4주)

#### Week 1 — 환경 세팅
- WSL + Python 환경 (conda 또는 venv)
- MuJoCo + Gymnasium 설치
- 4륜 rover URDF 로드
- Height field 지형 생성
- 중력 1/6, 마찰 계수 설정
- Terrain randomization 기본 구현

#### Week 2 — 기본 RL 훈련
- Observation / Action space 구현
- Reward 구현
- PPO 기본 훈련
- 기본 주행 확인 + 버그 수정

#### Week 3 — Reward 튜닝 + 고도화
- Reward shaping 튜닝
- 급경사 / slip 회피 확인
- PID baseline 구현 + 비교 실험
- 실패 케이스 분석

#### Week 4 — 안정화 + 포트폴리오 정리
- Domain randomization 강화
- 시각화 (주행 영상, 학습 곡선)
- GitHub README 작성
- RL vs PID 비교 그래프

---

### 포트폴리오 산출물

- 주행 영상 (다양한 terrain)
- RL vs PID 비교 실험 결과
- GitHub README + 코드
- 학습 곡선 그래프

---

### Future Work (README에 명시)

- High-level exploration policy 연결
- Observation delay 추가 (통신 딜레이 근사)
- Genesis MPM으로 실제 레골리스 sinkage 구현
- Isaac Lab으로 포팅

---

### 참고 논문

1. **달 표면 slip reward 반영 PPO path planning** (ResearchGate, 2026) — slip behavior를 reward에 반영한 lunar rover DRL 연구, PPO 기반, Gazebo 환경
2. **Constrained RL for Lunar Legged Manipulator** (arXiv:2510.12684, 2025) — 달 환경 CRL 프레임워크, 저중력 설정, domain randomization, 40ms 제어 딜레이 구현 참고
3. **4-Wheeled Lunar Rover Failure-Safe Motion Planning** (MDPI, 2023) — 바퀴 고장 시 slip 대응 RL

---

### 개발 환경

- OS: Windows + WSL (Ubuntu)
- Language: Python
- Simulator: MuJoCo + Gymnasium
- RL: PPO (Stable-Baselines3 또는 직접 구현)
- GPU: 있음 (훈련 시 사용)
