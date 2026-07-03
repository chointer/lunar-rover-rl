# Low-level Observation Space 설계 결정

### 최종 Obs Space

```
obs = [
    height_grid (7×7 = 49),   # 로컬 1인칭, 상대 높이, 플랫 벡터
    imu (8),                   # roll, pitch, 선속도(x,y,z), 각속도(xyz)
    wheel_vel (4),             # 바퀴별 회전속도, slip ratio 계산용
    goal_rel_x, goal_rel_y     # 로버 기준 상대 좌표
]
# 총 63개
# visible_mask: 초기 구현에서 제외. Week 3 이후 추가 검토
```

---

### 설계 결정 및 근거

#### 1인칭 로컬 좌표계

rover heading 기준으로 그리드를 회전 샘플링한다. 월드 좌표계로 샘플링하면 rover가 어느 방향을 보느냐에 따라 동일한 지형이 그리드에서 다른 위치에 나타난다. 로컬 좌표계에서는 rover가 어느 방향을 보든 전방이 항상 그리드 앞쪽이다.

goal 좌표도 동일한 이유로 월드 좌표가 아닌 로버 기준 상대 좌표로 변환한다. 스폰 위치가 달라져도 "goal이 앞 3m"는 항상 동일한 의미다.

yaw 변환이 두 군데 필요하다:
- height grid 샘플링 시
- goal 월드 좌표 → 로컬 상대 좌표 변환 시

#### 상대 높이 인코딩

각 셀의 절대 높이 대신 rover 현재 z 기준 상대 높이로 정규화한다. 맵의 어느 위치에 있든 "이 앞이 오르막인가 내리막인가"만 표현되어 일반화가 유리하다.

#### 플랫 벡터

7×7 = 49개 그리드를 row-major로 펼쳐서 1D 벡터로 입력한다. PPO + MLP 기반에서 CNN 커스텀 없이 사용 가능하다. 소규모 그리드에서 MLP가 공간 패턴을 암묵적으로 학습할 수 있다. 이미지(수백×수백) 수준이 아니라면 CNN이 필수는 아니다.

#### 그리드 크기 (7×7, 셀 0.5m)

전방 3.5m, 좌우 ±1.5m 범위를 커버한다. 후방도 포함한다 (미끄러짐 후 후진 대비, 후진 허용 여부와 무관하게 포함).

미결: 7행 중 전방/후방 행 수 비율 결정 필요. 예: 전방 5행(2.5m) + 후방 2행(1m).

#### visible_mask 별도 채널

장애물이나 언덕 뒤 지형은 실전 센서로 볼 수 없다. GT에서는 그냥 읽어버리므로 sim-to-real gap이 발생한다.

이를 줄이기 위해 `mj_ray`를 rover 눈높이에서 각 셀 방향으로 발사해서, 장애물에 막히면 뒤 셀을 unknown으로 처리한다. unknown을 높이값으로 인코딩하면 실제 높이값과 구분이 불가능하므로 별도 0/1 마스크 채널로 분리한다.

미결: rover 모델에서 센서(카메라/LiDAR) 위치 정의 필요.

#### goal 상대 좌표 정규화

high-level이 월드 좌표로 넘겨주면 low-level env에서 로컬 변환한다. goal이 그리드 범위(3.5m) 밖에 있을 때 값이 커지므로 최대 탐사 반경 기준으로 -1~1 정규화 필요. 정규화 범위 결정 필요.

#### IMU 구성 (8개)

roll(1), pitch(1), 선속도(x,y,z)(3), 각속도(xyz)(3) = 8개. yaw는 로컬 좌표계 변환에 이미 사용되므로 obs에서 제외.

gyro_x(roll rate), gyro_y(pitch rate)를 포함한 이유: 경사면 진입 시 기울기 변화 속도로 뒤집힘 위험을 조기 감지 가능. obs 크기 증가(2개)가 미미해서 포함.

---

### 실전 배포 스택

```
Stereo/LiDAR → local point cloud
IMU → pitch/roll 보정
        ↓
rover 자세 보정된 local height scan
        ↓
SLAM (위치 추정) → world coord 변환
        ↓
elevation mapping (world map 누적 갱신)
        ↓
rover 주변 잘라서 policy 입력
```

- SLAM: "어디 있냐" (위치/자세 추정)
- elevation mapping: "주변이 어떻게 생겼냐" (지형 지도 누적)
- 둘 다 ROS 기반 오픈소스 존재 (RTAB-Map, ANYbotics elevation_mapping)
- GT 사용은 Limitation이 아닌 의도된 설계. 실전 연결은 Low-level 완성 후 별도 태스크

---

### Limitations

1. **GT 위치/지형 사용**: 실제 배포 시 SLAM 필요. localization 오차 미반영
2. **Sensor noise 미구현**: noise-free height scan으로 학습. Week 3~4 domain randomization에서 위치/yaw noise 추가 예정
3. **Height map 한계**: 오버행, 천장 돌출 구조물 표현 불가. 달 용암 튜브 실제 환경에서 천장 붕괴 잔해 등 3D 구조물 존재 가능
4. **Sinkage 미구현**: 저마찰 + slip으로 근사
5. **시야각 raycast 미구현 (초기)**: Week 2에서는 수직 raycast로 시작, Week 3에서 시야각 raycast로 전환 가능

---

### Future Work

- SLAM + elevation mapping 연결 (ROS 기반, Low-level 완성 후)
- High-level exploration policy (frontier-based 대비 RL 차별점: low-level 실패 누적 반영)
- Sparse voxel 기반 3D occupancy grid로 obs 교체 → 오버행 및 복잡한 3D 장애물 처리
- Carrier + Explorer 구조 검토 (베이스 유닛 + 탐사 로버 분리)
- Teacher-student: GT policy → 실제 센서 입력 policy 모방 학습
- Sensor noise / odometry drift domain randomization

---

### Week 3 실험 항목

#### 과거 프레임 스택 비교

현재 obs만으로는 slip 진행 추이, 가속도 변화 등 동적 정보가 없다. 과거 프레임 스택으로 이를 보완할 수 있는지 실험한다.

| 실험 | 구성 | obs 크기 |
|------|------|----------|
| A | 현재 obs만 | 63개 |
| B | 센서(IMU+바퀴속도)만 과거 3프레임 스택 | 63 - 12 + 36 = 87개 |
| C | 전체 obs 과거 3프레임 스택 | 189개 |

실험 순서: B로 학습 안정화 → A로 줄여서 성능 하락 확인 → C로 전체 스택 효과 확인

B vs C 비교: height grid 과거 스택이 공간 어긋남으로 노이즈가 되는지, 실제로 도움이 되는지 확인

측정 지표: 학습 속도, slip 회피 성능, goal 도달률

#### 거리 기반 해상도 차등

근거리 고해상도, 원거리 저해상도 방식. 실제 depth sensor 특성과 유사.

```
근거리 (0~1m):   0.25m 셀
중거리 (1~2.5m): 0.5m 셀
원거리 (2.5~3.5m): 1.0m 셀
```

균일 7×7 대비 성능 및 obs 크기 효율 비교.

---

### 참고 연구

- **CERBERUS (ETH Zurich)**: SubT 우승팀. ANYmal + elevation mapping + RL locomotion. 오버행 환경에서 3D voxel occupancy map 사용. elevation mapping 20Hz, Jetson AGX Xavier 온보드 실행
- **ANYbotics elevation_mapping**: ROS 기반 오픈소스. robot-centric, pose uncertainty 반영 설계
- **RL for Wheeled Mobility on Vertically Challenging Terrain (2024)**: 4륜 + PPO + cropped elevation map obs. action space 동일 (선속도 + 조향각). elevation map을 SWAE로 차원 축소
- **RLRoverLAB (2024)**: 행성 로버 RL. 지형 + 장애물 합친 height scan 방식 사용
