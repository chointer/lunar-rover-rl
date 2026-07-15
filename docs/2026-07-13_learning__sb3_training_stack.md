# SB3 학습 스택 정리

> `train.py`를 구성하는 요소들(VecEnv · Monitor · VecNormalize · PPO · log_std)의 개념 정리.
> 학습 결과가 이상할 때 어디를 의심할지 판단하는 근거가 된다.
> 관련: train.py, #18(학습 스크립트) · 작성: 2026-07-13

---

### 전체 래퍼 구조

```
LunarRoverEnv  →  Monitor  →  SubprocVecEnv  →  VecNormalize  →  PPO
   (원본 env)     에피소드     env 8개 묶음      obs·보상 정규화     학습
                   통계
```

**순서가 의미를 바꾼다.** 특히 Monitor는 반드시 VecNormalize **안쪽**이어야 한다 (아래 참조).

---

### VecEnv — env 여러 개를 하나처럼 다루기

```python
venv_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
venv = venv_cls([make_env(seed, i) for i in range(n_envs)])
```

| | DummyVecEnv | SubprocVecEnv |
|---|---|---|
| 실행 | 한 프로세스에서 **순차** 루프 | env마다 **별도 프로세스** (진짜 병렬) |
| 병렬성 | ❌ (GIL) | ✅ |
| 오버헤드 | 0 | obs/action을 매 스텝 직렬화·파이프 전송 |

**주의**: env가 싸면 IPC 오버헤드가 병렬 이득을 잡아먹어 **DummyVecEnv가 더 빠를 수 있다.**
우리 env는 스텝당 ~0.39ms(2,565 steps/s)로 애매한 편 → **추측 말고 실측할 것.**

#### 왜 env가 아니라 "함수"의 리스트를 넘기나

```python
def make_env(seed, rank):
    def _init():
        env = LunarRoverEnv()
        env.reset(seed=seed + rank)   # rank가 env마다 다른 seed → 서로 다른 지형·마찰·목표
        return Monitor(env)
    return _init                       # ← env가 아니라 "만드는 함수"를 반환 (팩토리)
```

`SubprocVecEnv`는 env를 **각 프로세스 안에서 직접 생성**해야 한다. MuJoCo env는 pickle해서
프로세스로 보낼 수 없기 때문. 그래서 "만드는 방법(함수)"을 주면 각 프로세스가 자기 안에서 호출한다.

결과물 `venv`는 env 8개를 하나처럼 다룬다: `venv.step(actions)` — actions `(8, 2)` → obs `(8, 65)`.

---

### Monitor — 에피소드 통계

에피소드가 끝나면 `info`에 요약을 끼워 넣는다:

```python
info["episode"] = {"r": 8.49, "l": 187, "t": 12.3}   # return, 길이, 경과시간
```

SB3 로거가 이걸 읽어 `rollout/ep_rew_mean`, `rollout/ep_len_mean`을 계산한다.
**없으면 `ep_rew_mean`이 로그에 안 나온다** — sanity 학습의 핵심 지표라 필수.

#### 왜 VecNormalize 안쪽이어야 하나

VecNormalize는 **보상을 나눈다**. Monitor가 바깥에 있으면 **정규화된 보상**을 기록하게 되어,
`ep_rew_mean`이 해석 불가능한 숫자가 된다. 안쪽에 두면 **원본 보상**을 기록해 우리가 아는
값(휴리스틱 +8.49 등)과 직접 비교할 수 있다.

> "학습은 되는 것 같은데 return 숫자가 이상하다"의 흔한 원인이 이 순서 문제다.

---

### VecNormalize ① — obs 정규화 (차원별!)

```python
VecNormalize(venv, norm_obs=True, ...)
```

**핵심: 통계는 obs 차원마다 따로 잡힌다.** `obs_rms`의 shape = `(65,)`.

```
mean = [m₀, m₁, ..., m₆₄]                    ← 65개
var  = [v₀, v₁, ..., v₆₄]                    ← 65개
정규화: (obs - mean) / sqrt(var + ε)          ← 원소별(element-wise)
```

**8개 env는 통계의 "표본"일 뿐이다.** 매 스텝 `(8, 65)` 배치가 들어오면, 8개 행을 샘플 8개로
써서 65개 차원 각각의 running 평균/분산을 갱신한다. env가 많으면 통계가 빨리 수렴할 뿐,
통계 자체는 여전히 차원별 65쌍.

#### 왜 차원별이어야 하는가 (우리 문제의 핵심)

| obs 차원 | 원래 std |
|---|---|
| height_scan (0~48) | 0.14 |
| **goal_rel (63~64)** | **3.22** ← 23배 |
| wheel_vel | 0.14 (평균 4.86 **오프셋**) |

스칼라 하나로 나누면 **23배 비율이 그대로 유지**된다. 그러면 첫 레이어에서 goal_rel이 만드는
활성값이 압도적이라 **역전파가 그쪽 가중치만 크게 움직이고, height_scan은 사실상 무시**된다.
→ **지형을 안 보고 목표로 돌진하는 정책**이 된다 (지형 인식이 이 프로젝트의 핵심인데!).

**차원별로 나눠야** 각 차원이 std≈1이 되어 비율 차이가 사라진다.

#### clip_obs=10.0

정규화하면 값이 대략 ±3 안에 든다(정규분포 99.7%). `±10`은 훨씬 넉넉해서 **평소엔 무영향**이고,
다음 사고만 막는 **안전망**이다:
- 학습 초기엔 running 통계가 부정확해 정규화 결과가 튈 수 있음
- 드물게 극단적 지형(급경사)에서 obs가 크게 벗어남
→ 이상치가 그대로 들어가면 **그래디언트 폭주**.

---

### VecNormalize ② — reward 정규화 (스칼라, 나눗셈만)

```python
VecNormalize(venv, norm_reward=True, ...)
```

```python
returns = returns * γ + reward           # 할인 누적 return을 계속 추적
ret_rms.update(returns)                  # 그 return의 running 분산 (스칼라 하나)
reward = reward / sqrt(ret_rms.var + ε)  # ← 스케일만! 평균은 안 뺌
reward = clip(reward, -10, +10)
```

| | 통계 형태 | 중심화(평균 빼기) |
|---|---|---|
| **obs** | `(65,)` 차원별 | ✅ 함 |
| **reward** | `()` 스칼라 | ❌ **안 함** |

#### 왜 보상은 평균을 빼면 안 되는가

보상에 상수 `c`를 더하거나 빼면 **에피소드 길이에 유불리가 생긴다**:
- `c > 0` → 오래 살수록 이득 → 로버가 목표 안 가고 뭉갬
- `c < 0` → 빨리 끝내는 게 이득 → **자살 exploit**

반면 **양수로 나누기만 하면** 모든 return이 같은 비율로 줄어 **최적 정책이 보존**된다.

#### 왜 필요한가 — critic 때문

critic은 **return을 예측하는 회귀**를 푼다. 우리 보상은 스케일이 1000배 차이난다:

| | 크기 |
|---|---|
| 스텝당 progress·time | ~0.001 ~ 0.1 |
| 도달 보너스 / 전복 | **±10** (스파이크) |
| 에피소드 return | 휴리스틱 평균 +8.5 |

critic이 `+8.5` 같은 큰 값을 출력하려면 마지막 레이어 가중치가 커지고, MSE 손실의 그래디언트가
튀며 **학습이 불안정**해진다. return의 std로 나누면 O(1)이 되어 다루기 좋아진다.

#### 주의

1. **클리핑(±10)과 충돌 가능** — 도달 보너스 +10이 정규화 후에도 크면 잘릴 수 있다. 로그 확인 필요.
2. **running std가 학습 중 변함** → 보상 스케일이 서서히 바뀌는 약한 비정상성.
3. **통계를 모델과 함께 저장할 것** (`models/*_vecnorm.pkl`). 없으면 평가 시 정책이 **학습 때와
   다른 스케일의 obs**를 보고 엉뚱하게 움직인다. **아주 흔한 버그.**

---

### PPO 생성자

```python
PPO("MlpPolicy", venv, seed=..., verbose=1, tensorboard_log="runs/")
```

| 인자 | 의미 |
|---|---|
| `"MlpPolicy"` | 다층 퍼셉트론 정책 (이미지면 `"CnnPolicy"`, dict obs면 `"MultiInputPolicy"`) |
| `seed` | 재현성 (torch·numpy·행동 샘플링) |
| `verbose=1` | 학습 진행 표 출력 (0=조용, 2=디버그) |
| `tensorboard_log` | TensorBoard 이벤트 파일 디렉터리 |

#### MlpPolicy의 실제 구조 (SB3 기본값)

```
obs(65) ─→ [64] ─→ [64] ─→ action mean(2)    ← actor (정책)
        └─→ [64] ─→ [64] ─→ value(1)         ← critic (가치)
                    tanh 활성화, actor·critic 별도 네트워크
```

#### 일부러 안 건드린 것들 (SB3 기본값)

`learning_rate=3e-4`, `n_steps=2048`(env당 → 8×2048 = **16,384 샘플/rollout**), `batch_size=64`,
`n_epochs=10`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.0`

**sanity 단계라 의도적으로 전부 기본값.** 튜닝은 변수를 늘리는 일이고, 지금은 "env가 학습
가능한가"만 봐야 한다.

---

### log_std — 가우시안 정책의 표준편차

연속 행동에선 정책이 행동 하나가 아니라 **확률분포**를 낸다:

```
a ~ N(μ(s), σ²)
```

- **μ(s)**: 신경망이 obs를 보고 출력 (상태에 따라 다름)
- **σ**: 얼마나 흔들지 = **탐험(exploration) 폭**

학습 중엔 이 분포에서 **샘플링**(탐험), 평가 땐 **μ만** 쓴다 (`deterministic=True` — `sim_viewer --policy`가 그렇게 함).

#### 왜 σ가 아니라 log σ 를 파라미터로 두는가

**① 양수 제약을 파라미터화로 해결**

σ는 반드시 양수여야 하는데, 경사하강법은 파라미터를 **아무 방향으로나 움직인다**.
σ를 직접 파라미터로 두면 **음수로 내려갈 수 있다.**

```
log_std ∈ (-∞, +∞)      ← 아무 실수나 OK, 제약 없음
σ = exp(log_std) > 0     ← 항상 양수! 자동 보장
```

`exp`가 어떤 실수를 넣어도 양수를 뱉으므로, **제약 조건이 파라미터화 자체로 해결**된다.

**② 로그 공간에서 스케일이 자연스러움**

σ는 학습 중 자릿수가 크게 변한다 (1.0 → 0.01). 로그 공간에선 선형 변화(0 → −4.6)라 경사하강법이
다루기 쉽다. 선형 공간에서 0.01 근처를 미세 조정하려면 그래디언트가 극도로 작아져야 해서 학습이 안 된다.

#### SB3에서의 실제 모습

```
log_std = 학습 가능한 파라미터 벡터 (shape = 2, 행동 차원 수)
```

**중요: obs와 무관하다.** μ는 상태마다 다르지만 **σ는 전역 하나**다 (기본 `use_sde=False`).
즉 "상황이 어렵든 쉽든 같은 크기로 흔든다".

- **초기값** `log_std_init=0.0` → **σ = exp(0) = 1.0**
- 우리 행동 범위가 `[-1, 1]`인데 σ=1이면 **아주 크게 흔드는 것** — 샘플이 자주 ±1로 잘려
  초기 탐험이 거의 극단적 조향/전속력이 된다. **정상이다.**
- 학습이 진행되면 log_std가 내려가 σ가 줄고, 정책이 결정적으로 변한다.

---

### PPO 학습 루프 — rollout · epoch · minibatch

PPO는 일반 딥러닝의 배치 개념을 **버리지 않고 그 위에 한 겹을 더 얹은** 구조다.
지도학습은 데이터셋이 고정이지만, PPO는 **on-policy**라 현재 정책으로 갓 모은 데이터만 쓸 수 있어서
**데이터셋을 매 판 새로 만들어야** 한다. 그 "새로 만드는 단계"가 rollout이다.

```python
for rollout in range(18):           # ← PPO만의 바깥 루프: 데이터셋을 새로 수집 (on-policy라 필요)
    buffer = collect(16_384)        #    n_steps(2048) × n_envs(8) = 16,384 전이
    for epoch in range(10):         # ┐
        for mb in split(buffer, 64):#  ├ 여기부터는 평범한 미니배치 SGD (지도학습과 동일)
            loss.backward()         #  │
            optimizer.step()        # ┘
    # buffer 버림 → 다음 rollout
```

세 층위가 모두 살아 있다:

| 층 | 크기 | 정체 |
|---|---|---|
| **rollout(buffer)** | `n_steps × n_envs` = **16,384** | "이번 판의 임시 데이터셋" (수집 단위) |
| **epoch** | 10 | 그 데이터셋을 몇 번 우려먹나 |
| **minibatch** | `batch_size` = **64** | 경사하강 1스텝에 쓰는 조각 |

```
16,384 수집 → 64씩 쪼개 256 미니배치 → 10 epoch 반복 = 2,560번 경사하강 → 버림 → 반복
30만 스텝 = 300,000 / 16,384 ≈ 18 rollout = 업데이트 18번
```

TensorBoard 곡선의 **점 하나 = rollout 하나**이고, `RolloutLogger._on_rollout_end()`가 이 경계에서 호출된다.

#### clip_range=0.2 — "정책을 한 판에 ±20% 넘게 못 바꾼다"

rollout 하나를 10 epoch 재사용하는 동안 정책이 계속 바뀐다. **데이터는 "옛 정책"으로 모았는데
업데이트 대상은 "조금 바뀐 정책"** 이라, 너무 많이 바뀌면 그 데이터가 현재 정책을 대표하지 못해
엉뚱하게 크게 업데이트되고 **학습이 붕괴**한다 (policy gradient의 고질병).

PPO는 "얼마나 바뀌었나"를 **확률 비율**로 잰다:

```
r = π_new(a|s) / π_old(a|s)     # r=1 안 바뀜, r=1.3 확률 30%↑, r=0.7 30%↓
```

`clip_range=0.2`는 이 비율을 **`[0.8, 1.2]` 밖으로 나가면 잘라** 이득을 없앤다 → 한 업데이트에서
±20%를 벗어나 봤자 보상이 없으니 **급변을 억제**한다. **절벽(붕괴) 근처에서 보폭을 제한하며 걷는 것**
= 데이터가 유효한 **신뢰 구역(trust region)** 안에서만 움직임. PPO(**P**roximal)의 "proximal(근접)"이 이것.

| clip_range | 성격 |
|---|---|
| 0.1 | 보수적 (느리지만 안정) |
| **0.2** | **표준** (거의 모든 구현의 기본값) |
| 0.3+ | 공격적 (빠르지만 붕괴 위험) |

`train/clip_fraction`이 "얼마나 자주 잘렸나"다. 너무 높으면(>0.3) 정책이 급변 중이라는 신호.

---

### 콜백 (Callback) — 학습 루프에 끼어들기

`BaseCallback`을 상속하면 SB3 학습 루프의 특정 시점에 코드를 끼울 수 있다. 훅은 5개:

```
_on_training_start()
  ┌─ _on_rollout_start()
  │  _on_step()            ← 매 venv.step()마다 (rollout 수집 중)
  │  ...
  └─ _on_rollout_end()     ← rollout 끝, PPO 업데이트 직전
_on_training_end()
```

훅만이 아니라 **루프 내부 접근 통로**도 제공한다: `self.model`, `self.training_env`,
`self.logger`(TensorBoard 기록), **`self.locals`**(학습 루프의 지역 변수 딕셔너리 — `dones`·`infos`
등을 여기서 꺼냄), `self.n_calls`·`self.num_timesteps`. (전형적 옵저버 패턴)

#### CheckpointCallback — 주기적 저장

```python
CheckpointCallback(save_freq=max(50_000 // n_envs, 1), save_path="models/ckpt",
                   name_prefix=run_name, save_vecnormalize=True)
```

**save_freq의 함정 — 총 스텝이 아니라 `n_calls`(=venv.step 호출 횟수) 기준.** SB3 내부:

```python
def on_step(self):
    self.n_calls += 1                              # 콜백 자신의 카운터 (메인 프로세스에 1개)
    ...
if self.n_calls % self.save_freq == 0: save()      # ← n_calls 기준!
```

`venv.step()` 1회 = env 8개가 동시에 1스텝 = **총 8 timesteps**. 그래서 "총 5만 스텝마다"는
`50000 // n_envs`. 안 나누면 8배 드물게 저장된다. (카운터는 env 안이 아니라 **콜백 인스턴스**에 있음.)

**`save_vecnormalize=True` 필수** — 기본값 False라, 안 켜면 체크포인트에 정규화 통계가 안 담긴다.
그러면 그 체크포인트는 평가에 못 쓴다(정책이 다른 스케일 obs를 봄). 단, SB3 체크포인트 파일명은
`<prefix>_vecnormalize_<n>_steps.pkl`이라 `sim_viewer`가 찾는 `<정책명>_vecnorm.pkl`과 규칙이 다름 —
체크포인트를 뷰어로 돌리려면 이름 매칭을 따로 손봐야 함 (지금은 최종 모델만 쓰므로 보류).

#### RolloutLogger (우리 커스텀) — 수집과 보고의 시점이 다르다

info의 값을 TensorBoard에 남기는 콜백. **왜 두 훅에 나눠 쓰나:**

```python
def _on_step(self):                       # 매 스텝 — "수집"
    for done, info in zip(dones, infos):  # env 8개 순회
        for k in KEYS: self._sum[k] += info[k]   # 값 누적
        self._steps += 1                          # 샘플 수 (env마다 +1)
        if done:                                  # 종료는 이 순간에만 알 수 있음!
            self._outcome.append("goal"/"flip"/"timeout")

def _on_rollout_end(self):                # rollout 끝 — "보고"
    record(f"detail/{k}", sum/self._steps)         # 스텝당 평균
    record(f"outcome/{n}_rate", count/len)
    self._sum.clear(); self._steps = 0             # ← _sum·_steps만 리셋
```

- **`self._steps`**: 이번 rollout에서 본 (env,스텝) 샘플 수. `_on_step` 1회에 8씩 늘어 rollout당 16,384.
  `_sum`을 이걸로 나눠 **스텝당 평균** slip/energy/r_*를 낸다. rollout마다 초기화.
- **`self._outcome`**: `deque(maxlen=100)`. 최근 **100 에피소드**의 종료 원인. **초기화 안 함** —
  deque가 오래된 걸 밀어내며 이동창을 유지한다 (rollout마다 리셋하면 표본이 30~80개로 들쭉날쭉).
- **종료(`done`)는 그 스텝에서만 info에 드러나므로 `_on_step`에서 잡아야** 한다. `_on_rollout_end`
  시점엔 지나간 done을 알 수 없다 → 수집(_on_step)과 집계(_on_rollout_end)의 시점 분리가 필수.
- info를 **이름으로** 읽는다(obs 인덱스가 아니라) → obs 레이아웃이 바뀌어도 조용히 깨지지 않음.

---

### 학습 로그에서 볼 것

| 지표 | 의미 | 판단 |
|---|---|---|
| `rollout/ep_rew_mean` | 원본 return 평균 (Monitor 덕에 정규화 전) | **오르면 학습 중** ✅ |
| `outcome/goal_rate` | 도달률 | **무작위(0%)보다 높으면 통과** (휴리스틱 73%가 상한 감각) |
| `outcome/flip_rate` | 전복률 | 급증하면 **자살 exploit** 의심 |
| **`train/std`** | `exp(log_std)` | **1.0 → 0.5 → 0.2 하강 = 수렴 중** ✅ / 1.0 고정 = 못 배우는 중 / 급히 0 = 조기 수렴(탐험 부족) |
| `train/clip_fraction` | clip에 걸린 비율 | **>0.3이면 정책 급변 중** (불안정 신호) |
| `detail/r_*` | reward 항목별 기여도 | 가중치 단계적 활성화 시 각 항목 영향 확인 |
| `detail/slip`, `detail/energy` | 원시 측정값 | 가중치 0이어도 크기를 볼 수 있음 |
