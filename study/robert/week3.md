# Week 3: ALOHA Insertion 시뮬레이션 실습 가이드

## 🎯 학습 목표
- ALOHA 환경 설정 및 이해
- ACT (Action Chunking Transformer) 정책 활용
- 양팔(Bimanual) 조작 시뮬레이션 실행

---

## 📋 사전 준비

### Step 1: 환경 확인
```powershell
# 가상환경 활성화 확인
cd d:\git\lerobot
.\.venv\Scripts\Activate.ps1

# Week 2 완료 확인 (PushT가 동작해야 함)
python -c "from lerobot.datasets import LeRobotDataset; print('OK')"
```

### Step 2: ALOHA 의존성 설치
```powershell
# ALOHA 환경 설치 (gym-aloha 포함)
pip install -e ".[aloha]"
```

---

## 🤖 Part 1: ALOHA 환경 이해

### ALOHA란?
- **양팔 로봇 시스템**: 두 개의 ViperX 로봇 팔
- **Bimanual Manipulation**: 양손 협업 조작 작업
- **ACT 정책**: ALOHA를 위해 개발된 Action Chunking Transformer

### 주요 태스크
| 태스크 | 설명 | 난이도 |
|--------|------|--------|
| `AlohaInsertion-v0` | 페그를 구멍에 삽입 | ⭐⭐ 중급 |
| `AlohaTransferCube-v0` | 큐브를 손에서 손으로 전달 | ⭐⭐ 중급 |

---

## 📊 Part 2: 데이터셋 탐색

### Step 3: ALOHA 데이터셋 로드
```python
python

>>> from lerobot.datasets import LeRobotDataset

>>> # ALOHA Insertion 데이터셋 로드
>>> dataset = LeRobotDataset("lerobot/aloha_sim_insertion_human")

>>> # 데이터셋 정보 확인
>>> print(f"에피소드 수: {dataset.num_episodes}")
>>> print(f"총 프레임 수: {len(dataset)}")

>>> # 특성 확인
>>> for key, value in dataset.meta.features.items():
...     print(f"{key}: {value}")

>>> exit()
```

### Step 4: 데이터셋 구조 분석
```python
python

>>> from lerobot.datasets import LeRobotDataset
>>> dataset = LeRobotDataset("lerobot/aloha_sim_insertion_human")

>>> # 첫 번째 샘플 확인
>>> sample = dataset[0]

>>> # 관찰 데이터 확인
>>> print("=== 관찰 데이터 ===")
>>> for key in sample.keys():
...     if hasattr(sample[key], 'shape'):
...         print(f"{key}: {sample[key].shape}")
...     else:
...         print(f"{key}: {sample[key]}")

>>> # 행동 데이터 (양팔이므로 14차원: 7 + 7)
>>> print(f"\n행동 차원: {sample['action'].shape}")
>>> exit()
```

---

## 🎬 Part 3: ACT 정책 이해

### ACT (Action Chunking Transformer)란?
- **CVAE + Transformer**: 조건부 변분 오토인코더와 트랜스포머 결합
- **Action Chunking**: 여러 시간 스텝의 행동을 한번에 예측
- **양팔 조작**: ALOHA 시스템을 위해 설계됨

### Step 5: ACT 정책 로드
```python
python

>>> from lerobot.policies.act import ACTPolicy

>>> # 사전 학습된 ACT 모델 로드
>>> policy = ACTPolicy.from_pretrained("lerobot/act_aloha_sim_insertion_human")

>>> print(f"정책 타입: {type(policy)}")
>>> print(f"설정: {policy.config}")
>>> exit()
```

---

## 🎮 Part 4: 시뮬레이션 평가

### Step 6: ALOHA Insertion 평가
```powershell
# ALOHA Insertion 환경에서 ACT 정책 평가
python -m lerobot.scripts.lerobot_eval `
  --policy.path=lerobot/act_aloha_sim_insertion_human `
  --env.type=aloha `
  --env.task=AlohaInsertion-v0 `
  --eval.n_episodes=5 `
  --eval.batch_size=1 `
  --output_dir=outputs/eval_aloha_insertion
```

### Step 7: 평가 결과 확인
```powershell
# 출력 디렉토리 확인
dir outputs/eval_aloha_insertion

# 성공률 확인
type outputs/eval_aloha_insertion\eval_info.json
```

---

## 📹 Part 5: 시각화

### Step 8: 데이터셋 에피소드 시각화
```powershell
# 첫 번째 에피소드 시각화
python -m lerobot.scripts.visualize_dataset `
  --repo-id lerobot/aloha_sim_insertion_human `
  --episode-index 0
```

### Step 9: 정책 행동 시각화
```python
python

>>> import torch
>>> from lerobot.policies.act import ACTPolicy
>>> from lerobot.datasets import LeRobotDataset

>>> # 정책과 데이터셋 로드
>>> policy = ACTPolicy.from_pretrained("lerobot/act_aloha_sim_insertion_human")
>>> dataset = LeRobotDataset("lerobot/aloha_sim_insertion_human")

>>> # 샘플 관찰로 행동 예측
>>> sample = dataset[0]
>>> observation = {k: v.unsqueeze(0) for k, v in sample.items() if k.startswith("observation")}

>>> # 행동 예측
>>> with torch.no_grad():
...     action = policy.select_action(observation)
...     print(f"예측된 행동 shape: {action.shape}")
...     print(f"행동 값 범위: [{action.min():.3f}, {action.max():.3f}]")

>>> exit()
```

---

## 🏋️ Part 6: ACT 정책 학습 (선택사항)

### Step 10: 짧은 학습 테스트
```powershell
# ACT 정책 학습 (100 스텝만 테스트)
python -m lerobot.scripts.train `
  --policy.type=act `
  --dataset.repo_id=lerobot/aloha_sim_insertion_human `
  --output_dir=outputs/train_aloha_insertion `
  --steps=100 `
  --eval_freq=50 `
  --batch_size=4
```

### Step 11: 학습된 모델 평가
```powershell
# 학습된 모델로 평가
python -m lerobot.scripts.lerobot_eval `
  --policy.path=outputs/train_aloha_insertion `
  --env.type=aloha `
  --env.task=AlohaInsertion-v0 `
  --eval.n_episodes=3
```

---

## 🔍 Part 7: 코드 분석

### 주요 파일 위치
| 파일 | 설명 |
|------|------|
| `src/lerobot/policies/act/modeling_act.py` | ACT 모델 구현 |
| `src/lerobot/policies/act/configuration_act.py` | ACT 설정 |
| `src/lerobot/policies/act/processor_act.py` | 전처리/후처리 |

### ACT 핵심 구조
```python
# ACT 정책의 주요 구성 요소
# 
# 1. CVAE Encoder: 학습 시 행동을 latent space로 인코딩
# 2. Transformer Encoder: 관찰 데이터 처리
# 3. Transformer Decoder: latent + 관찰 → 행동 시퀀스 생성
# 4. Action Chunking: 한 번에 여러 스텝 예측 (예: 100 스텝)
```

### Step 12: ACT 모델 구조 확인
```python
python

>>> from lerobot.policies.act import ACTPolicy
>>> policy = ACTPolicy.from_pretrained("lerobot/act_aloha_sim_insertion_human")

>>> # 모델 구조 출력
>>> print(policy.model)

>>> # 파라미터 수 확인
>>> total_params = sum(p.numel() for p in policy.model.parameters())
>>> print(f"\n총 파라미터 수: {total_params:,}")

>>> exit()
```

---

## 🔄 Part 8: PushT vs ALOHA 비교

| 항목 | PushT | ALOHA Insertion |
|------|-------|-----------------|
| 차원 | 2D | 3D |
| 로봇 | 점(point) | 양팔 (14 DoF) |
| 정책 | Diffusion | ACT |
| 관찰 | 이미지 + 상태 | 이미지 + 관절 상태 |
| 행동 | 2D 위치 | 14D 관절 명령 |
| 학습 난이도 | 낮음 | 중간 |

---

## ✅ 체크리스트

- [ ] ALOHA 의존성 설치 완료
- [ ] 데이터셋 로드 및 구조 확인
- [ ] ACT 정책 로드
- [ ] ALOHA Insertion 시뮬레이션 평가 실행
- [ ] 데이터셋 시각화
- [ ] ACT 모델 구조 분석
- [ ] (선택) 짧은 학습 테스트

---

## 🚨 문제 해결

### 문제 1: `gym_aloha` 모듈 없음
```powershell
pip install gym-aloha
```

### 문제 2: MuJoCo 렌더링 오류
```powershell
# Windows에서 MuJoCo 렌더링 설정
$env:MUJOCO_GL="egl"
# 또는
$env:MUJOCO_GL="osmesa"
```

### 문제 3: CUDA 메모리 부족
```powershell
# 배치 사이즈 줄이기
python -m lerobot.scripts.train ... --batch_size=2
```

### 문제 4: WSL2 권장
ALOHA 시뮬레이션은 Linux에서 가장 안정적입니다.
Windows에서 문제 발생 시 WSL2 사용을 권장합니다.

---

## 📚 추가 학습 자료

### 논문
- **ACT**: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
- **ALOHA**: Stanford의 ALOHA 프로젝트

### 관련 데이터셋
```python
# ALOHA 관련 데이터셋 목록
datasets = [
    "lerobot/aloha_sim_insertion_human",
    "lerobot/aloha_sim_insertion_scripted", 
    "lerobot/aloha_sim_transfer_cube_human",
    "lerobot/aloha_sim_transfer_cube_scripted",
]
```

---

## 🎯 다음 단계

Week 3을 완료했다면:
1. **LIBERO** 또는 **MetaWorld** 벤치마크 도전
2. **실제 로봇** 통합 학습 (하드웨어 필요)
3. **VLA (Vision-Language-Action)** 모델 탐색

---

**축하합니다! 3D 시뮬레이션 환경을 마스터했습니다! 🎉🤖**
