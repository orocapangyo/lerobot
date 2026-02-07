# Week 2: PushT 시뮬레이션 실습 가이드

## 🎯 학습 목표
- PushT 환경 설정 및 실행
- Diffusion Policy 이해 및 활용
- 시뮬레이션 평가 파이프라인 구축

---

## 📋 사전 준비

### Step 1: 환경 확인
```powershell
# 가상환경 활성화 확인
cd d:\git\lerobot
.\.venv\Scripts\Activate.ps1

# Python 버전 확인 (3.10+ 필요)
python --version
```

### Step 2: PushT 의존성 설치
```powershell
# PushT 환경 설치 (gym-pusht 포함)
uv pip install -e ".[pusht]"
```

---

## 🎮 Part 1: PushT 환경 이해

### PushT란?
- **2D 푸싱 태스크**: T자 모양 블록을 목표 위치로 밀어서 옮기는 작업
- **간단한 환경**: 로봇 학습 입문에 적합
- **빠른 피드백**: 학습과 평가가 빠름

### Step 3: 데이터셋 확인
```python
# Python 대화형 모드에서 실행
python

>>> from lerobot.datasets import LeRobotDataset

>>> # PushT 데이터셋 로드
>>> dataset = LeRobotDataset("lerobot/pusht")

>>> # 데이터셋 정보 확인
>>> print(f"에피소드 수: {dataset.num_episodes}")
>>> print(f"총 프레임 수: {len(dataset)}")
>>> print(f"특성: {dataset.meta.features}")

>>> # 첫 번째 샘플 확인
>>> sample = dataset[0]
>>> print(f"관찰 키: {[k for k in sample.keys()]}")
>>> print(f"행동 shape: {sample['action'].shape}")
>>> exit()
```

---

## 🤖 Part 2: 사전 학습된 정책 사용

### Step 4: Diffusion Policy 로드
```python
python

>>> from lerobot.policies.diffusion import DiffusionPolicy

>>> # 사전 학습된 모델 로드
>>> policy = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht")

>>> print(f"정책 타입: {type(policy)}")
>>> print(f"디바이스: {policy.device}")
>>> exit()
```

---

## 🎬 Part 3: 시뮬레이션 평가

### Step 5: 사전 학습된 정책 평가
```powershell
# PushT 환경에서 Diffusion Policy 평가
python -m lerobot.scripts.lerobot_eval `
  --policy.path=lerobot/diffusion_pusht `
  --env.type=pusht `
  --eval.n_episodes=10 `
  --eval.batch_size=1 `
  --output_dir=outputs/eval_pusht
```

### Step 6: 평가 결과 확인
```powershell
# 출력 디렉토리 확인
dir outputs/eval_pusht

# 비디오 파일 확인 (생성된 경우)
dir outputs/eval_pusht\videos
```

---

## 📊 Part 4: 데이터셋 시각화

### Step 7: 에피소드 시각화
```powershell
# 첫 번째 에피소드 시각화
python -m lerobot.scripts.visualize_dataset `
  --repo-id lerobot/pusht `
  --episode-index 0
```

---

## 🏋️ Part 5: 정책 학습 (선택사항)

### Step 8: Diffusion Policy 학습
```powershell
# 짧은 학습 테스트 (100 스텝)
python -m lerobot.scripts.train `
  --policy.type=diffusion `
  --dataset.repo_id=lerobot/pusht `
  --output_dir=outputs/train_pusht `
  --steps=100 `
  --eval_freq=50
```

### Step 9: 학습된 모델 평가
```powershell
# 학습된 모델로 평가
python -m lerobot.scripts.lerobot_eval `
  --policy.path=outputs/train_pusht `
  --env.type=pusht `
  --eval.n_episodes=5
```

---

## 🔍 Part 6: 코드 분석

### 주요 파일 위치
| 파일 | 설명 |
|------|------|
| `src/lerobot/policies/diffusion/` | Diffusion Policy 구현 |
| `src/lerobot/envs/` | 환경 팩토리 |
| `src/lerobot/scripts/train.py` | 학습 스크립트 |
| `src/lerobot/scripts/lerobot_eval.py` | 평가 스크립트 |

### Diffusion Policy 핵심 개념
```python
# Diffusion Policy의 핵심 구조
# 1. Noise Scheduler: 노이즈 추가/제거 스케줄 관리
# 2. UNet: 노이즈 예측 네트워크
# 3. Action Chunking: 여러 시간 스텝의 행동을 한번에 예측
```

---

## ✅ 체크리스트

- [ ] PushT 의존성 설치 완료
- [ ] 데이터셋 로드 및 구조 확인
- [ ] 사전 학습된 정책 로드
- [ ] 시뮬레이션 평가 실행
- [ ] 데이터셋 시각화
- [ ] (선택) 짧은 학습 테스트

---

## 🚨 문제 해결

### 문제 1: `gym_pusht` 모듈 없음
```powershell
pip install gym-pusht
```

### 문제 2: CUDA 메모리 부족
```powershell
# CPU에서 실행
$env:CUDA_VISIBLE_DEVICES=""
python -m lerobot.scripts.lerobot_eval ...
```

### 문제 3: 렌더링 오류 (Windows)
```powershell
# EGL 대신 소프트웨어 렌더링 사용
$env:MUJOCO_GL="osmesa"
```

---

## 📚 다음 단계
- **Week 3**: ALOHA Insertion (3D 환경)으로 넘어가기
- Diffusion Policy와 ACT 정책 비교 학습

---

**완료 후 Week 3으로 진행하세요! 🎉**
