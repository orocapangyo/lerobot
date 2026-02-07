# LeRobot 코드베이스 분석

## 📌 프로젝트 개요

**LeRobot**은 Hugging Face에서 개발한 **PyTorch 기반 로봇 제어 및 학습 라이브러리**입니다.

- **버전**: 0.4.4
- **Python**: 3.10+
- **라이선스**: Apache 2.0
- **목표**: 모델, 데이터셋, 도구를 제공하여 로봇 공학의 진입 장벽을 낮추고 공유 데이터셋과 사전 학습 모델의 혜택을 누릴 수 있도록 함

---

## 📂 소스 코드 구조

```
src/lerobot/
├── __init__.py          # 사용 가능한 환경, 데이터셋, 정책 목록
├── async_inference/     # 비동기 추론 지원
├── cameras/             # 카메라 인터페이스 (OpenCV, RealSense, ZMQ)
├── configs/             # 설정 파일
├── data_processing/     # 데이터 전처리
├── datasets/            # LeRobotDataset 포맷 및 유틸리티
├── envs/                # 시뮬레이션 환경 (LIBERO, MetaWorld)
├── model/               # 모델 기반 클래스
├── motors/              # 모터 드라이버 (Dynamixel, Feetech, Damiao)
├── optim/               # 옵티마이저
├── policies/            # 정책 구현 (14종)
├── processor/           # 데이터 프로세서
├── rl/                  # 강화학습 유틸리티
├── robots/              # 로봇 인터페이스 (11종)
├── scripts/             # CLI 스크립트 (16개)
├── teleoperators/       # 원격 조종 장치 (12종)
├── templates/           # 템플릿
├── transport/           # 통신 레이어
└── utils/               # 유틸리티 함수
```

---

## 🤖 지원 로봇 (11종)

| 카테고리 | 로봇 |
|---------|------|
| **SO 시리즈** | SO100, SO101, bi_so_follower |
| **Koch** | koch_follower |
| **OpenArm** | openarm_follower, bi_openarm_follower |
| **휴머노이드** | hope_jr, unitree_g1 |
| **모바일** | lekiwi, earthrover_mini_plus |
| **고급** | reachy2, omx_follower |

---

## 🎮 원격조종 장치 (12종)

- **리더 암**: so_leader, bi_so_leader, koch_leader, openarm_leader, omx_leader
- **게임패드**: gamepad
- **입력장치**: keyboard, phone
- **휴머노이드**: homunculus, unitree_g1, reachy2_teleoperator

---

## 📷 카메라 지원 (4종)

| 타입 | 설명 |
|------|------|
| **OpenCV** | 일반 USB 카메라 |
| **Intel RealSense** | 깊이 카메라 |
| **ZMQ** | 네트워크 카메라 |
| **Reachy2** | Reachy2 전용 카메라 |

---

## 🧠 정책 (Policies) - 14종

### Imitation Learning
| 정책 | 설명 |
|------|------|
| **ACT** | Action Chunking with Transformers |
| **Diffusion** | Diffusion Policy |
| **VQ-BeT** | Vector Quantized Behavior Transformer |

### Reinforcement Learning
| 정책 | 설명 |
|------|------|
| **TDMPC** | Temporal Difference Model Predictive Control |
| **SAC** | Soft Actor-Critic |
| **RTC** | Real-Time Critic |

### Vision-Language-Action (VLA) Models
| 정책 | 설명 |
|------|------|
| **Pi0** | π₀ 기본 모델 |
| **Pi0.5** | π₀.5 개선 모델 |
| **Pi0 Fast** | 빠른 추론용 모델 |
| **SmolVLA** | 경량 VLA 모델 |
| **GR00T** | NVIDIA GR00T N1.5 |
| **XVLA** | 확장 VLA 모델 |
| **SARM** | Spatial Attention Robot Model |
| **Wall-X** | Qwen2.5-VL 기반 모델 |

---

## 🎮 시뮬레이션 환경

| 환경 | 태스크 |
|------|--------|
| **ALOHA** | AlohaInsertion-v0, AlohaTransferCube-v0 |
| **PushT** | PushT-v0 |
| **LIBERO** | 다양한 조작 태스크 |
| **MetaWorld** | ML1 벤치마크 태스크 |

---

## 📊 데이터셋 (LeRobotDataset)

### 포맷
- **영상**: MP4 또는 이미지 시퀀스
- **상태/액션**: Parquet 파일
- **Hugging Face Hub** 통합 지원

### 주요 클래스
- `LeRobotDataset`: 메인 데이터셋 클래스
- `StreamingDataset`: 스트리밍 데이터 로딩
- `OnlineBuffer`: 온라인 학습용 버퍼

---

## 🛠️ CLI 스크립트 (16개)

| 명령어 | 설명 |
|--------|------|
| `lerobot-train` | 정책 학습 |
| `lerobot-eval` | 정책 평가 |
| `lerobot-record` | 데이터 수집 |
| `lerobot-replay` | 에피소드 재생 |
| `lerobot-teleoperate` | 원격 조종 |
| `lerobot-calibrate` | 로봇 캘리브레이션 |
| `lerobot-find-cameras` | 카메라 탐색 |
| `lerobot-find-port` | 포트 탐색 |
| `lerobot-setup-motors` | 모터 설정 |
| `lerobot-setup-can` | CAN 버스 설정 |
| `lerobot-find-joint-limits` | 관절 한계 탐색 |
| `lerobot-dataset-viz` | 데이터셋 시각화 |
| `lerobot-edit-dataset` | 데이터셋 편집 |
| `lerobot-imgtransform-viz` | 이미지 변환 시각화 |
| `lerobot-info` | 시스템 정보 출력 |
| `lerobot-train-tokenizer` | 토크나이저 학습 |

---

## 🚀 빠른 시작

### 설치
```bash
pip install lerobot
lerobot-info
```

### 기본 사용법
```python
from lerobot.robots.so_follower import SOFollower
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 로봇 연결
robot = SOFollower(config=...)
robot.connect()

# 데이터셋 로드
dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")

# 관측 및 액션
obs = robot.get_observation()
action = model.select_action(obs)
robot.send_action(action)
```

### 학습
```bash
lerobot-train \
  --policy=act \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet
```

### 평가
```bash
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object
```

---

## ☁️ Google Colab에서 학습하기

### 1. 환경 설정

```python
# GPU 런타임 확인 (런타임 > 런타임 유형 변경 > GPU 선택)
!nvidia-smi

# LeRobot 설치
!pip install lerobot

# 추가 의존성 설치 (시뮬레이션 환경용)
!pip install "lerobot[pusht]"  # PushT 환경
!pip install "lerobot[aloha]"  # ALOHA 환경
```

### 2. Hugging Face 로그인

```python
from huggingface_hub import login

# 토큰 입력 (https://huggingface.co/settings/tokens 에서 발급)
login()
```

### 3. 데이터셋 로드 및 확인

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 데이터셋 로드
dataset = LeRobotDataset("lerobot/pusht")

# 데이터셋 정보 확인
print(f"에피소드 수: {dataset.num_episodes}")
print(f"샘플 수: {len(dataset)}")
print(f"특성: {dataset.features}")
```

### 4. ACT 정책 학습

```python
# 방법 1: CLI 명령어 사용
!lerobot-train \
    --policy=act \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --training.num_epochs=100 \
    --training.batch_size=8 \
    --output_dir=outputs/act_aloha
```

```python
# 방법 2: Python 코드 사용
from lerobot.policies.act import ACTPolicy, ACTConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch

# 설정
config = ACTConfig()
dataset = LeRobotDataset("lerobot/pusht")

# 정책 초기화
policy = ACTPolicy(config, dataset_stats=dataset.stats)
policy = policy.to("cuda")

# 학습 루프 (간략화)
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
for batch in dataloader:
    loss = policy.forward(batch)
    loss.backward()
    optimizer.step()
```

### 5. Diffusion Policy 학습

```python
!lerobot-train \
    --policy=diffusion \
    --dataset.repo_id=lerobot/pusht \
    --training.num_epochs=100 \
    --training.batch_size=64 \
    --output_dir=outputs/diffusion_pusht
```

### 6. 시뮬레이션에서 평가

```python
!lerobot-eval \
    --policy.path=outputs/act_aloha/checkpoints/last \
    --env.type=aloha \
    --env.task=AlohaInsertion-v0 \
    --eval.n_episodes=10
```

### 7. 모델 허브에 업로드

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="outputs/act_aloha",
    repo_id="your-username/act-aloha-trained",
    repo_type="model"
)
```

### 8. Colab 팁

| 팁 | 설명 |
|----|------|
| **GPU 메모리** | T4(16GB) 기준 batch_size=8~16 권장 |
| **런타임 유지** | 학습 중 브라우저 탭 유지 |
| **체크포인트** | Google Drive 마운트 후 저장 권장 |
| **Wandb** | `--wandb.enable=true`로 학습 모니터링 |

### 9. Google Drive 연동

```python
from google.colab import drive
drive.mount('/content/drive')

# 체크포인트를 Drive에 저장
!lerobot-train \
    --policy=act \
    --dataset.repo_id=lerobot/pusht \
    --output_dir=/content/drive/MyDrive/lerobot_outputs
```

---

## 📚 참고 자료

- **문서**: https://huggingface.co/docs/lerobot/index
- **GitHub**: https://github.com/huggingface/lerobot
- **Discord**: https://discord.gg/q8Dzzpym3f
- **한국어 튜토리얼**: https://zihao-ai.feishu.cn/wiki/space/7589642043471924447
