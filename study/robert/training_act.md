# LeRobot ACT Training (Colab)

LeRobot 라이브러리를 사용하여 **ACT (Action Chunking with Transformers)** 정책을 Google Colab에서 학습하는 가이드입니다.

---

## 요구사항

- **데이터셋**: Hugging Face Hub에 업로드된 LeRobot 포맷 데이터셋
- **GPU**: NVIDIA A100 권장 (T4도 가능)
- **계정**: Hugging Face, Weights & Biases (선택)

### 예상 학습 시간
- A100 GPU: 100,000 스텝 약 **1.5시간**
- T4 GPU: 약 3-4시간

---

## 1. 환경 설정

### Conda 설치 (Colab)
```python
!pip install -q condacolab
import condacolab
condacolab.install()
```

### LeRobot 설치
```python
!git clone https://github.com/huggingface/lerobot.git
!conda install ffmpeg=7.1.1 -c conda-forge
!cd lerobot && pip install -e .
```

---

## 2. 로그인

### Weights & Biases (학습 모니터링)
```python
!wandb login
```

### Hugging Face Hub
```python
from google.colab import userdata
import os

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

또는 CLI로 로그인:
```python
!hf auth login
```

---

## 3. ACT 학습

### 기본 학습 명령어
```python
!cd lerobot && python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=${HF_USER}/your_dataset \
  --policy.type=act \
  --output_dir=outputs/train/act_training \
  --job_name=act_training_job \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/act_policy
```

### 주요 파라미터

| 파라미터 | 설명 |
|---------|------|
| `--dataset.repo_id` | HF Hub 데이터셋 ID |
| `--policy.type` | 정책 타입 (`act`, `diffusion` 등) |
| `--output_dir` | 체크포인트 저장 경로 |
| `--policy.device` | `cuda`, `mps`, `cpu` |
| `--wandb.enable` | W&B 로깅 활성화 |
| `--policy.repo_id` | 학습된 모델 저장 repo |

### ACT 정책 설정 예시
```yaml
policy:
  chunk_size: 100
  dim_model: 512
  n_heads: 8
  n_encoder_layers: 4
  n_decoder_layers: 1
  vision_backbone: resnet18
  use_vae: true
  kl_weight: 10.0
```

---

## 4. 학습 결과 확인

### 로컬 체크포인트 확인
```python
!ls -l /content/lerobot/outputs/train/act_training/
```

### 출력 파일 구조
```
outputs/train/act_training/
├── checkpoints/
│   ├── 020000/
│   ├── 040000/
│   └── last/
│       └── pretrained_model/
│           ├── model.safetensors
│           └── config.json
├── train_config.json
└── logs/
```

---

## 5. 모델 업로드

### HF Hub에 모델 업로드
```python
!hf upload ${HF_USER}/act_policy \
  /content/lerobot/outputs/train/act_training/checkpoints/last/pretrained_model
```

### 설정 파일 별도 저장
```python
# 레포 생성
!hf repo create act-configs --type model --private

# 설정 업로드
!hf upload ${HF_USER}/act-configs \
  "/content/lerobot/outputs/train/act_training/train_config.json" \
  train_config.json
```

---

## 6. 모델 다운로드

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="${HF_USER}/act_policy", 
    filename="model.safetensors"
)
print(f"Downloaded to: {model_path}")
```

---

## 참고 사항

| 항목 | 권장 설정 |
|------|----------|
| **GPU** | A100 (최적), T4 (가능) |
| **batch_size** | T4: 8, A100: 32 |
| **steps** | 100,000 (기본) |
| **평가 주기** | 20,000 스텝마다 |
| **저장 주기** | 20,000 스텝마다 |

### 팁
- Colab 런타임 중 브라우저 탭 유지
- 체크포인트는 Google Drive에 주기적 백업 권장
- `--wandb.enable=true`로 학습 진행 실시간 모니터링
