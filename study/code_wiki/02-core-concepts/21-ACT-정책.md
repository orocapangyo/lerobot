# ACT (Action Chunking Transformer) 정책

## 개요

ACT는 **Action Chunking with Transformers**의 약자로, ALOHA 프로젝트에서 개발된 모방 학습 정책입니다. CVAE(Conditional Variational Autoencoder)와 Transformer를 결합하여 복잡한 양팔 조작 작업을 학습할 수 있습니다.

## 핵심 개념

### Action Chunking

ACT의 핵심 아이디어는 한 번에 여러 스텝의 액션을 예측하는 것입니다:

```python
# 일반 정책: 현재 관측 -> 1개 액션
action_t = policy(observation_t)

# ACT: 현재 관측 -> 100개 액션
actions_t_to_t+99 = act_policy(observation_t)
```

**장점:**
- 시간적 일관성 향상
- 더 부드러운 행동 궤적
- 긴 시퀀스를 고려한 계획

### CVAE (Conditional Variational Autoencoder)

ACT는 CVAE를 사용하여 행동의 다양성을 모델링합니다:

```
훈련 시:
observation -> encoder -> latent (z) -> decoder -> action_sequence
                            ↑
                        KL divergence loss

추론 시:
observation -> sample z ~ N(0,1) -> decoder -> action_sequence
```

---

## 아키텍처

### 전체 구조

```python
class ACTPolicy(PreTrainedPolicy):
    """
    Components:
    1. Vision Backbone (ResNet-18)
    2. CVAE Encoder (훈련 시만)
    3. CVAE Decoder
    4. Transformer (Encoder-Decoder)
    """

    def __init__(self, config):
        # 1. 비전 백본
        self.backbone = ResNet18(...)

        # 2. CVAE 인코더 (action -> latent)
        self.encoder = nn.Linear(action_dim * chunk_size, latent_dim)

        # 3. CVAE 디코더 (latent -> action)
        self.latent_proj = nn.Linear(latent_dim, dim_model)

        # 4. Transformer
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
        )

        # 5. Action head
        self.action_head = nn.Linear(dim_model, action_dim)
```

### 상세 구조

#### 1. Vision Backbone

```python
# ResNet-18으로 이미지 특징 추출
image: (B, C, H, W)
  ↓ ResNet-18
features: (B, D, H', W')
  ↓ Flatten & Project
vision_tokens: (B, num_patches, dim_model)
```

#### 2. State Encoding

```python
# State (proprioception)도 임베딩
state: (B, state_dim)
  ↓ Linear
state_embedding: (B, dim_model)
```

#### 3. CVAE Encoder (훈련 시)

```python
# Ground truth action을 latent로 인코딩
action_sequence: (B, chunk_size, action_dim)
  ↓ Flatten
action_flat: (B, chunk_size * action_dim)
  ↓ Linear
latent_params: (B, latent_dim * 2)
  ↓ Split
mu, log_var = latent_params.chunk(2, dim=-1)
  ↓ Reparameterization
latent: (B, latent_dim)
```

#### 4. CVAE Decoder (추론 시)

```python
# 추론 시: latent를 N(0,1)에서 샘플링
latent = torch.randn(B, latent_dim)
  ↓ Linear
latent_embedding: (B, dim_model)
```

#### 5. Transformer

```python
# Encoder: 관측 처리
encoder_input = [vision_tokens, state_embedding]  # (B, seq_len, dim_model)
encoder_output = transformer.encoder(encoder_input)

# Decoder: Action sequence 생성
decoder_input = latent_embedding + positional_encoding  # (B, chunk_size, dim_model)
decoder_output = transformer.decoder(decoder_input, encoder_output)

# Action 예측
action_sequence = action_head(decoder_output)  # (B, chunk_size, action_dim)
```

---

## 사용법

### 모델 로드

```python
from lerobot.policies.act import ACTPolicy

# 사전 훈련된 모델
policy = ACTPolicy.from_pretrained("lerobot/act_aloha_insertion")

# 설정 확인
print(f"Chunk size: {policy.config.chunk_size}")
print(f"Latent dim: {policy.config.latent_dim}")
print(f"Input shapes: {policy.config.input_shapes}")
```

### 추론

```python
import torch

policy.eval()
policy = policy.to("cuda")

# 관측 데이터 (ALOHA 예시)
observation = {
    "observation.images.cam_high": torch.randn(1, 3, 224, 224).cuda(),
    "observation.images.cam_left_wrist": torch.randn(1, 3, 224, 224).cuda(),
    "observation.images.cam_right_wrist": torch.randn(1, 3, 224, 224).cuda(),
    "observation.state": torch.randn(1, 14).cuda(),  # 14 joint positions
}

# Action sequence 생성
with torch.no_grad():
    actions = policy.select_action(observation)

print(actions.shape)  # (1, action_dim) - 첫 번째 action만 반환
```

### Action Chunking 사용

```python
# 전체 action chunk 가져오기
policy.eval()

observation = {...}  # 위와 동일

with torch.no_grad():
    # 내부적으로 chunk 생성
    output = policy(observation)
    action_chunk = output["action"]  # (1, chunk_size, action_dim)

# 환경에서 실행
for t in range(chunk_size):
    action_t = action_chunk[0, t]  # t번째 action
    observation, reward, done, info = env.step(action_t)

    if done:
        break
```

---

## 훈련

### 데이터셋 준비

```python
from lerobot.datasets import LeRobotDataset

# ACT는 action chunk를 위한 delta_timestamps 필요
dataset = LeRobotDataset(
    "lerobot/aloha_insertion",
    delta_timestamps={
        "observation.images.cam_high": [0],
        "observation.images.cam_left_wrist": [0],
        "observation.images.cam_right_wrist": [0],
        "observation.state": [0],
        "action": list(range(100)),  # 100-step chunk
    }
)

print(f"Action shape: {dataset[0]['action'].shape}")  # (100, action_dim)
```

### 훈련 루프

```python
from torch.utils.data import DataLoader
import torch.nn.functional as F

policy = ACTPolicy(config)
policy.train()
policy = policy.to("cuda")

optimizer = torch.optim.AdamW([
    {"params": policy.backbone.parameters(), "lr": 1e-5},  # 백본은 낮은 LR
    {"params": policy.transformer.parameters(), "lr": 1e-4},
], weight_decay=1e-4)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

for epoch in range(num_epochs):
    for batch in dataloader:
        # GPU로 이동
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # 순전파
        output = policy(batch)

        # 손실 계산
        loss = policy.compute_loss(output, batch)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")
```

### 손실 함수

ACT의 손실은 두 부분으로 구성됩니다:

```python
def compute_loss(self, output, batch):
    # 1. Action 예측 손실 (L2)
    predicted_actions = output["action"]  # (B, chunk_size, action_dim)
    target_actions = batch["action"]      # (B, chunk_size, action_dim)

    action_loss = F.mse_loss(predicted_actions, target_actions)

    # 2. KL Divergence 손실 (CVAE 정규화)
    mu = output["mu"]              # (B, latent_dim)
    log_var = output["log_var"]    # (B, latent_dim)

    kl_loss = -0.5 * torch.sum(
        1 + log_var - mu.pow(2) - log_var.exp(),
        dim=-1
    ).mean()

    # 총 손실
    total_loss = action_loss + self.config.kl_weight * kl_loss

    return total_loss
```

---

## 설정 (Configuration)

### 주요 파라미터

```python
from lerobot.policies.act import ACTConfig

config = ACTConfig(
    # 입출력 shape
    input_shapes={
        "observation.images.cam_high": (3, 224, 224),
        "observation.images.cam_left_wrist": (3, 224, 224),
        "observation.images.cam_right_wrist": (3, 224, 224),
        "observation.state": (14,),
    },
    output_shapes={
        "action": (14,),  # 양팔 7-DoF each
    },

    # ACT 특화 설정
    chunk_size=100,          # Action chunk 크기
    latent_dim=32,           # CVAE latent 차원
    kl_weight=10.0,          # KL loss 가중치

    # Transformer 설정
    dim_model=512,           # Hidden dimension
    n_heads=8,               # Attention heads
    dim_feedforward=3200,    # FFN dimension
    n_encoder_layers=4,      # Encoder layers
    n_decoder_layers=1,      # Decoder layers (보통 1)
    dropout=0.1,

    # 학습 설정
    lr=1e-5,                 # Learning rate
    lr_backbone=1e-5,        # Backbone LR
    weight_decay=1e-4,

    # Vision 설정
    use_vip=False,           # VIP backbone 사용 여부
    pretrained_backbone=True,
)
```

### ALOHA 프리셋

```python
from lerobot.configs.policies import ACT_ALOHA_PRESET

# 사전 정의된 ALOHA용 설정
policy = ACTPolicy(ACT_ALOHA_PRESET)
```

---

## CLI 훈련

### 기본 훈련

```bash
python -m lerobot.scripts.lerobot_train \
  --policy-name act \
  --dataset-repo-id lerobot/aloha_insertion \
  --output-dir outputs/act_aloha \
  --num-train-steps 100000 \
  --batch-size 8 \
  --eval-freq 10000 \
  --save-freq 25000 \
  --wandb-enable true
```

### 설정 커스터마이징

```bash
python -m lerobot.scripts.lerobot_train \
  --policy-name act \
  --dataset-repo-id lerobot/aloha_insertion \
  --policy.chunk_size 50 \
  --policy.latent_dim 64 \
  --policy.kl_weight 5.0 \
  --policy.n_encoder_layers 6 \
  --policy.lr 5e-5
```

---

## 평가

### CLI 평가

```bash
python -m lerobot.scripts.lerobot_eval \
  --policy-repo-id lerobot/act_aloha_insertion \
  --env-name AlohaInsertion-v0 \
  --num-eval-episodes 50 \
  --save-video true
```

### Python 평가

```python
from lerobot.policies.act import ACTPolicy
from lerobot.envs.factory import make_env

# 모델 로드
policy = ACTPolicy.from_pretrained("lerobot/act_aloha_insertion")
policy.eval()
policy = policy.to("cuda")

# 환경 생성
env = make_env("AlohaInsertion-v0")

# 에피소드 실행
num_success = 0
for episode_idx in range(50):
    observation, info = env.reset(seed=episode_idx)
    done = False
    step = 0

    while not done and step < 500:
        # Action 생성
        with torch.no_grad():
            action = policy.select_action(observation)

        # 환경에서 실행
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

    if info.get("is_success", False):
        num_success += 1

    print(f"Episode {episode_idx}: {step} steps, Success: {info.get('is_success', False)}")

print(f"\nSuccess rate: {num_success}/50 = {num_success*100/50}%")
```

---

## 고급 사용법

### Temporal Ensemble

여러 chunk를 앙상블하여 안정성 향상:

```python
# Temporal ensemble 파라미터
temporal_ensemble_coeff = 0.01  # Exponential decay

# 이전 chunk 저장
previous_chunk = None

for t in range(episode_length):
    # 새 chunk 생성 (매 스텝)
    with torch.no_grad():
        current_chunk = policy.select_action(observation)

    if previous_chunk is not None:
        # Exponential smoothing
        action = (
            (1 - temporal_ensemble_coeff) * previous_chunk[t]
            + temporal_ensemble_coeff * current_chunk[0]
        )
    else:
        action = current_chunk[0]

    previous_chunk = current_chunk

    # 환경에서 실행
    observation, reward, done, info = env.step(action)
```

### VIP Backbone

Visual Imitation Pre-training (VIP) 사용:

```python
config = ACTConfig(
    ...
    use_vip=True,  # VIP ResNet 사용
    vip_checkpoint="path/to/vip_checkpoint.pth"
)

policy = ACTPolicy(config)
```

---

## 성능 팁

### 메모리 최적화

```python
# Gradient checkpointing으로 메모리 절약
config.use_gradient_checkpointing = True

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = policy(batch)
    loss = policy.compute_loss(output, batch)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 배치 크기 조정

```python
# ACT는 메모리를 많이 사용하므로 배치 크기 조절
# GPU 메모리에 따라:
# - RTX 3090 (24GB): batch_size=8-16
# - RTX 4090 (24GB): batch_size=8-16
# - A100 (40GB): batch_size=16-32
```

---

## 문제 해결

### KL Loss가 0으로 수렴

**증상**: KL loss가 매우 작아져서 CVAE가 작동하지 않음

**해결**:
```python
# KL weight 증가
config.kl_weight = 20.0  # 기본값 10.0에서 증가

# 또는 KL annealing 사용
kl_weight = min(epoch / warmup_epochs, 1.0) * config.kl_weight
```

### Action이 부자연스러움

**증상**: 로봇 움직임이 떨림

**해결**:
```python
# Chunk size 증가
config.chunk_size = 200  # 더 긴 시퀀스

# Temporal ensemble 사용 (위 참조)
```

### 훈련이 불안정

**증상**: Loss가 발산하거나 NaN

**해결**:
```python
# Learning rate 감소
config.lr = 5e-6
config.lr_backbone = 1e-6

# Gradient clipping 추가
torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
```

---

## 다음 단계

- [Diffusion 정책](22-Diffusion-정책.md) - 또 다른 강력한 정책
- [첫 번째 모델 훈련](../04-tutorials/60-첫-모델-훈련.md) - ACT 훈련 튜토리얼
- [실제 로봇으로 IL 수행](../04-tutorials/61-실제-로봇-IL.md) - ALOHA 사용법

---

**참조 파일:**
- [src/lerobot/policies/act/modeling_act.py](../../../src/lerobot/policies/act/modeling_act.py)
- [src/lerobot/policies/act/configuration_act.py](../../../src/lerobot/policies/act/configuration_act.py)
- [docs/source/act.mdx](../../../docs/source/act.mdx)
- [examples/tutorial/act/](../../../examples/tutorial/act/)

**원문 논문:**
- "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ALOHA)
