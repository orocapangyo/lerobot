# X-VLA: LeRobot 통합 가이드

이 문서는 [X-VLA](https://thu-air-dream.github.io/X-VLA/) (Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model)를 현재 LeRobot 코드베이스에서 사용하는 방법을 설명합니다.

---

## 1. 개요 (Overview)

**X-VLA**는 범용적인 로봇 기초 모델(Robot Foundation Model)을 지향하는 프레임워크입니다.
- **Soft Prompts**: 로봇의 형태와 환경의 차이를 소수의 학습 가능한 임베딩(Soft Prompts)으로 흡수하여, 하나의 모델로 다양한 로봇(Cross-Embodiment)을 제어할 수 있습니다.
- **Scalability**: 순수 트랜스포머 기반의 Flow-matching 모델로, 0.9B 파라미터 규모에서도 효율적인 학습과 전이가 가능합니다.
- **High Performance**: 적은 양의 데이터로도 새로운 로봇에 빠르게 적응(Phase II Adaptation)하며, 복잡한 조작 작업(예: 옷 개기)에서 뛰어난 성능을 보입니다.

---

## 2. 설치 방법 (Installation)

LeRobot이 설치된 환경에서 X-VLA 관련 의존성을 추가로 설치해야 합니다.

```bash
# 로컬 소스 설치 모드인 경우
pip install -e .[xvla]

# (향후 릴리즈 시)
# pip install lerobot[xvla]
```

---

## 3. 기본 사용법 (Quick Start)

### 정책 타입 설정
LeRobot 프로젝트 설정에서 정책 타입을 `xvla`로 지정합니다.
```bash
policy.type=xvla
```

### 사전 학습된 체크포인트 평가
LIBERO 벤치마크를 예시로 한 평가 명령어입니다.

```bash
lerobot-eval \
  --policy.path="lerobot/xvla-libero" \
  --env.type=libero \
  --env.task=libero_spatial,libero_goal,libero_10 \
  --env.control_mode=absolute \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --env.episode_length=800 \
  --seed=142
```

### 실제 로봇에 적용 (Real-World Deployment)
학습된 모델을 실제 로봇에서 실행하려면 `lerobot-record` 명령어를 사용합니다. VLA 모델이므로 어떤 동작을 수행할지 **언어 지시문(Instruction)**을 함께 전달해야 합니다.

```bash
lerobot-record \
  --robot.type=so100_follower \
  --policy.path=outputs/xvla_training/checkpoints/last/pretrained_model \
  --dataset.single_task="집게로 컵을 들어서 오른쪽으로 옮겨줘" \
  --robot.cameras='{top: {type: opencv, index_or_path: 0}}'
```

- **`--policy.path`**: 학습 완료 후 저장된 체크포인트 경로를 지정합니다.
- **`--dataset.single_task`**: 로봇에게 내릴 **한글/영어 명령**을 입력합니다. X-VLA는 이 텍스트를 이해하여 동작을 생성합니다.
- **`---teleop.port` 제외**: 학습된 정책으로만 움직이길 원한다면 텔레옵(조종기) 포트 설정은 제외하면 됩니다.

---

## 4. X-VLA 학습 (Training)

새로운 로봇이나 작업에 대해 X-VLA를 미세조정(Fine-tuning)할 때 권장되는 설정입니다.

```bash
lerobot-train \
  --dataset.repo_id=YOUR_DATASET \
  --output_dir=./outputs/xvla_training \
  --job_name=xvla_training \
  --policy.path="lerobot/xvla-base" \
  --policy.repo_id="HF_USER/xvla-your-robot" \
  --policy.dtype=bfloat16 \
  --policy.action_mode=auto \
  --steps=20000 \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=false \
  --policy.freeze_language_encoder=false \
  --policy.train_policy_transformer=true \
  --policy.train_soft_prompts=true
```

### 권장 사항 (Best Practices)
- **VLM 고정 해제**: 새로운 형태의 로봇에 학습할 때는 시각(Vision) 및 언어(Language) 인코더의 가중치 고정을 해제하는 것이 성능이 더 좋습니다.
- **학습률 설정**: 기본적으로 VLM의 학습률은 다른 컴포넌트의 1/10로 설정되어 안정적인 학습을 돕습니다.
- **메모리(OOM) 방지**: `policy.dtype=bfloat16` 설정을 적극 권장합니다.

### 권장 학습 단계 (Training Steps)
X-VLA는 사전 학습된 파라미터가 0.9B에 달하는 대규모 모델이므로, 처음부터 모든 것을 배우는 ACT와 달리 적은 학습 단계로도 새로운 로봇에 적응할 수 있습니다.

- **권장 단계 (Fine-tuning)**: **20,000 ~ 30,000 스텝**
    - `lerobot-train` 가이드에서는 기본적으로 **20,000 스텝**을 권장합니다.
    - 모델 내부 스케줄러 기본 커스텀 값은 **30,000 스텝**입니다.
- **간단한 작업 (Small Adaptation)**: **3,000 스텝**
    - 양팔 로봇의 물체 전달과 같이 단순한 동작 적응의 경우 3,000 스텝 정도로도 충분할 수 있습니다.
- **비교 (vs ACT)**: ACT가 보통 100,000 스텝 이상을 요구하는 것에 비해 X-VLA는 약 **1/5 ~ 1/3 수준**의 학습만으로도 효율적인 동작이 가능합니다.

---

## 5. 핵심 개념 (Core Concepts)

### Action Modes
X-VLA는 **Action Registry** 시스템을 통해 다양한 로봇의 액션 공간을 처리합니다.
- `auto` (권장): 데이터셋의 액션 차원을 자동으로 감지하고 처리합니다. 대부분의 새로운 로봇에 적합합니다.
- `ee6d`: 엔드 이펙터(xyz, 6D 회전, 그리퍼) 기반 제어 (20차원).
- `joint`: 관절 공간 제어 (14차원).
- `so101_bimanual`: SO100/101 양팔 로봇 전용 모드.

### Domain IDs
로봇의 설정(기구학적 특징, 카메라 각도 등)을 구분하기 위한 식별자입니다.
- 학습 시 `domain_id`를 지정하여 특정 환경의 특성을 Soft Prompt에 저장할 수 있습니다.
- `lerobot/xvla-base`는 Bridge(0), RT1(1), Calvin(2), Libero(3) 등 다양한 도메인 ID가 정의되어 있습니다.

---

## 6. 주요 체크포인트 (Available Models)

Hugging Face Hub에서 다음 모델들을 사용할 수 있습니다.
- **Base Model**: [lerobot/xvla-base](https://huggingface.co/lerobot/xvla-base) (0.9B 기반 모델)
- **Simulation**: `xvla-libero`, `xvla-widowx`
- **Real-World**: `xvla-folding` (옷 개기 숙련형 모델)

## [실습] 양팔 로봇 + 카메라 2개 수건 개기

양팔 로봇(SO100/101 Bimanual)과 2개의 카메라(예: Top, Front) 환경에 최적화된 명령어입니다.

### 1단계: 데이터 수집 (Bimanual + Dual-Cam)
양팔의 동작과 두 카메라 시점을 모두 기록해야 합니다.
- **명령어**:
  ```bash
  lerobot-record \
    --robot.type=bi_so_follower \
    --robot.cameras='{top: {type: opencv, index_or_path: 0}, side: {type: opencv, index_or_path: 1}}' \
    --dataset.repo_id=YOUR_ID/bimanual_towel_fold
  ```
  *(팁: 양손을 조화롭게 사용하여 수건을 펴고 접는 과정을 100회 정도 기록하세요.)*

### 2단계: 학습 (2-Camera & Bimanual Config)
X-VLA가 두 개의 카메라 입력을 이해하고 양팔을 제어하도록 설정합니다.
- **명령어**:
  ```bash
  lerobot-train \
    --dataset.repo_id=YOUR_ID/bimanual_towel_fold \
    --policy.path="lerobot/xvla-base" \
    --policy.action_mode=auto \
    --policy.num_image_views=2 \
    --policy.dtype=bfloat16 \
    --steps=20000
  ```
  - **`--policy.num_image_views=2`**: 두 카메라 영상을 모두 학습에 사용합니다.
  - **`--policy.action_mode=auto`**: 양팔 로봇의 액션 차원을 자동으로 감지하여 최적화합니다.

### 3단계: 적용 (Inference)
학습된 모델로 로봇을 구동합니다.
- **명령어**:
  ```bash
  lerobot-record \
    --robot.type=bi_so_follower \
    --policy.path=outputs/xvla_training/checkpoints/last/pretrained_model \
    --dataset.single_task="양팔로 수건을 정성껏 개어줘" \
    --robot.cameras='{top: {type: opencv, index_or_path: 0}, side: {type: opencv, index_or_path: 1}}'
  ```

### 4단계: 실력 향상 (DAgger)
- 수건이 겹치거나 꼬인 특수한 상황에서 자꾸 실패한다면, 그 상황에서 수습하는 양팔 동작을 30~50개 추가 수집하여 재학습하세요. X-VLA는 이런 '에러 대처' 데이터를 아주 잘 학습합니다.

---

## 7. 문제 해결 (Troubleshooting)

- **액션 차원 불일치**: `policy.action_mode=auto`를 사용하고 있는지 확인하세요.
- **성능 저하**: `train_soft_prompts=true`가 설정되어 있는지, 그리고 데이터 전처리 과정에서 ImageNet 정규화가 올바르게 적용되었는지 확인이 필요합니다.
- **메모리 부족**: `batch_size`를 줄이거나 `chunk_size`를 조정(예: 32 -> 16)해 보세요.

---

## 참고 자료
- [X-VLA Paper (arXiv)](https://arxiv.org/pdf/2510.10274)
- [LeRobot XVLA Documentation](https://huggingface.co/docs/lerobot/xvla) (내부 문서: `docs/source/xvla.mdx`)
