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
- **체크포인트 재개**: 학습 중 중단되었다면 `--resume=true`와 `-c [체크포인트경로]`를 사용하여 이어서 학습할 수 있습니다.

### 권장 학습 단계 (Training Steps)
X-VLA는 사전 학습된 파라미터가 0.9B에 달하는 대규모 모델이므로, 처음부터 모든 것을 배우는 ACT와 달리 적은 학습 단계로도 새로운 로봇에 적응할 수 있습니다.

- **권장 단계 (Fine-tuning)**: **20,000 ~ 30,000 스텝**
    - `lerobot-train` 가이드에서는 기본적으로 **20,000 스텝**을 권장합니다.
    - 모델 내부 스케줄러 기본 커스텀 값은 **30,000 스텝**입니다.
- **간단한 작업 (Small Adaptation)**: **3,000 스텝**
    - 양팔 로봇의 물체 전달과 같이 단순한 동작 적응의 경우 3,000 스텝 정도로도 충분할 수 있습니다.
- **비교 (vs ACT)**: ACT가 보통 100,000 스텝 이상을 요구하는 것에 비해 X-VLA는 약 **1/5 ~ 1/3 수준**의 학습만으로도 효율적인 동작이 가능합니다.

---

## 5. 데이터셋 수집 및 관리 (Dataset Management)

새로운 작업을 학습시키기 위해 데이터셋을 **추가(기록)**하거나 상태를 확인할 때 사용하는 명령어들입니다.

### 데이터 기록 (Recording)
로컬에 새로운 데이터셋을 생성하고 로봇의 동작을 기록합니다.
```bash
lerobot-record \
  --robot.type=so100_follower \
  --teleop.type=so100_leader \
  --dataset.repo_id=YOUR_ID/new_task_name \
  --dataset.root=D:/lerobot_data \
  --dataset.single_task="작업 설명 입력" \
  --robot.cameras='{"top": {"type": "opencv", "index_or_path": 0}}'
```

> [!TIP]
> **데이터셋 이어서 기록하기 (Resume/Append)**
> - **데이터셋 하나로 묶기**: 이전에 녹화했던 `--dataset.repo_id`와 `--dataset.root`를 동일하게 지정하고 다시 실행하면, 기존 데이터 뒤에 새로운 에피소드로 이어서 저장됩니다.
> - **태스크 명(`task`) vs 데이터셋 명(`repo_id`)**:
>   - `repo_id`는 데이터를 담는 **상자(폴더)** 이름입니다.
>   - `task`는 각 에피소드에 붙는 **동작 설명(라벨)**입니다.
>   - X-VLA는 하나의 데이터셋(`repo_id`) 안에 여러 종류의 태스크(`task`)가 섞여 있어도 이를 언어 명령어로 구분하여 학습하고 수행할 수 있습니다.

### 데이터셋 정보 확인
설치된 데이터셋의 에피소드 수, 프레임 수, 통계 정보를 확인합니다.
```bash
# 로컬 데이터셋 정보 확인
lerobot-info --repo_id YOUR_ID/new_task_name --root D:/lerobot_data
```

### 데이터셋 시각화 (Visualization)
기록된 이미지가 올바른지, 동작이 잘 저장되었는지 GUI로 확인합니다.
```bash
lerobot-dataset-viz --repo_id YOUR_ID/new_task_name --root D:/lerobot_data
```

---

## 6. 핵심 개념 (Core Concepts)

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

## 7. 주요 체크포인트 (Available Models)

Hugging Face Hub에서 다음 모델들을 사용할 수 있습니다.
- **Base Model**: [lerobot/xvla-base](https://huggingface.co/lerobot/xvla-base) (0.9B 기반 모델)
- **Simulation**: `xvla-libero`, `xvla-widowx`
- **Real-World**: `xvla-folding` (옷 개기 숙련형 모델)

## 8. [실습] 양팔 로봇 + 카메라 2개 수건 개기

양팔 로봇(SO100/101 Bimanual)과 2개의 카메라(예: Top, Front) 환경에 최적화된 명령어입니다.

### [참고] 양팔 로봇 텔레오퍼레이션 (조종 방법)
수건 개기 데이터를 수집하기 위해 리더(Leader) 로봇으로 팔로워(Follower) 로봇을 조종하는 방법입니다.

- **단순 조종 (테스트용)**:
  ```bash
  lerobot-teleoperate \
    --robot.type=bi_so_follower \
    --robot.left_arm_config.port=/dev/tty.follower_left \
    --robot.right_arm_config.port=/dev/tty.follower_right \
    --teleop.type=bi_so_leader \
    --teleop.left_arm_config.port=/dev/tty.leader_left \
    --teleop.right_arm_config.port=/dev/tty.leader_right \
    --display_data=true
  ```

- **데이터 기록 시 (1단계에 적용)**:
  `lerobot-record` 명령에 `--teleop` 설정을 추가하면 조종과 동시에 기록이 가능합니다.
  ```bash
  lerobot-record \
    --robot.type=bi_so_follower \
    --teleop.type=bi_so_leader \
    --robot.cameras='{top: {type: opencv, index_or_path: 0}}' \
    --dataset.repo_id=YOUR_ID/bimanual_towel_fold
  ```

### 1단계: 데이터 수집 (Bimanual + Dual-Cam)
양팔의 동작과 두 카메라 시점을 모두 기록해야 합니다.
- **명령어**:
- **Windows (PowerShell)**:
  ```bash
  lerobot-record \
    --robot.type=bi_so_follower \
    --robot.cameras='{"top": {"type": "opencv", "index_or_path": 0}, "front": {"type": "opencv", "index_or_path": 1}}' \
    --dataset.repo_id=bimanual_towel_fold \
    --dataset.root=D:/lerobot_data
  ```

- **Linux (Bash/Zsh)**:
  ```bash
  lerobot-record \
    --robot.type=bi_so_follower \
    --robot.cameras='{"top": {"type": "opencv", "index_or_path": 0}, "front": {"type": "opencv", "index_or_path": 1}}' \
    --dataset.repo_id=bimanual_towel_fold \
    --dataset.root=~/lerobot_data
  ```
  *(팁: 로컬 데이터를 사용할 때는 `--dataset.repo_id`로 폴더 이름을, `--dataset.root`로 상위 폴더 경로를 지정하세요.)*

> [!TIP]
> **카메라 설정 주의사항**: `--robot.cameras`는 JSON 형식의 문자열이어야 합니다. 윈도우 환경에서는 전체를 `'`로 감싸고 내부 값들을 `"`로 감싸는 형식이 가장 안전합니다. 리눅스에서도 동일한 따옴표 규칙이 잘 작동합니다.

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

## 8. 실전 적용 및 추론 (Deployment & Inference)

학습된 X-VLA 모델을 실제 로봇에서 실행할 때는 **`lerobot-record`** 명령어를 사용합니다. LeRobot은 "모델 추론"과 "로봇 제어 루프"가 이 명령어 하나에 통합되어 있습니다.

### 핵심 개념: 왜 lerobot-record인가요?
- **평상시 (데이터 수집)**: 리더 로봇의 입력을 팔로워 로봇에게 전달하고 저장합니다.
- **추론 시 (Policy Inference)**: `--policy.path`가 지정되면, 리더 로봇 대신 **모델(Policy)의 출력값**을 팔로워 로봇에게 전달합니다.

### 추론 명령어 예시 (Single Task)
로봇에게 하나의 구체적인 명령을 내려 동작시키는 가장 기본적인 방법입니다.

```bash
lerobot-record \
  --robot.type=so100_follower \
  --policy.path=outputs/xvla_training/checkpoints/last/pretrained_model \
  --dataset.single_task="집게로 빨간 컵을 들어서 바구니에 넣어줘" \
  --robot.cameras='{"top": {"type": "opencv", "index_or_path": 0}}' \
  --device=cuda
```

- **`--policy.path`**: 학습 완료 후 저장된 체크포인트 폴더 경로입니다.
- **`--dataset.single_task`**: 로봇이 수행할 **언어 지시문**입니다. X-VLA는 이 텍스트를 분석하여 적절한 동작을 생성합니다.
- **`--device=cuda`**: GPU를 사용하여 실시간 추론 속도를 높입니다.

### [실습] 양팔 로봇 + 카메라 2개 수건 개기
양팔 로봇(SO100/101 Bimanual)과 2개의 카메라(예: Top, Front) 환경에서의 추론 사례입니다.

```bash
lerobot-record \
  --robot.type=bi_so_follower \
  --policy.path=outputs/xvla_bimanual/checkpoints/last/pretrained_model \
  --dataset.single_task="양팔로 수건을 정성껏 개어줘" \
  --robot.cameras='{"top": {"type": "opencv", "index_or_path": 0}, "front": {"type": "opencv", "index_or_path": 1}}' \
  --device=cuda
```

### 팁: DAgger (실력 향상시키기)
만약 모델이 특정 구간에서 자꾸 실수한다면, `lerobot-record` 실행 시 `--teleop.type` 설정을 함께 넣어 **사람이 개입**할 수 있게 하세요. 실패하는 순간 사람이 개입하여 올바른 동작을 보여주면, 그 데이터가 추가로 저장되어 다음 학습 시 더 똑똑해집니다.

---

## 9. 데이터셋 허깅페이스 업로드 (Dataset Upload)

기록된 데이터를 허깅페이스 허브(Hugging Face Hub)에 올리는 방법은 크게 두 가지가 있습니다.

### 방법 1: 기록 시 자동 업로드
`lerobot-record` 실행 시 `--dataset.push_to_hub=true` 옵션을 추가하면, 기록이 끝난 후 자동으로 업로드됩니다.

```bash
lerobot-record \
  ... \
  --dataset.push_to_hub=true \
  --dataset.repo_id=YOUR_ID/dataset_name
```

### 방법 2: 로컬 데이터를 수동으로 업로드 (권장)
이미 로컬에 저장된 데이터를 나중에 올리고 싶을 때 사용하는 가장 확실한 방법입니다. 간단한 파이썬 스크립트를 작성하여 실행하세요.

**`push_dataset.py`** (작성 및 실행):
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 설정값
REPO_ID = "YOUR_ID/bimanual_towel_fold"
LOCAL_ROOT = "D:/lerobot_data" # 또는 리눅스의 경우 "~/lerobot_data"

# 데이터셋 로드 및 업로드
dataset = LeRobotDataset(REPO_ID, root=LOCAL_ROOT)
dataset.push_to_hub()

print(f"✅ 업로드 완료: https://huggingface.co/datasets/{REPO_ID}")
```

**실행 방법**:
```bash
python push_dataset.py
```

### 업로드 확인 사항
*   **Hugging Face 로그인**: 미리 `huggingface-cli login` 명령어로 로그인이 되어 있어야 합니다.
*   **데이터셋 카드**: `push_to_hub()`를 사용하면 데이터셋의 통계 정보와 메타데이터가 포함된 데이터셋 카드가 자동으로 생성되어 시각적으로 확인하기 편리해집니다.

---

## 10. 문제 해결 (Troubleshooting)

- **액션 차원 불일치**: `policy.action_mode=auto`를 사용하고 있는지 확인하세요.
- **성능 저하**: `train_soft_prompts=true`가 설정되어 있는지, 그리고 데이터 전처리 과정에서 ImageNet 정규화가 올바르게 적용되었는지 확인이 필요합니다.
- **메모리 부족**: `batch_size`를 줄이거나 `chunk_size`를 조정(예: 32 -> 16)해 보세요.

---

## 참고 자료
- [X-VLA Paper (arXiv)](https://arxiv.org/pdf/2510.10274)
- [LeRobot XVLA Documentation](https://huggingface.co/docs/lerobot/xvla) (내부 문서: `docs/source/xvla.mdx`)
