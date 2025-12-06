# LeRobotDataset 이해하기

## 개요

`LeRobotDataset`은 LeRobot의 핵심 데이터 구조로, 로봇 학습을 위한 에피소드 기반 데이터셋을 표준화된 형식으로 제공합니다.

## LeRobotDataset v3.0 포맷

### 주요 특징

1. **에피소드 기반 구조**: 각 데모가 독립적인 에피소드로 저장
2. **Delta Timestamps**: 시간적 컨텍스트를 쉽게 쿼리
3. **효율적 저장**: Parquet (메타데이터) + MP4 (비디오) + JSON (설정)
4. **HuggingFace Hub 통합**: 원클릭 다운로드 및 공유

### 데이터 구조

```
dataset_repo/
├── meta/
│   └── info.json                    # 데이터셋 메타정보
├── data/
│   ├── chunk-000/
│   │   ├── episode_000000.parquet   # 에피소드 0 데이터
│   │   ├── episode_000001.parquet
│   │   └── ...
│   └── chunk-001/
│       └── ...
├── videos/
│   ├── chunk-000/
│   │   ├── observation.image_000000.mp4  # 에피소드 0 비디오
│   │   ├── observation.image_000001.mp4
│   │   └── ...
│   └── chunk-001/
│       └── ...
└── stats/
    └── default.json                 # 정규화 통계
```

---

## 기본 사용법

### 데이터셋 로드

```python
from lerobot.datasets import LeRobotDataset

# HuggingFace Hub에서 로드
dataset = LeRobotDataset("lerobot/pusht")

print(f"총 프레임 수: {len(dataset)}")
print(f"에피소드 수: {dataset.num_episodes}")
print(f"FPS: {dataset.fps}")
print(f"로봇 타입: {dataset.robot_type}")
```

### 데이터 접근

```python
# 인덱스로 프레임 가져오기
frame = dataset[0]

print("데이터 키:", frame.keys())
# 출력: dict_keys(['observation.image', 'observation.state', 'action',
#                  'episode_index', 'frame_index', 'timestamp', 'next.reward', ...])

# 각 데이터 확인
print("이미지 shape:", frame["observation.image"].shape)  # (C, H, W)
print("상태 shape:", frame["observation.state"].shape)    # (state_dim,)
print("액션 shape:", frame["action"].shape)               # (action_dim,)
print("에피소드 인덱스:", frame["episode_index"])
print("프레임 인덱스:", frame["frame_index"])
```

### 배치 로드

```python
import torch
from torch.utils.data import DataLoader

# DataLoader로 배치 처리
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# 배치 반복
for batch in dataloader:
    images = batch["observation.image"]      # (B, C, H, W)
    states = batch["observation.state"]      # (B, state_dim)
    actions = batch["action"]                # (B, action_dim)

    # 모델 훈련 코드...
    break
```

---

## Delta Timestamps

Delta timestamps는 LeRobot의 핵심 기능으로, 시간적 컨텍스트를 쉽게 가져올 수 있게 합니다.

### 개념

```python
# delta_timestamps 예시
delta_timestamps = {
    "observation.image": [-1, 0],      # 이전 프레임과 현재 프레임
    "observation.state": [-1, 0],
    "action": [0, 1, 2, 3]             # 현재부터 3스텝 미래
}
```

### 사용 예제

```python
from lerobot.datasets import LeRobotDataset

# Delta timestamps 설정
dataset = LeRobotDataset(
    "lerobot/pusht",
    delta_timestamps={
        "observation.image": [-1, 0],    # 과거 1프레임, 현재
        "observation.state": [-2, -1, 0], # 과거 2프레임, 과거 1프레임, 현재
        "action": [0, 1, 2, 3, 4]        # 현재부터 미래 4프레임 (action chunking)
    }
)

# 데이터 가져오기
frame = dataset[100]

# 이미지는 2개 프레임 (과거 1, 현재)
print(frame["observation.image"].shape)  # (2, C, H, W)

# 상태는 3개 프레임
print(frame["observation.state"].shape)  # (3, state_dim)

# 액션은 5개 프레임 (action chunking)
print(frame["action"].shape)             # (5, action_dim)
```

### ACT 정책 예제

```python
# ACT는 action chunking을 사용
dataset = LeRobotDataset(
    "lerobot/aloha_insertion",
    delta_timestamps={
        "observation.image": [0],           # 현재 이미지만
        "observation.state": [0],           # 현재 상태만
        "action": list(range(100))          # 100 스텝 action chunk
    }
)

frame = dataset[0]
print(frame["action"].shape)  # (100, action_dim)
```

---

## 메타데이터

### LeRobotDatasetMetadata

```python
# 메타데이터 접근
meta = dataset.meta

print("데이터셋 ID:", meta.repo_id)
print("FPS:", meta.fps)
print("로봇 타입:", meta.robot_type)
print("총 에피소드:", meta.total_episodes)
print("총 프레임:", meta.total_frames)
print("총 작업:", meta.total_tasks)
```

### 에피소드 정보

```python
# 에피소드 데이터 인덱스
episode_data = dataset.episode_data_index

print("에피소드별 시작 인덱스:", episode_data["from"])
print("에피소드별 종료 인덱스:", episode_data["to"])
print("에피소드별 프레임 수:", episode_data["to"] - episode_data["from"])

# 특정 에피소드의 모든 프레임
episode_idx = 0
start_idx = episode_data["from"][episode_idx]
end_idx = episode_data["to"][episode_idx]

episode_frames = [dataset[i] for i in range(start_idx, end_idx)]
print(f"에피소드 {episode_idx}: {len(episode_frames)}개 프레임")
```

### 통계 정보

```python
# 정규화 통계
stats = dataset.stats

print("관측 통계 키:", stats["observation"].keys())
print("액션 통계 키:", stats["action"].keys())

# 액션 평균 및 표준편차
action_mean = stats["action"]["mean"]
action_std = stats["action"]["std"]

print("액션 평균:", action_mean)
print("액션 표준편차:", action_std)
```

---

## 데이터 구조 상세

### Observation (관측)

```python
frame = dataset[0]

# 이미지 관측
if "observation.image" in frame:
    img = frame["observation.image"]
    print("이미지 타입:", img.dtype)        # torch.float32
    print("이미지 범위:", img.min(), img.max())  # [0, 1]
    print("이미지 shape:", img.shape)       # (C, H, W) 또는 (T, C, H, W)

# 상태 관측 (proprioception)
if "observation.state" in frame:
    state = frame["observation.state"]
    print("상태 shape:", state.shape)
    print("상태 예시:", state)
    # 예: joint positions, velocities, end-effector pose 등
```

### Action (행동)

```python
# 액션 데이터
action = frame["action"]

print("액션 타입:", action.dtype)
print("액션 shape:", action.shape)
print("액션 예시:", action)

# 액션은 보통 다음 중 하나:
# - Joint positions (관절 위치)
# - Joint velocities (관절 속도)
# - End-effector pose (말단 장치 위치/자세)
# - Gripper state (그리퍼 상태)
```

### 추가 필드

```python
# 에피소드 및 프레임 인덱스
print("에피소드 인덱스:", frame["episode_index"])
print("프레임 인덱스:", frame["frame_index"])
print("타임스탬프:", frame["timestamp"])

# 다음 프레임 정보 (RL용)
if "next.reward" in frame:
    print("보상:", frame["next.reward"])
if "next.done" in frame:
    print("종료 여부:", frame["next.done"])
if "next.success" in frame:
    print("성공 여부:", frame["next.success"])
```

---

## 에피소드 필터링

### 특정 에피소드만 로드

```python
# 에피소드 0-9만 사용
dataset = LeRobotDataset(
    "lerobot/pusht",
    episodes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
)

print(f"필터링된 에피소드 수: {dataset.num_episodes}")
```

### 태스크별 필터링

```python
# 특정 태스크의 에피소드만
dataset = LeRobotDataset(
    "lerobot/aloha_insertion",
    task="insert_peg"  # 특정 태스크만
)
```

### 성공한 에피소드만

```python
# 메타데이터에서 성공한 에피소드 찾기
if hasattr(dataset.meta, "episodes"):
    successful_episodes = [
        ep_idx for ep_idx, ep_info in enumerate(dataset.meta.episodes)
        if ep_info.get("success", False)
    ]

    # 성공한 에피소드만으로 새 데이터셋 생성
    dataset = LeRobotDataset(
        "lerobot/pusht",
        episodes=successful_episodes
    )
```

---

## 다중 카메라

여러 카메라가 있는 경우:

```python
frame = dataset[0]

# 여러 카메라 이미지
if "observation.images.cam_high" in frame:
    cam_high = frame["observation.images.cam_high"]
    print("상단 카메라:", cam_high.shape)

if "observation.images.cam_wrist" in frame:
    cam_wrist = frame["observation.images.cam_wrist"]
    print("손목 카메라:", cam_wrist.shape)

# ALOHA 예시
if "observation.images.cam_left_wrist" in frame:
    left_wrist = frame["observation.images.cam_left_wrist"]
if "observation.images.cam_right_wrist" in frame:
    right_wrist = frame["observation.images.cam_right_wrist"]
```

---

## 비디오 처리

### 비디오 백엔드

LeRobot은 효율적인 비디오 처리를 위해 여러 백엔드를 지원합니다:

```python
# PyAV 사용 (기본값, 가장 빠름)
dataset = LeRobotDataset(
    "lerobot/pusht",
    video_backend="pyav"
)

# OpenCV 사용
dataset = LeRobotDataset(
    "lerobot/pusht",
    video_backend="opencv"
)

# 비디오 디코딩 없이 로드 (테스트용)
dataset = LeRobotDataset(
    "lerobot/pusht",
    video_backend=None  # 이미지는 None으로 반환
)
```

### 이미지 변환

```python
from lerobot.datasets.transforms import ImageTransforms

# 이미지 변환 설정
image_transforms = ImageTransforms(
    brightness=0.1,
    contrast=0.1,
    saturation=0.1,
    hue=0.05,
    sharpness=0.1
)

dataset = LeRobotDataset(
    "lerobot/pusht",
    image_transforms=image_transforms
)
```

---

## 캐싱

### 메모리 캐싱

대용량 데이터셋의 경우 캐싱으로 성능 향상:

```python
# 첫 로드시 캐시 생성
dataset = LeRobotDataset(
    "lerobot/pusht",
    root="/path/to/cache"  # 캐시 디렉토리 지정
)

# 두 번째 로드시 캐시 사용 (훨씬 빠름)
dataset = LeRobotDataset(
    "lerobot/pusht",
    root="/path/to/cache"
)
```

### 비디오 캐싱

```python
# 비디오를 미리 디코딩하여 메모리에 캐시
# 주의: 메모리 사용량이 매우 클 수 있음
dataset = LeRobotDataset(
    "lerobot/pusht",
    cache_videos=True  # 실험적 기능
)
```

---

## 데이터셋 정보 출력

### 요약 정보

```python
from lerobot.datasets import LeRobotDataset

dataset = LeRobotDataset("lerobot/pusht")

# 데이터셋 요약
print(f"""
데이터셋 정보:
- ID: {dataset.meta.repo_id}
- 총 에피소드: {dataset.num_episodes}
- 총 프레임: {len(dataset)}
- FPS: {dataset.fps}
- 로봇: {dataset.robot_type}
- 데이터 키: {list(dataset.hf_dataset.features.keys())}
""")

# 첫 프레임 shape 정보
frame = dataset[0]
print("\n데이터 Shape:")
for key, value in frame.items():
    if hasattr(value, 'shape'):
        print(f"  {key}: {value.shape}")
    else:
        print(f"  {key}: {type(value)}")
```

---

## 데이터 검증

### 데이터 무결성 확인

```python
import torch

def validate_dataset(dataset, num_samples=100):
    """데이터셋 검증"""
    print(f"데이터셋 검증 중... (샘플: {num_samples})")

    for i in range(min(num_samples, len(dataset))):
        frame = dataset[i]

        # 이미지 검증
        if "observation.image" in frame:
            img = frame["observation.image"]
            assert img.min() >= 0 and img.max() <= 1, "이미지 범위 오류"
            assert not torch.isnan(img).any(), "이미지에 NaN 존재"

        # 액션 검증
        action = frame["action"]
        assert not torch.isnan(action).any(), "액션에 NaN 존재"
        assert not torch.isinf(action).any(), "액션에 Inf 존재"

        if i % 20 == 0:
            print(f"  {i}/{num_samples} 검증 완료")

    print("✓ 데이터셋 검증 성공!")

validate_dataset(dataset)
```

---

## 고급 사용법

### 커스텀 샘플러

```python
from lerobot.datasets.sampler import EpisodeAwareSampler

# 에피소드 경계를 고려하는 샘플러
sampler = EpisodeAwareSampler(
    episode_data_index=dataset.episode_data_index,
    shuffle=True,
    drop_last=False
)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler
)
```

### 멀티 데이터셋

```python
from lerobot.datasets import MultiLeRobotDataset

# 여러 데이터셋 결합
multi_dataset = MultiLeRobotDataset([
    "lerobot/pusht",
    "lerobot/pusht_image_aug",
    "my_username/my_pusht_data"
])

print(f"총 프레임 수: {len(multi_dataset)}")
```

---

## 다음 단계

- [데이터셋 사용법](11-데이터셋-사용법.md) - 실전 활용 방법
- [데이터셋 생성과 관리](12-데이터셋-생성-관리.md) - 자신만의 데이터셋 만들기
- [데이터 변환과 증강](13-데이터-변환-증강.md) - 데이터 전처리

---

**참조 파일:**
- [src/lerobot/datasets/lerobot_dataset.py](../../../src/lerobot/datasets/lerobot_dataset.py) - 메인 클래스 구현
- [docs/source/lerobot-dataset-v3.mdx](../../../docs/source/lerobot-dataset-v3.mdx) - 공식 문서
- [examples/dataset/load_lerobot_dataset.py](../../../examples/dataset/load_lerobot_dataset.py) - 예제 코드
