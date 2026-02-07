from lerobot.datasets.lerobot_dataset import LeRobotDataset

# PushT 데이터셋 로드
dataset = LeRobotDataset("lerobot/pusht")

# 데이터셋 정보 확인
print(f"에피소드 수: {dataset.num_episodes}")
print(f"총 프레임 수: {len(dataset)}")
print(f"특성: {dataset.meta.features}")

# 첫 번째 샘플 확인
sample = dataset[0]
print(f"관찰 키: {[k for k in sample.keys()]}")
print(f"행동 shape: {sample['action'].shape}")
