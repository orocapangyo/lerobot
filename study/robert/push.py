from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ---------------------------------------------------------
# 설정값 (본인의 환경에 맞게 수정하세요)
# ---------------------------------------------------------
REPO_ID = "YOUR_HF_ID/bimanual_towel_fold"  # 허깅페이스에 생성될 데이터셋 이름
LOCAL_ROOT = "D:/lerobot_data"             # 데이터가 저장된 로컬 폴더 경로 (상위 폴더)

def push():
    print(f"🚀 '{REPO_ID}' 데이터셋 업로드를 시작합니다 (경로: {LOCAL_ROOT})...")
    
    try:
        # 데이터셋 로드 및 업로드
        # push_to_hub()는 데이터셋 카드(README.md)를 자동으로 생성합니다.
        dataset = LeRobotDataset(REPO_ID, root=LOCAL_ROOT)
        dataset.push_to_hub()
        
        print("\n" + "="*50)
        print("✅ 업로드 완료!")
        print(f"🔗 링크: https://huggingface.co/datasets/{REPO_ID}")
        print("="*50)
        
    except Exception as e:
        print("\n" + "!"*50)
        print(f"❌ 오류 발생: {e}")
        print("먼저 'huggingface-cli login'이 되어 있는지 확인하세요.")
        print("!"*50)

if __name__ == "__main__":
    push()
