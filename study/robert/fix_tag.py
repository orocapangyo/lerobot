from huggingface_hub import HfApi, whoami

# 1. 설정값 (정박한지 꼭 확인하세요!)
REPO_ID = "robert0631/pnk"
TAG_NAME = "v3.0"

def fix():
    hub_api = HfApi()
    
    try:
        # 로그인 상태 확인
        try:
            user_info = whoami()
            print(f"👤 현재 로그인된 계정: {user_info['name']}")
        except Exception:
            print("❌ 로그인 정보가 없습니다.")
            print("💡 해결법: 'python -m huggingface_hub cli login' 명령어를 먼저 실행하세요.")
            return
        
        print(f"🚀 '{REPO_ID}' 데이터셋에 '{TAG_NAME}' 태그를 추가합니다...")
        hub_api.create_tag(
            repo_id=REPO_ID, 
            tag=TAG_NAME, 
            repo_type="dataset",
            exist_ok=True
        )
        print("✅ 태그 추가 완료! 이제 모델 훈련을 다시 실행해 보세요.")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("\n--- 해결 가이드 ---")
        print("1. 'python -m huggingface_hub cli login' 명령어를 실행하세요.")
        print("2. 토큰 입력 시 반드시 'Write' 권한이 있는 토큰을 사용해야 합니다.")
        print(f"3. 허깅페이스 웹사이트에서 '{REPO_ID}' 데이터셋이 실제로 존재하는지 확인하세요.")

if __name__ == "__main__":
    fix()
