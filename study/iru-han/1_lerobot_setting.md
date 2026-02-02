# 1. 개발 환경 구축 (Installation)
처음에 miniconda 를 설치한걸 제거해주고, 레퍼런스에 나와있는대로 다시 설치했다.

## 1단계: 기존 Conda 완전 삭제하기

**설치 폴더 삭제**: 터미널에서 아래 명령어를 입력. (보통 miniconda3나 anaconda3 폴더로 설치되어 있음.)
```bash
rm -rf ~/miniconda3
rm -rf ~/anaconda3
```

**설정 파일 삭제**: 숨겨진 설정 파일들을 지우기.
```bash
rm -rf ~/.condarc ~/.conda ~/.continuum
```

**환경 변수 정리**:
```bash
nano ~/.zshrc
```
>>> conda initialize >>> 로 시작해서 <<< conda initialize <<<로 끝나는 덩어리를 찾아서 다 지우기.  
Ctrl + O (저장), Enter, Ctrl + X (종료) 순으로 누르기.
```bash
source ~/.zshrc
```

---

## 2단계: Miniforge 설치 (wget 대신 curl 사용)

다운로드:
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
```

설치 실행:
```bash
bash Miniforge3-$(uname)-$(uname -m).sh
```
엔터를 계속 누르시고, 중간에 yes라고 입력하라는 메시지가 나오면 모두 yes를 입력.

**터미널 재시작**: 설치가 끝나면 터미널을 완전히 껐다가 다시 키기. 이름 앞에 (base)가 뜨면 성공.

---

## 1-1. 시스템 의존성 설치
macOS에서 비디오 인코딩과 통신을 원활하게 하기 위해 Homebrew를 통해 필수 도구를 먼저 설치했다.
```bash
brew install cmake ffmpeg pkg-config
```

---

## 1-2. 가상환경 및 소스코드 설치

기존 가상환경이 있다면 삭제 (초기화)
```bash
conda deactivate
conda remove -n lerobot --all -y
```

파이썬 3.10으로 가상환경 생성 및 활성화
```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

소스코드 클론 및 이동
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
```

ffmpeg 설치
```bash
conda install ffmpeg -c conda-forge
```

LeRobot 및 필수 기능 설치
```bash
pip install 'lerobot[feetech,pusht]'
```

Tip:
```bash
pip install -e .
```

---

# 2. 하드웨어 연결 및 장치 인식

## 2-1. 시리얼 포트 확인
```bash
ls /dev/tty.usbmodem*
```

USB 포트를 한개씩 뽑으면서 누가 리더인지 체크했다.

### 🍎 맥북용 USB 포트 고정 가이드

Alias 등록:
```bash
nano ~/.zshrc
```
```bash
# LeRobot Port Aliases
export LEADER_PORT="/dev/tty.usbmodem장치고유번호"
export FOLLOWER_PORT="/dev/tty.usbmodem장치고유번호"
```
```bash
source ~/.zshrc
```

---

## 2-2. 텔레오퍼레이션 테스트

리더 팔 보정
```bash
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=$LEADER_PORT \
    --teleop.id=leader
```

팔로워 팔 보정
```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=$FOLLOWER_PORT \
    --robot.id=follower
```

---

# 3. 데이터셋 녹화 및 업로드 (Recording)

## 3-1. 허깅페이스 로그인
```bash
huggingface-cli login
```

```bash
export HF_USER="허깅페이스_아이디"
```

## 3-2. 데이터 녹화 명령어 (최종 성공 버전)
```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=$FOLLOWER_PORT \
    --robot.id=follower \
    --teleop.type=so101_leader \
    --teleop.port=$LEADER_PORT \
    --teleop.id=leader \
    --robot.cameras='{"top": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \
    --dataset.repo_id=$HF_USER/lego_pickup_test \
    --dataset.num_episodes=5 \
    --dataset.fps=30 \
    --dataset.single_task="Pick up the object with the gripper"
```

기존 데이터 폴더 삭제
```bash
rm -rf /Users/유저이름/.cache/huggingface/lerobot/허깅페이스_아이디/lego_pickup_test
```

FPS 수정 버전
```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=$FOLLOWER_PORT \
    --robot.id=follower \
    --teleop.type=so101_leader \
    --teleop.port=$LEADER_PORT \
    --teleop.id=leader \
    --robot.cameras='{"wrist": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 15}}' \
    --dataset.repo_id=$HF_USER/lego_pickup_test \
    --dataset.num_episodes=5 \
    --dataset.single_task="Pick up the lego block and put it in the box"
```

확인
```bash
python src/lerobot/scripts/lerobot_dataset_viz.py \
    --repo-id $HF_USER/lego_pickup_test \
    --episode-index 0
```

---

# 4. 주요 트러블슈팅 및 노하우

## 4-1. macOS 보안 설정 (Accessibility)
시스템 설정 > 개인정보 보호 및 보안 > 손쉬운 사용에서 터미널 앱 권한 부여

## 4-2. 카메라 인덱스 및 시야각 문제
Eye-to-Hand(Top-down) 방식 채택

## 4-3. FPS 및 인코딩 에러
actual_fps 기준 설정값 일치

---

# 5. 데이터 검수 및 시각화 (Visualization)

```bash
python src/lerobot/scripts/lerobot_dataset_viz.py \
    --repo-id $HF_USER/lego_pickup_test \
    --episode-index 0 \
    --display-compressed-images False
```

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/$FOLLOWER_PORT \
    --robot.id=follower \
    --teleop.type=so101_leader \
    --teleop.port=/dev/$LEADER_PORT \
    --teleop.id=leader \
    --robot.cameras='{"wrist": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' \
    --display_data=true
```

---

# 6. 결론 및 향후 계획

일단 텔레오퍼레이션후 데이터 수집 했는데 데이터 수집한 것 보니까 처음엔 맥북의 웹캠 데이터가 저장되었고, 두번째는 로봇 집게가 제대로 안찍혀서 로봇 위치와 카메라 세팅하느라 시간을 허비해버렸다.

일단 집게에 하나 전체적으로 보이는곳에 하나 달아야한다는데 위치를 너무 낮게 잡으면 로봇이 잘려서 찍혀서 위치가 고민이다. 또 집게쪽에 카메라를 어떻게 달아야할지도 고민이다. 일단 집게에 일반 usb 카메라 달려니까 도저히 안돼서 이미지와 같이 임시로 집게 위치에 카메라를 달았다. 하지만 일정 범위를 벗어나면 로봇집게가 짤려나와서 수정을 좀 해야할 것 같다. 또 카메라 한개 더 가져와서 전체적으로 보이는곳에 하나 더 달아야한다.(전체적으로 보이는 카메라는 세팅이 어렵지 않을거라고 생각한다) 오늘 저녁엔 카메라 잘 조정해서 꼭 제대로된 데이터 수집을 하고싶다.


## 고민 후 카메라 임시 해결
이제 카메라 하나 더 달고, 데이터를 수집하면 될 것 같다