## 설치 진행 상황 (로봇 하드웨어 없음)
- 범위: Git clone + Python 패키지 설치(editable)까지만 수행
- 미수행: 로봇 연결/캘리브레이션/실제 구동 관련 단계

---

## 0. 개요
- 목적: LeRobot 설치 및 기본 실행 확인
- 환경:
  - OS(Ubuntu24.04)
- 참고 영상: https://youtu.be/ElZvzKRt9g8?si=M6vpU5S8oP5tWtQ7

---

## 1. 사전 준비
### 1.1 필수 설치/확인
- git 설치
```bash
sudo apt update
sudo apt install git
git --version
```

### 1.2 miniforge 설치
```bash
sudo apt install curl
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
터미널을 껐다가 켰을 때, 앞쪽에 (base)가 달리면 성공입니다.

## 2. conda 설치 (유튜브 허깅페이스 installation 참고)
```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg=7.1.1 -c conda-forge
```
설치 후 해당 버전이 뜨면 성공
```bash
conda --version
which conda
```


## 3. 르로봇 git 설치
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[feetech]"
```
설치 가이드에는 [feetech]를 설치하지 않지만, so-arm 101을 사용할 것이고, 해당 모터가 feetech 사 제품이므로 해당 명령어를 진행했다.

로봇을 산 후, youtube 7:00 부터 다시 시작할 예정.
    
