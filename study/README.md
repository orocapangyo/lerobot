# LeRobot 8주 스터디 커리큘럼

LeRobot 소스코드 기반 체계적 학습 과정

---

## 📚 커리큘럼 개요

이 커리큘럼은 LeRobot 프로젝트의 실제 소스코드를 기반으로 구성되었습니다.
총 8주 동안 기초부터 고급 주제까지 단계적으로 학습하며, 이론과 실습을 병행합니다.

**학습 목표:**
- LeRobot의 핵심 개념과 아키텍처 이해
- 데이터셋 처리 및 정책 학습 파이프라인 구축
- 시뮬레이션 및 실제 로봇에 정책 배포
- 최신 로봇 학습 알고리즘 구현 및 활용

---

## Week 1: LeRobot 기초 및 데이터셋 처리

### 학습 목표
- LeRobot 프로젝트 구조 이해
- LeRobotDataset 포맷 학습
- 데이터셋 로드 및 시각화

### 이론
- **LeRobot 개요**
  - 프로젝트 구조: `src/lerobot/` 주요 모듈
  - HuggingFace Hub 통합
  - 지원되는 로봇 및 환경

- **LeRobotDataset v3.0 포맷**
  - 에피소드 기반 데이터 구조
  - Delta timestamps를 이용한 시간적 쿼리
  - 관찰(observation), 행동(action), 상태(state) 데이터
  - 비디오 인코딩/디코딩

### 실습 코드
1. **설치 및 환경 설정**
   ```bash
   pip install lerobot
   ```

2. **데이터셋 로드** - [examples/dataset/load_lerobot_dataset.py](../examples/dataset/load_lerobot_dataset.py)
   ```python
   from lerobot.datasets import LeRobotDataset

   # HuggingFace Hub에서 데이터셋 로드
   dataset = LeRobotDataset("lerobot/pusht")
   ```

3. **데이터셋 구조 분석** - [lerobot/datasets/lerobot_dataset.py](../src/lerobot/datasets/lerobot_dataset.py)
   - `LeRobotDataset` 클래스 (2,074 lines)
   - Delta timestamps 메커니즘
   - Episode indexing

4. **데이터 시각화** - [scripts/lerobot_dataset_viz.py](../src/lerobot/scripts/lerobot_dataset_viz.py)
   ```bash
   python -m lerobot.scripts.lerobot_dataset_viz \
     --repo-id lerobot/pusht \
     --episode-index 0
   ```

### 핵심 소스 파일
- [lerobot/datasets/lerobot_dataset.py](../src/lerobot/datasets/lerobot_dataset.py) - 메인 데이터셋 클래스
- [lerobot/datasets/utils.py](../src/lerobot/datasets/utils.py) - 유틸리티 함수
- [lerobot/datasets/video_utils.py](../src/lerobot/datasets/video_utils.py) - 비디오 처리
- [lerobot/datasets/transforms.py](../src/lerobot/datasets/transforms.py) - 데이터 증강

### 과제
1. `lerobot/pusht` 데이터셋을 로드하고 첫 번째 에피소드 시각화
2. Delta timestamps를 사용하여 시간적 컨텍스트 추출 (t-1, t, t+1)
3. 비디오 프레임과 상태 데이터의 동기화 확인

---

## Week 2: 정책(Policy) 기초 - ACT & Diffusion

### 학습 목표
- 정책(Policy) 아키텍처 이해
- ACT 및 Diffusion Policy 구현 분석
- Configuration 시스템 학습

### 이론
- **정책 아키텍처 개요**
  - Configuration, Modeling, Processor 구조
  - 사전 처리(preprocessing)와 후처리(postprocessing)
  - HuggingFace 스타일 구현

- **ACT (Action Chunking Transformer)**
  - 논문: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
  - CVAE + Transformer 아키텍처
  - Action chunking 메커니즘

- **Diffusion Policy**
  - 논문: Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
  - Diffusion 기반 행동 생성
  - Noise scheduling

### 실습 코드
1. **ACT 정책 사용** - [examples/tutorial/act/act_using_example.py](../examples/tutorial/act/act_using_example.py)
   ```python
   from lerobot.policies.act import ACTPolicy

   policy = ACTPolicy.from_pretrained("lerobot/act_aloha_insertion")
   action = policy.select_action(observation)
   ```

2. **ACT 정책 구조 분석** - [lerobot/policies/act/](../src/lerobot/policies/act/)
   - [configuration_act.py](../src/lerobot/policies/act/configuration_act.py) - 설정 클래스
   - [modeling_act.py](../src/lerobot/policies/act/modeling_act.py) - 모델 구현
   - [processor_act.py](../src/lerobot/policies/act/processor_act.py) - 전처리/후처리

3. **Diffusion 정책 사용** - [examples/tutorial/diffusion/diffusion_using_example.py](../examples/tutorial/diffusion/diffusion_using_example.py)
   ```python
   from lerobot.policies.diffusion import DiffusionPolicy

   policy = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht")
   action = policy.select_action(observation)
   ```

4. **Configuration 시스템** - [lerobot/configs/](../src/lerobot/configs/)
   - Draccus 기반 dataclass 설정
   - CLI 인자 파싱

### 핵심 소스 파일
- [lerobot/policies/act/modeling_act.py](../src/lerobot/policies/act/modeling_act.py) - ACT 모델
- [lerobot/policies/diffusion/modeling_diffusion.py](../src/lerobot/policies/diffusion/modeling_diffusion.py) - Diffusion 모델
- [lerobot/processor/pipeline.py](../src/lerobot/processor/pipeline.py) (1,873 lines) - 처리 파이프라인
- [lerobot/processor/normalize_processor.py](../src/lerobot/processor/normalize_processor.py) - 정규화

### 과제
1. ACT 정책의 CVAE 구조 분석 및 latent action 생성 과정 이해
2. Diffusion 정책의 노이즈 스케줄링 구현 분석
3. Processor가 observation과 action을 어떻게 처리하는지 코드 추적

---

## Week 3: 학습 파이프라인 (Training Pipeline)

### 학습 목표
- 전체 학습 파이프라인 이해
- 학습 스크립트 작성 및 실행
- 체크포인팅 및 로깅

### 이론
- **학습 워크플로우**
  1. Configuration 설정
  2. Dataset 생성 (delta_timestamps 포함)
  3. Policy 인스턴스화
  4. Optimizer/Scheduler 설정
  5. 학습 루프 (Accelerate 사용)
  6. 체크포인팅
  7. WandB 로깅
  8. 평가 및 Hub 업로드

- **HuggingFace Accelerate**
  - 분산 학습 지원
  - 혼합 정밀도(Mixed Precision)
  - Gradient accumulation

- **최적화 전략**
  - Optimizer 선택 (AdamW, Lion 등)
  - Learning rate scheduling (cosine, linear 등)

### 실습 코드
1. **기본 학습 예제** - [examples/tutorial/act/act_training_example.py](../examples/tutorial/act/act_training_example.py)
   ```python
   from lerobot.scripts.lerobot_train import train

   config = TrainConfig(
       policy_name="act",
       dataset_repo_id="lerobot/aloha_insertion",
       ...
   )
   train(config)
   ```

2. **학습 스크립트 실행** - [scripts/lerobot_train.py](../src/lerobot/scripts/lerobot_train.py)
   ```bash
   python -m lerobot.scripts.lerobot_train \
     --policy-name act \
     --dataset-repo-id lerobot/pusht \
     --output-dir outputs/act_pusht \
     --wandb-enable true
   ```

3. **체크포인트 재개**
   ```bash
   python -m lerobot.scripts.lerobot_train \
     --resume-from-checkpoint outputs/act_pusht
   ```

4. **학습 유틸리티 분석** - [lerobot/utils/train_utils.py](../src/lerobot/utils/train_utils.py)
   - 학습 루프 구현
   - 평가 주기 관리
   - 모델 저장/로드

### 핵심 소스 파일
- [lerobot/scripts/lerobot_train.py](../src/lerobot/scripts/lerobot_train.py) - 학습 진입점
- [lerobot/configs/train.py](../src/lerobot/configs/train.py) - 학습 설정
- [lerobot/utils/train_utils.py](../src/lerobot/utils/train_utils.py) - 학습 유틸리티
- [lerobot/optim/](../src/lerobot/optim/) - Optimizer와 Scheduler
- [lerobot/utils/logging_utils.py](../src/lerobot/utils/logging_utils.py) - 로깅

### 과제
1. PushT 환경에서 Diffusion Policy 학습 (최소 100 에피소드)
2. WandB를 통한 학습 과정 모니터링
3. 체크포인트에서 학습 재개 기능 테스트

---

## Week 4: 평가 및 시뮬레이션 환경

### 학습 목표
- 시뮬레이션 환경 설정
- 정책 평가 파이프라인 구축
- Environment Processor 활용

### 이론
- **지원 환경**
  - **ALOHA**: 양팔 조작 (AlohaInsertion-v0, AlohaTransferCube-v0)
  - **PushT**: 2D 푸싱 태스크
  - **Libero**: 10가지 가정 작업 벤치마크
  - **MetaWorld**: 50가지 조작 작업

- **평가 메트릭**
  - Success rate
  - Episode length
  - Reward

- **Environment Processor**
  - 환경별 관찰 처리
  - 행동 후처리
  - Pre/post processor 체인

### 실습 코드
1. **정책 평가** - [scripts/lerobot_eval.py](../src/lerobot/scripts/lerobot_eval.py)
   ```bash
   python -m lerobot.scripts.lerobot_eval \
     --policy-repo-id lerobot/diffusion_pusht \
     --env-name PushT-v0 \
     --num-eval-episodes 50 \
     --save-video true
   ```

2. **환경 생성** - [lerobot/envs/factory.py](../src/lerobot/envs/factory.py)
   ```python
   from lerobot.envs.factory import make_env

   env = make_env("PushT-v0")
   ```

3. **Libero 벤치마크** - [lerobot/envs/libero.py](../src/lerobot/envs/libero.py)
   ```python
   env = make_env("libero_spatial_pick_cube_from_shelf")
   ```

4. **Environment Processor 분석** - [lerobot/processor/env_processor.py](../src/lerobot/processor/env_processor.py)
   - 관찰 정규화
   - 행동 스케일링
   - 환경별 커스터마이징

### 핵심 소스 파일
- [lerobot/scripts/lerobot_eval.py](../src/lerobot/scripts/lerobot_eval.py) - 평가 스크립트
- [lerobot/envs/factory.py](../src/lerobot/envs/factory.py) - 환경 생성
- [lerobot/envs/libero.py](../src/lerobot/envs/libero.py) - Libero 환경
- [lerobot/envs/metaworld.py](../src/lerobot/envs/metaworld.py) - MetaWorld 환경
- [lerobot/processor/env_processor.py](../src/lerobot/processor/env_processor.py) - 환경 프로세서

### 과제
1. 학습한 정책을 PushT 환경에서 평가하고 success rate 측정
2. Libero 벤치마크 중 3가지 작업에서 사전 학습된 모델 평가
3. 평가 비디오를 저장하고 정책 행동 분석

---

## Week 5: 고급 정책 (VQ-BeT, VLA, Flow-Matching)

### 학습 목표
- Vector-Quantized Behavior Transformer 이해
- Vision-Language-Action 모델 학습
- Flow-matching 기반 정책 분석

### 이론
- **VQ-BeT (Vector-Quantized Behavior Transformer)**
  - 행동 이산화(discretization)
  - Transformer 기반 시퀀스 모델링
  - Codebook 학습

- **Vision-Language-Action 모델**
  - **SmolVLA**: 소형 멀티모달 모델
  - **Pi0**: Flow-matching 기반 VLA
  - **Pi0.5**: 개선된 Pi0
  - **GROOT**: Eagle2 비전 + Action head

- **Flow-Matching**
  - Continuous normalizing flows
  - 확률적 행동 생성
  - Diffusion과의 차이점

- **Real-Time Chunking (RTC)**
  - 추론 시간 최적화
  - Flow-matching 정책 향상 (Pi0, Pi0.5, SmolVLA)

### 실습 코드
1. **VQ-BeT 분석** - [lerobot/policies/vqbet/](../src/lerobot/policies/vqbet/)
   - [modeling_vqbet.py](../src/lerobot/policies/vqbet/modeling_vqbet.py) - 메인 모델
   - [vqbet_utils.py](../src/lerobot/policies/vqbet/vqbet_utils.py) (1,398 lines) - VQ 유틸리티

2. **SmolVLA 사용** - [examples/tutorial/smolvla/using_smolvla_example.py](../examples/tutorial/smolvla/using_smolvla_example.py)
   ```python
   from lerobot.policies.smolvla import SmolVLAPolicy

   policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_pusht")
   action = policy.select_action(observation, text="push the block")
   ```

3. **Pi0 사용** - [examples/tutorial/pi0/using_pi0_example.py](../examples/tutorial/pi0/using_pi0_example.py)
   ```python
   from lerobot.policies.pi0 import Pi0Policy

   policy = Pi0Policy.from_pretrained("lerobot/pi0_aloha")
   ```

4. **Pi0 모델 분석** - [lerobot/policies/pi0/modeling_pi0.py](../src/lerobot/policies/pi0/modeling_pi0.py) (1,332 lines)
   - Flow-matching 구현
   - Action head 구조

5. **RTC (Real-Time Chunking)** - [examples/rtc/](../examples/rtc/)
   - 추론 시간 개선
   - Action chunk 최적화

### 핵심 소스 파일
- [lerobot/policies/vqbet/modeling_vqbet.py](../src/lerobot/policies/vqbet/modeling_vqbet.py) - VQ-BeT
- [lerobot/policies/smolvla/modeling_smolvla.py](../src/lerobot/policies/smolvla/modeling_smolvla.py) - SmolVLA
- [lerobot/policies/pi0/modeling_pi0.py](../src/lerobot/policies/pi0/modeling_pi0.py) - Pi0
- [lerobot/policies/pi05/modeling_pi05.py](../src/lerobot/policies/pi05/modeling_pi05.py) - Pi0.5
- [lerobot/policies/groot/modeling_groot.py](../src/lerobot/policies/groot/modeling_groot.py) - GROOT
- [lerobot/policies/rtc/](../src/lerobot/policies/rtc/) - RTC

### 과제
1. VQ-BeT의 codebook이 어떻게 학습되는지 코드 분석
2. SmolVLA에서 언어 조건(language conditioning)이 어떻게 처리되는지 추적
3. Flow-matching과 Diffusion의 구현 차이점 비교

---

## Week 6: 실제 로봇 통합

### 학습 목표
- 실제 로봇 하드웨어 설정
- 원격 조작(Teleoperation) 시스템 구축
- 데이터 수집 및 비동기 추론

### 이론
- **지원 로봇**
  - **SO-100/SO-101**: 저비용 로봇 팔
  - **Koch**: 모듈형 로봇 팔
  - **ALOHA**: 양팔 로봇
  - **HopeJR**: 휴머노이드 팔
  - **LeKiwi**: 모바일 로봇
  - **Reachy2**: 휴머노이드

- **하드웨어 구성 요소**
  - 모터: Dynamixel, Feetech
  - 카메라: OpenCV, RealSense
  - 모터 버스 통신

- **비동기 추론**
  - Policy Server / Robot Client 아키텍처
  - 실시간 제어
  - 저지연 추론

### 실습 코드
1. **로봇 설정** - [lerobot/robots/](../src/lerobot/robots/)
   - [lerobot/robots/so100_follower/](../src/lerobot/robots/so100_follower/) - SO-100 설정
   - [lerobot/robots/robot.py](../src/lerobot/robots/robot.py) - 베이스 로봇 클래스

2. **모터 캘리브레이션** - [scripts/lerobot_calibrate.py](../src/lerobot/scripts/lerobot_calibrate.py)
   ```bash
   python -m lerobot.scripts.lerobot_calibrate \
     --robot-path lerobot/robots/configs/so100.yaml
   ```

3. **카메라 설정** - [scripts/lerobot_find_cameras.py](../src/lerobot/scripts/lerobot_find_cameras.py)
   ```bash
   python -m lerobot.scripts.lerobot_find_cameras
   ```

4. **원격 조작** - [scripts/lerobot_teleoperate.py](../src/lerobot/scripts/lerobot_teleoperate.py)
   ```bash
   python -m lerobot.scripts.lerobot_teleoperate \
     --robot-path lerobot/robots/configs/so100.yaml \
     --teleop-path lerobot/teleoperators/configs/keyboard.yaml
   ```

5. **데이터 기록** - [scripts/lerobot_record.py](../src/lerobot/scripts/lerobot_record.py)
   ```bash
   python -m lerobot.scripts.lerobot_record \
     --robot-path lerobot/robots/configs/so100.yaml \
     --fps 30 \
     --repo-id username/my_dataset \
     --num-episodes 50
   ```

6. **비동기 추론** - [examples/tutorial/async-inf/](../examples/tutorial/async-inf/)
   - [policy_server.py](../examples/tutorial/async-inf/policy_server.py) - 정책 서버
   - [robot_client.py](../examples/tutorial/async-inf/robot_client.py) - 로봇 클라이언트

### 핵심 소스 파일
- [lerobot/robots/robot.py](../src/lerobot/robots/robot.py) - 로봇 베이스 클래스
- [lerobot/motors/motors_bus.py](../src/lerobot/motors/motors_bus.py) (1,240 lines) - 모터 통신
- [lerobot/cameras/camera.py](../src/lerobot/cameras/camera.py) - 카메라 인터페이스
- [lerobot/async_inference/policy_server.py](../src/lerobot/async_inference/policy_server.py) - 정책 서버
- [lerobot/async_inference/robot_client.py](../src/lerobot/async_inference/robot_client.py) - 로봇 클라이언트
- [lerobot/scripts/lerobot_record.py](../src/lerobot/scripts/lerobot_record.py) - 데이터 기록

### 과제
1. 로봇 구성 YAML 파일 작성 (실제 로봇이 없으면 시뮬레이션)
2. 원격 조작을 통한 데모 데이터 10 에피소드 수집
3. 비동기 추론 파이프라인 구축 및 지연 시간 측정

---

## Week 7: 강화학습 (Reinforcement Learning)

### 학습 목표
- RL 구성 요소 이해 (Actor, Learner, Buffer)
- HIL-SERL (Human-in-the-Loop) 학습
- 온라인 학습 파이프라인 구축

### 이론
- **RL 컴포넌트**
  - Actor: 환경과 상호작용
  - Learner: 정책 학습
  - Replay Buffer: 경험 저장

- **SAC (Soft Actor-Critic)**
  - Model-free RL
  - Maximum entropy framework
  - Continuous action spaces

- **TDMPC (Temporal Difference MPC)**
  - Model-based RL
  - 예측 모델 학습
  - MPC를 통한 계획

- **HIL-SERL (Human-in-the-Loop SERL)**
  - 오프라인 데이터 + 온라인 학습
  - 인간 개입을 통한 안전 학습
  - Reward classifier 학습

### 실습 코드
1. **RL 컴포넌트 분석** - [lerobot/rl/](../src/lerobot/rl/)
   - [actor.py](../src/lerobot/rl/actor.py) - RL Actor
   - [learner.py](../src/lerobot/rl/learner.py) (1,179 lines) - RL Learner
   - [buffer.py](../src/lerobot/rl/buffer.py) (905 lines) - Replay Buffer

2. **SAC 정책** - [lerobot/policies/sac/](../src/lerobot/policies/sac/)
   ```python
   from lerobot.policies.sac import SACPolicy

   policy = SACPolicy(config)
   ```

3. **TDMPC 정책** - [lerobot/policies/tdmpc/](../src/lerobot/policies/tdmpc/)
   ```python
   from lerobot.policies.tdmpc import TDMPCPolicy

   policy = TDMPCPolicy(config)
   ```

4. **HIL-SERL 예제** - [examples/tutorial/rl/hilserl_example.py](../examples/tutorial/rl/hilserl_example.py)
   ```python
   # 오프라인 데이터로 사전 학습
   # 온라인 학습 시작
   # 인간이 필요시 개입
   ```

5. **Reward Classifier** - [examples/tutorial/rl/reward_classifier_example.py](../examples/tutorial/rl/reward_classifier_example.py)
   - 성공/실패 분류기 학습
   - 온라인 reward 제공

### 핵심 소스 파일
- [lerobot/rl/actor.py](../src/lerobot/rl/actor.py) - RL Actor
- [lerobot/rl/learner.py](../src/lerobot/rl/learner.py) - RL Learner
- [lerobot/rl/buffer.py](../src/lerobot/rl/buffer.py) - Replay Buffer
- [lerobot/policies/sac/modeling_sac.py](../src/lerobot/policies/sac/modeling_sac.py) - SAC
- [lerobot/policies/tdmpc/modeling_tdmpc.py](../src/lerobot/policies/tdmpc/modeling_tdmpc.py) - TDMPC
- [lerobot/rl/eval_policy.py](../src/lerobot/rl/eval_policy.py) - RL 평가

### 과제
1. SAC 정책으로 MetaWorld 환경에서 학습
2. Replay buffer의 샘플링 전략 분석
3. HIL-SERL에서 인간 개입이 학습에 미치는 영향 실험

---

## Week 8: 고급 주제 및 프로젝트

### 학습 목표
- 멀티 GPU 학습
- 커스텀 정책 구현
- 프로젝트 기여 및 확장

### 이론
- **분산 학습**
  - Multi-GPU training
  - HuggingFace Accelerate 활용
  - Gradient accumulation

- **스트리밍 데이터셋**
  - 대용량 데이터 처리
  - 메모리 효율적 로딩

- **커스텀 구현**
  - 새로운 정책 추가
  - 새로운 환경 통합
  - 커스텀 processor 작성

- **데이터셋 포팅**
  - 다른 포맷을 LeRobotDataset으로 변환
  - Open X-Embodiment 데이터셋 통합

### 실습 코드
1. **멀티 GPU 학습**
   ```bash
   accelerate launch --num_processes=4 \
     lerobot/scripts/lerobot_train.py \
     --policy-name act \
     --dataset-repo-id lerobot/aloha_insertion
   ```

2. **스트리밍 데이터셋** - [examples/training/train_with_streaming.py](../examples/training/train_with_streaming.py)
   ```python
   from lerobot.datasets import StreamingLeRobotDataset

   dataset = StreamingLeRobotDataset("lerobot/pusht", streaming=True)
   ```

3. **커스텀 정책 구현**
   - Configuration 클래스 작성
   - Modeling 클래스 구현
   - Processor 추가

4. **데이터셋 포팅** - [examples/port_datasets/](../examples/port_datasets/)
   - 다른 포맷 데이터를 LeRobot 포맷으로 변환
   - 메타데이터 생성
   - HuggingFace Hub 업로드

5. **하드웨어 통합 가이드** - [docs/source/integrate_hardware.mdx](../docs/source/integrate_hardware.mdx)
   - 새로운 로봇 추가
   - 커스텀 카메라/모터 통합

### 핵심 소스 파일
- [lerobot/datasets/streaming_dataset.py](../src/lerobot/datasets/streaming_dataset.py) - 스트리밍
- [lerobot/datasets/aggregate.py](../src/lerobot/datasets/aggregate.py) - 데이터셋 병합
- [lerobot/processor/tokenizer_processor.py](../src/lerobot/processor/tokenizer_processor.py) - VLA 토크나이저
- [examples/port_datasets/](../examples/port_datasets/) - 포팅 유틸리티

### 프로젝트 아이디어
1. **새로운 정책 구현**
   - 최신 논문의 정책을 LeRobot 스타일로 구현
   - 예: Behavior Transformer, RT-1, Octo 등

2. **커스텀 환경 벤치마크**
   - 새로운 시뮬레이션 환경 추가
   - 실제 로봇 작업 정의 및 평가

3. **데이터셋 수집 및 공유**
   - 특정 작업에 대한 데모 데이터 수집
   - HuggingFace Hub에 공개

4. **성능 최적화**
   - 정책 추론 속도 개선
   - 메모리 사용량 최적화
   - RTC와 같은 추론 시간 개선 기법 적용

5. **멀티모달 확장**
   - 언어, 비전, 행동을 결합한 정책
   - 프롬프트 기반 작업 조건화

### 과제
1. 커스텀 정책 하나를 처음부터 구현하고 PushT에서 학습
2. 자신만의 데이터셋을 수집하고 LeRobot 포맷으로 변환
3. LeRobot 저장소에 기여할 수 있는 개선 사항 제안 (Issue/PR)

---

## 📖 추가 학습 자료

### 공식 문서
- [LeRobot Documentation](https://github.com/huggingface/lerobot)
- [HuggingFace Hub](https://huggingface.co/lerobot)

### 주요 논문
- **ACT**: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
- **Diffusion Policy**: Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
- **VQ-BeT**: Behavior Generation with Latent Actions
- **SmolVLA**: Small Vision-Language-Action Models
- **Pi0**: Flow Matching for Generalist Policies
- **HIL-SERL**: Human-in-the-Loop Imitation Learning

### 코드 탐색 가이드
- 정책 구현: [lerobot/policies/](../src/lerobot/policies/)
- 데이터셋 처리: [lerobot/datasets/](../src/lerobot/datasets/)
- 학습 스크립트: [lerobot/scripts/](../src/lerobot/scripts/)
- 예제 코드: [examples/](../examples/)

---

## 🛠 실습 환경 설정

### 필수 요구사항
```bash
# Python 3.10+
# CUDA 11.8+ (GPU 학습용)

# 기본 설치
pip install lerobot

# 전체 기능 설치
pip install "lerobot[all]"

# 개발 모드 설치
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[dev]"
```

### 선택 사항
- WandB 계정 (학습 로깅)
- HuggingFace 계정 (데이터셋/모델 공유)
- 실제 로봇 하드웨어 (Week 6)

---

## 🎯 학습 팁

1. **코드 우선**: 이론보다 코드를 먼저 읽고 실행해보세요
2. **작은 실험**: 전체 학습 전에 작은 데이터셋으로 빠르게 테스트
3. **디버깅**: 각 모듈의 입출력 shape을 확인하세요
4. **시각화**: 데이터와 정책 행동을 항상 시각화
5. **커뮤니티**: GitHub Issues와 Discussions 활용

---

## 📊 평가 기준

각 주차별 과제를 완료하면서:
- 코드 이해도 (소스 분석 및 설명)
- 실습 완성도 (실행 가능한 코드)
- 창의성 (Week 8 프로젝트)

---

## 🤝 기여하기

이 커리큘럼을 따라 학습하면서 발견한 개선 사항이나 오류가 있다면
LeRobot 프로젝트에 기여해보세요!

- 버그 리포트: GitHub Issues
- 코드 개선: Pull Requests
- 문서 개선: Documentation PRs

---

**Happy Learning! 🤖**
