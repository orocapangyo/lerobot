# LeRobot 한글 코드 위키

LeRobot 프로젝트의 포괄적인 한국어 문서입니다.

## 📚 목차

### 1. 개요 (Overview)
- [프로젝트 소개](01-overview/00-프로젝트-소개.md)
- [설치 가이드](01-overview/01-설치-가이드.md)
- [빠른 시작](01-overview/02-빠른-시작.md)

### 2. 핵심 개념 (Core Concepts)

#### 2.1 데이터셋
- [LeRobotDataset 이해하기](02-core-concepts/10-LeRobotDataset-개요.md)
- [데이터셋 사용법](02-core-concepts/11-데이터셋-사용법.md)
- [데이터셋 생성과 관리](02-core-concepts/12-데이터셋-생성-관리.md)
- [데이터 변환과 증강](02-core-concepts/13-데이터-변환-증강.md)

#### 2.2 정책
- [정책 아키텍처 개요](02-core-concepts/20-정책-아키텍처-개요.md)
- [ACT 정책](02-core-concepts/21-ACT-정책.md)
- [Diffusion 정책](02-core-concepts/22-Diffusion-정책.md)
- [VQ-BeT 정책](02-core-concepts/23-VQBeT-정책.md)
- [TDMPC 정책](02-core-concepts/24-TDMPC-정책.md)
- [비전-언어-행동 모델](02-core-concepts/25-VLA-정책.md)
- [SAC와 보상 모델](02-core-concepts/26-SAC-RL-정책.md)
- [Real-Time Chunking](02-core-concepts/27-RTC-정책.md)

#### 2.3 환경
- [시뮬레이션 환경 개요](02-core-concepts/30-시뮬레이션-환경-개요.md)
- [ALOHA 환경](02-core-concepts/31-ALOHA-환경.md)
- [PushT 환경](02-core-concepts/32-PushT-환경.md)
- [Libero와 MetaWorld](02-core-concepts/33-고급-시뮬레이션-환경.md)

#### 2.4 프로세서
- [프로세서 시스템 소개](02-core-concepts/40-프로세서-시스템-소개.md)
- [정규화 프로세서](02-core-concepts/41-정규화-프로세서.md)
- [배치 및 관측 프로세서](02-core-concepts/42-배치-관측-프로세서.md)
- [액션 및 환경 프로세서](02-core-concepts/43-액션-환경-프로세서.md)
- [커스텀 프로세서 구현](02-core-concepts/44-커스텀-프로세서-구현.md)

### 3. API 레퍼런스
- [데이터셋 API](03-api-reference/50-데이터셋-API.md)
- [정책 API](03-api-reference/51-정책-API.md)
- [환경 API](03-api-reference/52-환경-API.md)
- [프로세서 API](03-api-reference/53-프로세서-API.md)
- [설정 API](03-api-reference/54-설정-API.md)
- [유틸리티 API](03-api-reference/55-유틸리티-API.md)

### 4. 튜토리얼

#### 4.1 Imitation Learning
- [첫 번째 모델 훈련](04-tutorials/60-첫-모델-훈련.md)
- [실제 로봇으로 IL 수행](04-tutorials/61-실제-로봇-IL.md)

#### 4.2 Reinforcement Learning
- [시뮬레이션에서 RL 훈련](04-tutorials/62-시뮬레이션-RL.md)
- [실제 로봇으로 RL 수행](04-tutorials/63-실제-로봇-RL.md)

#### 4.3 데이터셋 작업
- [데이터셋 시각화](04-tutorials/64-데이터셋-시각화.md)
- [데이터셋 편집과 병합](04-tutorials/65-데이터셋-편집-병합.md)

#### 4.4 추론
- [동기 추론](04-tutorials/66-동기-추론.md)
- [비동기 추론](04-tutorials/67-비동기-추론.md)

### 5. 예제 코드
- [일반 작업 예제](05-examples/70-일반-작업-예제.md)
- [스트리밍 데이터셋](05-examples/71-스트리밍-데이터셋.md)
- [이미지 변환](05-examples/72-이미지-변환.md)
- [데이터셋 포팅](05-examples/73-데이터셋-포팅.md)

### 6. 하드웨어 통합

#### 6.1 카메라
- [카메라 시스템 개요](06-hardware/80-카메라-시스템-개요.md)
- [OpenCV 카메라](06-hardware/81-OpenCV-카메라.md)
- [Intel RealSense](06-hardware/82-RealSense-카메라.md)

#### 6.2 모터
- [모터 시스템 개요](06-hardware/84-모터-시스템-개요.md)
- [Dynamixel 모터](06-hardware/85-Dynamixel-모터.md)
- [Feetech 모터](06-hardware/86-Feetech-모터.md)
- [모터 보정 및 설정](06-hardware/87-모터-보정-설정.md)

#### 6.3 로봇
- [로봇 시스템 개요](06-hardware/88-로봇-시스템-개요.md)
- [SO-101 로봇](06-hardware/89-SO101-로봇.md)
- [SO-100 로봇](06-hardware/90-SO100-로봇.md)
- [Koch v1.1 로봇](06-hardware/91-Koch-로봇.md)
- [커스텀 로봇 통합](06-hardware/95-커스텀-로봇-통합.md)

#### 6.4 원격 조작
- [원격 조작 시스템 개요](06-hardware/96-원격조작-시스템-개요.md)
- [리더 로봇 원격 조작](06-hardware/97-리더-로봇-원격조작.md)
- [게임패드 및 키보드](06-hardware/98-게임패드-키보드.md)

### 7. 고급 주제
- [Multi-GPU 훈련](07-advanced/110-MultiGPU-훈련.md)
- [커스텀 정책 구현](07-advanced/111-커스텀-정책-구현.md)
- [커스텀 환경 구현](07-advanced/112-커스텀-환경-구현.md)
- [비디오 처리 최적화](07-advanced/113-비디오-처리-최적화.md)
- [온라인 버퍼와 RL](07-advanced/115-온라인-버퍼-RL.md)
- [WandB 실험 추적](07-advanced/116-WandB-실험-추적.md)

### 8. 스크립트 및 CLI
- [주요 CLI 명령어](08-scripts/120-주요-CLI-명령어.md)

### 9. 부록
- [용어집](09-appendix/140-용어집.md)
- [자주 묻는 질문](09-appendix/141-FAQ.md)
- [참고 자료](09-appendix/142-참고-자료.md)

---

## 🚀 빠른 네비게이션

### 처음 사용하시나요?
1. [설치 가이드](01-overview/01-설치-가이드.md)부터 시작하세요
2. [빠른 시작](01-overview/02-빠른-시작.md)으로 첫 실습을 해보세요
3. [첫 번째 모델 훈련](04-tutorials/60-첫-모델-훈련.md) 튜토리얼을 따라해보세요

### 데이터셋 작업을 하시나요?
- [LeRobotDataset 이해하기](02-core-concepts/10-LeRobotDataset-개요.md)
- [데이터셋 사용법](02-core-concepts/11-데이터셋-사용법.md)
- [데이터셋 시각화](04-tutorials/64-데이터셋-시각화.md)

### 정책 훈련을 하시나요?
- [정책 아키텍처 개요](02-core-concepts/20-정책-아키텍처-개요.md)
- [첫 번째 모델 훈련](04-tutorials/60-첫-모델-훈련.md)
- [Multi-GPU 훈련](07-advanced/110-MultiGPU-훈련.md)

### 실제 로봇을 사용하시나요?
- [로봇 시스템 개요](06-hardware/88-로봇-시스템-개요.md)
- [실제 로봇으로 IL 수행](04-tutorials/61-실제-로봇-IL.md)
- [비동기 추론](04-tutorials/67-비동기-추론.md)

---

## 📖 문서 작성 원칙

이 위키는 다음 원칙을 따릅니다:

1. **실용성**: 모든 개념은 실행 가능한 코드 예제와 함께 제공
2. **명확성**: 한국어로 명확하게 설명하되, 필요시 영어 원문 병기
3. **완전성**: 초보자부터 고급 사용자까지 모두 활용 가능
4. **최신성**: LeRobot 소스코드를 기반으로 최신 정보 유지
5. **상호 참조**: 관련 문서 간 링크로 쉬운 네비게이션 제공

---

## 🤝 기여하기

이 위키는 커뮤니티의 기여를 환영합니다!

- 오타나 오류 발견시 이슈 제출
- 새로운 튜토리얼이나 예제 추가
- 번역 개선 제안

---

**Happy Learning! 🤖**
