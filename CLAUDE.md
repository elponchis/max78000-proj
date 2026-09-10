# CLAUDE.md — MAX78000FTHR 음향 이벤트 감지 졸업 프로젝트

이 문서는 프로젝트의 **불변 컨텍스트**다. 작업 시작 전 반드시 전체를 읽고,
아래 제약과 규칙을 위반하는 코드를 생성하지 말 것.
실행 가능한 작업 목록은 `TASKS.md`에 있다.

---

## 0. 현재 상태 (2026-09 기준)

### 완료
- 로컬 WSL2 (Ubuntu 22.04) 환경 구축, 디스크 58GB 확보
- `ai8x-training` / `ai8x-synthesis` 클론, venv 레포별 분리 (Python 3.10, torch 2.3.1)
- **관문 B 통과**: 로컬 CPU에서 KWS20 1 epoch 완주 — **77분** → 로컬 학습 불가 확정
- **관문 C 통과**: 사전학습 KWS20 v3 체크포인트로 양자화 → 합성 성공
  - 모델 크기 **169,472 바이트** (442KB 중 38%) → **2× 규모 모델까지 여유 있음**
  - 구조: voice_conv 4층 + kws_conv 4층 + FC 1층, 전부 Conv1d(k=1/3/6)
- Colab 환경 구축 완료 (condacolab + Python 3.11 + torch 2.3.1 + CUDA, T4)
  - Drive 백업 및 복구 절차 3회 검증
  - GPU 학습 1 epoch 완주 확인

### 미완료 / 제약
- **MAX78000FTHR 보드 미확보** ← 최우선 해결 과제
- 전력 계측 장비 확보 여부 미확인
- Colab 무료 티어 RAM 12GB → 대용량 데이터셋 로딩 불가 (KWS20 325,353샘플에서 OOM)

### 보드가 필요한 작업은 착수 금지
`TASKS.md`의 Phase 3 이후는 전부 보드 대기. 착수 전에 명시적으로 알릴 것.

---

## 1. 주제

> **MAX78000 기반 상시 동작 음향 감지 시스템에서 종단간 추론 파이프라인 설계**
> — 전처리 위치가 실시간 제약 충족과 에너지 효율에 미치는 영향

**한 문장 정의**
배터리로 상시 구동되는 소형 음향 이벤트 감지 장치를 MAX78000FTHR 단독으로 구현하고,
전처리를 CPU에서 수행하는 구성과 NPU 친화적으로 재설계한 구성을 대조하여,
NPU 탑재 MCU에서의 파이프라인 설계 지침을 정량적으로 도출한다.

최종 산출물은 **논문 형식 보고서**다. 모든 실험은 논문의 표/그림에 대응되어야 한다.

---

## 2. 핵심 논거

**주 논거**: "NPU가 빠르다"가 아니라
**"NPU가 있으면 모델뿐 아니라 파이프라인 전체를 가속기 친화적으로 다시 설계해야 하며,
그러지 않으면 이득이 상쇄된다."**
그리고 M4 소프트웨어 추론으로는 상시 추론 주기(hop 250ms)를 충족하지 못해
**시스템이 성립 자체를 하지 않는다.** 성능 향상이 아니라 기능 실현의 문제로 프레이밍한다.

**보조 논거**
1. 추론당 에너지 감소 (평균 전류 차분법)
2. 추론 중 M4 여유 사이클 확보
3. SRAM 128KB 제약 하에서 동일 모델의 수용 가능 여부

### 프레이밍 금지 사항 (중요)

**"시간 ↔ 전력 트레이드오프"로 서술하지 말 것.**
에너지 = 전력 × 시간이므로, 추론이 빨라지면 시간이 크게 줄어 에너지도 함께 감소한다
(race to sleep). NPU는 M4 대비 빠르면서 **동시에** 에너지도 적다 — 트레이드오프가 아니라
일방적 우위다. "NPU는 빠른 대신 전력을 더 쓴다"는 서술은 틀렸으며 심사에서 지적당한다.

### 실제로 성립하는 트레이드오프 축

| 축 | 성립 | 대응 목표 |
|---|---|---|
| 시간 ↔ 전력 | ❌ 잘못된 프레이밍 | 사용 금지 |
| 정확도 ↔ 에너지 | ✅ | G9 (비트폭) |
| 응답 지연 ↔ 에너지 | ✅ | G10 (hop) |
| 오탐률 ↔ 에너지 | ✅ | 향후 과제 (캐스케이드) |
| 플랫폼 티어 ↔ 정확도 | ✅ | 도전 1.5순위 (OrangePi) |
| 추론 지연 | 트레이드오프가 아닌 **제약 조건** | 실현 가능 영역 경계 |

---

## 3. 하드웨어 제약 (절대 위반 금지)

### MAX78000 / FTHR
- Cortex-M4F 100MHz + RISC-V 60MHz, 512KB Flash, 128KB SRAM, 16KB Cache
- CNN 가속기: **가중치 메모리 442KB**, **데이터 메모리 512KB**
- 지원 연산: Conv1d, Conv2d(1x1 / 3x3만), ConvTranspose2d, MaxPool/AvgPool,
  element-wise(add/sub/or/xor), 제한적 MLP
- **미지원**: depthwise separable conv (MAX78002에서 추가), 임의 커널 크기,
  FFT / 로그 / DCT 등 MFCC 구성 연산 ← G8의 출발점
- Softmax는 CPU에서 처리, BatchNorm은 학습 후 fold
- 가중치 1/2/4/8bit, 활성화 8bit → **QAT 필수**
- **무선 라디오 없음** (BLE/WiFi 불가 → 외부 모듈 필요, 본 프로젝트 범위 외)
- 온보드: 디지털 마이크, 스테레오 오디오 코덱, VGA 카메라(미사용), microSD,
  1MB QSPI SRAM, RGB LED, 푸시버튼, MAX20303 PMIC(퓨얼게이지), DAPLink
- **온보드 디스플레이 없음**, **온보드 전력 계측 회로 없음** (EVKIT과의 차이)

모델 수정 시 442KB / 512KB 제약 초과 여부를 `ai8xize.py` 합성으로 조기 검증할 것.

---

## 4. 시스템 사양 (확정)

| 항목 | 값 |
|---|---|
| 입력 | 온보드 디지털 마이크, 16kHz mono |
| 윈도우 | 1초, hop 250ms (초당 4회 상시 추론) |
| 모델 입력 | raw waveform → 16384 샘플 → **(128, 128) reshape**, int8 [-128,127] |
| 백본 | ai8x KWS20 v3 파생 1D CNN (169KB @ 8bit) |
| 클래스 | 7종 (아래) |
| 대조군 | 동일 가중치를 CMSIS-NN으로 포팅한 M4 소프트웨어 추론 |
| 출력 | UART→호스트 GUI / SD 이벤트 로그 / RGB LED·부저 |
| 전원 | Li-Po 단독 구동 |

### 클래스 정의

**확정 (2026-09-10). 근거와 상세 수치는 `docs/results/class-mapping.md`.**

| ID | 이름 | 주 출처 | 알림 등급 | train/test 원본 |
|---|---|---|---|---|
| 0 | `siren` | FSD50K(Siren), US8K, ESC-50 | 즉시 | 137 / 65 |
| 1 | `glass` | FSD50K(Shatter, Glass−식기류), ESC-50 glass_breaking | 즉시 | 518 / 223 |
| 2 | `scream` | FSD50K(Screaming/Yell/Shout −군중류) | 즉시 | 412 / 191 |
| 3 | `dog_bark` | FSD50K(Bark/Dog), US8K, ESC-50 | 기록 | 788 / 339 |
| 4 | `background` | 위 4종 외 전부 + 하드 네거티브 + LibriSpeech + MSnoise | — | 샘플링 |

**제외한 클래스**
- `alarm` — vocabulary 200개에 `Alarm_clock`, `Smoke_detector_and_smoke_alarm`,
  `Fire_alarm`, `Buzzer` 가 전부 부재. 상위 라벨 `Alarm` 만 남는데 실제 내용이
  주차미터기·전등 스위치·전철 출입문·찻물 타이머다. 화재경보는 586클립 중 **24개**.
  → `docs/results/alarm-class-analysis.md`. 1,724클립 전량은 하드 네거티브로 편입.
- `baby_cry` — 전 소스 합쳐 176클립으로 다른 클래스의 1/5. 보강 소스 없음.
  억지로 넣으면 혼동행렬의 그 행만 무너져 macro-F1을 끌어내린다.

**라벨 판정 순서 (이 순서가 규칙이다)**

```
Siren → glass(Shatter | Glass−식기류) → scream(−군중류) → dog_bark → background
```

FSD50K는 AudioSet 조상 라벨을 함께 부여한다. `Siren` 클립 132개 **전부(100%)가
`Alarm`을 동시 보유**하므로, 다중라벨 필터를 라벨 문자열에 그대로 적용하면
siren이 0개가 된다. 구체 클래스가 먼저 판정되어야 한다.

- `scream`은 데이터 품질 리스크가 가장 크다고 보았으나 **실측 결과 반대였다.**
  scream 603원본으로 안전하고, 실제 취약 클래스는 **siren(고유 원본 202개)**이다.
  US8K siren 929개는 고유 원본이 74개뿐이다(원본당 중앙값 9슬라이스).
- 샘플 청취는 아직 하지 않았다.

---

## 5. 데이터 규칙 (위반 시 논문이 깨짐)

1. **분할은 Freesound 원본 ID 단위 전역 분할.** 같은 원본에서 뽑은 윈도우가
   train/test에 흩어지면 데이터 누수다.
   ⚠️ **FSD50K·UrbanSound8K·ESC-50은 모두 Freesound 파생이고 ID 체계가 같다**
   (`fname` / `fsID` / `src_file`). 각 데이터셋의 공식 분할을 그대로 쓰면
   1,002개 원본이 중복 등장하고 **337개가 train/test로 갈라진다.**
   → 분할은 `scripts/build_class_manifest.py` 가 **원본 ID 단위로 전역 1회** 결정한다.
   FSD50K 공식 eval 원본은 전량 test에 유지하고(전수 검증 라벨), 부족분만 dev에서
   결정적 해시로 이관해 클래스별 test 비율을 30%로 맞춘다. 공식 분할을 그대로 쓰지
   않은 사유는 논문 데이터셋 절에 명시한다 (`docs/results/class-mapping.md` A-3).
   ⚠️ US8K는 원본 하나에서 여러 슬라이스를 뜬다(siren 929슬라이스 = 원본 74개).
   **신뢰구간은 슬라이스·윈도우가 아니라 원본 수로 계산할 것.**
2. **약라벨 처리.** FSD50K는 클립 단위 weak label이고 길이가 0.3~30초로 제각각이다.
   RMS 에너지 최대 지점 기준으로 1초를 자르고, 에너지 임계값 미달 윈도우는 버린다.
   긴 클립은 상위 N개 구간만 취한다. **필터 전후 데이터 수를 반드시 기록**(논문 표).
3. **다중라벨 → 단일라벨.** 타깃 클래스 중 정확히 하나만 포함한 클립만 채택.
   ⚠️ 단, FSD50K는 AudioSet **조상 라벨을 함께 부여**하므로 라벨 문자열에 그대로
   적용하면 안 된다. `Siren`은 132/132(100%)가 `Alarm`을 동시 보유한다.
   포함관계는 4장의 판정 순서로 해소하고, 포함관계가 아닌 다중 히트만 폐기한다.
3-1. **배제한 클립은 버리지 말고 배경음 하드 네거티브로 편입한다.**
   초인종·주차미터기·식기 부딪힘·군중 환호를 배경음으로 학습시키면 실배치에서
   그 소리에 오탐하지 않는다. 현재 2,478클립. G7(무인 구동 오탐률)의 직접 근거다.
4. **배경음은 다른 클래스 합의 2~3배.** 실배치 환경의 99.9%가 배경음이므로
   균등하게 학습시키면 오탐률이 붕괴한다. 최대한 다양하게 구성.
5. 목표 수량: 클래스당 학습 윈도우 800~1500, 테스트 200 이상, 배경음 3000 이상.
   → 총 약 13,500 윈도우 ≈ **220MB** (KWS20의 24분의 1)
   ⚠️ **달성 여부는 `prepare_safesound.py` 실측 후에만 판정한다.** 클립 길이 기반
   상한으로 판정하지 말 것. 특히 `glass`는 유리 파손이 0.3초 내외 과도음인데 클립
   중앙값이 3.0초라, 에너지 필터를 걸면 클립당 1윈도우만 남아 상한의 절반이 될 수 있다.
6. 증강은 **학습셋에만**: 시간축 shift ±100ms, 볼륨 0.7~1.3배, MSnoise 혼합(SNR 0~20dB).
   테스트셋은 무증강. SNR 실험용 세트는 고정 SNR로 별도 생성.
   ⚠️ **`siren` 은 증강을 더 강하게 건다.** 고유 원본이 202개(train 137)뿐이라
   원본 다양성 부족을 증강으로 일부 보완한다. shift ±200ms, 볼륨 0.6~1.5배,
   MSnoise SNR 0~15dB 정도로 다른 클래스보다 공격적으로. 실측 후 조정.
7. **원본당 슬라이스 상한 4개** (`SLICES_PER_ORIGINAL`). UrbanSound8K는 긴 필드
   녹음을 4초씩 자른 데이터셋이라 한 원본에서 슬라이스가 최대 100개까지 나온다.
   전부 쓰면 모델이 사이렌 자체가 아니라 **그 녹음의 배경 잡음 패턴**을 외운다.
   취할 때는 녹음 전체에 균등 간격으로 퍼뜨린다.

### 데이터 소스
| 소스 | 역할 | 비고 |
|---|---|---|
| FSD50K | 주 소스 | 51,197클립 / 200클래스 / CC / Zenodo 직접 다운로드 / 30GB+ |
| UrbanSound8K | siren, dog_bark 보강 | 약 6GB |
| ESC-50 | 소량 보강 | 클래스당 40클립뿐 — 단독 사용 금지 |
| MSnoise | 잡음 증강 | ai8x-training 내장 |
| LibriSpeech | 배경음에 음성 투입 | kws20.py가 이미 다운로드 (EU 미러 필요) |

디스크 여유가 부족하면 **스트리밍 처리**: zip 파트 하나씩 받아 → 필요 클립만 추출·리샘플
→ interim에 저장 → 원본 파트 삭제 → 다음 파트. 피크 사용량을 수 GB로 유지.

---

## 6. 코드 규칙

### 개발 환경 분리 (호스트는 Windows + WSL2)

작업 성격에 따라 실행 환경이 다르다. **명령어를 안내할 때 반드시 어느 환경인지
명시할 것.** 특히 펌웨어 관련 명령을 WSL 기준으로 안내하지 말 것.

| 작업 | 환경 | 이유 |
|---|---|---|
| 데이터 다운로드·전처리, `safesound.py` | **WSL2 (Ubuntu 22.04)** | ffmpeg/sox/librosa, swap 32GB |
| `ai8x-synthesis` (양자화·합성) | **WSL2** | ADI 공식 지원 환경은 Ubuntu |
| `ai8x-training` (학습) | **Google Colab** | 로컬 GPU 없음, CPU는 1 epoch 77분 |
| MSDK 펌웨어 빌드·플래싱·디버깅 | **Windows 네이티브** | WSL2는 USB 미지원 |
| 호스트 GUI (PyQt + pyserial) | **Windows 네이티브** | 시리얼 포트 직접 접근 |

WSL2 주의:
- **데이터셋을 `/mnt/c/` 아래에 두지 말 것.** Windows 파일시스템 접근이 극단적으로 느림.
  반드시 WSL 내부 경로(`~/max78000/data/`)에 둔다.
- `.wslconfig`에 `memory=11GB`, `swap=32GB` 설정됨 (RAM 15GB 환경)
- 로컬에서 전처리 → `.pt` 캐시만 Drive 업로드 → Colab에서 학습.
  원본 30GB를 Drive에 올리지 않는다.

### 파이썬 환경
- `ai8x-training`과 `ai8x-synthesis`는 **반드시 별도 venv**. 의존성이 다르며 섞으면 충돌.
- 로컬: Python 3.10.12 (레포 공식 요구는 3.11.x이나 동작 확인됨), torch 2.3.1
- Colab: Python 3.11 (condacolab), torch 2.3.1+cu121
- 논문 재현 절차에 **실제 사용 버전을 그대로 기술**할 것.

### 데이터로더 — 메모리 설계 (필수)

`kws20.py`는 전체 데이터를 메모리에 올린 뒤 concat하는 구조라
**로컬(15GB)·Colab(12GB) 양쪽에서 OOM으로 3회 실패했다.**
`safesound.py`는 반드시:

- **전체를 메모리에 올리지 말 것.** `Dataset.__getitem__`에서 필요한 샘플만
  디스크에서 읽는 lazy loading
- `np.memmap` 또는 개별/청크 파일 저장. **하나의 거대한 `.pt` 금지**
- 라벨 단위 처리 후 즉시 디스크 저장, 메모리 누적 금지
- 중간에 죽어도 이어서 할 수 있게 진행 상태 파일 유지
- 전처리 결과는 캐싱. 매 epoch 재전처리 금지
- 전처리 스크립트(`prepare_safesound.py`)와 데이터로더는 **분리**
- 파일 하단 `datasets` 딕셔너리에 등록해야 `train.py --dataset SafeSound`로 잡힌다

### 펌웨어 (보드 도착 후)
- **빌드를 두 개로 분리.** 컴파일 타임 플래그로 제어:
  - `MEASURE_BUILD`: LED·UART·부저·SD 전부 비활성. GPIO 토글 마킹만.
    **모든 정량 실험은 이 빌드로 수행.**
  - `DEMO_BUILD`: UI 전부 활성.
  - 이유: UI 전력이 추론 전력에 섞이면 측정이 오염되고 심사에서 지적당한다.
- CMSIS-NN 베이스라인은 **새로 학습하지 말고**, `ai8x-synthesis`가 생성한 C 가중치
  배열을 그대로 `arm_convolve_*_s8` 계열에 매핑한다. 동일 가중치가 공정 비교의 전제.
- 이벤트 병합 상태머신 필수: 초당 4회 추론이므로 3초짜리 사이렌이 12개 레코드로
  쪼개진다. 연속 동일 클래스는 하나로 묶고, N프레임(약 2초) 이상 끊기면 종료 처리.

### 로깅
- SD 레코드 스키마: `timestamp | class | confidence | duration`
- **오디오 원본은 어떤 단계·어떤 경로로도 저장하거나 전송하지 않는다.**
  편의가 아니라 설계 제약이며 논문의 프라이버시 주장 근거다.
- RTC는 전원 차단 시 초기화된다. 부팅 시 UART로 호스트에서 시각 동기화한다.

---

## 7. 측정 방법론

### MAX78000FTHR
- **지연**: DWT 사이클 카운터. 1000회 측정 후 최악값 기준 보고.
- **에너지**: 온보드 계측 회로가 없으므로 외부 측정.
  - 1순위: 오실로스코프 + 1Ω 션트 (배터리 경로 직렬)
  - 2순위: Nordic PPK2 (약 15만원) / USB 전류계 / PMIC 퓨얼게이지 장시간 로깅
  - 추론 구간을 GPIO 토글로 마킹해 시간 정렬
  - 코어 레일 분리 불가 → **M4 경로 vs NPU 경로의 차분**으로 논증.
    베이스라인이 양쪽에 동일하게 깔리므로 차분이 오히려 깨끗하다.
  - idle 평균 전류를 따로 재서 차감 → 추론 순수 에너지
- **정확도**: PC 시뮬레이션과 **온디바이스 실측을 모두** 보고. 테스트셋을 SD에
  적재해 보드에서 직접 추론시켜 혼동행렬 산출, 괴리도 함께 기술.

### OrangePi 5 Pro (도전 1.5순위)
전력 특성이 근본적으로 달라 동일 방식 측정은 불공정하다.

| | MAX78000FTHR | OrangePi 5 Pro |
|---|---|---|
| Idle 전력 | 수 mW | 2~4 W |
| 추론 증분 | 수 mW | 1~3 W |
| Idle 대비 추론 비중 | 지배적 | 미미 |
| 전원 | Li-Po 3.7V | DC 5V/4A |

→ **세 지표를 모두 보고**한다.
1. 추론당 증분 에너지 = (추론 중 평균 P − idle P) × 추론 시간
2. 시스템 평균 전력
3. **하루 총 에너지** ← 결정적. 초당 4회 워크로드에서
   배터리 며칠 vs 상시 전원 필수가 숫자로 드러남

→ 측정 조건 통일
- 동일 모델 결과와 각 플랫폼 최적 모델 결과를 **따로** 보고
- OrangePi는 헤드리스, HDMI·네트워크 정지
- OS 지터 때문에 GPIO 마킹 정확도가 낮으므로,
  **추론 1000회 루프의 평균에서 idle을 차감**하는 통계적 방식 사용
- 각 조건 10분 이상, 3회 반복, 평균·표준편차 기록
- 주변 온도 기록 (RK3588S 발열 클럭 저하)
- PPK2는 최대 1A라 OrangePi에는 부족할 수 있음 → 벤치 전원공급기 또는 USB-C 전력계

---

## 8. 목표

### 필수 — 이것만으로 논문 완성

| ID | 목표 | 검증 |
|---|---|---|
| **G1** | 시스템 구현 | 배터리 구동 30분 연속, 프레임 드롭 0 |
| **G2** | 실시간 제약 충족 | 추론 지연 ≤ hop의 10%, DWT 1000회 최악값 |
| **G3** | NPU 효과 입증 | 지연·에너지·여유사이클·메모리 4축 + **베이스라인 미성립 실증** |
| **G4** | 인식 성능 | 온디바이스 실측 혼동행렬, PC 시뮬레이션과 괴리 병기 |
| **G5** | 데이터셋 구축 | 클래스당 800+ 윈도우, 원본 클립 단위 분할 |
| **G6** | 검증 인터페이스 | 호스트 GUI + NPU/CPU 실시간 전환 + LED/부저 |
| **G7** | 이벤트 로깅 | SD 영속 저장, 8시간 무인 구동, 오디오 원본 미저장 |
| **G8** | **전처리 오프로드 스펙트럼** ← 제목의 핵심 기여 | 아래 상세 |
| **G9** | 비트폭 파레토 분석 | 8/4/2bit 정확도-에너지 곡선, 최적 지점 제시 |
| **G10** | **hop 스윕** | 250ms/500ms/1s 응답지연-에너지 곡선 |

#### G8 상세 — 전처리 오프로드 스펙트럼

**문제**: MFCC(FFT/로그/DCT)는 NPU 지원 연산이 아니라 M4 CPU가 소프트웨어로 처리해야 한다.
전처리가 8ms, NPU 추론이 3ms라면 NPU를 아무리 빠르게 해도 총 시간은 8ms 아래로 못 간다.
**가속기를 달아놓고도 파이프라인 앞단이 병목이라 이득이 상쇄되는 상황.**

**실험**: 전처리를 CPU에서 얼마나 걷어내느냐에 따른 지연·에너지 곡선

| 구성 | CPU가 하는 일 | NPU가 하는 일 |
|---|---|---|
| ① 전량 CPU | MFCC 전체 | 2D CNN |
| ② 부분 오프로드 | FFT만 | 나머지 + CNN |
| ③ 학습 프론트엔드 | 없음 | 필터뱅크 역할 conv + CNN |
| ④ 전량 NPU | 없음 | raw 1D CNN (KWS20 v3 방식, 기본 구성) |

**최소 버전은 ①과 ④ 두 점 비교**로 성립한다. ②③은 스펙트럼을 채우는 확장.
④는 이미 기본 구성이므로, **①(MFCC + 2D CNN)을 별도로 학습·구현**해야 한다.

**도출할 결론 형태**: "전처리를 NPU로 옮기면 정확도 N%p를 잃는 대신 지연을 M% 줄인다.
NPU 탑재 MCU에서는 모델만 최적화하는 것으로 부족하며, 파이프라인 전체를 가속기 지원
연산으로 재구성해야 가속기의 이득이 실현된다."

#### G9 상세 — 비트폭 파레토
8/4/2bit로 각각 QAT 학습 후 (에너지, 정확도) 점을 찍어 파레토 곡선 도출.
"무릎(knee)"을 찾아 최적 비트폭을 제시한다.
**처음부터 학습하지 말고 8bit 모델을 fine-tuning**하면 시간이 크게 절약된다.

#### G10 상세 — hop 스윕
hop을 125/250/500/1000ms로 바꿔 (하루 에너지, 최악 검출 지연, 정확도) 측정.
**학습 재수행 불필요, 펌웨어에서 추론 주기만 바꿔 측정.** 비용 대비 산출이 가장 좋다.
결론: "화재경보처럼 1초 내 알림이면 충분한 용도에서는 hop 500ms로 에너지를 절반
줄일 수 있다. 유리 파손처럼 짧은 이벤트는 hop 250ms 이하가 필요하다."

### 도전

**1순위 · 모델 규모별 교차점**
백본 채널 수 **0.25×/0.5×/1×/1.5×**로 스윕. 전처리가 전체 지연·에너지에서 차지하는
비중 변화를 측정해 **"모델이 이 크기 이하일 때 전처리 위치가 지배적"**이라는 적용
조건 제시. 전처리는 모델 크기와 무관한 고정 비용이므로, 모델이 작을수록 전처리가
지배한다. → 사례를 원칙으로 격상. 추가 하드웨어 0, 학습 6~8회.

⚠️ **"169KB / 442KB이므로 2× 수용 가능"은 오류였다** (2026-09-10 정정).
Conv 파라미터는 채널 수에 **제곱**으로 비례한다(in_ch × out_ch). 8bit 기준 실측:

| 배율 | params | 442KB 대비 |
|---|---:|---:|
| 0.25× | 12,689 | 2.8% |
| 0.5× | 43,729 | 9.7% |
| 1× | 165,457 | 36.6% |
| 1.5× | 364,753 | **80.6% ← 8bit 상한** |
| 1.75× | 489,873 | 108.2% 초과 |
| 2× | 633,425 | 140.0% 초과 |

8bit 상한은 약 1.6×다. 4bit 가중치라면 2×도 들어가므로 G9(비트폭)와 조합할 여지는
있다. 확정은 `ai8xize.py` 합성 결과로 한다 (`models/ai85net-safesound.py`).

**1.5순위 · OrangePi 5 Pro 티어 비교**
같은 태스크를 세 티어에서 돌려 "이 태스크에 어느 하드웨어 티어가 적정한가"를 규명.

| 티어 | 플랫폼 |
|---|---|
| MCU CPU only | MAX78000 M4 (CMSIS-NN) — 실시간 미달 |
| MCU + NPU | MAX78000 CNN 가속기 — 본 연구 주인공 |
| SBC + NPU | OrangePi 5 Pro (RK3588S, RKNN) — 상위 티어 |

- 제목의 "파이프라인 설계"에 **하드웨어 티어 선택**이 포함되어 자연스럽게 확장
- "왜 하필 MAX78000인가"에 정면으로 답변
- RKNN은 지원 연산이 넓어 전처리 선택지가 다름 →
  **"제약이 클수록 파이프라인 재설계가 중요하다"**를 대조로 입증
- 측정 방법론은 7장 참조. 방법론 자체가 논문 기여가 된다.
- 리스크: RKNN 툴체인은 별개의 학습 곡선. **MAX78000 쪽 완성 전 착수 금지.**
- **퇴로로서의 가치**: 보드 확보가 지연·무산될 경우 "엣지 NPU 티어 비교"로
  무게중심을 옮길 수 있다. 데이터셋·전처리·모델 설계는 그대로 재사용된다.

**2순위 · RISC-V 저전력 게이팅**
RISC-V(60MHz, FPU 없음)를 전처리 오프로드에 쓰는 것은 부적절하다
— M4보다 느리고, raw 입력 설계에서는 오프로드할 전처리가 없다.
대신 **상시 감시 게이트**로 활용: RISC-V만 깨어 마이크 DMA + 에너지 임계값 검사,
M4는 deep sleep. 이벤트 후보 감지 시에만 M4 기상 → NPU 추론.
→ 평균 전력 대폭 감소. 난이도 높음, 다른 게 다 끝난 뒤에만 착수.

**보조**: SNR 20/10/0dB 강건성, PMIC 퓨얼게이지 실측 배터리 방전 곡선

### 향후 과제 — 실제로 하지 않음, 논문에만 기술
- BLE 원격 알림 (라디오 부재, 무선 전력이 측정 오염)
- 캐스케이드 게이팅 — 오탐률 ↔ 에너지 (도전 2순위 미수행 시 여기로)
- 다른 NPU MCU(MAX78002 등)로의 일반화
- 실환경 장기 현장 배치

### 범위 제외 (제안서에 명시)
화자 식별, 음원 방향 추정, 다중 이벤트 동시 검출, 클라우드 연동.

---

## 9. 대표 그림 (논문 Figure 1급)

```
가로축: 하루 총 에너지 (Wh/day)
세로축: macro-F1
점:     각 설정 (비트폭 × hop × 플랫폼)
색/크기: 응답 지연
회색 영역: 실현 불가능 (추론 지연 > hop 주기)
```

M4 소프트웨어 추론 설정들은 회색 영역으로 밀려나고 NPU 설정만 유효 영역에 남는다.
**핵심 논거가 그림 한 장으로 시각화된다.**

---

## 10. 수치 목표에 대한 경고

문서 내 수치 목표(hop 10%, 지연 배율, F1 값)는 **모두 가안**이다.
예비 측정 후 실측 기반으로 캘리브레이션한다.
**측정하지 않은 숫자를 논문이나 코드 주석에 확정값처럼 쓰지 말 것.** 측정 전이면 `TBD`.

---

## 11. 디렉터리 구조

```
~/max78000-proj/               # 본인 레포 — github.com/elponchis/max78000-proj
│                             # WSL2 배포판 `max78000` 내부 경로
├── CLAUDE.md
├── TASKS.md
├── scripts/
│   ├── check_vocabulary.py   # FSD50K 라벨 실재 확인 (완료)
│   ├── prepare_safesound.py  # 필터링 → 윈도우 추출 → 리샘플
│   ├── dataset_stats.py      # 클래스별 통계 표 생성
│   └── analyze_power.py      # 전력 CSV → 추론당 에너지 적분
├── datasets/safesound.py     # ai8x-training/datasets/로 심볼릭 링크
├── models/                   # 모델 정의
├── firmware/
│   ├── npu/ cmsisnn/ common/
├── host/
│   ├── gui.py                # PyQt 실시간 GUI
│   └── fake_serial.py        # 보드 없이 GUI 개발용 더미 생성기
├── colab/train.ipynb
├── patches/                  # ai8x 레포 수정 기록 (git diff)
├── data/                     # git 제외 — 구조는 docs/data-layout.md
└── docs/
    ├── proposal.md
    ├── data-layout.md
    └── results/
        └── vocabulary-check.md   # Phase 2.1 라벨 조사 결과

~/ai8x-training/              # 벤더 툴체인 — GitHub에 올리지 않음
~/ai8x-synthesis/             # 벤더 툴체인 — GitHub에 올리지 않음
```

**벤더 툴체인은 본인 레포에 포함하지 않는다.** 6.6GB이고 GitHub 용량 제한에 걸리며,
어디까지가 본인 작업인지 흐려진다. 본인 코드는 `~/max78000-proj`에 두고,
ai8x 레포에는 **심볼릭 링크**로 연결한다.

```bash
ln -s ~/max78000-proj/datasets/safesound.py ~/ai8x-training/datasets/safesound.py
```

ai8x 레포 수정은 패치로 기록:
```bash
cd ~/ai8x-training && git diff > ~/max78000-proj/patches/<이름>.patch
```

---

## 12. 환경 이슈 기록 (재현 절차 / 논문 부록)

### 로컬 WSL2
- LibriSpeech US 미러(`us.openslr.org`) 404 → **EU 미러(`openslr.elda.org`) 교체**
- `libsox3 libsox-dev libsox-fmt-all sox` 시스템 패키지 필요
  (torchaudio `speed_augment`가 사용)
- **`apt autoremove` 금지** — venv 의존 시스템 라이브러리를 제거해 학습이 깨진다.
  실제로 libsox 및 OpenOCD 의존성(libftdi1-2, libjaylink0, libusb-0.1-4,
  libgpiod2, libcapstone4)이 제거된 사고 발생. 공간 정리는 `apt clean`까지만.
- `.wslconfig`: `memory=11GB`, `swap=32GB` (RAM 15GB 환경). swap 없이는 전처리 중 OOM
- WSL 가상 디스크는 파일 삭제 후에도 축소되지 않음 → `diskpart`의 `compact vdisk` 필요
  (37.4GB → 28.5GB, C드라이브 27GB → 58GB 확보)

### 데이터 다운로드 (2026-09-10)
- **Zenodo(`zenodo.org`) 접속 불가.** TLS 핸드셰이크까지는 성공하나 HTTP 응답이
  오지 않고 타임아웃(25초 이상, `curl` 반환 코드 000). 같은 시점에 github.com /
  huggingface.co 는 0.1초 내 200 응답 → 네트워크 전반이 아니라 Zenodo 측 문제.
  → **FSD50K ground_truth는 HuggingFace 미러에서 취득**:
  `https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/labels/{vocabulary,dev,eval}.csv`
  무결성 확인: vocabulary 200 라벨, dev 40,966 + eval 10,231 = 51,197 클립 (공식 수치 일치).
  FSD50K 오디오 본체(30GB+)도 Zenodo가 유일 배포처이므로 **다운로드 경로를 별도로
  확보해야 한다** (미러 또는 접속 복구 대기).

### 데이터셋 간 원본 중복 (2026-09-10 발견)
- FSD50K / UrbanSound8K / ESC-50 은 **모두 Freesound 파생**이며 ID 체계가 같다.
  FSD50K `fname`, US8K `fsID`, ESC-50 `src_file` 이 동일한 Freesound 클립 ID다.
  실측: 2개 이상 데이터셋에 등장하는 원본 **1,002개**, 각 데이터셋 공식 분할을
  그대로 쓰면 **337개가 train/test로 갈라진다**(dog_bark 197, siren 62, glass 24,
  scream 3). → 5장 규칙 1의 원본 ID 단위 전역 분할로 차단.
- **US8K는 원본 하나에서 여러 슬라이스를 뜬다.** siren 929슬라이스의 고유 원본은
  74개뿐(원본당 중앙값 9, 최대 100), dog_bark 1000슬라이스는 원본 337개.
  데이터 양과 표본 독립성을 혼동하지 말 것. 신뢰구간은 원본 수로 계산한다.
- FSD50K 클립 길이는 메타데이터에 없다. `clips_info` JSON에도 없다.
  → HF tree API로 파일 크기를 받아 역산했다 (PCM 16bit/44.1kHz/mono,
  `duration = (size − 44) / 88200`). 검증: 최솟값 정확히 0.300초, 99퍼센타일
  28.9초로 공식 스펙과 일치. 결과는 `data/raw/FSD50K_meta/clip_sizes.json`.

### FSD50K 라벨 구조 (전처리 설계에 직결)
- FSD50K는 AudioSet 온톨로지를 따라 **조상 라벨을 함께 부여**한다.
  실측: `Siren` 클립 132개 중 **132개(100%)가 `Alarm`을 동시 보유**.
  → CLAUDE.md 5장 규칙 3(타깃 클래스 정확히 1개)을 라벨 문자열에 그대로 적용하면
  siren이 0개가 된다. **포함관계 쌍은 우선순위로 해소**하고(구체 클래스 우선),
  포함관계가 아닌 조합만 폐기할 것. `scripts/check_vocabulary.py`의
  `CLASS_PRIORITY` / `SUBSUMED_PAIRS` 참조. `prepare_safesound.py`도 동일 규칙 적용.

### Colab
- 기본 Python 3.13, ai8x는 3.11 요구 → **`condacolab`으로 3.11 환경 생성**,
  모든 명령을 `conda run -n ai8x`로 실행 (`--no-capture-output`으로 로그 실시간 표시)
- torch 2.3.1은 PyPI에서 제거됨 → **PyTorch 공식 cu121 인덱스**에서 설치
  (Python 3.13에서는 휠 자체가 없어 3.11 환경이 필수)
- `pyffmpeg==2.4.2.18.1` 부재 → 최신 버전 설치 (datasets/kinetics.py가 import)
- `visdom` 빌드 실패 (구식 setup.py) → 별도 처리. torchnet이 import 시점에 요구
- distiller `__init__.py`의 `except pkg_resources.DistributionNotFound`가
  `ContextualVersionConflict`를 못 잡아 **모듈 로딩이 중간에 중단됨**
  → `except Exception:`으로 변경. 이 하나가 `knowledge_distillation` 없음,
  `__file__` None, `__version__` 없음 증상의 공통 원인이었다.
- **`distiller/` 디렉터리명이 네임스페이스 패키지 충돌 유발**
  (`/content/ai8x-training/distiller`가 설치된 패키지를 가림)
  → `distiller_repo`로 이름 변경 후 재설치
- `MPLBACKEND=Agg` 필요 (Colab의 matplotlib_inline 백엔드가 conda 환경에 없음)
- `apt-get install libsox3 ...` 필요 — conda 환경 밖이라 환경 백업에 미포함
- 세션 재시작 시 `/content` 전체 초기화 → Drive 백업에서 복원
- 무료 티어 RAM 12GB → KWS20(325,353샘플, 5.7GB `.pt`) 로딩 시 OOM

### Colab 복구 절차

**셀 1** (재시작 유발)
```python
!pip install -q condacolab
import condacolab; condacolab.install()
```

**셀 2** (일괄 복구)
```python
from google.colab import drive
drive.mount('/content/drive')
!tar xzf /content/drive/MyDrive/max78000/ai8x-conda-env.tar.gz -C /usr/local/envs
!rm -rf /content/ai8x-training
!tar xzf /content/drive/MyDrive/max78000/ai8x-training-patched.tar.gz -C /content
!apt-get -qq install -y libsox3 libsox-dev libsox-fmt-all sox
import os; os.environ['MPLBACKEND'] = 'Agg'
%cd /content/ai8x-training
!rm -rf logs && ln -s /content/drive/MyDrive/max78000/logs logs
```

Drive 백업 파일: `ai8x-conda-env.tar.gz`(3.2GB), `ai8x-training-patched.tar.gz`

> tar `--exclude`는 경로 인자보다 **앞에** 와야 한다. 뒤에 두면 무시된다.

---

## 13. 작업 시 지켜야 할 것

- **보드가 필요한 작업은 착수하지 말고 명시적으로 알릴 것.**
- 대용량 다운로드(FSD50K 30GB+)나 장시간 학습 전 **반드시 먼저 확인을 구할 것.**
- 실험 결과 수치를 추정하거나 지어내지 말 것. 측정 전이면 `TBD`.
- 데이터 규칙(5장)을 우회하는 지름길 코드를 제안하지 말 것.
  특히 **윈도우 단위 랜덤 분할은 절대 금지.**
- 데이터로더는 **반드시 lazy loading**. 전체를 메모리에 올리는 구조 금지 (6장).
- 모델 구조를 바꿀 때는 442KB/512KB 제약 초과 여부를 함께 보고할 것.
- **"시간 ↔ 전력 트레이드오프"로 서술하지 말 것** (2장).
- **명령어 안내 시 실행 환경(WSL2 / Windows / Colab)을 반드시 명시할 것.**
  펌웨어 빌드·플래싱·시리얼 관련 명령을 WSL 기준으로 안내하지 말 것.
- 파일 경로를 제안할 때 WSL 내부 경로와 `/mnt/c/` 경로를 혼동하지 말 것.
- 한국어로 응답할 것. 코드 주석도 한국어.
