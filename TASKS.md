# TASKS.md — 실행 체크리스트

`CLAUDE.md`의 제약을 전제로 한다. 완료 시 `[ ]` → `[x]`로 갱신할 것.
**Phase 4 이후는 보드가 도착해야 착수 가능하다.**

---

## Phase 0 — 환경 구축 ✅ 완료

### 로컬 WSL2
- [x] Ubuntu 22.04, 디스크 58GB 확보 (`compact vdisk`로 회수)
- [x] `.wslconfig` memory=11GB / swap=32GB 설정
- [x] `ai8x-training` / `ai8x-synthesis` venv 레포별 분리 (Python 3.10, torch 2.3.1)
- [x] LibriSpeech EU 미러 교체, libsox 설치
- [x] **관문 B**: KWS20 1 epoch 완주 — **77분 (CPU)** → 로컬 학습 불가 확정
- [x] **관문 C**: 양자화 → 합성 통과. **169,472 바이트 / 442KB (38%)**

### Colab
- [x] condacolab + Python 3.11 + torch 2.3.1+cu121 + T4 GPU
- [x] 모든 패치 적용 (12장 참조)
- [x] Drive 백업 (`ai8x-conda-env.tar.gz`, `ai8x-training-patched.tar.gz`)
- [x] 복구 절차 3회 검증
- [x] GPU 학습 1 epoch 완주 확인
- [x] 노트북 Drive 저장
- [ ] 무료 티어 RAM 12GB 한계 확인됨 → `safesound.py` lazy loading으로 대응 예정

---

## Phase 1 — 프로젝트 레포 구성 (보드 불필요)

레포: **github.com/elponchis/max78000-proj** (로컬 `~/max78000-proj`, WSL2 `max78000` 배포판)

- [x] `~/max78000-proj` 디렉터리 생성, `git init` (main 브랜치)
- [x] `.gitignore` 작성 (data/, logs/, venv/, *.pt, *.pth, *.tar.gz, __pycache__)
- [x] GitHub 빈 레포 생성 후 `git remote add origin`
- [ ] 첫 push
- [x] `CLAUDE.md`, `TASKS.md` 배치
- [x] `patches/` 에 ai8x 레포 수정 기록
      `patches/ai8x-training-local-fixes.patch` (kws20.py LibriSpeech EU 미러 +
      requirements 2건). `ai8x-synthesis`는 추적 파일 변경 없음 → 패치 없음.
- [ ] Claude Code를 `~/max78000-proj`에서 실행하도록 설정

> 벤더 툴체인(`ai8x-training`, `ai8x-synthesis`)은 **GitHub에 올리지 않는다.**
> 심볼릭 링크로 연결한다.

---

## Phase 2 — 데이터셋 구축 (보드 불필요, 최우선)

### 2.1 클래스 확정 — **오디오 없이 가능, 지금 즉시 착수**
- [x] FSD50K 메타데이터 다운로드 — **Zenodo 접속 불가 → HF 미러 사용**
- [x] US8K / ESC-50 메타데이터 다운로드 및 집계
- [x] FSD50K 클립 길이 확보 (파일 크기 역산 — `clip_sizes.json`)
- [x] `scripts/check_vocabulary.py` — 라벨 실재 확인
- [x] `scripts/analyze_alarm_identity.py` — 제목·태그로 `alarm` 내용 검증
- [x] `scripts/build_class_manifest.py` — 확정 매핑 → `data/interim/manifest.csv`
- [x] `Smoke detector` 부재 확인 → **`alarm` 클래스 제외 확정**
- [x] `baby_cry` 제외 확정 (176클립, 보강 소스 없음)
- [x] `glass` 범위 확정 — `Shatter` ∪ (`Glass` − 식기류)
- [x] 데이터셋 간 Freesound 원본 중복 발견 → 원본 ID 단위 전역 분할로 차단
- [x] **최종 클래스 매핑 확정 (5클래스) 및 `CLAUDE.md` 4장 갱신**
- [x] 교차 데이터셋 승격 경계 케이스 7건 수정 (하드 네거티브 2,478 → 2,477)
- [x] **1차 청취 (listen_v2)** — 라벨 노이즈 발견: `scream` 에 말소리만 든 윈도우,
      `glass` 에 파손음이 아닌 울림. → glass 하위유형 게이트 + 클래스별 윈도우 선택
      + PANNs 태거 필터로 대응 (`docs/results/label-noise-filter.md`)
- [x] **2차 청취 (listen_v3 채택분 150창 + listen_rejected 경계 40창)** 완료
      → `docs/results/listening-verification.md`, 판정 기록 `data_overrides.csv` (48행)
      · glass 50창 정답 72% — **태거 점수가 파손음 여부와 무관**(0.02~0.05 는 78%,
        0.05~0.10 은 31% 로 순서가 뒤집힘) → glass 태거 해제, 순위는 onset 으로
      · scream 50창 정답 80% — 0.10 아래는 무작위(50~57%), 확인된 비명 최저 0.026
        → 임계값 0.05 → **0.025**
      · siren 30창 중 3창 오답(확률 0.40~0.50) — 임계값으로 못 거름, override 폐기
      · dog_bark·background 각 30창 오답 0 → 유지
- [ ] **3차 청취 — glass Glass-only 통과분 30창** ← 다음 청취
      onset 게이트 8.0 을 통과한 Glass-only 클립이 실제 파손음인지 미검증이다
      (2차에서 표본이 2창뿐이었고 둘 다 오답). 게이트 8.0 유지 / 상향 /
      `Shatter` 전용(train 461) 중 선택이 여기에 달려 있다
      ⚠️ US8K siren 원본 159738 / 159742 / 159747 의 슬라이스 6개는 최대 윈도우 RMS가
      −60~−74dBFS다. 먼 사이렌인지 사실상 무음인지 원본(`data/raw/US8K_audio/`)을 들어
      판정할 것 — 절대 하한 선택에 직결된다

> 확정 내역 전문: `docs/results/class-mapping.md`
> `alarm` 제외 근거: `docs/results/alarm-class-analysis.md` (논문 데이터셋 절에 사용)
> **주의 1**: FSD50K는 AudioSet 조상 라벨을 함께 부여한다(`Siren`⊂`Alarm` 100%).
> **주의 2**: US8K는 원본당 여러 슬라이스다(siren 929슬라이스 = 원본 74개).
> 신뢰구간은 슬라이스가 아니라 원본 수로 계산할 것.

### 2.2 데이터 다운로드 — **선별 다운로드** (전체 30GB를 받지 않는다)

매니페스트에 있는 클립만 받는다. `scripts/download_clips.py`
⚠️ Zenodo 접속 불가 → HuggingFace 미러 `Fhrozen/FSD50k` 사용.
미러 무결성은 표본 검증 완료 (44.1kHz/16bit/mono, 크기·길이 일치).

- [x] 미러 무결성 표본 검증
- [x] **1단계**: 이벤트 4클래스 + 하드 네거티브 — **4,883클립 / 3.1GB**
      `python3 scripts/download_clips.py --stage events`
      전량 `clip_sizes.json` 크기 일치. 매니페스트 수정(교차 데이터셋 승격 7건 폐기) 후
      사용분 4,876클립 — `docs/results/class-mapping.md` A-5
- [x] UrbanSound8K (siren·dog_bark 슬라이스만), ESC-50 — **매니페스트분만 선별**
      US8K 854슬라이스 / 599MB: HF 미러 `MahiA/UrbanSound8K`. 공식 Zenodo tar.gz와
      sha256 150/150 일치, 파일별 sha256 검증. ESC-50 52클립 / 22MB: 저자 GitHub.
      `python3 scripts/download_clips.py --source us8k` / `--source esc50`
- [ ] **2단계**: `prepare_safesound.py` 완성 → 에너지 필터 실측 윈도우 수 확정
      (작은 데이터로 스크립트를 디버깅할 수 있어 반복이 빠르다)
  - [x] 리샘플러 `scipy.signal.resample_poly` 교체 (이전: 이동평균 + 선형보간, 앨리어싱)
  - [x] 절대 하한 + 클립 내 상대 기준 병용, 임계값 인자화
  - [x] 민감도표 실측 → `docs/results/energy-filter-sweep.md` (5,782클립, 클래스×split)
        floor −70~−30 × rel 0.00~0.50 (−45/−35/−30 추가로 무릎 구간을 채웠다)
  - [x] 라벨 노이즈 대응 3종 → `docs/results/label-noise-filter.md`
        glass 하위유형 게이트(Shatter 필수 / Glass-only 는 onset ≥ 8.0),
        hop 250ms 중첩 후보 + 겹침 없는 최종 선택, PANNs Cnn14(16k) 태거 필터
  - [x] 청취 검증으로 임계값 확정 → `docs/results/listening-verification.md`
  - [x] **데이터셋 v1 확정 (`dataset-v1` 태그)** — 목표(train 800+/test 200+) 판정
        최종 실측 (hop 250ms, int8 0비율 하한 0.95(이벤트만) / rel 0.30,
        태거 glass 해제·scream 0.025·siren·dog_bark 0.10, override 49행,
        클립 9,202개, 저장 1.2초 창):

        | 클래스 | train 창 | test 창 | train 원본 | test 원본 | 목표 대비 |
        |---|---:|---:|---:|---:|---|
        | siren | 528 | 147 | 128 | 54 | train·test 미달 |
        | glass | 477 | 221 | 336 | 147 | train 미달 |
        | scream | 491 | 225 | 317 | 142 | train 미달 |
        | dog_bark | 1,934 | 776 | 727 | 284 | 충족 |
        | background | 9,577 | 3,838 | 4,509 | 1,604 | 이벤트 합의 2.79× / 2.80× |
        | **합계** | **13,007** | **5,207** | | | 342MB / tar 140MB |

        · 배경음 비율은 규칙 4(2~3배) 안이다. 첫 집계에서 3.09배로 넘겨
          `BG_GENERAL_TRAIN` 3100→2630, `_TEST` 900→790 으로 줄였다
          (하드 네거티브는 유지 — 오탐 억제의 직접 근거다)
        · **train 800 미달이 siren·glass·scream 3종이다.** 원본 자체가 부족한
          문제라 증강(규칙 6)으로 대응하고, 학습 후 클래스별 재현율로 재판정한다
        · 산출물: `data/safesound-v1.tar.gz` (sha256 `57557e07…`),
          `MANIFEST.json`(설정·커밋 해시 포함), Colab 절차는 `colab/train_baseline.md`
        · siren train 800 은 상한을 4로 올려도 578 이라 도달 불가 —
          원본 자체가 부족하다(고유 원본 train 129). 증강 강화(규칙 6)로 대응
        · siren 상한은 3 유지 확정 (2026-09-22). 상한을 올려도 고유 원본 수가
          늘지 않아 신뢰구간이 좁아지지 않는다
        · 배경음이 이벤트 합(3,260)의 1.06배다. 규칙 4(2~3배) 미달 → 3단계에서 확장
        · dog_bark 불일치 중 최상위 라벨 `Animal` 인 1,540윈도우는 자식(Dog) 확률
          중앙값 0.034 다. 0.05 로 내리면 544윈도우가 살아난다 — 청취로 판정할 것
        · siren 불일치 중 `Alarm` 최상위 87윈도우는 Siren 확률이 0.033 으로 낮다.
          FSD50K siren 이 전량 `Alarm` 을 함께 보유하는 것과 같은 문제일 수 있다
        이전 에너지 전용 실측에서 드러난 것 (모두 rel 0.30 기준):
        · `siren` train 은 **어떤 설정에서도 800 미달**이다. 필터를 전부 끄고
          상한 3만 걸어도 797이 최대다. 상한을 4로 올리든 목표를 낮추든 선택이 필요하다
        · `siren` test 200 선은 floor −45 (204) 에서 끊긴다. −40 은 198 로 미달
        · floor 를 −40 이하로 조이면 **`glass` 가 먼저 무너진다**. 클립 전량 탈락률이
          −40 에서 glass 10.5% vs scream 0.5% — 과도음 클래스 편향이 실측으로 확인됐다
        · 따라서 후보는 floor −45 ~ −50 / rel 0.30. 최종 판정은 청취 결과로 한다
  - [ ] 캐시 생성(build) 이어하기 버그 수정 — 샤드 flush 전에 진행 상태를 저장하고
        재실행 시 `shard_0000`부터 덮어쓴다
- [ ] **3단계**: 실측 윈도우 수의 2~3배로 배경음 목표량 계산 → 배경음 클립 선별
      후 매니페스트 확장 → `--stage background`

> 배경음 샘플링을 먼저 정해야 목록이 확정되는데 이벤트 윈도우 수가 잠정이라
> 순환이 생긴다. 위 3단계 순서가 그 순환을 끊는다.

### 2.3 전처리
- [ ] `scripts/prepare_safesound.py` 작성
  - [ ] 다중라벨 → 단일라벨 필터 (타깃 클래스 정확히 1개만 채택)
  - [ ] RMS 에너지 기반 1초 윈도우 추출 + 임계값 미달 폐기
  - [ ] 44.1kHz → 16kHz mono 리샘플
  - [ ] **원본 클립 단위 분할** (FSD50K dev/eval 준수)
  - [ ] 배경음 클래스 구성 (다른 클래스 합의 2~3배, 최대한 다양하게)
  - [ ] 라벨 단위 처리 후 즉시 `data/interim/`에 저장 — **메모리 누적 금지**
  - [ ] 진행 상태 파일로 중단 후 이어하기 지원
- [ ] `scripts/dataset_stats.py` — 클래스별 샘플 수·길이 분포·필터 전후 수량 표
      → **논문 데이터셋 섹션 표로 직결**
- [ ] **윈도우 수량 실측 후 재판정** ← 현재 수치는 클립 길이 기반 상한일 뿐이다
      RMS 에너지 필터 적용 후 목표(train 800+/test 200+) 달성 여부를 다시 판정한다.
      특히 `glass`는 과도음이라 클립당 1윈도우만 남을 수 있어 상한의 절반 예상.
      미달 시 대응: 정제 완화 / 클래스 축소 / 증강 강화 중 선택 — 실측 후 결정.
- [ ] (선택) **무필터 대조 학습** — 배제 규칙을 끈 버전으로 한 번 학습해
      오탐률 차이를 측정. 정제 규칙의 정당성을 데이터로 뒷받침한다.

### 2.4 데이터로더
- [ ] `datasets/safesound.py` 작성 (`kws20.py`를 템플릿으로 하되 메모리 구조는 다르게)
  - [ ] **lazy loading** — `__getitem__`에서 필요한 샘플만 디스크에서 읽기
  - [ ] `np.memmap` 또는 개별/청크 파일. **단일 거대 `.pt` 금지**
  - [ ] 16384 샘플 패딩 → (128, 128) reshape
  - [ ] int8 양자화 [-128, 127]
  - [ ] 증강: shift ±100ms, 볼륨 0.7~1.3, MSnoise SNR 0~20dB (**학습셋만**)
  - [ ] **`siren` 전용 강한 증강** — shift ±200ms, 볼륨 0.6~1.5, SNR 0~15dB
        고유 원본 202개(train 137)뿐이라 다양성 부족을 증강으로 보완한다
  - [ ] 파일 하단 `datasets` 딕셔너리 등록
- [ ] `ai8x-training/datasets/`에 심볼릭 링크
- [ ] `train.py --dataset SafeSound`로 로딩 성공 확인
- [ ] 소규모 학습 sanity check — 혼동행렬이 대각선을 형성하는지

---

## Phase 3 — 모델 학습 및 합성 (보드 불필요)

- [x] 클래스 수에 맞춰 KWS20 v3 백본 수정 (FC 층 **21 → 5**)
      `models/ai85net-safesound.py`. 1× 165,457 params (442KB의 36.6%)
      ⚠️ 채널 스윕은 2×가 아니라 **1.5×가 8bit 상한**이다 (params가 채널에 제곱 비례)
- [ ] `.pt` 캐시를 Drive 업로드 → Colab에서 본 학습
      (**대량 학습 전 사용자 확인 필수**)
- [ ] **Colab 1 epoch 소요 시간 기록** → 실험 일정 산출 기준값
- [ ] 혼동행렬 분석 → 필요 시 데이터 필터링 규칙 조정 후 재학습
- [ ] `quantize.py` 양자화 → 정확도 저하량 측정
- [ ] `ai8xize.py` 합성 → **442KB / 512KB 제약 통과 확인**
- [ ] 합성용 샘플 입력 `sample_*.npy` 저장 (`train.py --save-sample`)
- [ ] **G9** 비트폭 스윕 8/4/2bit → 파레토 곡선의 **정확도 축 완성**
      (4/2bit는 8bit fine-tuning으로, 처음부터 학습 금지)
- [ ] **G8 대조군**: MFCC-on-M4 + 2D CNN 모델 별도 학습
- [ ] **도전 1순위**: 채널 수 0.25×/0.5×/1×/1.5× 스윕 학습 (2×는 442KB 초과)
- [ ] SNR 20/10/0dB 고정 평가셋 생성 및 정확도 측정

---

## Phase 3.5 — 호스트 측 및 문서 (보드 불필요)

- [ ] `host/fake_serial.py` — 가짜 이벤트 스트림 생성기
- [ ] `host/gui.py` — PyQt 실시간 GUI (더미 데이터로 완성 가능)
  - [ ] 실시간 클래스 확률 막대 / 이벤트 타임라인 / 프레임 드롭 카운터
  - [ ] NPU/CPU 경로 표시 및 전환 버튼
  - [ ] SD 로그 조회 탭 (기간·클래스 필터)
- [ ] UART 패킷 포맷 설계 문서화
- [ ] SD 로그 스키마 및 이벤트 병합 상태머신 로직 설계
- [ ] `scripts/analyze_power.py` — 전력 CSV + GPIO 엣지 → 추론당 에너지 적분
      (모든 전력 실험에 재사용되는 핵심 도구)
- [ ] CMSIS-NN 베이스라인 코드 작성 + x86 빌드로 수치 검증
      (보드 없이 여기까지 가능. 도착 후 컴파일만)
- [ ] `docs/proposal.md` 제안서 작성
- [ ] 관련연구 조사 — 엣지 AI 추론, CMSIS-NN 벤치마크, MAX78000 선행연구, SED

---

## 확보 / 확인 필요 (병렬 진행)

- [ ] **MAX78000FTHR 확보 경로** — 연구실 보유분? 구매? 리드타임?
- [ ] **지도교수 면담** — 주제·기여 수준 피드백, OrangePi 확장 여부
- [ ] 연구실 오실로스코프 / 벤치 전원공급기 / PPK2 유무 확인
- [ ] 데이터 전송용 Micro USB 케이블 (충전 전용 아닌 것)
- [ ] microSD Class 10, 8~32GB
- [ ] Li-Po 3.7V 500~1000mAh (JST-PH 2.0mm) — **극성 멀티미터 확인 필수**
- [ ] 액티브 부저 3.3V + 점퍼선 (선택)

---
## 여기서부터 보드 필요
---

## Phase 4 — 환경 구축 관문 (보드 도착 직후)

- [ ] **관문 A**: Windows에 MSDK 설치 → `Hello_World` 빌드 → 플래싱 → UART 확인
      ⚠️ 데이터 전송 지원 USB 케이블 사용 (충전 전용이면 DAPLink 미인식)
- [ ] **관문 C 실기 검증**: 합성된 C 프로젝트 빌드 → **known-answer test 통과**
      ⚠️ 합성 성공 ≠ 정상 동작. 이 테스트가 최우선
- [ ] **관문 D**: KWS20 데모로 온보드 마이크 실시간 인식 확인
- [ ] **무음 환경 마이크 노이즈 플로어 dBFS 실측** ← 전처리 전제의 검증
      조용한 방에서 `micBuff`(int8) 를 수천 샘플 받아 RMS·피크를 dBFS 로 환산한다.
      `SAMPLE_SCALE_FACTOR` 를 1 과 4 로 각각 빌드해 두 값을 모두 잰다
      (MSDK `kws20_demo/main.c` 의 `MicReadChunk`, CLAUDE.md 7장 참조).
      이 값으로 다시 볼 것:
      · `--floor-db` −50 — 노이즈 플로어가 이보다 높으면 하한을 올려야 한다
      · 채움 풀 레벨 — 현재는 배경 구간 원래 레벨(실측 −16~−44dBFS)을 쓴다.
        실기기 암소음과 크게 다르면 풀 선정 기준을 바꾼다
      · 랜덤 게인 ±12dB 범위 (5장 규칙 6)
      ⚠️ 이 세 값은 전부 **측정 전 잠정치**다. 논문에 확정값으로 쓰지 말 것
- [ ] **관문 E**: 계측 준비
  - [ ] DWT 사이클 카운터로 추론 지연 UART 출력
  - [ ] 추론 시작/종료 GPIO 토글 관측
  - [ ] microSD 마운트 및 읽기/쓰기
  - [ ] 배터리 단독 부팅 (⚠️ 연결 전 멀티미터로 극성 확인)

## Phase 5 — 시스템 구현 및 측정

- [ ] 합성 모델 펌웨어 이식, 마이크 실시간 파이프라인 구성
- [ ] **I2S → int8 변환에 포화(saturation) 처리 — 필수** ⚠️
      MSDK `kws20_demo/main.c:1229` 는 `micBuff[i] = sample*SAMPLE_SCALE_FACTOR/256`
      로 int32 를 int8_t 에 그냥 대입한다. 클램프가 없어 **랩어라운드**다
      (int16 10,000 → 156 → −100, 부호 반전). `HPF()` 는 int16 범위를 클리핑하지만
      이 대입은 보호하지 않는다.
      전처리 `to_int8` 은 `np.clip` 으로 포화시키므로, 펌웨어를 그대로 두면 큰
      소리에서 학습 분포와 추론 분포가 정반대가 된다.
      → `__SSAT(v, 8)` 또는 명시적 `if (v > 127) v = 127; else if (v < -128) v = -128;`
      로 구현하고, **known-answer test 에 풀스케일 초과 입력을 포함**할 것.
      근거: CLAUDE.md 7장 "실기기 스케일"
- [ ] `MEASURE_BUILD` / `DEMO_BUILD` 컴파일 플래그 분리
- [ ] 이벤트 병합 상태머신 구현
- [ ] SD 이벤트 로깅 + 부팅 시 UART 시각 동기화
- [ ] LED 색상 코딩 (+ 부저)
- [ ] CMSIS-NN 베이스라인 펌웨어 이식 (동일 가중치)
- [ ] 호스트 GUI를 실제 보드에 연결
- [ ] **G2** 지연 측정 1000회, 최악값 보고
- [ ] **G3** 에너지 측정(차분법), 여유 사이클, 메모리 수용성
- [ ] **G3** 베이스라인의 hop 주기 초과 실증 ← 핵심 결과
- [ ] **G4** 테스트셋 SD 적재 → 온디바이스 실측 혼동행렬
- [ ] **G8** 전처리 위치별 지연·에너지 비교 (①전량CPU vs ④전량NPU 최소)
- [ ] **G9** 비트폭별 에너지 측정 → 파레토 곡선 완성
- [ ] **G10** hop 125/250/500/1000ms 스윕 → 응답지연-에너지 곡선
- [ ] **도전 1순위** 모델 규모별 전처리 비중 측정
- [ ] PMIC 퓨얼게이지 로깅 → 실측 배터리 방전 곡선
- [ ] **G7** 8시간 무인 구동 → 로그 무결성 + 실환경 오탐률
      ⚠️ **`siren` 을 특히 주시할 것.** 이 클래스만 실환경에서 무너지면 원본
      다양성 부족(고유 원본 202개)이 원인일 가능성이 크다. PC 시뮬레이션과의
      괴리도 여기서 가장 클 것으로 예상한다. 그 관찰 자체가 논의 절의 재료다.

## Phase 6 — 도전 확장 (여유 있을 때만)

- [ ] **1.5순위** OrangePi 5 Pro: RKNN 툴체인 → 모델 변환 → 추론
- [ ] 3티어 전력 비교 (7장 방법론 준수, 세 지표 모두 보고)
- [ ] **2순위** RISC-V 저전력 게이팅

## Phase 7 — 논문

- [ ] 서론 / 관련연구 / 시스템 설계 / 구현 / 실험 / 논의 / 결론
- [ ] **대표 그림**: 하루 에너지 vs macro-F1 파레토, 실현 불가 영역 회색 처리
- [ ] 수치 목표를 실측값으로 캘리브레이션
- [ ] 데이터셋 라이선스 명시 (FSD50K: CC, US8K·ESC-50: 비상업 연구용)
- [ ] 재현 절차에 환경 이슈 기록(`CLAUDE.md` 12장) 반영
- [ ] 한계 및 향후 과제 (BLE, 캐스케이드, 타 NPU MCU, 장기 배치)
- [ ] 시연 리허설 + **정상 동작 영상 사전 녹화** (라이브 데모 실패 대비)

---

## 진행 로그

| 날짜 | 완료 항목 | 비고 |
|---|---|---|
| 2026-09 | Phase 0 완료 | 로컬 77분/epoch, 합성 169KB, Colab 환경 구축 |
| 2026-09-10 | Phase 1 레포 구성 | `github.com/elponchis/max78000-proj` 연결, 구조·patches 배치 |
| 2026-09-10 | Phase 2.1 라벨 조사 | 부재 라벨 5종 확인, 클립 수 집계 완료 |
| 2026-09-10 | **클래스 매핑 확정** | 5클래스(siren/glass/scream/dog_bark/background). alarm·baby_cry 제외 |
| 2026-09-10 | 데이터 누수 차단 | Freesound 원본 중복 337건 발견 → 원본 ID 단위 전역 분할 |
| 2026-09-10 | 모델 정의 | `ai85net-safesound.py` FC 21→5. 채널 스윕 상한 2×→1.5× 정정 |
| 2026-09-10 | FSD50K 1단계 다운로드 | 4,883클립 / 3.1GB, 크기 전량 일치 |
| 2026-09-11 | 매니페스트 경계 케이스 수정 | 교차 데이터셋 승격 7건 폐기, 하드 네거티브 2,477 |
| 2026-09-11 | US8K·ESC-50 다운로드 | 매니페스트분 854 / 52, Zenodo 공식 해시 150/150 일치 |
| 2026-09-11 | 전처리 민감도표 | 리샘플러 교체, 필터 병용, `energy-filter-sweep.md` |
