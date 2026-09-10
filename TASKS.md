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
- [ ] **`scream` 샘플 수십 개 직접 청취** — 수량은 603원본으로 충분, 품질 미확인
- [ ] `siren` 샘플 청취 — 고유 원본 202개로 가장 취약한 클래스

> 확정 내역 전문: `docs/results/class-mapping.md`
> `alarm` 제외 근거: `docs/results/alarm-class-analysis.md` (논문 데이터셋 절에 사용)
> **주의 1**: FSD50K는 AudioSet 조상 라벨을 함께 부여한다(`Siren`⊂`Alarm` 100%).
> **주의 2**: US8K는 원본당 여러 슬라이스다(siren 929슬라이스 = 원본 74개).
> 신뢰구간은 슬라이스가 아니라 원본 수로 계산할 것.

### 2.2 데이터 다운로드
- [ ] FSD50K 오디오 (30GB+) — **시작 전 디스크 여유 확인 및 사용자 승인 필수**
- [ ] UrbanSound8K (6GB), ESC-50 (600MB)
- [ ] 디스크 부족 시 스트리밍 처리 (파트별 받아 추출 후 원본 삭제)

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
- [ ] **관문 E**: 계측 준비
  - [ ] DWT 사이클 카운터로 추론 지연 UART 출력
  - [ ] 추론 시작/종료 GPIO 토글 관측
  - [ ] microSD 마운트 및 읽기/쓰기
  - [ ] 배터리 단독 부팅 (⚠️ 연결 전 멀티미터로 극성 확인)

## Phase 5 — 시스템 구현 및 측정

- [ ] 합성 모델 펌웨어 이식, 마이크 실시간 파이프라인 구성
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
| | | |
