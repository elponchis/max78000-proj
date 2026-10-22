# 원고 수치 대조표 — 문장 단위 (수치 · 가리키는 구성 · numbers.json 키)

원고: `docs/paper/ksc2026/ksc2026_review.docx`. 방법: 문장마다 수치 토큰을 뽑고, numbers.json 의 어느 키가 그 표기(.0f/.1f/.2f/.3f/±/%/천 단위)를 내는지 찾은 뒤, 키 경로의 구성 id 가 문장에 언급된 구성(별칭 포함)과 맞는지 본다. 참고문헌·인용 번호·표/그림 번호는 제외.

판정 집계: ✓ 215 · ⚠ 구성 확인 17 · 상수 43 · ✗ 미확인 0

| 절 | 문장 (앞 70자) | 수치 | 문장의 구성 | numbers.json 키 (또는 상수 설명) | 판정 |
|---|---|---|---|---|---|
| (제목·요약) | NPU를 탑재한 MCU(MAX78000)에서 배터리 상시 구동 음향 이벤트 감지기를 구현하고, log-mel 전처리를 어디에  | 10 | CPU 추론 | firmware.stop_wait.ms_per_decision; latency.①′.pre; listening.B_ext_glass/전체.?; listening.S_train_scream/audioset_train.X; preproc.inc_v1_ms; welch.(a) − ④.ext_ | ✓ |
| (제목·요약) | 정확도와 판단 1회 지연은 같은 데이터·같은 보드에서 실측했고, 에너지는 실측 지연과 공개 전력값으로 추정했다. | 1 | — | acc.(a).ext_f1.sd; acc.(a).ext_map.mean; acc.(a).ext_map.sd; acc.(a).f1_L.mean; acc.(a).f1_L.sd; acc.(a).f1_S3.mean; acc.(a).f1_S3.sd; acc.①′.ext_f1.mean; acc.① | ✓ |
| (제목·요약) | 전처리를 없애고 원본 파형을 NPU에 직접 넣는 벤더 방식 구성은 가장 정확한 CPU log-mel 구성(간격 16 ms)보다 | 16 | 변형 3종, 변형 3종, 변형 3종, 16 ms, 원본 파형 | 16 kHz / 16 프레임 / 16 ms | 상수 |
| (제목·요약) | 전처리를 없애고 원본 파형을 NPU에 직접 넣는 벤더 방식 구성은 가장 정확한 CPU log-mel 구성(간격 16 ms)보다 | 0.166 | 변형 3종, 변형 3종, 변형 3종, 16 ms, 원본 파형 | 유도: f1(①′)−f1(④) 시험 | ✓ |
| (제목·요약) | 반면 거친 log-mel을 CPU에 남기고 작은 2D CNN을 NPU에 올린 구성은 원본 파형 구성보다 macro-F1이 +0 | 2 | 원본 파형 | acc.①′.ext_f1.sd; acc.②′ b64.ext_f1.sd; acc.②′ b64f31.ext_f1.sd; acc.②′ b64f31.ext_map.sd; acc.②′ b64f31.f1_L.sd; acc.②′ b64f31.f1_S3.sd; acc.②′.f1_S3.sd; acc.④ | ✓ |
| (제목·요약) | 반면 거친 log-mel을 CPU에 남기고 작은 2D CNN을 NPU에 올린 구성은 원본 파형 구성보다 macro-F1이 +0 | +0.085 | 원본 파형 | welch.가 u1000 − ④.test_f1_S3[0] | ⚠ |
| (제목·요약) | 반면 거친 log-mel을 CPU에 남기고 작은 2D CNN을 NPU에 올린 구성은 원본 파형 구성보다 macro-F1이 +0 | 95 | 원본 파형 | latency.②′ b64f31.load; listening.S_train_scream/audioset_train.O | ✓ |
| (제목·요약) | 반면 거친 log-mel을 CPU에 남기고 작은 2D CNN을 NPU에 올린 구성은 원본 파형 구성보다 macro-F1이 +0 | 0 | 원본 파형 | acc.(a).ext_f1.mean; acc.(a).ext_f1.sd; acc.(a).ext_map.sd; acc.(a).f1_L.sd; acc.(a).f1_S3.sd; acc.(a).scream_S3.mean; acc.(a).scream_S3.sd; acc.①′.ext_f1.sd; a | ✓ |
| 1. 서론 | 벤더의 음성 키워드 참조 설계는 후자를 택하며, MFCC를 쓰는 소형 모델들과 비슷하거나 높은 정확도(96.3% 대 90.2~ | 96.3 | — | [4] 보고 정확도 % | 상수 |
| 1. 서론 | 벤더의 음성 키워드 참조 설계는 후자를 택하며, MFCC를 쓰는 소형 모델들과 비슷하거나 높은 정확도(96.3% 대 90.2~ | 90.2 | — | [4] 비교 하한 % | 상수 |
| 1. 서론 | 벤더의 음성 키워드 참조 설계는 후자를 택하며, MFCC를 쓰는 소형 모델들과 비슷하거나 높은 정확도(96.3% 대 90.2~ | 96.4 | — | [4] 비교 상한 % | 상수 |
| 1. 서론 | (1) 전처리 위치를 양 끝(전부 CPU, 전부 NPU)과 그 사이로 나눈 10개 구성과 CPU 추론 대조군의 정확도·판단 1 | 1 | CPU 추론 | acc.(a).ext_f1.sd; acc.(a).ext_map.mean; acc.(a).ext_map.sd; acc.(a).f1_L.mean; acc.(a).f1_L.sd; acc.(a).f1_S3.mean; acc.(a).f1_S3.sd; acc.①′.ext_f1.mean; acc.① | ✓ |
| 1. 서론 | (1) 전처리 위치를 양 끝(전부 CPU, 전부 NPU)과 그 사이로 나눈 10개 구성과 CPU 추론 대조군의 정확도·판단 1 | 10 | CPU 추론 | firmware.stop_wait.ms_per_decision; latency.①′.pre; listening.B_ext_glass/전체.?; listening.S_train_scream/audioset_train.X; preproc.inc_v1_ms; welch.(a) − ④.ext_ | ✓ |
| 1. 서론 | (1) 전처리 위치를 양 끝(전부 CPU, 전부 NPU)과 그 사이로 나눈 10개 구성과 CPU 추론 대조군의 정확도·판단 1 | 1 | CPU 추론 | acc.(a).ext_f1.sd; acc.(a).ext_map.mean; acc.(a).ext_map.sd; acc.(a).f1_L.mean; acc.(a).f1_L.sd; acc.(a).f1_S3.mean; acc.(a).f1_S3.sd; acc.①′.ext_f1.mean; acc.① | ✓ |
| 1. 서론 | (2) 필터뱅크·로그 압축·분할 등 전처리를 NPU 연산으로 옮긴 변형은 원본 파형 구성을 유의하게 넘지 못하고, 선형 부분만 | 2 | 변형 3종, 변형 3종, 변형 3종, 원본 파형 | acc.①′.ext_f1.sd; acc.②′ b64.ext_f1.sd; acc.②′ b64f31.ext_f1.sd; acc.②′ b64f31.ext_map.sd; acc.②′ b64f31.f1_L.sd; acc.②′ b64f31.f1_S3.sd; acc.②′.f1_S3.sd; acc.④ | ✓ |
| 1. 서론 | (3) 거친 log-mel(프레임 간격 62.5 ms, 멜 32)을 CPU에 남기는 구성이 원본 파형 구성보다 정확도가 유의하 | 3 | 62.5 ms, 원본 파형 | acc.(a).ext_f1.n; acc.(a).ext_map.n; acc.(a).f1_L.n; acc.(a).f1_S3.n; acc.(a).scream_S3.n; acc.(a).scream_S3.sd; acc.①′.ext_f1.n; acc.①′.ext_map.n; acc.①′.f1_L. | ✓ |
| 1. 서론 | (3) 거친 log-mel(프레임 간격 62.5 ms, 멜 32)을 CPU에 남기는 구성이 원본 파형 구성보다 정확도가 유의하 | 62.5 | 62.5 ms, 원본 파형 | 프레임 간격 ms | 상수 |
| 1. 서론 | (3) 거친 log-mel(프레임 간격 62.5 ms, 멜 32)을 CPU에 남기는 구성이 원본 파형 구성보다 정확도가 유의하 | 32 | 62.5 ms, 원본 파형 | listening.B_ext_glass/전체/verdict_first.O | ✓ |
| 1. 서론 | CPU 추론 대조군은 CMSIS-NN [8]으로 구현했고, 데이터는 FSD50K [9]·UrbanSound8K [10]·ESC | 50 | CPU 추론 | 프레임 간격 ms / CNN 50 MHz | 상수 |
| 2. 시스템과 실험 설계 | MAX78000FTHR 단독으로 온보드 마이크 16 kHz 입력을 1.024 s 창(16,384 샘플, int8)으로 받는다. | 16 | 50 ms | 16 kHz / 16 프레임 / 16 ms | 상수 |
| 2. 시스템과 실험 설계 | MAX78000FTHR 단독으로 온보드 마이크 16 kHz 입력을 1.024 s 창(16,384 샘플, int8)으로 받는다. | 1.024 | 50 ms | 창 길이 s | 상수 |
| 2. 시스템과 실험 설계 | MAX78000FTHR 단독으로 온보드 마이크 16 kHz 입력을 1.024 s 창(16,384 샘플, int8)으로 받는다. | 16,384 | 50 ms | 창 샘플 | 상수 |
| 2. 시스템과 실험 설계 | MAX78000FTHR 단독으로 온보드 마이크 16 kHz 입력을 1.024 s 창(16,384 샘플, int8)으로 받는다. | 250 | 50 ms | hop_ms | ✓ |
| 2. 시스템과 실험 설계 | MAX78000FTHR 단독으로 온보드 마이크 16 kHz 입력을 1.024 s 창(16,384 샘플, int8)으로 받는다. | 256 | 50 ms | breakeven.CPU log-mel, 간격 50 ms < 원본 파형 (전처리 없음).by_source.datasheet_upper.E_Y; breakeven.CPU log-mel, 간격 62.5 ms (멜 32) < 원본 파형 (전처리 없음).by_source.datasheet_up | ✓ |
| 2. 시스템과 실험 설계 | MAX78000FTHR 단독으로 온보드 마이크 16 kHz 입력을 1.024 s 창(16,384 샘플, int8)으로 받는다. | 5 | 50 ms | acc.②′ b64f31.scream_S3.sd; acc.②′ f31.scream_S3.sd; avg_power.configs.②′ b64f31.share_datasheet_at_hi_pct; avg_power.configs.②′ b64f31.share_min_pct; breakeven | ✓ |
| 2. 시스템과 실험 설계 | 모델은 가중치 8 bit QAT, 가속기 가중치 메모리 442 KB 안에 들어간다. | 8 | — | acc.다 u800.scream_S3.sd; avg_power.configs.(a).share_datasheet_at_hi_pct; latency.②′.detail.infer1; listening.A_ext_scream/실사용만.?; welch.(a) − ④.test_f1_L[2]; w | ✓ |
| 2. 시스템과 실험 설계 | 모델은 가중치 8 bit QAT, 가속기 가중치 메모리 442 KB 안에 들어간다. | 442 | — | 가중치 메모리 KB | 상수 |
| 2. 시스템과 실험 설계 | 시험 데이터의 유리·비명 창은 저자가 전수 청취해 양성을 확정했다(채점 창 4,965). | 4,965 | — | truth.testL_S3_windows | ✓ |
| 2. 시스템과 실험 설계 | 외부 평가는 AudioSet의 별도 영상에서 뽑은 673창(양성 271, 음성 402)으로, 학습에 쓰지 않았다. | 673 | — | 유도: 외부 창 수 (양성 합+음성) | ✓ |
| 2. 시스템과 실험 설계 | 외부 평가는 AudioSet의 별도 영상에서 뽑은 673창(양성 271, 음성 402)으로, 학습에 쓰지 않았다. | 271 | — | 유도: 외부 양성 합 | ✓ |
| 2. 시스템과 실험 설계 | 외부 평가는 AudioSet의 별도 영상에서 뽑은 673창(양성 271, 음성 402)으로, 학습에 쓰지 않았다. | 402 | — | truth.ext_negatives_real31L | ✓ |
| 2. 시스템과 실험 설계 | 오경보율은 시험 데이터 배경에서 300회/h가 되는 문턱값을 잡아 외부에도 같은 값을 썼고, 지표는 macro-F1이다. | 300 | — | 오경보/h | 상수 |
| 2. 시스템과 실험 설계 | 전처리 없음 — 원본 파형: 파형을 128×128로 접어 NPU의 1D CNN에 넣는 벤더 방식. | 128 | 원본 파형 | listening.T_test_scream/screaming_test.O | ✓ |
| 2. 시스템과 실험 설계 | 전처리 없음 — 원본 파형: 파형을 128×128로 접어 NPU의 1D CNN에 넣는 벤더 방식. | 128 | 원본 파형 | listening.T_test_scream/screaming_test.O | ✓ |
| 2. 시스템과 실험 설계 | 전처리 없음 — 원본 파형: 파형을 128×128로 접어 NPU의 1D CNN에 넣는 벤더 방식. | 1 | 원본 파형 | acc.(a).ext_f1.sd; acc.(a).ext_map.mean; acc.(a).ext_map.sd; acc.(a).f1_L.mean; acc.(a).f1_L.sd; acc.(a).f1_S3.mean; acc.(a).f1_S3.sd; acc.①′.ext_f1.mean; acc.① | ✓ |
| 2. 시스템과 실험 설계 | NPU로 옮긴 전처리 — NPU 근사 log: 로그 압축을 NPU 안의 구간선형 층으로 근사한 1D 구성(보드 미실측). | 1 | NPU 근사 log | acc.(a).ext_f1.sd; acc.(a).ext_map.mean; acc.(a).ext_map.sd; acc.(a).f1_L.mean; acc.(a).f1_L.sd; acc.(a).f1_S3.mean; acc.(a).f1_S3.sd; acc.①′.ext_f1.mean; acc.① | ✓ |
| 2. 시스템과 실험 설계 | NPU 전처리 (Gabor 필터뱅크): 512탭 Gabor 필터(cos·sin 32쌍)를 NPU의 Conv1d로 두고 에너지  | 512 | 62.5 ms, NPU 전처리 (Gabor 필터뱅크) | 512탭 / 512 값 | 상수 |
| 2. 시스템과 실험 설계 | NPU 전처리 (Gabor 필터뱅크): 512탭 Gabor 필터(cos·sin 32쌍)를 NPU의 Conv1d로 두고 에너지  | 32 | 62.5 ms, NPU 전처리 (Gabor 필터뱅크) | listening.B_ext_glass/전체/verdict_first.O | ✓ |
| 2. 시스템과 실험 설계 | NPU 전처리 (Gabor 필터뱅크): 512탭 Gabor 필터(cos·sin 32쌍)를 NPU의 Conv1d로 두고 에너지  | 8 | 62.5 ms, NPU 전처리 (Gabor 필터뱅크) | acc.다 u800.scream_S3.sd; avg_power.configs.(a).share_datasheet_at_hi_pct; latency.②′.detail.infer1; listening.A_ext_scream/실사용만.?; welch.(a) − ④.test_f1_L[2]; w | ✓ |
| 2. 시스템과 실험 설계 | NPU 전처리 (Gabor 필터뱅크): 512탭 Gabor 필터(cos·sin 32쌍)를 NPU의 Conv1d로 두고 에너지  | 32 | 62.5 ms, NPU 전처리 (Gabor 필터뱅크) | listening.B_ext_glass/전체/verdict_first.O | ✓ |
| 2. 시스템과 실험 설계 | NPU 전처리 (Gabor 필터뱅크): 512탭 Gabor 필터(cos·sin 32쌍)를 NPU의 Conv1d로 두고 에너지  | 16 | 62.5 ms, NPU 전처리 (Gabor 필터뱅크) | 16 kHz / 16 프레임 / 16 ms | 상수 |
| 2. 시스템과 실험 설계 | NPU 전처리 (Gabor 필터뱅크): 512탭 Gabor 필터(cos·sin 32쌍)를 NPU의 Conv1d로 두고 에너지  | 62.5 | 62.5 ms, NPU 전처리 (Gabor 필터뱅크) | 프레임 간격 ms | 상수 |
| 2. 시스템과 실험 설계 | NPU 전처리 (Gabor 필터뱅크): 512탭 Gabor 필터(cos·sin 32쌍)를 NPU의 Conv1d로 두고 에너지  | 2 | 62.5 ms, NPU 전처리 (Gabor 필터뱅크) | acc.①′.ext_f1.sd; acc.②′ b64.ext_f1.sd; acc.②′ b64f31.ext_f1.sd; acc.②′ b64f31.ext_map.sd; acc.②′ b64f31.f1_L.sd; acc.②′ b64f31.f1_S3.sd; acc.②′.f1_S3.sd; acc.④ | ✓ |
| 2. 시스템과 실험 설계 | NPU 전처리 (Gabor 필터뱅크): 512탭 Gabor 필터(cos·sin 32쌍)를 NPU의 Conv1d로 두고 에너지  | 256 | 62.5 ms, NPU 전처리 (Gabor 필터뱅크) | breakeven.CPU log-mel, 간격 50 ms < 원본 파형 (전처리 없음).by_source.datasheet_upper.E_Y; breakeven.CPU log-mel, 간격 62.5 ms (멜 32) < 원본 파형 (전처리 없음).by_source.datasheet_up | ✓ |
| 2. 시스템과 실험 설계 | 변형 3종은 Gabor 64쌍(64 대역), 적분 간격 절반(31 프레임), 둘 다 적용이다. | 3 | 변형 3종, 변형 3종, 변형 3종, NPU 전처리 (Gabor 필터뱅크) | acc.(a).ext_f1.n; acc.(a).ext_map.n; acc.(a).f1_L.n; acc.(a).f1_S3.n; acc.(a).scream_S3.n; acc.(a).scream_S3.sd; acc.①′.ext_f1.n; acc.①′.ext_map.n; acc.①′.f1_L. | ✓ |
| 2. 시스템과 실험 설계 | 변형 3종은 Gabor 64쌍(64 대역), 적분 간격 절반(31 프레임), 둘 다 적용이다. | 64 | 변형 3종, 변형 3종, 변형 3종, NPU 전처리 (Gabor 필터뱅크) | acc.②′.f1_S3.mean; panns.ext_real31L_test300.recall[1] | ✓ |
| 2. 시스템과 실험 설계 | 변형 3종은 Gabor 64쌍(64 대역), 적분 간격 절반(31 프레임), 둘 다 적용이다. | 64 | 변형 3종, 변형 3종, 변형 3종, NPU 전처리 (Gabor 필터뱅크) | acc.②′.f1_S3.mean; panns.ext_real31L_test300.recall[1] | ✓ |
| 2. 시스템과 실험 설계 | 변형 3종은 Gabor 64쌍(64 대역), 적분 간격 절반(31 프레임), 둘 다 적용이다. | 31 | 변형 3종, 변형 3종, 변형 3종, NPU 전처리 (Gabor 필터뱅크) | latency.②′ b64f31.detail.load1; latency.②′ b64f31.log32.session_new.load1; latency.②′ b64f31.log32.session_old.load1; latency.②′ f31.detail.load1; latency.②′ f3 | ✓ |
| 2. 시스템과 실험 설계 | CPU 전처리 — CPU가 증분 log-mel을 계산하고 2D CNN을 NPU에 올린다(프레임 간격 62.5, 50, 25,  | 2 | 16 ms | acc.①′.ext_f1.sd; acc.②′ b64.ext_f1.sd; acc.②′ b64f31.ext_f1.sd; acc.②′ b64f31.ext_map.sd; acc.②′ b64f31.f1_L.sd; acc.②′ b64f31.f1_S3.sd; acc.②′.f1_S3.sd; acc.④ | ✓ |
| 2. 시스템과 실험 설계 | CPU 전처리 — CPU가 증분 log-mel을 계산하고 2D CNN을 NPU에 올린다(프레임 간격 62.5, 50, 25,  | 62.5 | 16 ms | 프레임 간격 ms | 상수 |
| 2. 시스템과 실험 설계 | CPU 전처리 — CPU가 증분 log-mel을 계산하고 2D CNN을 NPU에 올린다(프레임 간격 62.5, 50, 25,  | 50 | 16 ms | 프레임 간격 ms / CNN 50 MHz | 상수 |
| 2. 시스템과 실험 설계 | CPU 전처리 — CPU가 증분 log-mel을 계산하고 2D CNN을 NPU에 올린다(프레임 간격 62.5, 50, 25,  | 25 | 16 ms | listening.A_ext_scream/v1+실사용.n | ✓ |
| 2. 시스템과 실험 설계 | CPU 전처리 — CPU가 증분 log-mel을 계산하고 2D CNN을 NPU에 올린다(프레임 간격 62.5, 50, 25,  | 16 | 16 ms | 16 kHz / 16 프레임 / 16 ms | 상수 |
| 2. 시스템과 실험 설계 | CPU 전처리 — CPU가 증분 log-mel을 계산하고 2D CNN을 NPU에 올린다(프레임 간격 62.5, 50, 25,  | 32 | 16 ms | listening.B_ext_glass/전체/verdict_first.O | ✓ |
| 2. 시스템과 실험 설계 | CPU 전처리 — CPU가 증분 log-mel을 계산하고 2D CNN을 NPU에 올린다(프레임 간격 62.5, 50, 25,  | 32 | 16 ms | listening.B_ext_glass/전체/verdict_first.O | ✓ |
| 2. 시스템과 실험 설계 | CPU 전처리 — CPU가 증분 log-mel을 계산하고 2D CNN을 NPU에 올린다(프레임 간격 62.5, 50, 25,  | 64 | 16 ms | acc.②′.f1_S3.mean; panns.ext_real31L_test300.recall[1] | ✓ |
| 2. 시스템과 실험 설계 | CPU 전처리 — CPU가 증분 log-mel을 계산하고 2D CNN을 NPU에 올린다(프레임 간격 62.5, 50, 25,  | 64 | 16 ms | acc.②′.f1_S3.mean; panns.ext_real31L_test300.recall[1] | ✓ |
| 2. 시스템과 실험 설계 | 모든 구성의 전처리는 판단마다 새 프레임만 계산하는 증분 방식이며 일괄 계산과 비트 일치함을 보드에서 확인했고, 모든 구성은  | 150 | — | 에폭 | 상수 |
| 2. 시스템과 실험 설계 | 모든 구성의 전처리는 판단마다 새 프레임만 계산하는 증분 방식이며 일괄 계산과 비트 일치함을 보드에서 확인했고, 모든 구성은  | 3 | — | acc.(a).ext_f1.n; acc.(a).ext_map.n; acc.(a).f1_L.n; acc.(a).f1_S3.n; acc.(a).scream_S3.n; acc.(a).scream_S3.sd; acc.①′.ext_f1.n; acc.①′.ext_map.n; acc.①′.f1_L. | ✓ |
| 2. 시스템과 실험 설계 | 지연은 보드에서 SysTick(코어 클럭 사이클, 분해능 10 ns)으로 1,000회 재어 최악값을 보고한다(전원 투입 직후, | 10 | CPU 추론 | firmware.stop_wait.ms_per_decision; latency.①′.pre; listening.B_ext_glass/전체.?; listening.S_train_scream/audioset_train.X; preproc.inc_v1_ms; welch.(a) − ④.ext_ | ✓ |
| 2. 시스템과 실험 설계 | 지연은 보드에서 SysTick(코어 클럭 사이클, 분해능 10 ns)으로 1,000회 재어 최악값을 보고한다(전원 투입 직후, | 1,000 | CPU 추론 | firmware.cpu_split.n_iter; latency.②′ b64.log32.n_iter; latency.②′ b64f31.log32.n_iter; latency.②′ f31.log32.n_iter; latency.②′.log32.n_iter; latency.②′.new_fir | ✓ |
| 2. 시스템과 실험 설계 | 지연은 보드에서 SysTick(코어 클럭 사이클, 분해능 10 ns)으로 1,000회 재어 최악값을 보고한다(전원 투입 직후, | 100 | CPU 추론 | firmware.cpu_split.sum_vs_cpu_pct; listening.S_train_scream/screaming_train.n | ✓ |
| 2. 시스템과 실험 설계 | 지연은 보드에서 SysTick(코어 클럭 사이클, 분해능 10 ns)으로 1,000회 재어 최악값을 보고한다(전원 투입 직후, | 50 | CPU 추론 | 프레임 간격 ms / CNN 50 MHz | 상수 |
| 2. 시스템과 실험 설계 | 지연은 보드에서 SysTick(코어 클럭 사이클, 분해능 10 ns)으로 1,000회 재어 최악값을 보고한다(전원 투입 직후, | 24 | CPU 추론 | SysTick bit | 상수 |
| 2. 시스템과 실험 설계 | 지연은 보드에서 SysTick(코어 클럭 사이클, 분해능 10 ns)으로 1,000회 재어 최악값을 보고한다(전원 투입 직후, | 168 | CPU 추론 | SysTick 한계 ms | 상수 |
| 2. 시스템과 실험 설계 | 지연은 보드에서 SysTick(코어 클럭 사이클, 분해능 10 ns)으로 1,000회 재어 최악값을 보고한다(전원 투입 직후, | 1 | CPU 추론 | acc.(a).ext_f1.sd; acc.(a).ext_map.mean; acc.(a).ext_map.sd; acc.(a).f1_L.mean; acc.(a).f1_L.sd; acc.(a).f1_S3.mean; acc.(a).f1_S3.sd; acc.①′.ext_f1.mean; acc.① | ✓ |
| 2. 시스템과 실험 설계 | 변형 3종의 지연은 펌웨어 안정화 대기(NPU 패스당 약 49 μs)를 포함하며, 같은 대기를 넣으면 NPU 전처리 (Gabo | 3 | 변형 3종, 변형 3종, 변형 3종, NPU 전처리 (Gabor 필터뱅크) | acc.(a).ext_f1.n; acc.(a).ext_map.n; acc.(a).f1_L.n; acc.(a).f1_S3.n; acc.(a).scream_S3.n; acc.(a).scream_S3.sd; acc.①′.ext_f1.n; acc.①′.ext_map.n; acc.①′.f1_L. | ✓ |
| 2. 시스템과 실험 설계 | 변형 3종의 지연은 펌웨어 안정화 대기(NPU 패스당 약 49 μs)를 포함하며, 같은 대기를 넣으면 NPU 전처리 (Gabo | 49 | 변형 3종, 변형 3종, 변형 3종, NPU 전처리 (Gabor 필터뱅크) | firmware.stop_wait.us_per_pass; listening.A_ext_scream/실사용만.n; listening.S_train_scream/yellshout_test.X; variants.②′.measured.②′ f31.cpu_share_of_delta | ✓ |
| 2. 시스템과 실험 설계 | 변형 3종의 지연은 펌웨어 안정화 대기(NPU 패스당 약 49 μs)를 포함하며, 같은 대기를 넣으면 NPU 전처리 (Gabo | 1.95 | 변형 3종, 변형 3종, 변형 3종, NPU 전처리 (Gabor 필터뱅크) | firmware.cpu_split.total_ms_this_run; firmware.stop_wait.fe_total_with_wait_ms; latency.②′.log32.session_new.total; latency.②′.new_firmware.detail.total; latenc | ✓ |
| 2. 시스템과 실험 설계 | 정확도는 시드 1~3 평균 ± 표준편차(그림 1의 세로 막대), 오경보 300회/h. | 1 | — | acc.(a).ext_f1.sd; acc.(a).ext_map.mean; acc.(a).ext_map.sd; acc.(a).f1_L.mean; acc.(a).f1_L.sd; acc.(a).f1_S3.mean; acc.(a).f1_S3.sd; acc.①′.ext_f1.mean; acc.① | ✓ |
| 2. 시스템과 실험 설계 | 정확도는 시드 1~3 평균 ± 표준편차(그림 1의 세로 막대), 오경보 300회/h. | 3 | — | acc.(a).ext_f1.n; acc.(a).ext_map.n; acc.(a).f1_L.n; acc.(a).f1_S3.n; acc.(a).scream_S3.n; acc.(a).scream_S3.sd; acc.①′.ext_f1.n; acc.①′.ext_map.n; acc.①′.f1_L. | ✓ |
| 2. 시스템과 실험 설계 | 정확도는 시드 1~3 평균 ± 표준편차(그림 1의 세로 막대), 오경보 300회/h. | 300 | — | 오경보/h | 상수 |
| 2. 시스템과 실험 설계 | 지연은 보드 1,000회 최악값(NPU 근사 log는 보드 미실측, 합성 사이클로 추정: 그림 1의 *), 에너지는 추정(실측 | 1,000 | NPU 근사 log | firmware.cpu_split.n_iter; latency.②′ b64.log32.n_iter; latency.②′ b64f31.log32.n_iter; latency.②′ f31.log32.n_iter; latency.②′.log32.n_iter; latency.②′.new_fir | ✓ |
| 2. 시스템과 실험 설계 | 변형 3종은 펌웨어 안정화 대기 포함(2절). | 3 | 변형 3종, 변형 3종, 변형 3종 | acc.(a).ext_f1.n; acc.(a).ext_map.n; acc.(a).f1_L.n; acc.(a).f1_S3.n; acc.(a).scream_S3.n; acc.(a).scream_S3.sd; acc.①′.ext_f1.n; acc.①′.ext_map.n; acc.①′.f1_L. | ✓ |
| 2. 시스템과 실험 설계 | 변형 3종은 펌웨어 안정화 대기 포함(2절). | 2 | 변형 3종, 변형 3종, 변형 3종 | acc.①′.ext_f1.sd; acc.②′ b64.ext_f1.sd; acc.②′ b64f31.ext_f1.sd; acc.②′ b64f31.ext_map.sd; acc.②′ b64f31.f1_L.sd; acc.②′ b64f31.f1_S3.sd; acc.②′.f1_S3.sd; acc.④ | ✓ |
| 2. 시스템과 실험 설계 | NPU 전처리 계열의 정수 로그는 32비트 구현(64비트 구현과 출력 비트 일치, 교체분은 보드 실측 차이로 반영). | 32 | — | listening.B_ext_glass/전체/verdict_first.O | ✓ |
| 2. 시스템과 실험 설계 | NPU 전처리 계열의 정수 로그는 32비트 구현(64비트 구현과 출력 비트 일치, 교체분은 보드 실측 차이로 반영). | 64 | — | acc.②′.f1_S3.mean; panns.ext_real31L_test300.recall[1] | ✓ |
| 2. 시스템과 실험 설계 | [표] 구성 \| 판단 1회 지연 (ms) \| 에너지 μJ, 추정 \| macro-F1 시험 데이터 \| F1 외부 데이터 | 1 | — | acc.(a).ext_f1.sd; acc.(a).ext_map.mean; acc.(a).ext_map.sd; acc.(a).f1_L.mean; acc.(a).f1_L.sd; acc.(a).f1_S3.mean; acc.(a).f1_S3.sd; acc.①′.ext_f1.mean; acc.① | ✓ |
| 2. 시스템과 실험 설계 | [표] 원본 파형 (전처리 없음) \| 2.53 \| 71 / 130 / 78 \| 0.669 ± 0.025 \| 0.443 ± 0. | 2.53 | 원본 파형 | latency.④.total | ✓ |
| 2. 시스템과 실험 설계 | [표] 원본 파형 (전처리 없음) \| 2.53 \| 71 / 130 / 78 \| 0.669 ± 0.025 \| 0.443 ± 0. | 71 | 원본 파형 | avg_power.configs.②′ f31.E_min_uJ; avg_power.configs.④.E_min_uJ; breakeven.CPU log-mel, 간격 50 ms < 원본 파형 (전처리 없음).by_source.datasheet.E_Y; breakeven.CPU log-mel | ✓ |
| 2. 시스템과 실험 설계 | [표] 원본 파형 (전처리 없음) \| 2.53 \| 71 / 130 / 78 \| 0.669 ± 0.025 \| 0.443 ± 0. | 130 | 원본 파형 | avg_power.configs.④.E_max_uJ; breakeven.CPU log-mel, 간격 50 ms < 원본 파형 (전처리 없음).by_source.literature.E_Y; breakeven.CPU log-mel, 간격 62.5 ms (멜 32) < 원본 파형 (전처리 없 | ✓ |
| 2. 시스템과 실험 설계 | [표] 원본 파형 (전처리 없음) \| 2.53 \| 71 / 130 / 78 \| 0.669 ± 0.025 \| 0.443 ± 0. | 78 | 원본 파형 | breakeven.CPU log-mel, 간격 50 ms < 원본 파형 (전처리 없음).by_source.moosmann.E_Y; breakeven.CPU log-mel, 간격 62.5 ms (멜 32) < 원본 파형 (전처리 없음).by_source.moosmann.E_X; break | ✓ |
| 2. 시스템과 실험 설계 | [표] 원본 파형 (전처리 없음) \| 2.53 \| 71 / 130 / 78 \| 0.669 ± 0.025 \| 0.443 ± 0. | 0.669 | 원본 파형 | acc.④.f1_S3.mean; latency.간격 400.load | ✓ |
| 2. 시스템과 실험 설계 | [표] 원본 파형 (전처리 없음) \| 2.53 \| 71 / 130 / 78 \| 0.669 ± 0.025 \| 0.443 ± 0. | 0.025 | 원본 파형 | acc.②′.f1_S3.sd; acc.④.f1_S3.sd; acc.다 u800.ext_f1.sd; latency.②′ f31.detail.cfg1; latency.②′ f31.log32.session_new.cfg1; variants.②′.measured.②′ f31.detail_ms. | ✓ |
| 2. 시스템과 실험 설계 | [표] 원본 파형 (전처리 없음) \| 2.53 \| 71 / 130 / 78 \| 0.669 ± 0.025 \| 0.443 ± 0. | 0.443 | 원본 파형 | acc.④.ext_f1.mean; editions.ext_f1_real31L.④.v4+D.mean | ✓ |
| 2. 시스템과 실험 설계 | [표] 원본 파형 (전처리 없음) \| 2.53 \| 71 / 130 / 78 \| 0.669 ± 0.025 \| 0.443 ± 0. | 0.021 | 원본 파형 | acc.②′ b64.ext_f1.sd; acc.②′ b64f31.f1_S3.sd; acc.④.ext_f1.sd; editions.ext_f1_real31L.④.v4+D.sd | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 근사 log \| 6.59 (추정) \| 236 / 421 / 223 \| 0.703 ± 0.007 \| 0.500 ± | 6.59 | NPU 근사 log | latency.(a).total | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 근사 log \| 6.59 (추정) \| 236 / 421 / 223 \| 0.703 ± 0.007 \| 0.500 ± | 236 | NPU 근사 log | breakeven.NPU 근사 log < 원본 파형 (전처리 없음).by_source.datasheet.E_X; energy.(a).datasheet | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 근사 log \| 6.59 (추정) \| 236 / 421 / 223 \| 0.703 ± 0.007 \| 0.500 ± | 421 | NPU 근사 log | avg_power.configs.(a).E_max_uJ; breakeven.NPU 근사 log < 원본 파형 (전처리 없음).by_source.literature.E_X; energy.(a).literature | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 근사 log \| 6.59 (추정) \| 236 / 421 / 223 \| 0.703 ± 0.007 \| 0.500 ± | 223 | NPU 근사 log | avg_power.configs.(a).E_min_uJ; breakeven.NPU 근사 log < 원본 파형 (전처리 없음).by_source.moosmann.E_X; energy.(a).moosmann | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 근사 log \| 6.59 (추정) \| 236 / 421 / 223 \| 0.703 ± 0.007 \| 0.500 ± | 0.703 | NPU 근사 log | acc.(a).f1_S3.mean | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 근사 log \| 6.59 (추정) \| 236 / 421 / 223 \| 0.703 ± 0.007 \| 0.500 ± | 0.007 | NPU 근사 log | acc.(a).f1_L.sd; acc.(a).f1_S3.sd; acc.②′.ext_map.sd; acc.④.ext_map.sd; editions.ext_f1_real31L.간격 400.v1.sd; hop_fraction.②′; latency.②′ b64f31.detail.unload;  | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 근사 log \| 6.59 (추정) \| 236 / 421 / 223 \| 0.703 ± 0.007 \| 0.500 ± | 0.500 | NPU 근사 log | acc.(a).ext_f1.mean; variants.②′.measured.②′ f31.delta_vs_fe_ms.infer2 | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 근사 log \| 6.59 (추정) \| 236 / 421 / 223 \| 0.703 ± 0.007 \| 0.500 ± | 0.011 | NPU 근사 log | acc.(a).ext_f1.sd; acc.②′ f31.f1_S3.sd; acc.가 u1000.ext_map.sd; acc.가 u1000.f1_L.sd | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 (Gabor 필터뱅크) \| 1.85 \| 39 / 74 / 52 \| 0.643 ± 0.025 \| 0.460 | 1.85 | NPU 전처리 (Gabor 필터뱅크) | latency.②′.detail.total; latency.②′.total | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 (Gabor 필터뱅크) \| 1.85 \| 39 / 74 / 52 \| 0.643 ± 0.025 \| 0.460 | 39 | NPU 전처리 (Gabor 필터뱅크) | acc.(a).scream_S3.mean; acc.②′ b64.scream_S3.mean; avg_power.configs.②′.E_min_uJ; breakeven.NPU 전처리 (Gabor 필터뱅크) < CPU log-mel, 간격 62.5 ms (멜 32).by_source.data | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 (Gabor 필터뱅크) \| 1.85 \| 39 / 74 / 52 \| 0.643 ± 0.025 \| 0.460 | 74 | NPU 전처리 (Gabor 필터뱅크) | avg_power.configs.②′.E_max_uJ; breakeven.NPU 전처리 (Gabor 필터뱅크) < CPU log-mel, 간격 62.5 ms (멜 32).by_source.literature.E_X; breakeven.NPU 전처리 (Gabor 필터뱅크) < 원본 파형  | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 (Gabor 필터뱅크) \| 1.85 \| 39 / 74 / 52 \| 0.643 ± 0.025 \| 0.460 | 52 | NPU 전처리 (Gabor 필터뱅크) | breakeven.NPU 전처리 (Gabor 필터뱅크) < CPU log-mel, 간격 62.5 ms (멜 32).by_source.moosmann.E_X; breakeven.NPU 전처리 (Gabor 필터뱅크) < 원본 파형 (전처리 없음).by_source.moosmann.E_X;  | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 (Gabor 필터뱅크) \| 1.85 \| 39 / 74 / 52 \| 0.643 ± 0.025 \| 0.460 | 0.643 | NPU 전처리 (Gabor 필터뱅크) | acc.②′.f1_S3.mean | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 (Gabor 필터뱅크) \| 1.85 \| 39 / 74 / 52 \| 0.643 ± 0.025 \| 0.460 | 0.025 | NPU 전처리 (Gabor 필터뱅크) | acc.②′.f1_S3.sd; acc.④.f1_S3.sd; acc.다 u800.ext_f1.sd; latency.②′ f31.detail.cfg1; latency.②′ f31.log32.session_new.cfg1; variants.②′.measured.②′ f31.detail_ms. | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 (Gabor 필터뱅크) \| 1.85 \| 39 / 74 / 52 \| 0.643 ± 0.025 \| 0.460 | 0.460 | NPU 전처리 (Gabor 필터뱅크) | acc.②′.ext_f1.mean | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 (Gabor 필터뱅크) \| 1.85 \| 39 / 74 / 52 \| 0.643 ± 0.025 \| 0.460 | 0.034 | NPU 전처리 (Gabor 필터뱅크) | acc.②′.ext_f1.sd; welch.(a) − ④.test_f1_S3[0] | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 64 대역 \| 3.38 \| 77 / 145 / 98 \| 0.687 ± 0.005 \| 0.504 ± 0.0 | 64 | 변형 3종 | acc.②′.f1_S3.mean; panns.ext_real31L_test300.recall[1] | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 64 대역 \| 3.38 \| 77 / 145 / 98 \| 0.687 ± 0.005 \| 0.504 ± 0.0 | 3.38 | 변형 3종 | latency.②′ b64.detail.total; latency.②′ b64.total; variants.②′.measured.②′ b64.detail_ms.total; variants.②′.measured.②′ b64.total_ms | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 64 대역 \| 3.38 \| 77 / 145 / 98 \| 0.687 ± 0.005 \| 0.504 ± 0.0 | 77 | 변형 3종 | avg_power.configs.②′ b64.E_min_uJ; energy.②′ b64.datasheet; variants.②′.measured.②′ b64.E_uJ.datasheet | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 64 대역 \| 3.38 \| 77 / 145 / 98 \| 0.687 ± 0.005 \| 0.504 ± 0.0 | 145 | 변형 3종 | avg_power.configs.②′ b64.E_max_uJ; energy.②′ b64.literature; variants.②′.measured.②′ b64.E_uJ.literature | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 64 대역 \| 3.38 \| 77 / 145 / 98 \| 0.687 ± 0.005 \| 0.504 ± 0.0 | 98 | 변형 3종 | avg_power.configs.가 u1000.E_max_uJ; breakeven.CPU log-mel, 간격 50 ms < 원본 파형 (전처리 없음).by_source.moosmann.E_X; breakeven.CPU log-mel, 간격 62.5 ms (멜 32) < 원본 파형 (전 | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 64 대역 \| 3.38 \| 77 / 145 / 98 \| 0.687 ± 0.005 \| 0.504 ± 0.0 | 0.687 | 변형 3종 | acc.②′ b64.f1_S3.mean; variants.②′.f1_S3.mean.②′ b64; variants.②′.measured.②′ f31.delta_vs_fe_ms.cpu | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 64 대역 \| 3.38 \| 77 / 145 / 98 \| 0.687 ± 0.005 \| 0.504 ± 0.0 | 0.005 | 변형 3종 | acc.(a).ext_map.sd; acc.②′ b64.f1_S3.sd; latency.가 u1000.new_firmware.main_detail.out; latency.가 u1000.new_firmware.out; latency.가 u1000.out; variants.②′.measur | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 64 대역 \| 3.38 \| 77 / 145 / 98 \| 0.687 ± 0.005 \| 0.504 ± 0.0 | 0.504 | 변형 3종 | acc.②′ b64.ext_f1.mean | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 64 대역 \| 3.38 \| 77 / 145 / 98 \| 0.687 ± 0.005 \| 0.504 ± 0.0 | 0.021 | 변형 3종 | acc.②′ b64.ext_f1.sd; acc.②′ b64f31.f1_S3.sd; acc.④.ext_f1.sd; editions.ext_f1_real31L.④.v4+D.sd | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 31 프레임 \| 3.25 \| 71 / 136 / 93 \| 0.655 ± 0.011 \| 0.465 ± 0. | 31 | 변형 3종 | latency.②′ b64f31.detail.load1; latency.②′ b64f31.log32.session_new.load1; latency.②′ b64f31.log32.session_old.load1; latency.②′ f31.detail.load1; latency.②′ f3 | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 31 프레임 \| 3.25 \| 71 / 136 / 93 \| 0.655 ± 0.011 \| 0.465 ± 0. | 3.25 | 변형 3종 | latency.②′ f31.detail.total; latency.②′ f31.total; variants.②′.measured.②′ f31.detail_ms.total; variants.②′.measured.②′ f31.total_ms | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 31 프레임 \| 3.25 \| 71 / 136 / 93 \| 0.655 ± 0.011 \| 0.465 ± 0. | 71 | 변형 3종 | avg_power.configs.②′ f31.E_min_uJ; avg_power.configs.④.E_min_uJ; breakeven.CPU log-mel, 간격 50 ms < 원본 파형 (전처리 없음).by_source.datasheet.E_Y; breakeven.CPU log-mel | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 31 프레임 \| 3.25 \| 71 / 136 / 93 \| 0.655 ± 0.011 \| 0.465 ± 0. | 136 | 변형 3종 | avg_power.configs.②′ f31.E_max_uJ; avg_power.configs.②′.share_datasheet_at_hi_pct; avg_power.configs.②′.share_min_pct; energy.②′ f31.literature; variants.②′.mea | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 31 프레임 \| 3.25 \| 71 / 136 / 93 \| 0.655 ± 0.011 \| 0.465 ± 0. | 93 | 변형 3종 | energy.②′ f31.moosmann; variants.②′.measured.②′ f31.E_uJ.moosmann | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 31 프레임 \| 3.25 \| 71 / 136 / 93 \| 0.655 ± 0.011 \| 0.465 ± 0. | 0.655 | 변형 3종 | acc.②′ f31.f1_S3.mean; variants.②′.f1_S3.mean.②′ f31; variants.②′.f1_S3.min | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 31 프레임 \| 3.25 \| 71 / 136 / 93 \| 0.655 ± 0.011 \| 0.465 ± 0. | 0.011 | 변형 3종 | acc.(a).ext_f1.sd; acc.②′ f31.f1_S3.sd; acc.가 u1000.ext_map.sd; acc.가 u1000.f1_L.sd | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 31 프레임 \| 3.25 \| 71 / 136 / 93 \| 0.655 ± 0.011 \| 0.465 ± 0. | 0.465 | 변형 3종 | acc.②′ f31.ext_f1.mean | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 31 프레임 \| 3.25 \| 71 / 136 / 93 \| 0.655 ± 0.011 \| 0.465 ± 0. | 0.006 | 변형 3종 | acc.②′ f31.ext_f1.sd; acc.간격 400.f1_L.sd; acc.간격 400.f1_S3.sd; latency.②′.detail.unload; latency.②′.log32.session_old.unload; latency.②′.out; latency.간격 400.out | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 64 대역·31 프레임 \| 5.96 \| 133 / 252 / 172 \| 0.690 ± 0.021 \| 0. | 64 | 변형 3종, 변형 3종 | acc.②′.f1_S3.mean; panns.ext_real31L_test300.recall[1] | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 64 대역·31 프레임 \| 5.96 \| 133 / 252 / 172 \| 0.690 ± 0.021 \| 0. | 31 | 변형 3종, 변형 3종 | latency.②′ b64f31.detail.load1; latency.②′ b64f31.log32.session_new.load1; latency.②′ b64f31.log32.session_old.load1; latency.②′ f31.detail.load1; latency.②′ f3 | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 64 대역·31 프레임 \| 5.96 \| 133 / 252 / 172 \| 0.690 ± 0.021 \| 0. | 5.96 | 변형 3종, 변형 3종 | latency.②′ b64f31.detail.total; latency.②′ b64f31.total; variants.②′.measured.②′ b64f31.detail_ms.total; variants.②′.measured.②′ b64f31.total_ms | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 64 대역·31 프레임 \| 5.96 \| 133 / 252 / 172 \| 0.690 ± 0.021 \| 0. | 133 | 변형 3종, 변형 3종 | avg_power.configs.②′ b64f31.E_min_uJ; energy.②′ b64f31.datasheet; variants.②′.measured.②′ b64f31.E_uJ.datasheet | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 64 대역·31 프레임 \| 5.96 \| 133 / 252 / 172 \| 0.690 ± 0.021 \| 0. | 252 | 변형 3종, 변형 3종 | avg_power.configs.②′ b64f31.E_max_uJ; energy.②′ b64f31.literature; variants.②′.measured.②′ b64f31.E_uJ.literature | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 64 대역·31 프레임 \| 5.96 \| 133 / 252 / 172 \| 0.690 ± 0.021 \| 0. | 172 | 변형 3종, 변형 3종 | energy.②′ b64f31.moosmann; variants.②′.measured.②′ b64f31.E_uJ.moosmann | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 64 대역·31 프레임 \| 5.96 \| 133 / 252 / 172 \| 0.690 ± 0.021 \| 0. | 0.690 | 변형 3종, 변형 3종 | acc.②′ b64f31.f1_S3.mean; variants.②′.f1_S3.max; variants.②′.f1_S3.mean.②′ b64f31 | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 64 대역·31 프레임 \| 5.96 \| 133 / 252 / 172 \| 0.690 ± 0.021 \| 0. | 0.021 | 변형 3종, 변형 3종 | acc.②′ b64.ext_f1.sd; acc.②′ b64f31.f1_S3.sd; acc.④.ext_f1.sd; editions.ext_f1_real31L.④.v4+D.sd | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 64 대역·31 프레임 \| 5.96 \| 133 / 252 / 172 \| 0.690 ± 0.021 \| 0. | 0.508 | 변형 3종, 변형 3종 | acc.②′ b64f31.ext_f1.mean | ✓ |
| 2. 시스템과 실험 설계 | [표] NPU 전처리 64 대역·31 프레임 \| 5.96 \| 133 / 252 / 172 \| 0.690 ± 0.021 \| 0. | 0.023 | 변형 3종, 변형 3종 | acc.②′ b64f31.ext_f1.sd; latency.②′.detail.cfg1; welch.(a) − ④.ext_map[1]; welch.가 u1000 − (a).ext_map[2]; welch.가 u1000 − (a).test_f1_S3[1] | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 62.5 ms (멜 32) \| 2.93 \| 43 / 98 / 78 \| 0.754 ± 0.0 | 62.5 | 62.5 ms | 프레임 간격 ms | 상수 |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 62.5 ms (멜 32) \| 2.93 \| 43 / 98 / 78 \| 0.754 ± 0.0 | 32 | 62.5 ms | listening.B_ext_glass/전체/verdict_first.O | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 62.5 ms (멜 32) \| 2.93 \| 43 / 98 / 78 \| 0.754 ± 0.0 | 2.93 | 62.5 ms | latency.가 u1000.total | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 62.5 ms (멜 32) \| 2.93 \| 43 / 98 / 78 \| 0.754 ± 0.0 | 43 | 62.5 ms | avg_power.configs.가 u1000.E_min_uJ; breakeven.CPU log-mel, 간격 62.5 ms (멜 32) < 원본 파형 (전처리 없음).by_source.datasheet.E_X; breakeven.NPU 전처리 (Gabor 필터뱅크) < CPU log- | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 62.5 ms (멜 32) \| 2.93 \| 43 / 98 / 78 \| 0.754 ± 0.0 | 98 | 62.5 ms | avg_power.configs.가 u1000.E_max_uJ; breakeven.CPU log-mel, 간격 50 ms < 원본 파형 (전처리 없음).by_source.moosmann.E_X; breakeven.CPU log-mel, 간격 62.5 ms (멜 32) < 원본 파형 (전 | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 62.5 ms (멜 32) \| 2.93 \| 43 / 98 / 78 \| 0.754 ± 0.0 | 78 | 62.5 ms | breakeven.CPU log-mel, 간격 50 ms < 원본 파형 (전처리 없음).by_source.moosmann.E_Y; breakeven.CPU log-mel, 간격 62.5 ms (멜 32) < 원본 파형 (전처리 없음).by_source.moosmann.E_X; break | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 62.5 ms (멜 32) \| 2.93 \| 43 / 98 / 78 \| 0.754 ± 0.0 | 0.754 | 62.5 ms | acc.가 u1000.f1_S3.mean | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 62.5 ms (멜 32) \| 2.93 \| 43 / 98 / 78 \| 0.754 ± 0.0 | 0.014 | 62.5 ms | acc.가 u1000.ext_f1.sd; acc.가 u1000.f1_S3.sd; editions.ext_f1_real31L.간격 400.v4.sd; hop_fraction.②′ b64 | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 62.5 ms (멜 32) \| 2.93 \| 43 / 98 / 78 \| 0.754 ± 0.0 | 0.542 | 62.5 ms | acc.가 u1000.ext_f1.mean | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 62.5 ms (멜 32) \| 2.93 \| 43 / 98 / 78 \| 0.754 ± 0.0 | 0.014 | 62.5 ms | acc.가 u1000.ext_f1.sd; acc.가 u1000.f1_S3.sd; editions.ext_f1_real31L.간격 400.v4.sd; hop_fraction.②′ b64 | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 50 ms \| 3.66 \| 54 / 121 / 98 \| 0.758 ± 0.019 \| 0.5 | 50 | 50 ms | 프레임 간격 ms / CNN 50 MHz | 상수 |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 50 ms \| 3.66 \| 54 / 121 / 98 \| 0.758 ± 0.019 \| 0.5 | 3.66 | 50 ms | latency.②′ b64.log32.total_main_before_ms; latency.다 u800.total | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 50 ms \| 3.66 \| 54 / 121 / 98 \| 0.758 ± 0.019 \| 0.5 | 54 | 50 ms | acc.가 u1000.ext_f1.mean; avg_power.configs.①′.share_max_pct; avg_power.configs.다 u800.E_min_uJ; breakeven.CPU log-mel, 간격 50 ms < 원본 파형 (전처리 없음).by_source.datas | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 50 ms \| 3.66 \| 54 / 121 / 98 \| 0.758 ± 0.019 \| 0.5 | 121 | 50 ms | avg_power.configs.다 u800.E_max_uJ; breakeven.CPU log-mel, 간격 50 ms < 원본 파형 (전처리 없음).by_source.literature.E_X; energy.다 u800.literature | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 50 ms \| 3.66 \| 54 / 121 / 98 \| 0.758 ± 0.019 \| 0.5 | 98 | 50 ms | avg_power.configs.가 u1000.E_max_uJ; breakeven.CPU log-mel, 간격 50 ms < 원본 파형 (전처리 없음).by_source.moosmann.E_X; breakeven.CPU log-mel, 간격 62.5 ms (멜 32) < 원본 파형 (전 | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 50 ms \| 3.66 \| 54 / 121 / 98 \| 0.758 ± 0.019 \| 0.5 | 0.758 | 50 ms | acc.다 u800.f1_S3.mean | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 50 ms \| 3.66 \| 54 / 121 / 98 \| 0.758 ± 0.019 \| 0.5 | 0.019 | 50 ms | acc.다 u800.f1_L.sd; acc.다 u800.f1_S3.sd; editions.ext_f1_real31L.④.v4.sd | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 50 ms \| 3.66 \| 54 / 121 / 98 \| 0.758 ± 0.019 \| 0.5 | 0.566 | 50 ms | acc.다 u800.ext_f1.mean | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 50 ms \| 3.66 \| 54 / 121 / 98 \| 0.758 ± 0.019 \| 0.5 | 0.025 | 50 ms | acc.②′.f1_S3.sd; acc.④.f1_S3.sd; acc.다 u800.ext_f1.sd; latency.②′ f31.detail.cfg1; latency.②′ f31.log32.session_new.cfg1; variants.②′.measured.②′ f31.detail_ms. | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 25 ms \| 9.42 \| 167 / 353 / 261 \| 0.820 ± 0.006 \| 0 | 25 | 25 ms | listening.A_ext_scream/v1+실사용.n | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 25 ms \| 9.42 \| 167 / 353 / 261 \| 0.820 ± 0.006 \| 0 | 9.42 | 25 ms | latency.간격 400.total | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 25 ms \| 9.42 \| 167 / 353 / 261 \| 0.820 ± 0.006 \| 0 | 167 | 25 ms | avg_power.configs.간격 400.E_min_uJ; breakeven.CPU log-mel, 간격 25 ms < CPU log-mel, 간격 16 ms.by_source.datasheet.E_X; energy.간격 400.datasheet | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 25 ms \| 9.42 \| 167 / 353 / 261 \| 0.820 ± 0.006 \| 0 | 353 | 25 ms | avg_power.configs.간격 400.E_max_uJ; breakeven.CPU log-mel, 간격 25 ms < CPU log-mel, 간격 16 ms.by_source.literature.E_X; energy.간격 400.literature | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 25 ms \| 9.42 \| 167 / 353 / 261 \| 0.820 ± 0.006 \| 0 | 261 | 25 ms | breakeven.CPU log-mel, 간격 25 ms < CPU log-mel, 간격 16 ms.by_source.moosmann.E_X; energy.간격 400.moosmann | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 25 ms \| 9.42 \| 167 / 353 / 261 \| 0.820 ± 0.006 \| 0 | 0.820 | 25 ms | acc.간격 400.f1_S3.mean | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 25 ms \| 9.42 \| 167 / 353 / 261 \| 0.820 ± 0.006 \| 0 | 0.006 | 25 ms | acc.②′ f31.ext_f1.sd; acc.간격 400.f1_L.sd; acc.간격 400.f1_S3.sd; latency.②′.detail.unload; latency.②′.log32.session_old.unload; latency.②′.out; latency.간격 400.out | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 25 ms \| 9.42 \| 167 / 353 / 261 \| 0.820 ± 0.006 \| 0 | 0.651 | 25 ms | acc.간격 400.ext_f1.mean; editions.ext_f1_real31L.간격 400.v4+D.mean | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 25 ms \| 9.42 \| 167 / 353 / 261 \| 0.820 ± 0.006 \| 0 | 0.010 | 25 ms | acc.간격 400.ext_f1.sd; editions.ext_f1_real31L.간격 400.v4+D.sd; hop_fraction.④ | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 16 ms \| 15.06 \| 267 / 565 / 418 \| 0.836 ± 0.002 \|  | 16 | 16 ms | 16 kHz / 16 프레임 / 16 ms | 상수 |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 16 ms \| 15.06 \| 267 / 565 / 418 \| 0.836 ± 0.002 \|  | 15.06 | 16 ms | latency.①′.total | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 16 ms \| 15.06 \| 267 / 565 / 418 \| 0.836 ± 0.002 \|  | 267 | 16 ms | avg_power.configs.①′.E_min_uJ; breakeven.CPU log-mel, 간격 25 ms < CPU log-mel, 간격 16 ms.by_source.datasheet.E_Y; energy.①′.datasheet | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 16 ms \| 15.06 \| 267 / 565 / 418 \| 0.836 ± 0.002 \|  | 565 | 16 ms | avg_power.configs.①′.E_max_uJ; breakeven.CPU log-mel, 간격 25 ms < CPU log-mel, 간격 16 ms.by_source.literature.E_Y; energy.①′.literature | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 16 ms \| 15.06 \| 267 / 565 / 418 \| 0.836 ± 0.002 \|  | 418 | 16 ms | breakeven.CPU log-mel, 간격 25 ms < CPU log-mel, 간격 16 ms.by_source.moosmann.E_Y; energy.①′.moosmann | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 16 ms \| 15.06 \| 267 / 565 / 418 \| 0.836 ± 0.002 \|  | 0.836 | 16 ms | acc.①′.f1_S3.mean | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 16 ms \| 15.06 \| 267 / 565 / 418 \| 0.836 ± 0.002 \|  | 0.002 | 16 ms | acc.①′.f1_S3.sd; variants.②′.measured.②′ f31.delta_vs_fe_ms.cfg1 | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 16 ms \| 15.06 \| 267 / 565 / 418 \| 0.836 ± 0.002 \|  | 0.665 | 16 ms | acc.①′.ext_f1.mean; latency.②′ b64.load | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU log-mel, 간격 16 ms \| 15.06 \| 267 / 565 / 418 \| 0.836 ± 0.002 \|  | 0.015 | 16 ms | acc.①′.ext_f1.sd; hop_fraction.다 u800 | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU 추론 (원본 파형 가중치) \| 225.77 \| 2008 / 5599 / 5599 \| = 원본 파형 \| = 원본  | 225.77 | CPU 추론, 원본 파형 | breakeven.원본 파형 (전처리 없음) (NPU 추론) < CPU 추론 (Cortex-M4F).C_Y; latency.M4 CMSIS-NN.total | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU 추론 (원본 파형 가중치) \| 225.77 \| 2008 / 5599 / 5599 \| = 원본 파형 \| = 원본  | 2008 | CPU 추론, 원본 파형 | avg_power.configs.M4 CMSIS-NN.E_min_uJ; breakeven.원본 파형 (전처리 없음) (NPU 추론) < CPU 추론 (Cortex-M4F).by_source.datasheet.E_Y; breakeven.원본 파형 (전처리 없음) (NPU 추론) < CPU | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU 추론 (원본 파형 가중치) \| 225.77 \| 2008 / 5599 / 5599 \| = 원본 파형 \| = 원본  | 5599 | CPU 추론, 원본 파형 | avg_power.configs.M4 CMSIS-NN.E_max_uJ; breakeven.원본 파형 (전처리 없음) (NPU 추론) < CPU 추론 (Cortex-M4F).by_source.literature.E_Y; breakeven.원본 파형 (전처리 없음) (NPU 추론) < CP | ✓ |
| 2. 시스템과 실험 설계 | [표] CPU 추론 (원본 파형 가중치) \| 225.77 \| 2008 / 5599 / 5599 \| = 원본 파형 \| = 원본  | 5599 | CPU 추론, 원본 파형 | avg_power.configs.M4 CMSIS-NN.E_max_uJ; breakeven.원본 파형 (전처리 없음) (NPU 추론) < CPU 추론 (Cortex-M4F).by_source.literature.E_Y; breakeven.원본 파형 (전처리 없음) (NPU 추론) < CP | ✓ |
| 2. 시스템과 실험 설계 | 판단 1회 추정 에너지(로그축)와 정확도: (a) 시험 데이터 macro-F1, (b) 외부 데이터 F1. | 1 | — | acc.(a).ext_f1.sd; acc.(a).ext_map.mean; acc.(a).ext_map.sd; acc.(a).f1_L.mean; acc.(a).f1_L.sd; acc.(a).f1_S3.mean; acc.(a).f1_S3.sd; acc.①′.ext_f1.mean; acc.① | ✓ |
| 2. 시스템과 실험 설계 | 색은 전처리 위치, 마커는 데이터시트 전력 기준, 가로 막대는 세 전력 출처의 범위; 주황 점선은 NPU 전처리 계열(Gabo | 3 | 변형 3종, 변형 3종, 변형 3종, NPU 전처리 (Gabor 필터뱅크) | acc.(a).ext_f1.n; acc.(a).ext_map.n; acc.(a).f1_L.n; acc.(a).f1_S3.n; acc.(a).scream_S3.n; acc.(a).scream_S3.sd; acc.①′.ext_f1.n; acc.①′.ext_map.n; acc.①′.f1_L. | ✓ |
| 3. 결  과 | 판단 1회 지연 분해(보드 실측, 1,000회 최악값). | 1 | — | acc.(a).ext_f1.sd; acc.(a).ext_map.mean; acc.(a).ext_map.sd; acc.(a).f1_L.mean; acc.(a).f1_L.sd; acc.(a).f1_S3.mean; acc.(a).f1_S3.sd; acc.①′.ext_f1.mean; acc.① | ✓ |
| 3. 결  과 | 판단 1회 지연 분해(보드 실측, 1,000회 최악값). | 1,000 | — | firmware.cpu_split.n_iter; latency.②′ b64.log32.n_iter; latency.②′ b64f31.log32.n_iter; latency.②′ f31.log32.n_iter; latency.②′.log32.n_iter; latency.②′.new_fir | ✓ |
| 3. 결  과 | 원본 파형 구성은 CPU log-mel 16 ms 구성보다 시험 macro-F1이 0.166, 외부 F1이 0.222 낮다(표 | 16 | 16 ms, 원본 파형 | 16 kHz / 16 프레임 / 16 ms | 상수 |
| 3. 결  과 | 원본 파형 구성은 CPU log-mel 16 ms 구성보다 시험 macro-F1이 0.166, 외부 F1이 0.222 낮다(표 | 0.166 | 16 ms, 원본 파형 | 유도: f1(①′)−f1(④) 시험 | ✓ |
| 3. 결  과 | 원본 파형 구성은 CPU log-mel 16 ms 구성보다 시험 macro-F1이 0.166, 외부 F1이 0.222 낮다(표 | 0.222 | 16 ms, 원본 파형 | 유도: f1(①′)−f1(④) 외부 | ✓ |
| 3. 결  과 | CPU 추론 대조군(원본 파형 가중치) 대비 NPU 추론은 추론만 157배, 판단 1회 89배 빠르고 CPU 추론은 판단 주기 | 157 | CPU 추론, 원본 파형 | ratio.infer_m4cmsis_over_npu | ✓ |
| 3. 결  과 | CPU 추론 대조군(원본 파형 가중치) 대비 NPU 추론은 추론만 157배, 판단 1회 89배 빠르고 CPU 추론은 판단 주기 | 1 | CPU 추론, 원본 파형 | acc.(a).ext_f1.sd; acc.(a).ext_map.mean; acc.(a).ext_map.sd; acc.(a).f1_L.mean; acc.(a).f1_L.sd; acc.(a).f1_S3.mean; acc.(a).f1_S3.sd; acc.①′.ext_f1.mean; acc.① | ✓ |
| 3. 결  과 | CPU 추론 대조군(원본 파형 가중치) 대비 NPU 추론은 추론만 157배, 판단 1회 89배 빠르고 CPU 추론은 판단 주기 | 89 | CPU 추론, 원본 파형 | panns.ext_real31L_ext300.map; panns.ext_real31L_test300.map; ratio.decision_m4cmsis_over_npu | ✓ |
| 3. 결  과 | CPU 추론 대조군(원본 파형 가중치) 대비 NPU 추론은 추론만 157배, 판단 1회 89배 빠르고 CPU 추론은 판단 주기 | 90 | CPU 추론, 원본 파형 | acc.①′.ext_map.mean; acc.간격 400.ext_map.mean; hop_fraction.M4 CMSIS-NN; listening.S_train_scream/yellshout_train.n; 유도: 100·hop_fraction(M4) | ✓ |
| 3. 결  과 | NPU 근사 log는 원본 파형 구성 대비 +0.034 [-0.022, +0.090]로 CI가 0을 포함하고, 로그 근사 층  | +0.034 | NPU 근사 log, 원본 파형 | acc.②′.ext_f1.sd; welch.(a) − ④.test_f1_S3[0] | ✓ |
| 3. 결  과 | NPU 근사 log는 원본 파형 구성 대비 +0.034 [-0.022, +0.090]로 CI가 0을 포함하고, 로그 근사 층  | -0.022 | NPU 근사 log, 원본 파형 | welch.(a) − ④.test_f1_S3[1] | ✓ |
| 3. 결  과 | NPU 근사 log는 원본 파형 구성 대비 +0.034 [-0.022, +0.090]로 CI가 0을 포함하고, 로그 근사 층  | +0.090 | NPU 근사 log, 원본 파형 | latency.②′ b64f31.log32.session_new.cfg2; latency.②′ b64f31.log32.session_old.cfg2; welch.(a) − ④.test_f1_S3[2] | ✓ |
| 3. 결  과 | NPU 근사 log는 원본 파형 구성 대비 +0.034 [-0.022, +0.090]로 CI가 0을 포함하고, 로그 근사 층  | 0 | NPU 근사 log, 원본 파형 | acc.(a).ext_f1.mean; acc.(a).ext_f1.sd; acc.(a).ext_map.sd; acc.(a).f1_L.sd; acc.(a).f1_S3.sd; acc.(a).scream_S3.mean; acc.(a).scream_S3.sd; acc.①′.ext_f1.sd; a | ✓ |
| 3. 결  과 | NPU 근사 log는 원본 파형 구성 대비 +0.034 [-0.022, +0.090]로 CI가 0을 포함하고, 로그 근사 층  | 3.8 | NPU 근사 log, 원본 파형 | hop_fraction.간격 400; welch.(a) − ④.ext_map[0]; welch.가 u1000 − ④.test_f1_L[1]; welch.다 u800 − ④.test_f1_S3[1] | ✓ |
| 3. 결  과 | NPU 근사 log는 원본 파형 구성 대비 +0.034 [-0.022, +0.090]로 CI가 0을 포함하고, 로그 근사 층  | 3.3 | NPU 근사 log, 원본 파형 | latency.②′ b64f31.log32.cpu_main_before_ms; latency.②′ b64f31.log32.session_old.cpu; latency.②′ f31.detail.total; latency.②′ f31.log32.session_new.total; latenc | ✓ |
| 3. 결  과 | NPU 전처리 (Gabor 필터뱅크)는 사전에 정한 목표(원본 파형 구성 초과 또는 CPU log-mel 62.5 ms 구성과 | 62.5 | 62.5 ms, NPU 전처리 (Gabor 필터뱅크), 원본 파형 | 프레임 간격 ms | 상수 |
| 3. 결  과 | NPU 전처리 (Gabor 필터뱅크)는 사전에 정한 목표(원본 파형 구성 초과 또는 CPU log-mel 62.5 ms 구성과 | -0.026 | 62.5 ms, NPU 전처리 (Gabor 필터뱅크), 원본 파형 | welch.②′ − ④.test_f1_S3[0] | ✓ |
| 3. 결  과 | NPU 전처리 (Gabor 필터뱅크)는 사전에 정한 목표(원본 파형 구성 초과 또는 CPU log-mel 62.5 ms 구성과 | -0.082 | 62.5 ms, NPU 전처리 (Gabor 필터뱅크), 원본 파형 | welch.②′ − ④.test_f1_S3[1]; welch.②′ − 가 u1000.ext_f1[0] | ✓ |
| 3. 결  과 | NPU 전처리 (Gabor 필터뱅크)는 사전에 정한 목표(원본 파형 구성 초과 또는 CPU log-mel 62.5 ms 구성과 | +0.030 | 62.5 ms, NPU 전처리 (Gabor 필터뱅크), 원본 파형 | variants.②′.measured.②′ f31.delta_vs_fe_ms.load1; welch.(a) − ④.test_f1_L[0]; welch.②′ − ④.test_f1_S3[2] | ✓ |
| 3. 결  과 | NPU 전처리 (Gabor 필터뱅크)는 사전에 정한 목표(원본 파형 구성 초과 또는 CPU log-mel 62.5 ms 구성과 | 62.5 | 62.5 ms, NPU 전처리 (Gabor 필터뱅크), 원본 파형 | 프레임 간격 ms | 상수 |
| 3. 결  과 | NPU 전처리 (Gabor 필터뱅크)는 사전에 정한 목표(원본 파형 구성 초과 또는 CPU log-mel 62.5 ms 구성과 | -0.111 | 62.5 ms, NPU 전처리 (Gabor 필터뱅크), 원본 파형 | welch.②′ − 가 u1000.test_f1_S3[0] | ✓ |
| 3. 결  과 | NPU 전처리 (Gabor 필터뱅크)는 사전에 정한 목표(원본 파형 구성 초과 또는 CPU log-mel 62.5 ms 구성과 | -0.162 | 62.5 ms, NPU 전처리 (Gabor 필터뱅크), 원본 파형 | welch.②′ − 가 u1000.test_f1_S3[1] | ✓ |
| 3. 결  과 | NPU 전처리 (Gabor 필터뱅크)는 사전에 정한 목표(원본 파형 구성 초과 또는 CPU log-mel 62.5 ms 구성과 | -0.060 | 62.5 ms, NPU 전처리 (Gabor 필터뱅크), 원본 파형 | welch.②′ − 가 u1000.test_f1_S3[2] | ✓ |
| 3. 결  과 | 추정 에너지는 원본 파형 구성보다 세 전력 출처 모두에서 적지만(데이터시트 39 대 71 μJ, Moosmann 52 대 78 | 39 | 62.5 ms, 원본 파형 | acc.(a).scream_S3.mean; acc.②′ b64.scream_S3.mean; avg_power.configs.②′.E_min_uJ; breakeven.NPU 전처리 (Gabor 필터뱅크) < CPU log-mel, 간격 62.5 ms (멜 32).by_source.data | ✓ |
| 3. 결  과 | 추정 에너지는 원본 파형 구성보다 세 전력 출처 모두에서 적지만(데이터시트 39 대 71 μJ, Moosmann 52 대 78 | 71 | 62.5 ms, 원본 파형 | avg_power.configs.②′ f31.E_min_uJ; avg_power.configs.④.E_min_uJ; breakeven.CPU log-mel, 간격 50 ms < 원본 파형 (전처리 없음).by_source.datasheet.E_Y; breakeven.CPU log-mel | ✓ |
| 3. 결  과 | 추정 에너지는 원본 파형 구성보다 세 전력 출처 모두에서 적지만(데이터시트 39 대 71 μJ, Moosmann 52 대 78 | 52 | 62.5 ms, 원본 파형 | breakeven.NPU 전처리 (Gabor 필터뱅크) < CPU log-mel, 간격 62.5 ms (멜 32).by_source.moosmann.E_X; breakeven.NPU 전처리 (Gabor 필터뱅크) < 원본 파형 (전처리 없음).by_source.moosmann.E_X;  | ✓ |
| 3. 결  과 | 추정 에너지는 원본 파형 구성보다 세 전력 출처 모두에서 적지만(데이터시트 39 대 71 μJ, Moosmann 52 대 78 | 78 | 62.5 ms, 원본 파형 | breakeven.CPU log-mel, 간격 50 ms < 원본 파형 (전처리 없음).by_source.moosmann.E_Y; breakeven.CPU log-mel, 간격 62.5 ms (멜 32) < 원본 파형 (전처리 없음).by_source.moosmann.E_X; break | ✓ |
| 3. 결  과 | 추정 에너지는 원본 파형 구성보다 세 전력 출처 모두에서 적지만(데이터시트 39 대 71 μJ, Moosmann 52 대 78 | 1 | 62.5 ms, 원본 파형 | acc.(a).ext_f1.sd; acc.(a).ext_map.mean; acc.(a).ext_map.sd; acc.(a).f1_L.mean; acc.(a).f1_L.sd; acc.(a).f1_S3.mean; acc.(a).f1_S3.sd; acc.①′.ext_f1.mean; acc.① | ✓ |
| 3. 결  과 | 추정 에너지는 원본 파형 구성보다 세 전력 출처 모두에서 적지만(데이터시트 39 대 71 μJ, Moosmann 52 대 78 | 1.85 | 62.5 ms, 원본 파형 | latency.②′.detail.total; latency.②′.total | ⚠ |
| 3. 결  과 | 추정 에너지는 원본 파형 구성보다 세 전력 출처 모두에서 적지만(데이터시트 39 대 71 μJ, Moosmann 52 대 78 | 2.53 | 62.5 ms, 원본 파형 | latency.④.total | ✓ |
| 3. 결  과 | 추정 에너지는 원본 파형 구성보다 세 전력 출처 모두에서 적지만(데이터시트 39 대 71 μJ, Moosmann 52 대 78 | 62.5 | 62.5 ms, 원본 파형 | 프레임 간격 ms | 상수 |
| 3. 결  과 | 추정 에너지는 원본 파형 구성보다 세 전력 출처 모두에서 적지만(데이터시트 39 대 71 μJ, Moosmann 52 대 78 | 39 | 62.5 ms, 원본 파형 | acc.(a).scream_S3.mean; acc.②′ b64.scream_S3.mean; avg_power.configs.②′.E_min_uJ; breakeven.NPU 전처리 (Gabor 필터뱅크) < CPU log-mel, 간격 62.5 ms (멜 32).by_source.data | ✓ |
| 3. 결  과 | 추정 에너지는 원본 파형 구성보다 세 전력 출처 모두에서 적지만(데이터시트 39 대 71 μJ, Moosmann 52 대 78 | 43 | 62.5 ms, 원본 파형 | avg_power.configs.가 u1000.E_min_uJ; breakeven.CPU log-mel, 간격 62.5 ms (멜 32) < 원본 파형 (전처리 없음).by_source.datasheet.E_X; breakeven.NPU 전처리 (Gabor 필터뱅크) < CPU log- | ✓ |
| 3. 결  과 | 추정 에너지는 원본 파형 구성보다 세 전력 출처 모두에서 적지만(데이터시트 39 대 71 μJ, Moosmann 52 대 78 | 0.11 | 62.5 ms, 원본 파형 | 유도: \|welch ②′−가\| | ✓ |
| 3. 결  과 | 보드에서 가속기 출력·CPU 특징·최종 로짓이 호스트와 비트 일치했고, CPU 몫 0.70 ms의 80%는 가속기 메모리에서  | 0.70 | — | acc.(a).f1_S3.mean; firmware.cpu_split.main_cpu_ms; latency.②′.detail.cpu; latency.②′.log32.session_new.cpu; latency.②′.new_firmware.detail.cpu; latency.②′.new_ | ✓ |
| 3. 결  과 | 보드에서 가속기 출력·CPU 특징·최종 로짓이 호스트와 비트 일치했고, CPU 몫 0.70 ms의 80%는 가속기 메모리에서  | 80 | — | acc.간격 400.f1_L.mean; firmware.cpu_split.read_pct; firmware.cpu_split.read_share_med; firmware.cpu_split.read_share_worst; firmware.same_fw_fe_vs_ga.moosmann.ga | ✓ |
| 3. 결  과 | 보드에서 가속기 출력·CPU 특징·최종 로짓이 호스트와 비트 일치했고, CPU 몫 0.70 ms의 80%는 가속기 메모리에서  | 2,048 | — | 가속기 출력 개수 | 상수 |
| 3. 결  과 | 보드에서 가속기 출력·CPU 특징·최종 로짓이 호스트와 비트 일치했고, CPU 몫 0.70 ms의 80%는 가속기 메모리에서  | 8 | — | acc.다 u800.scream_S3.sd; avg_power.configs.(a).share_datasheet_at_hi_pct; latency.②′.detail.infer1; listening.A_ext_scream/실사용만.?; welch.(a) − ④.test_f1_L[2]; w | ✓ |
| 3. 결  과 | 보드에서 가속기 출력·CPU 특징·최종 로짓이 호스트와 비트 일치했고, CPU 몫 0.70 ms의 80%는 가속기 메모리에서  | 20 | — | avg_power.configs.다 u800.share_max_pct; firmware.cpu_split.log_pct; latency.②′ b64.detail.infer1; latency.②′ b64.log32.session_new.infer1; latency.②′ b64.log32. | ✓ |
| 3. 결  과 | 대역·프레임 수를 늘린 NPU 전처리 변형 3종은 정확도가 올랐으나 에너지가 더 빠르게 늘어 세 전력 출처 모두에서 CPU 전 | 3 | 변형 3종, 변형 3종, 변형 3종 | acc.(a).ext_f1.n; acc.(a).ext_map.n; acc.(a).f1_L.n; acc.(a).f1_S3.n; acc.(a).scream_S3.n; acc.(a).scream_S3.sd; acc.①′.ext_f1.n; acc.①′.ext_map.n; acc.①′.f1_L. | ✓ |
| 3. 결  과 | 시험 macro-F1 0.655~0.690 (표 1), 추정 에너지 39에서 71~133 μJ. | 0.655 | — | acc.②′ f31.f1_S3.mean; variants.②′.f1_S3.mean.②′ f31; variants.②′.f1_S3.min | ⚠ |
| 3. 결  과 | 시험 macro-F1 0.655~0.690 (표 1), 추정 에너지 39에서 71~133 μJ. | 0.690 | — | acc.②′ b64f31.f1_S3.mean; variants.②′.f1_S3.max; variants.②′.f1_S3.mean.②′ b64f31 | ⚠ |
| 3. 결  과 | 시험 macro-F1 0.655~0.690 (표 1), 추정 에너지 39에서 71~133 μJ. | 39 | — | acc.(a).scream_S3.mean; acc.②′ b64.scream_S3.mean; avg_power.configs.②′.E_min_uJ; breakeven.NPU 전처리 (Gabor 필터뱅크) < CPU log-mel, 간격 62.5 ms (멜 32).by_source.data | ✓ |
| 3. 결  과 | 시험 macro-F1 0.655~0.690 (표 1), 추정 에너지 39에서 71~133 μJ. | 71 | — | avg_power.configs.②′ f31.E_min_uJ; avg_power.configs.④.E_min_uJ; breakeven.CPU log-mel, 간격 50 ms < 원본 파형 (전처리 없음).by_source.datasheet.E_Y; breakeven.CPU log-mel | ✓ |
| 3. 결  과 | 시험 macro-F1 0.655~0.690 (표 1), 추정 에너지 39에서 71~133 μJ. | 133 | — | avg_power.configs.②′ b64f31.E_min_uJ; energy.②′ b64f31.datasheet; variants.②′.measured.②′ b64f31.E_uJ.datasheet | ⚠ |
| 3. 결  과 | 늘어난 비용은 CPU의 가속기 메모리 읽기(46~51%)와 NPU 연산(39~45%)이 비슷하게 나눴다. | 46 | — | acc.②′.ext_f1.mean; variants.②′.measured.②′ b64.cpu_share_of_delta | ⚠ |
| 3. 결  과 | 늘어난 비용은 CPU의 가속기 메모리 읽기(46~51%)와 NPU 연산(39~45%)이 비슷하게 나눴다. | 51 | — | acc.②′ b64f31.ext_f1.mean; variants.②′.measured.②′ b64f31.cpu_share_of_delta | ⚠ |
| 3. 결  과 | 늘어난 비용은 CPU의 가속기 메모리 읽기(46~51%)와 NPU 연산(39~45%)이 비슷하게 나눴다. | 39 | — | acc.(a).scream_S3.mean; acc.②′ b64.scream_S3.mean; avg_power.configs.②′.E_min_uJ; breakeven.NPU 전처리 (Gabor 필터뱅크) < CPU log-mel, 간격 62.5 ms (멜 32).by_source.data | ✓ |
| 3. 결  과 | 늘어난 비용은 CPU의 가속기 메모리 읽기(46~51%)와 NPU 연산(39~45%)이 비슷하게 나눴다. | 45 | — | firmware.same_fw_fe_vs_ga.datasheet.ga; latency.가 u1000.new_firmware.E.datasheet; variants.②′.measured.②′ b64.npu_share_of_delta | ✓ |
| 3. 결  과 | CPU log-mel 62.5 ms 구성은 원본 파형 구성보다 시험 macro-F1 +0.085 [+0.035, +0.135] | 62.5 | 62.5 ms, 원본 파형 | 프레임 간격 ms | 상수 |
| 3. 결  과 | CPU log-mel 62.5 ms 구성은 원본 파형 구성보다 시험 macro-F1 +0.085 [+0.035, +0.135] | +0.085 | 62.5 ms, 원본 파형 | welch.가 u1000 − ④.test_f1_S3[0] | ✓ |
| 3. 결  과 | CPU log-mel 62.5 ms 구성은 원본 파형 구성보다 시험 macro-F1 +0.085 [+0.035, +0.135] | +0.035 | 62.5 ms, 원본 파형 | acc.가 u1000.scream_S3.sd; welch.가 u1000 − ④.test_f1_S3[1] | ✓ |
| 3. 결  과 | CPU log-mel 62.5 ms 구성은 원본 파형 구성보다 시험 macro-F1 +0.085 [+0.035, +0.135] | +0.135 | 62.5 ms, 원본 파형 | latency.가 u1000.new_firmware.load; welch.가 u1000 − ④.test_f1_S3[2] | ✓ |
| 3. 결  과 | CPU log-mel 62.5 ms 구성은 원본 파형 구성보다 시험 macro-F1 +0.085 [+0.035, +0.135] | +0.099 | 62.5 ms, 원본 파형 | welch.가 u1000 − ④.ext_f1[0] | ✓ |
| 3. 결  과 | CPU log-mel 62.5 ms 구성은 원본 파형 구성보다 시험 macro-F1 +0.085 [+0.035, +0.135] | +0.056 | 62.5 ms, 원본 파형 | welch.가 u1000 − ④.ext_f1[1] | ✓ |
| 3. 결  과 | CPU log-mel 62.5 ms 구성은 원본 파형 구성보다 시험 macro-F1 +0.085 [+0.035, +0.135] | +0.142 | 62.5 ms, 원본 파형 | firmware.cpu_split.log_ms; firmware.cpu_split.log_ms_med; welch.가 u1000 − ④.ext_f1[2] | ✓ |
| 3. 결  과 | CPU log-mel 62.5 ms 구성은 원본 파형 구성보다 시험 macro-F1 +0.085 [+0.035, +0.135] | 78 | 62.5 ms, 원본 파형 | breakeven.CPU log-mel, 간격 50 ms < 원본 파형 (전처리 없음).by_source.moosmann.E_Y; breakeven.CPU log-mel, 간격 62.5 ms (멜 32) < 원본 파형 (전처리 없음).by_source.moosmann.E_X; break | ✓ |
| 3. 결  과 | CPU log-mel 62.5 ms 구성은 원본 파형 구성보다 시험 macro-F1 +0.085 [+0.035, +0.135] | 78 | 62.5 ms, 원본 파형 | breakeven.CPU log-mel, 간격 50 ms < 원본 파형 (전처리 없음).by_source.moosmann.E_Y; breakeven.CPU log-mel, 간격 62.5 ms (멜 32) < 원본 파형 (전처리 없음).by_source.moosmann.E_X; break | ✓ |
| 3. 결  과 | log-mel이 16,384 샘플을 512 값으로 줄여 NPU 입력 적재(1.09 → 0.13 ms)와 추론(1.44 → 0. | 16,384 | — | 창 샘플 | 상수 |
| 3. 결  과 | log-mel이 16,384 샘플을 512 값으로 줄여 NPU 입력 적재(1.09 → 0.13 ms)와 추론(1.44 → 0. | 512 | — | 512탭 / 512 값 | 상수 |
| 3. 결  과 | log-mel이 16,384 샘플을 512 값으로 줄여 NPU 입력 적재(1.09 → 0.13 ms)와 추론(1.44 → 0. | 1.09 | — | latency.(a).load; latency.④.load | ⚠ |
| 3. 결  과 | log-mel이 16,384 샘플을 512 값으로 줄여 NPU 입력 적재(1.09 → 0.13 ms)와 추론(1.44 → 0. | 0.13 | — | latency.②′.detail.load2; latency.②′.log32.session_new.infer1; latency.②′.log32.session_new.load2; latency.②′.log32.session_old.infer1; latency.②′.log32.session_ | ⚠ |
| 3. 결  과 | log-mel이 16,384 샘플을 512 값으로 줄여 NPU 입력 적재(1.09 → 0.13 ms)와 추론(1.44 → 0. | 1.44 | — | breakeven.CPU log-mel, 간격 25 ms < CPU log-mel, 간격 16 ms.by_source.moosmann.r; breakeven.CPU log-mel, 간격 50 ms < 원본 파형 (전처리 없음).N_Y; breakeven.CPU log-mel, 간격 50 | ✓ |
| 3. 결  과 | log-mel이 16,384 샘플을 512 값으로 줄여 NPU 입력 적재(1.09 → 0.13 ms)와 추론(1.44 → 0. | 0.53 | — | breakeven.CPU log-mel, 간격 62.5 ms (멜 32) < 원본 파형 (전처리 없음).N_X; breakeven.NPU 전처리 (Gabor 필터뱅크) < CPU log-mel, 간격 62.5 ms (멜 32).N_Y; latency.②′.detail.infer2; la | ✓ |
| 3. 결  과 | 대신 CPU 전처리 2.26 ms가 늘어 판단 1회 지연은 2.93 대 2.53 ms로 비슷하지만, 줄어든 적재·추론 구간의  | 2.26 | — | latency.가 u1000.new_firmware.pre; latency.가 u1000.pre | ⚠ |
| 3. 결  과 | 대신 CPU 전처리 2.26 ms가 늘어 판단 1회 지연은 2.93 대 2.53 ms로 비슷하지만, 줄어든 적재·추론 구간의  | 1 | — | acc.(a).ext_f1.sd; acc.(a).ext_map.mean; acc.(a).ext_map.sd; acc.(a).f1_L.mean; acc.(a).f1_L.sd; acc.(a).f1_S3.mean; acc.(a).f1_S3.sd; acc.①′.ext_f1.mean; acc.① | ✓ |
| 3. 결  과 | 대신 CPU 전처리 2.26 ms가 늘어 판단 1회 지연은 2.93 대 2.53 ms로 비슷하지만, 줄어든 적재·추론 구간의  | 2.93 | — | latency.가 u1000.total | ⚠ |
| 3. 결  과 | 대신 CPU 전처리 2.26 ms가 늘어 판단 1회 지연은 2.93 대 2.53 ms로 비슷하지만, 줄어든 적재·추론 구간의  | 2.53 | — | latency.④.total | ⚠ |
| 3. 결  과 | 대신 CPU 전처리 2.26 ms가 늘어 판단 1회 지연은 2.93 대 2.53 ms로 비슷하지만, 줄어든 적재·추론 구간의  | 71 | — | avg_power.configs.②′ f31.E_min_uJ; avg_power.configs.④.E_min_uJ; breakeven.CPU log-mel, 간격 50 ms < 원본 파형 (전처리 없음).by_source.datasheet.E_Y; breakeven.CPU log-mel | ✓ |
| 3. 결  과 | 대신 CPU 전처리 2.26 ms가 늘어 판단 1회 지연은 2.93 대 2.53 ms로 비슷하지만, 줄어든 적재·추론 구간의  | 23 | — | avg_power.configs.②′ b64.share_max_pct | ⚠ |
| 3. 결  과 | 대신 CPU 전처리 2.26 ms가 늘어 판단 1회 지연은 2.93 대 2.53 ms로 비슷하지만, 줄어든 적재·추론 구간의  | 20 | — | avg_power.configs.다 u800.share_max_pct; firmware.cpu_split.log_pct; latency.②′ b64.detail.infer1; latency.②′ b64.log32.session_new.infer1; latency.②′ b64.log32. | ✓ |
| 3. 결  과 | 대신 CPU 전처리 2.26 ms가 늘어 판단 1회 지연은 2.93 대 2.53 ms로 비슷하지만, 줄어든 적재·추론 구간의  | 43 | — | avg_power.configs.가 u1000.E_min_uJ; breakeven.CPU log-mel, 간격 62.5 ms (멜 32) < 원본 파형 (전처리 없음).by_source.datasheet.E_X; breakeven.NPU 전처리 (Gabor 필터뱅크) < CPU log- | ✓ |
| 3. 결  과 | 대신 CPU 전처리 2.26 ms가 늘어 판단 1회 지연은 2.93 대 2.53 ms로 비슷하지만, 줄어든 적재·추론 구간의  | 71 | — | avg_power.configs.②′ f31.E_min_uJ; avg_power.configs.④.E_min_uJ; breakeven.CPU log-mel, 간격 50 ms < 원본 파형 (전처리 없음).by_source.datasheet.E_Y; breakeven.CPU log-mel | ✓ |
| 3. 결  과 | 프레임 간격을 좁힐수록 정확도는 오르지만(25 → 16 ms의 차이 -0.016 [-0.030, -0.002]) 전처리가 지연 | 25 | NPU 전처리 (Gabor 필터뱅크), 16 ms | listening.A_ext_scream/v1+실사용.n | ✓ |
| 3. 결  과 | 프레임 간격을 좁힐수록 정확도는 오르지만(25 → 16 ms의 차이 -0.016 [-0.030, -0.002]) 전처리가 지연 | 16 | NPU 전처리 (Gabor 필터뱅크), 16 ms | 16 kHz / 16 프레임 / 16 ms | 상수 |
| 3. 결  과 | 프레임 간격을 좁힐수록 정확도는 오르지만(25 → 16 ms의 차이 -0.016 [-0.030, -0.002]) 전처리가 지연 | -0.016 | NPU 전처리 (Gabor 필터뱅크), 16 ms | welch.간격 400 − ①′.test_f1_S3[0] | ⚠ |
| 3. 결  과 | 프레임 간격을 좁힐수록 정확도는 오르지만(25 → 16 ms의 차이 -0.016 [-0.030, -0.002]) 전처리가 지연 | -0.030 | NPU 전처리 (Gabor 필터뱅크), 16 ms | welch.간격 400 − ①′.test_f1_S3[1] | ⚠ |
| 3. 결  과 | 프레임 간격을 좁힐수록 정확도는 오르지만(25 → 16 ms의 차이 -0.016 [-0.030, -0.002]) 전처리가 지연 | -0.002 | NPU 전처리 (Gabor 필터뱅크), 16 ms | welch.간격 400 − ①′.test_f1_S3[2] | ⚠ |
| 3. 결  과 | 프레임 간격을 좁힐수록 정확도는 오르지만(25 → 16 ms의 차이 -0.016 [-0.030, -0.002]) 전처리가 지연 | 2/3 | NPU 전처리 (Gabor 필터뱅크), 16 ms | 전처리 지연 비중 | 상수 |
| 3. 결  과 | 프레임 간격을 좁힐수록 정확도는 오르지만(25 → 16 ms의 차이 -0.016 [-0.030, -0.002]) 전처리가 지연 | 6 | NPU 전처리 (Gabor 필터뱅크), 16 ms | acc.②′.scream_S3.sd; acc.간격 400.scream_S3.sd; avg_power.configs.간격 400.share_datasheet_at_hi_pct; avg_power.configs.간격 400.share_min_pct; hop_fraction.①′; laten | ✓ |
| 3. 결  과 | 프레임 간격을 좁힐수록 정확도는 오르지만(25 → 16 ms의 차이 -0.016 [-0.030, -0.002]) 전처리가 지연 | 62.5 | NPU 전처리 (Gabor 필터뱅크), 16 ms | 프레임 간격 ms | 상수 |
| 3. 결  과 | 프레임 간격을 좁힐수록 정확도는 오르지만(25 → 16 ms의 차이 -0.016 [-0.030, -0.002]) 전처리가 지연 | 50 | NPU 전처리 (Gabor 필터뱅크), 16 ms | 프레임 간격 ms / CNN 50 MHz | 상수 |
| 3. 결  과 | 프레임 간격을 좁힐수록 정확도는 오르지만(25 → 16 ms의 차이 -0.016 [-0.030, -0.002]) 전처리가 지연 | 25 | NPU 전처리 (Gabor 필터뱅크), 16 ms | listening.A_ext_scream/v1+실사용.n | ✓ |
| 3. 결  과 | 프레임 간격을 좁힐수록 정확도는 오르지만(25 → 16 ms의 차이 -0.016 [-0.030, -0.002]) 전처리가 지연 | 16 | NPU 전처리 (Gabor 필터뱅크), 16 ms | 16 kHz / 16 프레임 / 16 ms | 상수 |
| 3. 결  과 | NPU 전처리 (Gabor 필터뱅크)와 62.5 ms 구성은 에너지 차이가 약 4 μJ로 사실상 정확도에서 갈리고, 원본 파형 | 62.5 | 62.5 ms, NPU 근사 log, NPU 전처리 (Gabor 필터뱅크), 원본 파형 | 프레임 간격 ms | 상수 |
| 3. 결  과 | NPU 전처리 (Gabor 필터뱅크)와 62.5 ms 구성은 에너지 차이가 약 4 μJ로 사실상 정확도에서 갈리고, 원본 파형 | 4 | 62.5 ms, NPU 근사 log, NPU 전처리 (Gabor 필터뱅크), 원본 파형 | acc.가 u1000.scream_S3.sd; avg_power.decisions_per_s; breakeven.CPU log-mel, 간격 25 ms < CPU log-mel, 간격 16 ms.N_Y; editions.ext_f1_real31L.간격 400.v3.1.sd; firmwa | ✓ |
| 3. 결  과 | PANNs(CNN14)는 같은 정답·문턱값 규칙에서 시험 macro-F1 0.845이고, CPU log-mel 16 ms 구성 | 0.845 | 16 ms | acc.(a).ext_map.mean; panns.test_S3.f1 | ✓ |
| 3. 결  과 | PANNs(CNN14)는 같은 정답·문턱값 규칙에서 시험 macro-F1 0.845이고, CPU log-mel 16 ms 구성 | 16 | 16 ms | 16 kHz / 16 프레임 / 16 ms | 상수 |
| 3. 결  과 | PANNs(CNN14)는 같은 정답·문턱값 규칙에서 시험 macro-F1 0.845이고, CPU log-mel 16 ms 구성 | 99 | 16 ms | 유도: 100·f1(①′)/PANNs | ✓ |
| 4. 한계 | 에너지는 직접 계측하지 않은 추정값이며 전력 출처에 따라 결론의 강도가 다르다(CPU log-mel 62.5 ms 구성의 우위 | 62.5 | 62.5 ms | 프레임 간격 ms | 상수 |
| 4. 한계 | 판단 1회 에너지를 평균 전력으로 옮기면(데이터시트 기반 추정) 디지털 마이크를 유지하는 대기 전력이 대부분이라, 가장 깊은  | 1 | 62.5 ms, 16 ms | acc.(a).ext_f1.sd; acc.(a).ext_map.mean; acc.(a).ext_map.sd; acc.(a).f1_L.mean; acc.(a).f1_L.sd; acc.(a).f1_S3.mean; acc.(a).f1_S3.sd; acc.①′.ext_f1.mean; acc.① | ✓ |
| 4. 한계 | 판단 1회 에너지를 평균 전력으로 옮기면(데이터시트 기반 추정) 디지털 마이크를 유지하는 대기 전력이 대부분이라, 가장 깊은  | 1.9 | 62.5 ms, 16 ms | acc.다 u800.f1_L.sd; acc.다 u800.f1_S3.sd; avg_power.P_wait_lo_mW; avg_power.configs.다 u800.share_datasheet_at_hi_pct; avg_power.configs.다 u800.share_min_pct; edi | ✓ |
| 4. 한계 | 판단 1회 에너지를 평균 전력으로 옮기면(데이터시트 기반 추정) 디지털 마이크를 유지하는 대기 전력이 대부분이라, 가장 깊은  | 62.5 | 62.5 ms, 16 ms | 프레임 간격 ms | 상수 |
| 4. 한계 | 판단 1회 에너지를 평균 전력으로 옮기면(데이터시트 기반 추정) 디지털 마이크를 유지하는 대기 전력이 대부분이라, 가장 깊은  | 17 | 62.5 ms, 16 ms | avg_power.configs.가 u1000.share_max_pct; breakeven.NPU 전처리 (Gabor 필터뱅크) < 원본 파형 (전처리 없음).r_star; latency.다 u800.load; listening.T_test_scream/screaming_test.?;  | ✓ |
| 4. 한계 | 판단 1회 에너지를 평균 전력으로 옮기면(데이터시트 기반 추정) 디지털 마이크를 유지하는 대기 전력이 대부분이라, 가장 깊은  | 16 | 62.5 ms, 16 ms | 16 kHz / 16 프레임 / 16 ms | 상수 |
| 4. 한계 | 판단 1회 에너지를 평균 전력으로 옮기면(데이터시트 기반 추정) 디지털 마이크를 유지하는 대기 전력이 대부분이라, 가장 깊은  | 54 | 62.5 ms, 16 ms | acc.가 u1000.ext_f1.mean; avg_power.configs.①′.share_max_pct; avg_power.configs.다 u800.E_min_uJ; breakeven.CPU log-mel, 간격 50 ms < 원본 파형 (전처리 없음).by_source.datas | ✓ |
| 4. 한계 | 판단 1회 에너지를 평균 전력으로 옮기면(데이터시트 기반 추정) 디지털 마이크를 유지하는 대기 전력이 대부분이라, 가장 깊은  | 11 | 62.5 ms, 16 ms | avg_power.P_wait_hi_mW; breakeven.CPU log-mel, 간격 25 ms < CPU log-mel, 간격 16 ms.C_Y; listening.B_ext_glass/전체/verdict_first.?; listening.S_train_scream/yellshou | ✓ |
| 4. 한계 | 판단 1회 에너지를 평균 전력으로 옮기면(데이터시트 기반 추정) 디지털 마이크를 유지하는 대기 전력이 대부분이라, 가장 깊은  | 42 | 62.5 ms, 16 ms | avg_power.configs.M4 CMSIS-NN.share_datasheet_at_hi_pct; avg_power.configs.M4 CMSIS-NN.share_min_pct; avg_power.configs.간격 400.share_max_pct; editions.ext_f1_re | ⚠ |
| 4. 한계 | 시드가 3개라 신뢰구간이 넓고, 정답 청취는 저자 1인이며, 비명과 환호의 경계는 음향만으로 가를 수 없어 비명 재현율은 그  | 3 | — | acc.(a).ext_f1.n; acc.(a).ext_map.n; acc.(a).f1_L.n; acc.(a).f1_S3.n; acc.(a).scream_S3.n; acc.(a).scream_S3.sd; acc.①′.ext_f1.n; acc.①′.ext_map.n; acc.①′.f1_L. | ✓ |
| 4. 한계 | 시드가 3개라 신뢰구간이 넓고, 정답 청취는 저자 1인이며, 비명과 환호의 경계는 음향만으로 가를 수 없어 비명 재현율은 그  | 1 | — | acc.(a).ext_f1.sd; acc.(a).ext_map.mean; acc.(a).ext_map.sd; acc.(a).f1_L.mean; acc.(a).f1_L.sd; acc.(a).f1_S3.mean; acc.(a).f1_S3.sd; acc.①′.ext_f1.mean; acc.① | ✓ |
| 5. 결론 및 향후 과제 | NPU 전처리의 CPU 몫 80%는 log를 NPU에서 수행할 수 없어 int32 중간 결과를 CPU로 읽어 오는 데이터 이동 | 80 | — | acc.간격 400.f1_L.mean; firmware.cpu_split.read_pct; firmware.cpu_split.read_share_med; firmware.cpu_split.read_share_worst; firmware.same_fw_fe_vs_ga.moosmann.ga | ✓ |
| 5. 결론 및 향후 과제 | 전력계로 에너지를 직접 계측해 전력 출처 간 차이를 없애고, 실배치 분포의 배경과 복수 판정자 정답으로 데이터셋을 보강하며,  | 8 | — | acc.다 u800.scream_S3.sd; avg_power.configs.(a).share_datasheet_at_hi_pct; latency.②′.detail.infer1; listening.A_ext_scream/실사용만.?; welch.(a) − ④.test_f1_L[2]; w | ✓ |

## 구간별 에너지 계산 — 원본 파형(④) 대 CPU log-mel 62.5 ms(가), 추정(실측 지연 × 공개 전력)

| 전력 출처 | 구성 | 전처리 ms × P_cpu | 적재+출력 ms × P_load | 추론 ms × P_npu | 합 μJ | numbers.json energy |
|---|---|---|---|---|---|---|
| datasheet | ④ | 0.00 × 8.89 = 0.0 | 1.10 × 11.26 = 12.3 | 1.44 × 40.79 = 58.7 | 71.0 | 71.0 |
| datasheet | 가 u1000 | 2.26 × 8.89 = 20.1 | 0.14 × 11.26 = 1.6 | 0.53 × 40.79 = 21.7 | 43.3 | 43.3 |
| literature | ④ | 0.00 × 24.8 = 0.0 | 1.10 × 24.8 = 27.2 | 1.44 × 71.7 = 103.2 | 130.4 | 130.4 |
| literature | 가 u1000 | 2.26 × 24.8 = 56.1 | 0.14 × 24.8 = 3.4 | 0.53 × 71.7 = 38.1 | 97.7 | 97.7 |
| moosmann | ④ | 0.00 × 24.8 = 0.0 | 1.10 × 24.8 = 27.2 | 1.44 × 35.6 = 51.2 | 78.4 | 78.4 |
| moosmann | 가 u1000 | 2.26 × 24.8 = 56.1 | 0.14 × 24.8 = 3.4 | 0.53 × 35.6 = 18.9 | 78.5 | 78.5 |

| 전력 출처 | 줄어든 적재+추론 에너지 (④ − 가) | 늘어난 전처리 에너지 (가 − ④) | 순 변화 (가 − ④) | 감소 주원인 |
|---|---|---|---|---|
| datasheet | 47.8 (적재 10.8 + 추론 37.1) | 20.1 | -27.7 | 추론 |
| literature | 88.9 (적재 23.7 + 추론 65.2) | 56.1 | -32.7 | 추론 |
| moosmann | 56.1 (적재 23.7 + 추론 32.3) | 56.1 | +0.1 | 추론 |

원고 문장(데이터시트 기준)은 '줄어든 적재·추론 구간의 에너지가 늘어난 CPU 구간보다 커서' 다 — 세 출처 모두 감소분 > 증가분이고, 감소의 주원인은 추론 구간이다(적재 구간도 줄었다). Moosmann 기준은 순 변화 +0.1 μJ 로 '같다'.

## 비교 문장 참/거짓 표 ('A 가 B 보다 작다/크다/같다' 류, 양쪽 값을 numbers.json 에서 계산)

| 절 | 주장 | 좌변 | 우변 | 판정 |
|---|---|---|---|---|
| 요약·3절 | 원본 파형은 CPU log-mel 16 ms 보다 시험 macro-F1 이 낮다 | ④ 0.669 | ①′ 0.836 (차 0.166) | 참 |
| 3절 | 외부 F1 도 낮다 | ④ 0.443 | ①′ 0.665 (차 0.222) | 참 |
| 요약·3절 | CPU log-mel 62.5 ms 는 원본 파형보다 macro-F1 이 유의하게 높다 (CI 가 0 제외) | 가 0.754 | ④ 0.669; Welch +0.085 [+0.035, +0.135] | 참 |
| 3절 | 외부 F1 도 유의하게 높다 | 가 0.542 | ④ 0.443; +0.099 [+0.056, +0.142] | 참 |
| 요약·3절·4절 | 가 의 추정 에너지는 ④ 보다 적다 (datasheet) | 가 43.3 μJ | ④ 71.0 μJ | 참 |
| 요약·3절·4절 | 가 의 추정 에너지는 ④ 보다 적다 (literature) | 가 97.7 μJ | ④ 130.4 μJ | 참 |
| 요약·3절·4절 | 가 의 추정 에너지는 ④ 보다 같다 (moosmann) | 가 78.5 μJ | ④ 78.4 μJ | 참 |
| 3절 | 가 와 ④ 의 판단 1회 지연은 비슷하다 | 가 2.93 ms | ④ 2.53 ms | 참 |
| 3절 | NPU 추론은 CPU 추론보다 추론만 157배 빠르다 | M4 225.50 ms | ④ 1.44 ms (비 156.7) | 참 |
| 3절 | 판단 1회 89배 | M4 225.77 ms | ④ 2.53 ms (비 89.1) | 참 |
| 3절 | CPU 추론은 판단 주기의 90% 를 추론에 쓴다 | M4 225.77 ms | 주기 250 ms (비 90.3%) | 참 |
| 3절 | NPU 근사 log 의 ④ 대비 CI 는 0 을 포함 |  | +0.034 [-0.022, +0.090] | 참 |
| 3절 | NPU 근사 log 의 추정 에너지는 ④ 의 3.3배 | (a) 236.5 μJ | ④ 71.0 μJ (비 3.33) | 참 |
| 3절 | ②′ 는 '원본 파형 초과' 목표를 넘지 못했다 (CI 가 0 포함) |  | -0.026 [-0.082, +0.030] | 참 |
| 3절 | ②′ 는 '가 와 동등' 목표를 넘지 못했다 (CI 가 0 제외, 음수) |  | -0.111 [-0.162, -0.060] | 참 |
| 3절 | ②′ 의 시험 macro-F1 은 가 보다 0.11 낮다 | ②′ 0.643 | 가 0.754 (Welch 차 -0.111) | 참 |
| 요약·3절·4절 | ②′ 의 추정 에너지는 ④ 보다 적다 (datasheet) | ②′ 38.9 μJ | ④ 71.0 μJ | 참 |
| 요약·3절·4절 | ②′ 의 추정 에너지는 ④ 보다 적다 (literature) | ②′ 74.5 μJ | ④ 130.4 μJ | 참 |
| 요약·3절·4절 | ②′ 의 추정 에너지는 ④ 보다 적다 (moosmann) | ②′ 52.4 μJ | ④ 78.4 μJ | 참 |
| 3절 | ②′ 와 가 는 비슷한 에너지 (데이터시트 차이 약 4 μJ) | ②′ 38.9 μJ | 가 43.3 μJ (차 4.5) | 참 |
| 3절 | ②′ 판단 1회 지연 1.85 < ④ 2.53 ms | ②′ 1.85 ms | ④ 2.53 ms | 참 |
| 3절 | 변형 3종은 ②′ 보다 정확도가 올랐다 | 변형 0.655~0.690 | ②′ 0.643 | 참 |
| 3절 | 변형 3종은 세 전력 출처 모두에서 CPU 전처리 곡선 아래 (지배됨) |  | {'②′ b64': {'datasheet': ['가 u1000', '다 u800'], 'literature': ['가 u1000', '다 u800'], 'moosmann': ['가 u1000', '다 u800']}, '②′ f31': {'datasheet': ['가 u1000', '다 u800'], 'literature': ['가 u1000', '다 u800'], 'moosmann': ['가 u1000']}, '②′ b64f31': {'datasheet': ['가 u1000', '다 u800'], 'literature': ['가 u1000', '다 u800'], 'moosmann': ['가 u1000', '다 u800']}} | 참 |
| 3절 | 늘어난 비용은 CPU 읽기와 NPU 연산이 비슷하게 나눴다 (CPU 몫이 50% 를 넘지 않음) | CPU 46~51% | NPU 39~45% | 참 |
| 3절 | NPU 입력 적재와 추론이 줄었다 (④ → 가) | 적재 1.09 → 0.13 ms | 추론 1.44 → 0.53 ms | 참 |
| 3절 | 줄어든 적재·추론 구간 에너지 > 늘어난 CPU 구간 에너지 (데이터시트) | 감소 47.8 μJ | 증가 20.1 μJ | 참 |
| 3절 | 프레임 간격을 좁힐수록 정확도가 오른다 (62.5 → 50 → 25 → 16 ms) | 0.754 → 0.758 → 0.820 → 0.836 |  | 참 |
| 3절 | CPU log-mel 16 ms 는 전처리가 지연의 2/3 | 전처리 9.87 ms | 총 15.06 ms (비 0.66) | 참 |
| 3절 | 16 ms 의 에너지는 62.5 ms 의 6배 | ①′ 266.8 μJ | 가 43.3 μJ (비 6.16) | 참 |
| 3절 | 25 ms 와 16 ms 의 차이는 작지만 CI 가 0 을 포함하지 않는다 | 25 ms 0.820 | 16 ms 0.836; -0.016 [-0.030, -0.002] | 참 |
| 3절 | ④ 와 (a) 는 파레토 앞면에 없다 (다른 점에 지배) | 앞면 ②′ → 가 u1000 → 다 u800 → 간격 400 → ①′ | ④ E 71/F1 0.669 vs 가 E 43/F1 0.754; (a) E 236/F1 0.703 | 참 |
| 3절 | 외부 데이터의 앞면 순서가 시험과 같다 | ②′ → 가 u1000 → 다 u800 → 간격 400 → ①′ | ②′ → 가 u1000 → 다 u800 → 간격 400 → ①′ | 참 |
| 3절 | CPU log-mel 16 ms 는 PANNs 의 99% | ①′ 0.836 | PANNs 0.845 (비 98.9%) | 참 |
| 3절 | ②′ CPU 몫의 80% 는 읽기, 20% 는 로그 | 읽기 0.570 ms | 로그 0.142 ms, CPU 0.712 ms | 참 |
| 4절 | 가 의 판단 비용은 평균 전력의 17% 이하 (대기 하한) | 16.9% |  | 참 |
| 4절 | ①′ 는 54% 이하 | 54.1% |  | 참 |
| 4절 | CPU 추론은 대기 상한에서도 42% 이상 | 41.7% |  | 참 |
| 4절 | 62.5 ms 구성의 우위는 '적다' 와 '같다' 사이 (출처별) | 데이터시트 43<71, 선행연구 98<130 | Moosmann 78.5 ≈ 78.4 | 참 |

비교 문장 38개 중 거짓 0개.

