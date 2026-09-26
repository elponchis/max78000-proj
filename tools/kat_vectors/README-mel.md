# KAT 벡터 (SafeSound 구성 ① — 로그 멜)

`tools/kat_safesound_mel.py` 가 생성한다. **손으로 고치지 말 것.**

| 파일 | 내용 |
|---|---|
| `melkat_window_int8.npy` | 1초 int8 창 (16384) — ④와 같은 지점의 값이다 |
| `melkat_logmel_int8.npy` | int8 로그 멜 (64×64, mel-major) |
| `melkat_logmel_db.npy` | 양자화 **전** float dB (float32) — int8 불일치가 반올림인지 계산 차이인지 가른다 |
| `melkat_vectors.h` | int8 둘의 C 배열 — 펌웨어용 |

## 회귀 검사 허용 오차 (PC↔PC)

최대 절대 차이 ≤ **1 LSB** 이고 불일치 원소 비율 ≤ **0.1%** 면 통과(경고)다. 같은 파이썬 구현이라 차이가
날 이유는 numpy/FFT 반올림뿐이고, 그건 int8 경계에 걸친 소수의
원소에서 1 LSB 로만 나타난다. float dB 차이가 1 LSB 의 절반 미만이면
반올림 문제임이 확정된다.

⚠️ 이 벡터는 **실제 샤드의 test 창 0번**에서 만든다. 샤드를 못 찾으면
검사는 생략이 아니라 **실패**다 — Colab 에서 경로가 달라 합성 톤으로
대체됐고, 그 때문에 회귀 검사만 실패해 수치 오차로 오진할 뻔했다.
`--data /content/ai8x-training/data/SafeSound` 로 넘길 것.

## 상수 (바꾸면 학습·KAT·펌웨어를 함께 재생성할 것)

n_fft 512 / hop 256 / reflect pad 128 / n_mels 64 / 20~8000Hz HTK / TOP_DB 0.0 / SPAN_DB 70.0 (1 LSB = 0.275dB)

근거는 `docs/results/mel-range-sweep.md` 다.

## 쓰는 곳

1. PC: `python3 tools/kat_safesound_mel.py` — 전처리 회귀 검사
2. 보드: `melkat_window` 를 M4 전처리(CMSIS-DSP)에 넣고 결과를
   `melkat_logmel` 과 비교. **허용 오차 ±2 LSB**(≈0.55dB).
   넘으면 창 함수·멜 계수·dB 기준점 중 하나가 다른 것이다.

⚠️ 구성 ④(raw 파형)에는 이 오차 항목이 **없다**. 전처리가 옮겨 담기뿐
이기 때문이다. 이 차이가 G4(온디바이스 vs PC 괴리)에서 ①이 ④보다
크게 어긋날 것으로 보는 근거이고, 그 자체가 G8 의 결과 중 하나다.
