# KAT 벡터 (SafeSound 구성 ① — 로그 멜)

`tools/kat_safesound_mel.py` 가 생성한다. **손으로 고치지 말 것.**

| 파일 | 내용 |
|---|---|
| `melkat_window_int8.npy` | 1초 int8 창 (16384) — ④와 같은 지점의 값이다 |
| `melkat_logmel_int8.npy` | int8 로그 멜 (64×64, mel-major) |
| `melkat_vectors.h` | 위 둘의 C 배열 — 펌웨어용 |

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
