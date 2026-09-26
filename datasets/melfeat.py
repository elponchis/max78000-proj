#!/usr/bin/env python3
"""로그 멜 프론트엔드 — G8 구성 ① 의 CPU 전처리 기준 구현 (numpy 전용).

`docs/results/config1-mel2d-design.md` 의 1절을 그대로 코드로 옮긴 것이다.
**torch 도 ai8x 도 import 하지 않는다** — 이유는 셋이다.

  · 학습(`datasets/safesound_mel.py`)·KAT(`tools/kat_safesound_mel.py`)·
    분포 스윕(`tools/mel_range_sweep.py`)이 **한 구현**을 공유해야 한다.
    구성 ① 의 전부가 "입력 표현" 이므로 구현이 갈리면 실험이 무의미하다
  · 보드 도착 후 M4 펌웨어(CMSIS-DSP)가 이 파일을 **이식 기준**으로 삼는다.
    의존성이 적을수록 한 줄씩 대조하기 쉽다
  · 전처리 분포를 재는 도구가 GPU 환경 없이 WSL2 에서 돌아야 한다

⚠️ **정규화를 하지 않는다** (CLAUDE.md 7장). 창 단위도 파일 단위도 아니다.
실기기 경로에 정규화가 없기 때문이다. dB 는 **절대 레벨**을 뜻한다 —
풀스케일 사인파가 필터 꼭대기에 맞으면 0dB 다.

── 실기기 이식 시 주의 ───────────────────────────────────────────────────
M4 는 `arm_rfft_fast_f32` 또는 q15 고정소수점으로 이 경로를 구현한다. PC float64
와 비트 단위로 같을 수 없으므로, `tools/kat_safesound_mel.py` 가 저장한 기준
벡터로 **허용 오차를 정하고 지킨다**. 이 오차가 G4(온디바이스 vs PC 괴리)의
구성 ① 쪽 원인이고, 구성 ④(raw 파형)에는 없는 항목이다 — 그 대조 자체가 결과다.
"""

import numpy as np

SR = 16000
N_FFT = 512          # 32ms
HOP = 256            # 16ms
PAD = (N_FFT - HOP) // 2     # 128 — reflect 패딩으로 프레임 수를 64에 맞춘다
N_MELS = 64
N_FRAMES = 64
FMIN = 20.0
FMAX = 8000.0
EPS = 1e-12          # log 보호. -120dB

# dB → int8 매핑 상수. **양쪽 끝이 물리적으로 유도된다** (2026-09-26 실측,
# docs/results/mel-range-sweep.md). 바꾸면 학습·KAT·펌웨어가 전부 어긋나므로
# 바꿀 때는 세 곳을 함께 재생성할 것.
#
#   위쪽 0dB  = 풀스케일 사인파. 실측 -0.75dB (1kHz 가 멜 필터 꼭대기와 정확히
#               겹치지 않아 생기는 차이). 이벤트 클래스 멜 빈의 0.006% 만 포화한다
#   아래쪽 -70dB = **int8 파형의 양자화 잡음 바닥보다 아래**다. ±1 LSB 잡음만 든
#               창의 멜 dB 를 재면 저역 -64.6 ~ 고역 -54.5, 중앙값 -58.6 이었다.
#               -70dB 아래는 실제 신호가 아니라 수치 잔여물이므로 잘라도 잃는
#               정보가 없다. 저역 최악값(-64.6)에도 5dB 여유를 둔다
#
# → 1 LSB = 70/255 = 0.275dB. 이벤트 멜 빈의 17.6%가 바닥에 붙지만, 그것들이
#   곧 양자화 잡음 구간이다. glass 가 특히 많은 것은 0.3초 과도음이라 창의
#   대부분이 조용한 것이 정상이기 때문이다 (CLAUDE.md 7장).
TOP_DB = 0.0         # 이 값 이상은 전부 +127 로 포화
SPAN_DB = 70.0       # TOP_DB - SPAN_DB 가 -128


def hz_to_mel(f):
    """HTK 멜 스케일. Slaney 가 아니라 HTK 를 쓰는 이유는 식이 한 줄이라
    고정소수점 이식이 단순하기 때문이다."""
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, sr=SR, fmin=FMIN, fmax=FMAX):
    """삼각 필터뱅크 (n_mels, n_fft//2+1).

    **면적 정규화를 하지 않는다** — 꼭대기가 1.0 인 순수 삼각형이다. librosa 의
    'slaney' 정규화는 필터마다 이득이 달라 고정소수점에서 스케일 관리가 번거롭고,
    우리는 어차피 로그를 씌운 뒤 고정 스케일로 양자화하므로 이득이 없다.
    """
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    freqs = np.arange(n_fft // 2 + 1, dtype=np.float64) * sr / n_fft
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(n_mels):
        lo, ctr, hi = edges[m], edges[m + 1], edges[m + 2]
        up = (freqs - lo) / max(ctr - lo, 1e-9)
        dn = (hi - freqs) / max(hi - ctr, 1e-9)
        fb[m] = np.maximum(0.0, np.minimum(up, dn))
    return fb


def hann(n=N_FFT):
    """주기형 Hann. CMSIS-DSP 및 일반 STFT 관례와 같다 (대칭형이 아니다)."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


_FB = mel_filterbank()
_WIN = hann()
# 풀스케일 사인파의 한 빈 진폭이 sum(win)/2 이므로, 이것의 제곱으로 나누면
# 풀스케일 사인 = 0dB 가 된다. 절대 레벨 정책(7장)에 맞춘 기준점이다.
_REF = (float(_WIN.sum()) / 2.0) ** 2


def stft_power(x):
    """(N_FRAMES, n_fft//2+1) 정규화 파워 스펙트럼.

    입력은 float 파형이며 풀스케일이 ±1.0 이다. reflect 패딩을 쓰는 이유는
    `center=True` 대신 **프레임 수를 64 에 정확히 맞추면서** 구성 ④ 와 같은
    16,384 샘플만 보기 위해서다 — 여유 샘플을 더 가져오면 두 구성이 서로 다른
    오디오를 보게 되어 비교가 깨진다.
    """
    xp = np.pad(np.asarray(x, dtype=np.float64), (PAD, PAD), mode="reflect")
    idx = np.arange(N_FFT)[None, :] + (np.arange(N_FRAMES) * HOP)[:, None]
    frames = xp[idx] * _WIN[None, :]
    spec = np.fft.rfft(frames, n=N_FFT, axis=1)
    return (spec.real ** 2 + spec.imag ** 2) / _REF


def log_mel_db(w_int8):
    """int8 파형 (16384,) → 로그 멜 dB (N_MELS, N_FRAMES) float64.

    int8 을 128 로 나눠 ±1.0 풀스케일로 맞춘다. 학습·평가·펌웨어가 모두 이
    지점에서 시작하므로 **여기서 정규화를 끼워 넣으면 안 된다**.
    """
    x = np.asarray(w_int8, dtype=np.float64) / 128.0
    mel = stft_power(x) @ _FB.T                     # (frames, mels)
    return 10.0 * np.log10(mel.T + EPS)             # (mels, frames)


def db_to_int8(lm_db, top_db=TOP_DB, span_db=SPAN_DB):
    """dB → int8. 고정 아핀이며 **데이터에 의존하지 않는다**.

    `top_db` 이상은 +127 로 포화하고 `top_db - span_db` 이하는 -128 이다.
    포화시키는 것이 중요하다 — MSDK 데모의 랩어라운드 버그와 같은 실수를
    전처리에서 반복하지 않는다 (CLAUDE.md 7장).
    """
    q = np.round((np.asarray(lm_db) - top_db) * (255.0 / span_db)) + 127.0
    return np.clip(q, -128.0, 127.0).astype(np.int8)


def log_mel_int8(w_int8, top_db=TOP_DB, span_db=SPAN_DB):
    """int8 파형 (16384,) → int8 로그 멜 (N_MELS, N_FRAMES). 모델 입력이다."""
    return db_to_int8(log_mel_db(w_int8), top_db, span_db)
