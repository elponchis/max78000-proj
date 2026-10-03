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

# ⚠️ **CPU 에서 이 모듈을 쓸 때는 `OPENBLAS_NUM_THREADS=1` 을 걸 것.**
# 멜 필터뱅크 행렬곱 `(64,257)@(257,64)` 은 2 MFLOP 짜리 **작은** 연산인데,
# OpenBLAS 의 스레드 동기화 오버헤드가 23ms 붙어 `mel_int8()` 전체가 41.5ms 가
# 된다 (창마다 한 번씩 부른다 -> 학습 배치당 5.6초). 스레드를 1로 두면 0.616ms 다.
# 결과는 **비트 단위로 동일**하고 torch 는 영향받지 않는다.
# 실측과 근거: CLAUDE.md 12장 "OpenBLAS 소행렬 오버헤드" (2026-09-29).

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


# ── 프레임 정의 ①′ — 증분 계산이 되는 판 (2026-10-02) ─────────────────────
# 위의 정의(hop 256 + 창 단위 reflect 패딩)는 **판단마다 64프레임을 전부**
# 다시 계산해야 한다: hop 256 이 판단 주기 4,000 샘플의 약수가 아니고, 프레임이
# 창의 양끝 패딩에 묶여 있다. 보드 실측 47.87 ms (synthesis-check.md 10절).
#
# ①′ 는 hop 을 **250** 으로 두고 패딩을 없앤다. 판단 주기 4,000 = 정확히
# 16프레임이라 판단마다 **새 프레임 16개만** 계산하고 나머지 48개는 링 버퍼에서
# 재사용한다. 프레임 k 는 스트림의 [250k, 250k+512) 에만 의존한다.
#   · 쓰는 샘플: 63*250 + 512 = 16,262. 1초 창의 **끝에 맞춘다** (샘플
#     122..16,383) — 실기기의 "가장 최근 64프레임" 과 같다
#   · 맨 앞 122 샘플(7.6 ms)은 보지 않는다 → ④ 와 0.7% 다른 오디오를 본다
#     (논문 방법·한계에 명시)
#   · n_fft·창·멜·로그 스케일(TOP_DB 0 / SPAN 70)은 그대로다
HOP_INC = 250
OFF_INC = 16384 - ((N_FRAMES - 1) * HOP_INC + N_FFT)      # 122

# ── "정확도 대 에너지 곡선" 변형 (2026-10-02) ─────────────────────────────
# ①′ 의 CPU 전처리 비용을 줄인 프레임 정의들. 전부 **증분 계산이 된다** (hop 이
# 판단 주기 4,000 의 약수), 끝 정렬·무패딩·고정 로그 스케일은 ①′ 와 같다.
# 프레임 수는 1초 창(16,384)에 들어가는 최대값이다.
#   이름      hop  프레임  멜   판단당 새 프레임  쓰는 샘플   겹침
#   inc       250    64   64        16           16,262     51%   ← ①′
#   h500      500    32   64         8           16,012      2%
#   h500m32   500    32   32         8           16,012      2%
#   m32       250    64   32        16           16,262     51%
#   h400      400    40   64        10           16,112     22%
INC_SPECS = {
    "inc": (250, 64, 64),
    "h500": (500, 32, 64),
    "h500m32": (500, 32, 32),
    "m32": (250, 64, 32),
    "h400": (400, 40, 64),
    # ── 2차 후보 (2026-10-03) — "추정 에너지가 ④ 이하인 로그 멜이 ④보다 정확한가"
    #   이름         hop   프레임  멜  n_fft  새 프레임  쓰는 샘플  미관측 구간
    #   u1000        1000    16   32   512      4       15,512     48.8%
    #   u1000f1024   1000    16   32  1024      4       16,024      0% (겹침 2%)
    #   u800          800    20   32   512      5       15,712     36.0%
    # hop > n_fft 이면 프레임 사이에 **어느 프레임도 보지 않는 샘플**이 생긴다.
    # u1000 대 u1000f1024 는 입력 모양이 같고 n_fft 만 달라 그 영향을 따로 본다.
    "u1000": (1000, 16, 32),
    "u1000f1024": (1000, 16, 32),
    "u800": (800, 20, 32),
}
# 프레임 정의별 n_fft. 여기 없으면 N_FFT(512).
INC_NFFT = {"u1000f1024": 1024}
_FB_CACHE = {(N_MELS, N_FFT): _FB}
_WIN_CACHE = {N_FFT: (_WIN, _REF)}


def inc_nfft(framing):
    return INC_NFFT.get(framing, N_FFT)


def _fb(n_mels, n_fft=N_FFT):
    if (n_mels, n_fft) not in _FB_CACHE:
        _FB_CACHE[(n_mels, n_fft)] = mel_filterbank(n_mels=n_mels, n_fft=n_fft)
    return _FB_CACHE[(n_mels, n_fft)]


def _win_ref(n_fft):
    """(Hann 창, 기준 파워). 기준은 풀스케일 사인 = 0dB 가 되게 창마다 다시 잡는다."""
    if n_fft not in _WIN_CACHE:
        w = hann(n_fft)
        _WIN_CACHE[n_fft] = (w, (float(w.sum()) / 2.0) ** 2)
    return _WIN_CACHE[n_fft]


def inc_offset(framing):
    """끝 정렬 오프셋 — 창의 맨 앞에서 쓰지 않는 샘플 수."""
    hop, n_frames, _ = INC_SPECS[framing]
    return 16384 - ((n_frames - 1) * hop + inc_nfft(framing))


def framing_shape(framing):
    """(멜 수, 프레임 수) — 모델 입력의 모양."""
    if framing in INC_SPECS:
        return INC_SPECS[framing][2], INC_SPECS[framing][1]
    return N_MELS, N_FRAMES


def stft_power(x, framing="reflect"):
    """(N_FRAMES, n_fft//2+1) 정규화 파워 스펙트럼.

    입력은 float 파형이며 풀스케일이 ±1.0 이다. reflect 패딩을 쓰는 이유는
    `center=True` 대신 **프레임 수를 64 에 정확히 맞추면서** 구성 ④ 와 같은
    16,384 샘플만 보기 위해서다 — 여유 샘플을 더 가져오면 두 구성이 서로 다른
    오디오를 보게 되어 비교가 깨진다.

    `framing="inc"` 는 ①′ 다 (위 주석): hop 250, 패딩 없음, 끝 정렬.
    """
    n_fft = N_FFT
    if framing in INC_SPECS:
        hop, n_frames, _ = INC_SPECS[framing]
        n_fft = inc_nfft(framing)
        xp = np.asarray(x, dtype=np.float64)
        idx = (inc_offset(framing) + np.arange(n_fft)[None, :]
               + (np.arange(n_frames) * hop)[:, None])
    else:
        xp = np.pad(np.asarray(x, dtype=np.float64), (PAD, PAD), mode="reflect")
        idx = np.arange(N_FFT)[None, :] + (np.arange(N_FRAMES) * HOP)[:, None]
    win, ref = _win_ref(n_fft)
    frames = xp[idx] * win[None, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    return (spec.real ** 2 + spec.imag ** 2) / ref


def log_mel_db(w_int8, framing="reflect"):
    """int8 파형 (16384,) → 로그 멜 dB (N_MELS, N_FRAMES) float64.

    int8 을 128 로 나눠 ±1.0 풀스케일로 맞춘다. 학습·평가·펌웨어가 모두 이
    지점에서 시작하므로 **여기서 정규화를 끼워 넣으면 안 된다**.
    """
    x = np.asarray(w_int8, dtype=np.float64) / 128.0
    fb = _fb(framing_shape(framing)[0],
             inc_nfft(framing) if framing in INC_SPECS else N_FFT)
    mel = stft_power(x, framing) @ fb.T             # (frames, mels)
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


# ─────────────────────────────────────────────────────────────────────────
# 압축 법칙 (G8 "로그 압축 가설", 2026-09-29)
#
# k=1/2/4 해상도 곡선이 전부 잡음 안이었으므로(docs/results/g8-d1-filterbank.md
# 부록 A), (1)-(4) 격차의 다음 후보는 **로그 압축**이다. 로그는 두 가지 일을
# 동시에 한다.
#
#   (a) 압축: 70dB 를 255단계에 담는다. 선형으로는 담기는 범위가 정해져 있다 —
#       1 LSB = 1/255 이므로 파워 선형 24.1dB / 진폭 선형 48.1dB /
#       세제곱근 72.2dB. **파워 선형은 이벤트 멜 빈의 94.2%를 -128 로 만든다.**
#   (b) 음량 불변: 게인 g 는 로그에서 **평행이동**이다. 실측 dq 중앙값 +-44,
#       IQR 1~3 (거의 완전한 평행이동). 거듭제곱 법칙은 빈마다 다르게 움직여
#       IQR 이 중앙값의 2배까지 간다.
#
# 근거 수치: docs/results/lin-mel-range.md (tools/lin_mel_range.py 실측).
#
# MAX78000 은 로그도 거듭제곱도 지원하지 않는다 — 둘 다 M4 가 한다. 구성 (1)
# 안에서는 선택이 자유롭고, 이 실험의 목적은 **(4)(압축 없음)가 왜 불리한지**를
# 가리는 것이다.

# 거듭제곱 판의 포화점. 이벤트 4클래스 멜 파워 p99.9 = -6.8dB 실측을 -7.0 으로
# 반올림했다 (클래스당 100창 표본이라 소수 첫째 자리는 표본 잡음 안이다).
# 위쪽을 풀스케일(0dB)이 아니라 여기에 두는 이유는, 담기는 범위가 로그보다
# 좁은 법칙일수록 **위쪽 7dB 를 버려서 아래쪽 7dB 를 얻는 쪽**이 이득이기
# 때문이다. 포화하는 이벤트 빈은 0.1% 다.
COMP_TOP_DB = -7.0

# alpha=None 은 로그다. 숫자면 p**alpha 거듭제곱 압축이다.
#   span(dB) = (10/alpha) * log10(255)
COMP_SCHEMES = {
    "log":  (None, TOP_DB, SPAN_DB),        # 현행 (1). 기준선
    "log72": (None, COMP_TOP_DB, 72.2),     # 로그, 담는 범위를 cbrt 와 맞춘 대조
    "lin":  (0.5, COMP_TOP_DB, None),       # 진폭 선형 (48.1dB)
    "linp": (1.0, COMP_TOP_DB, None),       # 파워 선형 (24.1dB) — 참고용
    "cbrt": (1.0 / 3.0, COMP_TOP_DB, None),  # 세제곱근 (72.2dB)
    # ①′ — 압축은 "log" 와 **같고 프레임 정의만** 다르다 (SCHEME_FRAMING)
    "loginc": (None, TOP_DB, SPAN_DB),
    # B 사전 확인 — STFT 를 **NPU 의 고정 Conv1d** 로 옮긴 판의 시뮬레이션.
    # 프레임·압축은 "loginc" 와 같고 **DFT 기저가 정수 가중치**라는 점만 다르다.
    "loginc_q8": (None, TOP_DB, SPAN_DB),
    "loginc_q4": (None, TOP_DB, SPAN_DB),
    # "정확도 대 에너지 곡선" 변형 — 압축은 같고 프레임 정의(INC_SPECS)만 다르다
    "log_h500": (None, TOP_DB, SPAN_DB),
    "log_h500m32": (None, TOP_DB, SPAN_DB),
    "log_m32": (None, TOP_DB, SPAN_DB),
    "log_h400": (None, TOP_DB, SPAN_DB),
    # 2차 후보 (초저비용)
    "log_u1000": (None, TOP_DB, SPAN_DB),
    "log_u1000f1024": (None, TOP_DB, SPAN_DB),
    "log_u800": (None, TOP_DB, SPAN_DB),
}

# 법칙 → 프레임 정의. 여기 없으면 "reflect"(현행 ①)다.
SCHEME_FRAMING = {"loginc": "inc", "loginc_q8": "inc", "loginc_q4": "inc",
                  "log_h500": "h500", "log_h500m32": "h500m32",
                  "log_m32": "m32", "log_h400": "h400",
                  "log_u1000": "u1000", "log_u1000f1024": "u1000f1024",
                  "log_u800": "u800"}


def scheme_shape(scheme):
    """법칙 이름 → (멜 수, 프레임 수)."""
    return framing_shape(SCHEME_FRAMING.get(scheme, "reflect"))
# 법칙 → STFT 기저의 가중치 비트폭 (NPU Conv1d 로 옮긴 판). 없으면 float FFT.
SCHEME_STFT_BITS = {"loginc_q8": 8, "loginc_q4": 4}
_QBASIS = {}


def _stft_basis_q(bits):
    """(2, n_fft//2+1, n_fft) 정수 DFT 기저와 스케일 S.

    `w[0,k,n] = round(S * hann[n] * cos(2πkn/N))`, `w[1,k,n] = round(S * hann[n] *
    -sin(...))`, S = 2^(bits-1) - 1. NPU Conv1d 한 층의 가중치가 된다 (층 안의
    가중치는 스케일 하나를 공유한다).
    """
    if bits not in _QBASIS:
        s = float(2 ** (bits - 1) - 1)
        n = np.arange(N_FFT)
        k = np.arange(N_FFT // 2 + 1)[:, None]
        ang = 2.0 * np.pi * k * n[None, :] / N_FFT
        w = np.stack([np.round(s * _WIN[None, :] * np.cos(ang)),
                      np.round(s * _WIN[None, :] * -np.sin(ang))]).astype(np.int64)
        _QBASIS[bits] = (w, s)
    return _QBASIS[bits]


def stft_power_q(w_int8, bits, framing="inc"):
    """NPU 고정 Conv1d STFT 의 시뮬레이션 — **정수 누산값**에서 파워를 만든다.

    NPU 는 int8 입력 × int8(또는 4bit) 가중치의 합을 int32 로 내고(활성화 없는
    wide 출력), CPU 가 re² + im² → 멜 → 로그를 한다. 누산은 정확한 정수라
    여기 numpy 계산이 곧 기기 값이다 (2의 거듭제곱 스케일만 다를 수 있다).
    반환은 `stft_power` 와 같은 눈금의 정규화 파워다.
    """
    assert framing == "inc", "NPU STFT 는 ①′ 프레임 정의에서만 정의한다"
    wq, s = _stft_basis_q(bits)
    x = np.asarray(w_int8, dtype=np.int64)
    idx = (OFF_INC + np.arange(N_FFT)[None, :]
           + (np.arange(N_FRAMES) * HOP_INC)[:, None])
    # float64 로 곱한다 — |누산값| <= 512*128*127 = 8.3e6 이라 **정확한 정수**이고
    # (2^53 한참 아래), int64 행렬곱(BLAS 미사용)보다 수십 배 빠르다.
    fr = x[idx].astype(np.float64)                    # (frames, n_fft)
    re = fr @ wq[0].T.astype(np.float64)              # (frames, bins) 정수값
    im = fr @ wq[1].T.astype(np.float64)
    p = re * re + im * im
    return p / (s * s * 128.0 * 128.0) / _REF


def comp_span_db(alpha):
    """거듭제곱 압축이 255단계에 담는 dB 범위."""
    return 10.0 / alpha * np.log10(255.0)


def power_to_int8(mel_p, alpha, top_db=COMP_TOP_DB):
    """정규화 멜 파워 -> int8, 거듭제곱 압축. 고정 아핀이며 데이터 무관이다.

    q = clip(round((p/top)**alpha * 255)) - 128.
    `top_db` 이상은 +127 로 포화하고, 1 LSB 아래는 -128 이다. 포화시키는 것이
    중요하다 — 랩어라운드를 전처리에서 반복하지 않는다 (CLAUDE.md 7장).
    """
    top = 10.0 ** (top_db / 10.0)
    c = np.power(np.maximum(np.asarray(mel_p, dtype=np.float64), 0.0) / top,
                 alpha)
    return np.clip(np.round(c * 255.0) - 128.0, -128.0, 127.0).astype(np.int8)


def mel_int8(w_int8, scheme="log"):
    """int8 파형 (16384,) -> int8 멜 (N_MELS, N_FRAMES). 모델 입력이다.

    `scheme` 은 COMP_SCHEMES 의 키다. "log" 는 `log_mel_int8` 과 **같은 값**을
    낸다 (같은 경로를 탄다 — 두 구현이 갈리면 비교가 오염된다).
    """
    alpha, top_db, span_db = COMP_SCHEMES[scheme]
    framing = SCHEME_FRAMING.get(scheme, "reflect")
    bits = SCHEME_STFT_BITS.get(scheme)
    if bits is not None:                              # NPU 정수 STFT 시뮬레이션
        mel = stft_power_q(w_int8, bits, framing) @ _FB.T
        return db_to_int8(10.0 * np.log10(mel.T + EPS), top_db, span_db)
    if alpha is None:
        return db_to_int8(log_mel_db(w_int8, framing), top_db, span_db)
    x = np.asarray(w_int8, dtype=np.float64) / 128.0
    mel_p = (stft_power(x, framing) @ _FB.T).T        # (mels, frames)
    return power_to_int8(mel_p, alpha, top_db)
