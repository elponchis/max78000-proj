"""구성 ②′ — **NPU 에너지 적분 필터뱅크** → CPU 정수 로그 → NPU 2D 분류기 (후보 가 의 모델).

사전 등록: `docs/results/fe-design.md` (2026-10-17). 왜 이 설계인가는 그 문서 1절.

## 지금까지의 NPU 프론트엔드가 공통으로 빠뜨린 것

D-1 · (a) · (c) 는 전부 "접힌 입력에 k=1 conv = 128탭 FIR 을 **프레임마다 한
샘플 위치에서** 평가 → |·|" 였다. 대역 통과 신호의 **순간값 하나**의 크기라
위상에 따라 요동한다. CPU log-mel 은 512샘플 창의 파워를 대역 안에서 **적분**
한다. 이 모델은 그 적분을 NPU 가 지원하는 연산(Conv1d, |·|, 합)으로 넣는다.

## 파이프라인 (학습 그래프 = 펌웨어 경로)

    s[0..16383]  (④ 와 같은 int8 창, 1.024 s)
    ── 접기 64 ──  X[r][p] = s[64p + r],  r = 0..63, p = 0..255
    ── NPU p1 ──   Conv1d(64 → 64, k=8, 활성화 없음, wide int32)
                   y[o][p] = Σ_{j<8} Σ_r W[o][r][j] X[r][p+j]
                           = Σ_{t<512} w[o][t] s[64p + t]      (512탭 FIR = 32 ms, 홉 64 = 4 ms)
                   유효 위치 p = 0..248
    ── CPU ──      mag[b][p] = |y[2b][p]| + |y[2b+1][p]|        (cos·sin 쌍 → 밴드 크기, L1)
                   E[b][f]   = Σ_{p=16f}^{16f+7} mag[b][p]      (프레임 f = 0..15, 홉 1024 = 64 ms,
                                                                 적분 구간 960 샘플 = 60 ms)
                   q[b][f]   = to_int8(log2_q8(round(E·2^SCALE_SH)), LO_Q8, SPAN_Q8)
                               ← tools/log2_q8.py == firmware/common/log2_q8.c (비트 동일)
    ── NPU p2 ──   가 의 2D CNN (`ai85safesoundmelnet_p4`, 입력 1×32×16) → 로짓 5

판단 1회(hop 250 ms ≈ 4 프레임)에 펌웨어는 **새 4,096 샘플만** p1 에 넣고
(접기 (64, 64) → 유효 위치 57 → 프레임 4개), 32×16 버퍼를 밀어 p2 를 돌린다.
학습은 창 전체(16 프레임)를 한 번에 계산하지만 conv 는 위치 불변이고 프레임
경계가 1024 의 배수라 같은 값이다. 판단 주기는 이 구성에서 **256 ms** 다
(4,096 샘플 = 1.024 s 창의 1/4; 다른 구성의 250 ms 와 2.4 % 차이).

## 왜 로그는 CPU 인가

`output_width: 32` 는 활성화와 함께 쓸 수 없다 (ai8x-synthesis, (c) 1단계 확인).
int8 로 내리고 NPU 에서 풀링하면 로그 **전에** 동적범위를 잃는다. 32bit 로
받아 CPU 가 |·|·합·로그를 하면 정밀도가 보존되고, 값이 32×4 = 128개뿐이라
비용은 수십 μs 다 ((c) 설계 4.1절의 로그 단가로 환산 — 측정 전이므로 추정).

## 로그단 상수 — 펌웨어와 공유

(c) 와 같은 논리 (`ai85net-safesound-cstage-train.py` 모듈 설명):
배율 차이는 로그 영역의 평행이동이라 LO_Q8 이 흡수한다. 지킬 것은 학습과
펌웨어가 **같은 (SCALE_SH, LO_Q8, SPAN_Q8)** 을 쓰는 것뿐이다.
SPAN_Q8 = 2976 (70.0 dB) 은 ① 의 SPAN_DB 와 같은 폭으로 **고정**하고, LO_Q8 은
`tools/init_fe_fb.py` 가 **학습셋** 표본의 E 분포 p99 에 창의 위끝을 맞춰
정한다 (시험셋을 보지 않는다). 경계 파일 이름에 세 상수가 들어가므로
모델 상수와 경계가 어긋나면 파일을 못 찾고 멈춘다 (조용히 틀리지 않는다).
"""
import importlib.util
import os

import numpy as np
import torch
import torch.nn as nn

import ai8x

# ── 구조 상수 ────────────────────────────────────────────────────────
FOLD = 64                 # 접기: 위치당 샘플 수 = p1 입력 채널 수
NPOS = 16384 // FOLD      # 256 위치 (학습 창 전체)
KERNEL = 8                # p1 커널 → FIR 탭 = KERNEL·FOLD = 512 (32 ms)
TAPS = KERNEL * FOLD
NBAND = 32                # 멜 밴드 수 = 가 의 멜 수
NCH = 2 * NBAND           # cos·sin 쌍 → p1 출력 채널 64
NFRAME = 16               # 프레임 수 = 가 와 같다 (32 × 16 입력)
POS_PER_FRAME = 16        # 프레임 홉 = 16 위치 = 1,024 샘플 = 64 ms
POOL_POS = 8              # 프레임당 적분 위치 수 (960 샘플 = 60 ms)
NVALID = NPOS - KERNEL + 1    # 249 — 마지막 프레임의 마지막 위치 16·15+7 = 247 ≤ 248 이라 패딩이 없다
assert POS_PER_FRAME * (NFRAME - 1) + POOL_POS <= NVALID

# ── 로그단 상수 — **펌웨어와 공유한다** (tools/init_fe_fb.py 가 정한 값) ──
SCALE_SH = 14             # E → 정수 스케일 (2 의 거듭제곱)
LO_Q8 = 1279              # 창 아래끝, Q8 log2 (init_fe_fb.py 2026-10-17: 학습셋 400창 E 분포 p99 16.62 − 70 dB → v≈32, 정수 해상도 지점(v≈30) 위)
SPAN_Q8 = 2976            # 창 폭, Q8 log2 (= 70.0 dB, ① 과 같은 폭 — 고정)
E_MIN = 2.0 ** -SCALE_SH  # STE 의 1/E 하한 (= 1 LSB)

_LN2 = float(np.log(2.0))
_REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))   # 심볼릭 링크를 푼다 ((c) 와 같은 이유)
_BOUNDS_NPY = os.path.join(_REPO, "data", "synth", f"fe_bounds_sh{SCALE_SH}_lo{LO_Q8}_span{SPAN_Q8}.npy")
_BOUNDS = None


def _bounds():
    """계단 경계 255개 (tools/gen_cstage_bounds.py --out …). import 시점에 읽지 않는다 ((c) 의 사고)."""
    global _BOUNDS                                  # noqa: PLW0603
    if _BOUNDS is None:
        if not os.path.isfile(_BOUNDS_NPY):
            raise FileNotFoundError(
                f"{_BOUNDS_NPY} 가 없다. `python3 tools/gen_cstage_bounds.py --scale-sh {SCALE_SH} "
                f"--lo-q8 {LO_Q8} --span-q8 {SPAN_Q8} --out {_BOUNDS_NPY}` 를 먼저 돌릴 것")
        b = torch.from_numpy(np.load(_BOUNDS_NPY))
        assert b.numel() == 255, "경계는 255개여야 한다 (레벨 256개)"
        _BOUNDS = b
    return _BOUNDS


class _IntLog(torch.autograd.Function):
    """E (≥ 0) → 정수 로그 → int8. 순전파는 펌웨어와 같고 (bucketize == round → CLZ+LUT → 포화), 역전파는 STE."""

    @staticmethod
    def forward(ctx, e):
        ctx.save_for_backward(e)
        q = torch.bucketize(e.detach().double(), _bounds().to(e.device)) - 128
        return q.to(e.dtype)

    @staticmethod
    def backward(ctx, g):
        (e,) = ctx.saved_tensors
        # d/dE [ (log2 E − lo) · 255/span ] = 255 / (span/256 · ln2 · E)
        denom = (SPAN_Q8 / 256.0) * _LN2 * e.clamp(min=E_MIN)
        return g * (255.0 / denom)


# ── 멜 필터뱅크 초기값 (모델이 스스로 만든다 — 체크포인트 없이 시드마다 분류기만 달라진다) ──
def _melfeat():
    spec = importlib.util.spec_from_file_location("melfeat_fe", os.path.join(_REPO, "datasets", "melfeat.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def filterbank_init(n_band=NBAND, taps=TAPS, fold=FOLD, kernel=KERNEL):
    """가 의 멜 32밴드와 같은 중심·폭의 **밴드폭 맞춤 Gabor** cos/sin 쌍 → (64, fold, kernel) conv 가중치.

    · 중심 주파수: melfeat.mel_filterbank 와 같은 식 (FMIN 20 ~ FMAX 8000, 34 경계의 안쪽 32)
    · 창 길이 L_b = clip(1.44·sr / 밴드폭, 64, taps) — Hann 주엽 폭이 멜 삼각 폭과 맞도록. 512탭 안에 가운데 정렬.
    · 각 필터 L2 노름 1 (init_filterbank.py 와 같은 이유: 출력 크기가 입력 RMS 자릿수).
    · 탭 배치 t = j·fold + r → reshape(n, kernel, fold).transpose(0, 2, 1)  (init_filterbank.to_conv_weight 와 같다).
    """
    MF = _melfeat()
    edges = MF.mel_to_hz(np.linspace(MF.hz_to_mel(MF.FMIN), MF.hz_to_mel(MF.FMAX), n_band + 2))
    centers, lengths = edges[1:-1], []
    rows = []
    for b, f in enumerate(centers):
        bw = max(float(edges[b + 2] - edges[b]) / 2.0, 1.0)          # 삼각 필터의 반폭 ≈ −3 dB 폭
        L = int(np.clip(round(1.44 * MF.SR / bw), fold, taps))
        lengths.append(L)
        w = np.zeros(taps)
        off = (taps - L) // 2
        w[off:off + L] = np.hanning(L + 1)[:L]
        t = (np.arange(taps) - (taps - 1) / 2.0) / MF.SR
        for v in (w * np.cos(2 * np.pi * f * t), w * np.sin(2 * np.pi * f * t)):
            rows.append(v / (np.linalg.norm(v) or 1.0))
    W = np.array(rows, dtype=np.float32)                                 # (64, 512), 2b = cos, 2b+1 = sin
    Wc = W.reshape(NCH, kernel, fold).transpose(0, 2, 1)                 # (64, fold, kernel)
    return np.ascontiguousarray(Wc), centers, np.array(lengths)


def _melnet():
    spec = importlib.util.spec_from_file_location("ai85net_safesound_mel_fe", os.path.join(_REPO, "models", "ai85net-safesound-mel.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class AI85SafeSoundFE(nn.Module):
    """②′ 학습 모델 — p1 (NPU) → CPU 로그단 → p2 (NPU, 가 의 2D CNN) 한 덩어리. 역전파는 끝까지 흐른다."""

    def __init__(
            self,
            num_classes=5,
            num_channels=128,       # pylint: disable=unused-argument  (데이터셋이 (128,128) 로 준다)
            dimensions=(128, 128),  # pylint: disable=unused-argument
            bias=False,
            fb_init=True,
            **kwargs
    ):
        super().__init__()
        # p1 — 활성화 없는 wide 층 (디바이스 int32). bias 없음 (FIR).
        self.fe = ai8x.Conv1d(FOLD, NCH, KERNEL, stride=1, padding=0, bias=False, wide=True, **kwargs)
        if fb_init:
            Wc, _, _ = filterbank_init()
            with torch.no_grad():
                assert tuple(self.fe.op.weight.shape) == Wc.shape, (self.fe.op.weight.shape, Wc.shape)
                self.fe.op.weight.copy_(torch.from_numpy(Wc))
        # p2 — 가 와 같은 모델·같은 입력 모양 (1, 32, 16), 풀링 4단
        self.cls = _melnet().AI85SafeSoundMelNet(num_classes=num_classes, num_channels=1,
                                                 dimensions=(NBAND, NFRAME), bias=bias, pool6=False, **kwargs)
        idx = (torch.arange(NFRAME) * POS_PER_FRAME)[:, None] + torch.arange(POOL_POS)[None, :]   # (16, 8)
        self.register_buffer("_idx", idx, persistent=False)

    def frontend(self, x):
        """(B, 128, 128) ④ 접기 → (B, 32, 16) 적분 밴드 에너지 E (로그 전, float). KAT·통계용으로 분리."""
        B = x.size(0)
        s = x.transpose(1, 2).reshape(B, -1)                 # x[r][c] = s[128c + r] → 샘플 순서
        x2 = s.reshape(B, NPOS, FOLD).transpose(1, 2)        # X[r][p] = s[64p + r]
        y = self.fe(x2)                                      # (B, 64, 249) wide
        mag = y.abs()
        mag = mag[:, 0::2] + mag[:, 1::2]                    # (B, 32, 249)
        return mag[:, :, self._idx].sum(-1)                  # (B, 32, 16)

    def forward(self, x):  # pylint: disable=arguments-differ
        e = self.frontend(x)
        # ── 여기가 CPU 다: 정수 로그 → int8 → p2 입력 규약 [-1, 1) ──
        q = _IntLog.apply(e) / 128.0
        return self.cls(q.unsqueeze(1))                      # (B, 1, 32, 16)


class AI85SafeSoundFEP1(nn.Module):
    """②′ 1패스 합성용 — p1 한 층. 입력 (64, L) int8, 출력 (64, L−7) int32. L = 64 (판단 1회, 4,096 샘플)."""

    def __init__(self, num_classes=5, num_channels=FOLD, dimensions=(64, 1), bias=False, **kwargs):  # pylint: disable=unused-argument
        super().__init__()
        self.fe = ai8x.Conv1d(FOLD, NCH, KERNEL, stride=1, padding=0, bias=False, wide=True, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        return self.fe(x)


def ai85safesoundfe(pretrained=False, **kwargs):
    """②′ 학습 모델 (멜 Gabor 초기값 내장)."""
    assert not pretrained
    return AI85SafeSoundFE(**kwargs)


def ai85safesoundfe_p1(pretrained=False, **kwargs):
    """②′ 1패스 — 합성 확인용."""
    assert not pretrained
    return AI85SafeSoundFEP1(**kwargs)


# ⚠️ 이 목록에 넣지 않으면 `train.py --model …` 이 받지 않는다.
models = [
    {'name': 'ai85safesoundfe', 'min_input': 1, 'dim': 1},      # 데이터셋 입력이 ④ 와 같은 (128, 128) 1D 접기다
    {'name': 'ai85safesoundfe_p1', 'min_input': 1, 'dim': 1},
]
