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

## 변형 (2026-10-17 저녁, `docs/results/fe-curve-design.md`) — NPU 전처리 쪽 곡선

②′ 한 점으로는 "NPU 전처리 곡선" 이 없다. 대역 수와 프레임 수를 늘린 변형 셋을
같은 설계·같은 학습 설정으로 학습해 CPU log-mel 곡선과 겹쳐 본다.

    fe      32 대역 × 16 프레임 (홉 64 ms)   ②′ 그대로
    b64     64 대역 × 16 프레임 (홉 64 ms)   p1 Conv1d(64→128), 분류기 입력 1×64×16
    f31     32 대역 × 31 프레임 (홉 32 ms)   프레임 홉 8 위치, 적분 8 위치 (60 ms 그대로), 입력 1×32×31
    b64f31  64 대역 × 31 프레임 (홉 32 ms)   둘 다

접기·탭·커널·적분 길이·로그단(SCALE_SH, SPAN_Q8)은 전부 같고, LO_Q8 만 변형마다
학습셋 E 분포로 다시 정한다 (`tools/init_fe_fb.py --variant`). 분류기는 가 의 2D CNN
(풀링 4단) 에 입력 모양만 바꾼 것이라 FC 입력이 256 (32 대역) 또는 512 (64 대역) 이다.
31 프레임인 이유: 유효 위치 249 안에서 8 위치 홉·8 위치 적분이면 8·30+8 = 248 이 한계다
(32 프레임은 패딩이 필요하다). 펌웨어는 판단마다 8 프레임을 새로 계산하므로 p1 입력을
청크 64 위치 + 직전 7 위치(448 샘플) 로 넣어야 한다 (fe 는 57 유효 위치로 충분했다).

## 후속 연구 (2026-10-24) — p1 을 8비트 + Abs 로 내리고 cos·sin·8위치 합을 NPU 에 두는 변형 (`p1_bits8=True`)

`docs/results/npu-readback-design.md` 7절. PC 시뮬레이션(6절)은 **학습된 ②′ 가중치를 그대로** 8비트로
내리면 포화 5~22 %·0 이 11~17 % 라 특징이 상해 F1 이 −0.040(p2 미세조정) 떨어진다고 했다.
이 변형은 8비트 단계를 **학습 그래프에 넣어** p1 이 그 창에 맞춰 게인·필터를 배우게 한다 (전체 QAT 재학습).

    ── NPU 층 A ──  Conv1d(64 → 2n, k=8), **8비트 출력, 활성화 Abs**  (디바이스: floor(y_int·2^(s−7) + 0.5) → 포화 ±128 → |·| → 127 상한)
                    a[o][p] = |y[o][p]| 을 8비트로 — s = 출력 시프트 (합성기가 가중치 범위로 정한다). 학습 그래프는 같은 순서로
                    ai8x 8비트 층(활성화 없음: 반올림·포화) 뒤 frontend() 에서 |·|·127 상한을 건다 (`tools/fe8_check.py` 가 정수 경로와 대조)
    ── NPU 층 B ──  Conv1d(2n → n, k=8), 가중치 64/128 = 0.5 고정, wide int32, 활성화 없음  (정수 정확: E'_int = 64·Σ a_int)
                    E'[b][f] = 0.5 · Σ_{p∈프레임 f, 8위치} (a[2b][p] + a[2b+1][p])         ← 학습 그래프는 이 합을 torch 로 직접 계산한다
    ── CPU ──       q = to_int8(log2_q8(round(E'·2^14)), LO_Q8 − 256, SPAN_Q8)        읽기 int32 n×F' 개 (②′ 2,048 → 128 워드)
    ── NPU p2 ──    ②′ 와 같다

  · 바뀐 것은 p1 층의 출력 폭·활성화와 로그단 LO 두 가지뿐이다. 접기·탭·커널·적분·SPAN·p2·학습 설정은 ②′ 와 같다.
  · LO − 256 인 이유: 포화·반올림이 없으면 E' = E/2 (|y| 합에 0.5 를 곱했다) 이므로 로그 창을 **물리적으로 같은 자리**에
    두려면 log2 로 −1 (= Q8 −256) 만큼 내린다. 출력 시프트 s 는 여기에 끼지 않는다 — 학습 그래프의 a = floor(|y_float|·128+0.5)/128
    은 s 와 무관하고, 디바이스는 y_int·2^(s−14) = y_float 가 되도록 s 를 고르기 때문이다 (float 가중치의 크기가 곧 게인이다).
  · QAT 시작(에폭 60) 전에는 ai8x 가 양자화·포화를 걸지 않으므로 그 전까지는 ②′ 와 같은 계산이다 (상수 1/2 는 LO 가 흡수).
    8비트 창은 60 에폭부터 보인다 — 다른 모든 층과 같은 규칙이다.
  · Abs 와 반올림의 순서는 **디바이스 순서(반올림 → 포화 → |·|)** 를 따른다 (ai8x-synthesis `izer/simulate.py` 가 그 순서고, 보드 KAT 가
    그 시뮬레이션과 일치해 왔다). ai8x-training 의 activation="Abs" 는 |·| 를 반올림 앞에 걸어 음수 v 가 정확히 k+0.5 인 자리에서
    1 LSB 다르고, 조용한 대역(합이 0~2 LSB)에서는 그 1 LSB 가 로그 뒤 q 를 최대 44 단계 움직인다 (2026-10-24 검산에서 확인) —
    그래서 쓰지 않았다. `tools/fe8_check.py` 가 학습 그래프와 정수 경로(`npu_readback_sim.Front.A_sum`)의 비트 일치를 확인한다.

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

try:
    from torch.fx._symbolic_trace import is_fx_tracing as _is_fx_tracing   # torch 2.3 — ai8x.apply_scales 의 symbolic_trace 감지
except ImportError:  # pragma: no cover
    def _is_fx_tracing():
        return False

# ── 구조 상수 (모든 변형 공통) ────────────────────────────────────────
FOLD = 64                 # 접기: 위치당 샘플 수 = p1 입력 채널 수
NPOS = 16384 // FOLD      # 256 위치 (학습 창 전체)
KERNEL = 8                # p1 커널 → FIR 탭 = KERNEL·FOLD = 512 (32 ms)
TAPS = KERNEL * FOLD
NVALID = NPOS - KERNEL + 1    # 249 유효 위치

# ── 로그단 상수 — **펌웨어와 공유한다** ──────────────────────────────
SCALE_SH = 14             # E → 정수 스케일 (2 의 거듭제곱)
SPAN_Q8 = 2976            # 창 폭, Q8 log2 (= 70.0 dB, ① 과 같은 폭 — 모든 변형에서 고정)
E_MIN = 2.0 ** -SCALE_SH  # STE 의 1/E 하한 (= 1 LSB)
LO8_SHIFT_Q8 = -256       # p1_bits8: E' = E/2 라 로그 창을 log2 −1 만큼 내린다 (모듈 설명 "후속 연구" 참조)
E8_SCALE = 0.5            # p1_bits8: 층 B 의 고정 가중치 (디바이스 int8 64/128). E'_int32 = 64·Σ a_int = 2^14·E'_float

# ── 변형 표 (docs/results/fe-curve-design.md). lo_q8 은 tools/init_fe_fb.py --variant 가 정한 값 ──
VARIANTS = {
    "fe":     dict(n_band=32, n_frame=16, pos_per_frame=16, pool_pos=8, lo_q8=1279),   # ②′ (init_fe_fb 2026-10-17: p99 16.62 − 70 dB)
    "b64":    dict(n_band=64, n_frame=16, pos_per_frame=16, pool_pos=8, lo_q8=1627),   # init_fe_fb 2026-10-17 저녁: p99 17.98 − 70 dB
    "f31":    dict(n_band=32, n_frame=31, pos_per_frame=8,  pool_pos=8, lo_q8=1279),   # fe 와 같은 E 분포 (p99 16.62)
    "b64f31": dict(n_band=64, n_frame=31, pos_per_frame=8,  pool_pos=8, lo_q8=1627),   # b64 와 같은 E 분포 (p99 17.98)
}
for _k, _v in VARIANTS.items():
    # 마지막 프레임의 마지막 위치가 유효 위치 안에 있어야 패딩이 없다 (fe: 16·15+7 = 247, f31: 8·30+7 = 247)
    assert _v["pos_per_frame"] * (_v["n_frame"] - 1) + _v["pool_pos"] <= NVALID, _k

# ②′ 기본 상수 — 기존 도구(fe_board_prep.py 등)가 모듈 상수로 읽는다. VARIANTS["fe"] 와 같다.
NBAND = VARIANTS["fe"]["n_band"]
NCH = 2 * NBAND
NFRAME = VARIANTS["fe"]["n_frame"]
POS_PER_FRAME = VARIANTS["fe"]["pos_per_frame"]
POOL_POS = VARIANTS["fe"]["pool_pos"]
LO_Q8 = VARIANTS["fe"]["lo_q8"]

_LN2 = float(np.log(2.0))
_REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))   # 심볼릭 링크를 푼다 ((c) 와 같은 이유)
_BOUNDS = {}


def bounds_path(lo_q8):
    return os.path.join(_REPO, "data", "synth", f"fe_bounds_sh{SCALE_SH}_lo{lo_q8}_span{SPAN_Q8}.npy")


def _bounds(lo_q8):
    """계단 경계 255개 (tools/gen_cstage_bounds.py --out …). import 시점에 읽지 않는다 ((c) 의 사고). 변형(LO)마다 따로."""
    if lo_q8 is None:
        raise ValueError("이 변형의 LO_Q8 이 아직 없다 — `tools/init_fe_fb.py --variant <이름>` 으로 정하고 VARIANTS 에 적을 것")
    if lo_q8 not in _BOUNDS:
        f = bounds_path(lo_q8)
        if not os.path.isfile(f):
            raise FileNotFoundError(
                f"{f} 가 없다. `python3 tools/gen_cstage_bounds.py --scale-sh {SCALE_SH} "
                f"--lo-q8 {lo_q8} --span-q8 {SPAN_Q8} --out {f}` 를 먼저 돌릴 것")
        b = torch.from_numpy(np.load(f))
        assert b.numel() == 255, "경계는 255개여야 한다 (레벨 256개)"
        _BOUNDS[lo_q8] = b
    return _BOUNDS[lo_q8]


class _IntLog(torch.autograd.Function):
    """E (≥ 0) → 정수 로그 → int8. 순전파는 펌웨어와 같고 (bucketize == round → CLZ+LUT → 포화), 역전파는 STE."""

    @staticmethod
    def forward(ctx, e, bounds):
        ctx.save_for_backward(e)
        q = torch.bucketize(e.detach().double(), bounds.to(e.device)) - 128
        return q.to(e.dtype)

    @staticmethod
    def backward(ctx, g):
        (e,) = ctx.saved_tensors
        # d/dE [ (log2 E − lo) · 255/span ] = 255 / (span/256 · ln2 · E)
        denom = (SPAN_Q8 / 256.0) * _LN2 * e.clamp(min=E_MIN)
        return g * (255.0 / denom), None


# ── 멜 필터뱅크 초기값 (모델이 스스로 만든다 — 체크포인트 없이 시드마다 분류기만 달라진다) ──
def _melfeat():
    spec = importlib.util.spec_from_file_location("melfeat_fe", os.path.join(_REPO, "datasets", "melfeat.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def filterbank_init(n_band=NBAND, taps=TAPS, fold=FOLD, kernel=KERNEL):
    """멜 n_band 밴드와 같은 중심·폭의 **밴드폭 맞춤 Gabor** cos/sin 쌍 → (2·n_band, fold, kernel) conv 가중치.

    · 중심 주파수: melfeat.mel_filterbank 와 같은 식 (FMIN 20 ~ FMAX 8000, n_band+2 경계의 안쪽 n_band)
    · 창 길이 L_b = clip(1.44·sr / 밴드폭, 64, taps) — Hann 주엽 폭이 멜 삼각 폭과 맞도록. 512탭 안에 가운데 정렬.
      64 밴드에서는 저역의 요구 길이가 512 를 넘어 잘린다 (분해능 한계 — 학습이 보정할 수 있는지는 결과가 말한다).
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
    W = np.array(rows, dtype=np.float32)                                 # (2n, 512), 2b = cos, 2b+1 = sin
    Wc = W.reshape(2 * n_band, kernel, fold).transpose(0, 2, 1)          # (2n, fold, kernel)
    return np.ascontiguousarray(Wc), centers, np.array(lengths)


def _melnet():
    spec = importlib.util.spec_from_file_location("ai85net_safesound_mel_fe", os.path.join(_REPO, "models", "ai85net-safesound-mel.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class AI85SafeSoundFE(nn.Module):
    """②′ 학습 모델 — p1 (NPU) → CPU 로그단 → p2 (NPU, 가 의 2D CNN) 한 덩어리. 역전파는 끝까지 흐른다.

    `variant` 가 VARIANTS 의 한 행을 고른다. 기본 "fe" 가 ②′ 다.
    """

    def __init__(
            self,
            num_classes=5,
            num_channels=128,       # pylint: disable=unused-argument  (데이터셋이 (128,128) 로 준다)
            dimensions=(128, 128),  # pylint: disable=unused-argument
            bias=False,
            fb_init=True,
            variant="fe",
            p1_bits8=False,
            qat_link_cut=False,
            **kwargs
    ):
        super().__init__()
        spec = VARIANTS[variant]
        self.variant = variant
        self.p1_bits8 = p1_bits8
        self.qat_link_cut = qat_link_cut
        self.n_band, self.n_frame = spec["n_band"], spec["n_frame"]
        self.pos_per_frame, self.pool_pos = spec["pos_per_frame"], spec["pool_pos"]
        self.lo_q8 = spec["lo_q8"] + (LO8_SHIFT_Q8 if p1_bits8 else 0)
        self.e_scale = E8_SCALE if p1_bits8 else 1.0
        self.n_ch = 2 * self.n_band
        if p1_bits8:
            # 후속 연구: p1 을 8비트로 (ai8x 가 출력 시프트·반올림·포화를 건다). |·| 는 frontend() 가 **반올림·포화 뒤에** 건다 —
            # 디바이스(izer/simulate.py: floor(v+0.5) → 포화 → |·| → 127 상한)와 같은 순서다. ai8x 의 activation="Abs" 는 |·| 를
            # 반올림 **앞에** 걸어 음수 v 가 정확히 k+0.5 인 자리에서 1 LSB 다르다 (조용한 대역에서 0↔1 이 뒤집혀 q 가 크게 움직인다).
            # 합성 YAML 은 그대로 activate: Abs 다 (디바이스가 |·| 를 맡는다). 합(층 B)은 frontend() 가 torch 로 계산한다.
            self.fe = ai8x.Conv1d(FOLD, self.n_ch, KERNEL, stride=1, padding=0, bias=False, wide=False, activation=None, **kwargs)
        else:
            # p1 — 활성화 없는 wide 층 (디바이스 int32). bias 없음 (FIR).
            self.fe = ai8x.Conv1d(FOLD, self.n_ch, KERNEL, stride=1, padding=0, bias=False, wide=True, **kwargs)
        if fb_init:
            Wc, _, _ = filterbank_init(n_band=self.n_band)
            with torch.no_grad():
                assert tuple(self.fe.op.weight.shape) == Wc.shape, (self.fe.op.weight.shape, Wc.shape)
                self.fe.op.weight.copy_(torch.from_numpy(Wc))
        # p2 — 가 와 같은 모델 (풀링 4단), 입력 모양 (1, n_band, n_frame)
        self.cls = _melnet().AI85SafeSoundMelNet(num_classes=num_classes, num_channels=1,
                                                 dimensions=(self.n_band, self.n_frame), bias=bias, pool6=False, **kwargs)
        idx = (torch.arange(self.n_frame) * self.pos_per_frame)[:, None] + torch.arange(self.pool_pos)[None, :]   # (n_frame, pool_pos)
        self.register_buffer("_idx", idx, persistent=False)

    def frontend(self, x):
        """(B, 128, 128) ④ 접기 → (B, n_band, n_frame) 적분 밴드 에너지 E (로그 전, float). KAT·통계용으로 분리."""
        B = x.size(0)
        s = x.transpose(1, 2).reshape(B, -1)                 # x[r][c] = s[128c + r] → 샘플 순서
        x2 = s.reshape(B, NPOS, FOLD).transpose(1, 2)        # X[r][p] = s[64p + r]
        y = self.fe(x2)                                      # (B, 2n, 249) wide — p1_bits8 이면 8비트로 반올림·포화된 값 (QAT 뒤)
        mag = y.abs()                                        # p1_bits8: 디바이스의 Abs (반올림·포화 뒤). |−128| = 128 은 127 로 (아래)
        if self.p1_bits8:
            # QAT(포화 켜짐) 뒤에만 127/128 상한. 파이썬 if 로 파라미터를 읽으면 ai8x.pre_qat 의 torch.fx 추적이 깨지므로 텐서 연산으로.
            cap = torch.where(self.fe.clamp_activation, torch.full((1,), 127.0 / 128.0), torch.full((1,), float("inf")))
            mag = torch.minimum(mag, cap.to(mag.dtype))
        mag = mag[:, 0::2] + mag[:, 1::2]                    # (B, n, 249)
        return mag[:, :, self._idx].sum(-1) * self.e_scale   # (B, n_band, n_frame); p1_bits8: 층 B 의 0.5 (정수 정확)

    def intlog(self, e):
        """E → int8 (펌웨어와 비트 동일한 정수 로그). 변형의 LO_Q8 경계를 쓴다."""
        return _IntLog.apply(e, _bounds(self.lo_q8))

    def forward(self, x):  # pylint: disable=arguments-differ
        e = self.frontend(x)
        # ── 여기가 CPU 다: 정수 로그 → int8 → p2 입력 규약 [-1, 1) ──
        q = self.intlog(e) / 128.0
        if self.qat_link_cut and _is_fx_tracing():
            # ai8x.pre_qat → apply_scales 의 torch.fx 추적 **때만**: p2 입력을 상수로 바꿔 fe → cls.conv1 의 '이전 층' 연결을 끊는다.
            # apply_scales 는 conv 노드에서 입력을 거슬러 올라가 만나는 첫 conv(fe)를 이전 층으로 보고 conv1 의 문턱에서 fe 의
            # 문턱(t_fe)을 뺀다 — fe 출력이 2^(−t_fe) 로 줄어 conv1 입력도 줄었다는 가정인데, 사이의 정수 로그가 스케일을 지우므로
            # 여기서는 틀린 보상이다. ②′ 원 학습(chain32·33)은 이 연결을 그대로 두어 conv1 문턱이 −3 (출력 ×8, 포화 14 %) 이었다
            # (docs/results/npu-readback-design.md 7절). 실제 순전파·역전파·평가에는 이 분기가 들어가지 않는다.
            q = torch.zeros(1, self.n_band, self.n_frame)   # dtype 은 고정 (추적 중 q.dtype 은 Proxy 라 못 쓴다)
        return self.cls(q.unsqueeze(1))                      # (B, 1, n_band, n_frame)


class AI85SafeSoundFEP1(nn.Module):
    """②′ 1패스 합성용 — p1 한 층. 입력 (64, L) int8, 출력 (2·n_band, L−7) int32. L = 64 (fe, 판단 1회 4,096 샘플) / 71 (f31)."""

    def __init__(self, num_classes=5, num_channels=FOLD, dimensions=(64, 1), bias=False, variant="fe", **kwargs):  # pylint: disable=unused-argument
        super().__init__()
        self.fe = ai8x.Conv1d(FOLD, 2 * VARIANTS[variant]["n_band"], KERNEL, stride=1, padding=0, bias=False, wide=True, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        return self.fe(x)


def ai85safesoundfe(pretrained=False, **kwargs):
    """②′ 학습 모델 (멜 Gabor 초기값 내장) — 32 대역 × 16 프레임."""
    assert not pretrained
    return AI85SafeSoundFE(variant="fe", **kwargs)


def ai85safesoundfe_b64(pretrained=False, **kwargs):
    """변형 b64 — 64 대역 × 16 프레임."""
    assert not pretrained
    return AI85SafeSoundFE(variant="b64", **kwargs)


def ai85safesoundfe_f31(pretrained=False, **kwargs):
    """변형 f31 — 32 대역 × 31 프레임 (홉 32 ms)."""
    assert not pretrained
    return AI85SafeSoundFE(variant="f31", **kwargs)


def ai85safesoundfe_b64f31(pretrained=False, **kwargs):
    """변형 b64f31 — 64 대역 × 31 프레임."""
    assert not pretrained
    return AI85SafeSoundFE(variant="b64f31", **kwargs)


def ai85safesoundfe_v2(pretrained=False, **kwargs):
    """후속 연구 대조군 — ②′ 그대로이되 QAT 문턱 연결(fe → cls.conv1)만 끊는다. fe8 과의 차이가 8비트 단계만 되게 하는 기준선."""
    assert not pretrained
    return AI85SafeSoundFE(variant="fe", qat_link_cut=True, **kwargs)


def ai85safesoundfe_v2_b64f31(pretrained=False, **kwargs):
    """후속 연구 대조군 — b64f31 에 같은 처리."""
    assert not pretrained
    return AI85SafeSoundFE(variant="b64f31", qat_link_cut=True, **kwargs)


def ai85safesoundfe_v2_b64(pretrained=False, **kwargs):
    """변형 b64 의 v2 (QAT 문턱 연결 끊음) — chain35 (H-1 유의 분기, 2026-10-24)."""
    assert not pretrained
    return AI85SafeSoundFE(variant="b64", qat_link_cut=True, **kwargs)


def ai85safesoundfe_v2_f31(pretrained=False, **kwargs):
    """변형 f31 의 v2 (QAT 문턱 연결 끊음) — chain35."""
    assert not pretrained
    return AI85SafeSoundFE(variant="f31", qat_link_cut=True, **kwargs)


def ai85safesoundfe8(pretrained=False, **kwargs):
    """후속 연구 — ②′ 에 p1 8비트 + Abs + NPU 합산 (안 A-sum) 을 넣어 전체 QAT 재학습. 32 대역 × 16 프레임. 문턱 연결 끊음(v2 와 짝)."""
    assert not pretrained
    return AI85SafeSoundFE(variant="fe", p1_bits8=True, qat_link_cut=True, **kwargs)


def ai85safesoundfe8_b64(pretrained=False, **kwargs):
    """후속 연구 — b64 에 같은 8비트 변형."""
    assert not pretrained
    return AI85SafeSoundFE(variant="b64", p1_bits8=True, qat_link_cut=True, **kwargs)


def ai85safesoundfe8_f31(pretrained=False, **kwargs):
    """후속 연구 — f31 에 같은 8비트 변형."""
    assert not pretrained
    return AI85SafeSoundFE(variant="f31", p1_bits8=True, qat_link_cut=True, **kwargs)


def ai85safesoundfe8_b64f31(pretrained=False, **kwargs):
    """후속 연구 — b64f31 에 같은 8비트 변형 (시뮬레이션에서 A-sum 이 유일하게 G-1 을 만족한 변형)."""
    assert not pretrained
    return AI85SafeSoundFE(variant="b64f31", p1_bits8=True, qat_link_cut=True, **kwargs)


def ai85safesoundfe_p1(pretrained=False, **kwargs):
    """②′ 1패스 (64 출력 채널) — 합성 확인용."""
    assert not pretrained
    return AI85SafeSoundFEP1(variant="fe", **kwargs)


def ai85safesoundfe_p1_b64(pretrained=False, **kwargs):
    """b64·b64f31 의 1패스 (128 출력 채널) — 합성 확인용."""
    assert not pretrained
    return AI85SafeSoundFEP1(variant="b64", **kwargs)


# ⚠️ 이 목록에 넣지 않으면 `train.py --model …` 이 받지 않는다.
models = [
    {'name': 'ai85safesoundfe', 'min_input': 1, 'dim': 1},      # 데이터셋 입력이 ④ 와 같은 (128, 128) 1D 접기다
    {'name': 'ai85safesoundfe_b64', 'min_input': 1, 'dim': 1},
    {'name': 'ai85safesoundfe_f31', 'min_input': 1, 'dim': 1},
    {'name': 'ai85safesoundfe_b64f31', 'min_input': 1, 'dim': 1},
    {'name': 'ai85safesoundfe_v2', 'min_input': 1, 'dim': 1},        # 후속 연구 대조군: ②′ + QAT 문턱 연결 끊음
    {'name': 'ai85safesoundfe_v2_b64f31', 'min_input': 1, 'dim': 1},
    {'name': 'ai85safesoundfe_v2_b64', 'min_input': 1, 'dim': 1},
    {'name': 'ai85safesoundfe_v2_f31', 'min_input': 1, 'dim': 1},
    {'name': 'ai85safesoundfe8', 'min_input': 1, 'dim': 1},          # 후속 연구: p1 8비트 + Abs + NPU 합산 (+ 연결 끊음)
    {'name': 'ai85safesoundfe8_b64', 'min_input': 1, 'dim': 1},
    {'name': 'ai85safesoundfe8_f31', 'min_input': 1, 'dim': 1},
    {'name': 'ai85safesoundfe8_b64f31', 'min_input': 1, 'dim': 1},
    {'name': 'ai85safesoundfe_p1', 'min_input': 1, 'dim': 1},
    {'name': 'ai85safesoundfe_p1_b64', 'min_input': 1, 'dim': 1},
]
