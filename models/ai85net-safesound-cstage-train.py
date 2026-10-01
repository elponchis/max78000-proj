"""구성 (c) 학습용 — NPU 필터뱅크 → **CPU 정수 로그** → NPU 분류기, 한 덩어리.

합성은 두 조각(`ai85net-safesound-cstage.py`)으로 하지만 **학습은 하나의
모델**이다. 중간에 CPU 연산이 낀 것뿐이고, 역전파는 끝까지 흐른다.

## 로그단의 정의 — 펌웨어와 같은 식

```
y   = voice_conv1(x)            # wide, 디바이스에서는 int32
v   = round(|y| · 2^SCALE_SH)   # 디바이스 int32 스케일로
lg  = log2_q8(v)                # tools/log2_q8.py == firmware/common/log2_q8.c
out = to_int8(lg, LO_Q8, SPAN_Q8)   # 포화
```

`log2_q8` 과 `to_int8` 은 **파이썬과 C 가 비트 단위로 같다**
(`tools/kat_logstage.py` 가 전수 확인, 2026-10-01 통과).

### ⚠️ `SCALE_SH` 가 디바이스 int32 스케일과 정확히 같지 않아도 된다

`log2(a·v) = log2(v) + log2(a)` 이므로 **배율 차이는 로그 영역의 상수
평행이동**이고, `LO_Q8` 이 그만큼 흡수한다. 지켜야 할 것은
**학습과 펌웨어가 같은 (SCALE_SH, LO_Q8, SPAN_Q8) 을 쓰는 것**뿐이다.
→ 세 상수를 펌웨어 헤더로 내보내고, 바뀌면 양쪽을 함께 고친다.

### 상수의 근거 (2026-10-01 실측)

D-1 의 voice_conv1 출력을 테스트 400창에 통과시켜 쟀다 (`|y|`, 비영):

| | 값 |
|---|---|
| 0 비율 | 0.30% |
| p1 / p50 / p99 | 0.00269 / 0.150 / 10.53 |
| **p1~p99 동적범위** | **11.93 log2 = 71.8 dB** |

→ `SPAN_Q8 = 2976` (= 11.625 log2 = **70.0 dB**). ① 의 `SPAN_DB=70` 과
  같은 폭이다 — ① 의 범위 스윕에서 **넓힐수록 LSB 해상도를 잃는다**가
  나왔으므로(48→72dB 에서 −0.038) 같은 값을 쓴다.
→ `LO_Q8 = 1472` 로 창의 위끝을 p99 에 맞춘다 (`SCALE_SH = 14` 기준).

## 역전파 — STE

`round`·`log2_q8`·`to_int8` 은 전부 계단 함수라 미분이 0 이다.
**STE** 로 `d(out)/d(y) = d/dy [ (log2|y| - lo) · 255/span ]`
`= 255 / (span · ln2 · |y|) · sign(y)` 를 흘린다.
⚠️ `|y|` 가 작으면 발산하므로 `y_min` 으로 막는다 — 막지 않으면 조용한
구간에서 그래디언트가 터진다.
"""
import os

import numpy as np
import torch
import torch.nn as nn

import ai8x

# ── 로그단 상수 — **펌웨어와 공유한다** ─────────────────────────────
SCALE_SH = 14          # |y| → 디바이스 int32 스케일 (2의 거듭제곱)
LO_Q8 = 1472           # 창 아래끝, Q8 log2
SPAN_Q8 = 2976         # 창 폭, Q8 log2 (= 70.0 dB)
Y_MIN = 2.0 ** -SCALE_SH   # STE 에서 1/|y| 를 막는 하한 (= 1 LSB)

_LN2 = float(np.log(2.0))

# ── 빠른 등가 구현 — **계단 경계 255개로 바꾼다** ─────────────────
# |y| → int8 은 **|y| 의 비감소 계단 함수**이고 출력 레벨이 256개뿐이다.
# 따라서 레벨이 바뀌는 |y| 경계 255개만 알면 `bucketize` 한 번으로 끝난다.
# 정수 경로(round → CLZ → LUT → 포화)를 그대로 돌면 배치당 0.224초인데
# 이쪽은 0.062초다 — **3.6배**. 150에폭에서 3.15시간 → 1.3시간.
#
# 경계는 `tools/gen_cstage_bounds.py` 가 **정수 경로에서 이진탐색으로**
# 뽑는다. 즉 등가성이 구성상 보장되고, 검증도 했다:
#   · 실제 p1 활성값 7,680,000개 — **불일치 0**
#   · 정규·극소·극대 난수 각 20만개 — **불일치 0**
#   · ⚠️ `|y|·2^SCALE_SH` 가 **정확히 x.5** 인 입력에서만 갈린다
#     (`torch.round` 는 half-to-even, 경계 비교는 half-up).
#     **기기에는 이 반올림이 없다** — NPU 가 int32 를 직접 주므로
#     `round` 는 학습 그래프가 float 를 정수로 옮기는 단계일 뿐이다.
#     따라서 train/serve 불일치가 아니다.
# ⚠️ **`realpath` 여야 한다.** 이 파일은 `~/ai8x-training/models/` 에
# 심볼릭 링크로 걸려 그쪽에서 import 된다. `abspath` 는 링크를 풀지 않아
# 레포가 `~/ai8x-training` 으로 잡히고 경계 파일을 못 찾는다.
_BOUNDS_NPY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "data/synth/cstage_bounds.npy")

# ⚠️ **import 시점에 읽지 않는다.** `train.py` 는 `models/` 의 **모든**
# 파일을 import 하므로(pydoc.locate), 여기서 예외가 나면 **이 모델과
# 무관한 학습까지 전부 죽는다.** 실제로 그렇게 chain15 의 마지막 시드가
# 죽었다 (2026-10-01). 모델을 만들 때 처음 한 번만 읽는다.
_BOUNDS = None


def _bounds():
    """계단 경계 255개. 처음 쓸 때 읽는다."""
    global _BOUNDS                                  # noqa: PLW0603
    if _BOUNDS is None:
        if not os.path.isfile(_BOUNDS_NPY):
            raise FileNotFoundError(
                f"{_BOUNDS_NPY} 가 없다. "
                "`python3 tools/gen_cstage_bounds.py` 를 먼저 돌릴 것")
        b = torch.from_numpy(np.load(_BOUNDS_NPY))
        assert b.numel() == 255, "경계는 255개여야 한다 (레벨 256개)"
        _BOUNDS = b
    return _BOUNDS


class _IntLog(torch.autograd.Function):
    """|y| → 정수 로그 → int8. 순전파는 펌웨어와 같고, 역전파는 STE."""

    @staticmethod
    def forward(ctx, y):
        ctx.save_for_backward(y)
        q = torch.bucketize(y.detach().abs().double(),
                            _bounds().to(y.device)) - 128
        return q.to(y.dtype)

    @staticmethod
    def backward(ctx, g):
        (y,) = ctx.saved_tensors
        # d/dy [ (log2|y| - lo) · 255/span ] = 255 / (span/256 · ln2 · |y|)
        denom = (SPAN_Q8 / 256.0) * _LN2 * y.abs().clamp(min=Y_MIN)
        return g * (255.0 / denom) * torch.sign(y)


class AI85SafeSoundCStageTrain(nn.Module):
    """(c) 학습 모델. D-1 과 **로그단 하나만** 다르다."""

    def __init__(
            self,
            num_classes=5,
            num_channels=128,
            dimensions=(128, 1),  # pylint: disable=unused-argument
            bias=False,
            **kwargs
    ):
        super().__init__()
        self.drop = nn.Dropout(p=0.2)
        # NPU-1 — 활성화 없는 wide 층 (디바이스에서 int32 출력)
        self.voice_conv1 = ai8x.Conv1d(
            num_channels, 100, 1, stride=1, padding=0, bias=bias,
            wide=True, **kwargs)
        # NPU-2 — D-1 의 voice_conv2 이하 그대로
        self.voice_conv2 = ai8x.FusedConv1dReLU(
            100, 96, 3, stride=1, padding=0, bias=bias, **kwargs)
        self.voice_conv3 = ai8x.FusedMaxPoolConv1dReLU(
            96, 64, 3, stride=1, pool_size=2, pool_stride=2, padding=1,
            bias=bias, **kwargs)
        self.voice_conv4 = ai8x.FusedConv1dReLU(
            64, 48, 3, stride=1, padding=0, bias=bias, **kwargs)
        self.kws_conv1 = ai8x.FusedMaxPoolConv1dReLU(
            48, 64, 3, stride=1, pool_size=2, pool_stride=2, padding=1,
            bias=bias, **kwargs)
        self.kws_conv2 = ai8x.FusedConv1dReLU(
            64, 96, 3, stride=1, padding=0, bias=bias, **kwargs)
        self.kws_conv3 = ai8x.FusedAvgPoolConv1dReLU(
            96, 100, 3, stride=1, pool_size=2, pool_stride=2, padding=1,
            bias=bias, **kwargs)
        self.kws_conv4 = ai8x.FusedMaxPoolConv1dReLU(
            100, 64, 6, stride=1, pool_size=2, pool_stride=2, padding=1,
            bias=bias, **kwargs)
        self.fc = ai8x.Linear(256, num_classes, bias=bias, wide=True, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.voice_conv1(x)
        # ── 여기가 CPU 다 ──────────────────────────────────────────
        # NPU-2 의 입력은 int8 [-128,127] 이고, ai8x 의 활성화 규약은
        # [-1, 1) 이므로 128 로 나눠 넣는다 (`ai8x.normalize` 와 같은 아핀).
        x = _IntLog.apply(x) / 128.0
        # ──────────────────────────────────────────────────────────
        x = self.voice_conv2(x)
        x = self.voice_conv3(x)
        x = self.drop(x)
        x = self.voice_conv4(x)
        x = self.kws_conv1(x)
        x = self.drop(x)
        x = self.kws_conv2(x)
        x = self.kws_conv3(x)
        x = self.drop(x)
        x = self.kws_conv4(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def ai85safesoundcstage(pretrained=False, **kwargs):
    """(c) 학습 모델."""
    assert not pretrained
    return AI85SafeSoundCStageTrain(**kwargs)


models = [
    {'name': 'ai85safesoundcstage', 'min_input': 1, 'dim': 1},
]
