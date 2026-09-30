###################################################################################################
#
# SafeSound + NPU 로그 근사 — G8 라운드 6 (a)
#
# `models/ai85net-safesound.py` (D-1) 에 **구간선형 오목 함수** 두 층을 끼운 것.
# 원본은 ai8x-training 의 ai85net-kws20-v3.py (Copyright (C) 2020 Maxim
# Integrated Products, Inc.) 파생이다.
#
###################################################################################################
"""D-1 의 대역 크기 뒤에 **NPU 안에서** 로그를 근사한다 (G8 라운드 6 (a)).

라운드 6 이 확정한 것: ①−④ 격차의 주원인은 **로그 압축**이고, MAX78000 은
로그를 지원하지 않는다 (`docs/results/g8-round6-logcompression.md`).

로그는 오목(concave)이다. 오목 구간선형 함수는 **기울기가 단조 감소하는
ReLU 의 음의 가중합**으로 정확히 쓸 수 있다.

    f(v) = a·v − Σ_k c_k · ReLU(v − t_k)        c_k > 0,  t_1 < … < t_K

MAX78000 지원 연산만으로:

    L0  voice_conv1 : Conv1d(128→100, k=1) + **Abs**     ← D-1 그대로 (대역 크기)
    La  log_knot    : Conv1d(100→100K, k=1) + **ReLU**, per-channel bias = −t_k
    Lb  log_sum     : Conv1d(100K→100, k=1), **활성화 없음** (음의 가중합)
    L1… voice_conv2 이하 : D-1 그대로

⚠️ **왜 낭비가 생기는가**: MAX78000 은 depthwise separable conv 를 지원하지
않는다 (CLAUDE.md 3장). 채널별 독립 연산을 full 1×1 conv 로 흉내 내야 하므로
`La`·`Lb` 가중치의 **99% 이상이 0** 이다. 이 낭비 자체가 "가속기 제약이
파이프라인 설계를 강제한다" 는 논문 주장의 사례라, 숨기지 않고 그대로 싣는다.

⚠️ **블록대각은 초기화일 뿐 고정이 아니다.** ai8x 에 가중치 마스킹이 없어서
학습 중 0 이 풀린다. 그래도 무방하다 — 채널 혼합이 생기면 그건 모델이
스스로 고른 것이고, 합성 비용은 어차피 full 1×1 기준으로 이미 지불했다.

사전 확인 (`tools/sim_late_log.py`, 2026-09-30): 이 구조가 흉내 내는 경로
(S1b, int8 → 구간선형 K=4)가 ① 모델 기준 시뮬레이션에서 **회복률 88.5%**.
⚠️ 그 값은 **① 모델로 평가한 시뮬레이션**이고 (a) 의 실측이 아니다.
"""
import numpy as np
import torch
from torch import nn

import ai8x


class AI85SafeSoundLogStage(nn.Module):
    """D-1 + NPU 로그 근사 두 층."""

    def __init__(
            self,
            num_classes=5,
            num_channels=128,
            dimensions=(128, 1),  # pylint: disable=unused-argument
            bias=False,
            knots=4,
            **kwargs
    ):
        super().__init__()
        self.knots = knots
        self.drop = nn.Dropout(p=0.2)
        c0, mid = 100, 100 * knots

        # ── L0: D-1 의 1층 그대로 (대역 크기) ─────────────────────────
        self.voice_conv1 = ai8x.FusedConv1dAbs(num_channels, c0, 1, stride=1,
                                               padding=0, bias=bias, **kwargs)
        # ── La: 마디. **bias 가 마디 위치 −t_k 다** → 항상 bias=True ──
        self.log_knot = ai8x.FusedConv1dReLU(c0, mid, 1, stride=1, padding=0,
                                             bias=True, **kwargs)
        # ── Lb: 음의 가중합. 활성화가 없어야 오목 합이 보존된다 ────────
        self.log_sum = ai8x.Conv1d(mid, c0, 1, stride=1, padding=0,
                                   bias=bias, activation=None, **kwargs)

        # ── 이하 D-1 과 동일 ─────────────────────────────────────────
        self.voice_conv2 = ai8x.FusedConv1dReLU(c0, 96, 3, stride=1, padding=0,
                                                bias=bias, **kwargs)
        self.voice_conv3 = ai8x.FusedMaxPoolConv1dReLU(96, 64, 3, stride=1,
                                                       padding=1, bias=bias,
                                                       **kwargs)
        self.voice_conv4 = ai8x.FusedConv1dReLU(64, 48, 3, stride=1, padding=0,
                                                bias=bias, **kwargs)
        self.kws_conv1 = ai8x.FusedMaxPoolConv1dReLU(48, 64, 3, stride=1,
                                                     padding=1, bias=bias,
                                                     **kwargs)
        self.kws_conv2 = ai8x.FusedConv1dReLU(64, 96, 3, stride=1, padding=0,
                                              bias=bias, **kwargs)
        self.kws_conv3 = ai8x.FusedAvgPoolConv1dReLU(96, 100, 3, stride=1,
                                                     padding=1, bias=bias,
                                                     **kwargs)
        self.kws_conv4 = ai8x.FusedMaxPoolConv1dReLU(100, 64, 6, stride=1,
                                                     padding=1, bias=bias,
                                                     **kwargs)
        self.fc = ai8x.Linear(4 * 64, num_classes, bias=bias, wide=True,
                              **kwargs)

    # ── 초기화 ────────────────────────────────────────────────────────
    def init_log_stage(self, lo=1.0 / 255.0):
        """`La`/`Lb` 를 **로그의 구간선형 오목 근사**로 초기화한다.

        활성화 축을 [0,1] 로 본다 (NPU int8 활성화). 마디 `t_1 … t_K` 를
        `lo`(1 LSB)부터 로그 간격으로 놓고, 출력도 **[0,1] 로 정규화**한다:

            g(v) = (log10 v − log10 lo) / (−log10 lo),   v ∈ [lo, 1]
                 = s_1·ReLU(v−t_1) + Σ_{k≥2} (s_k − s_{k−1})·ReLU(v−t_k)

        `g(t_1) = 0` 이라 **상수항이 없다** → `Lb` 에 bias 가 필요 없다.
        로그가 오목이라 `s_k` 가 단조 감소하고, 따라서 `k≥2` 의 계수가 전부
        음수다 — 그래야 ReLU 의 **음의 가중합**으로 오목 함수가 된다.

        ⚠️ 출력을 [0,1] 로 맞추는 이유: 그대로 두면 log10 범위가 [−2.4, 0]
        이라 int8 활성화 축에서 거의 0 에 뭉친다. 입력과 같은 축척으로 두면
        `output_shift` 가 잡기 쉽다.

        ⚠️ 인위적인 0 마디를 넣지 않는다. 넣으면 첫 구간 기울기가 0 이 되고
        다음 구간에서 **증가**해 오목성이 깨진다 (2026-09-30 실수).
        """
        K = self.knots
        t = np.exp(np.linspace(np.log(lo), 0.0, K + 1))[:K]   # t_1..t_K, t_1=lo
        span = -np.log10(lo)
        g = (np.log10(np.concatenate([t, [1.0]])) - np.log10(lo)) / span
        sl = np.diff(g) / np.diff(np.concatenate([t, [1.0]]))  # 구간 기울기 K개
        assert np.all(np.diff(sl) < 0), f"오목하지 않다: {sl}"
        coef = np.concatenate([[sl[0]], np.diff(sl)])          # k=1 은 s_1

        with torch.no_grad():
            wk = torch.zeros_like(self.log_knot.op.weight)     # (100K, 100, 1)
            bk = torch.zeros_like(self.log_knot.op.bias)       # (100K,)
            ws = torch.zeros_like(self.log_sum.op.weight)      # (100, 100K, 1)
            for c in range(100):
                for j in range(K):
                    o = c * K + j
                    wk[o, c, 0] = 1.0
                    bk[o] = -float(t[j])           # ReLU(v − t_j)
                    ws[c, o, 0] = float(coef[j])
            self.log_knot.op.weight.copy_(wk)
            self.log_knot.op.bias.copy_(bk)
            self.log_sum.op.weight.copy_(ws)
        return {"knots": [float(x) for x in t],
                "slopes": [float(x) for x in sl],
                "coef": [float(x) for x in coef]}

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.voice_conv1(x)      # 대역 크기
        x = self.log_knot(x)         # ReLU(v − t_k)
        x = self.log_sum(x)          # 음의 가중합 = 로그 근사
        x = self.voice_conv2(x)
        x = self.drop(x)
        x = self.voice_conv3(x)
        x = self.voice_conv4(x)
        x = self.drop(x)
        x = self.kws_conv1(x)
        x = self.kws_conv2(x)
        x = self.drop(x)
        x = self.kws_conv3(x)
        x = self.kws_conv4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ai85safesoundlogstage(pretrained=False, **kwargs):
    """K=4 마디 (기본)."""
    assert not pretrained
    return AI85SafeSoundLogStage(**kwargs)


def ai85safesoundlogstage_k2(pretrained=False, **kwargs):
    """K=2 마디 — 비용을 절반으로 줄인 판."""
    assert not pretrained
    return AI85SafeSoundLogStage(knots=2, **kwargs)


models = [
    {'name': 'ai85safesoundlogstage', 'min_input': 1, 'dim': 1},
    {'name': 'ai85safesoundlogstage_k2', 'min_input': 1, 'dim': 1},
]
