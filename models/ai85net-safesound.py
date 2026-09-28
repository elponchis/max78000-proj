###################################################################################################
#
# SafeSound — MAX78000 음향 이벤트 감지 백본
#
# ai8x-training 의 ai85net-kws20-v3.py (Copyright (C) 2020 Maxim Integrated
# Products, Inc.) 를 파생한 것이다. 원본 저작권 고지:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""SafeSound 음향 이벤트 감지 네트워크 (MAX78000 / AI85).

KWS20 v3 백본을 그대로 쓰되 출력층만 5클래스로 바꾼 것이다.

원본 대비 변경점
  - `num_classes` 21 → 5  (siren / glass / scream / dog_bark / background)
  - `width_mult` 추가: 채널 수 스윕(도전 1순위, TASKS.md Phase 3)을 위해
    0.25× / 0.5× / 1× / 2× 를 한 정의로 다룬다.

구조는 Conv1d(k=1/3/6) 8층 + FC 1층으로, MAX78000 CNN 가속기의 지원 연산만
사용한다. 입력은 raw waveform 16384샘플을 (128, 128) 로 reshape 한 int8 텐서다
(CLAUDE.md 4장). 전처리를 NPU 쪽에 두는 구성 ④에 해당하며, G8 대조군인
구성 ①(MFCC + 2D CNN)은 별도 모델로 정의한다.

메모리 제약 (8bit 가중치, 한계 442KB = 452,608바이트)
    0.25x     12,689 params    2.8%
    0.50x     43,729 params    9.7%
    1.00x    165,457 params   36.6%
    1.25x    252,753 params   55.8%
    1.50x    364,753 params   80.6%   ← 8bit 상한
    1.75x    489,873 params  108.2%   초과
    2.00x    633,425 params  140.0%   초과

⚠️ Conv 파라미터는 채널 수에 **제곱**으로 비례한다(in_ch × out_ch). CLAUDE.md
8장의 "169KB / 442KB 이므로 2× 수용 가능"은 선형 증가를 가정한 오류이며,
실제 8bit 상한은 약 1.6× 다. 채널 스윕은 0.25/0.5/1/1.5× 로 잡는다.
4bit 가중치라면 2× 도 들어가므로 G9(비트폭)와 조합할 여지는 있다.

위 수치는 파라미터 수 기준 추정이다. 확정은 `ai8xize.py` 합성 결과로 할 것
(CLAUDE.md 3장 — 구조 변경 시 442KB/512KB 초과 여부 조기 검증).
"""
import torch
from torch import nn

import ai8x


class AI85SafeSoundNet(nn.Module):
    """SafeSound 1D CNN — KWS20 v3 파생, 전부 Conv1d."""

    def __init__(
            self,
            num_classes=5,
            num_channels=128,
            dimensions=(128, 1),  # pylint: disable=unused-argument
            bias=False,
            width_mult=1.0,
            abs_first=False,
            first_kernel=1,
            **kwargs
    ):
        super().__init__()

        def ch(n):
            """채널 수 스윕. 가속기 정렬을 위해 4의 배수로 맞춘다."""
            return max(4, int(round(n * width_mult / 4)) * 4)

        self.drop = nn.Dropout(p=0.2)

        # T: 128  F: 128
        #
        # ⚠️ 이 층은 `k=1` 이라 **길이 128 FIR 필터 100개**와 수학적으로 같다
        # (입력 채널 축 r 이 프레임 내 잔 시간축이기 때문이다 —
        #  `tools/init_filterbank.py` 참조). 그래서 여기에 멜 대역통과 필터를
        # 초기값으로 넣으면 G8 구성 ③(학습 프론트엔드)이 된다.
        #
        # `abs_first=True` 면 활성화를 **Abs** 로 바꾼다. cos/sin 쌍의 |re|, |im|
        # 을 바로 얻기 위해서다 — ReLU 면 위상에 따라 두 채널이 동시에 0 이 되어
        # 밴드가 사라지므로 밴드당 4필터가 필요해지고 밴드 수가 반으로 준다.
        # MAX78000 은 Abs 를 지원한다 (`ai8x.FusedConv1dAbs`).
        # `first_kernel` 로 **FIR 길이**를 정한다. 입력 채널 축이 프레임 내 잔
        # 시간축이므로, 커널 j 는 **프레임 하나(=128샘플)만큼 떨어진 구간**을 본다:
        #   out[o][c] = Σ_r Σ_j W[o][r][j] · x[r][c+j],  x[r][c] = sample[c*128+r]
        #             = Σ_t W'[o][t] · sample[c*128 + t],  t = j*128 + r
        # 즉 **탭 수 = first_kernel × 128**, 홉은 128 그대로다(겹침이 생긴다).
        #   k=1 → 128탭 8ms,  k=2 → 256탭 16ms,  k=4 → 512탭 32ms(① 멜 n_fft 와 동일)
        #
        # 패딩은 `(k-1)//2` 다. 이러면 출력 길이가 k=1 에서 128, k=2·4 에서 127 이 되고
        # **뒤쪽 층의 차원과 FC 입력(4×64)이 셋 다 같아진다** — 비교에서 바뀌는 것이
        # conv1 하나뿐이도록 하기 위한 선택이다 (아래 flatten 자동 계산이 검증한다).
        fk, fp = first_kernel, (first_kernel - 1) // 2
        self.voice_conv1 = (
            ai8x.FusedConv1dAbs(num_channels, ch(100), fk, stride=1,
                                padding=fp, bias=bias, **kwargs)
            if abs_first else
            ai8x.FusedConv1dReLU(num_channels, ch(100), fk, stride=1,
                                 padding=fp, bias=bias, **kwargs))
        # T: 128  F: 100
        self.voice_conv2 = ai8x.FusedConv1dReLU(ch(100), ch(96), 3, stride=1,
                                                padding=0, bias=bias, **kwargs)
        # T: 126  F: 96
        self.voice_conv3 = ai8x.FusedMaxPoolConv1dReLU(ch(96), ch(64), 3, stride=1,
                                                       padding=1, bias=bias, **kwargs)
        # T: 63  F: 64
        self.voice_conv4 = ai8x.FusedConv1dReLU(ch(64), ch(48), 3, stride=1,
                                                padding=0, bias=bias, **kwargs)
        # T: 61  F: 48
        self.kws_conv1 = ai8x.FusedMaxPoolConv1dReLU(ch(48), ch(64), 3, stride=1,
                                                     padding=1, bias=bias, **kwargs)
        # T: 30  F: 64
        self.kws_conv2 = ai8x.FusedConv1dReLU(ch(64), ch(96), 3, stride=1,
                                              padding=0, bias=bias, **kwargs)
        # T: 28  F: 96
        self.kws_conv3 = ai8x.FusedAvgPoolConv1dReLU(ch(96), ch(100), 3, stride=1,
                                                     padding=1, bias=bias, **kwargs)
        # T: 14  F: 100
        self.kws_conv4 = ai8x.FusedMaxPoolConv1dReLU(ch(100), ch(64), 6, stride=1,
                                                     padding=1, bias=bias, **kwargs)
        # T: 4  F: 64  → flatten 시 4 * ch(64)
        #
        # ⚠️ flatten 길이를 **실제로 통과시켜 구한다.** `first_kernel` 을 바꾸면
        #    앞단 길이가 달라져 뒤쪽 풀링·컨볼루션의 내림 연산을 타고 FC 입력이
        #    조용히 바뀔 수 있다. 하드코딩(4×64)해 두면 그때 shape 오류로 죽거나,
        #    더 나쁘게는 죽지 않고 다른 모델이 된다.
        #    k=1/2/4 는 이 계산으로 셋 다 4×ch(64) 임을 확인했다.
        flat = self._flatten_len(num_channels, dimensions)
        self.fc = ai8x.Linear(flat, num_classes, bias=bias, wide=True, **kwargs)

    def _flatten_len(self, num_channels, dimensions):
        """conv 스택에 0 을 한 번 흘려 FC 입력 길이를 구한다 (파라미터 없음)."""
        length = dimensions[0] if isinstance(dimensions, (tuple, list)) else 128
        with torch.no_grad():
            x = torch.zeros(1, num_channels, length)
            for m in (self.voice_conv1, self.voice_conv2, self.voice_conv3,
                      self.voice_conv4, self.kws_conv1, self.kws_conv2,
                      self.kws_conv3, self.kws_conv4):
                x = m(x)
        return int(x.numel())

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.voice_conv1(x)
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
        # `view` 대신 `flatten` 을 쓴다. 1D 경로는 channels_last 가 없어 지금은
        # 문제가 되지 않지만, 구성 ①에서 바로 이 패턴이 학습을 죽였다
        # (channels_last 에서 view 불가). 두 모델을 같은 형태로 둔다 —
        # 평탄화 논리 순서는 동일하므로 FC 가중치 대응과 합성 결과는 그대로다.
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ai85safesoundnet(pretrained=False, **kwargs):
    """SafeSound 1× 모델."""
    assert not pretrained
    return AI85SafeSoundNet(**kwargs)


def ai85safesoundnet_w025(pretrained=False, **kwargs):
    """채널 0.25× (도전 1순위 스윕)."""
    assert not pretrained
    return AI85SafeSoundNet(width_mult=0.25, **kwargs)


def ai85safesoundnet_w050(pretrained=False, **kwargs):
    """채널 0.5× (도전 1순위 스윕)."""
    assert not pretrained
    return AI85SafeSoundNet(width_mult=0.5, **kwargs)


def ai85safesoundnet_bias(pretrained=False, **kwargs):
    """`bias=True` 판 — **KWS20 v3 사전학습 초기화 전용**이다 (TASKS.md B).

    ADI 공식 체크포인트 `ai85-kws20_v3-qat8.pth.tar` 에는 conv 8층의 **bias 가
    들어 있다**(실측). 우리 기준선은 `bias=False` 라 그대로 얹으면 bias 9개
    텐서가 버려진다 — 사전학습의 일부를 잃는 것이다. 이 진입점은 그것을 받기
    위해서만 쓴다.

    추가되는 파라미터는 conv bias 632개(100+96+64+48+64+96+100+64)뿐이라
    442KB 판정에 영향이 없다. 다만 **다른 ④ 실행과 설정이 달라지므로** 비교표에
    `bias=True` 를 반드시 명기한다.

    ⚠️ **`bias` 를 강제로 True 로 덮어쓴다** (`setdefault` 가 아니다).
    `train.py:797` 이 `model_args["bias"] = args.use_bias` 로 **항상** bias 를
    명시해 넘기고 `--use-bias` 기본값이 False 라, `setdefault` 로 두면 이 진입점이
    조용히 무력화된다. 실제로 그렇게 돌려서 체크포인트의 bias 9개가 버려졌고
    "contains 9 unexpected state keys" 경고만 남았다. 학습 명령에도
    `--use-bias` 를 함께 주어 의도가 명령줄에 보이게 한다.
    """
    assert not pretrained
    if kwargs.get("bias") is False:
        print("[ai85safesoundnet_bias] bias=False 로 들어왔으나 **True 로 덮어쓴다** "
              "— 이 진입점의 목적이 KWS20 체크포인트의 conv bias 를 받는 것이다. "
              "학습 명령에 --use-bias 를 함께 줄 것.")
    kwargs["bias"] = True
    return AI85SafeSoundNet(**kwargs)


def ai85safesoundnet_fb_k2(pretrained=False, **kwargs):
    """D-1 **256탭** 판 — 첫 층 커널 2 (FIR 256탭 = 16ms, 홉 128 로 50% 겹침).

    분해능이 약 500Hz → **250Hz** 로 좋아진다. 128탭에서 서로 분리되지 않던
    500Hz 아래 밴드들이 갈라지는지 보는 실험이다.
    conv1 params 12,800 → 25,600, 총 178,257 (442KB 의 39.4%).
    """
    assert not pretrained
    return AI85SafeSoundNet(abs_first=True, first_kernel=2, **kwargs)


def ai85safesoundnet_fb_k4(pretrained=False, **kwargs):
    """D-1 **512탭** 판 — 첫 층 커널 4 (FIR 512탭 = 32ms, 홉 128).

    **① 멜의  와 창 길이가 같다.** 그래서 "④와 ①의 격차가 주파수
    해상도 때문인가" 에 직접 답한다 — k=1·2·4 세 점이 해상도-성능 곡선이 된다.
    conv1 params 12,800 → 51,200, 총 203,857 (442KB 의 45.0%).
    """
    assert not pretrained
    return AI85SafeSoundNet(abs_first=True, first_kernel=4, **kwargs)


def ai85safesoundnet_fb(pretrained=False, relu_first=False, **kwargs):
    """**필터뱅크 초기화 판** — G8 구성 ③ (학습 프론트엔드). TASKS.md D-1.

    구조는 ④와 같고 첫 층의 활성화만 `Abs` 다 (`relu_first=True` 면 ReLU 유지).
    가중치 초기값은 `tools/init_filterbank.py` 가 멜 대역통과 FIR 로 채운다.
    파라미터 수는 ④와 **완전히 같다** — 활성화는 파라미터가 없다.

    `Abs` 를 쓰는 이유와 밴드 수 계산은 `tools/init_filterbank.py` 의 설명을
    볼 것. 요지는 cos/sin 쌍에 `Abs` 를 걸면 밴드당 2필터로 |re|, |im| 이 나와
    100채널에 **50밴드**가 들어가고, ReLU 면 부호 쌍까지 필요해 25밴드로 준다는
    것이다.
    """
    assert not pretrained
    return AI85SafeSoundNet(abs_first=not relu_first, **kwargs)


def ai85safesoundnet_fb_relu(pretrained=False, **kwargs):
    """필터뱅크 초기화 + **ReLU 유지** (25밴드). 초기화만 바꾸는 순수 대조용."""
    assert not pretrained
    return AI85SafeSoundNet(abs_first=False, **kwargs)


def ai85safesoundnet_w150(pretrained=False, **kwargs):
    """채널 1.5× — 8bit 가중치 상한(442KB의 80.6%). 스윕의 최대 지점."""
    assert not pretrained
    return AI85SafeSoundNet(width_mult=1.5, **kwargs)


models = [
    {'name': 'ai85safesoundnet', 'min_input': 1, 'dim': 1},
    {'name': 'ai85safesoundnet_w025', 'min_input': 1, 'dim': 1},
    {'name': 'ai85safesoundnet_w050', 'min_input': 1, 'dim': 1},
    {'name': 'ai85safesoundnet_w150', 'min_input': 1, 'dim': 1},
    {'name': 'ai85safesoundnet_bias', 'min_input': 1, 'dim': 1},
    {'name': 'ai85safesoundnet_fb', 'min_input': 1, 'dim': 1},
    {'name': 'ai85safesoundnet_fb_relu', 'min_input': 1, 'dim': 1},
    {'name': 'ai85safesoundnet_fb_k2', 'min_input': 1, 'dim': 1},
    {'name': 'ai85safesoundnet_fb_k4', 'min_input': 1, 'dim': 1},
]
