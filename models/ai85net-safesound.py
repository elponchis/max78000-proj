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
            **kwargs
    ):
        super().__init__()

        def ch(n):
            """채널 수 스윕. 가속기 정렬을 위해 4의 배수로 맞춘다."""
            return max(4, int(round(n * width_mult / 4)) * 4)

        self.drop = nn.Dropout(p=0.2)

        # T: 128  F: 128
        self.voice_conv1 = ai8x.FusedConv1dReLU(num_channels, ch(100), 1, stride=1,
                                                padding=0, bias=bias, **kwargs)
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
        self.fc = ai8x.Linear(4 * ch(64), num_classes, bias=bias, wide=True, **kwargs)

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
        x = x.view(x.size(0), -1)
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


def ai85safesoundnet_w150(pretrained=False, **kwargs):
    """채널 1.5× — 8bit 가중치 상한(442KB의 80.6%). 스윕의 최대 지점."""
    assert not pretrained
    return AI85SafeSoundNet(width_mult=1.5, **kwargs)


models = [
    {'name': 'ai85safesoundnet', 'min_input': 1, 'dim': 1},
    {'name': 'ai85safesoundnet_w025', 'min_input': 1, 'dim': 1},
    {'name': 'ai85safesoundnet_w050', 'min_input': 1, 'dim': 1},
    {'name': 'ai85safesoundnet_w150', 'min_input': 1, 'dim': 1},
]
