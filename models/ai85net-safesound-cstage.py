"""구성 (c) — NPU 필터뱅크 → CPU 정수 로그 → NPU 분류기.

**이 파일은 _합성 확인용_ 두 조각만 정의한다.** 학습용 단일 모델(중간에
CPU 로그가 낀 한 덩어리)은 1단계 통과 후에 만든다 — 설계는
`docs/results/g8-c-design.md` 3.3.

조각을 나누는 이유: MAX78000 은 네트워크 하나를 한 번에 올린다. (c) 는
중간에 CPU 연산이 끼므로 `cnn_configure()` 를 두 번 불러 **2패스**로 돈다.
합성도 따로 해야 하고, 두 패스의 가중치가 442KB 안에 **동시에** 들어가는지가
(c) 의 성패를 가른다.

| 패스 | 층 | 입력 | 출력 |
|---|---|---|---|
| p1 | voice_conv1 (128→100, k=1) | (128,128) int8 | (100,128) **int32** |
| — | **CPU**: \\|·\\| → log2(CLZ+LUT) → int8 | | |
| p2 | voice_conv2 … fc | (100,128) int8 | 로짓 5 |

두 조각 모두 **D-1(`ai85safesoundnet_fb`) 의 층을 그대로** 쓴다. 바꾼 것은
p1 의 활성화(Abs → 없음, 32bit 출력)뿐이다. 그래야 (c) 와 D-1 의 차이가
로그단 하나로 남는다.
"""
import torch.nn as nn

import ai8x


class AI85SafeSoundCStageP1(nn.Module):
    """(c) 1패스 — 멜 필터뱅크 한 층. **wide 32bit 출력, 활성화 없음.**

    `wide=True` 는 출력을 int32 로 내보낸다. Abs 를 붙이지 않는 이유는
    설계 문서 2.2 — 32bit 출력에 활성화가 붙는지가 불확실하고, `|·|` 는
    CPU 로그 루프에 융합하면 공짜다.
    """

    def __init__(
            self,
            num_classes=5,  # pylint: disable=unused-argument
            num_channels=128,
            dimensions=(128, 1),  # pylint: disable=unused-argument
            bias=False,
            **kwargs
    ):
        super().__init__()
        self.voice_conv1 = ai8x.FusedConv1dReLU(
            num_channels, 100, 1, stride=1, padding=0, bias=bias, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        return self.voice_conv1(x)


class AI85SafeSoundCStageP2(nn.Module):
    """(c) 2패스 — CPU 로그를 거친 100채널 int8 을 받는 분류기.

    `ai85net-safesound.py` 의 voice_conv2 이하와 **같은 층들**이다.
    여기서 다시 쓰는 대신 복사한 이유는, 합성이 모델 하나당 yaml 하나를
    요구해서 **입력층이 voice_conv2 인 모델**이 따로 있어야 하기 때문이다.
    층 정의가 갈라지지 않도록 수정 시 두 파일을 함께 본다.
    """

    def __init__(
            self,
            num_classes=5,
            num_channels=100,
            dimensions=(128, 1),  # pylint: disable=unused-argument
            bias=False,
            **kwargs
    ):
        super().__init__()
        self.drop = nn.Dropout(p=0.2)
        self.voice_conv2 = ai8x.FusedConv1dReLU(
            num_channels, 96, 3, stride=1, padding=0, bias=bias, **kwargs)
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


def ai85safesoundcstage_p1(pretrained=False, **kwargs):
    """(c) 1패스 — 필터뱅크, 32bit 출력."""
    assert not pretrained
    return AI85SafeSoundCStageP1(**kwargs)


def ai85safesoundcstage_p2(pretrained=False, **kwargs):
    """(c) 2패스 — 분류기."""
    assert not pretrained
    return AI85SafeSoundCStageP2(**kwargs)


# ⚠️ 이 목록에 넣지 않으면 `train.py --model …` 이 받지 않는다.
# (같은 실수를 네 번 했다 — `tools/kat_entrypoints.py` 가 검사한다)
models = [
    {'name': 'ai85safesoundcstage_p1', 'min_input': 1, 'dim': 1},
    {'name': 'ai85safesoundcstage_p2', 'min_input': 1, 'dim': 1},
]
