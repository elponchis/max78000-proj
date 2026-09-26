###################################################################################################
#
# SafeSound — G8 구성 ① 대조 모델 (로그 멜 + 2D CNN)
#
# ai8x-training 의 모델 정의 규약을 따른다.
# Copyright (C) 2020 Maxim Integrated Products, Inc. 의 원본 고지:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""SafeSound 로그 멜 2D CNN (MAX78000 / AI85) — G8 구성 ①.

구성 ④(`ai85net-safesound.py`, raw 파형 1D CNN)의 대조군이다. 전처리를 M4 CPU 가
전부 하고(STFT + 멜 필터뱅크 + 로그) NPU 는 2D CNN 만 한다.

**설계 원칙은 하나 — 파라미터 수를 ④에 맞춘다.** 용량이 다르면 정확도 차이가
입력 표현 때문인지 모델 크기 때문인지 갈리지 않는다.

    구성 ④ (1D, width 1.0)   165,457 params   36.6% of 442KB
    구성 ① (2D, width 1.0)   157,472 params   34.8% of 442KB

MAX78000 제약 (CLAUDE.md 3장)
  - Conv2d 는 **3×3 과 1×1 만**. 임의 커널 불가
  - depthwise separable conv **미지원** (MAX78002 에서 추가)
  - 가중치 442KB / 데이터 512KB
  - BatchNorm 은 학습 후 fold, Softmax 는 CPU

BatchNorm 을 **쓰지 않는다.** 구성 ④가 `FusedConv1dReLU`(BN 없음)이기 때문이다 —
정규화 유무는 구조 선택이고, 한쪽에만 주면 비교가 오염된다. ①이 학습되지 않으면
그때 **양쪽에 동시에** 넣는다.

데이터 메모리: 최대 중간 텐서는 1층 출력 32×64×64 = 131,072바이트 (512KB 의 26%).
채널당 4,096바이트로 프로세서당 한도 안이다. **확정은 `ai8xize.py` 합성 결과로
한다** (CLAUDE.md 3장 — 구조 변경 시 조기 검증).
"""
import torch
from torch import nn

import ai8x


class AI85SafeSoundMelNet(nn.Module):
    """로그 멜 (1, 64, 64) int8 → 5클래스. 전부 Conv2d 3×3 / 1×1."""

    def __init__(
            self,
            num_classes=5,
            num_channels=1,
            dimensions=(64, 64),  # pylint: disable=unused-argument
            bias=False,
            width_mult=1.0,
            pool_first=False,
            **kwargs
    ):
        super().__init__()

        def ch(n):
            """채널 수 스윕. 가속기 정렬을 위해 4의 배수로 맞춘다 (④와 동일)."""
            return max(4, int(round(n * width_mult / 4)) * 4)

        # 드롭아웃 비율도 ④와 같은 0.2 다
        self.drop = nn.Dropout(p=0.2)

        # 입력 1×64×64 (멜 64 × 프레임 64).
        #
        # `pool_first=True` 면 입력이 **128×128** 인 경우다 (raw 파형 2D, wave2D
        # 대조 실험). 앞에 2×2 풀링을 하나 더 넣어 이후 전부를 ①과 **동일하게**
        # 맞춘다 — 파라미터 수도 정확히 같다(157,535). 1층 출력이 32×64×64 =
        # 131KB 로 데이터 메모리도 ①과 같다.
        # ⚠️ 풀링을 안 넣으면 1층 출력이 32×128×128 = **524,288B = 512KB 데이터
        #    메모리를 정확히 100%** 쓴다(실측). 입력 16KB 와 다음 층이 들어갈
        #    자리가 없으므로 실제로는 들어가지 않는다. 즉 이 풀링은 선택이 아니라
        #    하드웨어 제약이다.
        #    (그 자체가 "raw 파형 입력은 공간 축소가 더 필요하다" 는 결과이고,
        #     비교의 한계로 문서에 적는다 — g8-first-comparison.md 4.1)
        self.conv1 = (
            ai8x.FusedMaxPoolConv2dReLU(num_channels, ch(32), 3, pool_size=2,
                                        pool_stride=2, stride=1, padding=1,
                                        bias=bias, **kwargs)
            if pool_first else
            ai8x.FusedConv2dReLU(num_channels, ch(32), 3, stride=1,
                                 padding=1, bias=bias, **kwargs))
        # 32×64×64
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(ch(32), ch(32), 3, pool_size=2,
                                                 pool_stride=2, stride=1,
                                                 padding=1, bias=bias, **kwargs)
        # 32×32×32
        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(ch(32), ch(64), 3, pool_size=2,
                                                 pool_stride=2, stride=1,
                                                 padding=1, bias=bias, **kwargs)
        # 64×16×16
        self.conv4 = ai8x.FusedMaxPoolConv2dReLU(ch(64), ch(64), 3, pool_size=2,
                                                 pool_stride=2, stride=1,
                                                 padding=1, bias=bias, **kwargs)
        # 64×8×8
        self.conv5 = ai8x.FusedMaxPoolConv2dReLU(ch(64), ch(128), 3, pool_size=2,
                                                 pool_stride=2, stride=1,
                                                 padding=1, bias=bias, **kwargs)
        # 128×4×4
        self.conv6 = ai8x.FusedMaxPoolConv2dReLU(ch(128), ch(128), 1, pool_size=2,
                                                 pool_stride=2, stride=1,
                                                 padding=0, bias=bias, **kwargs)
        # 128×2×2 → flatten 512
        self.fc = ai8x.Linear(ch(128) * 2 * 2, num_classes, bias=bias,
                              wide=True, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.drop(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # ⚠️ `view` 를 쓰면 안 된다. 입력이 channels_last 이면 (C,H,W) 가 메모리에서
        # 연속이 아니라 `view` 가 RuntimeError 로 죽는다. Colab 학습이 에폭 0
        # 마지막 배치(59샘플)에서 torch.compile 재컴파일 후 channels_last 로
        # 바뀌면서 여기서 터졌다. `flatten` 은 필요하면 복사해서 **논리 순서를
        # 보존**한다 — 즉 평탄화 결과는 C×H×W 순서 그대로이고 FC 가중치 대응도
        # 그대로다 (합성 결과에 영향 없음). `tools/kat_models.py` 가 이것을
        # channels_last 입력으로 회귀 검사한다.
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ai85safesoundmelnet(pretrained=False, **kwargs):
    """구성 ① 1× 모델 — 구성 ④와 파라미터 수를 맞춘 지점."""
    assert not pretrained
    return AI85SafeSoundMelNet(**kwargs)


def ai85safesoundmelnet_w050(pretrained=False, **kwargs):
    """채널 0.5× — 모델 규모별 교차점(도전 1순위)의 ① 쪽 점."""
    assert not pretrained
    return AI85SafeSoundMelNet(width_mult=0.5, **kwargs)


def ai85safesoundmelnet_w150(pretrained=False, **kwargs):
    """채널 1.5× — 8bit 가중치 상한 근처. 합성으로 확인할 것."""
    assert not pretrained
    return AI85SafeSoundMelNet(width_mult=1.5, **kwargs)


def ai85safesoundwave2dnet(pretrained=False, **kwargs):
    """**wave2D 대조 실험** — 구성 ①과 완전히 같은 2D 구조에 raw 파형을 넣는다.

    입력은 `(1, 128, 128)`, 즉 구성 ④가 쓰는 것과 **같은 파형·같은 접기**에 채널
    축만 붙인 것이다 (`datasets/safesound_wave2d.py`).

    이 구성이 있어야 G8 의 주장이 성립한다. ①(멜 2D) vs ④(파형 1D) 는 표현과
    구조가 동시에 달라 격차의 원인이 분리되지 않는다. 세 점을 놓으면 갈린다.

        ① mel      vs  wave2D     → **표현만** 다르다 (구조 고정)
        wave2D      vs  ④ wave1D  → **구조만** 다르다 (입력 정보 고정)

    ⚠️ 완벽한 통제는 아니다. (a) 128×128 입력이라 앞에 2×2 풀링이 하나 더 붙는다
    (데이터 메모리 512KB 제약 — 위 `pool_first` 주석). (b) `(128,128)` 접기는
    스펙트로그램이 아니다 — 행은 128샘플 간격의 값이라 3×3 conv 가 보는 이웃이
    물리적 의미를 갖지 않는다. 두 점 모두 논문에 적는다
    (`docs/results/g8-first-comparison.md` 4.1).

    파라미터 수는 ①과 **정확히 같다** (157,535).
    """
    assert not pretrained
    return AI85SafeSoundMelNet(pool_first=True, dimensions=(128, 128), **kwargs)


models = [
    {'name': 'ai85safesoundmelnet', 'min_input': 1, 'dim': 2},
    {'name': 'ai85safesoundmelnet_w050', 'min_input': 1, 'dim': 2},
    {'name': 'ai85safesoundmelnet_w150', 'min_input': 1, 'dim': 2},
    {'name': 'ai85safesoundwave2dnet', 'min_input': 1, 'dim': 2},
]
