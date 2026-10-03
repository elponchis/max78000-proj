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
import os

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
            pool6=True,
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
        # 채널당 4,096픽셀로 데이터 메모리도 ①과 같다.
        #
        # ⚠️ **풀링은 선택이 아니라 하드웨어 제약이다.** 데이터 메모리는 32KiB
        #    인스턴스 16개이고, HWC(모든 층 출력)에서는 4채널이 한 인스턴스를
        #    나눠 쓰므로 **채널당 8,192픽셀**이 한계다
        #    (`ai8x-synthesis/README.md:1085`). 128×128 = 16,384픽셀이라
        #    **채널 수와 무관하게** 초과한다 — 1층을 16채널로 줄여도 안 들어간다.
        #    (입력은 CHW 라 채널당 32,768픽셀까지 되므로 1×128×128 자체는 OK다.)
        #
        # ⚠️ **AvgPool 을 쓴다.** 이 풀링만은 ReLU 이전의 **부호 있는 파형**에
        #    걸리므로 MaxPool 이 부적절하다 — 양의 피크만 골라 파형의 부호 구조를
        #    망친다. conv2~conv6 의 MaxPool 은 ReLU 이후(비음수)라 그대로 둔다.
        #
        # ⚠️ 이 풀링이 무엇을 버리는지 정직하게: (128,128) 접기에서 행(r)은
        #    연속 샘플 축, 열(c)은 8ms 간격 프레임 축이다. 2×2 평균은
        #    (a) 연속 샘플 축에서 2탭 저역통과 + 2배 데시메이션 → **4kHz 이상을
        #    버린다** (유리 파손 같은 광대역 과도음에 불리하다), (b) 프레임 축에서
        #    8ms 떨어진 두 값을 섞는다. 따라서 wave2D 는 "구조만 바꾼 것" 이
        #    **아니다** — 선행 풀링이라는 변경이 함께 들어간다.
        #    정보를 버리지 않는 대안은 `ai8x.fold(2)` 로 입력을 (4,64,64)로
        #    인터레이스 접는 것이다 (ai8x-synthesis README 의 Data Folding).
        #    `docs/results/g8-first-comparison.md` 4.1 에 적었다.
        self.conv1 = (
            ai8x.FusedAvgPoolConv2dReLU(num_channels, ch(32), 3, pool_size=2,
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
        # `pool6=False` 는 **프레임이 32개 미만인 입력**(2차 후보: 32멜 × 16·20프레임)
        # 용이다. 풀링이 5단이면 16프레임이 16→8→4→2→1→0 으로 사라지므로 마지막
        # 단의 풀링 하나를 뺀다 (4단). 가중치 모양은 그대로다 (1×1 conv).
        self.conv6 = (
            ai8x.FusedMaxPoolConv2dReLU(ch(128), ch(128), 1, pool_size=2,
                                        pool_stride=2, stride=1,
                                        padding=0, bias=bias, **kwargs)
            if pool6 else
            ai8x.FusedConv2dReLU(ch(128), ch(128), 1, stride=1,
                                 padding=0, bias=bias, **kwargs))
        # 128×2×2 → flatten 512 (입력 64×64 일 때).
        #
        # ⚠️ flatten 길이를 **실제로 통과시켜 구한다** (2026-10-02). 입력이
        #    64×64 가 아닌 변형(프레임 32·40, 멜 32 — "정확도 대 에너지 곡선")에서
        #    FC 입력이 달라진다: 64×32 → 256, 32×32 → 128, 64×40 → 256.
        #    conv 만 통과시키므로 난수를 쓰지 않는다 — 기존 구성(64×64 → 512)의
        #    초기화·재현성은 그대로다.
        flat = self._flatten_len(num_channels, dimensions, pool_first)
        self.fc = ai8x.Linear(flat, num_classes, bias=bias,
                              wide=True, **kwargs)

    def _flatten_len(self, num_channels, dimensions, pool_first):
        """conv 스택에 0 을 한 번 흘려 FC 입력 길이를 구한다 (파라미터 없음)."""
        if isinstance(dimensions, (tuple, list)) and len(dimensions) == 2:
            h, w = int(dimensions[0]), int(dimensions[1])
        else:
            h = w = 128 if pool_first else 64
        with torch.no_grad():
            x = torch.zeros(1, num_channels, h, w)
            for m in (self.conv1, self.conv2, self.conv3, self.conv4,
                      self.conv5, self.conv6):
                x = m(x)
        return int(x.numel())

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


def ai85safesoundmelnet_p4(pretrained=False, **kwargs):
    """풀링 4단 판 — 프레임이 32개 미만인 입력용 (2차 후보 u1000 / u1000f1024 / u800).

    `ai85safesoundmelnet` 에서 **conv6 의 MaxPool 하나만** 뺐다. 세 후보가 같은
    구조를 쓴다. 입력 32×16 → FC 256, 32×20 → FC 256.
    """
    assert not pretrained
    return AI85SafeSoundMelNet(pool6=False, **kwargs)


def ai85safesoundmelnet_w050(pretrained=False, **kwargs):
    """채널 0.5× — 모델 규모별 교차점(도전 1순위)의 ① 쪽 점."""
    assert not pretrained
    return AI85SafeSoundMelNet(width_mult=0.5, **kwargs)


def ai85safesoundmelnet_w150(pretrained=False, **kwargs):
    """채널 1.5× — 8bit 가중치 상한 근처. 합성으로 확인할 것."""
    assert not pretrained
    return AI85SafeSoundMelNet(width_mult=1.5, **kwargs)


def ai85safesoundwave2dnet(pretrained=False, **kwargs):
    """wave2D **선행 AvgPool 판** — 남겨 두지만 **기본값이 아니다**.

    ⚠️ 이 판은 2×2 평균이 연속 샘플 축의 4kHz 이상을 버린다. 정보 손실이 없는
    (fold 판)을 쓸 것. 이 진입점은 두 판을 비교해야
    할 때만 남겨 둔다.

    구성 ①과 완전히 같은 2D 구조에 raw 파형을 넣는다.

    입력은 `(1, 128, 128)`, 즉 구성 ④가 쓰는 것과 **같은 파형·같은 접기**에 채널
    축만 붙인 것이다 (`datasets/safesound_wave2d.py`).

    이 구성이 있어야 G8 의 주장이 성립한다. ①(멜 2D) vs ④(파형 1D) 는 표현과
    구조가 동시에 달라 격차의 원인이 분리되지 않는다. 세 점을 놓으면 갈린다.

        ① mel      vs  wave2D     → **표현만** 다르다 (구조 고정)
        wave2D      vs  ④ wave1D  → **구조만** 다르다 (입력 정보 고정)

    ⚠️ **"표현만 다르다" 는 과장이다.** 세 가지가 함께 달라진다.
    (a) 128×128 입력이라 앞에 2×2 AvgPool 이 붙는다 — 채널당 8,192픽셀 한계
        때문에 피할 수 없다(위 `pool_first` 주석). 이 풀링이 연속 샘플 축의
        4kHz 이상을 버리고 8ms 떨어진 프레임 값을 섞는다
    (b) `(128,128)` 접기는 스펙트로그램이 아니다 — 행은 연속 샘플, 열은 8ms
        간격이라 3×3 conv 가 보는 이웃이 물리적으로 인접하지 않는다
    (c) 1D conv(k=1/3/6) 대 2D conv(3×3/1×1) 라는 연산 자체의 차이
    그래서 이 대조의 이름표는 **"구조 변경(선행 풀링 포함)"** 이다.
    정보를 버리지 않는 대안은 `ai8x.fold(2)` 로 입력을 (4,64,64)로 인터레이스
    접는 것이다 (`docs/results/g8-first-comparison.md` 4.1).

    파라미터 수는 ①과 **정확히 같다** (157,535).
    """
    assert not pretrained
    # fold 판과 같은 이유로 setdefault 만 쓴다 (위 주석 참조).
    kwargs.setdefault("dimensions", (128, 128))
    return AI85SafeSoundMelNet(pool_first=True, **kwargs)


def ai85safesoundwave2dfoldnet(pretrained=False, **kwargs):
    """**wave2D (fold) — 권장 기본값.** 입력 `(4, 64, 64)`, 정보 손실 없음.

    `ai85safesoundwave2dnet`(선행 AvgPool 판)의 문제를 없앤 것이다. 풀링판은
    채널당 8,192픽셀 한계를 2×2 평균으로 피했는데, 그 평균이 (128,128) 접기의
    행(연속 샘플) 축에서 **2탭 저역통과 + 2배 데시메이션**이라 **4kHz 이상을
    버렸다**. 유리 파손 같은 광대역 과도음에 직접 불리하다.

    fold 판은 `ai8x.fold(2)` 로 입력을 **인터레이스 접기**만 한다 —
    `1×128×128` → `4×64×64`. 버리는 값이 하나도 없고, 2×2 블록의 4개 위상
    오프셋이 서로 다른 **채널**로 분리되어 3×3 conv 가 함께 본다.
    (ADI 의 Data Folding, `ai8x.py:42`, ai8x-synthesis README.)

    | | 풀링판 | **fold 판** |
    |---|---|---|
    | 입력 | 1×128×128 → 2×2 avg | 1×128×128 → 4×64×64 |
    | 정보 | 4kHz 이상 손실 | **손실 없음** |
    | 1층 | Conv2d(1→32) 288 | Conv2d(4→32) 1,152 |
    | 총 파라미터 | 157,535 (①과 동일) | 158,399 (①의 +0.5%) |
    | 채널당 픽셀 | 4,096 ✅ | 4,096 ✅ |

    ⚠️ 그래도 **"표현만 다르다" 는 아니다.** (a) `(128,128)` 접기 자체가
    스펙트로그램이 아니고(행은 연속 샘플, 열은 8ms 간격), (b) 1D conv 대 2D conv
    라는 연산 차이가 남는다. 이름표는 **"구조 변경, 입력 정보 보존"** 이다.
    """
    assert not pretrained
    # ⚠️ `num_channels`·`dimensions` 는 **setdefault 로만** 둔다. `train.py:794-795`
    #    가 데이터셋의 `input` 에서 뽑아 **항상 명시해** 넘기기 때문이다
    #    (`(4,64,64)` → num_channels 4, dimensions (64,64) — 우리가 원하는 값과
    #    같다). 하드코딩하면 `got multiple values for keyword argument` 로 죽는다.
    #    기본값은 이 진입점을 직접 부르는 경우(평가·KAT)를 위한 것이다.
    kwargs.setdefault("num_channels", 4)
    kwargs.setdefault("dimensions", (64, 64))
    return AI85SafeSoundMelNet(pool_first=False, **kwargs)


class AI85SafeSoundMelTeacher(nn.Module):
    """지식 증류 **교사 어댑터** — 학생(④)의 파형 입력을 받아 멜 CNN 으로 넘긴다.

    ## 왜 어댑터가 필요한가

    distiller 의 `KnowledgeDistillationPolicy.forward(*inputs)` 가
    `self.teacher(*inputs)` 를 부른다 (`knowledge_distillation.py:118`). 즉
    **교사가 학생과 똑같은 입력을 받는다.** 교사는 멜 `(1,64,64)` 이 필요한데
    학생 입력은 파형 `(128,128)` 이므로 그대로는 못 쓴다. 데이터로더가 두 표현을
    동시에 내게 만드는 방법도 있지만 ai8x 의 로더 규약을 고쳐야 하고, 그러면
    학생 쪽 학습 경로가 기준선과 달라진다.

    → 이 모듈이 `forward` 안에서 **파형 → 멜** 변환을 한다. 데이터로더는 ④와
    완전히 같은 것을 쓴다.

    ## 무엇을 하는가 (세 단계)

    1. `ai8x.normalize` 를 **되돌려** int8 파형을 복원한다. 학생 입력은
       `transpose(reshape(-1,128),1,0)` 을 거친 `(128,128)` 이므로 역순으로 편다
    2. torch 로 로그 멜을 만든다 — 상수는 전부 `datasets/melfeat.py` 에서
       가져온다 (창 함수·멜 행렬·dB 기준점·양자화 구간)
    3. 내부 `AI85SafeSoundMelNet` 에 넣어 로짓을 낸다

    ## 정확도

    교사는 `torch.no_grad()` 로만 돌고 학습되지 않으므로(`--kd-teacher-wt 0`),
    numpy 구현과 1 LSB 수준으로 달라도 소프트 타깃에 미치는 영향은 작다.
    그래도 **`tools/kat_teacher.py` 가 torch↔numpy 최대 오차를 재서 기록**한다 —
    "작다" 를 측정 없이 주장하지 않는다.

    ⚠️ 이 모듈은 **학습 보조 장치이고 합성 대상이 아니다.** 실기기에 올라가는
    것은 학생(④)뿐이다. `models` 목록에 등록하는 것은 `--kd-teacher` 가 이름으로
    모델을 찾기 때문이며(`train.py:760`), `ai8xize.py` 로 넘기지 않는다.
    """

    def __init__(self, num_classes=5, num_channels=128, dimensions=(128, 1),
                 bias=False, **kwargs):
        super().__init__()
        # 학생과 같은 인자로 생성되므로(dimensions=(128,1)) 여기서 무시하고
        # 내부 멜 모델은 제 규격으로 만든다
        self.net = AI85SafeSoundMelNet(num_classes=num_classes, num_channels=1,
                                       dimensions=(64, 64), bias=bias, **kwargs)
        self._mel_ready = False
        # ★ **전역 로짓 스케일** (2026-09-28, C 실패로 추가).
        #
        # ① 은 QAT 체크포인트라 `act_mode_8bit` 없이 추론하면 `output_shift`
        # 스케일링이 빠져 로짓이 |값| 평균 1435 규모가 된다. 그 상태로 T=4 를
        # 쓰면 최대확률이 0.998 — 소프트 타깃이 사실상 one-hot 이라 증류가
        # "교사의 하드 라벨" 이 되고, 교사가 틀린 창에서 확신에 찬 오답을
        # 강요한다 (`docs/results/g8-c-distillation.md`).
        #
        # ⚠️ **샘플별 정규화가 아니라 전역 상수 하나로 나눈다.** 샘플 간 확신도
        #    차이(쉬운 창은 뾰족하고 어려운 창은 평평하다)가 증류가 전달하려는
        #    정보의 핵심이다. 샘플마다 정규화하면 그것을 지워 버린다.
        #
        # 값은 `tools/calib_teacher.py` 가 **학습셋에서** 구해 체크포인트에
        # 저장한다. 1.0 이면 스케일링 없음(옛 동작).
        self.register_buffer("logit_scale", torch.tensor(1.0))

    def _build_mel(self, device, dtype):
        """멜 상수를 텐서로 올린다 (한 번만). 값은 `melfeat` 이 유일한 출처다."""
        import sys as _sys

        _sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "datasets"))
        import melfeat as MF

        self.MF = MF
        self.register_buffer("win", torch.tensor(MF.hann(), dtype=dtype,
                                                 device=device),
                             persistent=False)
        self.register_buffer("fb", torch.tensor(MF.mel_filterbank(), dtype=dtype,
                                                device=device),
                             persistent=False)
        self._ref = (float(MF.hann().sum()) / 2.0) ** 2
        self._mel_ready = True

    def forward(self, x):  # pylint: disable=arguments-differ
        """`(B, 128, 128)` 학생 입력 → 멜 → 로짓 `(B, num_classes)`."""
        if not self._mel_ready:
            self._build_mel(x.device, torch.float32)
        MF = self.MF

        # ── 1. normalize 되돌리기 → int8 파형 (B, 16384)
        # 학생 입력은 `x8/256 - 0.5` 를 normalize 가 `(v-0.5)*256` 로 되돌린
        # 상태다. act_mode_8bit 이면 이미 [-128,127] 이고, float 모드면 그것을
        # 128 로 나눈 값이다 (ai8x.py:37). 두 경우를 크기로 가른다.
        w = x if float(x.abs().max()) > 2.0 else x * 128.0
        # (128,128) → 원래 순서: x[r][c] = sample[c*128+r]
        w = torch.transpose(w, 1, 2).reshape(w.size(0), -1)

        # ── 2. 로그 멜 (melfeat 과 같은 순서·상수)
        xp = nn.functional.pad(w.unsqueeze(1) / 128.0, (MF.PAD, MF.PAD),
                               mode="reflect").squeeze(1)
        frames = xp.unfold(1, MF.N_FFT, MF.HOP)[:, :MF.N_FRAMES] * self.win
        spec = torch.fft.rfft(frames, n=MF.N_FFT, dim=2)
        power = (spec.real ** 2 + spec.imag ** 2) / self._ref
        mel = torch.matmul(power, self.fb.t())                 # (B, frames, mels)
        db = 10.0 * torch.log10(mel + MF.EPS)
        q = torch.round((db - MF.TOP_DB) * (255.0 / MF.SPAN_DB)) + 127.0
        q = torch.clamp(q, -128.0, 127.0).transpose(1, 2)      # (B, mels, frames)

        # ── 3. 학생과 같은 스케일 규약으로 멜 모델에 넣는다
        f = q if float(x.abs().max()) > 2.0 else q / 128.0
        # ── 4. 전역 상수로 나눠 소프트 타깃이 실제로 소프트해지게 한다 (위 설명)
        return self.net(f.unsqueeze(1)) / self.logit_scale


def ai85safesoundmelteacher(pretrained=False, **kwargs):
    """C(지식 증류)의 교사. `--kd-teacher ai85safesoundmelteacher` 로 쓴다.

    가중치는 `--kd-resume <① qat_best>` 로 얹는다. 체크포인트의 키가 `net.` 접두사
    없이 저장돼 있으므로 `tools/wrap_teacher.py` 로 접두사를 붙여 변환한다.
    """
    assert not pretrained
    return AI85SafeSoundMelTeacher(**kwargs)


models = [
    {'name': 'ai85safesoundmelnet', 'min_input': 1, 'dim': 2},
    {'name': 'ai85safesoundmelnet_p4', 'min_input': 1, 'dim': 2},
    # 증류 교사 — 학습 보조 장치이고 **합성 대상이 아니다** (dim 은 학생 기준 1)
    {'name': 'ai85safesoundmelteacher', 'min_input': 1, 'dim': 1},
    {'name': 'ai85safesoundmelnet_w050', 'min_input': 1, 'dim': 2},
    {'name': 'ai85safesoundmelnet_w150', 'min_input': 1, 'dim': 2},
    {'name': 'ai85safesoundwave2dnet', 'min_input': 1, 'dim': 2},
    {'name': 'ai85safesoundwave2dfoldnet', 'min_input': 1, 'dim': 2},
]
