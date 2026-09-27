#!/usr/bin/env python3
"""SafeSoundWave2D — raw 파형을 **구성 ①의 2D 구조에** 넣기 위한 로더.

G8 첫 비교에서 ①(멜 2D) 0.673 vs ④(파형 1D) 0.387 이 나왔지만, 둘은 **표현과
구조가 동시에** 다르다 — ④는 KWS20 v3 파생 음성 토폴로지다. 격차의 원인을
가르려면 셋째 점이 필요하다 (`docs/results/g8-first-comparison.md` 4.1).

    ① mel      vs  wave2D     → **표현만** 다르다 (구조 고정)
    wave2D      vs  ④ wave1D  → **구조만** 다르다 (입력 정보 고정)

구현은 한 줄짜리다. `SafeSound` 가 이미 `(128, 128)` 을 내놓으므로 **채널 축만
붙인다**. 파형·창·증강·스케일·분할이 ④와 **완전히 동일**하다 — 같은 샤드를 같은
코드로 읽고, 달라지는 것은 텐서 모양 하나뿐이다.

    ln -s ~/max78000-proj/datasets/safesound_wave2d.py ~/ai8x-training/datasets/
    python train.py --dataset SafeSoundWave2D --model ai85safesoundwave2dnet ...

⚠️ `(128,128)` 접기는 스펙트로그램이 아니다. `kws20.py:558-560` 의
`transpose(reshape(-1,128),1,0)` 이므로 `x[r][c] = sample[c*128 + r]` 다 —
열(c)은 128샘플(8ms) 간격의 굵은 시간축이고, 행(r)은 그 안의 잔 시간축이다.
3×3 conv 가 보는 이웃이 물리적으로 인접한 주파수·시간이 아니라는 뜻이고, 그래서
이 대조는 "완벽한 구조 통제" 가 아니라 **"같은 정보를 ④의 배치 관례로, 구조는
①로"** 다. 논문에 그대로 적는다.
"""

import os
import sys

import ai8x

try:                                   # ai8x-training 이 datasets 패키지로 로드할 때
    from .safesound import CLASSES, SafeSound, V1_TRAIN_COUNTS, class_weights
except ImportError:                    # tools/ 가 단독 모듈로 import 할 때
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from safesound import CLASSES, SafeSound, V1_TRAIN_COUNTS, class_weights

__all__ = ["CLASSES", "SafeSoundWave2D", "SafeSoundWave2DFold",
           "safesound_wave2d_get_datasets", "safesound_wave2dfold_get_datasets",
           "V1_TRAIN_COUNTS"]


class SafeSoundWave2D(SafeSound):
    """`SafeSound` 그대로에 채널 축만 붙인다 — `(128,128)` → `(1,128,128)`.

    `__getitem__` 의 나머지 전부(mmap, shift 재절단, 랜덤 게인, int8 왕복,
    `ai8x.normalize`)를 상속으로 공유한다. 복제하지 않는 것이 핵심이다.
    """

    def __getitem__(self, i):
        x, target = super().__getitem__(i)
        return x.unsqueeze(0), target


class SafeSoundWave2DFold(SafeSoundWave2D):
    """**권장 기본값** — `ai8x.fold(2)` 로 `(1,128,128)` → `(4,64,64)`.

    풀링판은 채널당 8,192픽셀 한계를 2×2 평균으로 피했는데, 그 평균이 접기의
    행(연속 샘플) 축에서 **2탭 저역통과 + 2배 데시메이션**이라 **4kHz 이상을
    버렸다** — 유리 파손 같은 광대역 과도음에 직접 불리하다.

    fold 는 **인터레이스 부분표본**이다. 버리는 값이 하나도 없고, 2×2 블록의
    4개 위상 오프셋이 서로 다른 채널로 분리되어 3×3 conv 가 함께 본다.
    ADI 가 큰 입력에 권하는 기법이고 (`ai8x.py:42`), 채널당 64×64 = 4,096픽셀로
    HWC 한계 8,192 안에 들어간다.

    ⚠️ `ai8x.fold` 는 `img[:, i::2, j::2]` 를 `(i,j)` 순서로 이어 붙인다. 즉
    채널 순서가 `(0,0) (0,1) (1,0) (1,1)` 이다. 펌웨어도 같은 순서로 넣어야
    하므로 KAT 벡터로 고정한다.
    """

    def __getitem__(self, i):
        x, target = super().__getitem__(i)      # (1, 128, 128)
        return ai8x.fold(fold_ratio=2)(x), target


def safesound_wave2d_get_datasets(data, load_train=True, load_test=True):
    """ai8x-training 규약 로더. 파형 구성과 **같은 샤드**를 읽는다."""
    (data_dir, args) = data
    root = os.path.join(data_dir, "SafeSound")
    transform = ai8x.normalize(args=args)

    train_ds = (SafeSoundWave2D(root, "train", transform=transform)
                if load_train else None)
    test_ds = (SafeSoundWave2D(root, "test", transform=transform)
               if load_test else None)
    return train_ds, test_ds


def safesound_wave2dfold_get_datasets(data, load_train=True, load_test=True):
    """fold 판 로더 — **이쪽이 기본값이다** (정보 손실 없음)."""
    (data_dir, args) = data
    root = os.path.join(data_dir, "SafeSound")
    transform = ai8x.normalize(args=args)

    train_ds = (SafeSoundWave2DFold(root, "train", transform=transform)
                if load_train else None)
    test_ds = (SafeSoundWave2DFold(root, "test", transform=transform)
               if load_test else None)
    return train_ds, test_ds


# 클래스 가중치는 파형·멜 구성과 **완전히 같은 값**을 쓴다 (같은 샤드라 수량도 같다).
datasets = [
    {
        # 선행 AvgPool 판 — 남겨 두지만 기본값이 아니다 (4kHz 이상 손실)
        "name": "SafeSoundWave2D",
        "input": (1, 128, 128),
        "output": tuple(CLASSES),
        "weight": class_weights(),
        "loader": safesound_wave2d_get_datasets,
    },
    {
        # **권장 기본값** — 인터레이스 접기, 정보 손실 없음
        "name": "SafeSoundWave2DFold",
        "input": (4, 64, 64),
        "output": tuple(CLASSES),
        "weight": class_weights(),
        "loader": safesound_wave2dfold_get_datasets,
    },
]
