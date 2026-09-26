#!/usr/bin/env python3
"""SafeSoundMel 데이터로더 — G8 구성 ① (로그 멜 + 2D CNN).

구성 ④(raw 파형 1D CNN, `safesound.py`)와 **같은 int8 샤드를 읽는다.** 새 전처리
산출물을 만들지 않는다 — 같은 창, 같은 원본 단위 분할, 같은 필터·채움·오버라이드.
달라지는 것은 마지막 한 단계, **입력 표현**뿐이다. 그래야 "전처리 위치가 정확도에
미치는 영향" 이라는 결론이 성립한다 (`docs/results/config1-mel2d-design.md`).

    ln -s ~/max78000-proj/datasets/safesound_mel.py ~/ai8x-training/datasets/
    ln -s ~/max78000-proj/datasets/melfeat.py       ~/ai8x-training/datasets/
    python train.py --dataset SafeSoundMel --model ai85safesoundmelnet ...

── 왜 특징을 미리 계산하지 않고 즉석에서 만드는가 ─────────────────────────
증강이 **파형 영역**에 정의돼 있기 때문이다. 게인 ±12dB 는 로그 영역에서 상수
덧셈처럼 보이지만 실제로는 아니다 — 피크 클램프와 int8 포화가 비선형이다.
미리 계산한 멜에 상수를 더하면 **큰 소리에서 구성 ④와 학습 분포가 갈라지고**,
그러면 "입력 표현만 다른 비교" 가 깨진다.

비용은 창당 약 1ms 다. `--workers 4` 기준 에폭당 3~4초이고 GPU 학습과 겹쳐
사라진다. 증강이 없는 **테스트셋은 캐시**해서(`--mel-cache`) 반복 평가·스윕·
스트리밍이 파형 경로와 같은 속도로 돌게 한다.

⚠️ shift 해상도는 구성 ④와 다르다. ④는 1샘플 단위이고 여기서는 멜 프레임
격자(16ms) 로 반올림되지 않는다 — **파형에서 먼저 자르므로 샘플 단위 그대로다.**
프레임 경계와 이벤트의 상대 위상이 shift 마다 달라지는 것이 오히려 정상이고,
미리 계산 방식이었다면 16ms 격자에 묶였을 것이다. 즉석 계산의 부수 이득이다.
"""

import os
import sys

import numpy as np
import torch

import ai8x

try:                                   # ai8x-training 이 datasets 패키지로 로드할 때
    from . import melfeat as MF
    from .safesound import (CLASSES, MARGIN, SafeSound, V1_TRAIN_COUNTS,
                            class_weights)
except ImportError:                    # tools/ 가 단독 모듈로 import 할 때
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import melfeat as MF
    from safesound import (CLASSES, MARGIN, SafeSound, V1_TRAIN_COUNTS,
                           class_weights)

__all__ = ["CLASSES", "SafeSoundMel", "safesound_mel_get_datasets",
           "V1_TRAIN_COUNTS", "MF"]


class SafeSoundMel(SafeSound):
    """파형 로더를 그대로 상속하고 **입력 표현만 바꾼다**.

    `SafeSound.__getitem__` 의 앞부분(mmap → 재절단 shift → 랜덤 게인 → int8)을
    그대로 쓰고, 뒤의 `(128,128)` reshape 대신 로그 멜을 만든다. 증강 코드를
    복제하지 않는 것이 핵심이다 — 복제하면 두 구성이 조용히 갈라진다.

    인자 (상위 클래스 외):
      cache   테스트셋 특징을 담을 .npy 경로. 주면 첫 접근 때 만들고 이후 읽는다.
              증강이 걸린 학습셋에는 **쓰면 안 된다** (매 에폭 값이 달라야 한다)
    """

    def __init__(self, *args, cache=None, **kwargs):
        super().__init__(*args, **kwargs)
        if cache and self.augment:
            raise ValueError("증강이 켜진 split 에 캐시를 쓰면 안 된다 — "
                             "매 에폭 달라야 할 값이 고정된다")
        self.cache_path = cache
        self._cache = None

    def _load_cache(self):
        """워커별로 mmap 을 연다 (샤드와 같은 이유 — fork 후 새로 연다)."""
        if self._cache is None and self.cache_path:
            if os.path.isfile(self.cache_path):
                self._cache = np.load(self.cache_path, mmap_mode="r")
            else:
                self._cache = False        # 없음 — 즉석 계산으로 간다
        return self._cache if self._cache is not False else None

    def build_cache(self, path=None, verbose=True):
        """무증강 특징을 통째로 만들어 저장한다. 평가 경로 전용이다."""
        if self.augment:
            raise ValueError("증강 split 은 캐시할 수 없다")
        path = path or self.cache_path
        n = len(self)
        out = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.int8, shape=(n, MF.N_MELS, MF.N_FRAMES))
        for i in range(n):
            out[i] = self._features(i)
            if verbose and (i + 1) % 2000 == 0:
                print(f"  캐시 {i+1:,}/{n:,}", flush=True)
        out.flush()
        self._cache = None
        return path

    def _features(self, i):
        """int8 로그 멜 (N_MELS, N_FRAMES). 파형 경로와 증강을 공유한다."""
        target, shard, row, left, right = self.index[i]
        stored = np.asarray(self._shard(shard)[row])
        rng = np.random.default_rng()
        w = self._crop(stored, left, right, rng)
        if self.augment:
            w = self._rand_gain(w, rng)
            w = self._mix_noise(w, rng)
        return MF.log_mel_int8(w)

    def __getitem__(self, i):
        target = self.index[i][0]
        c = self._load_cache()
        f = np.asarray(c[i]) if c is not None else self._features(i)

        # int8 [-128,127] → [0,1) → ai8x.normalize 가 [-128,127] 로 되돌린다.
        # 파형 경로(safesound.py)와 **같은 관례**다. 여기서 평균·분산 정규화를
        # 끼워 넣으면 실기기 경로와 어긋난다 (CLAUDE.md 7장).
        x = torch.from_numpy(np.ascontiguousarray(f).astype(np.int16)) + 128
        x = (x.float() / 256.0).unsqueeze(0)          # (1, N_MELS, N_FRAMES)
        if self.transform is not None:
            x = self.transform(x)
        return x, target


def safesound_mel_get_datasets(data, load_train=True, load_test=True):
    """ai8x-training 규약 로더. `data` 는 (data_dir, args)."""
    (data_dir, args) = data
    root = os.path.join(data_dir, "SafeSound")     # 파형과 **같은 샤드**를 읽는다
    transform = ai8x.normalize(args=args)

    train_ds = (SafeSoundMel(root, "train", transform=transform)
                if load_train else None)
    # 테스트셋은 무증강이므로 캐시해 둔다. 없으면 즉석 계산으로 돌아간다.
    cache = os.path.join(root, "test", f"melcache_{MF.N_MELS}x{MF.N_FRAMES}.npy")
    test_ds = (SafeSoundMel(root, "test", transform=transform,
                            cache=cache if os.path.isfile(cache) else None)
               if load_test else None)
    return train_ds, test_ds


# 클래스 가중치는 **파형 구성과 완전히 같은 값**을 쓴다. 여기서 따로 계산하면
# 두 구성의 손실이 달라져 비교가 오염된다 (같은 샤드라 수량도 같다).
datasets = [
    {
        "name": "SafeSoundMel",
        "input": (1, MF.N_MELS, MF.N_FRAMES),
        "output": tuple(CLASSES),
        "weight": class_weights(),
        "loader": safesound_mel_get_datasets,
    },
]
