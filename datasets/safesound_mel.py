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
    from .safesound import MIX_PROB as S_MIX_PROB
    from .safesound import (CLASSES, MARGIN, SafeSound, V1_TRAIN_COUNTS,
                            class_weights)
except ImportError:                    # tools/ 가 단독 모듈로 import 할 때
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import melfeat as MF
    from safesound import MIX_PROB as S_MIX_PROB
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

    def __init__(self, *args, cache=None, scheme="log", **kwargs):
        super().__init__(*args, **kwargs)
        # 압축 법칙. "log" 가 현행 (1) 이고 나머지는 G8 로그 압축 가설용이다
        # (melfeat.COMP_SCHEMES). 표현 외에는 아무것도 달라지지 않는다.
        if scheme not in MF.COMP_SCHEMES:
            raise ValueError(f"모르는 압축 법칙: {scheme}")
        self.scheme = scheme
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
        n_mels, n_frames = MF.scheme_shape(self.scheme)
        out = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.int8, shape=(n, n_mels, n_frames))
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
            w = self._mix_noise(w, rng, target)
        return MF.mel_int8(w, self.scheme)

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


class SafeSoundMel1D(SafeSoundMel):
    """같은 로그 멜을 **1D 모델용 배치**로 낸다 — (멜 64 = 채널, 프레임 64 = 길이).

    구성 A (`ai85safesoundnet_mel1d`)용이다. 특징 값은 SafeSoundMel 과 **완전히
    같고** 텐서 모양만 (1, 64, 64) → (64, 64) 로 다르다.
    """

    def __getitem__(self, i):
        x, target = super().__getitem__(i)        # (1, N_MELS, N_FRAMES), 변환 적용됨
        return x.squeeze(0), target


def _mel_loader(scheme, subset_frac=None, clip_prob=0.0, cls=None, mix_prob=0.0):
    """압축 법칙 하나에 대한 ai8x-training 규약 로더를 만든다.

    ⚠️ **캐시 파일 이름에 법칙을 넣는다.** 넣지 않으면 (1) 의 로그 캐시를
    선형 판이 그대로 읽어, 표현을 바꾼 줄 알았는데 안 바뀐 실행이 된다.
    """
    def get_datasets(data, load_train=True, load_test=True):
        (data_dir, args) = data
        root = os.path.join(data_dir, "SafeSound")   # 파형과 **같은 샤드**다
        transform = ai8x.normalize(args=args)
        DS = cls or SafeSoundMel

        # ⚠️ 부분집합은 **train 에만** 건다. test 를 줄이면 네 점이
        # 서로 다른 테스트셋을 보게 되어 곡선이 성립하지 않는다.
        train_ds = (DS(root, "train", transform=transform,
                                 scheme=scheme, subset_frac=subset_frac,
                                 clip_prob=clip_prob, mix_prob=mix_prob)
                    if load_train else None)
        # 테스트셋은 무증강이므로 캐시해 둔다. 없으면 즉석 계산으로 돌아간다.
        tag = "" if scheme == "log" else f"_{scheme}"
        n_mels, n_frames = MF.scheme_shape(scheme)
        cache = os.path.join(root, "test",
                             f"melcache{tag}_{n_mels}x{n_frames}.npy")
        test_ds = (DS(root, "test", transform=transform,
                      scheme=scheme,
                      cache=cache if os.path.isfile(cache) else None)
                   if load_test else None)
        return train_ds, test_ds
    get_datasets.__name__ = (f"safesound_mel_{scheme}"
                             f"{'_1d' if cls is SafeSoundMel1D else ''}_get_datasets")
    return get_datasets


# 기존 이름을 유지한다 — 노트북·도구가 이 심볼을 쓴다
safesound_mel_get_datasets = _mel_loader("log")


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
    {
        # 제곱근 역빈도 — 파형 쪽 SafeSoundW05 와 **같은 값**을 쓴다.
        "name": "SafeSoundMelW05",
        "input": (1, MF.N_MELS, MF.N_FRAMES),
        "output": tuple(CLASSES),
        "weight": class_weights(power=0.5),
        "loader": safesound_mel_get_datasets,
    },
]

# ── G8 로그 압축 가설 (2026-09-29) ────────────────────────────────────────
# 해상도 곡선(k=1/2/4)이 전부 잡음 안이었으므로 다음 후보는 압축이다.
# 모델·스케줄·가중치·창 집합이 전부 (1) 과 같고 **압축 법칙 하나만** 다르다.
#
#   MelLin   진폭 선형  (48.1dB) — "로그를 뺀" 판. 이벤트 빈 40.3% 가 바닥
#   MelCbrt  세제곱근   (72.2dB) — 압축은 있으나 로그가 아닌 판
#   MelLog72 로그, top/span 을 cbrt 와 맞춘 판 — **Cbrt 의 대조군**
#   MelLinP  파워 선형  (24.1dB) — 참고용. 이벤트 빈 87.3% 가 바닥이라
#            학습해도 "int8 에 안 담긴다" 는 것만 확인된다
#
# ⚠️ MelCbrt 를 (1) 과 바로 비교하면 **압축 법칙과 담는 범위가 함께** 달라진다.
#    MelLog72 가 그 교란을 없앤다 (같은 범위, 법칙만 다름).
# 근거 수치: docs/results/lin-mel-range.md
# ── siren 학습 곡선 (V-4) ────────────────────────────────────────────────
# ⚠️ 클래스 가중치는 **v1 상수 그대로** 쓴다. class_weights() 가 실측 수량
# 에서 나오므로 siren 을 줄이면 가중치가 자동으로 올라가 **두 가지가 동시에**
# 바뀐다 — 그러면 "원본 수" 만의 효과가 아니게 된다.
for _f in (25, 50, 75):
    datasets.append({
        "name": f"SafeSoundMelSiren{_f}",
        "input": (1, MF.N_MELS, MF.N_FRAMES),
        "output": tuple(CLASSES),
        "weight": class_weights(),          # v1 상수 고정
        "loader": _mel_loader("log", {"siren": _f / 100.0}),
    })

# 혼합 증강 (되돌리기/포화 50:50, 학습 전용)
datasets.append({
    "name": "SafeSoundMelMix50",
    "input": (1, MF.N_MELS, MF.N_FRAMES),
    "output": tuple(CLASSES),
    "weight": class_weights(),
    "loader": _mel_loader("log", clip_prob=0.5),
})

# ①′ — 증분 계산이 되는 프레임 정의 (melfeat.HOP_INC). 모델·스케줄·가중치·
# 창 집합·증강은 ① 과 같고 **프레임 격자만** 다르다.
datasets.append({
    "name": "SafeSoundMelInc",
    "input": (1, MF.N_MELS, MF.N_FRAMES),
    "output": tuple(CLASSES),
    "weight": class_weights(),
    "loader": _mel_loader("loginc"),
})

# 구성 A — ①′ 와 **같은 특징**을 1D 모델용 (채널 = 멜, 길이 = 프레임)으로.
datasets.append({
    "name": "SafeSoundMelInc1D",
    "input": (MF.N_MELS, MF.N_FRAMES),
    "output": tuple(CLASSES),
    "weight": class_weights(),
    "loader": _mel_loader("loginc", cls=SafeSoundMel1D),
})

# "정확도 대 에너지 곡선" 변형 (2026-10-02) — ①′ 의 CPU 전처리를 줄인 프레임
# 정의들 (melfeat.INC_SPECS). 모델은 같은 2D CNN 이고 입력 모양만 다르다
# (FC 입력 길이는 모델이 스스로 계산한다).
for _name, _scheme in (("SafeSoundMelH500", "log_h500"),
                       ("SafeSoundMelH500M32", "log_h500m32"),
                       ("SafeSoundMelM32", "log_m32"),
                       ("SafeSoundMelH400", "log_h400"),
                       # 2차 후보 (초저비용) — 모델은 ai85safesoundmelnet_p4
                       ("SafeSoundMelU1000", "log_u1000"),
                       ("SafeSoundMelU1000F1024", "log_u1000f1024"),
                       ("SafeSoundMelU800", "log_u800")):
    datasets.append({
        "name": _name,
        "input": (1,) + MF.scheme_shape(_scheme),
        "output": tuple(CLASSES),
        "weight": class_weights(),
        "loader": _mel_loader(_scheme),
    })

# v2 — 같은 특징·같은 시험셋, 학습에만 배경 혼합 증강 (dataset-v2-design.md 8절)
for _name, _scheme in (("SafeSoundMelIncV2", "loginc"),
                       ("SafeSoundMelH400V2", "log_h400"),
                       ("SafeSoundMelU1000V2", "log_u1000"),
                       ("SafeSoundMelU800V2", "log_u800")):
    datasets.append({
        "name": _name,
        "input": (1,) + MF.scheme_shape(_scheme),
        "output": tuple(CLASSES),
        "weight": class_weights(),
        "loader": _mel_loader(_scheme, mix_prob=S_MIX_PROB),
    })

for _name, _scheme in (("SafeSoundMelLin", "lin"),
                       ("SafeSoundMelCbrt", "cbrt"),
                       ("SafeSoundMelLog72", "log72"),
                       ("SafeSoundMelLinP", "linp")):
    datasets.append({
        "name": _name,
        "input": (1, MF.N_MELS, MF.N_FRAMES),
        "output": tuple(CLASSES),
        "weight": class_weights(),
        "loader": _mel_loader(_scheme),
    })
