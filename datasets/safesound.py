#!/usr/bin/env python3
"""SafeSound 데이터로더 — `prepare_safesound.py` 가 만든 int8 샤드를 읽는다.

`ai8x-training/datasets/safesound.py` 로 심볼릭 링크해서 쓴다 (CLAUDE.md 11장):
    ln -s ~/max78000-proj/datasets/safesound.py ~/ai8x-training/datasets/safesound.py
    python train.py --dataset SafeSound ...

── 메모리 설계 (CLAUDE.md 6장, OOM 3회의 교훈) ─────────────────────────────
`kws20.py` 는 전체를 메모리에 올린 뒤 concat 해서 로컬(15GB)·Colab(12GB) 양쪽에서
터졌다. 여기서는 **한 샘플도 미리 올리지 않는다.**
  - 인덱스(CSV)만 읽어 (클래스, 샤드, 행) 목록을 만든다. 16,000창 기준 수 MB다
  - 샤드는 `np.load(mmap_mode='r')` 로 매핑만 한다. 실제 페이지는 `__getitem__`
    이 그 행을 건드릴 때 OS 가 올린다
  - 매핑 핸들은 **워커 프로세스마다** 따로 연다 (fork 후 첫 접근 때). mmap 객체를
    부모에서 열어 물려주면 워커 간에 상태가 꼬인다
샤드 하나는 2048 × 16384 int8 = 32MB 이므로, 워커 4개가 각자 몇 개를 열어도
상주 메모리는 수백 MB 를 넘지 않는다.

── 증강 (학습셋 전용, CLAUDE.md 5장 규칙 6) ───────────────────────────────
  - 시간축 shift ±100ms — **재절단**이다. 샤드가 1.2초(STORE)로 저장돼 있어
    가운데 1초를 어디서 자를지만 바꾼다. 순환(roll)은 연속음에서 이음매 클릭을
    만들고 0 으로 미는 것은 전처리에서 없앤 디지털 0 을 되살린다. 클립 가장자리
    창은 여유가 모자랄 수 있어 인덱스의 `left_margin`/`right_margin` 만큼만 민다
  - **랜덤 게인 ±12dB, 전 클래스 동일** (background 포함). 범위를 클래스마다
    달리하면 레벨 분포 자체가 단서가 된다. 근거는 실기기 `SAMPLE_SCALE_FACTOR`
    가 빌드 타임 상수라는 것 — CLAUDE.md 7장
  - 게인 결과가 **int8 에서 비면 다시 뽑는다.** 판정 기준(`--floor-zero`)은
    전처리의 절대 하한과 같은 값이어야 한다. 다르면 하한이 거른 창을 증강이
    되살리거나 그 반대가 된다
  - **MSnoise 혼합은 v1 에 넣지 않는다.** v1 기준선을 학습한 뒤 별도 비교 실험으로
    추가한다 (증강을 한꺼번에 넣으면 어느 것이 효과였는지 분리되지 않는다).
    `_mix_noise` 는 후크만 남겨 두었다

테스트셋은 무증강이다.
"""

import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

import ai8x

CLASSES = ["siren", "glass", "scream", "dog_bark", "background"]
WIN = 16384          # 1초 @ 16kHz — 모델 입력
MARGIN = 1600        # 샤드에 저장된 양옆 여유 (±100ms). shift 재절단용
STORE = WIN + 2 * MARGIN
ROW = 128            # (128,128) reshape

# 전처리와 **반드시 같은 값**이어야 한다 (prepare_safesound.py --floor-zero).
# 게인 증강 후 창이 int8 에서 비었는지 판정하는 기준이다.
FLOOR_ZERO = 0.95
GAIN_DB = 12.0       # 랜덤 게인 범위 ±dB
SHIFT_MS = 100       # 시간축 shift 범위 ±ms
# ── 음량 정규화 (라운드 6 (b) 1단계, 2026-09-30) ─────────────────────────
# CLAUDE.md 7장 개정 규칙: **학습과 펌웨어에 비트 단위로 같은 구현**일 때만
# 허용하고 **적용 위치**를 명시한다. 여기는 **int8 변환 _후_** 다.
#
# ⚠️ **정밀도 이득은 없다.** x 가 이미 int8 이라 게인은 양자화 계단을 함께
# 키울 뿐이고, int8 로 깎이며 잃은 정보는 돌아오지 않는다. 이 단계가 검증하는
# 것은 **음량 불변성 가설뿐**이다. 결과를 "정규화의 효과" 라고 쓰지 말 것 —
# 개선이 나오면 그건 하한이다 (docs/results/g8-round6-design.md b.2.5).
#
# **2의 거듭제곱만 쓴다.** int8 값에 2^n 을 곱하는 것은 정확한 시프트라
# 반올림 오차가 없다. 임의 게인은 int8 위에서 한 번 더 깎이므로 int16 샤드가
# 생기는 2단계로 미룬다.
#
# 펌웨어 대응 (MicReadChunk 뒤, NPU 입력 직전):
#     peak = max |x[n]|
#     sh   = 0..NORM_GMAX_SH 중 (peak << (sh+1)) <= NORM_TARGET 인 최대
#     y[n] = __SSAT(x[n] << sh, 8)
NORM_TARGET = 96        # 풀스케일 127 대비 -2.4dB. 포화 여유
NORM_GMAX_SH = 2        # 최대 시프트 2 = x4 = +12dB. 조용한 창의 잡음 증폭 상한

GAIN_TRIES = 8       # 재추출 횟수 상한


class SafeSound(Dataset):
    """int8 샤드 lazy 로더.

    인자:
      root      `prepare_safesound.py --out` 경로 (기본 data/processed/safesound)
      d_type    'train' 또는 'test'
      transform ai8x.normalize 등
      augment   학습 증강 적용 여부 (test 는 무조건 False)
      noise_dir MSnoise wav 디렉터리 (미연결, TODO)
    """

    def __init__(self, root, d_type, transform=None, augment=None,
                 floor_zero=FLOOR_ZERO, gain_db=GAIN_DB, shift_ms=SHIFT_MS,
                 noise_dir=None, seed=0, norm_pow2=False,
                 subset_frac=None, clip_prob=0.0):
        if d_type not in ("train", "test"):
            raise ValueError(f"d_type 은 train/test 여야 한다: {d_type}")
        self.root = root
        self.d_type = d_type
        self.transform = transform
        self.augment = (d_type == "train") if augment is None else augment
        self.floor_zero = floor_zero
        # 음량 정규화 (기본 꺼짐). 켜면 int8 변환 **후**에 적용된다.
        # 게인 증강에서 피크 초과 시 **포화**로 처리할 확률.
        # 0.0 = 항상 되돌리기(기존), 1.0 = 항상 포화, 0.5 = 혼합
        self.clip_prob = float(clip_prob)
        self.norm_pow2 = bool(norm_pow2)
        self.norm_target = NORM_TARGET
        self.norm_gmax_sh = NORM_GMAX_SH
        self.norm_sh_hist = [0] * (NORM_GMAX_SH + 1)   # 게인 분포 보고용
        self.gain_db = gain_db
        self.shift = max(1, int(16000 * shift_ms / 1000))
        self.noise_dir = noise_dir
        self.seed = seed

        # ── 클래스별 원본 부분집합 (학습 곡선용, V-4) ────────────────
        # `subset_frac={"siren": 0.5}` 처럼 준다. **원본 ID(fsid) 단위**로
        # 자른다 — 창 단위로 줄이면 같은 원본이 흩어져 누수와 같은 효과가
        # 난다 (CLAUDE.md 5장 규칙 1).
        #
        # ⚠️ **포함 관계를 지킨다**: 25% ⊂ 50% ⊂ 75% ⊂ 100%. fsid 를
        # 결정적 해시로 [0,1) 에 사상하고 그 값이 frac 미만인 것만 남긴다 —
        # frac 을 키우면 이전 집합이 그대로 포함된다. 무작위 표본추출이면
        # 점마다 다른 원본이 뽑혀 곡선이 표본 차이와 섞인다.
        self.subset_frac = dict(subset_frac or {})
        self.subset_kept = {}
        self.index = []          # (target, 샤드 경로, 행, 왼쪽 여유, 오른쪽 여유)
        self.meta = []           # (clip_id, fsid) — 분석·디버깅용
        for target, cls in enumerate(CLASSES):
            d = os.path.join(root, d_type, cls)
            idx_path = os.path.join(d, "index.csv")
            if not os.path.isfile(idx_path):
                continue
            frac = self.subset_frac.get(cls)
            kept, seen = set(), set()
            with open(idx_path, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if frac is not None:
                        seen.add(r["fsid"])
                        if _fsid_unit(r["fsid"]) >= frac:
                            continue
                        kept.add(r["fsid"])
                    shard = os.path.join(d, f"shard_{int(r['shard']):04d}.npy")
                    self.index.append((target, shard, int(r["row"]),
                                       int(r.get("left_margin", MARGIN)),
                                       int(r.get("right_margin", MARGIN))))
                    # start_sample 은 평가에서 **클립 내 창 순서**로 쓴다
                    # (tools/eval_confusion.py 의 N프레임 다수결)
                    self.meta.append((r["clip_id"], r["fsid"],
                                      int(r["start_sample"])))
            if frac is not None:
                self.subset_kept[cls] = (len(kept), len(seen))
                print(f"  [부분집합] {cls} 원본 {len(kept)}/{len(seen)} "
                      f"({frac:.0%} 목표)")
        if not self.index:
            sys.exit(f"[에러] 샤드가 없다: {root}/{d_type}. "
                     "먼저 prepare_safesound.py 를 실행할 것.")
        self._maps = {}          # 워커별 mmap 핸들 (fork 후 새로 연다)
        self._pid = None

    def __len__(self):
        return len(self.index)

    def _shard(self, path):
        """샤드 mmap 핸들. 프로세스가 바뀌면(fork) 새로 연다."""
        pid = os.getpid()
        if self._pid != pid:
            self._maps = {}
            self._pid = pid
        m = self._maps.get(path)
        if m is None:
            m = np.load(path, mmap_mode="r")
            self._maps[path] = m
        return m

    # ────────────────────────────────────────────────────────────── 증강
    def _crop(self, row, left, right, rng):
        """STORE 행에서 1초를 잘라낸다. 학습이면 여유 안에서 위치를 흔든다.

        이것이 시간축 shift 다 — 파형을 미는 게 아니라 **자르는 위치**를 바꾼다.
        순환 시프트는 연속음(사이렌 등)에서 이음매 클릭을 만들고, 그 클릭은
        전 클래스에 고르게 퍼지지 않아 인공 단서가 된다.
        """
        if not self.augment:
            return row[MARGIN:MARGIN + WIN]
        lo = MARGIN - min(left, self.shift)
        hi = MARGIN + min(right, self.shift)
        off = int(rng.integers(lo, hi + 1)) if hi > lo else MARGIN
        return row[off:off + WIN]

    def _norm_pow2(self, w):
        """2의 거듭제곱 음량 정규화. int8 -> int8, **시프트뿐이라 오차 없음**.

        (시프트량, 결과) 를 돌려준다. 시프트량은 게인 분포 보고용이다
        (`tools/norm_gain_stats.py`).

        감쇠는 하지 않는다 (sh >= 0). 큰 소리를 줄여도 이미 포화된 정보가
        돌아오지 않고, 줄여야 할 만큼 크면 그건 클리핑 문제이지 정규화
        문제가 아니다.
        """
        peak = int(np.abs(w.astype(np.int16)).max())
        sh = 0
        while sh < self.norm_gmax_sh and (peak << (sh + 1)) <= self.norm_target:
            sh += 1
        if sh == 0:
            return 0, w
        # 포화까지 펌웨어의 __SSAT(v, 8) 과 같게 둔다. TARGET 96 이라 위
        # 조건상 넘지 않지만, 경계와 펌웨어 일치를 위해 남긴다.
        y = np.clip(w.astype(np.int16) << sh, -128, 127)
        return sh, y.astype(np.int8)

    def _rand_gain(self, w, rng):
        """랜덤 게인 ±gain_db. 클리핑을 만들지 않고, int8 에서 비면 다시 뽑는다.

        반환: int8 배열. `GAIN_TRIES` 번 안에 조건을 못 맞추면 원본을 그대로 쓴다
        (창을 버리는 것은 데이터 손실이고, 원본은 이미 하한을 통과했다).
        """
        f = w.astype(np.float32)
        for _ in range(GAIN_TRIES):
            g = 10.0 ** (float(rng.uniform(-self.gain_db, self.gain_db)) / 20.0)
            y = f * g
            peak = float(np.abs(y).max())
            # 피크가 넘을 때 두 가지 처리가 있고, 실기기와 학습이 다르다.
            #   되돌리기 — 창 전체를 줄인다. 파형 모양 보존, 레벨만 하락
            #   포화     — 피크만 자른다. 레벨 유지, 모양 왜곡 ← **실기기**
            # 기본은 되돌리기였고(기존 실행과의 비교를 위해 유지), `clip_prob`
            # 로 포화를 섞는다. 실측: 포화 판 +12dB 에서 macro-F1 이 되돌리기
            # 판보다 ① -0.044 / (a) -0.035 / D-1 -0.011 낮다 — 학습이 포화를
            # 못 본 탓으로 보인다 (docs/results/g8-round6-design.md 부록).
            if peak > 127.0 and rng.random() >= self.clip_prob:
                y *= 127.0 / peak
            q = np.clip(np.round(y), -128, 127)   # 포화 (wraparound 아님)
            if float((q == 0).mean()) <= self.floor_zero:
                return q.astype(np.int8)
        return w

    def _mix_noise(self, w, rng):
        """MSnoise 혼합 (SNR 0~20dB). **v1 에서는 쓰지 않는다** (CLAUDE.md 규칙 6).

        v1 기준선 학습 뒤 별도 비교 실험으로 붙인다. 붙일 때는 ai8x-training 의
        MSnoise 전처리 결과를 16kHz 창으로 변환해 `noise_dir` 에 넣고, 채움 베드와
        같은 규칙(같은 split 안에서만)을 적용한다.
        """
        return w

    # ──────────────────────────────────────────────────────────── 샘플
    def __getitem__(self, i):
        target, shard, row, left, right = self.index[i]
        stored = np.asarray(self._shard(shard)[row])      # int8, (STORE,)

        rng = np.random.default_rng()
        w = self._crop(stored, left, right, rng)          # (WIN,)
        if self.augment:
            w = self._rand_gain(w, rng)
            w = self._mix_noise(w, rng)

        # 음량 정규화는 **증강 뒤**다. 랜덤 게인은 실기기의 레벨 변동을 흉내
        # 내는 것이고, 실기기에서도 정규화는 그 변동을 받은 뒤에 일어난다.
        # 앞에 두면 정규화가 게인을 되돌려 증강이 무의미해진다.
        # ⚠️ **테스트셋에도 적용된다** — 증강이 아니라 전처리이고, 펌웨어가
        #    하는 일이므로 평가 경로에도 있어야 한다.
        if self.norm_pow2:
            sh, w = self._norm_pow2(w)
            self.norm_sh_hist[sh] += 1

        return self._to_tensor(w), target

    def _to_tensor(self, w):
        """int8 창 (WIN,) → 모델 입력 텐서 (128,128)."""
        # int8 [-128,127] → [0,1) → ai8x.normalize 가 다시 [-128,127] 로 되돌린다.
        # kws20.py 와 같은 관례다: 거기서는 uint8 로 저장해 `inp /= 256`
        # (datasets/kws20.py:592) 한 뒤 `ai8x.normalize` (ai8x.py:29, 즉
        # `img.sub(0.5).mul(256.).round().clamp(-128,127)`) 를 태운다.
        # 우리는 int8 로 저장하므로 +128 을 먼저 더해 같은 [0,1) 구간으로 맞춘다.
        x = (torch.from_numpy(np.ascontiguousarray(w).astype(np.int16)) + 128)
        x = x.float() / 256.0
        # (128,128) 변환도 kws20.py 와 동일하다 — `__reshape_audio`
        # (datasets/kws20.py:558-560): `torch.transpose(audio.reshape((-1,128)), 1, 0)`.
        # 즉 16384 를 128행씩 끊어 (128,128) 로 만든 뒤 전치한다. 전치를 빼먹으면
        # 시간축과 채널축이 뒤바뀌어 합성 결과와 어긋난다.
        x = torch.transpose(x.reshape((-1, ROW)), 1, 0)   # (128,128)
        if self.transform is not None:
            x = self.transform(x)
        return x


# ── 음량 정규화 2단계 — int8 변환 **전** (라운드 6 (b) 2단계, 2026-10-02) ──
# 샤드가 int16 (실기기 스케일, prepare_safesound.py --store-dtype int16) 이다.
# 1단계와 달리 **정밀도가 보존된다**: 조용한 창을 키운 뒤에 int8 로 줄이므로
# 양자화 계단이 함께 커지지 않는다. 또 int8 포화점(int16 ±8,128) 위 12dB 가
# int16 에 살아 있어 **큰 소리는 줄여서** 포화를 피한다 (1단계는 증폭만 했다).
#
# 연산 (학습·평가·펌웨어 공통, **정수만** 쓴다 — 비트 단위 재현):
#     peak = max |x16[n]|                         n = 0..16383
#     G    = clamp( (TARGET << 16) / max(peak,1), 1, GMAX << 10 )   # Q16, 정수 나눗셈
#     y[n] = sat8( (x16[n] * G + 32768) >> 16 )
# 기준(정규화 없음)은 `>> 6` 이므로 G = 1<<10 이 게인 1 이다. `x16 * G` 는
# peak·G <= TARGET<<16 이라 int32 를 넘지 않는다.
N16_TARGET = 96         # 정규화 후 int8 피크. 1단계와 같은 값 (-2.4dB)
N16_Q = 16
N16_UNITY = 1 << 10     # Q16 에서 `>> 6` 과 같은 게인


def int16_to_int8(x16):
    """정규화 없는 기준 변환 — `sat8((x + 32) >> 6)`. int8 0 비율 판정용."""
    return np.clip((x16.astype(np.int32) + 32) >> 6, -128, 127).astype(np.int8)


def norm16(x16, gmax):
    """int16 창 → (Q16 게인, 정규화된 int8 창). 위 주석의 연산 그대로."""
    x = x16.astype(np.int64)
    peak = int(np.abs(x).max())
    g = (N16_TARGET << N16_Q) // max(peak, 1)
    g = max(1, min(g, int(gmax) * N16_UNITY))
    y = (x * g + (1 << (N16_Q - 1))) >> N16_Q
    return g, np.clip(y, -128, 127).astype(np.int8)


class SafeSound16(SafeSound):
    """int16 샤드 + int8 변환 전 음량 정규화. 인덱스·증강 규칙은 SafeSound 와 같다.

    `gmax` 는 조용한 창에 거는 최대 증폭(기준 대비 배수)이다.
      4  — 1단계와 같은 상한 (+12dB). 조용한 방의 잡음을 덜 키운다
      64 — int16 정밀도가 허락하는 한 끝까지 (+36dB). kws20.py 의 peak 정규화에
           가장 가까운, **창 하나만 보고 재현 가능한** 형태
    """

    def __init__(self, root, d_type, transform=None, augment=None, gmax=4, **kw):
        super().__init__(root, d_type, transform=transform, augment=augment, **kw)
        self.gmax = int(gmax)

    def _rand_gain16(self, w, rng):
        """랜덤 게인 ±gain_db 를 **int16 위에서** 건다.

        int8 판(`_rand_gain`)과 다른 점: 피크가 int8 포화점을 넘어도 되돌리지
        않는다 — 실기기에서 그 소리는 int16 에 그대로 들어오고 정규화가 줄인다.
        되돌리는 것은 int16 자체(마이크 풀스케일)를 넘을 때뿐이다.
        int8 0 비율 판정은 **정규화 전** 기준 변환으로 한다 (CLAUDE.md 7장
        규칙 3 — 데이터 준비 기준은 정규화 이전 절대 레벨).
        """
        f = w.astype(np.float32)
        for _ in range(GAIN_TRIES):
            g = 10.0 ** (float(rng.uniform(-self.gain_db, self.gain_db)) / 20.0)
            y = f * g
            peak = float(np.abs(y).max())
            if peak > 32767.0:
                y *= 32767.0 / peak
            q = np.clip(np.round(y), -32768, 32767).astype(np.int16)
            if float((int16_to_int8(q) == 0).mean()) <= self.floor_zero:
                return q
        return w

    def __getitem__(self, i):
        target, shard, row, left, right = self.index[i]
        stored = np.asarray(self._shard(shard)[row])      # int16, (STORE,)
        if stored.dtype != np.int16:
            raise TypeError(f"int16 샤드가 아니다: {shard} ({stored.dtype})")

        rng = np.random.default_rng()
        w = self._crop(stored, left, right, rng)
        if self.augment:
            w = self._rand_gain16(w, rng)
        # 정규화는 증강 뒤 — 테스트셋에도 걸린다 (전처리이지 증강이 아니다)
        _, w8 = norm16(w, self.gmax)
        return self._to_tensor(w8), target


def safesound_get_datasets(data, load_train=True, load_test=True,
                           norm_pow2=False, clip_prob=0.0):
    """ai8x-training 규약 로더. `data` 는 (data_dir, args).

    `norm_pow2` 는 라운드 6 (b) 1단계의 음량 정규화다. **테스트셋에도 켠다** —
    증강이 아니라 전처리이고 펌웨어가 하는 일이기 때문이다.
    """
    (data_dir, args) = data
    root = os.path.join(data_dir, "SafeSound")
    transform = ai8x.normalize(args=args)

    # ⚠️ clip_prob 는 **증강**이라 train 에만 건다. 평가·펌웨어 경로는
    #    포화(np.clip) 하나로 고정이다 (2026-09-30 원칙).
    train_ds = (SafeSound(root, "train", transform=transform,
                          norm_pow2=norm_pow2, clip_prob=clip_prob)
                if load_train else None)
    test_ds = (SafeSound(root, "test", transform=transform,
                         norm_pow2=norm_pow2) if load_test else None)
    return train_ds, test_ds


def safesound_mix50_get_datasets(data, load_train=True, load_test=True):
    """게인 증강에서 되돌리기/포화를 **50:50** 으로 섞는다 (학습 전용)."""
    return safesound_get_datasets(data, load_train, load_test, clip_prob=0.5)


def safesound_norm2_get_datasets(data, load_train=True, load_test=True):
    """SafeSoundNorm2 로더 (람다를 쓰지 않는다 — ai8x 가 이름을 로그에 찍는다)."""
    return safesound_get_datasets(data, load_train, load_test, norm_pow2=True)


def _n16_get_datasets(data, load_train, load_test, gmax):
    (data_dir, args) = data
    root = os.path.join(data_dir, "SafeSound16")
    transform = ai8x.normalize(args=args)
    train_ds = (SafeSound16(root, "train", transform=transform, gmax=gmax)
                if load_train else None)
    test_ds = (SafeSound16(root, "test", transform=transform, gmax=gmax)
               if load_test else None)
    return train_ds, test_ds


def safesound_n16g4_get_datasets(data, load_train=True, load_test=True):
    """SafeSoundN16G4 로더 — int8 변환 전 정규화, 최대 증폭 x4."""
    return _n16_get_datasets(data, load_train, load_test, 4)


def safesound_n16g64_get_datasets(data, load_train=True, load_test=True):
    """SafeSoundN16G64 로더 — int8 변환 전 정규화, 최대 증폭 x64."""
    return _n16_get_datasets(data, load_train, load_test, 64)


def _fsid_unit(fsid):
    """fsid → [0,1) 결정적 사상. 부분집합의 **포함 관계**를 만드는 핵심이다.

    `build_class_manifest.py` 가 train/test 를 가를 때 쓴 것과 같은 방식
    (blake2b 앞 8바이트). 시드가 없다 — 같은 fsid 는 언제나 같은 값이다.
    """
    import hashlib
    h = hashlib.blake2b(str(fsid).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def class_weights(root=None, d_type="train", counts=None, power=1.0):
    """역빈도 클래스 가중치 — `nn.CrossEntropyLoss(weight=...)` 로 들어간다.

    `w_c = N / (K · n_c)`. 이 정규화는 **표본당 평균 가중치를 1로** 유지하므로
    손실 크기가 변하지 않는다(학습률을 다시 잡을 필요가 없다).

    배경음만 낮추는 것으로는 부족하다. v1 실측에서 이벤트 클래스끼리도
    dog_bark 1,934 vs glass 477 로 **4배** 차이가 난다 — 가중치를 주지 않으면
    모델이 dog_bark 쪽으로 기울고, 정작 취약한 glass·siren 의 재현율이 낮게
    수렴한다. 5클래스 전부에 건다.

    `root` 를 주면 인덱스 CSV 에서 실측해 계산한다(재집계 후 값 갱신용).
    주지 않으면 v1 상수를 쓴다 — ai8x 는 `datasets` 딕셔너리를 import 시점에
    읽는데 그때 데이터가 없을 수 있어 상수가 필요하다.

    `power` 는 완화 지수다. **기준선은 1.0(순수 역빈도)** 이고, 오탐률이 나쁘면
    `0.5`(제곱근 역빈도)로 낮춘다 — 배경음 가중치가 0.272 에서 0.637 로 올라가
    배경음을 더 배우고 이벤트 재현율을 일부 내준다. 오탐률/재현율 맞교환이므로
    `tools/eval_confusion.py` 의 시간당 오경보 수치를 보고 정한다.
    """
    if counts is None and root:
        counts = []
        for cls in CLASSES:
            idx = os.path.join(root, d_type, cls, "index.csv")
            with open(idx, encoding="utf-8") as f:
                counts.append(sum(1 for _ in csv.DictReader(f)))
    counts = counts or V1_TRAIN_COUNTS
    k = len(counts)
    raw = [(1.0 / c) ** power if c else 0.0 for c in counts]
    # 표본당 평균 가중치를 1로 맞춘다 — 손실 크기가 변하지 않아 학습률을 그대로 쓴다
    s = sum(c * w for c, w in zip(counts, raw)) / sum(counts)
    return tuple(round(w / s, 4) for w in raw)


# v1 실측 train 창 수 (docs/results/label-noise-filter.md B절, 태그 dataset-v1).
# ⚠️ 재집계하면 `python3 -c "import safesound; print(safesound.class_weights('<root>'))"`
# 로 다시 뽑아 아래를 갱신할 것 — 수량이 바뀌었는데 가중치가 그대로면 조용히 틀어진다.
V1_TRAIN_COUNTS = (528, 477, 491, 1934, 9577)   # siren/glass/scream/dog_bark/background

# → siren 4.93 / glass 5.45 / scream 5.30 / dog_bark 1.35 / background 0.27
# background 0.27 은 오탐률을 보고 조정할 파라미터다 — 낮추면 배경음을 덜 배워
# 오탐이 늘고, 높이면 이벤트 재현율이 떨어진다. G7 측정 후 재설정하고, 바꾼 값과
# 그때의 오탐률을 논문에 함께 적는다 (CLAUDE.md 7장).
# 균형 샘플러(WeightedRandomSampler)도 대안이지만 ai8x 의 train.py 가 DataLoader 를
# 직접 만들기 때문에 패치가 필요하다. 가중 손실이 같은 목적을 패치 없이 달성한다.
datasets = [
    {
        "name": "SafeSound",
        "input": (128, 128),
        "output": tuple(CLASSES),
        "weight": class_weights(),
        "loader": safesound_get_datasets,
    },
    {
        # 혼합 증강 — 게인 초과 시 되돌리기/포화 50:50 (학습 전용).
        # 실기기는 항상 포화인데 학습은 되돌리기만 봤다는 불일치를 줄인다.
        "name": "SafeSoundMix50",
        "input": (128, 128),
        "output": tuple(CLASSES),
        "weight": class_weights(),
        "loader": safesound_mix50_get_datasets,
    },
    {
        # 라운드 6 (b) 1단계 — int8 위 2의 거듭제곱 음량 정규화.
        # D-1 과 **정규화 하나만** 다르다. 정밀도 이득은 없고 음량 불변성만
        # 본다 (docs/results/g8-round6-design.md b.2.5).
        "name": "SafeSoundNorm2",
        "input": (128, 128),
        "output": tuple(CLASSES),
        "weight": class_weights(),
        "loader": safesound_norm2_get_datasets,
    },
    {
        # 라운드 6 (b) 2단계 — int16 샤드, int8 변환 **전** 정규화.
        # ④ 와 모델·학습 설정이 같고 **입력 전처리만** 다르다.
        "name": "SafeSoundN16G4",
        "input": (128, 128),
        "output": tuple(CLASSES),
        "weight": class_weights(),
        "loader": safesound_n16g4_get_datasets,
    },
    {
        "name": "SafeSoundN16G64",
        "input": (128, 128),
        "output": tuple(CLASSES),
        "weight": class_weights(),
        "loader": safesound_n16g64_get_datasets,
    },
    {
        # 제곱근 역빈도 가중치 (power=0.5). 데이터·모델·나머지 설정은 전부
        # 같고 **손실 가중치 하나만** 다르다 — 재현율 ↔ 오탐률 맞교환을 본다.
        # 실측: background 0.272 → 0.637, siren 4.93 → 2.71 (이벤트 쪽이 절반으로).
        "name": "SafeSoundW05",
        "input": (128, 128),
        "output": tuple(CLASSES),
        "weight": class_weights(power=0.5),
        "loader": safesound_get_datasets,
    },
]
