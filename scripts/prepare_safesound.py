#!/usr/bin/env python3
"""매니페스트 → 1초 윈도우 추출 → int8 캐시.

TASKS.md Phase 2.3. 데이터로더(`datasets/safesound.py`)와 **분리**되어 있다
(CLAUDE.md 6장). 이 스크립트는 디스크에 캐시를 만들고, 데이터로더는 읽기만 한다.

── 메모리 설계 (CLAUDE.md 6장, 3회 OOM의 교훈) ────────────────────────────
전체를 메모리에 올리지 않는다. 처음부터 lazy 구조로 간다.
  - 클립 하나씩 읽어 윈도우를 뽑고 **즉시** 샤드에 append 한다
  - 출력은 클래스·split 별 `.npy` 샤드(기본 2048윈도우/샤드) + 인덱스 CSV.
    단일 거대 `.pt` 를 만들지 않는다
  - 진행 상태를 `progress.json` 에 남겨 중단 후 이어하기를 지원한다
샤드 하나는 2048 × 16384 int8 = 32MB 다. 데이터로더는 `np.memmap` 으로 필요한
샤드만 매핑한다.

── 판정 파이프라인 (클립 하나당, `analyze_clip` → `choose`) ────────────────
  0. glass 하위유형 게이트 — 매니페스트 note 가 `glass:glass_only`(Shatter 없이
     Glass 만 보유)이면 클립 최대 onset strength 가 `--glass-onset-thr` 이상일
     때만 glass 로 둔다. 미달은 울림·부딪힘으로 보고 배경음 하드 네거티브로
     보낸다 (5장 규칙 3-1, `--glass-reject`).
  1. 후보 윈도우 생성 — **겹치는 격자** (`--hop-ms`, 기본 250ms).
     1초 비중첩 격자는 0.3초짜리 파손음·짧은 비명이 경계에 걸리면 두 윈도우로
     쪼개져 어느 쪽도 온전하지 않다. 250ms 로 겹쳐 생성하고 마지막에 겹치지 않게
     고른다. 후보는 4배가 되지만 태거는 에너지 통과분만 판정한다.
  2. 에너지 필터 — 절대 하한(`--floor-db`) + 클립 내 상대 기준(`--rel-ratio`) 병용.
     상대 기준의 분모는 그 클립 **후보** 중 최대 RMS 다.
     ⚠️ 절대 RMS 임계값 하나만 전 클래스에 적용하면 glass 가 먼저 무너진다
     (`docs/results/energy-filter-sweep.md` 실측).
  3. 태거 필터 — PANNs Cnn14(16kHz, AudioSet 527) 로 후보 윈도우를 판정해
     우리 라벨과 일치하는 것만 남긴다 (`--tag-thr`, `--tag-thr-cls`, `--tag-bg-thr`).
       이벤트 클래스 : 해당 AudioSet 라벨 묶음(`LABEL_SETS`)의 최대 확률 ≥ 임계값
       background    : 이벤트 라벨 묶음 전체의 최대 확률 < 배경 임계값
     판정 결과는 윈도우 단위로 디스크에 캐시한다(`--tag-cache`). 한 번 돌면
     임계값만 바꿔 다시 보는 데 태거를 재실행하지 않는다.
  4. 최종 선택 — **태거 확률이 높은 순**으로, 서로 겹치지 않게 클립당 최대
     `--max-win` 개 (`--policy tag`, 기본). 같은 소리의 이웃 윈도우를 중복해서
     담지 않으려고 겹침을 금지한다.
       background 는 예외로 RMS 큰 순이다. 이벤트 확률이 낮은 순으로 고르면
       가장 헷갈리지 않는 배경음만 남아 하드 네거티브의 의미가 사라진다.
     `--policy class` : 태거 없이 클래스별 기준 (glass=onset 피크 중심,
       scream=스펙트럴 중심 높은 순, 그 외 RMS). `--policy rms` : 전부 RMS.
       둘 다 대조군용이며 `--no-tagger` 일 때의 기본 동작이기도 하다.
민감도표·보고서·청취 표본·캐시 생성이 **모두 이 두 함수**를 쓴다.

── 리샘플 ──────────────────────────────────────────────────────────────────
`scipy.signal.resample_poly` (안티앨리어싱 FIR 포함). `resample` 참조.

사용법 (WSL2):
    python3 scripts/prepare_safesound.py --sweep          # 에너지 민감도표 (태거 미사용)
    python3 scripts/prepare_safesound.py --filter-report  # 라벨 노이즈 필터 전후 표
    python3 scripts/prepare_safesound.py --export-samples --max-win 1 \\
        --n-samples 30 --per-class 120 \\
        --sample-dir data/interim/listen_v3               # 청취용 표본 (채택분)
    python3 scripts/prepare_safesound.py --export-rejected --n-samples 20 \\
        --sample-dir data/interim/listen_rejected         # 태거가 거른 경계 윈도우
    python3 scripts/prepare_safesound.py                  # 캐시 생성

`--per-class` 는 `--n-samples` 보다 넉넉히 준다. 무음·태거 탈락, 읽기 실패로
표본이 비는 것을 막기 위한 후보 풀이다 (채워지면 나머지는 읽지 않는다).

태거 준비 (WSL2, 1회):
    python3 -m pip install --user librosa panns_inference
    체크포인트 Cnn14_16k_mAP=0.438.pth (Zenodo 3987831, 342MB) 를 ~/panns_data/ 에
    sha256 e2ee543a27919542c2ea03eabaa70b24dcd4e6c8e05621de6b67a94e4c5058e6
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from scipy.signal import resample_poly
except ImportError:
    resample_poly = None

SR = 16000              # 목표 샘플레이트 (CLAUDE.md 4장)
WIN = 16384             # 1초 윈도우 = 16384 샘플 → (128,128) reshape
MAX_WIN_PER_CLIP = 3    # 클립당 윈도우 상한
SHARD = 2048            # 샤드당 윈도우 수 (2048 × 16384 int8 = 32MB)
CLASSES = ["siren", "glass", "scream", "dog_bark", "background"]
EVENTS = CLASSES[:4]
SPLITS = ["train", "test"]

# onset strength 설정. hop 10ms 로 어택 위치를 잡는다 (librosa 기본 hop 32ms 는
# 0.3초짜리 파손음에서 너무 거칠다).
ONSET_HOP = 160
ONSET_KW = dict(sr=SR, hop_length=ONSET_HOP, n_fft=1024, n_mels=64, fmax=SR // 2)

# 클래스 → AudioSet 라벨 묶음 (PANNs class_labels_indices.csv 의 display_name)
# glass 에 `Glass`·`Chink, clink` 를 넣지 않는다 — 울림·부딪힘도 그 라벨로 잡히므로
# 이번 필터의 목적(파손음만 남기기)과 반대로 작동한다.
LABEL_SETS = {
    "siren": ["Siren", "Civil defense siren", "Police car (siren)",
              "Ambulance (siren)", "Fire engine, fire truck (siren)",
              "Emergency vehicle"],
    "glass": ["Shatter", "Breaking", "Smash, crash"],
    "scream": ["Screaming", "Yell", "Shout", "Battle cry", "Children shouting"],
    "dog_bark": ["Bark", "Dog", "Yip", "Bow-wow"],
}

# 온톨로지 상위 라벨. 태거가 최상위로 이것을 내놓으면 실제로는 우리 클래스인데
# 구체 라벨 확률이 분산돼 걸러졌을 수 있다 → 보고서 F 절에서 자식 확률을 본다.
PARENT_LABELS = {
    "dog_bark": ["Animal", "Domestic animals, pets", "Canidae, dogs, wolves"],
    "siren": ["Alarm", "Vehicle"],
}

HOME = os.path.expanduser("~")


# ────────────────────────────────────────────────────────────── 오디오 로드
def load_audio(path):
    """16kHz mono float32 로 읽는다. 실패 시 None."""
    if sf is None:
        sys.exit("[에러] soundfile 미설치. WSL2에서: pip install soundfile")
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as e:                                  # noqa: BLE001
        print(f"  읽기 실패 {os.path.basename(path)}: {e}", file=sys.stderr)
        return None
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = resample(x, sr, SR)
    return x


def resample(x, sr_in, sr_out):
    """다상(polyphase) 리샘플 — `scipy.signal.resample_poly`.

    up/down 을 최대공약수로 약분해 넘긴다 (44.1k→16k = 160/441). scipy 가 Kaiser 창
    FIR 저역통과를 함께 걸어 새 나이퀴스트(8kHz) 위 성분을 제거한다.

    이전 구현(3샘플 이동평균 후 선형보간)은 이동평균의 주파수 응답상 7.35kHz에서
    −3.5dB, 10kHz에서 −7.3dB 밖에 감쇠하지 못해 고주파가 접혀 들어왔다.
    고주파 성분이 많은 glass 에 가장 불리했다.
    US8K 는 파일마다 샘플레이트가 달라 비율을 고정하지 않는다.
    """
    if sr_in == sr_out:
        return x
    if resample_poly is None:
        sys.exit("[에러] scipy 미설치. WSL2에서: pip install scipy")
    g = int(np.gcd(int(sr_in), int(sr_out)))
    return resample_poly(x, sr_out // g, int(sr_in) // g).astype(np.float32)


def pad_to_win(x):
    """WIN 보다 짧은 클립을 0으로 채운다 — `--fill-mode zero` 전용(대조군).

    기본 경로는 `FillerPool` 을 쓰는 `prepare_audio` 다. 아래 설명을 볼 것.
    """
    return np.pad(x, (0, WIN - len(x))) if len(x) < WIN else x


class FillerPool:
    """1초보다 짧은 클립의 빈 구간을 채울 **조용한 배경 구간** 풀.

    0 으로 채우면 두 가지가 깨진다.
      1. 마이크 스트림에는 디지털 0 이 나오지 않는다. 그런 창에 대한 정확도는
         현장 성능을 대변하지 못한다. glass 는 0.3초 과도음이라 test 창의 10%가
         이 상태였다 (`tools/audit_silence.py` 실측).
      2. 0 이 특정 클래스에만 몰리면 "정확한 0 → glass" 지름길이 생긴다
         (`tools/zero_ratio_by_class.py`).
    실배치에서 1초 창에 담기는 것은 **파손음 + 방 안의 암소음**이다. 그대로 만든다.

    ⚠️ **채움용 배경은 같은 split 안에서만 뽑는다.** test 창을 train 배경으로
    채우면 그 자체가 새로운 누수다.
    조용한 구간을 고르되 **디지털 0 이 섞인 구간은 제외**한다 — 0 을 지우려고
    채우면서 다시 0 을 들여오면 의미가 없다.
    """

    def __init__(self, manifest, roots, args):
        self.roots = roots
        self.args = args
        self.rows = defaultdict(list)
        for r in csv.DictReader(open(manifest, encoding="utf-8")):
            if r["cls"] == "background":
                self.rows[r["split"]].append(r)
        self.cache = {}

    def _build(self, split):
        rows = sorted(self.rows.get(split, []), key=lambda r: rank_id(r["clip_id"]))
        segs, used = [], []
        for r in rows:
            if len(segs) >= self.args.fill_pool_clips:
                break
            p = clip_path(r, self.roots)
            if not p:
                continue
            x = load_audio(p)
            if x is None or len(x) < WIN:
                continue
            starts = np.arange(0, len(x) - WIN + 1, WIN, dtype=np.int64)
            rms = window_rms(x, starts)
            for i in np.argsort(rms):            # 조용한 구간부터
                w = x[starts[i]:starts[i] + WIN]
                if float((w == 0.0).mean()) > 0.001:
                    continue                     # 디지털 무음이 섞인 구간은 제외
                if rms[i] <= 0:
                    continue
                segs.append(w.astype(np.float32))
                used.append((r["clip_id"], int(starts[i])))
                break
        if not segs:
            sys.exit(f"[에러] {split} 배경음 채움 구간을 찾지 못했다. "
                     "먼저 download_clips.py --stage background 를 실행할 것.")
        self.cache[split] = (segs, used)
        return self.cache[split]

    def pick(self, split, key):
        """클립 ID 로 결정적으로 하나 고른다 (재실행 시 같은 결과)."""
        segs, used = self.cache.get(split) or self._build(split)
        i = rank_id(key) % len(segs)
        return segs[i], used[i]


def rank_id(s):
    """문자열 → 결정적 정수. 난수 시드 없이 재현 가능한 선택에 쓴다."""
    import hashlib
    return int(hashlib.md5(str(s).encode()).hexdigest(), 16)


def prepare_audio(x, r, args, pool=None):
    """윈도우 추출 전 오디오를 WIN 이상으로 만든다.

    WIN 이상이면 그대로. 짧으면 `--fill-mode` 에 따라 0(대조군) 또는 배경음으로
    채운다. 배경음 채움은 **결정적**이다 — 클립 ID 해시로 구간과 위치를 정하므로
    build / 보고서 / 청취 표본이 모두 같은 파형을 본다.

    합성 방식: 배경 구간을 창 전체에 깔고(bed) 그 위에 이벤트를 더한다.
    구간 밖만 채우면 경계에서 파형이 튀어 클릭이 생기고, 그 클릭이야말로
    모델이 잡기 좋은 인공 단서다. 이벤트 가장자리에는 짧은 페이드를 건다.

    반환: (오디오, 채움 정보 dict 또는 None)
    """
    if len(x) >= WIN:
        return x, None
    if args.fill_mode == "zero" or pool is None:
        return pad_to_win(x), None

    seg, src = pool.pick(r["split"], r["clip_id"])
    n = len(x)
    off = rank_id(r["clip_id"] + ":off") % (WIN - n + 1)
    # 기본은 **배경 구간의 원래 레벨 그대로** 다. 이벤트 RMS 에 맞춰 스케일하면
    # 정규화를 하지 않는다는 절대 레벨 정책(7장)과 어긋난다 — 실기기에서 암소음의
    # 크기는 이벤트 크기와 무관하다. `--fill-db` 를 주면 이벤트 상대 레벨로 맞춘다.
    gain = 1.0
    if args.fill_db is not None:
        ev_rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        bg_rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
        gain = (ev_rms * 10 ** (args.fill_db / 20.0) / bg_rms) if bg_rms > 0 else 0.0
    y = seg * gain

    ev = x.astype(np.float32).copy()
    f = max(1, int(SR * args.fill_fade_ms / 1000))
    if n > 2 * f:                                 # 경계 클릭 방지용 짧은 페이드
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, f, dtype=np.float32))
        ev[:f] *= ramp
        ev[-f:] *= ramp[::-1]
    y = y.copy()
    y[off:off + n] += ev
    bg_db = 20 * np.log10(max(float(np.sqrt(np.mean((seg * gain) ** 2))), 1e-12))
    return np.clip(y, -1.0, 1.0), {
        "src_clip": src[0], "src_start": src[1], "offset": off,
        "event_len": n, "fill_db": args.fill_db, "bg_dbfs": bg_db}


def fill_note(info):
    """채움 정보를 인덱스 CSV 한 칸에 넣을 문자열로."""
    if not info:
        return ""
    lvl = "orig" if info["fill_db"] is None else f"{info['fill_db']:.0f}dBrel"
    return (f"bgfill:{lvl}:{info['bg_dbfs']:.1f}dBFS:off={info['offset']}:"
            f"len={info['event_len']}:src={info['src_clip']}@{info['src_start']}")


# ────────────────────────────────────────────────────────── 후보 윈도우 생성
def onset_env(x):
    import librosa
    return librosa.onset.onset_strength(y=x, **ONSET_KW)


def onset_candidates(env, n):
    """onset 피크를 강한 순으로, 서로 겹치지 않게 윈도우 중심에 둔다.

    반환: (시작점 배열, 점수=onset strength 배열). 클립 끝에서는 시작점을
    [0, n−WIN] 으로 당기므로 피크가 정중앙에서 벗어날 수 있다.
    """
    starts, score = [], []
    for f in np.argsort(-env, kind="stable"):
        if env[f] <= 0:
            break
        s = int(np.clip(f * ONSET_HOP - WIN // 2, 0, n - WIN))
        if all(abs(s - t) >= WIN for t in starts):
            starts.append(s)
            score.append(float(env[f]))
    if not starts:
        starts, score = [0], [0.0]
    return np.array(starts, dtype=np.int64), np.array(score)


def spectral_centroid(w):
    """윈도우 전체의 전력 가중 스펙트럴 중심 (Hz). 무음 프레임이 끼어도
    전력 가중이라 평균을 끌어내리지 않는다."""
    p = np.abs(np.fft.rfft(w.astype(np.float64))) ** 2
    f = np.fft.rfftfreq(len(w), 1.0 / SR)
    return float((f * p).sum() / max(p.sum(), 1e-20))


# 클래스별 태거 일치 기준. **0 이면 그 클래스는 태거를 쓰지 않는다**
# (순위도 태거 확률이 아니라 클래스 기준으로 되돌아간다).
# 값의 근거는 청취 검증 50창 × 4클래스 — `docs/results/listening-verification.md`.
#   glass 0 : 태거 점수가 파손음 여부와 맞지 않는다. 0.02~0.05 구간이 78% 파손음인데
#             0.05~0.10 구간은 31% 로 **순서가 뒤집혀** 있어 임계값으로 해결 불가.
#             PANNs 가 `Shatter` 를 거의 내놓지 않고 `Chink, clink` 로 몰아준다.
#   scream 0.025 : 귀로 확인한 비명이 0.026 까지 내려온다. 0.05 는 너무 높다.
#   siren·dog_bark 0.10 : 청취 30창 중 오답 2 (siren) — 유지.
DEFAULT_TAG_THR = {"siren": 0.10, "glass": 0.0, "scream": 0.025, "dog_bark": 0.10}


def policy_of(cls, args):
    """후보 **생성** 방식. 선택 순서는 `rank_of` 가 정한다."""
    if args.policy == "class" and cls == "glass":
        return "onset"
    if args.policy == "class" and cls == "scream":
        return "centroid"
    return "rms"


def window_rms(x, starts):
    return np.array([float(np.sqrt(np.mean(x[s:s + WIN].astype(np.float64) ** 2)))
                     for s in starts])


# ───────────────────────────────────────────────────────────────── 태거
class Tagger:
    """PANNs Cnn14 (16kHz) 윈도우 판정기. 결과는 클립별 npz 로 캐시한다.

    캐시 키는 (소스, 클립, 시작 샘플) 이다. 리샘플러·윈도우 길이를 바꾸면
    캐시를 지울 것 — 같은 시작점이라도 입력이 달라진다.
    """

    def __init__(self, args):
        try:
            import torch
            from panns_inference.models import Cnn14
        except ImportError:
            sys.exit("[에러] 태거 의존성 미설치. WSL2에서: "
                     "python3 -m pip install --user librosa panns_inference")
        if not os.path.isfile(args.tag_ckpt):
            sys.exit(f"[에러] 체크포인트 없음: {args.tag_ckpt} (모듈 docstring 참조)")
        torch.set_num_threads(args.threads)
        self.torch = torch
        self.model = Cnn14(sample_rate=SR, window_size=512, hop_size=160,
                           mel_bins=64, fmin=50, fmax=SR // 2, classes_num=527)
        ck = torch.load(args.tag_ckpt, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        with open(args.tag_labels, encoding="utf-8") as f:
            self.names = [r["display_name"] for r in csv.DictReader(f)]
        missing = [l for v in LABEL_SETS.values() for l in v if l not in self.names]
        if missing:
            sys.exit(f"[에러] AudioSet 라벨명 불일치: {missing}")
        self.idx = {c: [self.names.index(l) for l in LABEL_SETS[c]] for c in EVENTS}
        self.ev_idx = sorted({i for v in self.idx.values() for i in v})
        self.cache = args.tag_cache
        os.makedirs(self.cache, exist_ok=True)

    def probs(self, key, x, starts):
        """(len(starts), 527) float32. 캐시에 없는 시작점만 추론한다."""
        path = os.path.join(self.cache, key + ".npz")
        have = {}
        if os.path.isfile(path):
            z = np.load(path)
            have = dict(zip(z["starts"].tolist(), z["probs"]))
        miss = [int(s) for s in starts if int(s) not in have]
        if miss:
            X = np.stack([x[s:s + WIN] for s in miss]).astype(np.float32)
            outs = []
            with self.torch.no_grad():
                for i in range(0, len(X), 64):
                    o = self.model(self.torch.from_numpy(X[i:i + 64]))
                    outs.append(o["clipwise_output"].numpy())
            for s, p in zip(miss, np.concatenate(outs)):
                have[s] = p.astype(np.float16)
            ks = sorted(have)
            np.savez(path, starts=np.array(ks, dtype=np.int64),
                     probs=np.stack([have[k] for k in ks]))
        return np.stack([have[int(s)] for s in starts]).astype(np.float32)

    def reduce(self, P, cls):
        """윈도우별 판정 점수. 이벤트: 자기 라벨 묶음 최대 확률.
        background: 이벤트 라벨 묶음 전체의 최대 확률(낮을수록 깨끗)."""
        cols = self.ev_idx if cls == "background" else self.idx[cls]
        return P[:, cols].max(axis=1)

    def parents(self, P, cls):
        """온톨로지 상위 라벨 확률 — 상위 라벨 때문에 걸러졌는지 보려고 남긴다."""
        return {l: P[:, self.names.index(l)] for l in PARENT_LABELS.get(cls, [])}


def clip_key(r):
    return f"{r['source']}_{os.path.splitext(r['clip_id'])[0]}"


# ──────────────────────────────────────────────────── 청취 판정 override
def load_overrides(path):
    """사람이 듣고 내린 윈도우 단위 판정. 자동 판정보다 우선한다.

    열: clip_id, start_sample, action, cls, reason
      keep    태거가 불일치로 판정했어도 채택한다. **에너지 필터는 그대로 적용**한다
              — 무음은 귀로 확인했든 아니든 무음이다
      drop    채택하지 않는다
      relabel `cls` 열의 클래스로 옮긴다
    `cls` 는 keep/drop 행에서는 청취로 확인한 실제 내용(기록용), relabel 행에서는
    목적지 클래스다. `#` 로 시작하는 줄은 주석.

    ⚠️ 키가 (clip_id, start_sample) 이므로 **`--hop-ms` 를 바꾸면 시작점이 달라져
    매칭이 깨진다.** 실행 때마다 매칭 수를 보고하니 확인할 것.
    """
    if not path or not os.path.isfile(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(l for l in f if not l.startswith("#")):
            if r["action"] not in ("keep", "drop", "relabel"):
                sys.exit(f"[에러] 알 수 없는 override action: {r['action']}")
            out[ov_key(r)] = r
    return out


def ov_key(row):
    """override 행 → 조회 키. `start_sample` 이 `*` 또는 빈 칸이면 **그 클립의
    전 윈도우**에 적용한다. 창 하나가 아니라 원본 전체를 버릴 때 쓴다."""
    s = str(row["start_sample"]).strip()
    return (row["clip_id"], "*" if s in ("", "*") else int(s))


def override_masks(r, res, ov):
    """후보별 (강제 채택, 강제 폐기) 마스크와 적용된 행 목록."""
    n = len(res["starts"])
    keep = np.zeros(n, dtype=bool)
    drop = np.zeros(n, dtype=bool)
    hit = []
    if ov:
        for i, s in enumerate(res["starts"]):
            row = ov.get((r["clip_id"], int(s))) or ov.get((r["clip_id"], "*"))
            if not row:
                continue
            hit.append(row)
            if row["action"] == "keep":
                keep[i] = True
            elif row["action"] == "drop":
                drop[i] = True
            else:                                  # relabel
                drop[i] = res["cls"] != row["cls"]
                keep[i] = not drop[i]
    return keep, drop, hit


# ──────────────────────────────────────────────────────────── 클립 판정
def analyze_clip(r, x, args, tagger=None):
    """클립 하나 → 후보 윈도우와 판정 재료. 윈도우 선택은 `choose` 가 한다.

    반환 dict:
      cls      실효 클래스 (glass 게이트 탈락 시 background, 폐기 시 None)
      note     실효 note (게이트 탈락분은 `hard_neg:glass_no_onset`)
      gate     glass 게이트 결과 shatter/pass/reject, 그 외 None
      onset    클립 최대 onset strength (glass 만, 그 외 nan)
      starts, rms, score   후보 윈도우 시작점·RMS·정책 점수
      rank     최종 선택 순서 점수 (클수록 먼저) — `--policy tag` 이면 태거 확률
      energy   에너지 필터 통과 마스크. 태거는 이 마스크 안에서만 돈다
      tag      후보별 태거 점수 (에너지 탈락분과 태거 미사용 시 nan)
      top      후보별 AudioSet 최상위 라벨, topp 그 확률
      own      자기 라벨 묶음의 라벨별 확률 (n, len(LABEL_SETS[cls]))
      par      상위 라벨 확률 dict (siren/dog_bark 만)
    """
    x, fill = prepare_audio(x, r, args, getattr(args, "pool", None))
    cls, note = r["cls"], r.get("note", "")
    gate, onset, env = None, float("nan"), None
    if cls == "glass":
        env = onset_env(x)
        onset = float(env.max())
        if note == "glass:glass_only":
            if onset >= args.glass_onset_thr:
                gate = "pass"
            else:
                gate = "reject"
                cls = "background" if args.glass_reject == "background" else None
                note = "hard_neg:glass_no_onset"
        else:
            gate = "shatter"

    out = dict(cls=cls, split=r["split"], orig=r["cls"], note=note, gate=gate,
               onset=onset, starts=None, rms=None, score=None, rank=None,
               energy=None, tag=None, top=None, topp=None, own=None, par=None,
               onset_win=None, fill=fill)
    if cls is None:
        return out

    pol = policy_of(cls, args)
    if pol == "onset":
        starts, score = onset_candidates(env, len(x))
        rms = window_rms(x, starts)
    else:
        starts = np.arange(0, len(x) - WIN + 1, args.hop, dtype=np.int64)
        rms = window_rms(x, starts)
        score = (np.array([spectral_centroid(x[s:s + WIN]) for s in starts])
                 if pol == "centroid" else rms)
    # glass 는 어택 세기를 윈도우 점수로도 쓴다 (태거를 끈 경우의 순위 기준).
    onset_score = None
    if env is not None and pol != "onset":
        onset_score = np.array([float(env[s // ONSET_HOP:(s + WIN) // ONSET_HOP]
                                      .max(initial=0.0)) for s in starts])
    energy = (rms >= 10 ** (args.floor_db / 20.0)) & (rms >= rms.max() * args.rel_ratio)
    out.update(starts=starts, rms=rms, score=score, rank=score, energy=energy,
               onset_win=onset_score)
    ov_keep, ov_drop, ov_hit = override_masks(r, out, getattr(args, "ov", None))
    out.update(ov_keep=ov_keep, ov_drop=ov_drop, ov_hit=ov_hit)

    if tagger is not None:
        # 태거는 에너지 통과분만 돌린다. 어차피 탈락할 윈도우를 판정할 이유가 없고,
        # hop 250ms 에서는 후보가 4배라 이 절약이 크다.
        n = len(starts)
        idx = np.where(energy)[0]
        tag = np.full(n, np.nan)
        topp = np.full(n, np.nan)
        top = [""] * n
        own = np.full((n, len(LABEL_SETS[cls])) if cls != "background" else (n, 1),
                      np.nan)
        par = {l: np.full(n, np.nan) for l in PARENT_LABELS.get(cls, [])}
        if len(idx):
            P = tagger.probs(clip_key(r), x, starts[idx])
            tag[idx] = tagger.reduce(P, cls)
            am = P.argmax(axis=1)
            topp[idx] = P[np.arange(len(idx)), am]
            for k, i in enumerate(idx):
                top[i] = tagger.names[am[k]]
            if cls != "background":
                own[idx] = P[:, tagger.idx[cls]]
            for l, v in tagger.parents(P, cls).items():
                par[l][idx] = v
        out.update(tag=tag, top=top, topp=topp, own=own, par=par)
        if args.policy == "tag":
            thr = getattr(args, "thr", DEFAULT_TAG_THR)
            if cls == "background":
                # 이벤트 확률이 낮은 순으로 고르면 가장 안 헷갈리는 배경음만 남는다.
                # 하드 네거티브의 목적과 반대라 RMS 를 유지한다.
                out["rank"] = rms
            elif thr[cls] > 0:
                out["rank"] = np.nan_to_num(tag, nan=-1.0)
            elif onset_score is not None:
                out["rank"] = onset_score      # 태거를 끈 glass — 어택이 센 순
            else:
                out["rank"] = score
    return out


def tag_ok(res, thr, bg_thr):
    """후보별 태거 일치 마스크. 태거 미사용이면 None (= 전부 통과).

    에너지 탈락분의 태거 점수는 nan 이며 비교 결과가 False 가 된다. 어차피
    에너지 마스크와 AND 되므로 결과는 같다.
    """
    if res["tag"] is None:
        return None
    if res["cls"] != "background" and thr[res["cls"]] <= 0:
        return None                      # 임계값 0 = 이 클래스는 태거를 쓰지 않는다
    with np.errstate(invalid="ignore"):
        if res["cls"] == "background":
            return res["tag"] < bg_thr
        return res["tag"] >= thr[res["cls"]]


def choose(res, floor_lin, rel, max_win, agree=None):
    """에너지 필터(+태거 마스크) 통과 후보를 순위 점수 순으로 최대 max_win 개.

    hop 이 WIN 보다 짧으면 후보가 서로 겹친다. 겹친 윈도우를 함께 담으면 같은
    소리가 중복 학습되므로, 이미 고른 윈도우와 겹치는 후보는 건너뛴다.

    청취 override 가 있으면 자동 판정을 덮는다 (`load_overrides` 참조).

    반환: (선택 인덱스[시간순], 에너지 통과 수, 에너지+태거+override 통과 수)
    """
    rms = res["rms"]
    starts = res["starts"]
    energy = (rms >= floor_lin) & (rms >= rms.max() * rel)
    keep = energy if agree is None else energy & agree
    if res.get("ov_keep") is not None:
        keep = (keep | (energy & res["ov_keep"])) & ~res["ov_drop"]
    picked = []
    for i in np.argsort(-res["rank"], kind="stable"):
        if len(picked) >= max_win:
            break
        if not keep[i]:
            continue
        s = int(starts[i])
        if any(abs(s - int(starts[j])) < WIN for j in picked):
            continue
        picked.append(int(i))
    return sorted(picked, key=lambda i: starts[i]), int(energy.sum()), int(keep.sum())


def to_int8(w, key=None):
    """[-1,1] float → int8 [-128,127]. 클리핑 포함. **정규화는 하지 않는다.**

    스케일이 고정(×127)인 이유는 실기기 경로에 정규화가 없기 때문이다. 펌웨어는
    마이크 샘플을 그대로 NPU 에 넣으므로, 학습 데이터만 정규화하면 train/serve
    불일치가 된다. ai8x 의 `kws20.py` 는 파일 단위 peak 정규화를 하지만
    (`data / max(abs(data))`) 우리는 따르지 않는다 — CLAUDE.md 7장.

    `key` 를 주면 **원본이 정확히 0인 샘플**에만 ±1 LSB 잡음을 넣는다.
    디지털 0 은 마이크 스트림에서 나올 수 없는 값이고, 노이즈 게이트로 편집된
    원본에만 몰려 있어 "정확한 0 → 그 클래스" 지름길이 된다
    (`tools/zero_ratio_by_class.py` 실측: 채움 적용 후에도 glass 가 배경음의 4.1배).
    1 LSB 는 8bit 양자화 잡음과 같은 크기라 정보 손실이 없고, 전 클래스에 같은
    규칙으로 적용하므로 새 단서를 만들지 않는다. `key` 로 시드를 고정해 재현한다.
    """
    q = np.round(w * 127.0)
    if key is not None:
        z = (w == 0.0)
        n = int(z.sum())
        if n:
            rng = np.random.default_rng(rank_id(key) % (2 ** 32))
            q[z] = rng.integers(0, 2, n) * 2 - 1        # −1 또는 +1
    return np.clip(q, -128, 127).astype(np.int8)


# ──────────────────────────────────────────────────────────── 경로 해석
def clip_path(row, roots):
    """매니페스트 행 → 실제 오디오 경로. 없으면 None."""
    src = row["source"]
    if src == "FSD50K":
        p = os.path.join(roots["fsd"], f"{row['clip_id']}.wav")
    elif src == "US8K":
        p = os.path.join(roots["us8k"], row["clip_id"])
        if not os.path.isfile(p):
            for fold in range(1, 11):
                q = os.path.join(roots["us8k"], f"fold{fold}", row["clip_id"])
                if os.path.isfile(q):
                    return q
    elif src == "ESC-50":
        p = os.path.join(roots["esc50"], row["clip_id"])
    else:
        return None
    return p if os.path.isfile(p) else None


# ──────────────────────────────────────────────────────────────── 샤드 기록
class ShardWriter:
    """클래스·split 별 int8 샤드를 순차 기록한다. 메모리에 쌓지 않는다."""

    def __init__(self, out_dir, cls, split, shard_size=SHARD):
        self.dir = os.path.join(out_dir, split, cls)
        os.makedirs(self.dir, exist_ok=True)
        self.cls, self.split = cls, split
        self.shard_size = shard_size
        self.buf = []
        self.index = []          # (shard, row, clip_id, fsid, start_sample, note)
        self.shard_id = 0

    def add(self, w_int8, clip_id, fsid, start, note=""):
        self.index.append((self.shard_id, len(self.buf), clip_id, fsid, start, note))
        self.buf.append(w_int8)
        if len(self.buf) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        arr = np.stack(self.buf)
        np.save(os.path.join(self.dir, f"shard_{self.shard_id:04d}.npy"), arr)
        self.shard_id += 1
        self.buf = []

    def close(self):
        self.flush()
        with open(os.path.join(self.dir, "index.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["shard", "row", "clip_id", "fsid", "start_sample", "note"])
            w.writerows(self.index)
        return len(self.index)


# ──────────────────────────────────────────────────────────────────── 공통
def iter_rows(manifest, roots, limit=0, per_class=0, seed=0):
    """매니페스트 → [(행, 오디오경로)]. 오디오가 없는 행은 제외한다.

    `per_class` 를 주면 클래스별로 그만큼만 무작위 추출한다(민감도표·청취 표본용).
    매니페스트가 클래스순으로 정렬돼 있어 단순 `limit` 은 한 클래스에 쏠린다.
    """
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    out = []
    for r in rows:
        p = clip_path(r, roots)
        if p:
            out.append((r, p))

    if per_class:
        rng = np.random.default_rng(seed)
        by = defaultdict(list)
        for item in out:
            by[item[0]["cls"]].append(item)
        picked = []
        for c in CLASSES:
            v = by.get(c, [])
            if len(v) > per_class:
                sel = rng.choice(len(v), per_class, replace=False)
                v = [v[i] for i in sorted(sel)]
            picked.extend(v)
        out = picked

    if limit:
        out = out[:limit]
    return out


def tag_thresholds(args):
    """클래스별 태거 임계값. `--tag-thr` 를 주면 전 클래스를 그 값으로 덮고,
    `--tag-thr-cls` 가 다시 클래스별로 덮는다."""
    thr = dict(DEFAULT_TAG_THR) if args.tag_thr is None \
        else {c: args.tag_thr for c in EVENTS}
    for kv in args.tag_thr_cls:
        c, v = kv.split("=")
        if c not in thr:
            sys.exit(f"[에러] --tag-thr-cls 클래스명 오류: {c}")
        thr[c] = float(v)
    return thr


def analyze_all(rows, args, tagger):
    """전 클립 판정 결과 리스트. 오디오는 버리고 판정 재료만 남긴다(메모리)."""
    recs, n_fail = [], 0
    for i, (r, path) in enumerate(rows, 1):
        x = load_audio(path)
        if x is None:
            n_fail += 1
            continue
        res = analyze_clip(r, x, args, tagger)
        res["source"] = r["source"]
        recs.append(res)
        if i % 250 == 0:
            print(f"  판정 {i}/{len(rows)}", flush=True)
    return recs, n_fail


def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---:"] * len(head)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def write_md(path, lines):
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"저장: {path}")


# ─────────────────────────────────────────────────────── 에너지 민감도표
def sweep(rows, args):
    """에너지 임계값 민감도표 — 클래스·split 별 잔존 윈도우 수. 태거는 쓰지 않는다.

    후보·RMS 는 임계값과 무관하므로 클립당 한 번만 계산하고, 임계값 조합은
    그 배열로 평가한다. 오디오는 메모리에 쌓지 않는다.
    """
    combos = [(f, r) for f in args.sweep_floor for r in args.sweep_rel]
    cols = [(c, sp) for c in CLASSES for sp in SPLITS]
    recs, n_fail = analyze_all(rows, args, None)
    recs = [r for r in recs if r["cls"]]
    n_clip = Counter((r["cls"], r["split"]) for r in recs)

    def tally(floor_lin, rel, max_win):
        win, dead = Counter(), 0
        for r in recs:
            picked = choose(r, floor_lin, rel, max_win)[0]
            win[(r["cls"], r["split"])] += len(picked)
            dead += not picked
        return win, dead

    def row(a, b, win, dead):
        return [a, b] + [win[k] for k in cols] + [sum(win.values()), dead]

    head = ["floor(dBFS)", "rel"] + [f"{c}/{sp}" for c, sp in cols] + ["합계", "하한 탈락 클립"]
    src = Counter(r["source"] for r in recs)
    body = [row("필터 없음", "상한 없음", tally(0.0, 0.0, 10 ** 9)[0], 0),
            row("필터 없음", f"상한 {args.max_win}", tally(0.0, 0.0, args.max_win)[0], 0)]
    for floor_db, rel in combos:
        win, dead = tally(10 ** (floor_db / 20.0), rel, args.max_win)
        body.append(row(f"{floor_db:.0f}", f"{rel:.2f}", win, dead))

    lines = [
        "# 에너지 필터 민감도표",
        "",
        "`scripts/prepare_safesound.py --sweep` 자동 생성. 손으로 고치지 말 것.",
        "",
        f"- 입력: 오디오가 있는 클립 {len(recs)}개 ("
        + ", ".join(f"{k} {v}" for k, v in src.most_common())
        + f"), 읽기 실패 {n_fail}",
        f"- 윈도우: {WIN}샘플 @ {SR}Hz, hop {args.hop_ms}ms, 클립당 최대 {args.max_win}개",
        f"- 후보 생성: hop {args.hop_ms}ms 격자"
        + (" (glass 는 onset 피크 중심)" if args.policy == "class" else "")
        + ". 최종 선택은 겹치지 않게 한다",
        "- ⚠️ 이 표는 태거를 쓰지 않으므로 선택 순서가 RMS"
        + ("·스펙트럴 중심·onset" if args.policy == "class" else "")
        + " 기준이다. 실제 캐시 생성은 `--policy tag`(태거 확률 순)를 쓴다",
        f"- glass 게이트: Glass-only 클립은 onset ≥ {args.glass_onset_thr} 만 glass, "
        f"미달은 {args.glass_reject}",
        "- floor: 윈도우 RMS 절대 하한 (dBFS, 최대 진폭 1.0 기준)",
        "- rel: 클립 후보 최대 RMS 대비 비율 (진폭 기준). 두 기준 모두 통과해야 채택",
        "- 태거 필터는 적용하지 않은 수치다 → `label-noise-filter.md`",
        "- 값은 **윈도우 수**다. 원본 수가 아니므로 신뢰구간 계산에 쓰지 말 것",
        "",
        "## 클립 수",
        "",
    ]
    lines += md_table([f"{c}/{sp}" for c, sp in cols], [[n_clip[k] for k in cols]])
    lines += ["", "## 잔존 윈도우 수", ""] + md_table(head, body)

    # 클립이 통째로 빠지는 것은 절대 하한뿐이다(최대 후보는 상대 기준을 항상 통과).
    n_cls = Counter(r["cls"] for r in recs)
    lines += ["", "## 절대 하한으로 통째로 탈락한 클립 (클래스별)", "",
              "최대 후보 RMS 조차 floor 미만인 클립 수 / 전체. rel 과 무관하다.", ""]
    body = []
    for floor_db in args.sweep_floor:
        fl = 10 ** (floor_db / 20.0)
        dead = Counter(r["cls"] for r in recs if r["rms"].max() < fl)
        body.append([f"{floor_db:.0f}"] + [
            f"{dead[c]}/{n_cls[c]} ({100 * dead[c] / max(n_cls[c], 1):.1f}%)" for c in CLASSES])
    lines += md_table(["floor(dBFS)"] + CLASSES, body)
    write_md(args.sweep_out, lines)


# ─────────────────────────────────────────────── 라벨 노이즈 필터 보고서
def filter_report(rows, args, tagger):
    """glass 게이트·태거 필터 전후 클래스별 수량표 + 임계값 민감도.

    태거 점수는 캐시되므로 임계값을 바꿔 다시 돌리면 오디오 로드·onset 계산만
    다시 한다.
    """
    if tagger is None:
        sys.exit("[에러] --filter-report 는 태거가 필요하다 (--no-tagger 와 함께 쓸 수 없음)")
    recs, n_fail = analyze_all(rows, args, tagger)
    floor_lin = 10 ** (args.floor_db / 20.0)
    thr = tag_thresholds(args)
    cols = [(c, sp) for c in CLASSES for sp in SPLITS]
    live = [r for r in recs if r["cls"]]

    lines = [
        "# 라벨 노이즈 필터 — 전후 수량",
        "",
        "`scripts/prepare_safesound.py --filter-report` 자동 생성. 손으로 고치지 말 것.",
        "",
        f"- 입력: 오디오가 있는 클립 {len(recs)}개, 읽기 실패 {n_fail}",
        f"- 에너지 필터: floor {args.floor_db:.0f}dBFS, rel {args.rel_ratio:.2f}, "
        f"클립당 최대 {args.max_win}윈도우, hop {args.hop_ms}ms",
        f"- 윈도우 선택 정책: `--policy {args.policy}`",
        "- 태거: PANNs Cnn14 16kHz (`Cnn14_16k_mAP=0.438.pth`), AudioSet 527클래스, "
        "1초 윈도우 단위",
        "- 태거 일치 규칙: 이벤트는 자기 라벨 묶음 최대 확률 ≥ 임계값, "
        "background 는 이벤트 라벨 묶음 전체 최대 확률 < 배경 임계값",
        "- 임계값: " + ", ".join(
            f"{c} {thr[c]:.3f}" if thr[c] > 0 else f"**{c} 태거 미사용**"
            for c in EVENTS)
        + f", background < {args.tag_bg_thr:.2f}",
        "- 값은 **윈도우 수**(클립 수 명시한 곳 제외). 신뢰구간은 원본 수로 계산할 것",
        "",
        "## 태거 라벨 묶음",
        "",
    ]
    lines += md_table(["클래스", "AudioSet 라벨"],
                      [[c, ", ".join(LABEL_SETS[c])] for c in EVENTS])

    # ── A. glass 게이트
    g = [r for r in recs if r["orig"] == "glass"]
    lines += ["", "## A. glass 하위유형 게이트 (onset strength)", "",
              "클립 최대 onset strength 분포 (librosa `onset_strength`, hop 10ms, "
              "mel 64). Shatter 보유 클립은 무조건 채택, Glass-only 만 임계값 판정.", ""]
    body = []
    for name, key in (("Shatter 보유 (+ESC-50 glass_breaking)", ("shatter",)),
                      ("Glass-only", ("pass", "reject"))):
        v = np.array([r["onset"] for r in g if r["gate"] in key])
        if len(v):
            body.append([name, len(v)] + [f"{np.percentile(v, q):.1f}"
                                         for q in (5, 10, 25, 50, 75, 90)])
    lines += md_table(["하위유형", "클립", "p5", "p10", "p25", "p50", "p75", "p90"], body)
    lines += ["", "Glass-only 클립의 임계값별 잔존 (클립 수, train / test). "
              "탈락분은 배경음 하드 네거티브.", ""]
    go = [r for r in g if r["gate"] in ("pass", "reject")]
    body = []
    for t in sorted(set(args.onset_sweep) | {args.glass_onset_thr}):
        tr = sum(r["onset"] >= t for r in go if r["split"] == "train")
        te = sum(r["onset"] >= t for r in go if r["split"] == "test")
        mark = " ← 현재" if t == args.glass_onset_thr else ""
        body.append([f"{t:g}{mark}", tr, te, len(go) - tr - te])
    lines += md_table(["onset 임계값", "채택 train", "채택 test", "탈락(→배경)"], body)

    # ── B. 단계별 수량
    off = [c for c in EVENTS if thr[c] <= 0]
    lines += ["", "## B. 단계별 수량 (현재 임계값)", "",
              "클래스는 **실효 클래스**다(glass 게이트 탈락분은 background 에 포함). "
              "`최종` 은 클립당 상한 적용 후.",
              ("임계값 0 인 클래스(" + ", ".join(off) + ")는 태거를 쓰지 않으므로 "
               "`+태거 일치` 가 에너지 통과와 같다 — 차이는 청취 override 뿐이다."
               if off else ""), ""]
    body = []
    for c, sp in cols:
        rs = [r for r in live if (r["cls"], r["split"]) == (c, sp)]
        cand = en = ag = f_off = f_on = k_off = k_on = 0
        for r in rs:
            p_off, n_en, _ = choose(r, floor_lin, args.rel_ratio, args.max_win)
            p_on, _, n_ag = choose(r, floor_lin, args.rel_ratio, args.max_win,
                                   tag_ok(r, thr, args.tag_bg_thr))
            cand += len(r["starts"])
            en += n_en
            ag += n_ag
            f_off += len(p_off)
            f_on += len(p_on)
            k_off += bool(p_off)
            k_on += bool(p_on)
        body.append([f"{c}/{sp}", len(rs), cand, en, ag, f_off, f_on,
                     f"{100 * (f_off - f_on) / max(f_off, 1):.1f}%", k_off, k_on])
    lines += md_table(["클래스/split", "클립", "후보", "에너지 통과", "+태거 일치",
                       "최종(태거 전)", "최종(태거 후)", "태거 제거율",
                       "클립(태거 전)", "클립(태거 후)"], body)
    extra = Counter(r["split"] for r in live if r["note"] == "hard_neg:glass_no_onset")
    lines += ["", f"background 중 glass 게이트 탈락 재배정분: 클립 train {extra['train']} "
              f"/ test {extra['test']}"]

    # ── C. 이벤트 태거 임계값 민감도
    def final_counts(t_ev, t_bg):
        th = {c: t_ev for c in EVENTS} if t_ev is not None else thr
        win = Counter()
        for r in live:
            p = choose(r, floor_lin, args.rel_ratio, args.max_win,
                       tag_ok(r, th, t_bg))[0]
            win[(r["cls"], r["split"])] += len(p)
        return win

    ev_cols = [(c, sp) for c in EVENTS for sp in SPLITS]
    lines += ["", "## C. 이벤트 태거 임계값 민감도 (최종 윈도우 수, 전 이벤트 클래스 동일 임계값)", ""]
    base = final_counts(-1.0, 2.0)          # 태거 무효화 = 필터 전
    body = [["필터 전"] + [base[k] for k in ev_cols]]
    for t in args.tag_sweep:
        w = final_counts(t, args.tag_bg_thr)
        body.append([f"{t:.3f}"] + [w[k] for k in ev_cols])
    lines += md_table(["임계값"] + [f"{c}/{sp}" for c, sp in ev_cols], body)

    # ── D. 배경 임계값 민감도
    lines += ["", "## D. 배경음 태거 임계값 민감도 (최종 윈도우 수)", "",
              "배경음 윈도우 중 이벤트 라벨 확률이 임계값 이상인 것을 제거한다.", ""]
    body = [["필터 전", base[("background", "train")], base[("background", "test")]]]
    for t in args.tag_bg_sweep:
        w = final_counts(None, t)
        body.append([f"{t:.2f}", w[("background", "train")], w[("background", "test")]])
    lines += md_table(["배경 임계값", "background/train", "background/test"], body)

    # ── E. 불일치 윈도우의 태거 최상위 라벨 — 무엇이 섞여 있었나
    lines += ["", "## E. 태거 불일치 윈도우의 AudioSet 최상위 라벨 (에너지 통과분, 상위 8)", "",
              "우리 라벨과 불일치로 판정된 윈도우에 실제로 무엇이 들어 있었는지.", ""]
    body = []
    for c in CLASSES:
        cnt, n = Counter(), 0
        for r in live:
            if r["cls"] != c:
                continue
            rms = r["rms"]
            en = (rms >= floor_lin) & (rms >= rms.max() * args.rel_ratio)
            ok = tag_ok(r, thr, args.tag_bg_thr)
            if ok is None:                       # 태거를 쓰지 않는 클래스
                n = -1
                break
            for i in np.where(en & ~ok)[0]:
                cnt[r["top"][i]] += 1
                n += 1
        body.append([c, "태거 미사용" if n < 0 else n,
                     "—" if n < 0 else ", ".join(f"{k} {v}" for k, v in cnt.most_common(8))])
    lines += md_table(["클래스", "불일치 윈도우", "최상위 라벨 (빈도)"], body)

    # ── F. 온톨로지 상위 라벨로 걸러진 것인가
    lines += ["", "## F. 상위 라벨이 최상위인 불일치 윈도우의 자식 라벨 확률", "",
              "AudioSet 온톨로지에서 `Animal`·`Alarm` 은 `Dog`·`Siren` 의 조상이다. "
              "태거가 조상 라벨을 최상위로 내놓았다면 실제로는 우리 클래스인데 구체 "
              "라벨 확률이 낮아 걸러졌을 수 있다. 자식 확률이 임계값 근처에 몰려 "
              "있으면 임계값을 내리거나 상위 라벨을 라벨 묶음에 넣어야 한다는 신호다.", ""]
    body, body2 = [], []
    for c, parents in PARENT_LABELS.items():
        for p in parents:
            sel = [(r, i) for r in live if r["cls"] == c
                   and tag_ok(r, thr, args.tag_bg_thr) is not None
                   for i in np.where(r["energy"] & ~tag_ok(r, thr, args.tag_bg_thr))[0]
                   if r["top"][i] == p]
            if not sel:
                continue
            pp = np.array([r["topp"][i] for r, i in sel])
            ch = np.array([r["tag"][i] for r, i in sel])
            body.append([c, p, len(sel), f"{np.median(pp):.3f}",
                         f"{np.median(ch):.3f}", f"{np.percentile(ch, 90):.3f}",
                         int((ch >= 0.02).sum()), int((ch >= 0.05).sum()),
                         int((ch >= 0.10).sum()), f"{thr[c]:.2f}"])
            med = np.median(np.stack([r["own"][i] for r, i in sel]), axis=0)
            body2.append([c, p, len(sel)] + [f"{v:.3f}" for v in med])
    lines += md_table(["클래스", "상위 라벨", "윈도우", "상위 확률 p50",
                       "자식최대 p50", "자식최대 p90", "≥0.02", "≥0.05", "≥0.10",
                       "현재 임계값"], body)
    lines += ["", "같은 윈도우들의 **자식 라벨별** 확률 중앙값. 확률이 여러 자식 "
              "라벨로 쪼개졌는지(→ 최대만 보면 과소평가) 확인용.", ""]
    for c in PARENT_LABELS:
        rows_c = [b for b in body2 if b[0] == c]
        if rows_c:
            lines += md_table(["클래스", "상위 라벨", "윈도우"] + LABEL_SETS[c], rows_c)
            lines += [""]

    # ── G. 청취 override 적용 현황
    if args.ov:
        hit = Counter()
        seen = set()
        for r in live:
            for row in r["ov_hit"]:
                hit[(row["action"], r["cls"])] += 1
                seen.add(ov_key(row))
        miss = [k for k in args.ov if k not in seen]
        lines += ["## G. 청취 판정 override", "",
                  f"`{args.overrides}` 의 {len(args.ov)}행 중 **{len(seen)}행이 현재 후보 "
                  f"윈도우와 매칭**됐다. 근거는 `docs/results/listening-verification.md`.", ""]
        lines += md_table(["action", "클래스", "행"],
                          [[a, c, n] for (a, c), n in sorted(hit.items())])
        if miss:
            lines += ["", f"⚠️ 매칭 실패 {len(miss)}행 — `--hop-ms` 가 바뀌어 시작점이 "
                      "달라졌거나 클립이 다른 클래스로 재배정됐다:", "",
                      "  " + ", ".join(f"{c}@{s}" for c, s in miss[:20])
                      + (" …" if len(miss) > 20 else "")]
        lines += [""]
    write_md(args.report_out, lines)


# ────────────────────────────────────────────────── 거부 윈도우 청취 표본
def export_rejected(rows, args, tagger):
    """태거가 거른 **경계** 윈도우를 뽑아 준다 — 임계값이 옳은지 귀로 판정한다.

    대상은 에너지는 통과했는데 태거 확률이 `--reject-band` 구간(기본 0.02~0.10)
    안이라 1차 임계값 0.10 에서 잘려나간 윈도우다. 이 구간은 현재 임계값
    (glass·scream 0.05)이 지나는 자리이므로, 표본의 절반은 지금은 채택되는 쪽이다.
    `samples.csv` 의 `accepted_now` 로 구분한다 — 들어보고
      · 채택분(accepted_now=1)에 잡음이 많으면 임계값을 올린다
      · 거부분(accepted_now=0)에 진짜 이벤트가 많으면 더 내린다
    클립당 1개만 뽑아 같은 녹음이 표본을 채우지 않게 한다.
    """
    if tagger is None:
        sys.exit("[에러] --export-rejected 는 태거가 필요하다")
    lo, hi = args.reject_band
    out = args.sample_dir
    thr = tag_thresholds(args)
    rng = np.random.default_rng(args.seed)
    rows = [rows[i] for i in rng.permutation(len(rows))]
    written = Counter()
    records = []
    for r, path in rows:
        c = r["cls"]
        if c not in args.reject_classes or written[c] >= args.n_samples:
            continue
        x = load_audio(path)
        if x is None:
            continue
        res = analyze_clip(r, x, args, tagger)
        if res["cls"] != c:                     # glass 게이트 탈락분은 대상이 아니다
            continue
        band = res["energy"] & (res["tag"] >= lo) & (res["tag"] < hi)
        idx = np.where(band)[0]
        if not len(idx):
            continue
        i = int(idx[np.argmax(res["tag"][idx])])   # 클립당 경계에 가장 가까운 1개
        s = int(res["starts"][i])
        x = prepare_audio(x, r, args, getattr(args, "pool", None))[0]
        t = float(res["tag"][i])
        db = 20.0 * np.log10(max(float(res["rms"][i]), 1e-12))
        d = os.path.join(out, c)
        os.makedirs(d, exist_ok=True)
        stem = os.path.splitext(r["clip_id"])[0]
        name = f"{c}_t{round(1000 * t):03d}_{abs(db):03.0f}dBFS_{stem}_{s}.wav"
        sf.write(os.path.join(d, name), x[s:s + WIN], SR)
        records.append([f"{c}/{name}", c, r["split"], r["source"], r["clip_id"],
                        r["fsid"], res["note"], s, f"{db:.1f}", f"{t:.4f}",
                        int(t >= thr[c]), res["top"][i], f"{res['topp'][i]:.3f}"])
        written[c] += 1

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "samples.csv"), "w", newline="",
              encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "cls", "split", "source", "clip_id", "fsid", "note",
                     "start_sample", "rms_dbfs", "tag_score", "accepted_now",
                     "tag_top_label", "tag_top_prob"])
        wr.writerows(sorted(records))
    print(f"거부 경계 표본을 {out} 에 저장했다 (태거 확률 {lo}~{hi}, 클립당 1개).")
    for c in args.reject_classes:
        n_acc = sum(1 for x in records if x[1] == c and x[10])
        print(f"  {c:<12}{written[c]:>4}개 (현재 임계값 {thr[c]:.2f} 기준 "
              f"채택 {n_acc} / 거부 {written[c] - n_acc})")


# ──────────────────────────────────────────────────────────── 청취 표본
def export_samples(rows, args, tagger):
    """청취용 표본 추출 — 자동화가 안 되는 검증이므로 사람이 직접 듣는다.

    캐시 생성과 **같은 판정**(glass 게이트 → 클래스별 후보 → 에너지 → 태거)을
    거친 윈도우만 뽑는다. `--max-win 1` 이면 표본 30개가 클립 30개다.

    파일명: `{클래스}_t{태거점수×100}_{|dBFS|}dBFS_{클립}_{시작샘플}.wav`
    이벤트는 태거 점수가 낮을수록 경계선이므로 **이름순 앞쪽부터 들으면
    의심스러운 것부터 듣게 된다.** background 의 점수는 이벤트 라벨 최대 확률이라
    반대로 이름순 **끝쪽**이 의심 후보다.
    `samples.csv` 에 출처·게이트·onset·태거 최상위 라벨을 함께 남긴다.
    """
    out = args.sample_dir
    floor_lin = 10 ** (args.floor_db / 20.0)
    thr = tag_thresholds(args)
    # 매니페스트는 클래스·소스 순으로 묶여 있다. 앞에서부터 N개를 취하면
    # 소스 편향이 생기므로 섞는다 (시드 고정).
    rng = np.random.default_rng(args.seed)
    rows = [rows[i] for i in rng.permutation(len(rows))]
    written = Counter()
    records = []
    for r, path in rows:
        if all(written[c] >= args.n_samples for c in CLASSES):
            break
        # 게이트 탈락으로 클래스가 바뀔 수 있는 glass 는 미리 거르지 않는다
        if r["cls"] != "glass" and written[r["cls"]] >= args.n_samples:
            continue
        x = load_audio(path)
        if x is None:
            continue
        res = analyze_clip(r, x, args, tagger)
        c = res["cls"]
        if c is None or written[c] >= args.n_samples:
            continue
        picked = choose(res, floor_lin, args.rel_ratio, args.max_win,
                        tag_ok(res, thr, args.tag_bg_thr))[0]
        x = prepare_audio(x, r, args, getattr(args, "pool", None))[0]
        d = os.path.join(out, c)
        os.makedirs(d, exist_ok=True)
        for i in picked:
            if written[c] >= args.n_samples:
                break
            s = int(res["starts"][i])
            w = x[s:s + WIN]
            db = 20.0 * np.log10(max(float(res["rms"][i]), 1e-12))
            t = float(res["tag"][i]) if res["tag"] is not None else float("nan")
            stem = os.path.splitext(r["clip_id"])[0]   # US8K clip_id 는 .wav 를 포함
            tt = f"t{min(99, round(100 * t)):02d}_" if t == t else ""
            name = f"{c}_{tt}{abs(db):03.0f}dBFS_{stem}_{s}.wav"
            sf.write(os.path.join(d, name), w, SR)
            records.append([f"{c}/{name}", c, r["cls"], r["split"], r["source"],
                            r["clip_id"], r["fsid"], res["note"], res["gate"] or "",
                            "" if res["onset"] != res["onset"] else f"{res['onset']:.1f}",
                            s, f"{db:.1f}", "" if t != t else f"{t:.3f}",
                            res["top"][i] if res["top"] else ""])
            written[c] += 1

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "samples.csv"), "w", newline="",
              encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "cls", "manifest_cls", "split", "source", "clip_id",
                     "fsid", "note", "glass_gate", "onset_max", "start_sample",
                     "rms_dbfs", "tag_score", "tag_top_label"])
        wr.writerows(sorted(records))

    print(f"청취용 표본을 {out} 에 저장했다 "
          f"(floor {args.floor_db}dBFS, rel {args.rel_ratio}, "
          f"클립당 최대 {args.max_win}윈도우, 시드 {args.seed}, "
          f"태거 {'끔' if tagger is None else thr}, 배경 < {args.tag_bg_thr}).")
    for c in CLASSES:
        print(f"  {c:<12}{written[c]:>4}개")
    print("\n⚠️ 직접 들어볼 것. 숫자로는 통과인데 무음이거나 엉뚱한 소리일 수 있다.")
    print("   이벤트 클래스는 이름순 앞쪽(태거 점수 낮음)이 경계선이다.")


# ────────────────────────────────────────────── 한 클래스·split 전수 감사
def export_audit(rows, args, tagger):
    """한 클래스·split 의 **실제 채택 윈도우**를 청취용으로 내보낸다.

    `--export-samples` 와 다른 점은 클래스당 균등 표본이 아니라 **그 클래스가
    데이터셋에 실제로 싣는 윈도우**를 대상으로 한다는 것이다. `--audit-n 0` 이면
    전량 — test 셋 라벨 품질을 통째로 확인할 때 쓴다.

    파일명: `{클래스}_{하위유형}_on{윈도우 onset}_{|dBFS|}dBFS_{클립}_{시작}.wav`
    이름순으로 정렬하면 하위유형별로 묶이고 그 안에서 onset 오름차순이 된다 —
    **각 묶음 앞쪽이 어택이 가장 약한, 즉 게이트 경계에 있는 표본**이다.
    """
    want = args.audit_cls
    got = []
    for r, path in rows:
        if r["cls"] != want or r["split"] != args.audit_split:
            continue
        x = load_audio(path)
        if x is None:
            continue
        res = analyze_clip(r, x, args, tagger)
        if res["cls"] != want:                 # 게이트 탈락분은 이 클래스가 아니다
            continue
        picked = choose(res, 10 ** (args.floor_db / 20.0), args.rel_ratio,
                        args.max_win, tag_ok(res, tag_thresholds(args), args.tag_bg_thr))[0]
        for i in picked:
            got.append((r, res, int(i), prepare_audio(x, r, args, getattr(args, "pool", None))[0]))

    rng = np.random.default_rng(args.seed)
    if args.audit_n and len(got) > args.audit_n:
        sel = sorted(rng.choice(len(got), args.audit_n, replace=False))
        got = [got[i] for i in sel]

    out = args.sample_dir
    os.makedirs(out, exist_ok=True)
    records = []
    for r, res, i, x in got:
        s = int(res["starts"][i])
        sub = {"shatter": "shatter", "pass": "glassonly"}.get(res["gate"], "-")
        onw = float(res["onset_win"][i]) if res["onset_win"] is not None else float("nan")
        db = 20.0 * np.log10(max(float(res["rms"][i]), 1e-12))
        stem = os.path.splitext(r["clip_id"])[0]
        name = (f"{res['cls']}_{sub}_on{onw:03.0f}_{abs(db):03.0f}dBFS_"
                f"{stem}_{s}.wav")
        sf.write(os.path.join(out, name), x[s:s + WIN], SR)
        tag = float(res["tag"][i]) if res["tag"] is not None else float("nan")
        records.append([name, res["cls"], sub, r["split"], r["source"], r["clip_id"],
                        r["fsid"], s, f"{onw:.1f}", f"{res['onset']:.1f}",
                        f"{db:.1f}", "" if tag != tag else f"{tag:.4f}",
                        res["top"][i] if res["top"] else "", fill_note(res["fill"])])
    with open(os.path.join(out, "samples.csv"), "w", newline="",
              encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "cls", "subtype", "split", "source", "clip_id", "fsid",
                     "start_sample", "onset_window", "onset_clip", "rms_dbfs",
                     "tag_score", "tag_top_label", "fill"])
        wr.writerows(sorted(records))

    sub_n = Counter(x[2] for x in records)
    print(f"{want}/{args.audit_split} 감사 표본 {len(records)}개 → {out}")
    print("  하위유형: " + ", ".join(f"{k} {v}" for k, v in sub_n.most_common()))
    if records:
        onw = np.array([float(x[8]) for x in records])
        print(f"  윈도우 onset: 최소 {onw.min():.1f} / 중앙 {np.median(onw):.1f} "
              f"/ 최대 {onw.max():.1f}")


# ──────────────────────────────────────────────────────────── 캐시 생성
def build(rows, args, tagger):
    """실제 캐시 생성. 클립 하나씩 처리해 즉시 샤드에 append."""
    floor_lin = 10 ** (args.floor_db / 20.0)
    thr = tag_thresholds(args)
    os.makedirs(args.out, exist_ok=True)
    prog_path = os.path.join(args.out, "progress.json")
    done = set()
    if os.path.isfile(prog_path) and not args.restart:
        done = set(json.load(open(prog_path))["done"])
        print(f"이어하기: 이미 처리한 클립 {len(done)}개 건너뜀")

    writers = {}
    stats = Counter()
    drops = Counter()
    for i, (r, path) in enumerate(rows, 1):
        key = f"{r['source']}:{r['clip_id']}"
        if key in done:
            continue
        x = load_audio(path)
        if x is None:
            drops["읽기 실패"] += 1
            continue
        res = analyze_clip(r, x, args, tagger)
        c = res["cls"]
        if res["gate"] == "reject":
            drops[f"glass 게이트 탈락 → {args.glass_reject}"] += 1
        if c is None:
            done.add(key)
            continue
        picked, n_en, n_ag = choose(res, floor_lin, args.rel_ratio, args.max_win,
                                    tag_ok(res, thr, args.tag_bg_thr))
        n_cand = len(res["starts"])
        stats[f"후보:{c}"] += n_cand
        drops["에너지 필터 미달"] += n_cand - n_en
        drops["태거 불일치"] += n_en - n_ag
        drops["클립당 상한 초과"] += n_ag - len(picked)
        if not picked:
            drops["전 윈도우 탈락 클립"] += 1
            done.add(key)
            continue

        x = prepare_audio(x, r, args, getattr(args, "pool", None))[0]

        def emit(cls_out, j, note):
            wk = (cls_out, r["split"])
            if wk not in writers:
                writers[wk] = ShardWriter(args.out, cls_out, r["split"])
            s = int(res["starts"][j])
            key = f"{r['clip_id']}:{s}" if args.zero_dither else None
            writers[wk].add(to_int8(x[s:s + WIN], key), r["clip_id"], r["fsid"], s, note)
            stats[f"채택:{cls_out}"] += 1

        for j in picked:
            emit(c, j, " ".join(filter(None, [res["note"], fill_note(res["fill"])])))
        # relabel override — 다른 클래스로 보낸다. choose 는 현재 클래스에서
        # 이미 뺐으므로 여기서만 기록한다.
        for i, s in enumerate(res["starts"]):
            row = args.ov.get((r["clip_id"], int(s))) if args.ov else None
            if row and row["action"] == "relabel" and row["cls"] != c and res["energy"][i]:
                emit(row["cls"], i, f"relabel:{c}→{row['cls']}")
                drops["override relabel"] += 1

        done.add(key)
        if i % 200 == 0:
            json.dump({"done": sorted(done)}, open(prog_path, "w"))
            print(f"  {i}/{len(rows)} 처리", flush=True)

    totals = {}
    for (c, sp), w in writers.items():
        totals[(c, sp)] = w.close()
    json.dump({"done": sorted(done)}, open(prog_path, "w"))

    print("\n=== 필터 전후 데이터 수 (논문 데이터셋 표) ===")
    print(f"{'클래스':<12}{'후보 윈도우':>12}{'채택 윈도우':>12}{'통과율':>9}")
    print("-" * 45)
    for c in CLASSES:
        cand = stats[f"후보:{c}"]
        keep = stats[f"채택:{c}"]
        if cand:
            print(f"{c:<12}{cand:>12}{keep:>12}{100 * keep / cand:>8.1f}%")
    print("\n=== 폐기 사유 ===")
    for k, v in drops.most_common():
        print(f"  {v:>8}  {k}")
    print("\n=== 샤드 (split/class) ===")
    for (c, sp), n in sorted(totals.items()):
        print(f"  {sp:<6}{c:<12}{n:>8} 윈도우")
    print(f"\n캐시: {args.out}")


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/interim/manifest.csv")
    ap.add_argument("--fsd-dir", default="data/raw/FSD50K_clips")
    ap.add_argument("--us8k-dir", default="data/raw/US8K_audio")
    ap.add_argument("--esc50-dir", default="data/raw/ESC50_audio")
    ap.add_argument("--out", default="data/processed/safesound")
    ap.add_argument("--sample-dir", default="data/interim/listen")
    ap.add_argument("--overrides", default="data_overrides.csv",
                    help="청취 판정 override CSV (없으면 무시). `load_overrides` 참조")

    g = ap.add_argument_group("에너지 필터")
    g.add_argument("--floor-db", type=float, default=-50.0,
                   help="절대 하한 (dBFS). 기본 −50 은 하드웨어에서 유도한 값이다 "
                        "(MAX78000 입력 8bit ±127 → −50dBFS 는 크레스트 팩터 15dB 를 "
                        "가정해도 피크가 1 LSB 언저리다). 끄려면 --floor-db=-inf")
    g.add_argument("--rel-ratio", type=float, default=0.30,
                   help="클립 후보 최대 RMS 대비 비율 (0~1, 진폭 기준)")
    g.add_argument("--hop-ms", type=int, default=250,
                   help="격자 후보 hop (ms). 250=4배 중첩, 1000=비중첩. "
                        "겹친 후보 중 최종 선택은 서로 겹치지 않게 한다")
    g.add_argument("--max-win", type=int, default=MAX_WIN_PER_CLIP)

    g = ap.add_argument_group("윈도우 선택 / glass 게이트")
    g.add_argument("--policy", choices=["tag", "class", "rms"], default="tag",
                   help="tag: 태거 확률 높은 순(배경음은 RMS). "
                        "class: glass=onset 중심, scream=스펙트럴 중심 우선, 그 외 RMS. "
                        "rms: 전 클래스 RMS. class/rms 는 대조군용")
    g.add_argument("--glass-onset-thr", type=float, default=8.0,
                   help="Glass-only 클립 채택 기준 — 클립 최대 onset strength. "
                        "기본 8.0 은 Shatter 클립 분포의 약 p25 (잠정, 청취로 확정)")
    g.add_argument("--glass-reject", choices=["background", "drop"],
                   default="background",
                   help="게이트 탈락 Glass-only 클립 처리 (기본: 하드 네거티브, 규칙 3-1)")

    g = ap.add_argument_group("짧은 클립 채움")
    g.add_argument("--fill-mode", choices=["background", "zero"], default="background",
                   help="1초 미만 클립의 빈 구간 처리. background=같은 split 의 조용한 "
                        "배경 구간(기본), zero=0 패딩(이전 동작, 대조용)")
    g.add_argument("--fill-db", type=float, default=None,
                   help="채움 배경 레벨을 이벤트 RMS 대비 dB 로 맞춘다. "
                        "기본은 미지정 = **배경 구간 원래 레벨 그대로** (절대 레벨 정책)")
    g.add_argument("--fill-fade-ms", type=float, default=2.0,
                   help="이벤트 가장자리 페이드 (ms). 경계 클릭이 인공 단서가 되는 것을 막는다")
    g.add_argument("--fill-pool-clips", type=int, default=48,
                   help="split 별 채움용 배경 클립 수")
    g.add_argument("--no-zero-dither", dest="zero_dither", action="store_false",
                   help="디지털 0 샘플에 ±1 LSB 잡음을 넣지 않는다 (대조용)")

    g = ap.add_argument_group("태거 필터 (PANNs)")
    g.add_argument("--no-tagger", action="store_true", help="태거 필터 끄기")
    g.add_argument("--tag-thr", type=float, default=None,
                   help="이벤트 클래스 일치 기준 — 자기 라벨 묶음 최대 확률. "
                        "생략하면 클래스별 기본값 "
                        + ", ".join(f"{c} {v:.2f}" for c, v in DEFAULT_TAG_THR.items()))
    g.add_argument("--tag-thr-cls", nargs="*", default=[], metavar="CLS=THR",
                   help="클래스별 덮어쓰기. 예: --tag-thr-cls glass=0.05 scream=0.2")
    g.add_argument("--tag-bg-thr", type=float, default=0.50,
                   help="배경음 일치 기준 — 이벤트 라벨 최대 확률이 이 값 미만 (잠정)")
    g.add_argument("--tag-ckpt",
                   default=os.path.join(HOME, "panns_data", "Cnn14_16k_mAP=0.438.pth"))
    g.add_argument("--tag-labels",
                   default=os.path.join(HOME, "panns_data", "class_labels_indices.csv"))
    g.add_argument("--tag-cache", default="data/interim/tagger_cache/cnn14_16k")
    g.add_argument("--threads", type=int, default=8, help="태거 CPU 스레드")

    g = ap.add_argument_group("모드")
    g.add_argument("--sweep", action="store_true", help="에너지 민감도표만 출력")
    g.add_argument("--sweep-floor", type=float, nargs="+",
                   default=[-70, -60, -50, -45, -40, -35, -30])
    g.add_argument("--sweep-rel", type=float, nargs="+",
                   default=[0.0, 0.2, 0.3, 0.5])
    g.add_argument("--sweep-out", default="docs/results/energy-filter-sweep.md")
    g.add_argument("--filter-report", action="store_true",
                   help="glass 게이트·태거 필터 전후 수량표")
    g.add_argument("--tag-sweep", type=float, nargs="+",
                   default=[0.02, 0.05, 0.10, 0.20, 0.30, 0.50])
    g.add_argument("--tag-bg-sweep", type=float, nargs="+",
                   default=[0.20, 0.30, 0.50, 0.70])
    g.add_argument("--onset-sweep", type=float, nargs="+",
                   default=[2, 4, 6, 8, 10, 12, 15])
    g.add_argument("--report-out", default="docs/results/label-noise-filter.md")
    g.add_argument("--export-samples", action="store_true",
                   help="청취용 표본 wav 추출 (채택된 윈도우)")
    g.add_argument("--export-audit", action="store_true",
                   help="한 클래스·split 의 실제 채택 윈도우를 내보낸다 (전수 감사)")
    g.add_argument("--audit-cls", default="glass")
    g.add_argument("--audit-split", choices=SPLITS, default="test")
    g.add_argument("--audit-n", type=int, default=0, help="0=전량, N=무작위 N개")
    g.add_argument("--export-rejected", action="store_true",
                   help="태거가 거른 경계 윈도우 추출 (임계값 검증용)")
    g.add_argument("--reject-band", type=float, nargs=2, default=[0.02, 0.10],
                   metavar=("LO", "HI"), help="거부 표본을 뽑을 태거 확률 구간")
    g.add_argument("--reject-classes", nargs="+", default=["glass", "scream"])
    g.add_argument("--n-samples", type=int, default=10)
    g.add_argument("--limit", type=int, default=0, help="처음 N클립만 (디버깅)")
    g.add_argument("--per-class", type=int, default=0,
                   help="클래스별 N클립만 무작위 추출 (민감도표·청취 표본용)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--restart", action="store_true", help="진행 상태 무시하고 처음부터")
    return ap


def finalize(args, quiet=False):
    """파싱된 인자를 실행 가능한 상태로 만든다 — 파생값·override·채움 풀.

    CLI 와 `tools/` 의 분석 도구가 **같은 기본값**을 쓰도록 한 곳에 모았다.
    """
    args.hop = max(1, int(SR * args.hop_ms / 1000))
    rels = [args.rel_ratio] + (args.sweep_rel if args.sweep else [])
    if any(not 0.0 <= v <= 1.0 for v in rels):
        sys.exit("[에러] rel 비율은 0~1 이어야 한다")
    args.thr = tag_thresholds(args)              # 형식 오류를 먼저 잡는다
    args.ov = load_overrides(args.overrides)
    if args.ov and not quiet:
        print(f"청취 override {len(args.ov)}행 — {args.overrides}")
    args.roots = {"fsd": args.fsd_dir, "us8k": args.us8k_dir, "esc50": args.esc50_dir}
    # 채움 풀은 lazy 다 — 1초 미만 클립이 없으면 배경음을 한 개도 읽지 않는다.
    args.pool = (FillerPool(args.manifest, args.roots, args)
                 if args.fill_mode == "background" else None)
    return args


def default_args(**kw):
    """CLI 기본값으로 만든 args (도구·노트북용). 키워드로 일부만 덮어쓴다."""
    args = build_parser().parse_args([])
    for k, v in kw.items():
        if not hasattr(args, k):
            sys.exit(f"[에러] 알 수 없는 인자: {k}")
        setattr(args, k, v)
    return finalize(args, quiet=True)


def main():
    args = finalize(build_parser().parse_args())
    roots = args.roots
    rows = iter_rows(args.manifest, roots, args.limit, args.per_class, args.seed)
    if not rows:
        sys.exit("[에러] 처리할 오디오가 없다. 먼저 download_clips.py 를 실행할 것.")

    have = Counter(r["source"] for r, _ in rows)
    print(f"오디오가 있는 클립 {len(rows)}개  " +
          " ".join(f"{k}:{v}" for k, v in have.most_common()) + "\n")

    if args.sweep:
        sweep(rows, args)
        return
    tagger = None if args.no_tagger else Tagger(args)
    if args.filter_report:
        filter_report(rows, args, tagger)
    elif args.export_audit:
        export_audit(rows, args, tagger)
    elif args.export_rejected:
        export_rejected(rows, args, tagger)
    elif args.export_samples:
        export_samples(rows, args, tagger)
    else:
        build(rows, args, tagger)


if __name__ == "__main__":
    main()
