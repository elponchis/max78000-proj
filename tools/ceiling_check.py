#!/usr/bin/env python3
"""상한선 점검 — PANNs CNN14 로 v1 테스트셋의 "어려움" 자체를 잰다.

**묻는 것**: ① 로그 멜의 @300/h 0.673 이 낮은 이유가 *데이터가 어려워서* 인가
*모델·학습이 부족해서* 인가. 대규모 사전학습 모델(AudioSet 527클래스, 2M 클립)
을 같은 테스트셋에 그대로 태워 참조점을 얻는다.

⚠️ **편향 경고 — 반드시 함께 보고한다.**
우리 테스트 창 선별에 **PANNs 판정이 쓰였다** (`prepare_safesound.py` 의 태거
필터, `DEFAULT_TAG_THR`). 즉 PANNs 가 이벤트라고 본 창이 이벤트 클래스에 더
많이 남아 있다. **PANNs 에 유리한 쪽으로 기울어 있고, 상한이 과대추정일 수
있다.** glass 는 태거 임계값이 0(=태거 미사용)이라 그 클래스만 편향이 약하다.

**두 가지 모드**
  (A) zero-shot  — AudioSet 라벨 묶음을 5클래스로 접는다. 결정적이라 1회.
  (B) linear probe — 2048차원 임베딩을 고정하고 선형 분류기만 학습. 3시드.

**입력 조건을 ①과 같게 맞춘다** — v1 테스트셋 5,207창, 증강 없음,
**(b) 게인 정규화 끔**, 같은 창 잘라내기(augment=False → row[MARGIN:MARGIN+WIN]).
PANNs 체크포인트가 16kHz 판(`Cnn14_16k_mAP=0.438.pth`)이라 리샘플이 없다 —
필터링 때 쓴 것과 같은 모델·같은 전처리다.

사용 (WSL2):
    python3 tools/ceiling_check.py --mode zeroshot --json data/ceiling-zs.json
    python3 tools/ceiling_check.py --mode zeroshot --limit 64   # 속도 재기
    python3 tools/ceiling_check.py --mode probe --seeds 1,2,3
"""

import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI8X = os.environ.get("AI8X_DIR") or os.path.expanduser("~/ai8x-training")
sys.path[:0] = [AI8X, os.path.join(REPO, "datasets"), os.path.join(REPO, "tools"),
                os.path.join(REPO, "scripts")]

SR = 16000
WIN = 16384
MARGIN = 1600
CLASSES = ["siren", "glass", "scream", "dog_bark", "background"]
BG = 4

# `scripts/prepare_safesound.py` 의 LABEL_SETS 를 **그대로** 재사용한다.
# 복제하면 필터링과 평가가 조용히 갈린다.
from prepare_safesound import LABEL_SETS  # noqa: E402


def build_tagger(ckpt, labels_csv, threads):
    import csv as _csv
    import torch
    from panns_inference.models import Cnn14
    torch.set_num_threads(threads)
    m = Cnn14(sample_rate=SR, window_size=512, hop_size=160, mel_bins=64,
              fmin=50, fmax=SR // 2, classes_num=527)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"])
    m.eval()
    with open(labels_csv, encoding="utf-8") as f:
        names = [r["display_name"] for r in _csv.DictReader(f)]
    idx = {c: [names.index(l) for l in LABEL_SETS[c]] for c in CLASSES[:4]}
    return torch, m, idx


def iter_windows(data_root, split, limit=None):
    """①과 **같은 창**을 **같은 순서로** 내놓는다.

    `datasets/safesound.py` 의 인덱스 구성(클래스 순서대로 index.csv 를 읽고,
    `augment=False` 면 `row[MARGIN:MARGIN+WIN]`)을 그대로 재현한다.
    ⚠️ safesound 를 import 하지 않는다 — 그 모듈은 `ai8x` 를 요구하는데
    `panns_inference` 는 시스템 python3 에만 있고 ai8x venv 에는 없다.
    환경을 건드리지 않으려고 샤드를 직접 읽는다. 순서가 어긋나면 라벨이
    밀리므로, 아래 재현이 safesound 와 **한 줄씩 대응**하는지 확인할 것.
    """
    import csv as _csv
    root = os.path.join(data_root, "SafeSound")
    maps, n = {}, 0
    for target, cls in enumerate(CLASSES):
        d = os.path.join(root, split, cls)
        ip = os.path.join(d, "index.csv")
        if not os.path.isfile(ip):
            continue
        with open(ip, encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                if limit is not None and n >= limit:
                    return
                sp = os.path.join(d, f"shard_{int(r['shard']):04d}.npy")
                m = maps.get(sp)
                if m is None:
                    m = np.load(sp, mmap_mode="r")
                    maps[sp] = m
                w = np.asarray(m[int(r["row"])][MARGIN:MARGIN + WIN])
                n += 1
                yield (n - 1, target, r["clip_id"], r["fsid"],
                       int(r["start_sample"]), w)


def run_panns(a, want_embed=False):
    """(y_true, clips, fsids, starts, probs(527), embed(2048)) 를 모은다."""
    torch, model, idx = build_tagger(a.ckpt, a.labels, a.threads)
    ys, cls_, fs_, st_, P, E = [], [], [], [], [], []
    buf, t0 = [], time.time()

    def flush():
        if not buf:
            return
        X = torch.from_numpy(np.stack(buf).astype(np.float32) / 128.0)
        with torch.no_grad():
            o = model(X)
        P.append(o["clipwise_output"].numpy().astype(np.float32))
        if want_embed:
            E.append(o["embedding"].numpy().astype(np.float32))
        buf.clear()

    for i, t, c, f, s, w in iter_windows(a.data, a.split, a.limit):
        ys.append(t); cls_.append(c); fs_.append(f); st_.append(s)
        buf.append(w)
        if len(buf) >= a.batch_size:
            flush()
            if len(ys) % (a.batch_size * 8) == 0:
                el = time.time() - t0
                print(f"  {len(ys)}창  {el:.0f}s  "
                      f"({1000*el/len(ys):.0f} ms/창)", flush=True)
    flush()
    el = time.time() - t0
    print(f"  완료 {len(ys)}창  {el:.0f}s  ({1000*el/max(len(ys),1):.0f} ms/창)")
    return (np.array(ys), np.array(cls_), np.array(fs_), np.array(st_),
            np.concatenate(P), np.concatenate(E) if want_embed else None,
            idx, el)


def zeroshot_logits(P, idx):
    """527 확률 → 5클래스 점수.

    이벤트: 그 클래스의 AudioSet 라벨 묶음 **최대 확률** (필터링의 `reduce` 와
    같은 규칙). 배경: `1 − max(이벤트 4개)`.

    ⚠️ 이 점수를 로짓 자리에 그대로 넣는다. 고정 오경보 스윕은
    `margin = max(이벤트) − 배경` 의 **분위수** 위에서 돌므로, 단조 변환에
    불변이다 — 확률이든 로짓이든 같은 동작점을 집는다.
    """
    ev = np.stack([P[:, idx[c]].max(axis=1) for c in CLASSES[:4]], axis=1)
    bg = 1.0 - ev.max(axis=1)
    return np.concatenate([ev, bg[:, None]], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("zeroshot", "probe"), default="zeroshot")
    ap.add_argument("--data", default=os.path.expanduser("~/ai8x-training/data"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default=os.path.expanduser(
        "~/panns_data/Cnn14_16k_mAP=0.438.pth"))
    ap.add_argument("--labels", default=os.path.expanduser(
        "~/panns_data/class_labels_indices.csv"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threads", type=int, default=3,
                    help="chain9 와 CPU 를 나눠 쓰므로 기본을 낮게 둔다")
    ap.add_argument("--limit", type=int, default=None, help="속도 재기용")
    ap.add_argument("--seeds", default="1,2,3", help="probe 모드")
    ap.add_argument("--npz", default=None, help="점수·임베딩 캐시 경로")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    want_embed = (a.mode == "probe")
    cache = a.npz
    if cache and os.path.isfile(cache):
        print(f"캐시 사용: {cache}")
        z = np.load(cache, allow_pickle=True)
        y, clips, fsids, starts = z["y"], z["clips"], z["fsids"], z["starts"]
        P = z["P"]
        E = z["E"] if "E" in z and z["E"].size else None
        idx = z["idx"].item()
        el = float(z["sec"])
    else:
        y, clips, fsids, starts, P, E, idx, el = run_panns(a, want_embed)
        if cache:
            np.savez_compressed(
                cache, y=y, clips=clips, fsids=fsids, starts=starts, P=P,
                E=E if E is not None else np.zeros(0, dtype=np.float32),
                idx=np.array(idx, dtype=object), sec=el)
            print(f"저장: {cache}")

    if a.mode != "zeroshot":
        sys.exit("probe 모드는 아직 구현 전이다 (zeroshot 먼저)")

    logits = zeroshot_logits(P, idx)

    import eval_confusion as EC
    y_pred = logits.argmax(1)
    print()
    EC.report(fsids, y, y_pred, CLASSES, n_boot=0, seed=0,
              clips=clips, starts=starts, hop_ms=250)
    fx = EC.fixed_fa_points(logits, y, CLASSES, hop_ms=250)
    print()
    EC.print_fixed_fa(fx, CLASSES)

    if a.json:
        import json
        out = EC.summary_json(fsids, y, y_pred, CLASSES, hop_ms=250)
        out["fixed_fa"] = fx
        out["panns_sec"] = el
        out["n_windows"] = int(len(y))
        json.dump(out, open(a.json, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"저장: {a.json}")

    print()
    print("⚠️ 편향: 테스트 창 선별에 PANNs 판정이 쓰였다 "
          "(prepare_safesound.py 태거 필터).")
    print("   이 수치는 **상한의 과대추정일 수 있다**. glass 는 태거 임계값이")
    print("   0(미사용)이라 그 클래스만 편향이 약하다.")
    print("⚠️ zero-shot 은 결정적이라 시드 자가 없다. ① 과의 차이는 ① 자")
    print("   (0.0008)가 아니라 **① 3시드 범위(0.6728~0.6736)** 와 비교할 것.")


if __name__ == "__main__":
    main()
