#!/usr/bin/env python3
"""데이터셋 v3 KAT — `dataset-v3-design.md` 9절. 전부 통과해야 1단계 학습을 시작한다.

  1 v3 루트의 v1 행(= note 가 audioset 로 시작하지 않는 행)이 v1 루트와 **창 단위 비트 일치** (무증강 경로).
    index.csv 의 앞부분이 v1 index.csv 와 행 단위로 같은지도 본다.
  2 창 수가 MANIFEST.counts 와 `safesound.V3_TRAIN_COUNTS` 와 일치 (가중치가 맞는 수량으로 계산됐는지).
  3 AudioSet 행의 영상 ID(`yt:` fsid)가 external_v1 목록·유효 meta 와 겹치지 않음.
  4 채움 베드 출처(note 의 `src=`)가 전부 train_v2 **배경** 구간.
  5 v3 test 가 v1 test 와 같은 디렉터리 (링크).
  6 혼합 풀 크기 = v2.1 풀 + AudioSet 배경 창 중 int8 0 비율 ≤ 0.5 인 것 (증강 규칙은 그대로, 풀만 커진다).
  7 (보고) 청취용 wav 클래스별 10개 → data/interim/listen_v3/ (`--listen`).

사용 (WSL2, ai8x venv):  python tools/kat_v3.py [--listen]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI8X = os.environ.get("AI8X_DIR") or os.path.expanduser("~/ai8x-training")
sys.path[:0] = [AI8X, os.path.join(REPO, "datasets")]
import safesound as S                                     # noqa: E402

V1 = os.path.join(REPO, "data", "processed", "safesound")
V3 = os.path.join(REPO, "data", "processed", "safesound_v3")
EXT = os.path.expanduser("~/safesound-external")


def read_index(root, cls):
    with open(os.path.join(root, "train", cls, "index.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", action="store_true")
    ap.add_argument("--edition", default="v3", choices=["v3", "v31", "v4"], help="v31/v4: 루트 safesound_v31/v4, V31/V4_TRAIN_COUNTS")
    a = ap.parse_args()
    global V3
    if a.edition != "v3":
        V3 = os.path.join(REPO, "data", "processed", "safesound_" + a.edition)
    COUNTS = {"v3": S.V3_TRAIN_COUNTS, "v31": S.V31_TRAIN_COUNTS, "v4": S.V4_TRAIN_COUNTS}[a.edition]
    man = json.load(open(os.path.join(V3, "MANIFEST.json"), encoding="utf-8"))
    fails = []

    # 1 비트 일치 + index 앞부분
    ds1 = S.SafeSound(V1, "train", augment=False)
    ds3 = S.SafeSound(V3, "train", augment=False)
    n_v1_rows = 0
    excl = set()
    if a.edition == "v4":                      # v4: 제외 목록(MANIFEST.exclude.file)을 뺀 v1 행이 앞부분이어야 한다
        import csv as _csv
        with open(man["exclude"]["file"], encoding="utf-8") as f:
            excl = {(r["cls"], r["clip_id"], int(r["start_sample"])) for r in _csv.DictReader(f)}
    for cls in S.CLASSES:
        i1, i3 = read_index(V1, cls), read_index(V3, cls)
        i1k = [r for r in i1 if (cls, r["clip_id"], int(r["start_sample"])) not in excl]
        if i3[:len(i1k)] != i1k:
            fails.append(f"1 index 앞부분 불일치: {cls}")
        if excl and any((cls, r["clip_id"], int(r["start_sample"])) in excl for r in i3):
            fails.append(f"1 제외 창이 남아 있다: {cls}")
        n_v1_rows += len(i1k)
    key = lambda m: (m[0], m[1], m[2])                          # noqa: E731  (clip_id, fsid, start)
    pos3 = {key(m): i for i, m in enumerate(ds3.meta)}
    mism, checked = 0, 0
    for i, m in enumerate(ds1.meta):
        j = pos3.get(key(m))
        if j is None:
            if (S.CLASSES[ds1.index[i][0]], m[0], m[2]) in excl:
                continue                                        # v4 에서 일부러 뺀 창
            mism += 1; continue
        w1 = np.asarray(ds1._shard(ds1.index[i][1])[ds1.index[i][2]])
        w3 = np.asarray(ds3._shard(ds3.index[j][1])[ds3.index[j][2]])
        checked += 1
        if not np.array_equal(w1, w3):
            mism += 1
    print(f"[1] v1 행 {len(ds1)} 중 v3 에서 찾아 비교 {checked}, 불일치 {mism}")
    if mism:
        fails.append(f"1 비트 불일치 {mism}")

    # 2 창 수
    counts = [len(read_index(V3, c)) for c in S.CLASSES]
    man_counts = [man["counts"][c]["total"] for c in S.CLASSES]
    print(f"[2] index 창 수 {counts}  MANIFEST {man_counts}  {a.edition.upper()}_TRAIN_COUNTS {list(COUNTS)}")
    if counts != man_counts or counts != list(COUNTS):
        fails.append("2 창 수 불일치 (safesound.V3_TRAIN_COUNTS 갱신 필요?)")

    # 3 영상 ID 분리
    banned = {r["ytid"] for r in csv.DictReader(open(os.path.join(REPO, "tools", "external_set", "sampling_list.csv"), encoding="utf-8"))}
    banned |= {r["ytid"] for r in csv.DictReader(open(os.path.join(EXT, "external_v1", "meta.csv"), encoding="utf-8"))}
    rows_as = [r for c in S.CLASSES for r in read_index(V3, c) if r["note"].startswith("audioset")]
    yts = {r["fsid"][3:] for r in rows_as}
    hit = yts & banned
    print(f"[3] AudioSet 행 {len(rows_as)}, 영상 {len(yts)}, external_v1 과 겹침 {len(hit)}")
    if hit:
        fails.append(f"3 영상 ID 겹침 {len(hit)}")

    # 4 베드 출처
    bg_segs = set(man["audioset_segments"].get("background", []))
    bad, n_fill = 0, 0
    for r in rows_as:
        for tok in r["note"].split(";"):
            if tok.startswith("bgfill:"):
                n_fill += 1
                src = [t[4:] for t in tok.split(":") if t.startswith("src=")]
                seg = src[0].split("@")[0] if src else ""
                if seg not in bg_segs:
                    bad += 1
    print(f"[4] 채움 창 {n_fill}, 베드 출처가 train_v2 배경이 아닌 것 {bad}")
    if bad:
        fails.append(f"4 베드 출처 {bad}")

    # 5 test 링크
    t3, t1 = os.path.realpath(os.path.join(V3, "test")), os.path.realpath(os.path.join(V1, "test"))
    print(f"[5] v3 test → {t3}  (v1 test 와 같음: {t3 == t1})")
    if t3 != t1:
        fails.append("5 test 가 v1 과 다르다")

    # 6 혼합 풀
    p1 = len(S.SafeSound(V1, "train", mix_prob=0.5, mix_bg=True)._bg_pool())
    p3 = len(S.SafeSound(V3, "train", mix_prob=0.5, mix_bg=True)._bg_pool())
    bg_idx = S.CLASSES.index("background")
    n_as_bg_ok = 0
    for i, (t, shard, row, _l, _r) in enumerate(ds3.index):
        if t == bg_idx and ds3.meta[i][1].startswith("yt:"):
            w = np.asarray(ds3._shard(shard)[row])[S.MARGIN:S.MARGIN + S.WIN]
            n_as_bg_ok += float((w == 0).mean()) <= S.MIX_POOL_MAX_ZERO
    print(f"[6] 혼합 풀 v2.1 {p1} + AudioSet 배경(0 비율 ≤ 0.5) {n_as_bg_ok} = {p1 + n_as_bg_ok}  v3 풀 {p3}")
    if p3 != p1 + n_as_bg_ok:
        fails.append("6 혼합 풀 크기 불일치")

    # 7 청취용
    if a.listen:
        import soundfile as sf
        out = os.path.join(REPO, "data", "interim", "listen_" + a.edition)
        os.makedirs(out, exist_ok=True)
        rng = np.random.default_rng(78003)
        for c in S.CLASSES:
            rows = [r for r in read_index(V3, c) if r["note"].startswith("audioset")]
            for r in rng.choice(rows, size=min(10, len(rows)), replace=False):
                shard = os.path.join(V3, "train", c, f"shard_{int(r['shard']):04d}.npy")
                w = np.load(shard, mmap_mode="r")[int(r["row"])][S.MARGIN:S.MARGIN + S.WIN]
                sf.write(os.path.join(out, f"{c}_{r['clip_id']}_{r['start_sample']}.wav"),
                         (w.astype(np.int16) * 256), S.SR if hasattr(S, "SR") else 16000)
        print(f"[7] 청취용 wav → {out}")

    # 8 (v4) 교사 로짓 정렬·라벨 없는 창 분리·KD 로더 타깃
    if a.edition == "v4":
        import csv as _csv
        tdir = os.path.join(V3, "teacher")
        for cls in S.CLASSES:
            n = len(read_index(V3, cls))
            for name in (f"emb_{cls}.npy", f"teacher_{cls}.npy"):
                pth = os.path.join(tdir, name)
                if not os.path.isfile(pth):
                    fails.append(f"8 없음 {name}"); continue
                sh = np.load(pth, mmap_mode="r").shape
                if sh[0] != n:
                    fails.append(f"8 {name} 행 {sh[0]} ≠ index {n}")
        U = os.path.join(REPO, "data", "processed", "safesound_v4u", "train", "unlabeled")
        rows_u = list(_csv.DictReader(open(os.path.join(U, "index.csv"), encoding="utf-8")))
        tu = np.load(os.path.join(U, "teacher.npy"), mmap_mode="r")
        if tu.shape != (len(rows_u), len(S.CLASSES)):
            fails.append(f"8 라벨 없는 교사 로짓 {tu.shape} ≠ ({len(rows_u)}, 5)")
        yt_u = {r["fsid"][3:] for r in rows_u}
        hit_u = yt_u & banned
        print(f"[8] 교사 로짓 정렬 확인, 라벨 없는 창 {len(rows_u):,} (영상 {len(yt_u):,}), external_v1 과 겹침 {len(hit_u)}")
        if hit_u:
            fails.append(f"8 라벨 없는 창 영상 겹침 {len(hit_u)}")
        # KD 로더: 타깃 모양·라벨 없는 항목·에폭 길이, 무증강 라벨 창은 v4 로더와 비트 일치
        kd = S.SafeSound(V3, "train", augment=False).attach_kd(tdir, os.path.join(REPO, "data", "processed", "safesound_v4u"))
        x0, t0 = kd[0]; xu, tu0 = kd[len(kd) - 1]
        if tuple(t0.shape) != (6,) or int(t0[0]) != kd.index[0][0] or int(tu0[0]) != -1 or len(kd) != 2 * len(kd.index):
            fails.append("8 KD 로더 타깃/길이 불일치")
        if not np.array_equal(x0.numpy(), ds3[0][0].numpy()):
            fails.append("8 KD 로더 라벨 창이 v4 로더와 다르다")
        kdh = S.SafeSound(V3, "train", augment=False).attach_kd(tdir, os.path.join(REPO, "data", "processed", "safesound_v4u"), unlab_half=True)
        print(f"[8] KD 로더 길이 {len(kd):,} (라벨 {len(kd.index):,}), 라벨 없음 전체 {len(kd._kd['unlab']):,} / 절반 {len(kdh._kd['unlab']):,}")
        if not (0.4 < len(kdh._kd["unlab"]) / len(kd._kd["unlab"]) < 0.6):
            fails.append("8 절반 분할 비율 이상")

    print("\nKAT", "전부 통과" if not fails else f"실패 {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
