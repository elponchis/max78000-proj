#!/usr/bin/env python3
"""데이터셋 v3 빌더 — v1 샤드(링크) + AudioSet strong train 창 (`docs/results/dataset-v3-design.md` 1·2·9절).

규칙 (결과 보기 전에 고정, 2026-10-07):
  · 정의: v1 동등 (siren = `Siren` + 하위 4종, glass = `Glass shatter`, scream = Screaming·Yell·Shout 이고
    군중 목록 없음, dog_bark = `Bark` 또는 `Dog`). Screaming류 + 군중 → background `hard_neg:scream_crowd`.
    실사용 정의는 note 의 `real=` 태그로만 남긴다 (학습에 안 씀).
  · 창: 모델 창 WIN(16,384) 격자 hop 250 ms. 이벤트 구간은 **겹침 ≥ min(0.5초, 이벤트 총 길이)** 인 창만.
    이벤트 창은 int8 0 비율 ≤ FLOOR_ZERO(0.95), 배경은 하한 없음. 겹침 길이 → RMS 순으로 서로 겹치지 않게
    구간당 최대 MAX_WIN(2). 배경은 RMS 순. scream_crowd 는 비명 구간과 겹치는 창.
  · 가공: 16 kHz wav(이미 모노 16 kHz) → `fill_zero_runs`(베드 = train_v2 배경, 같은 split·같은 출처)
    → `store_row`(가장자리 여유 축소, 모자란 여유는 베드) → `to_int8`. 전부 prepare_safesound 의 함수를 import.
  · webm 폴백(download_state reason=webm_trim)은 포함하고 note `webm_trim=1`.
  · 분리: 영상 ID 가 external_v1 목록·유효 meta 와 겹치면 **중단**한다.

산출 (`--out`, 기본 data/processed/safesound_v3):
  train/<cls>/shard_0000..  → v1 샤드 심볼릭 링크,  shard_0100..  AudioSet 창,  index.csv = v1 행 + AudioSet 행
  test → ../safesound/test 링크,  MANIFEST.json (창 수·규칙·구간 ID·커밋)
  `--dry-run` 은 집계 표만 낸다 (파일 안 씀).

사용 (WSL2, safesound-external venv 또는 numpy+soundfile 가 있는 python):
    python3 scripts/build_v3_windows.py --dry-run
    python3 scripts/build_v3_windows.py
"""
import argparse
import collections
import csv
import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tools", "external_set"))
import prepare_safesound as P                                           # noqa: E402
from build_sampling import (CROWD_V1, DOG_V1, DOG_REAL, GLASS, SCREAM_REAL,  # noqa: E402
                            SCREAM_V1, SIREN_SUB, SIREN_V1)

EXT = os.path.expanduser("~/safesound-external")
CLASSES = P.CLASSES
HOP = P.SR // 4            # 250 ms
MAX_WIN = 2                # 구간당 창 상한 (D3)
MIN_OVERLAP_S = 0.5        # 겹침 규칙
FLOOR_ZERO = 0.95          # v1 --floor-zero
SHARD_BASE = 100           # AudioSet 샤드 번호 시작
WIN, MARGIN, STORE, SR = P.WIN, P.MARGIN, P.STORE, P.SR


V1_TRAIN_COUNTS = (528, 477, 491, 1934, 9577)


def class_weights(counts, power=1.0):
    """`datasets/safesound.class_weights` 와 같은 식 (torch 없이). w_c = N/(K·n_c), 표본당 평균 1."""
    raw = [(1.0 / c) ** power if c else 0.0 for c in counts]
    s = sum(c * w for c, w in zip(counts, raw)) / sum(counts)
    return tuple(round(w / s, 4) for w in raw)


def fill_args():
    """prepare_safesound 의 채움 기본값과 같은 인자 묶음."""
    return argparse.Namespace(zero_run_min=16, fill_snr_db=10.0, fill_fade_ms=2.0,
                              fill_pool_clips=48, fill_bed_min_db=-45.0, store_dtype="int8")


class AudioSetBedPool(P.FillerPool):
    """채움 베드 풀 — train_v2 **배경** 구간에서 (같은 split·같은 출처). pick() 은 부모 그대로."""

    def __init__(self, bg_paths, args):          # noqa: D107  (부모 __init__ 은 매니페스트를 읽는다)
        self.args = args
        self.bg_paths = sorted(bg_paths, key=lambda p: P.rank_id(os.path.basename(p)))
        self.cache = {}

    def _build(self, split):
        segs, used, rmss = [], [], []
        floor_bed = 10 ** (self.args.fill_bed_min_db / 20.0)
        for p in self.bg_paths:
            if len(segs) >= self.args.fill_pool_clips:
                break
            x = P.load_audio(p)
            if x is None or len(x) < WIN:
                continue
            starts = np.arange(0, len(x) - WIN + 1, WIN, dtype=np.int64)
            rms = P.window_rms(x, starts)
            order = list(np.argsort(rms))
            order = [i for i in order if rms[i] >= floor_bed] or order
            for i in order:
                w = x[starts[i]:starts[i] + WIN]
                if float((w == 0.0).mean()) > 0.001 or rms[i] <= 0:
                    continue
                segs.append(w.astype(np.float32))
                used.append((os.path.basename(p)[:-4], int(starts[i])))
                rmss.append(float(rms[i]))
                break
        if not segs:
            sys.exit("[에러] train_v2 배경에서 채움 베드를 찾지 못했다")
        self.cache[split] = (segs, used, np.array(rmss))
        return self.cache[split]


def load_strong(path):
    seg = collections.defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            seg[r["segment_id"]].append((float(r["start_time_seconds"]), float(r["end_time_seconds"]), r["label"]))
    return seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default=os.path.join(REPO, "tools", "external_set", "train_v2_list.csv"))
    ap.add_argument("--raw", default=os.path.join(EXT, "train_v2", "raw"))
    ap.add_argument("--state", default=os.path.join(EXT, "train_v2", "download_state.csv"))
    ap.add_argument("--meta", default=os.path.join(EXT, "meta"))
    ap.add_argument("--v1-root", default=os.path.join(REPO, "data", "processed", "safesound"))
    ap.add_argument("--out", default=os.path.join(REPO, "data", "processed", "safesound_v3"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--scream-screaming-only", action="store_true",
                    help="v3.1: AudioSet 구간 중 Yell·Shout 만 있고 Screaming 이 없는 구간을 학습에서 제외한다 "
                         "(양성·배경 어느 쪽으로도 안 씀). 군중 동반은 v1 규칙 그대로 하드 네거티브.")
    ap.add_argument("--tag", default=None, help="MANIFEST tag (기본 dataset-v3 / --scream-screaming-only 면 dataset-v3.1)")
    a = ap.parse_args()
    tag = a.tag or ("dataset-v3.1" if a.scream_screaming_only else "dataset-v3")

    # ── 라벨 사전 ──
    nm = dict(l.rstrip("\n").split("\t") for l in open(os.path.join(a.meta, "mid_to_display_name.tsv"), encoding="utf-8"))
    by = {v: k for k, v in nm.items()}
    M = lambda names: {by[n] for n in names if n in by}                 # noqa: E731
    sv1, ssub, gl = M(SIREN_V1), M(SIREN_SUB), M(GLASS)
    dv1, dre, cv1, cre, crowd = M(DOG_V1), M(DOG_REAL), M(SCREAM_V1), M(SCREAM_REAL), M(CROWD_V1)
    MIDS = {"siren": sv1 | ssub, "glass": gl, "scream": cv1, "dog_bark": dv1}

    def v1eq(s):
        c = set()
        if s & (sv1 | ssub): c.add("siren")
        if s & gl: c.add("glass")
        if (s & cv1) and not (s & crowd): c.add("scream")
        if s & dv1: c.add("dog_bark")
        return c

    def real(s):
        c = set()
        if s & (sv1 | ssub): c.add("siren")
        if s & gl: c.add("glass")
        if s & cre: c.add("scream")
        if s & dre: c.add("dog_bark")
        return c

    strong = load_strong(os.path.join(a.meta, "audioset_train_strong.tsv"))
    rows = list(csv.DictReader(open(a.list, encoding="utf-8")))
    state = {r["segment_id"]: r for r in csv.DictReader(open(a.state, encoding="utf-8"))}
    ok = [r for r in rows if state.get(r["segment_id"], {}).get("status") == "ok"
          and os.path.isfile(os.path.join(a.raw, r["segment_id"] + ".wav"))]
    webm = {s for s, r in state.items() if r["status"] == "ok" and r["reason"] == "webm_trim"}

    # ── 분리 확인 (겹치면 중단) ──
    banned = {r["ytid"] for r in csv.DictReader(open(os.path.join(REPO, "tools", "external_set", "sampling_list.csv"), encoding="utf-8"))}
    meta_ext = os.path.join(EXT, "external_v1", "meta.csv")
    if os.path.isfile(meta_ext):
        banned |= {r["ytid"] for r in csv.DictReader(open(meta_ext, encoding="utf-8"))}
    banned_eval = {s.rsplit("_", 1)[0] for s in load_strong(os.path.join(a.meta, "audioset_eval_strong.tsv"))}
    hit = [r["segment_id"] for r in ok if r["ytid"] in banned or r["ytid"] in banned_eval]
    if hit:
        sys.exit(f"[에러] external_v1/eval 과 영상 ID 가 겹친다 ({len(hit)}): {hit[:5]}")
    print(f"유효 구간 {len(ok)} (목록 {len(rows)}), webm 폴백 {len(webm)}, 영상 ID 겹침 0 (external_v1 {len(banned)} / eval {len(banned_eval)})")

    # ── 구간 배정 ──
    assign = {}                        # sid → (cls, hard_neg, real_cls, mids 또는 None)
    tally = collections.Counter()
    for r in ok:
        sid = r["segment_id"]
        s = {l for _, _, l in strong[sid]}
        c1, c2 = v1eq(s), real(s)
        rc = next(iter(c2), "-") if len(c2) == 1 else "-"
        if r["group"] == "background":
            assign[sid] = ("background", "", rc, None); tally["background"] += 1
        elif len(c1) > 1:
            tally["제외(다중)"] += 1
        elif c1:
            c = next(iter(c1))
            if a.scream_screaming_only and c == "scream" and not (s & cre):
                tally["제외(v3.1: Yell·Shout 만, Screaming 없음)"] += 1
                continue
            assign[sid] = (c, "", rc, MIDS[c]); tally[c] += 1
        elif (s & cv1) and (s & crowd):
            assign[sid] = ("background", "scream_crowd", rc, cv1); tally["background:scream_crowd"] += 1
        else:
            tally["제외(어느 정의도 아님)"] += 1
    print("구간 배정:", dict(tally))

    args = fill_args()
    bg_paths = [os.path.join(a.raw, sid + ".wav") for sid, v in assign.items() if v[0] == "background" and not v[1]]
    pool = AudioSetBedPool(bg_paths, args)

    # ── 창 추출 ──
    out_rows = collections.defaultdict(list)      # cls → [(stored int8, clip_id, fsid, start, left, right, note)]
    stat = {k: collections.Counter() for k in ("cand", "floor_drop", "nowin", "win", "fill_seg", "seg")}
    for r in ok:
        sid = r["segment_id"]
        if sid not in assign:
            continue
        cls, hard, rc, mids = assign[sid]
        key = cls + (":" + hard if hard else "")
        stat["seg"][key] += 1
        x = P.load_audio(os.path.join(a.raw, sid + ".wav"))
        if x is None or len(x) < WIN:
            stat["nowin"][key] += 1
            continue
        rr = {"split": "train", "clip_id": sid}
        x, finfo = P.fill_zero_runs(x, rr, args, pool)
        if finfo:
            stat["fill_seg"][key] += 1
        n = len(x)
        starts = np.arange(0, n - WIN + 1, HOP, dtype=np.int64)
        if mids is None:
            cand, ov = list(starts), {int(t): 1.0 for t in starts}
        else:
            iv = [(s0, e0) for s0, e0, l in strong[sid] if l in mids]
            need = min(MIN_OVERLAP_S, sum(e0 - s0 for s0, e0 in iv))
            cand, ov = [], {}
            for t in starts:
                w0, w1 = t / SR, (t + WIN) / SR
                o = sum(max(0.0, min(w1, e0) - max(w0, s0)) for s0, e0 in iv)
                if o > 0 and o >= need - 1e-6:
                    cand.append(int(t)); ov[int(t)] = o
        stat["cand"][key] += len(cand)
        zf = {t: float((P.to_int8(x[t:t + WIN]) == 0).mean()) for t in cand}
        rms = {t: float(np.sqrt(np.mean(x[t:t + WIN].astype(np.float64) ** 2))) for t in cand}
        if cls != "background":
            keep = [t for t in cand if zf[t] <= FLOOR_ZERO]
            stat["floor_drop"][key] += len(cand) - len(keep)
            cand = keep
        chosen = []
        for t in sorted(cand, key=lambda t: (-ov[t], -rms[t], t)):
            if len(chosen) >= MAX_WIN:
                break
            if all(abs(t - c) >= WIN for c in chosen):
                chosen.append(t)
        if not chosen:
            stat["nowin"][key] += 1
            continue
        for t in sorted(chosen):
            row, left, right = P.store_row(x, t, rr, args, pool)
            note = (f"audioset;v1eq={cls};real={rc};webm_trim={int(sid in webm)};ev_overlap={ov[t]:.3f}"
                    + (f";hard_neg:{hard}" if hard else "") + (";" + P.fill_note(finfo) if finfo else ""))
            out_rows[cls].append((row, sid, "yt:" + r["ytid"], int(t), left, right, note))
            stat["win"][key] += 1

    # ── 집계 표 ──
    v1_counts = {}
    for cls in CLASSES:
        with open(os.path.join(a.v1_root, "train", cls, "index.csv"), encoding="utf-8") as f:
            v1_counts[cls] = sum(1 for _ in csv.DictReader(f))
    print("\n| 클래스 | v1 train 창 | AudioSet 구간 | 후보 창 | 하한 탈락 | 창 0개 구간 | 채움 구간 | AudioSet 창 (최대 %d) | v3 합계 | AudioSet 비중 |" % MAX_WIN)
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    tot = collections.Counter()
    for cls in CLASSES:
        keys = [k for k in stat["seg"] if k.split(":")[0] == cls]
        g = lambda name: sum(stat[name][k] for k in keys)                  # noqa: E731
        w = len(out_rows[cls])
        extra = " (하드 네거티브 scream_crowd 구간 %d, 창 %d)" % (stat["seg"]["background:scream_crowd"], stat["win"]["background:scream_crowd"]) if cls == "background" else ""
        print(f"| {cls} | {v1_counts[cls]:,} | {g('seg'):,}{extra} | {g('cand'):,} | {g('floor_drop')} | {g('nowin')} | {g('fill_seg')} | **{w:,}** | {v1_counts[cls] + w:,} | {100 * w / (v1_counts[cls] + w):.0f}% |")
        for name in ("seg", "cand", "floor_drop", "nowin", "fill_seg"):
            tot[name] += g(name)
        tot["win"] += w; tot["v1"] += v1_counts[cls]
    print(f"| **합계** | **{tot['v1']:,}** | {tot['seg']:,} | {tot['cand']:,} | {tot['floor_drop']} | {tot['nowin']} | {tot['fill_seg']} | **{tot['win']:,}** | **{tot['v1'] + tot['win']:,}** | {100 * tot['win'] / (tot['v1'] + tot['win']):.0f}% |")
    counts_v3 = [v1_counts[c] + len(out_rows[c]) for c in CLASSES]
    print("\n클래스 가중치 v1:", class_weights(V1_TRAIN_COUNTS), "\n클래스 가중치 v3:", class_weights(counts_v3), " (창 수", counts_v3, ")")
    if a.dry_run:
        return

    # ── 쓰기 ──
    os.makedirs(a.out, exist_ok=True)
    test_link = os.path.join(a.out, "test")
    if not os.path.lexists(test_link):
        os.symlink(os.path.relpath(os.path.join(a.v1_root, "test"), a.out), test_link)
    seg_ids = {}
    for cls in CLASSES:
        d = os.path.join(a.out, "train", cls)
        os.makedirs(d, exist_ok=True)
        src = os.path.join(a.v1_root, "train", cls)
        with open(os.path.join(src, "index.csv"), encoding="utf-8") as f:
            rd = csv.DictReader(f)
            header = rd.fieldnames
            v1_rows = list(rd)
        for sh in sorted({int(r["shard"]) for r in v1_rows}):
            if sh >= SHARD_BASE:
                sys.exit(f"[에러] v1 샤드 번호 {sh} 가 {SHARD_BASE} 이상이다")
            link = os.path.join(d, f"shard_{sh:04d}.npy")
            if not os.path.lexists(link):
                os.symlink(os.path.relpath(os.path.join(src, f"shard_{sh:04d}.npy"), d), link)
        # 기존 AudioSet 샤드는 지우고 다시 쓴다
        for fn in os.listdir(d):
            if fn.startswith("shard_") and int(fn[6:10]) >= SHARD_BASE:
                os.remove(os.path.join(d, fn))
        new_rows, buf, sh = [], [], SHARD_BASE
        for i, (row, sid, fsid, t, left, right, note) in enumerate(out_rows[cls]):
            if len(buf) == P.SHARD:
                np.save(os.path.join(d, f"shard_{sh:04d}.npy"), np.stack(buf)); sh += 1; buf = []
            new_rows.append({"shard": sh, "row": len(buf), "clip_id": sid, "fsid": fsid, "start_sample": t,
                             "left_margin": left, "right_margin": right, "note": note})
            buf.append(row)
            seg_ids.setdefault(cls, []).append(sid)
        if buf:
            np.save(os.path.join(d, f"shard_{sh:04d}.npy"), np.stack(buf))
        with open(os.path.join(d, "index.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, header)
            w.writeheader(); w.writerows(v1_rows); w.writerows(new_rows)
        print(f"  {cls}: v1 {len(v1_rows):,} + audioset {len(new_rows):,} 행, 샤드 {sh - SHARD_BASE + 1}개")

    def git(*g):
        try:
            return subprocess.run(("git",) + g, cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:                                   # noqa: BLE001
            return ""
    man = {"tag": tag, "created": datetime.now().astimezone().isoformat(timespec="seconds"),
           "commit": git("rev-parse", "HEAD"), "v1_root": a.v1_root, "v1_manifest": json.load(open(os.path.join(a.v1_root, "MANIFEST.json"))).get("tag"),
           "rules": {"definition": "v1eq (siren=Siren+하위4종)" + (" + scream=Screaming 필수 (Yell·Shout 만 제외)" if a.scream_screaming_only else ""), "hop": HOP, "max_win": MAX_WIN, "min_overlap_s": MIN_OVERLAP_S,
                     "floor_zero": FLOOR_ZERO, "fill": vars(args), "webm_fallback": "included, note webm_trim=1"},
           "counts": {c: {"v1": v1_counts[c], "audioset": len(out_rows[c]), "total": v1_counts[c] + len(out_rows[c])} for c in CLASSES},
           "class_weights_v3": class_weights(counts_v3),
           "stat": {k: dict(v) for k, v in stat.items()},
           "audioset_segments": {c: sorted(set(v)) for c, v in seg_ids.items()},
           "webm_fallback_ids": sorted(webm)}
    with open(os.path.join(a.out, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    print("MANIFEST:", os.path.join(a.out, "MANIFEST.json"))


if __name__ == "__main__":
    main()
