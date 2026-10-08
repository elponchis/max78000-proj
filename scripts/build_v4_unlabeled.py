#!/usr/bin/env python3
"""v4+D 의 **라벨 없는 창** — 받아 둔 AudioSet train_v2 클립 전체를 1초 창·250 ms 간격으로 (v4-distill-design.md 7절).

가공은 v3 빌더(`build_v3_windows.py`)와 같은 경로: 16 kHz → 디지털 0 런 베드 채움(train_v2 배경 베드) → int8 (np.clip 포화) →
1.2 s 행 저장(`store_row`). 창 선별은 없다 — int8 전부 0 인 창만 뺀다. 라벨·정의는 쓰지 않는다 (교사 확률만으로 학습).
분리: 클립 ytid ∩ external_v1 / sampling_list / eval strong = 0 이어야 한다 (겹치면 중단).

출력: <out>/train/unlabeled/shard_NNNN.npy + index.csv (clip_id, fsid=yt:<ytid>, start_sample, margins, note), MANIFEST.json.

사용 (WSL2, 시스템 python3 또는 ai8x venv):  python3 scripts/build_v4_unlabeled.py [--dry-run]
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
sys.path[:0] = [os.path.join(REPO, "scripts")]
import prepare_safesound as P                      # noqa: E402
import build_v3_windows as B                       # noqa: E402

SR, WIN, HOP = P.SR, P.WIN, P.SR // 4
EXT = os.path.expanduser("~/safesound-external")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default=os.path.join(REPO, "tools", "external_set", "train_v2_list.csv"))
    ap.add_argument("--raw", default=os.path.join(EXT, "train_v2", "raw"))
    ap.add_argument("--state", default=os.path.join(EXT, "train_v2", "download_state.csv"))
    ap.add_argument("--meta", default=os.path.join(EXT, "meta"))
    ap.add_argument("--out", default=os.path.join(REPO, "data", "processed", "safesound_v4u"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rows = {r["segment_id"]: r for r in csv.DictReader(open(a.list, encoding="utf-8"))}
    state = {r["segment_id"]: r for r in csv.DictReader(open(a.state, encoding="utf-8"))}
    clips = sorted(fn[:-4] for fn in os.listdir(a.raw) if fn.endswith(".wav"))
    yt = {rows[s]["ytid"] if s in rows else s.rsplit("_", 1)[0] for s in clips}
    banned = {r["ytid"] for r in csv.DictReader(open(os.path.join(REPO, "tools", "external_set", "sampling_list.csv"), encoding="utf-8"))}
    banned |= {r["ytid"] for r in csv.DictReader(open(os.path.join(EXT, "external_v1", "meta.csv"), encoding="utf-8"))}
    banned_eval = {s.rsplit("_", 1)[0] for s in B.load_strong(os.path.join(a.meta, "audioset_eval_strong.tsv"))}
    hit = yt & (banned | banned_eval)
    if hit:
        sys.exit(f"[에러] external_v1/eval 영상과 겹친다 ({len(hit)}): {sorted(hit)[:5]}")
    print(f"클립 {len(clips)} (영상 {len(yt)}), external_v1·sampling_list {len(banned)} / eval {len(banned_eval)} 과 겹침 0")

    args = B.fill_args()
    bg_paths = [os.path.join(a.raw, s + ".wav") for s in clips
                if s in rows and rows[s]["group"] == "background" and state.get(s, {}).get("status") == "ok"]
    pool = B.AudioSetBedPool(bg_paths, args)
    webm = {s for s, r in state.items() if r["status"] == "ok" and r["reason"] == "webm_trim"}

    out_dir = os.path.join(a.out, "train", "unlabeled")
    if not a.dry_run:
        os.makedirs(out_dir, exist_ok=True)
        for fn in os.listdir(out_dir):
            if fn.startswith("shard_"):
                os.remove(os.path.join(out_dir, fn))
    stat = collections.Counter()
    index, buf, sh = [], [], 0
    for k, sid in enumerate(clips):
        x = P.load_audio(os.path.join(a.raw, sid + ".wav"))
        if x is None or len(x) < WIN:
            stat["짧음/실패"] += 1
            continue
        rr = {"split": "train", "clip_id": sid}
        x, finfo = P.fill_zero_runs(x, rr, args, pool)
        if finfo:
            stat["채움 클립"] += 1
        ytid = rows[sid]["ytid"] if sid in rows else sid.rsplit("_", 1)[0]
        grp = rows[sid]["group"] if sid in rows else "?"
        for t in np.arange(0, len(x) - WIN + 1, HOP, dtype=np.int64):
            t = int(t)
            w8 = P.to_int8(x[t:t + WIN])
            if not np.any(w8):
                stat["전부 0 창 제외"] += 1
                continue
            stat["창"] += 1
            stat["창:" + grp] += 1
            if a.dry_run:
                continue
            row, left, right = P.store_row(x, t, rr, args, pool)
            if len(buf) == P.SHARD:
                np.save(os.path.join(out_dir, f"shard_{sh:04d}.npy"), np.stack(buf)); sh += 1; buf = []
            index.append({"shard": sh, "row": len(buf), "clip_id": sid, "fsid": "yt:" + ytid, "start_sample": t,
                          "left_margin": left, "right_margin": right,
                          "note": f"unlabeled;group={grp};webm_trim={int(sid in webm)}" + (";" + P.fill_note(finfo) if finfo else "")})
            buf.append(row)
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(clips)} 클립, 창 {stat['창']:,}", flush=True)
    print("집계:", dict(stat))
    if a.dry_run:
        return
    if buf:
        np.save(os.path.join(out_dir, f"shard_{sh:04d}.npy"), np.stack(buf))
    with open(os.path.join(out_dir, "index.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["shard", "row", "clip_id", "fsid", "start_sample", "left_margin", "right_margin", "note"])
        w.writeheader(); w.writerows(index)

    def git(*g):
        try:
            return subprocess.run(("git",) + g, cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:                                   # noqa: BLE001
            return ""
    man = {"tag": "dataset-v4u", "created": datetime.now().astimezone().isoformat(timespec="seconds"), "commit": git("rev-parse", "HEAD"),
           "rules": {"hop": HOP, "selection": "none (int8 all-zero windows dropped)", "fill": vars(args), "labels": "none"},
           "n_clips": len(clips), "n_videos": len(yt), "n_windows": len(index), "shards": sh + 1, "stat": dict(stat),
           "separation": {"external_v1+sampling_list": len(banned), "eval_strong": len(banned_eval), "overlap": 0}}
    with open(os.path.join(a.out, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    print(f"창 {len(index):,}, 샤드 {sh + 1} → {out_dir}")


if __name__ == "__main__":
    main()
