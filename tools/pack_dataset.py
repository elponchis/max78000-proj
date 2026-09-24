#!/usr/bin/env python3
"""샤드를 tar 하나로 묶는다 — Colab 업로드용.

Drive 에는 **이 tar 하나만** 올린다. 원본 오디오 30GB 를 올리지 않는 것이
CLAUDE.md 6장의 전제다 (로컬 전처리 → 캐시만 업로드).

함께 넣는 것
  - `MANIFEST.json` — 클래스·split 별 창 수, 전처리 설정, 레포 커밋 해시.
    **학습 로그와 데이터를 나중에 대조하려면 이게 있어야 한다.** 수치만 있고
    어떤 설정으로 만든 건지 모르는 캐시는 논문에 쓸 수 없다
  - `.sha256` — Colab 에서 받은 tar 가 온전한지 확인용

사용법 (WSL2):
    python3 tools/pack_dataset.py
    python3 tools/pack_dataset.py --out /mnt/c/dev/safesound-v1.tar.gz
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import prepare_safesound as P                              # noqa: E402


def git(*args):
    try:
        return subprocess.run(("git",) + args, cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def counts(root):
    """클래스·split 별 창 수와 원본 수 — 인덱스 CSV 에서 센다."""
    out, origins = {}, {}
    for split in P.SPLITS:
        for cls in P.CLASSES:
            idx = os.path.join(root, split, cls, "index.csv")
            if not os.path.isfile(idx):
                continue
            with open(idx, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            out[f"{cls}/{split}"] = len(rows)
            origins[f"{cls}/{split}"] = len({r["fsid"] for r in rows})
    return out, origins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/safesound")
    ap.add_argument("--out", default="data/safesound-v1.tar.gz")
    ap.add_argument("--tag", default="dataset-v1")
    a = ap.parse_args()

    if not os.path.isdir(a.root):
        sys.exit(f"[에러] 샤드가 없다: {a.root}. "
                 "먼저 prepare_safesound.py 를 실행할 것.")
    win, orig = counts(a.root)
    if not win:
        sys.exit(f"[에러] {a.root} 에 index.csv 가 없다.")

    args = P.default_args()
    manifest = {
        "tag": a.tag,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "counts": win,
        "originals": orig,
        "window": {"samples": P.WIN, "stored": P.STORE, "margin": P.MARGIN,
                   "sample_rate": P.SR, "dtype": "int8", "reshape": [128, 128]},
        "settings": {
            "hop_ms": args.hop_ms, "policy": args.policy,
            "floor_zero": args.floor_zero, "rel_ratio": args.rel_ratio,
            "max_win": args.max_win, "bg_quiet_frac": args.bg_quiet_frac,
            "tag_thr": args.thr, "tag_bg_thr": args.tag_bg_thr,
            "fill_mode": args.fill_mode, "fill_snr_db": args.fill_snr_db,
            "fill_bed_min_db": args.fill_bed_min_db,
            "zero_run_min": args.zero_run_min,
            "glass_onset_thr": args.glass_onset_thr,
        },
        "notes": "생성: tools/pack_dataset.py. 설정 근거는 CLAUDE.md 5·7장, "
                 "docs/results/listening-verification.md",
    }
    if manifest["dirty"]:
        print("⚠️ 커밋되지 않은 변경이 있다. 재현하려면 먼저 커밋할 것.")

    mpath = os.path.join(a.root, "MANIFEST.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    print(f"묶는 중 → {a.out}")
    with tarfile.open(a.out, "w:gz") as tf:
        for name in sorted(os.listdir(a.root)):
            tf.add(os.path.join(a.root, name), arcname=name)

    h = hashlib.sha256()
    with open(a.out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    with open(a.out + ".sha256", "w", encoding="utf-8") as f:
        f.write(f"{h.hexdigest()}  {os.path.basename(a.out)}\n")

    size = os.path.getsize(a.out)
    print(f"\n{'클래스/split':<22}{'창':>8}{'원본':>8}")
    print("-" * 38)
    for k in sorted(win):
        print(f"{k:<22}{win[k]:>8}{orig[k]:>8}")
    print("-" * 38)
    print(f"{'합계':<22}{sum(win.values()):>8}{sum(orig.values()):>8}")
    print(f"\ntar   {a.out}  ({size / 1e6:.1f}MB)")
    print(f"sha256 {h.hexdigest()}")
    print(f"매니페스트 {mpath}")
    print("\nDrive 에 tar 와 .sha256 을 함께 올릴 것 "
          "(colab/train_baseline.md 셀 4 가 해시를 확인한다).")


if __name__ == "__main__":
    main()
