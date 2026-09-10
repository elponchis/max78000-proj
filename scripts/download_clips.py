#!/usr/bin/env python3
"""매니페스트에 있는 클립만 선별 다운로드한다.

FSD50K 전체(30GB+)를 받지 않는다. 로컬 디스크가 58GB인데 30GB를 받으면 전처리
중간 산출물 공간이 빠듯해진다. 실제로 쓰는 건 매니페스트에 있는 클립뿐이다.

⚠️ Zenodo(공식 배포처)는 2026-09-10 기준 접속 불가다 (TLS 핸드셰이크는 되지만
HTTP 응답 없음). HuggingFace 미러 `Fhrozen/FSD50k` 를 쓴다. 미러가 원본과
동일함은 표본 검증으로 확인했다 — 44.1kHz/16bit/mono PCM, 파일 크기 및 길이 일치
(`CLAUDE.md` 12장).

단계별로 받는다 (TASKS.md Phase 2.2):
  1) 이벤트 클래스 + 하드 네거티브   ← `--stage events`
  2) prepare_safesound.py 로 에너지 필터 실측 → 배경음 목표량 확정
  3) 배경음 클립 추가                ← `--stage background`

사용법 (WSL2):
    python3 scripts/download_clips.py --stage events --dry-run   # 용량만 확인
    python3 scripts/download_clips.py --stage events
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request

HF_BASE = "https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/clips"
UA = {"User-Agent": "curl/7.81"}


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def load_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fsd_subdir(fsid, sizes):
    """clip_sizes.json 기준으로 dev/eval 중 어디에 있는지 판정."""
    if fsid in sizes.get("dev", {}):
        return "dev"
    if fsid in sizes.get("eval", {}):
        return "eval"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/interim/manifest.csv")
    ap.add_argument("--sizes", default="data/raw/FSD50K_meta/clip_sizes.json")
    ap.add_argument("--out", default="data/raw/FSD50K_clips")
    ap.add_argument("--stage", choices=("events", "hardneg", "background", "all"),
                    default="events",
                    help="events=이벤트4클래스+하드네거티브 / hardneg=하드네거티브만 "
                         "/ background=일반 배경음(3단계, 매니페스트 확장 후)")
    ap.add_argument("--dry-run", action="store_true",
                    help="다운로드하지 않고 대상 수와 예상 용량만 출력")
    ap.add_argument("--limit", type=int, default=0, help="처음 N개만 (디버깅용)")
    args = ap.parse_args()

    rows = load_manifest(args.manifest)
    sizes = json.load(open(args.sizes, encoding="utf-8"))

    # 이 스크립트는 FSD50K 만 받는다. US8K / ESC-50 은 별도 배포처에서 받는다.
    rows = [r for r in rows if r["source"] == "FSD50K"]

    if args.stage == "events":
        rows = [r for r in rows
                if r["cls"] != "background" or r["note"].startswith("hard_neg")]
    elif args.stage == "hardneg":
        rows = [r for r in rows if r["note"].startswith("hard_neg")]
    elif args.stage == "background":
        rows = [r for r in rows
                if r["cls"] == "background" and not r["note"].startswith("hard_neg")]
        if not rows:
            sys.exit(
                "[중단] 매니페스트에 일반 배경음 항목이 없다.\n"
                "  일반 배경음은 3단계다. prepare_safesound.py 로 에너지 필터 실측\n"
                "  윈도우 수를 확정한 뒤, 그 2~3배를 목표로 배경음을 샘플링해\n"
                "  매니페스트에 추가하고 나서 이 단계를 실행할 것.")

    if args.limit:
        rows = rows[:args.limit]

    # 예상 용량 — clip_sizes.json 에 실제 바이트가 있으므로 추정이 아니라 실측이다
    total = 0
    targets = []
    missing = 0
    for r in rows:
        sub = fsd_subdir(r["fsid"], sizes)
        if sub is None:
            missing += 1
            continue
        size = sizes[sub][r["fsid"]]
        total += size
        targets.append((r["fsid"], sub, size, r["cls"]))

    by_cls = {}
    for _f, _s, size, c in targets:
        n, b = by_cls.get(c, (0, 0))
        by_cls[c] = (n + 1, b + size)

    print(f"단계: {args.stage}")
    print(f"{'클래스':<14}{'클립':>8}{'용량':>12}")
    print("-" * 34)
    for c, (n, b) in sorted(by_cls.items(), key=lambda x: -x[1][1]):
        print(f"{c:<14}{n:>8}{human(b):>12}")
    print("-" * 34)
    print(f"{'합계':<14}{len(targets):>8}{human(total):>12}")
    if missing:
        print(f"  (clip_sizes.json 에 없는 항목 {missing}개는 제외)")

    if args.dry_run:
        return

    os.makedirs(args.out, exist_ok=True)
    done = skipped = failed = 0
    t0 = time.time()
    for i, (fsid, sub, size, _c) in enumerate(targets, 1):
        dest = os.path.join(args.out, f"{fsid}.wav")
        if os.path.isfile(dest) and os.path.getsize(dest) == size:
            skipped += 1
            continue
        url = f"{HF_BASE}/{sub}/{fsid}.wav"
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  실패 {fsid}: {e}", file=sys.stderr)
            failed += 1
            continue
        if len(data) != size:
            # 크기가 다르면 미러가 재인코딩한 것이다. 조용히 넘어가지 않는다.
            print(f"  크기 불일치 {fsid}: {len(data)} != {size}", file=sys.stderr)
            failed += 1
            continue
        with open(dest, "wb") as f:
            f.write(data)
        done += 1
        if i % 100 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(targets)}  받음 {done} 건너뜀 {skipped} "
                  f"실패 {failed}  {el:.0f}s", flush=True)

    print(f"\n완료: 받음 {done} / 건너뜀 {skipped} / 실패 {failed}")
    if failed:
        print("실패분은 스크립트를 다시 실행하면 이어받는다.")


if __name__ == "__main__":
    main()
