#!/usr/bin/env python3
"""매니페스트에 있는 클립만 선별 다운로드한다.

FSD50K 전체(30GB+)를 받지 않는다. 로컬 디스크가 58GB인데 30GB를 받으면 전처리
중간 산출물 공간이 빠듯해진다. 실제로 쓰는 건 매니페스트에 있는 클립뿐이다.

── 배포처 ──────────────────────────────────────────────────────────────────
FSD50K : HF 미러 `Fhrozen/FSD50k`. Zenodo(공식 배포처)가 2026-09-10 오전 접속
         불가였다 (TLS 핸드셰이크는 되지만 HTTP 응답 없음). 미러가 원본과 동일함은
         표본 검증으로 확인했다 — 44.1kHz/16bit/mono PCM, 파일 크기 및 길이 일치.
US8K   : HF 미러 `MahiA/UrbanSound8K` (파일 단위 wav). 공식 배포는 Zenodo 의 단일
         tar.gz(6.0GB)라 필요한 슬라이스만 골라 받을 수 없다.
         파일마다 미러의 LFS sha256 으로 검증한다.
ESC-50 : 저자 GitHub 저장소(공식 배포처)에서 파일 단위로 받는다.
         전 파일이 5초 44.1kHz/16bit/mono 이므로 헤더로 검증한다.
미러 검증 절차와 결과는 `CLAUDE.md` 12장.

단계별로 받는다 (TASKS.md Phase 2.2):
  1) 이벤트 클래스 + 하드 네거티브   ← `--stage events`
  2) prepare_safesound.py 로 에너지 필터 실측 → 배경음 목표량 확정
  3) 배경음 클립 추가                ← `--stage background`

사용법 (WSL2):
    python3 scripts/download_clips.py --stage events --dry-run   # 용량만 확인
    python3 scripts/download_clips.py --stage events             # FSD50K
    python3 scripts/download_clips.py --source us8k              # US8K 매니페스트분
    python3 scripts/download_clips.py --source esc50             # ESC-50 매니페스트분
"""

import argparse
import csv
import hashlib
import http.client
import io
import json
import os
import sys
import time
import urllib.request
import wave

HF_BASE = "https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/clips"
US8K_REPO = "MahiA/UrbanSound8K"
US8K_BASE = f"https://huggingface.co/datasets/{US8K_REPO}/resolve/main/audios"
US8K_TREE_API = f"https://huggingface.co/api/datasets/{US8K_REPO}/tree/main/audios"
ESC50_BASE = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio"
ESC50_BYTES = 44 + 220500 * 2          # 5초 × 44.1kHz × 16bit mono + 헤더
UA = {"User-Agent": "curl/7.81"}

SOURCES = {"fsd50k": "FSD50K", "us8k": "US8K", "esc50": "ESC-50"}
DEFAULT_OUT = {"fsd50k": "data/raw/FSD50K_clips",
               "us8k": "data/raw/US8K_audio",
               "esc50": "data/raw/ESC50_audio"}


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def load_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def http_get(url, timeout=120):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fsd_subdir(fsid, sizes):
    """clip_sizes.json 기준으로 dev/eval 중 어디에 있는지 판정."""
    if fsid in sizes.get("dev", {}):
        return "dev"
    if fsid in sizes.get("eval", {}):
        return "eval"
    return None


def us8k_mirror_tree(path):
    """미러의 파일별 크기·sha256 목록. 없으면 HF API 로 받아 캐시한다."""
    if os.path.isfile(path):
        return json.load(open(path, encoding="utf-8"))
    tree = {}
    for fold in range(1, 11):
        for e in json.loads(http_get(f"{US8K_TREE_API}/fold{fold}", 60)):
            if e["type"] == "file" and e["path"].endswith(".wav"):
                tree[e["path"]] = {"size": e["size"],
                                   "sha256": (e.get("lfs") or {}).get("oid")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=0)
    return tree


# ──────────────────────────────────────────────────────────────── 검증기
def check_sha256(expected):
    def check(data):
        got = hashlib.sha256(data).hexdigest()
        return None if got == expected else f"sha256 불일치 {got[:12]} != {expected[:12]}"
    return check


def check_esc50_wav(data):
    """ESC-50 은 전 파일이 5초 44.1kHz/16bit/mono 다."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            p = (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes())
    except (wave.Error, EOFError) as e:
        return f"wav 헤더 오류: {e}"
    return None if p == (44100, 1, 2, 220500) else f"형식 불일치 {p}"


# ──────────────────────────────────────────────────────────── 대상 목록
def fsd50k_targets(rows, args, out):
    sizes = json.load(open(args.sizes, encoding="utf-8"))
    targets, missing = [], 0
    for r in rows:
        sub = fsd_subdir(r["fsid"], sizes)
        if sub is None:
            missing += 1
            continue
        targets.append({"url": f"{HF_BASE}/{sub}/{r['fsid']}.wav",
                        "dest": os.path.join(out, f"{r['fsid']}.wav"),
                        "size": sizes[sub][r["fsid"]], "cls": r["cls"],
                        "check": None})
    return targets, missing


def us8k_targets(rows, args, out):
    tree = us8k_mirror_tree(args.us8k_tree)
    fold_of = {r["slice_file_name"]: r["fold"]
               for r in csv.DictReader(open(args.us8k_csv, encoding="utf-8"))}
    targets, missing = [], 0
    for r in rows:
        name = r["clip_id"]
        fold = fold_of.get(name)
        meta = tree.get(f"audios/fold{fold}/{name}")
        if meta is None or not meta["sha256"]:
            missing += 1
            continue
        targets.append({"url": f"{US8K_BASE}/fold{fold}/{name}",
                        "dest": os.path.join(out, f"fold{fold}", name),
                        "size": meta["size"], "cls": r["cls"],
                        "check": check_sha256(meta["sha256"])})
    return targets, missing


def esc50_targets(rows, _args, out):
    return [{"url": f"{ESC50_BASE}/{r['clip_id']}",
             "dest": os.path.join(out, r["clip_id"]),
             "size": ESC50_BYTES, "cls": r["cls"], "check": check_esc50_wav}
            for r in rows], 0


TARGETS = {"fsd50k": fsd50k_targets, "us8k": us8k_targets, "esc50": esc50_targets}


def verify(data, t):
    if len(data) != t["size"]:
        # 크기가 다르면 미러가 재인코딩한 것이다. 조용히 넘어가지 않는다.
        return f"크기 불일치 {len(data)} != {t['size']}"
    return t["check"](data) if t["check"] else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/interim/manifest.csv")
    ap.add_argument("--source", choices=tuple(SOURCES), default="fsd50k")
    ap.add_argument("--sizes", default="data/raw/FSD50K_meta/clip_sizes.json")
    ap.add_argument("--us8k-csv", default="data/raw/US8K_meta/UrbanSound8K.csv")
    ap.add_argument("--us8k-tree", default="data/raw/US8K_meta/mirror_MahiA_tree.json",
                    help="US8K 미러 파일 목록 캐시 (없으면 HF API 로 생성)")
    ap.add_argument("--out", default=None, help="기본값은 소스별 data/raw/*")
    ap.add_argument("--stage", choices=("events", "hardneg", "background", "all"),
                    default="events",
                    help="events=이벤트4클래스+하드네거티브 / hardneg=하드네거티브만 "
                         "/ background=일반 배경음(3단계, 매니페스트 확장 후)")
    ap.add_argument("--cls", nargs="+", default=None, help="이 클래스만 (예: siren)")
    ap.add_argument("--dry-run", action="store_true",
                    help="다운로드하지 않고 대상 수와 예상 용량만 출력")
    ap.add_argument("--limit", type=int, default=0, help="처음 N개만 (디버깅용)")
    args = ap.parse_args()
    out = args.out or DEFAULT_OUT[args.source]

    rows = [r for r in load_manifest(args.manifest)
            if r["source"] == SOURCES[args.source]]

    if args.stage == "events":
        rows = [r for r in rows
                if r["cls"] != "background" or r["note"].startswith("hard_neg")]
    elif args.stage == "hardneg":
        rows = [r for r in rows if r["note"].startswith("hard_neg")]
    elif args.stage == "background":
        rows = [r for r in rows
                if r["cls"] == "background" and not r["note"].startswith("hard_neg")]
        if not rows and args.source == "fsd50k":
            sys.exit(
                "[중단] 매니페스트에 일반 배경음 항목이 없다.\n"
                "  일반 배경음은 3단계다. prepare_safesound.py 로 에너지 필터 실측\n"
                "  윈도우 수를 확정한 뒤, 그 2~3배를 목표로 배경음을 샘플링해\n"
                "  매니페스트에 추가하고 나서 이 단계를 실행할 것.")
    if args.cls:
        rows = [r for r in rows if r["cls"] in args.cls]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        sys.exit("[중단] 조건에 맞는 매니페스트 행이 없다.")

    targets, missing = TARGETS[args.source](rows, args, out)

    by_cls = {}
    for t in targets:
        n, b = by_cls.get(t["cls"], (0, 0))
        by_cls[t["cls"]] = (n + 1, b + t["size"])

    print(f"소스: {SOURCES[args.source]}  단계: {args.stage}  → {out}")
    print(f"{'클래스':<14}{'클립':>8}{'용량':>12}")
    print("-" * 34)
    for c, (n, b) in sorted(by_cls.items(), key=lambda x: -x[1][1]):
        print(f"{c:<14}{n:>8}{human(b):>12}")
    print("-" * 34)
    print(f"{'합계':<14}{len(targets):>8}{human(sum(t['size'] for t in targets)):>12}")
    if missing:
        print(f"  (배포처 목록에 없는 항목 {missing}개는 제외)")

    if args.dry_run:
        return

    done = skipped = failed = 0
    t0 = time.time()
    for i, t in enumerate(targets, 1):
        dest = t["dest"]
        if os.path.isfile(dest) and os.path.getsize(dest) == t["size"]:
            with open(dest, "rb") as f:
                if verify(f.read(), t) is None:
                    skipped += 1
                    continue
        try:
            data = http_get(t["url"])
        except (OSError, http.client.HTTPException) as e:
            print(f"  실패 {os.path.basename(dest)}: {e}", file=sys.stderr)
            failed += 1
            continue
        err = verify(data, t)
        if err:
            print(f"  {os.path.basename(dest)}: {err}", file=sys.stderr)
            failed += 1
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest + ".part", "wb") as f:
            f.write(data)
        os.replace(dest + ".part", dest)
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
