#!/usr/bin/env python3
"""추출된 윈도우 wav 의 무음·잘림 점검.

청취 표본을 귀로 듣기 전에 **기계로 먼저 거르는** 도구다. 에너지 필터를 통과했는데도
실제로는 무음에 가까운 파일, 그리고 원본 길이를 넘겨 0으로 패딩된 파일을 찾는다.

재는 것 (파일마다):
  길이 / 샘플레이트 / peak dBFS / RMS dBFS / **정확히 0인 샘플의 비율**
0 샘플 비율이 핵심이다. RMS 가 낮은 것은 조용한 녹음일 수도 있지만, 0 이 대량으로
섞여 있으면 그건 녹음이 아니라 **패딩이거나 잘린 구간**이다. 둘을 구별해야 한다.

표본 매니페스트(`samples.csv`)가 있으면 조인해서 원본 클립 ID·원본 길이·윈도
시작 오프셋·onset 점수를 함께 찍는다. 어느 클립의 어느 지점에서 나온 무음인지
바로 추적할 수 있다.

**윈도 끝이 원본 길이를 넘는 항목은 따로 표시한다.** 파이프라인은 1초보다 짧은
클립을 0 으로 채우므로(`prepare_safesound.pad_to_win`), 이런 항목은 뒤쪽이 통째로
무음이다. 학습에 넣을지 판단이 필요하다.

사용법 (WSL2):
    python3 tools/audit_silence.py data/interim/listen_glass_test
    python3 tools/audit_silence.py <디렉터리> --bottom 30
    python3 tools/audit_silence.py <디렉터리> --samples <다른 경로>/samples.csv
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sys.exit("[에러] soundfile 미설치. WSL2에서: python3 -m pip install --user soundfile")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from prepare_safesound import SR, WIN, clip_path          # noqa: E402

NEG_INF = float("-inf")


def db(x):
    """진폭 → dBFS (전체 스케일 1.0 기준). 0 은 −inf."""
    return 20.0 * math.log10(x) if x > 0 else NEG_INF


def fmt_db(v):
    return "  -inf" if v == NEG_INF else f"{v:6.1f}"


def measure(path):
    """wav 하나의 기본 통계. 실패하면 None."""
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as e:                                   # noqa: BLE001
        print(f"  읽기 실패 {os.path.basename(path)}: {e}", file=sys.stderr)
        return None
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = len(x)
    peak = float(np.abs(x).max()) if n else 0.0
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if n else 0.0
    return {
        "file": os.path.basename(path),
        "n": n,
        "sr": sr,
        "dur": n / sr if sr else 0.0,
        "peak_db": db(peak),
        "rms_db": db(rms),
        # 정확히 0 인 샘플. PCM16 로 저장했으므로 양자화 후 0 이 된 샘플도 포함된다.
        "zero_frac": float((x == 0.0).mean()) if n else 1.0,
    }


def load_samples(path):
    """export_audit / export_samples 가 남긴 samples.csv → {파일명: 행}."""
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        # export_audit 는 파일명만, export_samples 는 `클래스/파일명` 을 넣는다
        out[os.path.basename(r["file"])] = r
    return out


def source_length(row, roots):
    """원본 클립의 16kHz 기준 샘플 수. 알 수 없으면 None.

    헤더만 읽는다(`sf.info`). 44.1kHz 원본을 16kHz 로 다시 샘플링했을 때의 길이는
    `ceil(frames × 16000 / sr)` 로 계산한다 — `resample_poly` 의 출력 길이와 같다.
    """
    p = clip_path(row, roots)
    if not p:
        return None, None
    try:
        info = sf.info(p)
    except Exception:                                        # noqa: BLE001
        return None, None
    return math.ceil(info.frames * SR / info.samplerate), info.duration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav_dir", help="점검할 wav 디렉터리 (하위 디렉터리 포함)")
    ap.add_argument("--samples", default=None,
                    help="표본 매니페스트 CSV (기본: wav_dir/samples.csv)")
    ap.add_argument("--manifest", default="data/interim/manifest.csv",
                    help="클립 매니페스트 — 원본 길이·출처 조인용")
    ap.add_argument("--fsd-dir", default="data/raw/FSD50K_clips")
    ap.add_argument("--us8k-dir", default="data/raw/US8K_audio")
    ap.add_argument("--esc50-dir", default="data/raw/ESC50_audio")
    ap.add_argument("--bottom", type=int, default=20,
                    help="RMS 오름차순 하위 N개를 출력 (기본 20)")
    args = ap.parse_args()

    paths = []
    for root, _dirs, files in os.walk(args.wav_dir):
        paths += [os.path.join(root, f) for f in files if f.lower().endswith(".wav")]
    if not paths:
        sys.exit(f"[에러] wav 가 없다: {args.wav_dir}")
    paths.sort()

    stats = [s for s in (measure(p) for p in paths) if s]
    samples = load_samples(args.samples or os.path.join(args.wav_dir, "samples.csv"))
    roots = {"fsd": args.fsd_dir, "us8k": args.us8k_dir, "esc50": args.esc50_dir}

    manifest = {}
    if os.path.isfile(args.manifest):
        with open(args.manifest, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                manifest[r["clip_id"]] = r

    # ── 조인: 표본 매니페스트 → 원본 길이·윈도 오프셋
    src_cache = {}
    for s in stats:
        row = samples.get(s["file"])
        s["row"] = row
        s["start"] = int(row["start_sample"]) if row else None
        s["end"] = s["start"] + WIN if row else None
        s["src_n"] = s["src_dur"] = None
        if row:
            cid = row["clip_id"]
            if cid not in src_cache:
                mrow = manifest.get(cid) or row
                src_cache[cid] = source_length(mrow, roots)
            s["src_n"], s["src_dur"] = src_cache[cid]
        s["over"] = (s["src_n"] is not None and s["end"] is not None
                     and s["end"] > s["src_n"] + 1)     # 1샘플 오차 허용

    # ── 1. 전체 요약
    n_sr = {}
    for s in stats:
        n_sr[s["sr"]] = n_sr.get(s["sr"], 0) + 1
    durs = np.array([s["dur"] for s in stats])
    rmss = np.array([s["rms_db"] for s in stats])
    finite = rmss[np.isfinite(rmss)]
    zf = np.array([s["zero_frac"] for s in stats])
    print(f"=== {args.wav_dir} — wav {len(stats)}개 ===")
    print("  샘플레이트: " + ", ".join(f"{k}Hz {v}개" for k, v in sorted(n_sr.items())))
    print(f"  길이(초):   최소 {durs.min():.3f} / 중앙 {np.median(durs):.3f} "
          f"/ 최대 {durs.max():.3f}")
    print(f"  RMS dBFS:   최소 {finite.min():6.1f} / 중앙 {np.median(finite):6.1f} "
          f"/ 최대 {finite.max():6.1f}" + (f"  (−inf {int((~np.isfinite(rmss)).sum())}개)"
                                          if len(finite) != len(rmss) else ""))
    print(f"  0 샘플 비율: 중앙 {100*np.median(zf):.2f}% / 최대 {100*zf.max():.2f}%")
    if not samples:
        print("  ⚠️ samples.csv 를 찾지 못해 원본 조인은 생략한다")

    # ── 2. RMS 하위 N
    worst = sorted(stats, key=lambda s: s["rms_db"])[:args.bottom]
    print(f"\n=== RMS 오름차순 하위 {len(worst)}개 ===")
    head = (f"{'#':>3} {'RMS':>6} {'peak':>6} {'0샘플':>7} {'길이':>6} "
            f"{'원본ID':>10} {'원본길이':>9} {'윈도오프셋':>11} {'윈도끝':>9} {'onset':>6}")
    print(head)
    print("-" * len(head))
    for i, s in enumerate(worst, 1):
        row = s["row"]
        cid = row["clip_id"] if row else "-"
        onset = row.get("onset_window", "") if row else ""
        src = f"{s['src_n']}" if s["src_n"] else "?"
        if s["src_dur"]:
            src += f"({s['src_dur']:.1f}s)"
        mark = " ⚠︎넘침" if s["over"] else ""
        print(f"{i:>3} {fmt_db(s['rms_db'])} {fmt_db(s['peak_db'])} "
              f"{100*s['zero_frac']:6.2f}% {s['dur']:6.3f} "
              f"{cid[:10]:>10} {src:>9} {str(s['start']):>11} {str(s['end']):>9} "
              f"{onset:>6}{mark}")
        print(f"     {s['file']}")

    # ── 3. 윈도 끝이 원본을 넘는 항목 (하위 N 밖도 전부)
    over = [s for s in stats if s["over"]]
    print(f"\n=== 윈도 끝 > 원본 길이: {len(over)}개 ===")
    if not over:
        print("  없음. 모든 윈도가 원본 안에 들어간다 (패딩 없음).")
    else:
        print("  원본보다 뒤쪽이 0 으로 채워진 윈도다. 패딩 길이만큼 무음이 섞여 있다.")
        print(f"  {'파일':<52}{'원본':>9}{'윈도끝':>9}{'패딩':>8}{'0샘플':>8}")
        print("  " + "-" * 85)
        for s in sorted(over, key=lambda s: s["src_n"] - s["end"]):
            pad = s["end"] - s["src_n"]
            print(f"  {s['file'][:52]:<52}{s['src_n']:>9}{s['end']:>9}"
                  f"{pad:>8}{100*s['zero_frac']:>7.2f}%")

    # ── 4. 무음 의심 요약 — 판단 기준을 숫자로 남긴다
    quiet = [s for s in stats if s["rms_db"] < -50]
    zeros = [s for s in stats if s["zero_frac"] > 0.10]
    print(f"\n=== 요약 ===")
    print(f"  RMS < −50dBFS      : {len(quiet)}개")
    print(f"  0 샘플 10% 초과     : {len(zeros)}개")
    print(f"  윈도 끝 > 원본 길이 : {len(over)}개")
    print("  RMS 가 낮은 것과 0 이 많은 것은 다르다. 전자는 조용한 녹음일 수 있지만")
    print("  후자는 패딩이거나 잘린 구간이다 — 후자를 먼저 볼 것.")


if __name__ == "__main__":
    main()
