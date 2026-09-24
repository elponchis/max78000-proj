#!/usr/bin/env python3
"""클래스별 "정확히 0인 샘플" 비율 분포 — 지름길 학습 검사.

FSD50K 원본 중 일부는 노이즈 게이트로 편집돼 디지털 0 구간을 갖는다. 그 자체는
데이터 잘못이 아니지만, **0 구간이 특정 클래스에만 몰려 있으면 모델이 "정확한 0이
있다 → 그 클래스"라는 지름길을 배운다.** 마이크 스트림에는 디지털 0 이 나오지
않으므로 실기기에서는 절대 재현되지 않는 단서다. train 에도 있으므로 test 쪽
패딩보다 위험하다.

`prepare_safesound` 의 **실제 선택 파이프라인**을 그대로 돌려, 데이터셋에 실릴
윈도우만 대상으로 잰다. 표본이므로 클래스당 `--per-class` 개 클립만 본다.

판정 기준
  · 분포가 클래스 간에 겹친다            → 그대로 둔다
  · 특정 클래스에만 0 이 몰려 있다        → 배경음에도 같은 비율로 게이트된 클립을
                                           섞거나, 미세한 디더로 0 을 깬다

사용법 (WSL2):
    python3 tools/zero_ratio_by_class.py --per-class 250
    python3 tools/zero_ratio_by_class.py --per-class 250 --cls glass background
"""

import argparse
import os
import sys

from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import prepare_safesound as P                              # noqa: E402


def pct(v, q):
    return np.percentile(v, q) if len(v) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/interim/manifest.csv")
    ap.add_argument("--fsd-dir", default="data/raw/FSD50K_clips")
    ap.add_argument("--us8k-dir", default="data/raw/US8K_audio")
    ap.add_argument("--esc50-dir", default="data/raw/ESC50_audio")
    ap.add_argument("--overrides", default="data_overrides.csv")
    ap.add_argument("--per-class", type=int, default=250,
                    help="클래스별 표본 클립 수 (0=전량)")
    ap.add_argument("--cls", nargs="+", default=P.CLASSES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--fill-mode", choices=["background", "zero"], default="background",
                    help="짧은 클립 채움 방식. zero 로 두면 변경 전 상태를 잰다")
    ap.add_argument("--floor-zero", type=float, default=1.01,
                    help="int8 0비율 상한. 기본 1.01 = 하한을 끄고 **분포 전체**를 "
                         "본다 (임계값을 정하려면 잘리지 않은 분포가 필요하다)")
    a = ap.parse_args()

    # 파이프라인 기본값을 그대로 쓴다 (`prepare_safesound.default_args`).
    # 도구가 설정을 따로 들고 있으면 스크립트와 어긋난다.
    args = P.default_args(manifest=a.manifest, fsd_dir=a.fsd_dir, us8k_dir=a.us8k_dir,
                          esc50_dir=a.esc50_dir, overrides=a.overrides,
                          threads=a.threads, fill_mode=a.fill_mode,
                          floor_zero=a.floor_zero)
    print(f"설정: int8 0비율 ≤ {args.floor_zero}, rel {args.rel_ratio}, hop {args.hop_ms}ms, "
          f"채움 {args.fill_mode}, 태거 {args.thr}")
    roots = args.roots
    rows = P.iter_rows(a.manifest, roots, 0, a.per_class, a.seed)
    rows = [(r, p) for r, p in rows if r["cls"] in a.cls]
    print(f"표본 클립 {len(rows)}개 (클래스당 최대 {a.per_class or '전량'})")

    tagger = P.Tagger(args)
    floor = args.floor_zero
    zr = defaultdict(list)          # 실효 클래스 -> 윈도우별 0 비율 (원본 float)
    zq = defaultdict(list)          # 같은 것을 int8 양자화 후로
    rms_db = defaultdict(list)
    for i, (r, path) in enumerate(rows, 1):
        x = P.load_audio(path)
        if x is None:
            continue
        res = P.analyze_clip(r, x, args, tagger)
        if res["cls"] is None:
            continue
        picked = P.choose(res, floor, args.rel_ratio, args.max_win,
                          P.tag_ok(res, args.thr, args.tag_bg_thr))[0]
        xp = P.prepare_audio(x, r, args, args.pool)[0]
        for j in picked:
            s = int(res["starts"][j])
            w = xp[s:s + P.WIN]
            zr[res["cls"]].append(float((w == 0.0).mean()))
            # 모델이 실제로 보는 것은 int8 이다. 조용한 구간은 원본이 0 이 아니어도
            # 양자화 후 0 이 되므로, 두 영역을 나란히 봐야 판단이 선다.
            # 캐시 생성과 **같은 키**로 디더를 적용한다 — 디더를 빼고 재면 실제
            # 학습 데이터가 아니라 그 전 단계를 재는 셈이 된다.
            zq[res["cls"]].append(float((P.to_int8(w) == 0).mean()))
            rms_db[res["cls"]].append(20 * np.log10(max(float(res["rms"][j]), 1e-12)))
        if i % 250 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    print(f"\n{'클래스':<12}{'창':>7}{'0비율 p50':>11}{'p90':>8}{'p99':>8}{'최대':>8}"
          f"{'>1%':>8}{'>10%':>8}{'>50%':>8}")
    print("-" * 79)
    for c in a.cls:
        v = np.array(zr.get(c, []))
        if not len(v):
            continue
        print(f"{c:<12}{len(v):>7}{100*pct(v,50):>10.2f}%{100*pct(v,90):>7.2f}%"
              f"{100*pct(v,99):>7.2f}%{100*v.max():>7.2f}%"
              f"{100*(v>0.01).mean():>7.1f}%{100*(v>0.10).mean():>7.1f}%"
              f"{100*(v>0.50).mean():>7.1f}%")
    print("\n  p50/p90/p99/최대 = 창 하나의 0 샘플 비율 분포")
    print("  >1% / >10% / >50% = 그 비율을 넘는 창의 **비중**")

    print(f"\n★ int8 기준 (모델이 보는 값) — 1 LSB = −42.1dBFS, "
          f"채움 {args.fill_mode}")
    print(f"{'클래스':<12}{'창':>7}{'int8 0비율 p50':>16}{'p90':>8}{'p99':>8}"
          f"{'>10%':>8}{'>50%':>8}")
    print("-" * 67)
    for c in a.cls:
        v = np.array(zq.get(c, []))
        if not len(v):
            continue
        print(f"{c:<12}{len(v):>7}{100*pct(v,50):>15.2f}%{100*pct(v,90):>7.2f}%"
              f"{100*pct(v,99):>7.2f}%{100*(v>0.10).mean():>7.1f}%"
              f"{100*(v>0.50).mean():>7.1f}%")

    # 베드를 깔 수 없는 창 — 이벤트가 너무 조용하면 "이벤트−10dB" 베드가
    # 1 LSB(−42.1dBFS) 아래가 되어 int8 에서 사라진다. 경계는 약 −32dBFS 다.
    print(f"\n[베드 표현 한계] 창 RMS < −32dBFS = 10dB 아래 베드가 int8 에서 소멸")
    print(f"  {'클래스':<12}{'창':>7}{'<−32dBFS':>11}{'비중':>8}")
    for c in a.cls:
        v = np.array(rms_db.get(c, []))
        if len(v):
            print(f"  {c:<12}{len(v):>7}{int((v < -32).sum()):>11}"
                  f"{100*(v < -32).mean():>7.1f}%")

    ev = [c for c in a.cls if c in P.EVENTS and len(zr.get(c, []))]
    if "background" in zr and ev:
        print("\n[지름길 위험도] 이벤트 클래스 vs 배경음 — >10% 창의 비중")
        print(f"  {'클래스':<12}{'원본':>10}{'배수':>8}{'int8':>10}{'배수':>8}")
        b_f = max((np.array(zr["background"]) > 0.10).mean(), 1e-9)
        b_q = max((np.array(zq["background"]) > 0.10).mean(), 1e-9)
        for c in ev + ["background"]:
            f = (np.array(zr[c]) > 0.10).mean()
            q = (np.array(zq[c]) > 0.10).mean()
            print(f"  {c:<12}{100*f:>9.1f}%{f/b_f:>7.1f}배{100*q:>9.1f}%{q/b_q:>7.1f}배")
        print("\n  배수가 1에 가까우면 분포가 겹치는 것이라 그대로 둔다.")
        print("  이벤트 쪽만 크게 높으면 0 자체가 클래스 단서가 된다 — 조치 필요.")


if __name__ == "__main__":
    main()
