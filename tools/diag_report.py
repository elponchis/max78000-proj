#!/usr/bin/env python3
"""`diag_external_drop.py` 의 JSON → 마크다운 표 (external-drop-diagnosis.md 2절용)."""

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(REPO, "data", "external_eval", "diag_drop.json"),
                   encoding="utf-8"))
EV = ["siren", "glass", "scream", "dog_bark"]
MIX = [f"mix-{p}-{s}" for p in ("gen", "hard") for s in (20, 10, 5, 0)]
ORDER = MIX + ["resample", "aac", "opus", "aac32", "comb"]
LABEL = {"resample": "대조: 44.1 kHz 무손실 왕복", "aac": "AAC 128 kbps",
         "opus": "Opus 160 kbps", "aac32": "대조: AAC 32 kbps 모노",
         "comb": "(다) 기준 조건 + AAC"}
for k in MIX:
    _m, p, s = k.split("-")
    LABEL[k] = f"{'일반' if p == 'gen' else '하드 네거티브'} 배경 {'+' if s != '0' else ''}{s} dB"

print("### 2.1 int8 0 비율 중앙값 (기준 조건을 정한 값)\n")
print("| 조건 | " + " | ".join(EV) + " |\n|---|" + "---:|" * 4)
for k in ["clean"] + MIX[:4] + ["external"]:
    z = R["zero_frac"][k]
    name = {"clean": "시험셋 원본", "external": "**외부 세트 (v1 양성)**"}.get(k, LABEL.get(k))
    print(f"| {name} | " + " | ".join(f"{z[c]:.3f}" for c in EV) + " |")
print("\n기준 SNR: " + ", ".join(f"{c} {'+' if v else ''}{v} dB" for c, v in R["ref_snr"].items()))

print("\n### 2.2 요약 재현 비 (3구성 × 4클래스 평균)\n")
print("| 조건 | 재현 비 |\n|---|---:|")
print(f"| **(가) 기준 조건** | **{R['summary_ratio']['ref_mix']:.3f}** |")
for k in ORDER:
    print(f"| {LABEL[k]} | {R['summary_ratio'][k]:.3f} |")

for i, (cfg, r) in enumerate(R["results"].items()):
    print(f"\n### 2.{3 + i} {cfg} — recall % (괄호: 재현 비), 시험셋 @300/h 문턱값 고정\n")
    print("| 조건 | " + " | ".join(EV) + " | 재현 비 평균 | 배경 오경보/h |\n|---|" + "---:|" * 6)
    print("| 시험셋 원본 | " + " | ".join(f"{100*v:.1f}" for v in r["clean"]) + " | — | 300 |")
    for k in ORDER:
        c, q = r["conds"][k], r["ratio"][k]
        fa = "—" if c["fa_per_hour"] is None else f"{c['fa_per_hour']:.0f}"
        print(f"| {LABEL[k]} | " + " | ".join(
            f"{100*v:.1f} ({x:.2f})" for v, x in zip(c["recall"], q))
            + f" | {sum(q)/4:.2f} | {fa} |")
    q = r["ratio"]["ref_mix"]
    print("| **(가) 기준 조건** | " + " | ".join(
        f"**{100*v:.1f}** ({x:.2f})" for v, x in zip(r["ref_mix_recall"], q))
        + f" | **{sum(q)/4:.2f}** | — |")
    print("| 외부 세트 (v1 정의) | " + " | ".join(
        f"{100*v:.1f}" for v in R["external_recall"][cfg]) + " | 1 | |")
