#!/usr/bin/env python3
"""chain34 판정 — H-1 (②′ v2 − ②′), H-2 (fe8 − ②′ v2), H-3 (가 u1000 과의 거리). docs/results/npu-readback-design.md 7.3 의 식 그대로.

입력: data/external_eval/scream_listened_test.json 의 score.configs[이름].seeds[i] (testL_S3.f1 / .recall, ext.f1 / .recall)
      — 논문 표 1 과 같은 채점 경로(시험 배경 @300/h 문턱, testL-S3, real31L). 먼저
        OPENBLAS_NUM_THREADS=1 python tools/fe_ext_eval.py --variant fe_v2
        OPENBLAS_NUM_THREADS=1 python tools/scream_listened_test.py --score --configs "②′ v2"
      를 돌려 두어야 한다 (fe8 도 같은 식).
출력: 표(stdout) + data/external_eval/h1_judge.json. 판정 문구는 7.3 사전 등록 그대로 — 결과를 보고 바꾸지 않는다.

사용 (WSL2): ~/ai8x-training/venv/bin/python tools/h1_judge.py [--fe8]
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
from eval_v2 import welch                      # noqa: E402  (b − a 의 평균, CI 하한, 상한)

EXT = os.path.join(REPO, "data", "external_eval")
CLS = ["siren", "glass", "scream", "dog_bark", "background"]
BASE, V2, FE8, GA = "②′ fe", "②′ v2", "②′ fe8", "가 u1000"


def fmt(c):
    d, lo, hi = c
    return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def ci(a, b):
    d, lo, hi = welch(a, b)
    return float(d), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fe8", action="store_true", help="fe8 3시드가 채점돼 있으면 H-2·H-3(fe8) 도")
    a = ap.parse_args()
    slt = json.load(open(os.path.join(EXT, "scream_listened_test.json"), encoding="utf-8"))["score"]["configs"]
    need = [BASE, V2] + ([FE8] if a.fe8 else [])
    for n in need:
        if n not in slt:
            sys.exit(f"[에러] {n} 이 scream_listened_test.json 에 없다 — --score 를 먼저")
    S = {n: slt[n]["seeds"] for n in need + [GA]}
    f1 = {n: [x["testL_S3"]["f1"] for x in S[n]] for n in S}
    ext = {n: [x["ext"]["f1"] for x in S[n]] for n in S}
    rec = {n: np.array([x["testL_S3"]["recall"] for x in S[n]]) for n in S}        # (시드, 클래스)
    erec = {n: np.array([x["ext"]["recall"] for x in S[n]]) for n in S}
    out = {"note": "npu-readback-design.md 7.3 사전 등록 판정. testL-S3 macro-F1 @ 시험 배경 300/h, 외부 real31L. Welch 95 % CI.", "points": {}, "judge": {}}
    print("| 구성 | n | 시험 F1 (시드) | 평균 ± sd | 외부 F1 평균 | " + " | ".join(f"재현율 {c}" for c in CLS) + " |")
    print("|---|---:|---|---|---|" + "---|" * len(CLS))
    for n in S:
        v = f1[n]
        out["points"][n] = {"n": len(v), "f1_S3": v, "f1_S3_mean": float(np.mean(v)), "f1_S3_sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                            "ext_f1": ext[n], "ext_f1_mean": float(np.mean(ext[n])), "test_recall_mean": rec[n].mean(0).tolist(),
                            "ext_recall_mean": erec[n].mean(0).tolist()}
        print(f"| {n} | {len(v)} | {' / '.join(f'{x:.4f}' for x in v)} | {np.mean(v):.4f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.4f} | {np.mean(ext[n]):.4f} | "
              + " | ".join(f"{100 * r:.1f} %" for r in rec[n].mean(0)) + " |")

    def pair(tag, x, y, rule):
        c = ci(f1[x], f1[y]); ce = ci(ext[x], ext[y])
        crec = [ci(rec[x][:, k], rec[y][:, k]) for k in range(len(CLS))]
        cerec = [ci(erec[x][:, k], erec[y][:, k]) for k in range(len(CLS))]
        out["judge"][tag] = {"pair": f"{y} − {x}", "test_f1": c, "ext_f1": ce, "test_recall": crec, "ext_recall": cerec, "verdict": rule(c)}
        print(f"\n**{tag}  {y} − {x}**")
        print(f"  시험 F1 {fmt(c)}   외부 F1 {fmt(ce)}")
        print("  시험 재현율 차 (pp): " + ", ".join(f"{CLS[k]} {100 * crec[k][0]:+.1f} [{100 * crec[k][1]:+.1f}, {100 * crec[k][2]:+.1f}]" for k in range(len(CLS))))
        print("  외부 재현율 차 (pp): " + ", ".join(f"{CLS[k]} {100 * cerec[k][0]:+.1f} [{100 * cerec[k][1]:+.1f}, {100 * cerec[k][2]:+.1f}]" for k in range(len(CLS))))
        print(f"  → 판정: {rule(c)}")

    def h1_rule(c):
        d, lo, hi = c
        if lo > 0:
            return f"결함 (다) 가 ②′ 를 {d:+.3f} 깎았다 (CI 하한 > 0) — 7.6 의 최종본 변경 후보가 살아난다"
        if hi < 0:
            return "②′ v2 가 더 낮다 (CI 상한 < 0) — 예상 밖, 원인 조사 전까지 결론 보류"
        return "CI 가 0 을 포함 — 결함은 동작점만 나쁘게 했고 F1 에는 보이지 않는다 (한계 절 후보)"

    def h2_rule(c):
        d, lo, hi = c
        if lo > -0.01:
            return "8비트 출력 재학습으로 손실 없이 읽기 2,048 → 128 워드 (추정 38.9 → 35.9 μJ)"
        if lo < -0.03:
            return "8비트 단계는 재학습으로도 메우지 못한다"
        return "미확정 (CI 하한 −0.03 ~ −0.01) — 그대로 보고, 시드 추가 제안 없음"

    pair("H-1", BASE, V2, h1_rule)
    if a.fe8:
        pair("H-2", V2, FE8, h2_rule)
    # H-3: 가 u1000 과의 거리 (앞면) — ②′ v2 (및 fe8) − 가
    for n in [V2] + ([FE8] if a.fe8 else []):
        c = ci(f1[GA], f1[n])
        out["judge"][f"H-3 {n}"] = {"pair": f"{n} − {GA}", "test_f1": c,
                                    "verdict": "가 와 겹침 (CI 상한 ≥ 0) — 4절 G-2 재검토" if c[2] >= 0 else "가 아래 (CI 상한 < 0) — 앞면 순서 불변"}
        print(f"\n**H-3  {n} − {GA}**  시험 F1 {fmt(c)} → {out['judge'][f'H-3 {n}']['verdict']}")
    fn = os.path.join(EXT, "h1_judge.json")
    json.dump(out, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n저장: {fn}")


if __name__ == "__main__":
    main()
