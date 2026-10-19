#!/usr/bin/env python3
"""②′ 곡선 변형 (b64 / f31 / b64f31) 의 사전 등록 판정 A~D (docs/results/fe-curve-design.md 5절).

입력  data/external_eval/scream_listened_test.json   변형·기존 구성의 시드별 testL-S3 macro-F1, 외부 F1 (--score 결과)
      data/external_eval/fe_eval_<변형>.json          변형의 외부 mAP (tools/fe_ext_eval.py --variant)
      docs/paper/numbers.json                         기존 점의 추정 에너지 (데이터시트) — 보드 실측 지연 기반
출력  data/external_eval/fe_curve_judge.json, 콘솔 표 (마크다운)

판정 (사전 등록, 결과를 본 뒤 바꾸지 않는다):
  A  변형 X 가 지배되지 않는다 — 에너지 ≤ X 인 기존 점(④·②′·가·다·간격 400·①′) 중 시험 macro-F1 점추정 ≥ X 가 없다
  B  A 인 X 에 대해, 에너지 ≤ X 이면서 F1 이 가장 높은 기존 점 대비 X 의 Welch 95% CI 하한 > 0
  C  X − ②′ 의 시험 macro-F1 CI 하한 > 0
  D  같은 에너지에서의 격차 — CPU 곡선(가-다-간격 400-①′ 선형 보간) F1 − X 의 F1 (보고용)

⚠️ 변형의 에너지는 보드 실측 전에는 fe-curve-design.md 4절의 **예측값**이다 (`--energy "이름=µJ"` 로 실측값을 넣으면 교체).
   예측값으로 낸 판정은 "잠정" 으로 표시한다 — 실측 전에는 논문 그림·표에 넣지 않는다 (5절 결과의 사용 4).

사용 (WSL2): ~/ai8x-training/venv/bin/python tools/fe_curve_judge.py [--energy "②′ b64=76" ...]
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
from eval_v2 import welch                             # noqa: E402  (b − a 의 평균, CI 하한, 상한)

EXT = os.path.join(REPO, "data", "external_eval")
NUMBERS = os.path.join(REPO, "docs", "paper", "numbers.json")
# 기존 점: scream_listened_test.json 의 구성 이름 → numbers.json 의 키
EXISTING = {"④": "④", "②′ fe": "②′", "가 u1000": "가 u1000", "다 u800": "다 u800", "간격 400": "간격 400", "①′": "①′"}
CPU_CURVE = ["가 u1000", "다 u800", "간격 400", "①′"]          # D 의 CPU log-mel 곡선
# 변형 → (fe_eval 파일, 사전 등록 4절 예측 에너지 µJ 데이터시트). 실측 전 잠정값.
VARIANTS = {"②′ b64": ("fe_eval_b64.json", 76.0), "②′ f31": ("fe_eval_f31.json", 72.0), "②′ b64f31": ("fe_eval_b64f31.json", 140.0)}
BASE = "②′ fe"


def load(fn):
    with open(os.path.join(EXT, fn), encoding="utf-8") as f:
        return json.load(f)


def mean(v):
    return float(np.mean(v))


def sd(v):
    return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def ci(a, b):                                          # b − a
    d, lo, hi = welch(a, b)
    return [float(d), float(lo), float(hi)]


def fmt(c):
    return "—" if c is None else f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", nargs="*", default=[], help='변형 에너지 실측값 "이름=µJ" (예: "②′ b64=80.1")')
    ap.add_argument("--out", default=os.path.join(EXT, "fe_curve_judge.json"))
    a = ap.parse_args()
    measured = {}
    for s in a.energy:
        k, v = s.rsplit("=", 1)
        measured[k.strip()] = float(v)

    slt = load("scream_listened_test.json")["score"]["configs"]
    with open(NUMBERS, encoding="utf-8") as f:
        num = json.load(f)
    pts = {}                                           # 이름 → energy, f1 시드 리스트, ext_f1, ext_map, n, src
    for name, ek in EXISTING.items():
        s = slt[name]["seeds"]
        pts[name] = {"energy": num[f"energy.{ek}"]["value"]["datasheet"], "src": "보드 실측 × 데이터시트",
                     "f1": [x["testL_S3"]["f1"] for x in s], "ext_f1": [x["ext"]["f1"] for x in s],
                     "ext_map": num[f"acc.{ek}.ext_map"]["value"]["mean"], "n": len(s)}
    for name, (fn, pred) in VARIANTS.items():
        if name not in slt:
            print(f"[건너뜀] {name}: scream_listened_test.json 에 없다 (--score 전?)")
            continue
        s = slt[name]["seeds"]
        fe = load(fn)["seeds"] if os.path.exists(os.path.join(EXT, fn)) else []
        for x, y in zip(s, fe):
            assert abs(x["thr"] - y["thr"]) < 1e-9 and abs(x["testL"]["f1"] - y["test_f1_L"]) < 1e-9, f"{name}: fe_eval 과 시드 정렬이 다르다"
        e = measured.get(name)
        if e is None and f"energy.{name}" in num:          # 보드 실측 뒤 make_paper_assets 가 적은 값 (2026-10-19 오후 지시로 실측)
            e = num[f"energy.{name}"]["value"]["datasheet"]
        pts[name] = {"energy": pred if e is None else e, "src": "예측 (fe-curve-design.md 4절) — 잠정" if e is None else "보드 실측 × 데이터시트",
                     "f1": [x["testL_S3"]["f1"] for x in s], "ext_f1": [x["ext"]["f1"] for x in s],
                     "ext_map": mean([y["ext_map_real31L"] for y in fe]) if fe else None, "n": len(s)}

    # D 용 CPU 곡선 보간 (에너지 오름차순, 범위 밖은 끝값)
    cx = np.array([pts[n]["energy"] for n in CPU_CURVE]); cy = np.array([mean(pts[n]["f1"]) for n in CPU_CURVE])
    o = np.argsort(cx); cx, cy = cx[o], cy[o]

    out = {"note": "fe-curve-design.md 5절 판정. 변형 에너지가 예측값이면 잠정 (논문 그림·표에 넣지 않는다).", "points": {}, "judge": {}}
    for n, p in pts.items():
        out["points"][n] = {"energy_uJ": p["energy"], "energy_src": p["src"], "n": p["n"], "f1_S3": p["f1"], "f1_S3_mean": mean(p["f1"]),
                            "f1_S3_sd": sd(p["f1"]), "ext_f1": p["ext_f1"], "ext_f1_mean": mean(p["ext_f1"]), "ext_map_mean": p["ext_map"]}

    print("| 변형 | n | 에너지 µJ | testL-S3 F1 | 외부 F1 | 외부 mAP | A 지배되지 않음 | 지배하는 점 | B (에너지 이하 최고 기존 점 대비 Welch) | C (②′ 대비 Welch) | D CPU−X | 외부 F1 − ②′ |")
    print("|---|---:|---:|---|---|---|---|---|---|---|---:|---|")
    for name in VARIANTS:
        if name not in pts:
            continue
        X = pts[name]; fx = mean(X["f1"]); n = X["n"]
        below = {k: v for k, v in pts.items() if k in EXISTING and v["energy"] <= X["energy"]}
        dom = sorted([k for k, v in below.items() if mean(v["f1"]) >= fx], key=lambda k: -mean(pts[k]["f1"]))
        A = len(dom) == 0
        bb = max(below, key=lambda k: mean(pts[k]["f1"])) if below else None
        Bci = ci(pts[bb]["f1"], X["f1"]) if (bb and n >= 2) else None
        B = bool(A and Bci and Bci[1] > 0)
        Cci = ci(pts[BASE]["f1"], X["f1"]) if n >= 2 else None
        C = bool(Cci and Cci[1] > 0)
        D = float(np.interp(X["energy"], cx, cy) - fx)
        eci = ci(pts[BASE]["ext_f1"], X["ext_f1"]) if n >= 2 else None
        prov = "예측" in X["src"]
        out["judge"][name] = {"energy_uJ": X["energy"], "provisional_energy": prov, "n": n, "f1_S3": fx, "A": A, "dominated_by": dom,
                              "best_below": bb, "B": B, "B_welch": Bci, "C": C, "C_welch_vs_fe": Cci, "ext_f1_welch_vs_fe": eci, "D_cpu_minus_X": D}
        emap = "—" if X["ext_map"] is None else f"{X['ext_map']:.3f}"
        print(f"| {name} | {n} | {X['energy']:.0f}{' (예측)' if prov else ''} | {fx:.3f} ± {sd(X['f1']):.3f} | {mean(X['ext_f1']):.3f} ± {sd(X['ext_f1']):.3f} | {emap} | "
              f"{'O' if A else 'X'} | {', '.join(dom) if dom else '—'} | {bb}: {fmt(Bci)} → {'O' if B else 'X'} | {fmt(Cci)} → {'O' if C else 'X'} | {D:+.3f} | {fmt(eci)} |")
    print("\n기존 점 (testL-S3 F1 평균 / 외부 F1 평균 / 에너지 µJ 데이터시트):")
    for k in EXISTING:
        p = pts[k]
        print(f"  {k:8s} {mean(p['f1']):.3f}  {mean(p['ext_f1']):.3f}  {p['energy']:.1f}")
    with open(a.out, "w", encoding="utf-8") as fo:
        json.dump(out, fo, ensure_ascii=False, indent=1)
    print("저장:", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
