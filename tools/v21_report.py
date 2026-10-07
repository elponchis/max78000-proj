#!/usr/bin/env python3
"""`eval_v21.py` 의 JSON → 세 판(v1 / v2 / v2.1) 비교표 (dataset-v2.1-design.md 8절용)."""

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
from eval_v2 import welch                                      # noqa: E402

R = json.load(open(os.path.join(REPO, "data", "external_eval", "v21_eval.json"), encoding="utf-8"))
P = {p["name"]: p for p in R["pairs"]}
EV = ["siren", "glass", "scream", "dog_bark"]
VERS = ("v1", "v2", "v21")
LAB = {"v1": "v1", "v2": "v2", "v21": "v2.1"}


def ci(d, k=1, f=".4f"):
    return f"{k*d[0]:+{f}} [{k*d[1]:+{f}}, {k*d[2]:+{f}}]"


def seeds(name, ver, key):
    return P[name]["seeds"][ver][key]


print("### 8.1 채택 판정 (2절 규칙 그대로) 과 본 표 선택 (7절 일탈 규칙)\n")
print("| 구성 | 조건 1 외부 F1 (v2.1 − v1) | 조건 2 시험셋 F1 (v2.1 − v1) | 조건 3 외부 오경보 맞춤 (v2.1 − v1) | 조건 4 배경+배경 발화 v2.1 | 넷 다 | 조건 2·4 |")
print("|---|---|---|---|---|---|---|")
n24 = 0
for n, p in P.items():
    c = p["cond"]
    ok24 = c["2_test"] and c["4_bgbg"]
    n24 += ok24
    print(f"| {n} | {ci(p['delta_v1']['ext_f1'])} {'○' if c['1_ext'] else '×'} | "
          f"{ci(p['delta_v1']['test_f1'])} {'○' if c['2_test'] else '×'} | "
          f"{ci(p['delta_v1']['ext_matched'])} {'○' if c['3_matched'] else '×'} | "
          f"{100*np.mean(p['seeds']['v21']['bgbg_fire'][:p['n_old']]):.2f}% {'○' if c['4_bgbg'] else '×'} | "
          f"{'**○**' if p['pass_all'] else '×'} | {'**○**' if ok24 else '×'} |")
print(f"\n네 조건 충족 {R['n_pass']}/{R['n_total']} → 채택 판정 **{'채택' if R['adopt'] else '기각'}**. "
      f"조건 2·4 충족 {n24}/{R['n_total']} → 본 표 **{'v2.1' if n24 >= 3 else 'v1'}** (7절).")

print("\n### 8.2 세 판 나란히 — 시드 평균 (v2.1 은 전 시드)\n")
print("| 구성 | 시험셋 F1 v1 / v2 / v2.1 | 외부 F1 (시험셋 문턱값) | 외부 F1 (외부 오경보 맞춤) | 외부 오경보/h | 배경+배경 발화 % |")
print("|---|---|---|---|---|---|")
for n, p in P.items():
    m = p["mean"]
    print(f"| {n} | " + " / ".join(f"{m[v]['test_f1']:.4f}" for v in VERS) + " | "
          + " / ".join(f"{m[v]['ext_f1']:.4f}" for v in VERS) + " | "
          + " / ".join(f"{m[v]['ext_matched']:.4f}" for v in VERS) + " | "
          + " / ".join(f"{m[v]['ext_fa']:.0f}" for v in VERS) + " | "
          + " / ".join(f"{100*m[v]['bgbg_fire']:.2f}" for v in VERS) + " |")
print("\n시드별 시험셋 F1 (v2.1):")
for n, p in P.items():
    print(f"- {n}: " + " / ".join(f"{x:.4f}" for x in p["seeds"]["v21"]["test_f1"]))

print("\n### 8.3 구성 순위 — 세 판 (시험셋 F1 / 외부 F1 시험셋 문턱값)\n")
names = list(P)
def rank(vals):
    o = np.argsort(-np.asarray(vals), kind="stable"); r = np.empty(len(vals), int); r[o] = np.arange(1, len(vals) + 1); return r
print("| 구성 | " + " | ".join(f"{LAB[v]} 시험셋" for v in VERS) + " | " + " | ".join(f"{LAB[v]} 외부" for v in VERS) + " |")
print("|---|" + "---:|" * 6)
rk = {(v, k): rank([P[n]["mean"][v][k] for n in names]) for v in VERS for k in ("test_f1", "ext_f1")}
for i, n in enumerate(names):
    print(f"| {n} | " + " | ".join(str(rk[(v, 'test_f1')][i]) for v in VERS) + " | "
          + " | ".join(str(rk[(v, 'ext_f1')][i]) for v in VERS) + " |")

print("\n### 8.4 재판정 항목 — 세 판 (시험셋 F1 @300/h, Welch 95% CI)\n")
print("v1·v2 는 시드 1~3(④ 는 1~5), v2.1 은 전 시드(④·간격 400·①′ 5, 가·다 3).\n")
print("| 비교 | v1 | v2 | v2.1 | v2.1 판정 |")
print("|---|---|---|---|---|")
for a, b, kind in (("간격 400", "①′", "eq"), ("가 u1000", "①′", ""), ("다 u800", "①′", ""),
                   ("가 u1000", "④", ""), ("다 u800", "④", "")):
    cells = []
    for v in VERS:
        d = welch(seeds(b, v, "test_f1"), seeds(a, v, "test_f1"))
        cells.append(ci(d))
    d = welch(seeds(b, "v21", "test_f1"), seeds(a, "v21", "test_f1"))
    if kind == "eq":
        verdict = "**동등** (±0.02 안)" if d[1] >= -0.02 and d[2] <= 0.02 else (
            "못 가른다" if d[1] < 0 < d[2] else ("유의하게 높다" if d[1] > 0 else "유의하게 낮다"))
    else:
        verdict = "못 가른다" if d[1] < 0 < d[2] else ("유의하게 높다" if d[1] > 0 else "유의하게 낮다")
    print(f"| {a} − {b} | " + " | ".join(cells) + f" | {verdict} |")

for key, title in (("test_recall", "시험셋 recall % v1 / v2 / v2.1"),
                   ("ext_recall", "외부 recall % (시험셋 문턱값) v1 / v2 / v2.1"),
                   ("mix_mix-gen-10", "혼합 시험셋 +10 dB recall % v1 / v2 / v2.1")):
    print(f"\n### {title}\n")
    print("| 구성 | " + " | ".join(EV) + " |\n|---|" + "---|" * 4)
    for n, p in P.items():
        c = p["class"]
        print(f"| {n} | " + " | ".join(
            " / ".join(f"{100*c[v][key][k]:.1f}" for v in VERS) for k in range(4)) + " |")
