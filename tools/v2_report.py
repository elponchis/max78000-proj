#!/usr/bin/env python3
"""`eval_v2.py` 의 JSON → 마크다운 표 (dataset-v2-design.md 9절용)."""

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(REPO, "data", "external_eval", "v2_eval.json"), encoding="utf-8"))
EV = ["siren", "glass", "scream", "dog_bark"]


def ci(d, pp=False):
    k = 100 if pp else 1
    f = ".1f" if pp else ".4f"
    return f"{k*d[0]:+{f}} [{k*d[1]:+{f}}, {k*d[2]:+{f}}]"


print("### 9.1 채택 판정 (사전 등록 8-c)\n")
print("| 구성 | 시드 | 시험셋 F1 v1 → v2 | v2 − v1 [95% CI] | 조건 2 (하한 ≥ −0.02) | 외부 F1 v1 → v2 | v2 − v1 [95% CI] | 조건 1 (하한 > 0) | 둘 다 |")
print("|---|---:|---|---|---|---|---|---|---|")
for p in R["pairs"]:
    print(f"| {p['name']} | {p['n_seeds']} | {p['v1']['test_f1']:.4f} → {p['v2']['test_f1']:.4f} | "
          f"{ci(p['delta']['test_f1'])} | {'충족' if p['cond_test'] else '**미충족**'} | "
          f"{p['v1']['ext_f1']:.4f} → {p['v2']['ext_f1']:.4f} | {ci(p['delta']['ext_f1'])} | "
          f"{'충족' if p['cond_ext'] else '**미충족**'} | {'**○**' if p['cond_ext'] and p['cond_test'] else '×'} |")
print(f"\n두 조건 충족: **{R['n_pass']} / {R['n_total']}** (채택 기준 5개 중 3개 이상).")

print("\n### 9.2 시드별 값\n")
print("| 구성 | 시험셋 v1 | 시험셋 v2 | 외부 v1 | 외부 v2 |\n|---|---|---|---|---|")
for p in R["pairs"]:
    f = lambda xs: " / ".join(f"{x:.4f}" for x in xs)
    print(f"| {p['name']} | {f(p['v1']['test_f1_seeds'])} | {f(p['v2']['test_f1_seeds'])} | "
          f"{f(p['v1']['ext_f1_seeds'])} | {f(p['v2']['ext_f1_seeds'])} |")

for key, title in (("test_recall", "기존 시험셋 recall % (v1 → v2, [v2 − v1 95% CI, pp])"),
                   ("test_f1_class", "기존 시험셋 클래스별 F1 %"),
                   ("ext_recall", "외부 세트 recall % (v1 정의, 시험셋 문턱값 고정)"),
                   ("mix_mix-gen-10", "혼합 시험셋 (일반 배경 +10 dB) recall % — 배경 불변, 같은 오경보"),
                   ("mix_mix-gen-5", "혼합 시험셋 (일반 배경 +5 dB) recall %")):
    print(f"\n### {title}\n")
    print("| 구성 | " + " | ".join(EV) + " |\n|---|" + "---|" * 4)
    for p in R["pairs"]:
        cells = []
        for c in range(4):
            d = p["delta"][key][c]
            warn = " ⚠️" if key == "test_recall" and 100 * d[1] < -5 else ""
            cells.append(f"{100*p['v1'][key][c]:.1f} → {100*p['v2'][key][c]:.1f} [{100*d[1]:+.1f}, {100*d[2]:+.1f}]{warn}")
        print(f"| {p['name']} | " + " | ".join(cells) + " |")

print("\n### 외부 배경 오경보와 지름길 점검 3-(다)\n")
print("| 구성 | 외부 배경 오경보/h (시험셋 문턱값) v1 → v2 | 시험셋 배경 발화 % | 배경+배경 혼합 창 발화 % v1 → v2 |\n|---|---|---|---|")
for p in R["pairs"]:
    print(f"| {p['name']} | {p['v1']['ext_fa']:.0f} → {p['v2']['ext_fa']:.0f} | "
          f"{100*p['v1']['bg_fire']:.2f} (고정) | {100*p['v1']['bgbg_fire']:.2f} → {100*p['v2']['bgbg_fire']:.2f} |")
rk = R["rank"]
print("\n### 순위 3-(라)\n")
print("| 구성 | v1 시험셋 | v2 시험셋 | v2 외부 |\n|---|---:|---:|---:|")
for i, n in enumerate(rk["names"]):
    print(f"| {n} | {rk['v1_test'][i]} | {rk['v2_test'][i]} | {rk['v2_ext'][i]} |")
