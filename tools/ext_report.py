#!/usr/bin/env python3
"""`eval_external.py` 의 JSON → 마크다운 표 (docs/results/external-eval.md 용).

사용 (WSL2):  python3 tools/ext_report.py [json]
"""

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "data", "external_eval", "external_eval.json")
EV = ["siren", "glass", "scream", "dog_bark"]


def pc(v):
    return "—" if v != v else f"{100 * v:.1f}"


def main():
    R = json.load(open(SRC, encoding="utf-8"))
    C = R["configs"]
    print(f"모집단: {R['population']}")
    print(f"외부 배경 1창 = {R['resolution_ext_bg_per_hour']}회/h\n")

    for d, title in (("v1", "v1 정의"), ("real", "실사용 정의 (real_primary=1)")):
        print(f"### {title} — 시험셋 @300/h 문턱값 고정\n")
        print("| 구성 | 시드 | 시험셋 F1 | 외부 F1 [95% CI] | 낙폭 | 외부 배경 오경보/h "
              "| 외부 배경 기준 @300/h F1 |")
        print("|---|---:|---:|---|---:|---:|---:|")
        for c in C:
            x = c["defs"][d]
            print(f"| {c['name']} | {c['n_seeds']} | {c['test_f1']:.4f} | "
                  f"**{x['f1']:.4f}** [{x['f1_ci'][0]:.3f}, {x['f1_ci'][1]:.3f}] | "
                  f"{x['drop_vs_test']:+.4f} | {x['fa_per_hour']:.0f} | "
                  f"{x['ext_bg_point']['f1']:.4f} |")
        print(f"\n클래스별 recall % [95% CI] — {title}\n")
        print("| 구성 | " + " | ".join(EV) + " |")
        print("|---|" + "---|" * 4)
        for c in C:
            x = c["defs"][d]
            lo, hi = x["recall_ci"]
            print(f"| {c['name']} | " + " | ".join(
                f"{pc(x['recall'][i])} [{pc(lo[i])}, {pc(hi[i])}]" for i in range(4)) + " |")
        print(f"\n시험셋 대비 recall 낙폭 (pp, 시험셋 − 외부) — {title}\n")
        print("| 구성 | " + " | ".join(EV) + " |")
        print("|---|" + "---:|" * 4)
        for c in C:
            x = c["defs"][d]
            print(f"| {c['name']} | " + " | ".join(
                f"{100 * (c['test_recall'][i] - x['recall'][i]):+.1f}" for i in range(4)) + " |")
        print()

    rk = R["rank"]
    print("### 순위\n")
    print("| 구성 | 시험셋 | 외부 v1 | 외부 실사용 |")
    print("|---|---:|---:|---:|")
    for i, n in enumerate(rk["names"]):
        print(f"| {n} | {rk['test'][i]} | {rk['v1'][i]} | {rk['real'][i]} |")
    print(f"\nSpearman: v1 {rk['spearman_v1']:.3f} / 실사용 {rk['spearman_real']:.3f}. "
          f"완전 일치: v1 {rk['identical_v1']} / 실사용 {rk['identical_real']}\n")

    print("### siren 하위 종류별 recall % (siren 그룹 121, 시험셋 문턱값)\n")
    keys = sorted(C[0]["sub"]["siren"], key=lambda k: -C[0]["sub"]["siren"][k]["n"])
    keys = [k for k in keys if C[0]["sub"]["siren"][k]["n"] >= 4]
    print("| 구성 | `Siren` 포함 (n=" + str(C[0]["sub"]["siren_generic"]["with_Siren"]["n"])
          + ") | 하위 종류만 (n=" + str(C[0]["sub"]["siren_generic"]["subtype_only"]["n"])
          + ") | " + " | ".join(f"{k} ({C[0]['sub']['siren'][k]['n']})" for k in keys) + " |")
    print("|---|---:|---:|" + "---:|" * len(keys))
    for c in C:
        g = c["sub"]["siren_generic"]
        print(f"| {c['name']} | {pc(g['with_Siren']['recall'])} | "
              f"{pc(g['subtype_only']['recall'])} | "
              + " | ".join(pc(c["sub"]["siren"][k]["recall"]) for k in keys) + " |")

    print("\n### scream 하위 라벨별 — scream 으로 발화한 비율 % (괄호: 아무 이벤트로든 발화)\n")
    keys = sorted(C[0]["sub"]["scream"], key=lambda k: -C[0]["sub"]["scream"][k]["n"])
    keys = [k for k in keys if C[0]["sub"]["scream"][k]["n"] >= 4]
    print("| 구성 | " + " | ".join(f"{k} ({C[0]['sub']['scream'][k]['n']})" for k in keys) + " |")
    print("|---|" + "---:|" * len(keys))
    for c in C:
        print(f"| {c['name']} | " + " | ".join(
            f"{pc(c['sub']['scream'][k]['recall'])} ({pc(c['sub']['scream'][k]['fire_any'])})"
            for k in keys) + " |")

    print("\n### def_gap 발화율 % — 그 클래스로 발화 (괄호: 아무 이벤트로든)\n")
    dk = list(C[0]["sub"]["def_gap"])
    print("| 구성 | " + " | ".join(f"{k} (n={C[0]['sub']['def_gap'][k]['n']})" for k in dk) + " |")
    print("|---|" + "---:|" * len(dk))
    for c in C:
        print(f"| {c['name']} | " + " | ".join(
            f"{pc(c['sub']['def_gap'][k]['fire_as_class'])} "
            f"({pc(c['sub']['def_gap'][k]['fire_any'])})" for k in dk) + " |")

    b0 = C[0]["sub"]["bg_fire"]
    print("\n### 배경 발화율 % (시험셋 문턱값) 과 포함·제외\n")
    print(f"| 구성 | 배경 전체 ({b0['n']}) | 거의 무음 ({b0['n_near_silent']}) | "
          f"webm 출신 ({b0['n_webm']}) | F1 v1: 무음 제외 | F1 v1: webm 제외 | "
          "F1 실사용: 무음 제외 | F1 실사용: webm 제외 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for c in C:
        b = c["sub"]["bg_fire"]
        v, r = c["defs"]["v1"], c["defs"]["real"]
        print(f"| {c['name']} | {pc(b['all'])} | {pc(b['near_silent'])} | {pc(b['webm'])} | "
              f"{v['excl_bg_near_silent']['f1']:.4f} ({v['excl_bg_near_silent']['f1'] - v['f1']:+.4f}) | "
              f"{v['excl_webm_origin']['f1']:.4f} ({v['excl_webm_origin']['f1'] - v['f1']:+.4f}) | "
              f"{r['excl_bg_near_silent']['f1']:.4f} ({r['excl_bg_near_silent']['f1'] - r['f1']:+.4f}) | "
              f"{r['excl_webm_origin']['f1']:.4f} ({r['excl_webm_origin']['f1'] - r['f1']:+.4f}) |")
    x = C[0]["defs"]
    print(f"\n제외 수: 무음 v1 {x['v1']['excl_bg_near_silent']['n_removed']} / "
          f"webm v1 {x['v1']['excl_webm_origin']['n_removed']} / "
          f"webm 실사용 {x['real']['excl_webm_origin']['n_removed']}")


if __name__ == "__main__":
    main()
