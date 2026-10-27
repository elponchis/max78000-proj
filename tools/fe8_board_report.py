#!/usr/bin/env python3
"""fe8 보드 실측 보고 (후속 연구, docs/results/npu-readback-design.md 7절) — 보고 전용. 원고·numbers.json 은 건드리지 않는다.

입력 (WSL, POR 뒤 1000회):
  data/logs-local/board_v4d_fe8.log            fe8 (safesound_fe8_v4d, 지연 정지 펌웨어, 32비트 로그)
  data/logs-local/board_v4d_fe_log32_1027.log  같은 날 ②′ (safesound_fe_v4d 10-20 빌드, -DFE_LOG_AB) — 새 로그 패스 B 를 쓴다
출력: docs/results/fe8-board.md, docs/results/fe8-board.json

주 값 환산 규칙(2026-10-21, 사용자 결정)을 그대로 쓴다: fe8 환산 = ②′ 주 값(numbers.json latency.②′) + (같은 세션 fe8 − 같은 세션 ②′) 구간별.
에너지는 전부 **추정(실측 지연 × 공개 전력)**, 표 1 과 같은 식 (②′ 계열: 전처리는 NPU 켜진 채 → P_load).
실행 (WSL2): python3 tools/fe8_board_report.py [--fe8 LOG] [--ref LOG]
"""
import argparse
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(REPO, "data", "logs-local")
SEGS = ["load1", "cfg1", "infer1", "cpu", "load2", "cfg2", "infer2", "unload", "total"]
NAMES = {"load1": "청크 적재 (load1)", "cfg1": "p1 설정·전환 (cfg1)", "infer1": "p1 추론 (infer1, 대기 49 µs 포함)", "cpu": "CPU 몫 (cpu)",
         "load2": "p2 입력 적재 (load2)", "cfg2": "p2 설정·전환 (cfg2)", "infer2": "p2 추론 (infer2, 대기 포함)", "unload": "출력 (unload)",
         "total": "**판단 1회 (total)**"}
WAIT_MS = 0.04905          # numbers.json firmware.stop_wait us_per_pass (지연 정지 펌웨어, 패스당)
LOG_SAVE_CYC = 106         # 32비트 로그가 64비트 대비 절감한 사이클 (보드 실측, log32-board.md)
BLOCK_RE = r"(\d+) iterations, output mismatches: (\d+)\s+\(p1 (\d+), feature (\d+), p2 (\d+)\)(.*?)=== (?:PASS )?END ==="


def parse_block(txt):
    out = {}
    for m in re.finditer(r"^(\w+)\s+us\s+min\s+([\d.]+)\s+med\s+([\d.]+)\s+max\s+([\d.]+)", txt, re.M):
        out[m.group(1)] = {"min": float(m.group(2)), "med": float(m.group(3)), "max": float(m.group(4))}
    return out


def parse_log(path, which=None):
    """which=None: 단일 패스(첫 '... iterations, output mismatches' 블록). which='B': -DFE_LOG_AB 의 패스 B."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    kat = {}
    for key, k in (("KAT p1 sampleoutput", "p1"), ("KAT cpu feature", "feat"), ("KAT sampleoutput:", "p2")):
        line = txt.split(key)[1].split("\n")[0] if key in txt else ""
        kat[k] = ("PASS" in line, line.strip())
    build = re.search(r"^build (.*?)  SystemCoreClock (\d+)", txt, re.M)
    if which:
        m = re.search(r"=== PASS %s \(.*?\): " % which + BLOCK_RE, txt, re.S)
    else:
        m = re.search(r"^" + BLOCK_RE, txt, re.S | re.M)
    if not m:
        return None
    seg = parse_block(m.group(6))
    net = txt.split("two networks co-resident:")[1].split("\n")[0].strip() if "two networks co-resident:" in txt else ""
    return {"n": int(m.group(1)), "mismatch": int(m.group(2)), "p1": int(m.group(3)), "feat": int(m.group(4)), "p2": int(m.group(5)),
            "seg": seg, "kat": kat, "build": build.group(1) if build else "?", "clock": int(build.group(2)) if build else None, "stage_line": net}


def parts(d):
    """d: 구간 → ms. 표 1 분해 (전처리 / 적재(전환 포함) / 추론 / 출력)."""
    return {"pre": d["cpu"], "load": d["load1"] + d["cfg1"] + d["load2"] + d["cfg2"], "infer": d["infer1"] + d["infer2"], "out": d["unload"],
            "total": sum(d[s] for s in SEGS if s != "total")}


def energy(p, PS):
    return {s: p["pre"] * ps["P_load_mW"] + (p["load"] + p["out"]) * ps["P_load_mW"] + p["infer"] * ps["P_npu_mW"] for s, ps in PS.items()}


def cost_fe8(d, c_log, wait_ms):
    """비용 모델 (tools/npu_readback_sim.py cost(), A-sum, n=32 fph=4 P=57) 을 ②′ 구간 d 에 적용 → fe8 예상 구간."""
    n, fph, P = 32, 4, 57
    n_val = n * fph
    mac_p1 = 64 * 2 * n * 8 * P
    mac_b = 2 * n * n * 8 * (P - 7)
    r = 1 + mac_b / mac_p1
    m = dict(d)
    m["cpu"] = (1400 + n_val * 32 + n_val * c_log) / 1e5
    m["infer1"] = (d["infer1"] - wait_ms) * r + wait_ms
    m["cfg1"] = d["cfg1"] + 0.011
    m["total"] = sum(m[s] for s in SEGS if s != "total")
    return m, r


def write(out, L, J):
    open(out + ".md", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    json.dump(J, open(out + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fe8", default=os.path.join(LOGS, "board_v4d_fe8.log"))
    ap.add_argument("--ref", default=os.path.join(LOGS, "board_v4d_fe_log32_1027.log"))
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "results", "fe8-board"))
    a = ap.parse_args()
    C = json.load(open(os.path.join(REPO, "docs", "paper", "paper_config.json"), encoding="utf-8"))
    PS = {k: v for k, v in C["power_sources"].items() if v.get("use_in_curve")}
    N = json.load(open(os.path.join(REPO, "docs", "paper", "numbers.json"), encoding="utf-8"))
    main_fe = N["latency.②′"]["value"]["detail"]
    old_main = {"load1": 0.28151, "cfg1": 0.02298, "infer1": 0.08032, "cpu": N["latency.②′.log32"]["value"]["cpu_main_before_ms"],
                "load2": 0.13386, "cfg2": 0.08603, "infer2": 0.53153, "unload": 0.00643}
    old_main["total"] = sum(old_main[s] for s in SEGS if s != "total")

    F = parse_log(a.fe8) if os.path.isfile(a.fe8) else None
    R = parse_log(a.ref, which="B") if os.path.isfile(a.ref) else None
    L = ["# fe8 보드 실측 — ②′ 와 같은 날 나란히 (`tools/fe8_board_report.py` 생성, 보고 전용)", "",
         "지연은 POR 뒤 1,000회 **최악값**(ms). 두 구성 모두 지연 정지 펌웨어(완료 뒤 49 µs 안정화 대기 → 정지, NPU 패스당)·32비트 정수 로그. "
         "에너지는 전부 **추정(실측 지연 × 공개 전력)**, 논문 표 1 과 같은 식 (②′ 계열은 전처리 중 NPU 가 켜져 있어 P_load). **원고·numbers.json 미반영.**", ""]
    J = {}
    if not F:
        L += ["## fe8 로그 없음 또는 형식 불일치: " + os.path.relpath(a.fe8, REPO)]
        write(a.out, L, J)
        return
    kat_ok = all(v[0] for v in F["kat"].values())
    L += ["## fe8 (safesound_fe8_v4d, 빌드 %s, 코어 %s Hz)" % (F["build"], F["clock"]), "", "- 네트워크: " + F["stage_line"],
          "- KAT p1 출력(청크 3) / CPU 특징 / p2 출력: " + " / ".join(("**일치**" if F["kat"][k][0] else "**불일치**") for k in ("p1", "feat", "p2")) +
          " — " + F["kat"]["feat"][1],
          "- %d회 불일치: %d (p1 %d, 특징 %d, p2 %d)" % (F["n"], F["mismatch"], F["p1"], F["feat"], F["p2"]), ""]
    J["fe8"] = {"build": F["build"], "kat": {k: v[0] for k, v in F["kat"].items()}, "n": F["n"], "mismatch": [F["p1"], F["feat"], F["p2"]],
                "seg_max_ms": {s: F["seg"][s]["max"] / 1e3 for s in SEGS if s in F["seg"]}}
    if not kat_ok:
        L += ["**KAT 불일치 → 지연 수치는 쓰지 않는다 (사용자 지시 5).**"]
        write(a.out, L, J)
        return
    fe8 = {s: F["seg"][s]["max"] / 1e3 for s in SEGS}
    if R:
        L += ["## ②′ 같은 날 (safesound_fe_v4d, 빌드 %s, 패스 B = 32비트 로그)" % R["build"], "",
              "- KAT: " + " / ".join(("일치" if R["kat"][k][0] else "**불일치**") for k in ("p1", "feat", "p2")) +
              "; %d회 불일치 %d (p1 %d, 특징 %d, p2 %d)" % (R["n"], R["mismatch"], R["p1"], R["feat"], R["p2"]), ""]
        ref = {s: R["seg"][s]["max"] / 1e3 for s in SEGS}
        J["ref_fe"] = {"build": R["build"], "kat": {k: v[0] for k, v in R["kat"].items()}, "n": R["n"], "mismatch": [R["p1"], R["feat"], R["p2"]], "seg_max_ms": ref}
    else:
        ref = {s: N["latency.②′.new_firmware"]["value"]["detail"][s] for s in SEGS}
        L += ["## ②′ 같은 날 로그 없음 → 2026-10-21 지연 정지 펌웨어 직접 측정(latency.②′.new_firmware)을 대신 쓴다 (같은 세션 비교가 아니다)", ""]
    prev_fe = N["latency.②′.new_firmware"]["value"]["detail"]
    exp_sess, r = cost_fe8(ref, 190 - LOG_SAVE_CYC, WAIT_MS)     # 이 세션 기준 예상 (STATUS ②)
    exp_reg, _ = cost_fe8(old_main, 190, 0.0)                    # 사전 등록값 (옛 주 값 1.98 기준, 로그 190) → 1.49 / 35.9 (STATUS ①)
    conv = {s: main_fe[s] + fe8[s] - ref[s] for s in SEGS}        # 주 값 환산
    conv["total"] = sum(conv[s] for s in SEGS if s != "total")
    exp_conv = {s: main_fe[s] + exp_sess[s] - ref[s] for s in SEGS}
    exp_conv["total"] = sum(exp_conv[s] for s in SEGS if s != "total")
    L += ["## 구간별 지연 (최악, ms) — 같은 날 나란히", "",
          "| 구간 | ②′ (같은 날) | **fe8** | fe8 − ②′ | fe8 예상 (비용 모델, 이 세션 기준) | 실측 − 예상 | ②′ 10-21 직접 측정 (참고) |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for s in SEGS:
        dv = fe8[s] - ref[s]
        de = fe8[s] - exp_sess[s]
        pct = " (%+.0f %%)" % (100 * de / exp_sess[s]) if exp_sess[s] > 0.02 else ""
        L.append("| %s | %.3f | **%.3f** | %+.3f | %.3f | %+.3f%s | %.3f |" % (NAMES[s], ref[s], fe8[s], dv, exp_sess[s], de, pct, prev_fe[s]))
    pf, pr, pe = parts(fe8), parts(ref), parts(exp_sess)
    net1 = (fe8["infer1"] - WAIT_MS) / (ref["infer1"] - WAIT_MS)
    L += ["", "- CPU 몫: ②′ %.3f → fe8 **%.3f ms** (%+.3f; 예상 %.3f = 1,400 + 128×32 읽기 + 128×%d 로그 사이클). 판단당 fe8 CPU 사이클 %s."
          % (pr["pre"], pf["pre"], pf["pre"] - pr["pre"], pe["pre"], 190 - LOG_SAVE_CYC, format(int(round(pf["pre"] * 1e5)), ",")),
          "- NPU 연산: p1 추론 ②′ %.3f → fe8 %.3f ms (%+.3f; 예상 ×%.3f MAC 비 → %.3f). 대기 49 µs 를 뺀 순 연산 %.3f → %.3f ms (×%.2f; 합성기 사이클 비 5,505 / 3,784 = 1.455). "
          "p1 설정 %.3f → %.3f (층 하나 추가, 예상 +0.011)."
          % (ref["infer1"], fe8["infer1"], fe8["infer1"] - ref["infer1"], r, exp_sess["infer1"], ref["infer1"] - WAIT_MS, fe8["infer1"] - WAIT_MS, net1,
             ref["cfg1"], fe8["cfg1"]),
          "- 판단 1회: ②′ %.3f → fe8 **%.3f ms** (%+.3f, %+.1f %%)." % (pr["total"], pf["total"], pf["total"] - pr["total"], 100 * (pf["total"] - pr["total"]) / pr["total"]), ""]
    E_fe8, E_ref, E_exp = energy(pf, PS), energy(pr, PS), energy(pe, PS)
    E_conv, E_exp_conv = energy(parts(conv), PS), energy(parts(exp_conv), PS)
    E_main = N["energy.②′"]["value"]
    E_reg = energy(parts(exp_reg), PS)
    lab = [PS[s]["label"] for s in PS]

    def row(name, E, bold=False):
        f = "**%.1f**" if bold else "%.1f"
        return "| %s | " % name + " | ".join(f % E[s] for s in PS) + " |"
    L += ["## 에너지 (추정, µJ) — 세 전력 출처", "", "| 값 | " + " | ".join(lab) + " |", "|---|" + "---:|" * len(PS),
          row("②′ 같은 날 직접 측정", E_ref), row("**fe8 직접 측정**", E_fe8, True), row("fe8 예상 (비용 모델, 이 세션 기준)", E_exp),
          "| fe8 − ②′ (직접) | " + " | ".join("%+.1f" % (E_fe8[s] - E_ref[s]) for s in PS) + " |",
          row("②′ 주 값 (numbers.json, 환산 1.846 ms)", E_main),
          row("**fe8 주 값 환산** (②′ 주 값 + 같은 날 Δ, %.3f ms)" % conv["total"], E_conv, True),
          row("fe8 사전 등록 예상 (7.3.2, 옛 주 값 기준 %.2f ms)" % exp_reg["total"], E_reg),
          row("fe8 예상 환산 (STATUS ③, %.3f ms)" % exp_conv["total"], E_exp_conv), ""]
    checks = [("판단 1회 직접 측정 (ms)", pf["total"], pe["total"]), ("판단 1회 주 값 환산 (ms)", conv["total"], exp_conv["total"]),
              ("판단 1회 환산 vs 사전 등록 1.49", conv["total"], exp_reg["total"]),
              ("데이터시트 에너지 직접 (µJ)", E_fe8["datasheet"], E_exp["datasheet"]),
              ("데이터시트 에너지 환산 (µJ)", E_conv["datasheet"], E_exp_conv["datasheet"]),
              ("데이터시트 에너지 환산 vs 사전 등록 35.9", E_conv["datasheet"], E_reg["datasheet"]),
              ("CPU 몫 (ms)", pf["pre"], pe["pre"]), ("p1 추론 (ms)", fe8["infer1"], exp_sess["infer1"])]
    L += ["## 예상 대비 (10 % 기준)", "", "| 항목 | 실측 | 예상 | 차 | 10 % 안 |", "|---|---:|---:|---:|:--:|"]
    J["checks"] = []
    for nm, v, e in checks:
        pct = 100 * (v - e) / e
        L.append("| %s | %.3f | %.3f | %+.1f %% | %s |" % (nm, v, e, pct, "예" if abs(pct) <= 10 else "**아니오**"))
        J["checks"].append({"item": nm, "measured": v, "expected": e, "pct": pct, "within10": abs(pct) <= 10})
    L += ["", "원인 서술은 docs/results/npu-readback-design.md 7.7 에 적는다 (이 파일은 수치만)."]
    J.update({"fe8_seg": fe8, "ref_seg": ref, "fe8_parts": pf, "ref_parts": pr, "exp_session": exp_sess, "exp_registered": exp_reg, "conv": conv,
              "exp_conv": exp_conv, "mac_ratio": r,
              "E": {"fe8": E_fe8, "ref": E_ref, "exp": E_exp, "conv": E_conv, "exp_conv": E_exp_conv, "registered": E_reg, "main_fe": E_main}})
    write(a.out, L, J)


if __name__ == "__main__":
    main()
