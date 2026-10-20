#!/usr/bin/env python3
"""feat/log32 — 보드 A/B 로그(`-DFE_LOG_AB`, 같은 POR 세션의 옛 64비트 → 새 32비트 정수 로그)를 읽어
구성별 전·후 구간 지연, 세 출처 에너지, 주 값 로그와의 대조, CPU 몫 비율(읽기/로그), 파레토 앞면 변화를 적는다.
원고·numbers.json 은 건드리지 않는다 (보고 전용).

실행 (WSL2): python3 tools/log32_report.py  → docs/results/log32-board.md + log32-board.json
입력: data/logs-local/board_v4d_<tag>_log32.log (tag: fe, fe_split, fe_b64, fe_f31, fe_b64f31)
"""
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(REPO, "data", "logs-local")
SEGS = ["load1", "cfg1", "infer1", "cpu", "load2", "cfg2", "infer2", "unload", "total", "cpu_rd", "cpu_lg"]
# tag → (numbers.json 이름, 주 값 로그 태그, 보조(지연 정지 펌웨어) 로그 태그, 로그 횟수/판단)
CFG = {"fe": ("②′", "fe_oldfw", "fe", 128), "fe_b64": ("②′ b64", "fe_b64", None, 256),
       "fe_f31": ("②′ f31", "fe_f31", None, 256), "fe_b64f31": ("②′ b64f31", "fe_b64f31", None, 512)}
FRONT_NAMES = ["④", "②′", "②′ b64", "②′ f31", "②′ b64f31", "가 u1000", "다 u800", "간격 400", "①′"]


def parse_block(txt):
    out = {}
    for m in re.finditer(r"^(\w+)\s+us\s+min\s+([\d.]+)\s+med\s+([\d.]+)\s+max\s+([\d.]+)", txt, re.M):
        out[m.group(1)] = {"min": float(m.group(2)), "med": float(m.group(3)), "max": float(m.group(4))}
    return out


def parse_ab(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    passes = {}
    for m in re.finditer(r"=== PASS ([AB]) \((.*?)\): (\d+) iterations, output mismatches: (\d+)\s+\(p1 (\d+), feature (\d+), p2 (\d+)\)(.*?)=== PASS END ===", txt, re.S):
        passes[m.group(1)] = {"label": m.group(2), "n": int(m.group(3)), "mismatch": int(m.group(4)), "p1": int(m.group(5)),
                              "feat": int(m.group(6)), "p2": int(m.group(7)), "seg": parse_block(m.group(8))}

    def kat(key):
        return key in txt and "PASS" in txt.split(key)[1].split("\n")[0]
    k = {"p1": kat("KAT p1 sampleoutput"), "feat": kat("KAT cpu feature"), "p2": kat("KAT sampleoutput:")}
    build = re.search(r"^build (.*?)  SystemCoreClock", txt, re.M)
    return passes, k, (build.group(1) if build else "?")


def parse_main(tag):
    p = os.path.join(LOGS, f"board_v4d_{tag}.log")
    if not os.path.isfile(p):
        return None
    txt = open(p, encoding="utf-8", errors="replace").read()
    return parse_block(txt.split("iterations, output mismatches")[-1])


def energy_from(pre, load, infer, out, PS):
    """표 2 와 같은 식 (②′ 계열: 전처리는 NPU 켜진 채 → P_load). 단위 μJ."""
    return {s: pre * ps["P_load_mW"] + (load + out) * ps["P_load_mW"] + infer * ps["P_npu_mW"] for s, ps in PS.items()}


def parts(seg):
    def ms(k):
        return seg[k]["max"] / 1e3
    return {"pre": ms("cpu"), "load": ms("load1") + ms("cfg1") + ms("load2") + ms("cfg2"), "infer": ms("infer1") + ms("infer2"),
            "out": ms("unload"), "total": ms("total")}


def pareto(points):
    """points: name → (E, F1). 에너지 ≤ 이고 F1 ≥ 인 다른 점이 없으면 앞면."""
    front = []
    for n, (e, f) in points.items():
        dominated = any(e2 <= e and f2 >= f and (e2 < e or f2 > f) for n2, (e2, f2) in points.items() if n2 != n)
        if not dominated:
            front.append(n)
    return sorted(front, key=lambda n: points[n][0])


def main():
    C = json.load(open(os.path.join(REPO, "docs", "paper", "paper_config.json"), encoding="utf-8"))
    PS = {k: v for k, v in C["power_sources"].items() if v.get("use_in_curve")}
    N = json.load(open(os.path.join(REPO, "docs", "paper", "numbers.json"), encoding="utf-8"))
    L = ["# 정수 로그 32비트화 — 보드 A/B 실측 (feat/log32, `tools/log32_report.py` 생성)", "",
         "같은 바이너리·같은 POR 세션에서 옛(64비트 나눗셈 `__aeabi_ldivmod`) → 새(32비트 `sdiv`) 순서로 1,000회씩 (`-DFE_LOG_AB`). "
         "지연은 1,000회 최악값(ms). 에너지는 **추정(실측 지연 × 공개 전력)**, 표 2 와 같은 식. 사전 등록: `fe-curve-design.md` 10절. **원고·numbers.json 미반영.**", ""]
    res = {}
    for tag, (name, main_tag, aux_tag, n_log) in CFG.items():
        p = os.path.join(LOGS, f"board_v4d_{tag}_log32.log")
        if not os.path.isfile(p):
            L += [f"## {name}: 로그 없음 ({os.path.relpath(p, REPO)})", ""]
            continue
        passes, kat, build = parse_ab(p)
        if "A" not in passes or "B" not in passes:
            L += [f"## {name}: PASS A/B 블록을 찾지 못함", ""]
            continue
        A, B = passes["A"], passes["B"]
        PA, PB = parts(A["seg"]), parts(B["seg"])
        main_seg, aux_seg = parse_main(main_tag), (parse_main(aux_tag) if aux_tag else None)
        L += [f"## {name} (빌드 {build}; KAT p1 / CPU 특징 / p2 비트 일치 = {kat['p1']} / {kat['feat']} / {kat['p2']})", "",
              f"1,000회 불일치: 옛 {A['mismatch']} (p1 {A['p1']}, 특징 {A['feat']}, p2 {A['p2']}) / 새 {B['mismatch']} (p1 {B['p1']}, 특징 {B['feat']}, p2 {B['p2']})", ""]
        hdr = "| 구간 (최악, ms) | 주 값 로그 |" + (" 지연 정지 펌웨어 로그 |" if aux_seg else "") + " 옛 (이 세션) | 새 (이 세션) | 새 − 옛 | 옛 vs 주 값 |"
        L += [hdr, "|---|---:|" + ("---:|" if aux_seg else "") + "---:|---:|---:|---:|"]
        for s in SEGS:
            if s not in A["seg"]:
                continue
            a, b = A["seg"][s]["max"] / 1e3, B["seg"][s]["max"] / 1e3
            m = main_seg[s]["max"] / 1e3 if main_seg and s in main_seg else None
            x = aux_seg[s]["max"] / 1e3 if aux_seg and s in aux_seg else None
            row = f"| {s} | " + (f"{m:.3f}" if m is not None else "—") + " |" + ((f" {x:.3f} |" if x is not None else " — |") if aux_seg else "")
            row += f" {a:.3f} | {b:.3f} | {b - a:+.3f} | " + (f"{100 * (a - m) / m:+.1f} % |" if m else "— |")
            L.append(row)
        d_cpu = PB["pre"] - PA["pre"]
        per_log = -d_cpu * 1e3 * 100 / n_log      # ms → us → cycles @100 MHz
        EA = energy_from(PA["pre"], PA["load"], PA["infer"], PA["out"], PS)
        EB = energy_from(PB["pre"], PB["load"], PB["infer"], PB["out"], PS)
        L += ["", f"로그 {n_log}회/판단, CPU 몫 **{d_cpu:+.3f} ms** → 로그 1회 **{per_log:.0f} 사이클** 절감. "
                  f"판단 지연 {PA['total']:.3f} → **{PB['total']:.3f} ms** ({PB['total'] - PA['total']:+.3f}).", ""]
        # 주 값 기준 환산 (CPU 몫만 Δ 만큼 뺀다 — CPU 구간은 네트워크 정지 방식과 무관)
        Lm = N[f"latency.{name}"]["value"]
        Em = N[f"energy.{name}"]["value"]
        conv = {"pre": Lm["pre"] + d_cpu, "load": Lm["load"], "infer": Lm["infer"], "out": Lm["out"], "total": Lm["total"] + d_cpu}
        Ec = energy_from(conv["pre"], conv["load"], conv["infer"], conv["out"], PS)
        L += ["| 에너지 μJ (추정) | " + " | ".join(PS[s]["label"] for s in PS) + " |", "|---|" + "---:|" * len(PS),
              "| 주 값 (numbers.json) | " + " | ".join(f"{Em[s]:.1f}" for s in PS) + " |",
              "| 옛 (이 세션) | " + " | ".join(f"{EA[s]:.1f}" for s in PS) + " |",
              "| 새 (이 세션) | " + " | ".join(f"{EB[s]:.1f}" for s in PS) + " |",
              "| 새 − 옛 (이 세션) | " + " | ".join(f"{EB[s] - EA[s]:+.1f}" for s in PS) + " |",
              f"| **주 값 환산** (주 값 CPU 몫 {Lm['pre']:.3f} {d_cpu:+.3f} ms, 나머지 주 값; 판단 {conv['total']:.3f} ms) | "
              + " | ".join(f"**{Ec[s]:.1f}**" for s in PS) + " |", ""]
        res[tag] = {"name": name, "old": PA, "new": PB, "E_old": EA, "E_new": EB, "main": Lm, "E_main": Em, "converted": conv,
                    "E_converted": Ec, "d_cpu_ms": d_cpu, "per_log_cycles": per_log, "mismatch": [A["mismatch"], B["mismatch"]],
                    "kat": kat, "build": build}
    # 분해 로그 (별도 파일, 실험 II 조건)
    sp = os.path.join(LOGS, "board_v4d_fe_split_log32.log")
    if os.path.isfile(sp):
        passes, kat, build = parse_ab(sp)
        if "A" in passes and "B" in passes and "cpu_rd" in passes["B"]["seg"]:
            A, B = passes["A"], passes["B"]
            rd, lg = B["seg"]["cpu_rd"]["max"] / 1e3, B["seg"]["cpu_lg"]["max"] / 1e3
            rd0, lg0 = A["seg"]["cpu_rd"]["max"] / 1e3, A["seg"]["cpu_lg"]["max"] / 1e3
            old = N["firmware.cpu_split"]["value"]
            v = "A" if rd / (rd + lg) >= 0.70 else "B"
            v0 = "A" if rd0 / (rd0 + lg0) >= 0.70 else "B"
            L += [f"## ②′ CPU 몫 분해 (`-DFE_CPU_SPLIT -DFE_LOG_AB`, 빌드 {build}; 불일치 옛 {A['mismatch']} / 새 {B['mismatch']})", "",
                  "| | 읽기 (ms) | 로그 (ms) | 읽기 / 로그 (%) | CPU 몫 (ms) | 판정 |", "|---|---:|---:|---:|---:|---|",
                  f"| 원고 (numbers.json firmware.cpu_split) | {old['read_ms']:.3f} | {old['log_ms']:.3f} | {old['read_pct']} / {old['log_pct']} | {old['cpu_ms_this_run']:.3f} | {old['verdict']} |",
                  f"| 옛 (이 세션) | {rd0:.3f} | {lg0:.3f} | {100 * rd0 / (rd0 + lg0):.0f} / {100 * lg0 / (rd0 + lg0):.0f} | {A['seg']['cpu']['max'] / 1e3:.3f} | {v0} |",
                  f"| **새 (이 세션)** | {rd:.3f} | {lg:.3f} | **{100 * rd / (rd + lg):.0f} / {100 * lg / (rd + lg):.0f}** | {B['seg']['cpu']['max'] / 1e3:.3f} | **{v}** |", "",
                  f"로그 1회: 옛 {lg0 * 1e3 * 100 / 128:.0f} → 새 {lg * 1e3 * 100 / 128:.0f} 사이클 (128회/판단, 밀기 memmove 포함). "
                  f"읽기는 워드당 {rd * 1e3 * 100 / 2048:.1f} 사이클 그대로.", ""]
            res["split"] = {"rd": rd, "lg": lg, "rd_old": rd0, "lg_old": lg0, "rd_pct": round(100 * rd / (rd + lg)),
                            "lg_pct": round(100 * lg / (rd + lg)), "verdict": v, "verdict_old": old["verdict"]}
    # 파레토 앞면: 데이터시트 에너지 × 시험 macro-F1(시드 평균). 전 = numbers.json, 후 = ②′ 계열을 주 값 환산으로 치환
    before = {n: (N[f"energy.{n}"]["value"]["datasheet"], N[f"acc.{n}.f1_S3"]["value"]["mean"]) for n in FRONT_NAMES}
    after = dict(before)
    for r in res.values():
        if isinstance(r, dict) and "E_converted" in r:
            after[r["name"]] = (r["E_converted"]["datasheet"], before[r["name"]][1])
    fb, fa = pareto(before), pareto(after)
    L += ["## 파레토 앞면 (데이터시트 에너지 μJ × 시험 macro-F1 시드 평균; ②′ 계열은 주 값 환산)", "",
          "| 구성 | μJ 전 | μJ 후 | F1 | 앞면 전 | 앞면 후 |", "|---|---:|---:|---:|:-:|:-:|"]
    for n in sorted(after, key=lambda n: after[n][0]):
        L.append(f"| {n} | {before[n][0]:.1f} | {after[n][0]:.1f} | {after[n][1]:.3f} | {'○' if n in fb else '—'} | {'○' if n in fa else '—'} |")
    L += ["", f"앞면 순서 전: {' → '.join(fb)}", "", f"앞면 순서 후: {' → '.join(fa)}", "",
          f"→ 앞면·순서 **{'같음' if fb == fa else '바뀜'}**.", ""]
    if "fe" in res:
        for s in PS:
            e_fe, e_ga = res["fe"]["E_converted"][s], N["energy.가 u1000"]["value"][s]
            L.append(f"- {PS[s]['label']}: ②′ 환산 {e_fe:.1f} μJ vs 가 u1000 {e_ga:.1f} μJ (차이 {e_ga - e_fe:+.1f}; 전 "
                     f"{e_ga - N['energy.②′']['value'][s]:+.1f}) / ④ {N['energy.④']['value'][s]:.1f}")
        L.append("")
    res["pareto"] = {"before": fb, "after": fa, "points_before": before, "points_after": after}
    # 변형 3종의 ②′ 대비 증가분 분해 (원고 4절 "주된 비용은 … CPU 몫" 문장의 근거, numbers.json variants.②′.measured 와 같은 정의)
    if "fe" in res and all(t in res for t in ("fe_b64", "fe_f31", "fe_b64f31")):
        VM = N["variants.②′.measured"]["value"]
        L += ["## 변형 3종의 ②′ 대비 증가분 분해 (cpu 몫 / NPU 추론 몫, 주 값 환산)", "",
              "| 변형 | Δ판단 전 (ms) | CPU 몫 전 | NPU 몫 전 | Δ판단 후 (ms) | CPU 몫 후 | NPU 몫 후 |", "|---|---:|---:|---:|---:|---:|---:|"]
        shares = {}
        for tag in ("fe_b64", "fe_f31", "fe_b64f31"):
            n = res[tag]["name"]; c, f = res[tag]["converted"], res["fe"]["converted"]
            dt, dc, dn = c["total"] - f["total"], c["pre"] - f["pre"], c["infer"] - f["infer"]
            shares[n] = {"cpu": dc / dt, "npu": dn / dt, "d_total": dt}
            L.append(f"| {n} | {VM[n]['delta_total_vs_fe_ms']:.3f} | {100 * VM[n]['cpu_share_of_delta']:.0f} % | {100 * VM[n]['npu_share_of_delta']:.0f} % | "
                     f"{dt:.3f} | **{100 * dc / dt:.0f} %** | {100 * dn / dt:.0f} % |")
        cpu_main_before = all(VM[n]["cpu_share_of_delta"] > 0.5 for n in VM)
        cpu_main_after = all(s["cpu"] > 0.5 for s in shares.values())
        L += ["", f"원고 규칙 `cpu_main` (세 변형 모두 CPU 몫 > 50 %): 전 **{cpu_main_before}** → 후 **{cpu_main_after}**. "
              + ("→ 4절 변형 문단의 '주된 비용은 NPU 연산이 아니라 CPU 몫' 문장이 **성립하지 않게 된다** (분기가 '나뉘었다' 로 바뀐다)." if cpu_main_before and not cpu_main_after else ""), ""]
        res["delta_shares"] = {"after": shares, "cpu_main_before": cpu_main_before, "cpu_main_after": cpu_main_after}
    out = os.path.join(REPO, "docs", "results", "log32-board.md")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(res, open(os.path.join(REPO, "docs", "results", "log32-board.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
