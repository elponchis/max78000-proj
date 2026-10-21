#!/usr/bin/env python3
"""원고의 수치를 **문장 단위로** numbers.json 과 대조한다 (2026-10-21 밤 지시).

집합 비교(수치 토큰이 어디엔가 있는가)가 아니라, 문장마다 (수치, 그 문장이 가리키는 구성) 쌍을 뽑아
numbers.json 의 어느 키가 그 값을 내는지, 그 키의 구성이 문장에 언급된 구성과 맞는지를 본다.
다른 구간·다른 구성의 값이 들어간 경우를 잡기 위한 것이다.

판정
  ✓ 키 일치      : 값을 내는 키가 있고, 키의 구성(또는 구성 무관 키)이 문장에 언급된 구성과 맞는다
  ⚠ 구성 확인    : 값을 내는 키는 있으나 그 키의 구성이 문장에 없다 (사람이 본다)
  상수           : 설계 상수·고정값 목록(CONST)에 있는 값
  ✗ 미확인       : 어느 키도 내지 않고 상수 목록에도 없다

사용 (WSL2):
    ~/ai8x-training/venv/bin/python tools/check_paper_numbers.py [docx] [--out docs/paper/ksc2026/number-check.md]
"""
import argparse
import json
import os
import re
import sys

import docx

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(REPO, "docs", "paper")
N = json.load(open(os.path.join(PAPER, "numbers.json"), encoding="utf-8"))
CFG = json.load(open(os.path.join(PAPER, "paper_config.json"), encoding="utf-8"))
DN = {k: v["name"] for k, v in CFG["display"]["configs"].items()}

# 구성 id ↔ 문장에서 쓰는 이름 (make_ksc_paper.py 의 s_w 등 별칭 포함)
ALIAS = {
    "④": ["원본 파형"], "(a)": [DN["(a)"]], "②′": [DN["②′"], "Gabor"], "가 u1000": ["62.5 ms", "62.5·50·25·16"],
    "다 u800": ["50 ms", "62.5·50·25·16"], "간격 400": ["25 ms", "62.5·50·25·16"], "①′": ["16 ms", "62.5·50·25·16"],
    "M4 CMSIS-NN": ["CPU 추론", "CMSIS-NN", "Cortex-M4F", "CPU 로 추론"],
    "②′ b64": ["변형 3종", "변형"], "②′ f31": ["변형 3종", "변형"], "②′ b64f31": ["변형 3종", "변형"],
}
for k, n in DN.items():
    ALIAS.setdefault(k, []).append(n)
# 키 경로에서 구성 id 를 찾는 순서 (긴 것 먼저)
IDS = sorted(ALIAS, key=len, reverse=True)

# 설계 상수·본문 고정값 (결과 수치가 아닌 것). 출처는 CLAUDE.md 4·7장, 펌웨어, 데이터 통계.
CONST = {
    "16": "16 kHz / 16 프레임 / 16 ms", "1.024": "창 길이 s", "16,384": "창 샘플", "250": "hop ms", "256": "②′ 판단 주기 ms",
    "5": "클래스 수 / 절 번호", "8": "8 bit / 8위치", "442": "가중치 메모리 KB", "128": "128×128 접기", "1": "1D / 1 μs / 인용",
    "2": "2D / 인용", "512": "512탭 / 512 값", "32": "cos·sin 32쌍 / 멜 32 / 32×16 / 32비트", "64": "64 대역 / 멜 64 / 64비트",
    "31": "31 프레임", "62.5": "프레임 간격 ms", "50": "프레임 간격 ms / CNN 50 MHz", "25": "프레임 간격 ms",
    "150": "에폭", "3": "시드 수 / 인용", "10": "SysTick 분해능 ns / 인용", "100": "코어 MHz", "24": "SysTick bit",
    "168": "SysTick 한계 ms", "300": "오경보/h", "96.3": "[4] 보고 정확도 %", "90.2": "[4] 비교 하한 %", "96.4": "[4] 비교 상한 %",
    "1,000": "지연 반복 횟수", "2,048": "가속기 출력 개수", "3.8": "(a) NPU 사이클 배수 (synthesis-check)", "2/3": "전처리 지연 비중",
    "0.59": "RISC-V 전력비 하한 (riscv-feasibility.md)", "0.97": "RISC-V 전력비 상한", "2.2": "손익분기 ms 하한", "4.2": "손익분기 ms 상한",
    "60": "RISC-V MHz", "4": "인용 / 절 번호", "6": "인용", "7": "인용", "9": "인용", "11": "인용", "12": "인용", "13": "인용",
    "1.15": "줄간격", "0": "CI 가 0 을 포함", "2022": "연도", "2021": "연도", "2024": "연도", "2023": "연도", "2018": "연도", "2020": "연도",
    "2014": "연도", "2015": "연도", "2017": "연도",
}


def fmts(x):
    """numbers.json 의 값 하나가 원고에 나타날 수 있는 표기들."""
    out = set()
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return out
    for f in (".0f", ".1f", ".2f", ".3f"):
        out.add(format(x, f))
        out.add(format(x, "+" + f))
    out.add(format(x, ",.0f"))
    if isinstance(x, float) and -1.5 <= x <= 1.5:
        out.add(format(100 * x, ".0f"))          # 비율 → %
        out.add(format(100 * x, ".1f"))
    out = {s.replace("-", "−") for s in out} | out
    return out


def derived():
    """생성기가 numbers.json 값 둘 이상으로 **계산해** 적는 값 (make_ksc_paper.py 와 같은 식)."""
    V = lambda k: N[k]["value"]
    f = lambda n, k="f1_S3": V(f"acc.{n}.{k}")["mean"]
    E = lambda n, s="datasheet": V(f"energy.{n}")[s]
    ext_pos = V("truth.ext_positives_real31L")
    out = {
        "f1(①′)−f1(④) 시험": f("①′") - f("④"), "f1(①′)−f1(④) 외부": f("①′", "ext_f1") - f("④", "ext_f1"),
        "100·f1(①′)/PANNs": 100 * f("①′") / V("panns.test_S3")["f1"],
        "E(a)/E(④) 데이터시트": E("(a)") / E("④"), "E(①′)/E(가) 데이터시트": E("①′") / E("가 u1000"),
        "|welch ②′−가|": abs(V("welch.②′ − 가 u1000")["test_f1_S3"][0]), "E(가)−E(②′) 데이터시트": abs(E("가 u1000") - E("②′")),
        "외부 창 수 (양성 합+음성)": sum(ext_pos.values()) + V("truth.ext_negatives_real31L"), "외부 양성 합": sum(ext_pos.values()),
        "100·hop_fraction(M4)": 100 * V("hop_fraction.M4 CMSIS-NN"),
    }
    return out


def index():
    idx = {}
    for name, val in derived().items():
        for s in fmts(val):
            idx.setdefault(s, set()).add("유도: " + name)

    def walk(path, val):
        if isinstance(val, dict):
            for k, v in val.items():
                walk(path + "." + k, v)
        elif isinstance(val, (list, tuple)):
            for i, v in enumerate(val):
                walk(path + f"[{i}]", v)
        else:
            for s in fmts(val):
                idx.setdefault(s, set()).add(path)
    for key, ent in N.items():
        walk(key, ent["value"] if isinstance(ent, dict) and "value" in ent else ent)
    return idx


def config_of(path):
    if path.startswith("유도:"):
        return None
    for cid in IDS:
        if "." + cid + "." in path + "." or "." + cid + "[" in path or path.endswith("." + cid):
            return cid
    for cid in IDS:                              # welch 쌍 'A − B' 처럼 사이에 끼는 경우
        if cid in path:
            return cid
    return None


def sentences(text):
    parts = re.split(r"(?<=[.。])\s+(?=[^\d])", text)
    return [p.strip() for p in parts if p.strip()]


NUM = re.compile(r"(?<![\w.])[+−-]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def segment_table():
    """'이유' 문장의 구간별 에너지 계산 (세 전력 출처). E_구간 = 구간 지연 × 그 구간 전력."""
    V = lambda k: N[k]["value"]
    L = ["## 구간별 에너지 계산 — 원본 파형(④) 대 CPU log-mel 62.5 ms(가), 추정(실측 지연 × 공개 전력)", "",
         "| 전력 출처 | 구성 | 전처리 ms × P_cpu | 적재+출력 ms × P_load | 추론 ms × P_npu | 합 μJ | numbers.json energy |", "|---|---|---|---|---|---|---|"]
    res = {}
    for src, ps in CFG["power_sources"].items():
        if not ps.get("use_in_curve"):
            continue
        for cfg in ("④", "가 u1000"):
            l = V(f"latency.{cfg}")
            e = {"pre": l["pre"] * ps["P_cpu_mW"], "load": (l["load"] + l["out"]) * ps["P_load_mW"], "infer": l["infer"] * ps["P_npu_mW"]}
            res[(src, cfg)] = e
            L.append(f"| {src} | {cfg} | {l['pre']:.2f} × {ps['P_cpu_mW']} = {e['pre']:.1f} | {l['load'] + l['out']:.2f} × {ps['P_load_mW']} = {e['load']:.1f} | "
                     f"{l['infer']:.2f} × {ps['P_npu_mW']} = {e['infer']:.1f} | {sum(e.values()):.1f} | {V(f'energy.{cfg}')[src]:.1f} |")
    L += ["", "| 전력 출처 | 줄어든 적재+추론 에너지 (④ − 가) | 늘어난 전처리 에너지 (가 − ④) | 순 변화 (가 − ④) | 감소 주원인 |", "|---|---|---|---|---|"]
    for src in dict.fromkeys(k[0] for k in res):
        a, b = res[(src, "④")], res[(src, "가 u1000")]
        cut = (a["load"] + a["infer"]) - (b["load"] + b["infer"]); add = b["pre"] - a["pre"]
        main = "추론" if (a["infer"] - b["infer"]) > (a["load"] - b["load"]) else "적재"
        L.append(f"| {src} | {cut:.1f} (적재 {a['load'] - b['load']:.1f} + 추론 {a['infer'] - b['infer']:.1f}) | {add:.1f} | {add - cut:+.1f} | {main} |")
    L += ["", "원고 문장(데이터시트 기준)은 '줄어든 적재·추론 구간의 에너지가 늘어난 CPU 구간보다 커서' 다 — 세 출처 모두 감소분 > 증가분이고, 감소의 주원인은 추론 구간이다(적재 구간도 줄었다). "
          "Moosmann 기준은 순 변화 +0.1 μJ 로 '같다'.", ""]
    return L


def comparisons():
    """원고의 'A 가 B 보다 작다/크다/같다' 류 비교 문장 — 양쪽 값을 계산해 참/거짓."""
    V = lambda k: N[k]["value"]
    f = lambda n, k="f1_S3": V(f"acc.{n}.{k}")["mean"]
    E = lambda n, s="datasheet": V(f"energy.{n}")[s]
    Lt = lambda n: V(f"latency.{n}")["total"]
    W = lambda pair, k="test_f1_S3": V(f"welch.{pair}")[k]
    vm = V("variants.②′.measured")
    rows = []
    def add(where, claim, lhs, rhs, ok):
        rows.append((where, claim, lhs, rhs, "참" if ok else "**거짓**"))
    d = f("①′") - f("④"); add("요약·3절", "원본 파형은 CPU log-mel 16 ms 보다 시험 macro-F1 이 낮다", f"④ {f('④'):.3f}", f"①′ {f('①′'):.3f} (차 {d:.3f})", d > 0)
    d = f("①′", "ext_f1") - f("④", "ext_f1"); add("3절", "외부 F1 도 낮다", f"④ {f('④','ext_f1'):.3f}", f"①′ {f('①′','ext_f1'):.3f} (차 {d:.3f})", d > 0)
    w = W("가 u1000 − ④"); add("요약·3절", "CPU log-mel 62.5 ms 는 원본 파형보다 macro-F1 이 유의하게 높다 (CI 가 0 제외)", f"가 {f('가 u1000'):.3f}", f"④ {f('④'):.3f}; Welch {w[0]:+.3f} [{w[1]:+.3f}, {w[2]:+.3f}]", w[1] > 0)
    w = W("가 u1000 − ④", "ext_f1"); add("3절", "외부 F1 도 유의하게 높다", f"가 {f('가 u1000','ext_f1'):.3f}", f"④ {f('④','ext_f1'):.3f}; {w[0]:+.3f} [{w[1]:+.3f}, {w[2]:+.3f}]", w[1] > 0)
    for src, word in (("datasheet", "적다"), ("literature", "적다"), ("moosmann", "같다")):
        a, b = E("가 u1000", src), E("④", src)
        add("요약·3절·4절", f"가 의 추정 에너지는 ④ 보다 {word} ({src})", f"가 {a:.1f} μJ", f"④ {b:.1f} μJ", (a < b) if word == "적다" else (abs(a - b) < 0.5))
    add("3절", "가 와 ④ 의 판단 1회 지연은 비슷하다", f"가 {Lt('가 u1000'):.2f} ms", f"④ {Lt('④'):.2f} ms", abs(Lt("가 u1000") - Lt("④")) < 0.5)
    r = V("ratio.infer_m4cmsis_over_npu"); add("3절", "NPU 추론은 CPU 추론보다 추론만 157배 빠르다", f"M4 {V('latency.M4 CMSIS-NN')['infer']:.2f} ms", f"④ {V('latency.④')['infer']:.2f} ms (비 {r:.1f})", round(r) == 157)
    r = V("ratio.decision_m4cmsis_over_npu"); add("3절", "판단 1회 89배", f"M4 {Lt('M4 CMSIS-NN'):.2f} ms", f"④ {Lt('④'):.2f} ms (비 {r:.1f})", round(r) == 89)
    h = V("hop_fraction.M4 CMSIS-NN"); add("3절", "CPU 추론은 판단 주기의 90% 를 추론에 쓴다", f"M4 {Lt('M4 CMSIS-NN'):.2f} ms", f"주기 {V('hop_ms')} ms (비 {100*h:.1f}%)", round(100 * h) == 90)
    w = W("(a) − ④"); add("3절", "NPU 근사 log 의 ④ 대비 CI 는 0 을 포함", "", f"{w[0]:+.3f} [{w[1]:+.3f}, {w[2]:+.3f}]", w[1] <= 0 <= w[2])
    r = E("(a)") / E("④"); add("3절", "NPU 근사 log 의 추정 에너지는 ④ 의 3.3배", f"(a) {E('(a)'):.1f} μJ", f"④ {E('④'):.1f} μJ (비 {r:.2f})", round(r, 1) == 3.3)
    w = W("②′ − ④"); add("3절", "②′ 는 '원본 파형 초과' 목표를 넘지 못했다 (CI 가 0 포함)", "", f"{w[0]:+.3f} [{w[1]:+.3f}, {w[2]:+.3f}]", w[1] <= 0)
    w2 = W("②′ − 가 u1000"); add("3절", "②′ 는 '가 와 동등' 목표를 넘지 못했다 (CI 가 0 제외, 음수)", "", f"{w2[0]:+.3f} [{w2[1]:+.3f}, {w2[2]:+.3f}]", w2[2] < 0)
    add("3절", "②′ 의 시험 macro-F1 은 가 보다 0.11 낮다", f"②′ {f('②′'):.3f}", f"가 {f('가 u1000'):.3f} (Welch 차 {w2[0]:+.3f})", round(abs(w2[0]), 2) == 0.11)
    for src in ("datasheet", "literature", "moosmann"):
        add("요약·3절·4절", f"②′ 의 추정 에너지는 ④ 보다 적다 ({src})", f"②′ {E('②′', src):.1f} μJ", f"④ {E('④', src):.1f} μJ", E("②′", src) < E("④", src))
    add("3절", "②′ 와 가 는 비슷한 에너지 (데이터시트 차이 약 4 μJ)", f"②′ {E('②′'):.1f} μJ", f"가 {E('가 u1000'):.1f} μJ (차 {abs(E('가 u1000') - E('②′')):.1f})", round(abs(E("가 u1000") - E("②′"))) == 4)
    add("3절", "②′ 판단 1회 지연 1.85 < ④ 2.53 ms", f"②′ {Lt('②′'):.2f} ms", f"④ {Lt('④'):.2f} ms", Lt("②′") < Lt("④"))
    vf = V("variants.②′.f1_S3"); add("3절", "변형 3종은 ②′ 보다 정확도가 올랐다", f"변형 {vf['min']:.3f}~{vf['max']:.3f}", f"②′ {f('②′'):.3f}", vf["min"] > f("②′"))
    add("3절", "변형 3종은 세 전력 출처 모두에서 CPU 전처리 곡선 아래 (지배됨)", "", str({n: vm[n]["dominated_by_cpu_curve"] for n in vm}), all(vm[n]["dominated_by_cpu_curve"][k] for n in vm for k in ("datasheet", "literature", "moosmann")))
    cs = [100 * vm[n]["cpu_share_of_delta"] for n in vm]; ns = [100 * vm[n]["npu_share_of_delta"] for n in vm]
    add("3절", "늘어난 비용은 CPU 읽기와 NPU 연산이 비슷하게 나눴다 (CPU 몫이 50% 를 넘지 않음)", f"CPU {min(cs):.0f}~{max(cs):.0f}%", f"NPU {min(ns):.0f}~{max(ns):.0f}%", all(c <= 51 for c in cs))
    l4, lg = V("latency.④"), V("latency.가 u1000")
    add("3절", "NPU 입력 적재와 추론이 줄었다 (④ → 가)", f"적재 {l4['load']:.2f} → {lg['load']:.2f} ms", f"추론 {l4['infer']:.2f} → {lg['infer']:.2f} ms", lg["load"] < l4["load"] and lg["infer"] < l4["infer"])
    ps = CFG["power_sources"]["datasheet"]
    cut = (l4["load"] + l4["out"] - lg["load"] - lg["out"]) * ps["P_load_mW"] + (l4["infer"] - lg["infer"]) * ps["P_npu_mW"]; addE = (lg["pre"] - l4["pre"]) * ps["P_cpu_mW"]
    add("3절", "줄어든 적재·추론 구간 에너지 > 늘어난 CPU 구간 에너지 (데이터시트)", f"감소 {cut:.1f} μJ", f"증가 {addE:.1f} μJ", cut > addE)
    seq = [f(n) for n in ("가 u1000", "다 u800", "간격 400", "①′")]
    add("3절", "프레임 간격을 좁힐수록 정확도가 오른다 (62.5 → 50 → 25 → 16 ms)", " → ".join(f"{x:.3f}" for x in seq), "", all(x < y for x, y in zip(seq, seq[1:])))
    l1 = V("latency.①′"); add("3절", "CPU log-mel 16 ms 는 전처리가 지연의 2/3", f"전처리 {l1['pre']:.2f} ms", f"총 {l1['total']:.2f} ms (비 {l1['pre'] / l1['total']:.2f})", 0.6 <= l1["pre"] / l1["total"] <= 0.72)
    r = E("①′") / E("가 u1000"); add("3절", "16 ms 의 에너지는 62.5 ms 의 6배", f"①′ {E('①′'):.1f} μJ", f"가 {E('가 u1000'):.1f} μJ (비 {r:.2f})", round(r) == 6)
    w = W("간격 400 − ①′"); add("3절", "25 ms 와 16 ms 의 차이는 작지만 CI 가 0 을 포함하지 않는다", f"25 ms {f('간격 400'):.3f}", f"16 ms {f('①′'):.3f}; {w[0]:+.3f} [{w[1]:+.3f}, {w[2]:+.3f}]", w[2] < 0 or w[1] > 0)
    front = V("pareto.front")["test_S3"]
    add("3절", "④ 와 (a) 는 파레토 앞면에 없다 (다른 점에 지배)", "앞면 " + " → ".join(front), f"④ E {E('④'):.0f}/F1 {f('④'):.3f} vs 가 E {E('가 u1000'):.0f}/F1 {f('가 u1000'):.3f}; (a) E {E('(a)'):.0f}/F1 {f('(a)'):.3f}", "④" not in front and "(a)" not in front and E("가 u1000") < E("④") and f("가 u1000") > f("④"))
    add("3절", "외부 데이터의 앞면 순서가 시험과 같다", " → ".join(front), " → ".join(V("pareto.front").get("ext", [])), V("pareto.front").get("ext") == front)
    pn = V("panns.test_S3")["f1"]; add("3절", "CPU log-mel 16 ms 는 PANNs 의 99%", f"①′ {f('①′'):.3f}", f"PANNs {pn:.3f} (비 {100 * f('①′') / pn:.1f}%)", round(100 * f("①′") / pn) == 99)
    sp = V("firmware.cpu_split"); add("3절", "②′ CPU 몫의 80% 는 읽기, 20% 는 로그", f"읽기 {sp['read_ms']:.3f} ms", f"로그 {sp['log_ms']:.3f} ms, CPU {sp['cpu_ms_this_run']:.3f} ms", round(100 * sp["read_ms"] / sp["cpu_ms_this_run"]) == 80)
    ap = V("avg_power")["configs"]
    add("4절", "가 의 판단 비용은 평균 전력의 17% 이하 (대기 하한)", f"{ap['가 u1000']['share_max_pct']:.1f}%", "", round(ap["가 u1000"]["share_max_pct"]) == 17)
    add("4절", "①′ 는 54% 이하", f"{ap['①′']['share_max_pct']:.1f}%", "", round(ap["①′"]["share_max_pct"]) == 54)
    add("4절", "CPU 추론은 대기 상한에서도 42% 이상", f"{ap['M4 CMSIS-NN']['share_min_pct']:.1f}%", "", round(ap["M4 CMSIS-NN"]["share_min_pct"]) == 42)
    add("4절", "62.5 ms 구성의 우위는 '적다' 와 '같다' 사이 (출처별)", f"데이터시트 {E('가 u1000'):.0f}<{E('④'):.0f}, 선행연구 {E('가 u1000','literature'):.0f}<{E('④','literature'):.0f}", f"Moosmann {E('가 u1000','moosmann'):.1f} ≈ {E('④','moosmann'):.1f}", E("가 u1000") < E("④") and abs(E("가 u1000", "moosmann") - E("④", "moosmann")) < 0.5)
    L = ["## 비교 문장 참/거짓 표 ('A 가 B 보다 작다/크다/같다' 류, 양쪽 값을 numbers.json 에서 계산)", "",
         "| 절 | 주장 | 좌변 | 우변 | 판정 |", "|---|---|---|---|---|"]
    for r in rows:
        L.append("| " + " | ".join(str(x).replace("|", "\|") for x in r) + " |")
    nbad = sum(1 for r in rows if "거짓" in r[4])
    L += ["", f"비교 문장 {len(rows)}개 중 거짓 {nbad}개.", ""]
    return L, nbad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("docx", nargs="?", default=os.path.join(PAPER, "ksc2026", "ksc2026_review.docx"))
    ap.add_argument("--out", default=os.path.join(PAPER, "ksc2026", "number-check.md"))
    a = ap.parse_args()
    idx = index()
    d = docx.Document(a.docx)
    rows, sec = [], "(제목·요약)"
    items = []
    for el in d.element.body.iterchildren():
        tag = el.tag.split("}")[1]
        if tag == "p":
            p = docx.text.paragraph.Paragraph(el, d)
            t = p.text.strip().replace("‑", "-")      # 줄바꿈 금지 하이픈(음수 기호) → 대조용 "-"
            if not t:
                continue
            if re.match(r"^(\d\. |참고문헌)", t) and any(r.bold for r in p.runs if r.text.strip()):
                sec = t
                continue
            items.append((sec, t))
        elif tag == "tbl":
            tb = docx.table.Table(el, d)
            for r in tb.rows:
                cells = [c.text.strip().replace("‑", "-") for c in r.cells]
                items.append((sec, "[표] " + " | ".join(cells)))
    stats = {"✓": 0, "⚠": 0, "상수": 0, "✗": 0}
    for sec, text in items:
        if sec.startswith("참고문헌"):
            continue
        for s in sentences(text):
            s_clean = re.sub(r"\[[\d, ]+\]", "", s)              # 인용 번호 제외
            s_clean = re.sub(r"(표|그림)\s*\d(\(b\))?", "", s_clean)   # 표·그림 번호 제외
            mentioned = [cid for cid in IDS if any(al in s for al in ALIAS[cid])]
            toks = [m.group(0).replace("−", "-") for m in NUM.finditer(s_clean)]
            toks = [t.rstrip(".,") for t in toks]
            if not toks:
                continue
            for t in toks:
                t0 = t.lstrip("+")
                keys = idx.get(t, set()) | idx.get(t0, set())
                if t.startswith("+") or t.startswith("-"):
                    keys = idx.get(t, set())
                if keys:
                    cids = {config_of(k) for k in keys}
                    ok = any(c is None or c in mentioned for c in cids) or ("[표]" in s and cids & set(mentioned))
                    if "[표]" in s:
                        ok = bool(cids & set(mentioned)) or (None in cids)
                    verdict = "✓" if ok else "⚠"
                    kstr = "; ".join(sorted(keys))[:160]
                    if not ok and (t0 in CONST or t in CONST):      # 구성이 안 맞는 키보다 상수 설명을 우선
                        verdict, kstr = "상수", CONST.get(t0, CONST.get(t))
                elif t0 in CONST or t in CONST:
                    verdict, kstr = "상수", CONST.get(t0, CONST.get(t))
                else:
                    verdict, kstr = "✗", ""
                stats[verdict] += 1
                rows.append((sec, s[:70].replace("|", "\\|"), t, ", ".join(ALIAS[c][0] for c in mentioned) or "—", kstr.replace("|", "\\|"), verdict))
    L = ["# 원고 수치 대조표 — 문장 단위 (수치 · 가리키는 구성 · numbers.json 키)", "",
         f"원고: `{os.path.relpath(a.docx, REPO)}`. 방법: 문장마다 수치 토큰을 뽑고, numbers.json 의 어느 키가 그 표기(.0f/.1f/.2f/.3f/±/%/천 단위)를 내는지 찾은 뒤, "
         "키 경로의 구성 id 가 문장에 언급된 구성(별칭 포함)과 맞는지 본다. 참고문헌·인용 번호·표/그림 번호는 제외.", "",
         f"판정 집계: ✓ {stats['✓']} · ⚠ 구성 확인 {stats['⚠']} · 상수 {stats['상수']} · ✗ 미확인 {stats['✗']}", "",
         "| 절 | 문장 (앞 70자) | 수치 | 문장의 구성 | numbers.json 키 (또는 상수 설명) | 판정 |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| " + " | ".join(r) + " |")
    cmp_lines, nbad = comparisons()
    L += [""] + segment_table() + cmp_lines
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"{a.out}: {len(rows)} 수치, {stats}; 비교 문장 거짓 {nbad}")
    bad = [r for r in rows if r[5] in ("⚠", "✗")]
    for r in bad:
        print(f"  [{r[5]}] {r[0]} | {r[2]} | 구성 {r[3]} | {r[4][:90]} | {r[1][:50]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
