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
            t = p.text.strip()
            if not t:
                continue
            if re.match(r"^(\d\. |참고문헌)", t) and any(r.bold for r in p.runs if r.text.strip()):
                sec = t
                continue
            items.append((sec, t))
        elif tag == "tbl":
            tb = docx.table.Table(el, d)
            for r in tb.rows:
                cells = [c.text.strip() for c in r.cells]
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
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"{a.out}: {len(rows)} 수치, {stats}")
    bad = [r for r in rows if r[5] in ("⚠", "✗")]
    for r in bad:
        print(f"  [{r[5]}] {r[0]} | {r[2]} | 구성 {r[3]} | {r[4][:90]} | {r[1][:50]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
