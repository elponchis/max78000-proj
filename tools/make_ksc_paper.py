#!/usr/bin/env python3
"""KSC2026 심사용 논문 초안 (2~3쪽, 국문) 을 **결과 파일에서** DOCX 로 생성한다.

원칙 (2026-10-16 지시와 같다): 본문의 모든 수치는 `docs/paper/numbers.json` / `table*.csv` / `paper_config.json`
에서 읽는다. 이 스크립트에 결과 수치를 직접 적지 않는다. 예외는 아래 `V1_NEG` 하나뿐이고 (v1 데이터 시절의
NPU 쪽 변형 결과, 결과 파일이 아니라 docs/STATUS.md 1.3절의 표에서 옮김) 출처를 적어 두었다.

양식: KSC2026 심사용 (A4, 위 30 / 아래 20 / 좌우 10 mm, 제목·요약 1단, 본문 2단, 9pt 이상, 저자 정보 없음,
참고문헌 포함 2~3쪽). 글꼴은 맑은 고딕 (양식은 글꼴 자유).

사용 (WSL2):
    ~/ai8x-training/venv/bin/python tools/make_ksc_paper.py
    # → docs/paper/ksc2026/ksc2026_review.docx (+ soffice 가 있으면 .pdf 와 쪽수)
"""
import csv
import json
import os
import shutil
import subprocess
import sys

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(REPO, "docs", "paper")
OUT_DIR = os.path.join(PAPER, "ksc2026")
N = json.load(open(os.path.join(PAPER, "numbers.json"), encoding="utf-8"))
CFG = json.load(open(os.path.join(PAPER, "paper_config.json"), encoding="utf-8"))
DN = {k: v["name"] for k, v in CFG["display"]["configs"].items()}
FONT = "맑은 고딕"
BODY_PT = 9

# ⚠️ 유일한 예외 — v1 데이터 시절 NPU 쪽 변형의 ④ 대비 차이 (@300/h macro-F1, Welch 95% CI).
#    출처: docs/STATUS.md 1.3절 표 (g8-first-comparison.md, g8-d1-filterbank.md, errata-update-model.md).
#    결과 파일(json)이 없어 여기 옮겼다. 논문에는 "v1 데이터" 임을 밝힌다.
#    다섯 가지 = D-1 (멜 필터뱅크 초기화 Conv1d+Abs) / D-1 탭 256·512 (k=2·k=4, 한 묶음) / (a) v1 (구간선형 로그) /
#    (c) (NPU 필터뱅크 + CPU 정수 로그 분할) / wave2D (원본 파형 2D 재배열). CI 가 있는 셋(D-1, (a), (c))은 모두 0 을 포함한다.
V1_NEG = [("D-1", -0.026, (-0.057, +0.006)), ("D-1 k=2", -0.050, None), ("D-1 k=4", -0.032, None),
          ("(a) v1", +0.003, (-0.028, +0.034)), ("(c)", -0.012, (-0.042, +0.018)), ("wave2D", -0.019, None)]
assert all(ci is None or ci[0] <= 0 <= ci[1] for _, _, ci in V1_NEG)
V1_LO, V1_HI = min(d for _, d, _ in V1_NEG), max(d for _, d, _ in V1_NEG)

# ⚠️ 두 번째 예외 — RISC-V 검토 (docs/results/riscv-gate.md 1.3~1.4절·5절, riscv-feasibility.md 0절·4절, 2026-10-20).
#    데이터시트 19-100868 Rev 1 p.9~10 전류 행으로 계산한 "데이터시트 기반 추정" 이며 결과 파일(json)이 없어 여기 옮겼다.
#    판단 비용 0.173 mW 는 아래 build() 에서 numbers.json 의 가 u1000 에너지 × 4/s 로 다시 계산한다.
#    평균 전력 분해(판단 비용 비중, 대기 전력 하한·상한)는 numbers.json 의 avg_power 항목에서 읽는다 (make_paper_assets.py 가 계산).
RISCV = {"p_rv_over_m4": (0.59, 0.97),    # RV32 활성 전력 / M4 활성 전력 (riscv-feasibility.md 1.1절 사례 A~C)
         "breakeven_ms": (2.2, 4.2)}      # RV32 전처리가 가 보다 에너지가 적으려면 끝나야 하는 전처리 지연 (riscv-feasibility.md 4절)


def v(key):
    return N[key]["value"]


def f1(name, k="f1_S3"):
    a = v(f"acc.{name}.{k}")
    return a["mean"], a["sd"]


def fmt_pm(name, k="f1_S3", d=3):
    m, s = f1(name, k)
    return f"{m:.{d}f} ± {s:.{d}f}"


def welch(pair, k="test_f1_S3"):
    d, lo, hi = v(f"welch.{pair}")[k]
    return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def energy(name, src="datasheet"):
    return v(f"energy.{name}")[src]


def latency(name):
    return v(f"latency.{name}")["total"]


# ── DOCX 도우미 ──────────────────────────────────────────────────────────
def set_font(run, size=BODY_PT, bold=False, italic=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), FONT)


def para(doc, text="", size=BODY_PT, bold=False, align=None, space_after=2, indent=False, line=1.05, italic=False):
    p = doc.add_paragraph()
    if text:
        r = p.add_run(text)
        set_font(r, size, bold, italic)
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(0)
    pf.line_spacing = line
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "just":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if indent:
        pf.first_line_indent = Pt(9)
    return p


def body(doc, text):
    return para(doc, text, align="just", indent=True)


def heading(doc, text, size=10):
    return para(doc, text, size=size, bold=True, space_after=2)


def set_cols(section, n, space_mm=6.35):
    sectPr = section._sectPr
    cols = sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sectPr.append(cols)
    cols.set(qn("w:num"), str(n))
    cols.set(qn("w:space"), str(int(space_mm / 25.4 * 1440)))


def new_section(doc, n_cols):
    s = doc.add_section(WD_SECTION.CONTINUOUS)
    set_cols(s, n_cols)
    return s


def page_setup(section):
    section.page_height, section.page_width = Mm(297), Mm(210)
    section.top_margin, section.bottom_margin = Mm(30), Mm(20)
    section.left_margin, section.right_margin = Mm(10), Mm(10)
    section.header_distance = section.footer_distance = Mm(0)


def figure(doc, path, width_mm, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(1)
    p.add_run().add_picture(path, width=Mm(width_mm))
    para(doc, caption, size=8, align="just", space_after=4, line=1.05)


def table(doc, header, rows, caption, col_mm=None, size=7.5):
    para(doc, caption, size=8, align="just", space_after=1, line=1.05)
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = "Table Grid"
    if col_mm:                       # 열 폭 고정 (autofit 이면 LibreOffice 가 내용 길이로 다시 나눈다)
        t.autofit = False
        for j, w in enumerate(col_mm):
            t.columns[j].width = Mm(w)
    for j, h in enumerate(header):
        c = t.rows[0].cells[j]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        set_font(r, size, bold=True)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.rows[i + 1].cells[j]
            c.text = ""
            r = c.paragraphs[0].add_run(str(val))
            set_font(r, size)
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if j else WD_ALIGN_PARAGRAPH.LEFT
    for row in t.rows:
        for j, c in enumerate(row.cells):
            c.paragraphs[0].paragraph_format.space_after = Pt(0)
            c.paragraphs[0].paragraph_format.space_before = Pt(0)
            if col_mm:
                c.width = Mm(col_mm[j])
    para(doc, "", size=4, space_after=2)
    return t


# ── 본문 ─────────────────────────────────────────────────────────────────
def build():
    os.makedirs(OUT_DIR, exist_ok=True)
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(BODY_PT)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    page_setup(doc.sections[0])
    set_cols(doc.sections[0], 1)

    # 이름
    W, A, GA, DA, H25, H16, M4, FE = "④", "(a)", "가 u1000", "다 u800", "간격 400", "①′", "M4 CMSIS-NN", "②′"
    n_w, n_a, n_ga, n_da, n_25, n_16 = (DN[k] for k in (W, A, GA, DA, H25, H16))
    s_w, s_ga, s_25, s_16 = "원본 파형 구성", "CPU log-mel 62.5 ms 구성", "CPU log-mel 25 ms", "CPU log-mel 16 ms"
    tr = v("truth.testL_S3_windows")
    ext_pos, ext_neg = v("truth.ext_positives_real31L"), v("truth.ext_negatives_real31L")
    n_ext = sum(ext_pos.values()) + ext_neg
    r_inf, r_dec = v("ratio.infer_m4cmsis_over_npu"), v("ratio.decision_m4cmsis_over_npu")
    pre_first, pre_inc = v("preproc.first_impl_ms"), v("preproc.inc_v1_ms")
    hop_m4 = 100 * v("hop_fraction.M4 CMSIS-NN")
    pann = v("panns.test_S3")["f1"]
    lat_ga = v("latency.가 u1000")

    # ── 1단: 제목·요약 ────────────────────────────────────────────────
    para(doc, "[학부생논문] NPU 탑재 MCU 의 상시 음향 이벤트 감지에서 전처리 위치가 정확도와 에너지에 미치는 영향: MAX78000 실측",
         size=14, bold=True, align="center", space_after=4, line=1.2)
    para(doc, "Where Should the Front-End Go? Measured Accuracy–Energy Trade-offs of Preprocessing Placement "
              "for Always-On Acoustic Event Detection on an NPU-Equipped MCU (MAX78000)",
         size=11, align="center", space_after=8, line=1.15)
    para(doc, "요   약", size=10, bold=True, align="center", space_after=3)
    pf = [DN[n] for n in v("pareto.front")["test_S3"]]
    fe_cpu_ms = v("latency.②′")["detail"]["cpu"]
    para(doc,
         f"NPU 를 탑재한 MCU(MAX78000) 에서 배터리 상시 구동 음향 이벤트 감지기를 구현하고, log-mel 전처리를 "
         f"어디에 둘지가 정확도와 판단당 에너지에 미치는 영향을 {len(CFG['configs'])}개 구성과 CPU 추론 대조군으로 비교했다. "
         f"정확도와 판단 1회 지연은 같은 데이터·같은 보드에서 실측했고, 에너지는 실측 지연과 공개 전력값으로 추정했다. "
         f"벤더 참조 설계처럼 원본 파형을 NPU 에 직접 넣어 전처리를 없앤 구성은 가장 정확한 CPU log-mel 구성(간격 16 ms)보다 시험 데이터 macro-F1 이 {f1(H16)[0]-f1(W)[0]:.3f} 낮았고, "
         f"로그 압축을 NPU 연산으로 근사한 변형은 이 격차를 되찾지 못했으며, 선형 부분(필터뱅크와 에너지 적분)만 NPU 로 옮긴 구성은 "
         f"원본 파형 구성보다 에너지가 적지만 정확도는 원본 파형 구성과 유의한 차이가 없었다. 반면 프레임 간격을 넓힌 거친 log-mel을 CPU 에 남기고 작은 2D CNN 을 "
         f"NPU 에 올린 구성은 원본 파형 구성보다 macro-F1 이 {v('welch.가 u1000 − ④')['test_f1_S3'][0]:+.3f} 높으면서(Welch 95% CI 가 0 을 "
         f"포함하지 않음) 추정 에너지는 같거나 적었다. NPU 추론은 같은 가중치의 CMSIS-NN 추론보다 판단 1회 기준 {r_dec:.0f}배 빨랐다. "
         f"전처리 위치는 가속기 친화성의 문제가 아니라 정확도–에너지 설계 변수이며, 파레토 앞면은 양 끝이 아니라 전처리를 나눈 "
         f"중간 구성들로 이루어짐을 보인다.",
         align="just", space_after=6, line=1.15)

    # ── 1단 띠: 그림 1 + 표 1 ─────────────────────────────────────────
    figure(doc, os.path.join(PAPER, "fig1_acc_energy.png"), 145,
           f"그림 1. 판단 1회 추정 에너지(가로, 로그축)와 정확도. (a) 시험 데이터 macro-F1, (b) 외부 데이터 F1. 색은 전처리 위치, "
           f"마커 위치는 데이터시트 전력 기준, 가로 막대는 세 전력 출처의 범위, 세로 막대는 시드 표준편차(n=3). "
           f"{n_a}(*) 는 보드 미실측으로 지연을 합성 사이클로 추정한 점. 주황 점선은 NPU 전처리 계열(NPU 전처리 (Gabor 필터뱅크)와 대역·프레임 수를 늘린 변형 3종, 모두 보드 실측)을 에너지 순으로 이은 것. "
           f"큰 고리와 회색 점선은 파레토 앞면(에너지가 같거나 작으면서 정확도가 같거나 높은 다른 점이 없는 구성).")
    rows = []
    VARS = [c["name"] for c in CFG["configs"] if c.get("variant_of") == FE]       # ②′ 변형 3종 (보드 실측, 2026-10-19 오후 지시)
    for key in (W, FE, *VARS, A, GA, DA, H25, H16):
        rows.append([DN[key], f"{latency(key):.2f}" + (" (추정)" if key == A else ""),
                     f"{energy(key):.0f} / {energy(key, 'literature'):.0f} / {energy(key, 'moosmann'):.0f}",
                     fmt_pm(key), fmt_pm(key, "ext_f1")])
    lm4 = v("latency.M4 CMSIS-NN")["total"]
    rows.append([CFG["display"]["m4"]["name"] + " (원본 파형 가중치)", f"{lm4:.2f}",
                 f"{energy(M4):.0f} / {energy(M4, 'literature'):.0f} / {energy(M4, 'moosmann'):.0f}", "= 원본 파형", "= 원본 파형"])
    table(doc, ["구성", "판단 1회 지연 (ms)", "에너지 μJ, 추정 (데이터시트 / 선행연구 / Moosmann)", "macro-F1 시험 데이터", "F1 외부 데이터"],
          rows, f"표 1. 주 결과. 정확도는 시드 1~3 평균 ± 표준편차, 오경보 300회/h. 지연은 보드 1,000회 최악값, 에너지는 추정(실측 지연 × 공개 전력). "
                f"변형 3종은 펌웨어 안정화 대기(NPU 패스당 약 {v('firmware.stop_wait')['us_per_pass']:.0f} μs) 포함. "
                f"같은 대기를 넣으면 {DN[FE]} 는 {v('firmware.stop_wait')['fe_total_with_wait_ms']:.2f} ms. "
                f"NPU 전처리 계열의 정수 로그는 32비트 나눗셈 구현(이전 64비트 구현과 출력 비트 일치)이며, 지연은 교체분을 보드에서 잰 차이로 반영했다."
                + (f" 같은 펌웨어끼리 비교하면 데이터시트 기준 {DN[FE]} {N['firmware.same_fw_fe_vs_ga']['value']['datasheet']['fe']:.0f} μJ > {DN[GA]} "
                   f"{N['firmware.same_fw_fe_vs_ga']['value']['datasheet']['ga']:.0f} μJ."
                   if "firmware.same_fw_fe_vs_ga" in N and N["firmware.same_fw_fe_vs_ga"]["value"]["datasheet"]["fe_gt_ga"] else ""),
          col_mm=[68, 22, 38, 31, 31])   # 2026-10-21: 구성 이름이 길어져 열 폭 재배분 (합 190 mm 그대로)

    # ── 2단: 서론·관련 연구·방법 ──────────────────────────────────────
    new_section(doc, 2)
    heading(doc, "1. 서  론")
    body(doc, "소형 MCU 에 신경망 가속기(NPU)를 얹은 칩이 늘고 있다. MAX78000 은 Cortex-M4F 와 CNN 가속기를 "
              "한 칩에 두어 상시 동작 음향 감지에 적합하지만, 가속기는 합성곱·풀링·요소별 연산만 지원하고 "
              "FFT·로그 같은 전처리 연산은 지원하지 않는다 [1, 2]. 따라서 log-mel 전처리는 CPU 가 맡거나, 아예 없애고 "
              "원본 파형을 가속기에 직접 넣어야 한다. 벤더의 음성 키워드 참조 설계는 후자를 택하며, MFCC 를 쓰는 "
              "소형 모델들과 비슷하거나 높은 정확도(96.3% 대 90.2~96.4%)를 보고했다 [4].")
    body(doc, "본 연구는 이 선택이 환경음(사이렌·유리 파손·비명·개 짖음) 감지에서도 성립하는지를 같은 데이터·같은 "
              "학습 설정·같은 보드에서 실측으로 답한다. 기여는 셋이다. (1) 전처리 위치를 양 끝(전부 CPU, 전부 NPU)과 "
              f"그 사이의 {len(CFG['configs'])}개 구성(과 CPU 추론 대조군)으로 나누어 정확도와 판단 1회 지연을 보드에서 재고 에너지를 추정한 정확도–에너지 "
              "곡선. (2) 필터뱅크·로그 압축·분할 등 전처리를 NPU 연산으로 옮긴 변형이 원본 파형 구성을 유의하게 넘지 "
              "못한다는 부정 결과와, 선형 부분만 옮긴 구성은 원본 파형 구성보다 에너지가 적지만 정확도 이득이 없어, CPU 전처리의 "
              "저에너지 구성과 비슷한 에너지에서 앞면 끝에 머문다는 실측. (3) 거친 log-mel(프레임 간격 62.5 ms, 멜 32)을 CPU 에 "
              "남기는 구성이 원본 파형 구성보다 정확도가 유의하게 높으면서 추정 에너지는 같거나 적다는 관찰과, 전처리가 NPU 가 처리할 데이터량을 줄이는 단계라는 해석.")

    heading(doc, "2. 관련 연구")
    body(doc, "MAX78000 의 자원·전력 특성은 Moss 등 [3] 이, 키워드 인식의 초저전력 구현은 Ulkar 와 Okman [4] 이, "
              "물체 검출은 Moosmann 등 [5] 이 보고했다. 이들은 전처리를 고정한 채 모델을 다룬다. 원본 파형에서 직접 "
              "배우는 프론트엔드로는 SincNet [6] 과 LEAF [7] 가 있으나 로그·PCEN 같은 비선형 압축을 포함하므로 "
              "MAX78000 가속기에 그대로 올릴 수 없다. CPU 추론 대조군은 CMSIS-NN [8] 으로 구현했다. 데이터는 "
              "FSD50K [9]·UrbanSound8K [10]·ESC-50 [11] 과 AudioSet [12] 을 썼고, 대형 모델 기준선과 증류 교사로 "
              "PANNs [13] 를 썼다.")

    heading(doc, "3. 시스템과 실험 설계")
    body(doc, "시스템. MAX78000FTHR 단독으로 온보드 마이크 16 kHz 입력을 1.024 s 창(16,384 샘플, int8)으로 받아 "
              "250 ms 마다(가속기 전처리 구성은 프레임 격자에 맞춰 256 ms 마다) 5클래스(사이렌·유리 파손·비명·개 짖음·배경)를 판단한다. 모델은 가중치 8 bit QAT, "
              "가속기 가중치 메모리 442 KB 안에 들어간다.")
    body(doc, f"데이터와 정답. 세 공개 데이터셋은 모두 Freesound 파생이라 원본 ID 단위로 전역 분할해 누수를 막았고, "
              f"AudioSet 의 실환경 녹음 구간을 학습에 더했다. 시험 데이터의 유리·비명 창은 저자가 전수 청취해 양성을 "
              f"확정했다(채점 창 {tr:,}). 외부 평가는 AudioSet 의 별도 영상에서 뽑은 {n_ext}창(양성 "
              f"{sum(ext_pos.values())}, 음성 {ext_neg}) 으로, 학습에 쓰지 않았다. 오경보율은 시험 데이터 배경에서 "
              f"300회/h 가 되는 문턱값을 잡아 외부에도 같은 값을 썼고, 지표는 macro-F1 이다.")
    body(doc, f"구성. {n_w}: 원본 파형을 128×128 로 접어 1D CNN 에 넣는 벤더 방식. {n_a}: 로그 압축을 NPU 안의 "
              f"구간선형 층으로 근사한 1D 구성. {DN[FE]}: 멜 대역폭에 맞춘 512탭 Gabor 필터(cos·sin 32쌍)를 가속기의 "
              f"Conv1d 로 두고 프레임 안 에너지 적분(|cos|+|sin| 의 8위치 합)과 정수 로그만 CPU 가 맡은 뒤, 그 32×16 특징을 "
              f"아래 62.5 ms 구성과 같은 2D CNN 에 넣는 구성 — 두 네트워크를 가속기에 동시 적재하고 판단마다 전환한다(판단 주기 256 ms). "
              f"그 변형 3종은 Gabor 필터를 64쌍으로 늘려 64 대역을 얻은 {DN['②′ b64']}, 적분 위치 간격을 반으로 좁혀 프레임 간격 64 ms 를 32 ms 로 "
              f"하고 프레임 수를 16 에서 31 로 늘린 {DN['②′ f31']}, 둘 다 적용한 {DN['②′ b64f31']} 이다. "
              f"CPU log-mel 네 구성: 프레임 간격 62.5 ms(멜 32)·50 ms(멜 32)·"
              f"25 ms(멜 64)·16 ms(멜 64) 의 증분 log-mel을 CPU 가 계산하고 2D CNN 을 NPU 에 올린다. "
              f"모든 구성의 전처리는 판단마다 새 프레임만 계산하는 증분 방식이며, 일괄 계산과 비트 일치함을 보드에서 확인했다. "
              f"CPU 추론: {n_w} 의 가중치를 그대로 CMSIS-NN 으로 Cortex-M4F 에서 돌린 대조군(NPU 출력과 비트 일치). "
              f"모든 구성은 같은 데이터로 150 에폭, 같은 스케줄·증강·교사 증류(PANNs 임베딩 위 선형 프로브)로 "
              f"시드 3개씩 학습했다.")
    body(doc, f"지연과 에너지. 지연은 보드에서 SysTick(코어 클럭 사이클, 분해능 10 ns)으로 1,000회 재어 최악값을 보고한다(전원 투입 직후, "
              f"코어 100 MHz, CNN 50 MHz; CPU 추론만 SysTick 의 24 bit 한계(168 ms)를 넘어 하드웨어 타이머 TMR0, 1 μs 분해능). 에너지는 외부 계측 장비 없이 "
              f"E = t전처리·Pcpu + (t적재 + t출력)·Pload + t추론·Pnpu 로 추정했으며, 전력값은 데이터시트 [1] 와 동일 칩 "
              f"실측 선행연구 두 벌(Moss·Ulkar & Okman [3, 4], Moosmann [5])의 세 출처를 나란히 써 범위로 보고한다. 측정값이 아님을 모든 수치에 밝힌다.")

    # ── 결과·결론·참고문헌 (2단 계속) ────────────────────────────────
    # ②′ (NPU 전처리: Gabor 필터뱅크 + CPU 정수 로그) — 보드 실측 뒤 정식 점. 사용자 지시 (2026-10-17) 4·5항:
    #   실측 에너지가 원본 파형보다 전 출처에서 작으면 "저에너지 끝" 문장, 아니면 부정 결과 한 줄만. 수치는 전부 numbers.json.
    fe_w, fe_g = v("welch.②′ − ④")["test_f1_S3"], v("welch.②′ − 가 u1000")["test_f1_S3"]
    fe_cheaper = all(energy(FE, k) < energy(W, k) for k in ("datasheet", "literature", "moosmann"))
    fci = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
    # 2026-10-20 지시: "②′ 가 가 보다 에너지가 적다" 는 주장으로 쓰지 않는다. 원본 파형보다 세 출처 모두 적다는 것만 유지, 손익분기 문장은 본문에서 뺀다
    #   (docs/paper/table5_breakeven.csv 에 남는다). 비슷한 에너지(데이터시트 차이 수 µJ = 측정 펌웨어 차이 수준)에서 정확도 0.11 낮다 로 쓴다.
    d_ga = abs(energy(GA) - energy(FE))
    SP = N.get("firmware.cpu_split", {}).get("value")        # 9절 실험 II (2026-10-21): CPU 몫 분해 실측 — 비율만 쓴다 (②′ 주 값 불변)
    if SP and not SP["valid"]:
        sys.exit(f"[에러] CPU 분해 (a)+(b) 가 CPU 구간과 2 % 넘게 다르다: {SP}")
    cpu_split_sent = (f"CPU 몫 {fe_cpu_ms:.2f} ms 중 {SP['read_pct']}%는 가속기 메모리에서 int32 출력 2,048개를 읽어 절댓값과 8위치 합산까지 하는 "
                      f"시간, {SP['log_pct']}%는 정수 로그였다(보드 실측). " if SP else
                      f"CPU 몫 {fe_cpu_ms:.2f} ms 의 대부분은 로그가 아니라 가속기 메모리에서 int32 출력 2,048 개를 읽어 오는 시간이었다. ")
    if fe_cheaper:
        fe_neg = (f"선형 부분만 옮기면 에너지는 줄지만 정확도는 유의한 차이가 없다. {DN[FE]} 는 사전에 정한 목표({s_w} 초과 또는 "
                  f"{s_ga}과 동등)는 넘지 못했다({s_w} 대비 {fci(fe_w)}, {s_ga} 대비 {fci(fe_g)}). 추정 에너지는 {s_w}보다 세 전력 출처 모두에서 "
                  f"적지만(데이터시트 {energy(FE):.0f} 대 {energy(W):.0f} μJ, Moosmann {energy(FE, 'moosmann'):.0f} 대 {energy(W, 'moosmann'):.0f} μJ; "
                  f"판단 1회 {latency(FE):.2f} ms 대 {latency(W):.2f} ms), {s_ga}과는 비슷한 에너지(데이터시트 기준 {energy(FE):.0f} 대 {energy(GA):.0f} μJ)"
                  f"에서 시험 macro-F1 이 {abs(fe_g[0]):.2f} 낮다. "
                  f"두 네트워크를 동시 적재한 보드에서 가속기 출력·CPU 특징·최종 로짓이 전부 호스트와 비트 일치했고, " + cpu_split_sent)
    else:
        fe_neg = (f"선형 부분만 NPU 로 옮기고 정수 로그만 CPU 에 남긴 {DN[FE]} 도 {s_w} 대비 {fci(fe_w)}, {s_ga} 대비 {fci(fe_g)} 로 열세였고 "
                  f"에너지도 {s_w} 보다 적지 않았다({energy(FE):.0f} 대 {energy(W):.0f} μJ). ")
    heading(doc, "4. 결  과")
    body(doc, f"전처리 제거의 비용. {s_w}은 {s_16} 구성보다 시험 macro-F1 이 {f1(H16)[0]-f1(W)[0]:.3f}, 외부 F1 이 "
              f"{f1(H16,'ext_f1')[0]-f1(W,'ext_f1')[0]:.3f} 낮다(표 1). 음성 키워드에서 손실이 없던 선택이 환경음에서는 "
              f"큰 비용을 치른다. 같은 가중치를 CPU 로 돌린 CMSIS-NN 대조군과 비교하면 NPU 추론은 추론만 "
              f"{r_inf:.0f}배, 판단 1회 {r_dec:.0f}배 빠르고 CPU 추론은 판단 주기의 {hop_m4:.0f}% 를 추론에 쓴다(그림 2).")
    body(doc, f"NPU 로 옮긴 전처리는 되찾지 못한다. {n_a} 는 {s_w} 대비 {welch('(a) − ④')} 로 CI 가 0 을 포함하고, "
              f"로그 근사 층 때문에 NPU 사이클이 3.8배 늘어 추정 에너지는 {energy(A)/energy(W):.1f}배다. 이전 데이터 버전에서 시험한 NPU 전처리 변형들도 "
              f"{s_w} 대비 {V1_LO:+.3f}~{V1_HI:+.3f} 으로, 신뢰구간이 0 을 넘는 개선은 없었다. 지원하지 않는 비선형을 "
              f"합성곱으로 흉내 내는 순간 NPU 는 CPU 보다 비싸진다.")
    body(doc, fe_neg)
    # ②′ 곡선 변형 3종 (chain33, fe-curve-design.md 7·8절) — 보드 실측 (2026-10-19 오후 지시). 수치는 전부 numbers.json.
    #   지시문: "대역·프레임 수를 늘린 NPU 전처리 변형은 정확도가 0.655~0.690으로 올랐으나 에너지가 더 빠르게 늘어 CPU 전처리 곡선
    #   아래에 머물렀다. 주된 비용은 NPU 연산이 아니라 CPU가 가속기 메모리에서 결과를 읽어 오는 시간이었다." — 실측이 다르면 결과대로.
    #   "64 대역이 원본 파형을 넘었다" 는 사후 관찰은 넣지 않는다 (5항).
    vf, vm = v("variants.②′.f1_S3"), v("variants.②′.measured")
    srcs3 = ("datasheet", "literature", "moosmann")
    below_all = all(vm[n]["dominated_by_cpu_curve"][k] for n in vm for k in srcs3)          # 세 출처 모두에서 CPU 곡선의 어떤 점에 지배
    below_ds = all(vm[n]["dominated_by_cpu_curve"]["datasheet"] for n in vm)
    cpu_main = all((vm[n]["cpu_share_of_delta"] or 0.0) > 0.5 for n in vm)                 # ②′ 대비 증가분의 절반 넘게가 CPU 구간
    e_lo, e_hi = min(vm[n]["E_uJ"]["datasheet"] for n in vm), max(vm[n]["E_uJ"]["datasheet"] for n in vm)
    var_sent = (f"대역·프레임 수를 늘린 NPU 전처리 변형 3종은 시험 macro-F1 이 {vf['min']:.3f}~{vf['max']:.3f} 으로 올랐으나"
                f"(표 1) 추정 에너지가 {energy(FE):.0f} 에서 {e_lo:.0f}~{e_hi:.0f} μJ 로 더 빠르게 늘어 ")
    if below_all:
        var_sent += "세 전력 출처 모두에서 CPU 전처리 곡선 아래에 머물렀다(그림 1). "
    elif below_ds:
        flip = [n for n in vm for k in srcs3 if not vm[n]["dominated_by_cpu_curve"][k]]
        var_sent += f"데이터시트 전력 기준으로는 CPU 전처리 곡선 아래에 머물렀으나 다른 출처에서는 일부({', '.join(DN[n] for n in dict.fromkeys(flip))})가 지배되지 않았다(그림 1). "
    else:
        var_sent += "CPU 전처리 곡선 아래에 머물지 않은 점이 있었다(그림 1). "
    cs = [100 * (vm[n]["cpu_share_of_delta"] or 0) for n in vm]; ns = [100 * (vm[n]["npu_share_of_delta"] or 0) for n in vm]
    # 비율(읽기/로그)은 위 ②′ 문단에 한 번만 쓴다 (2026-10-21 지시: 중복 제거)
    # 2026-10-21 (32비트 정수 로그 반영 뒤) 지시: CPU 몫이 세 변형 모두 50 % 를 넘지 않으므로 '비슷하게 나눴다' 로 쓴다.
    if cpu_main:
        var_sent += f"늘어난 비용의 절반 넘게({min(cs):.0f}~{max(cs):.0f}%)가 CPU 의 가속기 메모리 읽기였고 NPU 연산은 {min(ns):.0f}~{max(ns):.0f}% 였다."
    else:
        var_sent += f"늘어난 비용은 CPU의 가속기 메모리 읽기({min(cs):.0f}~{max(cs):.0f}%)와 NPU 연산({min(ns):.0f}~{max(ns):.0f}%)이 비슷하게 나눴다."
    body(doc, var_sent)
    body(doc, f"최적점은 중간에 있다. {s_ga}은 {s_w}보다 시험 macro-F1 {welch('가 u1000 − ④')}, 외부 F1 "
              f"{welch('가 u1000 − ④', 'ext_f1')} 로 유의하게 높고, 판단 1회 지연은 {latency(GA):.2f} ms 대 {latency(W):.2f} ms 로 "
              f"비슷하며, 추정 에너지는 데이터시트 기준 {energy(GA):.0f} 대 {energy(W):.0f} μJ 로 적고 Moosmann 기준 "
              f"{energy(GA,'moosmann'):.0f} 대 {energy(W,'moosmann'):.0f} μJ 로 같다. 이유는 그림 2 에 있다. log-mel이 16,384 샘플을 "
              f"512 값으로 줄이므로 NPU 추론이 {v('latency.④')['infer']:.2f} ms 에서 {lat_ga['infer']:.2f} ms 로 줄고, "
              f"CPU 전처리 {lat_ga['pre']:.2f} ms 가 그 차이보다 작다. 전처리는 병목이 아니라 데이터 축소 단계였고, "
              f"NPU 가 원시 샘플 전량을 처리하는 순간 그 이득이 사라진다. 프레임 간격을 좁힐수록 정확도는 오르지만 "
              f"({s_25} {fmt_pm(H25)}, {s_16} {fmt_pm(H16)}) 전처리가 지연의 2/3 를 차지해 에너지가 {energy(H16)/energy(GA):.0f}배가 된다. "
              f"{s_25} 와 {s_16} 의 차이는 {welch('간격 400 − ①′')} 로 작지만 0 을 포함하지 않았다. "
              + (f"그림 1 의 파레토 앞면은 {' → '.join(pf)} 이다. 다만 {DN[FE]} 와 {s_ga}의 에너지 차이(데이터시트 기준 약 "
                 f"{abs(energy(GA) - energy(FE)):.0f} μJ)는 작아, 두 구성의 차이는 사실상 정확도에서 난다. {s_w}과 {n_a} 는 모두 다른 점에 지배된다."
                 if FE in v("pareto.front")["test_S3"] else
                 f"그림 1 의 파레토 앞면은 {' → '.join(pf)} 이고, {s_w}과 {n_a} 는 모두 다른 점에 지배된다."))
    body(doc, f"대형 모델 기준선. 같은 정답·같은 문턱값 규칙으로 PANNs(CNN14) 를 채점하면 시험 macro-F1 {pann:.3f} 이다. "
              f"{s_16} 구성은 그 {100*f1(H16)[0]/pann:.0f}% 에 이른다.")
    # 4절 끝 (2026-10-20 지시): 비용을 결정한 것은 데이터 이동. 마지막 문장은 추측 — '할 수 있다' 를 넘지 않는다.
    body(doc, f"log 연산을 NPU에서 수행할 수 없어 중간 결과를 int32로 CPU에 읽어 와야 하는 데이터 이동이 NPU 전처리 CPU 몫의 대부분({SP['read_pct']}%)을 차지했고, "
              "대역·프레임을 늘리면 NPU 연산도 비슷한 비중으로 늘었다. 가속기 내부에서 비선형 압축을 지원하면 이 경계가 달라질 수 있다. "
              # 3) 증분 없이 창 전체를 매번 계산했을 때 (추정, docs/results/incremental-design.md 4절)
              + "증분 계산 없이 창 전체를 매번 계산했다면 " + f"{s_ga}과 {DN[FE]} 모두 {s_w}보다 에너지가 컸을 것으로 추정되어"
              + "(일괄 전처리 사이클 = 새 프레임 비용 × 전체 프레임 수, 같은 전력 상수), 전처리의 이점은 증분 계산을 전제로 한다.")
    figure(doc, os.path.join(PAPER, "fig2_latency_breakdown.png"), 90,
           f"그림 2. 판단 1회 지연 분해(보드 실측, 1,000회 최악값). 왼쪽은 NPU 추론 {len([c for c in CFG['configs'] if not c.get('variant_of')])}개 구성"
           f"(CPU 전처리 / 입력 적재 / NPU 추론; {DN[FE]} 의 변형 3종은 표 1 에만), "
           "오른쪽은 같은 가중치의 CPU 추론(Cortex-M4F, CMSIS-NN)과 판단 주기 250 ms.")

    heading(doc, "5. 한계와 결론")
    body(doc, "한계. 에너지는 직접 계측하지 않은 추정값이며 전력 출처에 따라 결론의 강도가 다르다(CPU log-mel 62.5 ms 구성의 우위는 "
              "'적다' 와 '같다' 사이). 시드가 3개라 신뢰구간이 넓다. 정답 청취는 저자 1인이다. 비명과 환호의 경계는 "
              "음향만으로 가를 수 없어 비명 재현율은 그 모호성을 포함한다. 시험 데이터 배경의 절반이 하드 네거티브라 "
              f"오경보율은 실배치보다 부풀려져 있다. {DN[FE]} 의 CPU 구간은 가속기 전원이 켜진 채 돌므로 적재 전력으로 "
              "보수적으로 계산했고, 가속기 메모리 읽기를 DMA 로 바꾸면 더 줄 수 있다. 음량 정규화·필터 해상도·용량 확대 등 원본 파형 쪽 변형을 더 시도할 "
              f"여지는 남아 있다. 향후 과제로, CPU 몫의 {SP['read_pct']}%를 차지하는 읽기와 위치 합산을 NPU 에서 수행하려면 8비트 출력에 맞춘 재학습이 필요하다.")
    # 평균 전력 조건형 문단 (2026-10-21 지시, f83c152 이후): 수치는 numbers.json avg_power (데이터시트 p.10 대기 전류 × 실측 지연 에너지) 에서 읽는다.
    AP = v("avg_power"); apc = AP["configs"]
    body(doc, f"평균 전력. 판단 1회 에너지를 평균 전력으로 옮기면(데이터시트 기반 추정), 디지털 마이크를 유지하는 대기 전력이 대부분을 차지한다. "
              f"I2S 를 살린 가장 깊은 대기 모드의 고정 전류(LPM 0.64 mA [1])로 구한 대기 전력 하한 {AP['P_wait_lo_mW']:.1f} mW 이상이면 "
              f"{s_ga}의 판단 비용은 평균 전력의 {apc[GA]['share_max_pct']:.0f}% 이하, {s_16} 구성은 {apc[H16]['share_max_pct']:.0f}% 이하다"
              f"(세 전력 출처 중 가장 큰 에너지 기준). 반면 NPU 없이 CPU 로 추론하면 현 구성의 대기 전력 상한 {AP['P_wait_hi_mW']:.0f} mW 에서도 "
              f"판단 비용이 평균 전력의 {apc[M4]['share_min_pct']:.0f}% 이상을 차지한다. 즉 NPU 는 거친 log-mel 구성에서 판단 비용을 평균 전력의 작은 몫으로 "
              f"낮추고, 전처리 구성 간 에너지 차이는 대기 전력을 줄일수록 평균 전력에 더 크게 드러난다. "
              f"RISC-V 코어를 상시 게이트로 두어 이벤트 후보에서만 판단하는 설계의 이득 상한은 현 대기 모드에서 판단 비용 전부인 "
              f"약 {apc[GA]['share_datasheet_at_hi_pct']:.0f}% 로 추정되고, RISC-V 로 전처리를 옮기는 안도 FPU 없는 60 MHz 코어의 전력이 Cortex-M4F 의 "
              f"{RISCV['p_rv_over_m4'][0]:.1f}~{RISCV['p_rv_over_m4'][1]:.1f}배로 추정되어 손익분기 전처리 지연 {RISCV['breakeven_ms'][0]:.1f}~{RISCV['breakeven_ms'][1]:.1f} ms 를 "
              f"지키기 어려워 제외했다.")
    body(doc, "결론. NPU 탑재 MCU 에서 '전처리를 전부 가속기로' 는 환경음 감지에서 최적이 아니다. 지원 연산이 "
              "제한된 가속기에 비선형 전처리를 옮기는 시도는 정확도도 에너지도 얻지 못했고" +
              (", 선형 부분만 옮긴 구성은 원본 파형 구성보다 에너지를 줄였지만 CPU 전처리의 저에너지 구성과 비슷한 에너지에서 정확도를 잃어 앞면 끝에 머물렀으며" if fe_cheaper else "") +
              ", 작은 CPU 전처리로 "
              "데이터량을 줄인 뒤 작은 모델을 가속기에 올리는 구성이 정확도–에너지 전선 위에 있었다. 이 칩에서는 어떤 연산을 "
              "가속기에 두느냐보다, 가속기에 들어가는 데이터율이 구성의 우열을 더 잘 설명했다.")

    heading(doc, "참고문헌", size=9)
    refs = [
        "Maxim Integrated, \"MAX78000: Artificial Intelligence Microcontroller with Ultra-Low-Power Convolutional Neural Network Accelerator,\" Data Sheet 19-100868, Rev. 1, May 2021.",
        "Analog Devices, \"MAX78000 User Guide,\" UG7456, Rev. 1, Mar. 2024.",
        "A. Moss, H. Lee, L. Xun, C. Min, F. Kawsar, A. Montanari, \"Ultra-low power DNN accelerators for IoT: resource characterization of the MAX78000,\" Proc. SenSys Workshop AIChallengeIoT, 2022.",
        "M. G. Ulkar, O. E. Okman, \"Ultra-low power keyword spotting at the edge,\" arXiv:2111.04988, 2021.",
        "J. Moosmann, M. Giordano, C. Vogt, M. Magno, \"TinyissimoYOLO: a quantized, low-memory and ultralow-power object detection network for microcontrollers,\" Proc. IEEE AICAS, 2023.",
        "M. Ravanelli, Y. Bengio, \"Speaker recognition from raw waveform with SincNet,\" Proc. IEEE SLT, pp. 1021–1028, 2018.",
        "N. Zeghidour, O. Teboul, F. de Chaumont Quitry, M. Tagliasacchi, \"LEAF: a learnable frontend for audio classification,\" Proc. ICLR, 2021.",
        "L. Lai, N. Suda, V. Chandra, \"CMSIS-NN: efficient neural network kernels for Arm Cortex-M CPUs,\" arXiv:1801.06601, 2018.",
        "E. Fonseca, X. Favory, J. Pons, F. Font, X. Serra, \"FSD50K: an open dataset of human-labeled sound events,\" IEEE/ACM TASLP, vol. 30, pp. 829–852, 2022.",
        "J. Salamon, C. Jacoby, J. P. Bello, \"A dataset and taxonomy for urban sound research,\" Proc. ACM Multimedia, pp. 1041–1044, 2014.",
        "K. J. Piczak, \"ESC: dataset for environmental sound classification,\" Proc. ACM Multimedia, pp. 1015–1018, 2015.",
        "J. F. Gemmeke et al., \"Audio Set: an ontology and human-labeled dataset for audio events,\" Proc. ICASSP, pp. 776–780, 2017.",
        "Q. Kong, Y. Cao, T. Iqbal, Y. Wang, W. Wang, M. D. Plumbley, \"PANNs: large-scale pretrained audio neural networks for audio pattern recognition,\" IEEE/ACM TASLP, vol. 28, pp. 2880–2894, 2020.",
    ]
    for i, r in enumerate(refs, 1):
        para(doc, f"[{i}] {r}", size=7.5, align="just", space_after=0, line=1.0)

    out = os.path.join(OUT_DIR, "ksc2026_review.docx")
    doc.save(out)
    print("저장:", out)
    return out


def to_pdf(docx_path):
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        print("soffice 없음 — PDF 변환 생략 (Windows 의 Word 로 PDF 저장)")
        return None
    r = subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", OUT_DIR, docx_path],
                       capture_output=True, text=True, timeout=300)
    pdf = docx_path[:-5] + ".pdf"
    if not os.path.isfile(pdf):
        print("PDF 변환 실패:", r.stdout[-500:], r.stderr[-500:])
        return None
    try:
        from pypdf import PdfReader                  # noqa: PLC0415
        n = len(PdfReader(pdf).pages)
    except Exception:                                # noqa: BLE001
        n = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout
        n = next((int(l.split()[-1]) for l in n.splitlines() if l.startswith("Pages")), "?")
    print(f"PDF: {pdf} — {n} 쪽 (허용 2~3)")
    return pdf


if __name__ == "__main__":
    p = build()
    to_pdf(p)
    sys.exit(0)
