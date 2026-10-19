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
V1_NEG = [("멜 필터뱅크 초기화 Conv1d + Abs (D-1)", "-0.026 [-0.057, +0.006]"),
          ("D-1 탭 256 / 512 (k=2 / k=4)", "-0.050 / -0.032"),
          ("NPU 구간선형 로그 ((a), v1)", "+0.003 [-0.028, +0.034]"),
          ("NPU 필터뱅크 + CPU 정수 로그 분할 ((c))", "-0.012 [-0.042, +0.018]"),
          ("원파형 2D 재배열 (wave2D)", "-0.019")]


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


def para(doc, text="", size=BODY_PT, bold=False, align=None, space_after=2, indent=False, line=1.15, italic=False):
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
    brk_fe_ga = v("breakeven")[f"{DN[FE]} < {DN[GA]}"]["r_star"]
    fe_cpu_ms = v("latency.②′")["detail"]["cpu"]
    para(doc,
         f"NPU 를 탑재한 MCU(MAX78000) 에서 배터리 상시 구동 음향 이벤트 감지기를 구현하고, 로그 멜 전처리를 "
         f"어디에 둘지가 정확도와 판단당 에너지에 미치는 영향을 {len(CFG['configs'])}개 구성으로 같은 데이터·같은 보드에서 실측했다. "
         f"벤더 참조 설계처럼 원파형을 NPU 에 직접 넣어 전처리를 없앤 구성은 CPU 로그 멜 구성보다 시험셋 macro-F1 이 {f1(H16)[0]-f1(W)[0]:.3f} 낮았고, "
         f"로그 압축을 NPU 연산으로 근사한 변형은 이 격차를 되찾지 못했으며, 선형 부분(필터뱅크와 에너지 적분)만 NPU 로 옮긴 구성은 "
         f"에너지를 가장 적게 쓰지만 정확도는 원파형과 같았다. 반면 프레임 간격을 넓힌 거친 로그 멜을 CPU 에 남기고 작은 2D CNN 을 "
         f"NPU 에 올린 구성은 원파형 구성보다 macro-F1 이 {v('welch.가 u1000 − ④')['test_f1_S3'][0]:+.3f} 높으면서(Welch 95% CI 가 0 을 "
         f"포함하지 않음) 추정 에너지는 같거나 적었다. NPU 추론은 같은 가중치의 CMSIS-NN 추론보다 판단 1회 기준 {r_dec:.0f}배 빨랐다. "
         f"전처리 위치는 가속기 친화성의 문제가 아니라 정확도–에너지 설계 변수이며, 파레토 앞면은 양 끝이 아니라 전처리를 나눈 "
         f"중간 구성들로 이루어짐을 보인다.",
         align="just", space_after=6, line=1.15)

    # ── 1단 띠: 그림 1 + 표 1 ─────────────────────────────────────────
    figure(doc, os.path.join(PAPER, "fig1_acc_energy.png"), 150,
           f"그림 1. 판단 1회 추정 에너지(가로, 로그축)와 정확도. (a) 시험 데이터 macro-F1, (b) 외부 데이터 F1. 색은 전처리 위치, "
           f"마커 위치는 데이터시트 전력 기준, 가로 막대는 공개 전력값 범위, 세로 막대는 시드 표준편차(n=3). "
           f"{n_a}(*) 는 보드 미실측으로 지연을 합성 사이클로 추정한 점. 주황 점선은 NPU 전처리 계열(Gabor 필터뱅크와 대역·프레임 수를 늘린 변형 3종, 모두 보드 실측)을 에너지 순으로 이은 것. "
           f"큰 고리와 회색 점선은 파레토 앞면(에너지가 같거나 작으면서 정확도가 같거나 높은 다른 점이 없는 구성).")
    rows = []
    VARS = [c["name"] for c in CFG["configs"] if c.get("variant_of") == FE]       # ②′ 변형 3종 (보드 실측, 2026-10-19 오후 지시)
    for key in (W, FE, *VARS, A, GA, DA, H25, H16):
        rows.append([DN[key], f"{latency(key):.2f}" + (" (추정)" if key == A else ""),
                     f"{energy(key):.0f} / {energy(key, 'moosmann'):.0f}",
                     fmt_pm(key), fmt_pm(key, "ext_f1")])
    lm4 = v("latency.M4 CMSIS-NN")["total"]
    rows.append([CFG["display"]["m4"]["name"] + " (원파형 가중치)", f"{lm4:.2f}",
                 f"{energy(M4):.0f} / {energy(M4, 'moosmann'):.0f}", "= 원본 파형", "= 원본 파형"])
    table(doc, ["구성", "판단 1회 지연 (ms)", "에너지 μJ, 추정 (데이터시트 / Moosmann)", "macro-F1 시험 데이터", "F1 외부 데이터"],
          rows, f"표 1. 주 결과. 정확도는 시드 1~3 평균 ± 표준편차, 오경보 300회/h. 지연은 보드 1,000회 최악값, 에너지는 추정(실측 지연 × 공개 전력).",
          col_mm=[60, 24, 40, 33, 33])

    # ── 2단: 서론·관련 연구·방법 ──────────────────────────────────────
    new_section(doc, 2)
    heading(doc, "1. 서  론")
    body(doc, "소형 MCU 에 신경망 가속기(NPU)를 얹은 칩이 늘고 있다. MAX78000 은 Cortex-M4F 와 CNN 가속기를 "
              "한 칩에 두어 상시 동작 음향 감지에 적합하지만, 가속기는 합성곱·풀링·요소별 연산만 지원하고 "
              "FFT·로그 같은 전처리 연산은 지원하지 않는다 [1, 2]. 따라서 로그 멜 전처리는 CPU 가 맡거나, 아예 없애고 "
              "원파형을 가속기에 직접 넣어야 한다. 벤더의 음성 키워드 참조 설계는 후자를 택하며 음성에서는 정확도 "
              "손실이 거의 없다고 알려져 있다 [4].")
    body(doc, "본 연구는 이 선택이 환경음(사이렌·유리 파손·비명·개 짖음) 감지에서도 성립하는지를 같은 데이터·같은 "
              "학습 설정·같은 보드에서 실측으로 답한다. 기여는 셋이다. (1) 전처리 위치를 양 끝(전부 CPU, 전부 NPU)과 "
              f"그 사이의 {len(CFG['configs'])}개 구성으로 나누어 정확도와 판단 1회 지연을 보드에서 재고 에너지를 추정한 정확도–에너지 "
              "곡선. (2) 필터뱅크·로그 압축·분할 등 전처리를 NPU 연산으로 옮긴 변형이 원파형 구성을 유의하게 넘지 "
              "못한다는 부정 결과와, 선형 부분만 옮긴 구성이 정확도 이득 없이 에너지만 줄여 곡선의 저에너지 끝이 된다는 "
              "실측. (3) 거친 로그 멜(프레임 간격 62.5 ms, 멜 32)을 CPU 에 남기는 구성이 원파형 구성을 "
              "정확도와 에너지 양쪽에서 이긴다는 관찰과, 전처리가 NPU 가 처리할 데이터량을 줄이는 단계라는 해석.")

    heading(doc, "2. 관련 연구")
    body(doc, "MAX78000 의 자원·전력 특성은 Moss 등 [3] 이, 키워드 인식의 초저전력 구현은 Ulkar 와 Okman [4] 이, "
              "물체 검출은 Moosmann 등 [5] 이 보고했다. 이들은 전처리를 고정한 채 모델을 다룬다. 원파형에서 직접 "
              "배우는 프론트엔드로는 SincNet [6] 과 LEAF [7] 가 있으나 로그·PCEN 같은 비선형 압축을 포함하므로 "
              "MAX78000 가속기에 그대로 올릴 수 없다. CPU 추론 대조군은 CMSIS-NN [8] 으로 구현했다. 데이터는 "
              "FSD50K [9]·UrbanSound8K [10]·ESC-50 [11] 과 AudioSet [12] 을 썼고, 대형 모델 기준선과 증류 교사로 "
              "PANNs [13] 를 썼다.")

    heading(doc, "3. 시스템과 실험 설계")
    body(doc, "시스템. MAX78000FTHR 단독으로 온보드 마이크 16 kHz 입력을 1.024 s 창(16,384 샘플, int8)으로 받아 "
              "250 ms 마다 5클래스(사이렌·유리 파손·비명·개 짖음·배경)를 판단한다. 모델은 가중치 8 bit QAT, "
              "가속기 가중치 메모리 442 KB 안에 들어간다.")
    body(doc, f"데이터와 정답. 세 공개 데이터셋은 모두 Freesound 파생이라 원본 ID 단위로 전역 분할해 누수를 막았고, "
              f"AudioSet 의 실환경 녹음 구간을 학습에 더했다. 시험셋의 유리·비명 창은 저자가 전수 청취해 양성을 "
              f"확정했다(채점 창 {tr:,}). 외부 평가는 AudioSet 의 별도 영상에서 뽑은 {n_ext}창(양성 "
              f"{sum(ext_pos.values())}, 음성 {ext_neg}) 으로, 학습에 쓰지 않았다. 오경보율은 시험셋 배경에서 "
              f"300회/h 가 되는 문턱값을 잡아 외부에도 같은 값을 썼고, 지표는 macro-F1 이다.")
    body(doc, f"구성. {n_w}: 원파형을 128×128 로 접어 1D CNN 에 넣는 벤더 방식. {n_a}: 로그 압축을 NPU 안의 "
              f"구간선형 층으로 근사한 1D 구성. {DN[FE]}: 멜 대역폭에 맞춘 512탭 Gabor 필터뱅크(cos·sin 32쌍)를 가속기의 "
              f"Conv1d 로 두고 프레임 안 에너지 적분(|cos|+|sin| 의 8위치 합)과 정수 로그만 CPU 가 맡은 뒤, 그 32×16 특징을 "
              f"아래 62.5 ms 구성과 같은 2D CNN 에 넣는 구성 — 두 네트워크를 가속기에 동시 적재하고 판단마다 전환한다(판단 주기 256 ms). "
              f"CPU log-mel 네 구성: 프레임 간격 62.5 ms(멜 32)·50 ms(멜 32)·"
              f"25 ms(멜 64)·16 ms(멜 64) 의 증분 로그 멜을 CPU 가 계산하고 2D CNN 을 NPU 에 올린다. "
              f"CPU 추론: {n_w} 의 가중치를 그대로 CMSIS-NN 으로 Cortex-M4F 에서 돌린 대조군(NPU 출력과 비트 일치). "
              f"모든 구성은 같은 데이터로 150 에폭, 같은 스케줄·증강·교사 증류(PANNs 임베딩 위 선형 프로브)로 "
              f"시드 3개씩 학습했다.")
    body(doc, f"지연과 에너지. 지연은 보드에서 SysTick 으로 1,000회 재어 최악값을 보고한다(전원 투입 직후, "
              f"코어 100 MHz, CNN 50 MHz). 에너지는 외부 계측 장비 없이 "
              f"E = t전처리·Pcpu + (t적재 + t출력)·Pload + t추론·Pnpu 로 추정했으며, 전력값은 데이터시트 [1] 와 동일 칩 "
              f"실측 선행연구 [3–5] 두 출처를 나란히 써 범위로 보고한다. 측정값이 아님을 모든 수치에 밝힌다.")

    # ── 결과·결론·참고문헌 (2단 계속) ────────────────────────────────
    # ②′ (NPU 전처리: Gabor 필터뱅크 + CPU 정수 로그) — 보드 실측 뒤 정식 점. 사용자 지시 (2026-10-17) 4·5항:
    #   실측 에너지가 원본 파형보다 전 출처에서 작으면 "저에너지 끝" 문장, 아니면 부정 결과 한 줄만. 수치는 전부 numbers.json.
    fe_w, fe_g = v("welch.②′ − ④")["test_f1_S3"], v("welch.②′ − 가 u1000")["test_f1_S3"]
    fe_cheaper = all(energy(FE, k) < energy(W, k) for k in ("datasheet", "literature", "moosmann"))
    fci = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
    if fe_cheaper:
        fe_neg = (f"선형 부분만 옮기면 에너지는 줄지만 정확도는 그대로다. {DN[FE]} 는 사전에 정한 목표({s_w} 초과 또는 "
                  f"{s_ga}과 동등)는 넘지 못했으나({s_w} 대비 {fci(fe_w)}, {s_ga} 대비 {fci(fe_g)}), {s_w}과 구분되지 않는 정확도를 "
                  f"더 낮은 에너지로 달성해 정확도–에너지 곡선의 저에너지 끝을 이룬다. 판단 1회 {latency(FE):.2f} ms 대 {latency(W):.2f} ms, "
                  f"추정 에너지는 데이터시트 기준 {energy(FE):.0f} 대 {energy(W):.0f} μJ, Moosmann 기준 {energy(FE, 'moosmann'):.0f} 대 "
                  f"{energy(W, 'moosmann'):.0f} μJ 이며, {s_ga}보다 적다는 결론은 NPU 와 CPU 의 전력비가 {brk_fe_ga:.1f} 미만일 때 성립한다"
                  f"(공개 전력값은 전부 그 안). 두 네트워크를 동시 적재한 보드에서 가속기 출력·CPU 특징·최종 로짓이 전부 호스트와 비트 일치했고, "
                  f"CPU 몫 {fe_cpu_ms:.2f} ms 의 대부분은 로그가 아니라 가속기 메모리에서 int32 출력 2,048 개를 읽어 오는 시간이었다. ")
    else:
        fe_neg = (f"선형 부분만 NPU 로 옮기고 정수 로그만 CPU 에 남긴 {DN[FE]} 도 {s_w} 대비 {fci(fe_w)}, {s_ga} 대비 {fci(fe_g)} 로 열세였고 "
                  f"에너지도 {s_w} 보다 적지 않았다({energy(FE):.0f} 대 {energy(W):.0f} μJ). ")
    heading(doc, "4. 결  과")
    body(doc, f"전처리 제거의 비용. {s_w}은 {s_16} 구성보다 시험 macro-F1 이 {f1(H16)[0]-f1(W)[0]:.3f}, 외부 F1 이 "
              f"{f1(H16,'ext_f1')[0]-f1(W,'ext_f1')[0]:.3f} 낮다(표 1). 음성 키워드에서 손실이 없던 선택이 환경음에서는 "
              f"큰 비용을 치른다. 같은 가중치를 CPU 로 돌린 CMSIS-NN 대조군과 비교하면 NPU 추론은 추론만 "
              f"{r_inf:.0f}배, 판단 1회 {r_dec:.0f}배 빠르고 CPU 추론은 판단 주기의 {hop_m4:.0f}% 를 추론에 쓴다(그림 2).")
    body(doc, f"NPU 로 옮긴 전처리는 되찾지 못한다. {n_a} 는 {s_w} 대비 {welch('(a) − ④')} 로 CI 가 0 을 포함하고, "
              f"로그 근사 층 때문에 NPU 사이클이 3.8배 늘어 추정 에너지는 {energy(A)/energy(W):.1f}배다. 앞선 라운드(v1 데이터)에서 "
              f"시험한 변형들도 같았다: " + "; ".join(f"{a} {b}" for a, b in V1_NEG) + ". 지원하지 않는 비선형을 "
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
    if cpu_main:
        var_sent += (f"주된 비용은 NPU 연산({min(ns):.0f}~{max(ns):.0f}%)이 아니라 CPU 가 가속기 메모리에서 결과를 읽어 오는 "
                     f"시간({min(cs):.0f}~{max(cs):.0f}%)이었다.")
    else:
        shares = ", ".join(f"{DN[n]} CPU {100 * (vm[n]['cpu_share_of_delta'] or 0):.0f}% / NPU {100 * (vm[n]['npu_share_of_delta'] or 0):.0f}%" for n in vm)
        var_sent += f"늘어난 비용은 CPU 의 가속기 메모리 읽기와 NPU 연산에 나뉘었다({shares})."
    body(doc, var_sent)
    body(doc, f"최적점은 중간에 있다. {s_ga}은 {s_w}보다 시험 macro-F1 {welch('가 u1000 − ④')}, 외부 F1 "
              f"{welch('가 u1000 − ④', 'ext_f1')} 로 유의하게 높고, 판단 1회 지연은 {latency(GA):.2f} ms 대 {latency(W):.2f} ms 로 "
              f"비슷하며, 추정 에너지는 데이터시트 기준 {energy(GA):.0f} 대 {energy(W):.0f} μJ 로 적고 Moosmann 기준 "
              f"{energy(GA,'moosmann'):.0f} 대 {energy(W,'moosmann'):.0f} μJ 로 같다. 이유는 그림 2 에 있다. 로그 멜이 16,384 샘플을 "
              f"512 값으로 줄이므로 NPU 추론이 {v('latency.④')['infer']:.2f} ms 에서 {lat_ga['infer']:.2f} ms 로 줄고, "
              f"CPU 전처리 {lat_ga['pre']:.2f} ms 가 그 차이보다 작다. 전처리는 병목이 아니라 데이터 축소 단계였고, "
              f"NPU 가 원시 샘플 전량을 처리하는 순간 그 이득이 사라진다. 프레임 간격을 좁힐수록 정확도는 오르지만 "
              f"({s_25} {fmt_pm(H25)}, {s_16} {fmt_pm(H16)}) 전처리가 지연의 2/3 를 차지해 에너지가 {energy(H16)/energy(GA):.0f}배가 된다. "
              f"{s_25} 와 {s_16} 의 차이는 {welch('간격 400 − ①′')} 로 작지만 0 을 포함하지 않았다. "
              f"그림 1 의 파레토 앞면은 {' → '.join(pf)} 이고, {s_w}과 {n_a} 는 모두 다른 점에 지배된다.")
    body(doc, f"대형 모델 기준선. 같은 정답·같은 문턱값 규칙으로 PANNs(CNN14) 를 채점하면 시험 macro-F1 {pann:.3f} 이다. "
              f"{s_16} 구성은 그 {100*f1(H16)[0]/pann:.0f}% 에 이른다.")
    figure(doc, os.path.join(PAPER, "fig2_latency_breakdown.png"), 88,
           f"그림 2. 판단 1회 지연 분해(보드 실측, 1,000회 최악값). 왼쪽은 NPU 추론 {len([c for c in CFG['configs'] if not c.get('variant_of')])}개 구성"
           f"(CPU 전처리 / 입력 적재 / NPU 추론; {DN[FE]} 의 변형 3종은 표 1 에만), "
           "오른쪽은 같은 가중치의 CPU 추론(Cortex-M4F, CMSIS-NN)과 판단 주기 250 ms.")

    heading(doc, "5. 한계와 결론")
    body(doc, "한계. 에너지는 직접 계측하지 않은 추정값이며 전력 출처에 따라 결론의 강도가 다르다(CPU log-mel 62.5 ms 구성의 우위는 "
              "'적다' 와 '같다' 사이). 시드가 3개라 신뢰구간이 넓다. 정답 청취는 저자 1인이다. 비명과 환호의 경계는 "
              "음향만으로 가를 수 없어 비명 재현율은 그 모호성을 포함한다. 시험셋 배경의 절반이 하드 네거티브라 "
              f"오경보율은 실배치보다 부풀려져 있다. {DN[FE]} 의 CPU 구간은 가속기 전원이 켜진 채 돌므로 적재 전력으로 "
              "보수적으로 계산했고, 가속기 메모리 읽기를 DMA 로 바꾸면 더 줄 수 있다. 음량 정규화·필터 해상도·용량 확대 등 원파형 쪽 변형을 더 시도할 "
              "여지는 남아 있다.")
    body(doc, "결론. NPU 탑재 MCU 에서 '전처리를 전부 가속기로' 는 환경음 감지에서 최적이 아니다. 지원 연산이 "
              "제한된 가속기에 비선형 전처리를 옮기는 시도는 정확도도 에너지도 얻지 못했고" +
              (", 선형 부분만 옮긴 구성은 에너지를 조금 줄이는 대신 정확도를 잃어 곡선의 저에너지 끝에 머물렀으며" if fe_cheaper else "") +
              ", 작은 CPU 전처리로 "
              "데이터량을 줄인 뒤 작은 모델을 가속기에 올리는 구성이 정확도–에너지 전선 위에 있었다. 가속기 설계 "
              "지침은 연산 위치가 아니라 가속기에 들어가는 데이터율을 기준으로 세워야 한다.")

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
        para(doc, f"[{i}] {r}", size=7.5, align="just", space_after=1, line=1.05)

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
