#!/usr/bin/env python3
"""논문용 표·그림을 실측 결과 파일에서 자동 생성한다 → docs/paper/.

원칙 (2026-10-16 지시)
- 모든 숫자는 결과 파일에서 읽는다. 이 스크립트에는 숫자를 적지 않는다.
  예외인 공개 전력값은 docs/paper/paper_config.json 에 출처와 함께 둔다.
- 기준: 데이터 v4+D, 정답 testL-S3 (시험셋) · real31L (외부), 지연은 v4+D 시드 1 보드 실측
  (data/logs-local/board_v4d_*.log), 에너지는 "추정(실측 지연 × 공개 전력)".

원천 파일
  data/external_eval/scream_listened_test.json   v4+D 7구성 × 시드 — testL / testL-S3 / 외부 (F1·recall)
  data/external_eval/scream_stage3.json          v4+D 5구성 testL-S3 (위와 교차 검증), 정답 창 수
  data/external_eval/v4_eval.json                판 v1~v4+D 외부 F1·mAP (real31L), testL — 그림 4·외부 mAP
  data/external_eval/v4_panns.json               PANNs CNN14 zero-shot 외부 기준선 (real31L)
  data/ceiling-panns-zs.npz                      PANNs 시험셋 점수 → testL-S3 로 다시 채점 (그림 3 점선)
  data/logs-local/board_v4d_{wave,u1000,u800,h400,melinc,m4cmsis}.log   보드 실측 (적재·추론·출력, 1000회 최악)
  docs/results/synthesis-check.md                CPU 전처리 사이클 (§14.1, §12.1b, §10 — 보드 실측의 기록)
  data/synth/out/<구성>/log.txt, cnn.h           합성 NPU 사이클, MAC, 가중치 메모리
  datasets/melfeat.py                            프레임 정의 (INC_SPECS, N_FFT 등)
  data_listened.csv, data_listened_scream.csv, data_listened_test_scream.csv   청취 판정 (창 ID·판정만)
  docs/results/curve-points-v4d.csv              지연·에너지 교차 검증용

사용 (WSL2, ai8x venv):  OPENBLAS_NUM_THREADS=1 ~/ai8x-training/venv/bin/python tools/make_paper_assets.py
"""
import collections
import csv
import json
import math
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "tools"), os.path.join(REPO, "datasets"), os.path.join(REPO, "scripts")]
import melfeat as MF                                   # noqa: E402
import eval_external as E                              # noqa: E402
import eval_v2 as V2                                   # noqa: E402
import label_audit as LA                               # noqa: E402

OUT = os.path.join(REPO, "docs", "paper")
CFG_PATH = os.path.join(OUT, "paper_config.json")
EXT = os.path.join(REPO, "data", "external_eval")
LOGS = os.path.join(REPO, "data", "logs-local")
SYNTH_CHECK = os.path.join(REPO, "docs", "results", "synthesis-check.md")
CLS = ["siren", "glass", "scream", "dog_bark"]
CLS_KO = ["사이렌", "유리 파손", "비명", "개 짖음"]
NUMBERS = {}          # numbers.json — 값마다 원천 파일과 열


def rel(p):
    return os.path.relpath(p, REPO).replace(os.sep, "/")


def num(key, value, unit, source, field, note=None):
    d = {"value": value, "unit": unit, "source": source, "field": field}
    if note:
        d["note"] = note
    NUMBERS[key] = d
    return value


def fnum(s):
    return float(s.replace(",", ""))


# ── 1. 보드 실측 로그 ────────────────────────────────────────────────────────
def read_board_log(tag):
    """board_v4d_<tag>.log → {load/infer/unload: 최악 µs, core_hz, cnn_hz}."""
    path = os.path.join(LOGS, f"board_v4d_{tag}.log")
    r = {"path": rel(path)}
    with open(path, encoding="utf-8", errors="replace") as f:
        txt = f.read()
    for seg in ("load", "infer", "unload"):
        m = re.search(rf"^{seg}\s+us\s+.*?max\s+([\d.]+)", txt, re.M)
        if m:
            r[seg] = float(m.group(1)) / 1000.0            # ms
    m = re.search(r"(?:SystemCoreClock|core clock) (\d+) Hz", txt)
    r["core_hz"] = int(m.group(1))
    m = re.search(r"CNN clock PCLK (\d+) Hz", txt)
    r["cnn_hz"] = int(m.group(1)) if m else None
    m = re.search(r"N_ITER (\d+)", txt)
    r["n_iter"] = int(m.group(1))
    r["kat"] = "PASS (bit-exact)" in txt
    m = re.search(r"output mismatches: (\d+)", txt)
    r["mismatch"] = int(m.group(1))
    return r


FE_SEGS = ("load1", "cfg1", "infer1", "cpu", "load2", "cfg2", "infer2", "unload", "total")


def read_board_log_fe(tag):
    """②′ 두 네트워크 측정 펌웨어(firmware/common/fe_measure_main.c) 로그 → 구간 8개 최악 ms. KAT 셋(p1·CPU 특징·p2) 전부 PASS 여야 한다."""
    path = os.path.join(LOGS, f"board_v4d_{tag}.log")
    r = {"path": rel(path)}
    with open(path, encoding="utf-8", errors="replace") as f:
        txt = f.read()
    for seg in FE_SEGS:
        m = re.search(rf"^{seg}\s+us\s+.*?max\s+([\d.]+)", txt, re.M)
        assert m, (tag, seg)
        r[seg] = float(m.group(1)) / 1000.0
    r["core_hz"] = int(re.search(r"core clock (\d+) Hz", txt).group(1))
    r["cnn_hz"] = int(re.search(r"CNN clock PCLK (\d+) Hz", txt).group(1))
    r["n_iter"] = int(re.search(r"N_ITER (\d+)", txt).group(1))
    r["kat"] = txt.count("PASS (bit-exact)") >= 3 and "FAIL" not in txt
    r["mismatch"] = int(re.search(r"output mismatches: (\d+)", txt).group(1))
    return r


def pareto_front(pts):
    """pts = {이름: (에너지, 정확도)} → 지배되지 않는 이름들 (에너지 ≤ 이고 정확도 ≥ 인 다른 점이 없다; 둘 다 같으면 둘 다 남긴다)."""
    front = []
    for n, (e, f) in pts.items():
        dominated = any((e2 <= e and f2 >= f) and (e2 < e or f2 > f) for m, (e2, f2) in pts.items() if m != n)
        if not dominated:
            front.append(n)
    return front


# ── 2. CPU 전처리 사이클 — synthesis-check.md 의 실측 표 ─────────────────────
def md_rows(path, section=None):
    """표 행(| 로 시작)만. section 을 주면 그 제목(예: '### 12.1b')부터 다음 제목 전까지."""
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    if section:
        start = next(i for i, ln in enumerate(lines) if ln.startswith(section + " "))
        end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("#")), len(lines))
        lines = lines[start:end]
    return [ln for ln in lines if ln.startswith("|")]


def row_numbers(row, min_digits=4):
    """표 행에서 1,000 단위 콤마 숫자(사이클)를 뽑는다 — 작은 수(KAT 개수 등)는 거른다."""
    return [fnum(x) for x in re.findall(r"\d{1,3}(?:,\d{3})+|\d+", row) if len(x.replace(",", "")) >= min_digits]


def read_preproc_cycles():
    """{framing: (cycles, 출처 설명)} — §14.1 (u1000/u800/h400: push 최악 + 링 버퍼 풀기 최악), §12.1b (①′: V1 push 최악 + get)."""
    src = rel(SYNTH_CHECK)

    def find(section, prefix):
        hit = [r for r in md_rows(SYNTH_CHECK, section) if r.startswith(prefix)]
        if len(hit) != 1:
            sys.exit(f"[에러] {src} {section} 에서 행 '{prefix}' 를 {len(hit)}개 찾았다 — 표 형식이 바뀌었나?")
        return hit[0]

    hdr = find("### 14.1", "| | u1000 (")
    cols = [c.strip() for c in hdr.strip("|").split("|")][1:]
    order = [re.match(r"(u1000|u800|h400)", c).group(1) for c in cols]
    push = row_numbers(find("### 14.1", "| push 최악 (사이클) |"))
    get = row_numbers(find("### 14.1", "| 링 버퍼 풀기 최악 |"))
    assert len(push) == len(get) == len(order) == 3, (push, get, order)
    out = {k: (p + g, f"{src} §14.1 'push 최악' + '링 버퍼 풀기 최악' ({k})") for k, p, g in zip(order, push, get)}
    v1 = row_numbers(find("### 12.1b", "| **V1 float32 rfft + LUT 로그** |"), 6)     # 최소/중앙/최악 push 사이클
    getc = row_numbers(find("### 12.1", "| get — 링 버퍼 64열을 시간 순으로 |"))
    out["inc"] = (max(v1) + max(getc), f"{src} §12.1b 'V1 float32 rfft + LUT 로그' 최악 push + §12.1 'get' 최악")
    first = row_numbers(find("## 10.", "| **전처리 전체** (코어 사이클) |"), 6)
    out["_first_impl"] = (max(first), f"{src} §10 '전처리 전체' 최악 (최적화 전 첫 구현, 일괄 64프레임)")
    return out


# ── 3. 합성 로그 ─────────────────────────────────────────────────────────────
def read_synth(d):
    base = os.path.join(REPO, "data", "synth", "out", d)
    r = {"dir": rel(base)}
    with open(os.path.join(base, "log.txt"), encoding="utf-8", errors="replace") as f:
        log = f.read()
    r["cycles"] = int(fnum(re.search(r"^Total\s+([\d,]+) cycles", log, re.M).group(1)))
    r["macc"] = int(fnum(re.search(r"Hardware: [\d,]+ ops \(([\d,]+) macc", log).group(1)))
    r["layers"] = len(re.findall(r"^\s*Layer \d+: [\d,]+ ops", log, re.M))
    with open(os.path.join(base, "cnn.h"), encoding="utf-8", errors="replace") as f:
        h = f.read()
    r["weight_bytes"] = int(fnum(re.search(r"Weight memory:\s+([\d,]+) bytes out of ([\d,]+)", h).group(1)))
    r["weight_total"] = int(fnum(re.search(r"Weight memory:\s+([\d,]+) bytes out of ([\d,]+)", h).group(2)))
    return r


# ── 4. 정확도 — 결과 JSON ────────────────────────────────────────────────────
def load_json(name):
    with open(os.path.join(EXT, name), encoding="utf-8") as f:
        return json.load(f)


def mean_sd(v):
    v = np.asarray(v, float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(ln for ln in f if not ln.startswith("#")))


# ── 5. PANNs 를 testL-S3 로 다시 채점 ────────────────────────────────────────
def test_keep_s3(zt, truth):
    """시험셋 채점 마스크: 유리는 C_test_glass 청취 O 만, 비명은 S(yellshout_test)+T(screaming_test) 청취 O 만."""
    L = load_csv(os.path.join(REPO, "data_listened.csv"))
    S = load_csv(os.path.join(REPO, "data_listened_scream.csv"))
    T = load_csv(os.path.join(REPO, "data_listened_test_scream.csv"))
    glass = {(r["clip"], int(r["start_sample"])): r["verdict"] for r in L if r["set"] == "C_test_glass" and r["role"] == "target"}
    scream = {(r["clip"], int(r["start_sample"])): r["verdict"] for r in S if r["definition"] == "yellshout_test"}
    scream |= {(r["clip"], int(r["start_sample"])): r["verdict"] for r in T if r["definition"] == "screaming_test"}
    keep = np.ones(len(zt["y"]), bool)
    cnt = collections.Counter()
    for i, (c, s, y) in enumerate(zip(zt["clips"], zt["starts"], zt["y"])):
        k = (str(c), int(s))
        if y == 1:
            v = glass.get(k)
            if v is None:
                sys.exit(f"[에러] 시험셋 유리 창에 청취 판정이 없다: {k}")
            cnt["glass_" + v] += 1
            keep[i] = v == "O"
        elif y == 2:
            v = scream.get(k)
            if v is None:
                sys.exit(f"[에러] 시험셋 비명 창에 청취 판정이 없다: {k}")
            cnt["scream_" + v] += 1
            keep[i] = v == "O"
    if int(keep.sum()) != truth["testL_S3_n"] or cnt["scream_O"] != truth["test_scream_n"]:
        sys.exit(f"[에러] testL-S3 마스크가 scream_stage3.json 의 정답과 다르다: {int(keep.sum())} / {cnt}")
    return keep, cnt


def panns_test_s3(keep):
    zt = np.load(LA.TEST_CACHE, allow_pickle=True)
    S_t = LA.zeroshot(zt["P"], LA.label_names())
    thr = E.thr_at(S_t, zt["y"] == E.BG)
    p = E.predict(S_t, thr)
    f1, rec, _ = E.metrics(zt["y"][keep], p[keep])
    return {"f1": float(f1), "recall": [float(v) for v in rec[:4]], "source": rel(LA.TEST_CACHE)}


# ── 6. 그림 공통 ─────────────────────────────────────────────────────────────
def setup_mpl(fonts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    fam = None
    for p in fonts:
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            fam = fm.FontProperties(fname=p).get_name()
            break
    if fam is None:
        print("[경고] 한글 글꼴을 찾지 못했다 — paper_config.json fonts 확인", file=sys.stderr)
    else:
        plt.rcParams["font.family"] = fam
    plt.rcParams.update({"axes.unicode_minus": False, "svg.fonttype": "none", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                         "figure.dpi": 100, "savefig.dpi": 300})
    return plt, fam


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return [rel(os.path.join(OUT, f"{name}.{ext}")) for ext in ("png", "svg")]


HATCH = ["", "////", "....", "xxxx", "\\\\\\\\", "++++", "oo"]


# ── 7. 표 저장 ───────────────────────────────────────────────────────────────
def write_table(name, header, rows, meta_lines=()):
    path = os.path.join(OUT, f"{name}.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        for ln in meta_lines:
            f.write(f"# {ln}\n")
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return header, rows, list(meta_lines)


def write_xlsx(tables):
    from openpyxl import Workbook
    from openpyxl.styles import Font
    wb = Workbook()
    wb.remove(wb.active)
    for name, (header, rows, meta) in tables.items():
        ws = wb.create_sheet(title=name[:31])
        for ln in meta:
            ws.append([ln])
            ws.cell(row=ws.max_row, column=1).font = Font(italic=True, color="666666")
        ws.append(header)
        for c in range(1, len(header) + 1):
            ws.cell(row=ws.max_row, column=c).font = Font(bold=True)
        for r in rows:
            ws.append(list(r))
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(60, max(10, width * 1.1))
    path = os.path.join(OUT, "paper_tables.xlsx")
    wb.save(path)
    return rel(path)


def fmt_ms(mu, sd, nd=3):
    return f"{mu:.{nd}f} ± {sd:.{nd}f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(CFG_PATH, encoding="utf-8") as f:
        C = json.load(f)
    PS = C["power_sources"]
    curve_src = [k for k, v in PS.items() if v.get("use_in_curve")]
    # 표시 이름 (내부 이름 → 논문 표기) — paper_config.json display 절 한 곳에서만 읽는다
    D = C["display"]
    DN = {k: v["name"] for k, v in D["configs"].items()}          # 긴 이름 (표)
    DS = {k: v["short"] for k, v in D["configs"].items()}         # 짧은 이름 (축 눈금)
    DT = {k: v["tag"] for k, v in D["configs"].items()}           # 점 옆 이름 (그림 1)
    DG = {k: v["group"] for k, v in D["configs"].items()}
    GC = {g: v["color"] for g, v in D["groups"].items()}
    GL = {g: v["label"] for g, v in D["groups"].items()}
    M4N, M4S = D["m4"]["name"], D["m4"]["short"]
    TR, PN, EDN = D["truth"], D["panns"], D["editions"]
    assert set(DN) == {c["name"] for c in C["configs"]}, "display.configs 와 configs 의 이름이 다르다"

    # ── 결과 JSON
    slt = load_json("scream_listened_test.json")
    st3 = load_json("scream_stage3.json")
    v4e = load_json("v4_eval.json")
    panns_rows = load_json("v4_panns.json")
    truth = st3["truth"]
    src_slt, src_st3, src_v4e, src_panns = (rel(os.path.join(EXT, n)) for n in
                                           ("scream_listened_test.json", "scream_stage3.json", "v4_eval.json", "v4_panns.json"))
    n_test_all = slt["score"]["n"]["testL"]
    num("truth.testL_S3_windows", truth["testL_S3_n"], "창", src_st3, "truth.testL_S3_n")
    num("truth.testL_windows", n_test_all, "창", src_slt, "score.n.testL")
    num("truth.test_scream_positives_S3", truth["test_scream_n"], "창", src_st3, "truth.test_scream_n")
    num("truth.test_scream_verdicts", slt["score"]["n"]["test_scream_verdicts"], "창", src_slt, "score.n.test_scream_verdicts")
    num("truth.ext_scream_positives_real31L", truth["ext_scream_n"], "창", src_st3, "truth.ext_scream_n")
    num("truth.ext_positives_real31L", v4e["labels"]["ext_n_pos"]["real31L"], "창", src_v4e, "labels.ext_n_pos.real31L")
    num("truth.ext_negatives_real31L", v4e["labels"]["ext_n_neg"]["real31L"], "창", src_v4e, "labels.ext_n_neg.real31L")

    # ── 구성별 정확도 (시드 단위)
    ACC = {}
    for c in C["configs"]:
        s = slt["score"]["configs"][c["slt"]]["seeds"]
        if c["v4eval"]:
            v = v4e["editions"]["v4+D"][c["v4eval"]]["seeds"]
            map_src = (src_v4e, f"editions.v4+D[{c['v4eval']}].seeds[*].ext_map_real31L")
        else:                                 # ②′: eval_v4.score_ckpt 와 같은 코드로 따로 뽑은 fe_eval.json (tools/fe_ext_eval.py)
            v = load_json(c["fe_eval"])["seeds"]
            map_src = (rel(os.path.join(EXT, c["fe_eval"])), "seeds[*].ext_map_real31L")
        assert len(s) == len(v), (c["name"], len(s), len(v))
        for a, b in zip(s, v):
            assert abs(a["thr"] - b["thr"]) < 1e-9, f"{c['name']}: 시드 정렬이 다르다 (thr {a['thr']} vs {b['thr']})"
            assert abs(a["testL"]["f1"] - b["test_f1_L"]) < 1e-9 and abs(a["ext"]["f1"] - b["ext_f1_real31L"]) < 1e-9
        if c["stage3"]:
            t = st3["editions"]["v4+D"][c["stage3"]]["seeds"]
            for a, b in zip(s, t):
                assert abs(a["testL_S3"]["f1"] - b["test_f1"]) < 1e-9, f"{c['name']}: S3 값이 stage3 와 다르다"
        ACC[c["name"]] = {
            "n": len(s),
            "f1_S3": [x["testL_S3"]["f1"] for x in s],
            "f1_L": [x["testL"]["f1"] for x in s],
            "rec_S3": np.array([x["testL_S3"]["recall"][:4] for x in s]),
            "scream_S3": [x["testL_S3"]["scream_recall"] for x in s],
            "ext_f1": [x["ext"]["f1"] for x in s],
            "ext_rec": np.array([x["ext"]["recall"][:4] for x in s]),
            "ext_map": [x["ext_map_real31L"] for x in v],
            "ext_fire_n": [x["ext"]["ext_fire_n"] for x in s],
            "bgbg": [x["ext"]["bgbg_fire"] for x in s],
            "ckpts": [x["ckpt"] for x in s],
        }
        for k, fld in (("f1_S3", "score.configs[%s].seeds[*].testL_S3.f1"), ("f1_L", "score.configs[%s].seeds[*].testL.f1"),
                       ("ext_f1", "score.configs[%s].seeds[*].ext.f1"), ("scream_S3", "score.configs[%s].seeds[*].testL_S3.scream_recall")):
            mu, sd = mean_sd(ACC[c["name"]][k])
            num(f"acc.{c['name']}.{k}", {"mean": mu, "sd": sd, "n": len(s)}, "", src_slt, fld % c["slt"])
        mu, sd = mean_sd(ACC[c["name"]]["ext_map"])
        num(f"acc.{c['name']}.ext_map", {"mean": mu, "sd": sd, "n": len(v)}, "", map_src[0], map_src[1])

    # ── 지연: 보드 로그 + 전처리 사이클 + 합성
    pre = read_preproc_cycles()
    LAT = {}
    wave_log = None
    for c in C["configs"]:
        syn = read_synth(c["synth_dir"])
        if c.get("synth_dir_p1"):              # ②′: 두 네트워크 — 합성 수치는 p1 + p2 의 합
            s1 = read_synth(c["synth_dir_p1"])
            syn = {"dir": f"{s1['dir']} + {syn['dir']}", "cycles": s1["cycles"] + syn["cycles"], "macc": s1["macc"] + syn["macc"],
                   "layers": s1["layers"] + syn["layers"], "weight_bytes": s1["weight_bytes"] + syn["weight_bytes"], "weight_total": syn["weight_total"]}
        if c.get("two_networks"):              # ②′: 보드 실측, 구간 8개. CPU 구간은 NPU 전원이 켜진 채 돌므로 에너지는 P_load 로 (보수적)
            b = read_board_log_fe(c["board_log"])
            assert b["kat"] and b["mismatch"] == 0, (c["name"], b)
            LAT[c["name"]] = {"pre": b["cpu"], "load": b["load1"] + b["load2"] + b["cfg1"] + b["cfg2"], "infer": b["infer1"] + b["infer2"],
                              "out": b["unload"], "est": False, "pre_power": "load", "detail": {k: b[k] for k in FE_SEGS}, "src": b["path"],
                              "pre_src": "보드 실측 cpu 구간 (|cos|+|sin| → 8위치 합 → 정수 로그 32×4 값 → 버퍼 밀기 → 입력 워드 포장), NPU 전원 켜진 채 → P_load 로 계산; 적재 열에 네트워크 전환(cfg1+cfg2) 포함",
                              "core_hz": b["core_hz"], "cnn_hz": b["cnn_hz"], "n_iter": b["n_iter"], "synth": syn}
        elif c["board_log"]:
            b = read_board_log(c["board_log"])
            assert b["kat"] and b["mismatch"] == 0, (c["name"], b)
            if c["board_log"] == "wave":
                wave_log = b
            core = b["core_hz"]
            t_pre, pre_src = (pre[c["framing"]][0] / core * 1e3, pre[c["framing"]][1]) if c["framing"] else (0.0, "전처리 없음")
            LAT[c["name"]] = {"pre": t_pre, "load": b["load"], "infer": b["infer"], "out": b["unload"], "est": False,
                              "src": b["path"], "pre_src": pre_src, "core_hz": core, "cnn_hz": b["cnn_hz"], "n_iter": b["n_iter"], "synth": syn}
        else:                                     # (a): 보드 미실측 — 적재·출력은 ④ 실측, 추론은 합성 사이클 ÷ CNN 클럭
            assert wave_log is not None
            LAT[c["name"]] = {"pre": 0.0, "load": wave_log["load"], "infer": syn["cycles"] / wave_log["cnn_hz"] * 1e3, "out": wave_log["unload"],
                              "est": True, "src": f"{syn['dir']}/log.txt (Total cycles) ÷ CNN 클럭 {wave_log['cnn_hz']} Hz; 적재·출력은 {wave_log['path']} (④ 실측)",
                              "pre_src": "전처리 없음", "core_hz": wave_log["core_hz"], "cnn_hz": wave_log["cnn_hz"], "n_iter": None, "synth": syn}
        L = LAT[c["name"]]
        L["total"] = L["pre"] + L["load"] + L["infer"] + L["out"]
    m4 = read_board_log(C["m4"]["board_log"])
    assert m4["kat"] and m4["mismatch"] == 0
    LAT_M4 = {"pre": 0.0, "load": m4["load"], "infer": m4["infer"], "out": 0.0, "total": m4["load"] + m4["infer"], "src": m4["path"], "n_iter": m4["n_iter"]}
    hop_ms = E.HOP_MS

    # 교차 검증: curve-points-v4d.csv 의 time_ms 와 0.02 ms 안에서 같아야 한다
    cp = {r["id"]: r for r in load_csv(os.path.join(REPO, "docs", "results", "curve-points-v4d.csv"))}
    for c in C["configs"]:
        cid = {"wave": "wave", "u1000": "u1000", "u800": "u800", "h400": "h400", "melinc": "melinc"}.get(c["board_log"])
        if cid and abs(float(cp[cid]["time_ms"]) - LAT[c["name"]]["total"]) > 0.02:
            sys.exit(f"[에러] {c['name']} 판단 1회 {LAT[c['name']]['total']:.3f} ms 가 curve-points-v4d.csv {cp[cid]['time_ms']} 와 다르다")
    if abs(float(cp["m4cmsis"]["time_ms"]) - LAT_M4["total"]) > 0.02:
        sys.exit("[에러] CMSIS-NN 판단 1회가 curve-points-v4d.csv 와 다르다")

    # ── 에너지 (추정)
    def energy(L, ps, m4=False):
        if m4:
            return L["load"] * ps["P_load_mW"] + L["infer"] * ps["P_cpu_mW"]
        p_pre = ps["P_load_mW"] if L.get("pre_power") == "load" else ps["P_cpu_mW"]   # ②′: CPU 구간이 NPU 전원 켜진 채 돈다
        return L["pre"] * p_pre + (L["load"] + L["out"]) * ps["P_load_mW"] + L["infer"] * ps["P_npu_mW"]

    for name, L in LAT.items():
        L["E"] = {k: energy(L, PS[k]) for k in PS}
    LAT_M4["E"] = {k: energy(LAT_M4, PS[k], m4=True) for k in PS}
    for c in C["configs"]:
        cid = {"wave": "wave", "u1000": "u1000", "u800": "u800", "h400": "h400", "melinc": "melinc"}.get(c["board_log"])
        if cid and abs(float(cp[cid]["e_datasheet_uJ"]) - LAT[c["name"]]["E"]["datasheet"]) > 1.0:
            sys.exit(f"[에러] {c['name']} 에너지(가) {LAT[c['name']]['E']['datasheet']:.1f} 가 curve-points-v4d.csv 와 1 µJ 넘게 다르다")
    for name, L in LAT.items():
        num(f"latency.{name}", dict({k: L[k] for k in ("pre", "load", "infer", "out", "total")},
                                  **({"detail": L["detail"], "pre_power": "load"} if "detail" in L else {})), "ms", L["src"],
            "load/infer/unload us max (1000회 최악)" + ("; 전처리 " + L["pre_src"] if L["pre"] else ""), "추정(보드 미실측)" if L["est"] else None)
        num(f"energy.{name}", {k: L["E"][k] for k in curve_src}, "µJ", f"{L['src']} × {rel(CFG_PATH)} power_sources",
            C["energy_note"], "추정(실측 지연 × 공개 전력)" + (" — 지연도 추정" if L["est"] else ""))
    num("latency.M4 CMSIS-NN", {k: LAT_M4[k] for k in ("load", "infer", "total")}, "ms", LAT_M4["src"], "load/infer us max (1000회 최악)")
    num("energy.M4 CMSIS-NN", {k: LAT_M4["E"][k] for k in curve_src}, "µJ", f"{LAT_M4['src']} × {rel(CFG_PATH)} power_sources", C["energy_note"],
        "추정(실측 지연 × 공개 전력)")
    L4 = LAT["④"]
    num("ratio.infer_m4cmsis_over_npu", LAT_M4["infer"] / L4["infer"], "배", f"{LAT_M4['src']}, {L4['src']}", "infer us max 비")
    num("ratio.decision_m4cmsis_over_npu", LAT_M4["total"] / L4["total"], "배", f"{LAT_M4['src']}, {L4['src']}", "(load+infer) / (load+infer+unload) 비")
    num("ratio.energy_m4cmsis_over_npu", {k: LAT_M4["E"][k] / L4["E"][k] for k in curve_src}, "배", "위 energy 항목", "같은 전력 상수 쌍", "추정")
    first_ms = pre["_first_impl"][0] / L4["core_hz"] * 1e3
    num("preproc.first_impl_ms", first_ms, "ms", pre["_first_impl"][1], "최악 사이클 ÷ 코어 클럭", "최적화 전 일괄 구현 (①, 2026-10-02)")
    num("preproc.inc_v1_ms", LAT["①′"]["pre"], "ms", pre["inc"][1], "최악 사이클 ÷ 코어 클럭", "①′ 증분 V1")
    num("hop_ms", hop_ms, "ms", rel(os.path.join(REPO, "tools", "eval_external.py")), "HOP_MS")
    for name, L in LAT.items():
        num(f"hop_fraction.{name}", L["total"] / hop_ms, "비", "latency 항목", "판단 1회 / hop")
    num("hop_fraction.M4 CMSIS-NN", LAT_M4["total"] / hop_ms, "비", "latency 항목", "판단 1회 / hop")

    # ── PANNs
    zt = np.load(LA.TEST_CACHE, allow_pickle=True)
    keep, kcnt = test_keep_s3(zt, truth)
    pt = panns_test_s3(keep)
    pe = {(r["definition"], r.get("point")): r for r in panns_rows if r["where"] == "ext"}
    p_ext = pe[("real31L", "test300")]
    p_ext300 = pe[("real31L", "ext300")]
    num("panns.test_S3", pt, "", pt["source"], "zeroshot(P) → 시험셋 배경 @300/h 문턱값 → testL-S3 마스크 (이 스크립트가 다시 채점)")
    num("panns.ext_real31L_test300", {"f1": p_ext["f1"], "recall": p_ext["recall"][:4], "map": p_ext["map"]}, "", src_panns,
        "where=ext, definition=real31L, point=test300 (시험셋 문턱값 — 모델과 같은 규칙)")
    num("panns.ext_real31L_ext300", {"f1": p_ext300["f1"], "recall": p_ext300["recall"][:4], "map": p_ext300["map"]}, "", src_panns,
        "where=ext, definition=real31L, point=ext300 (외부 배경 문턱값 — 참고)")
    num("truth.test_glass_verdicts", {k.split("_")[1]: v for k, v in kcnt.items() if k.startswith("glass_")}, "창",
        rel(os.path.join(REPO, "data_listened.csv")), "set=C_test_glass, role=target, verdict")

    # ── Welch
    WEL = {}
    for a_name, b_name in C["welch_pairs"]:          # a − b
        row = {}
        for k, lab in (("f1_S3", "test_f1_S3"), ("f1_L", "test_f1_L"), ("ext_f1", "ext_f1"), ("ext_map", "ext_map")):
            d, lo, hi = V2.welch(ACC[b_name][k], ACC[a_name][k])
            row[lab] = (float(d), float(lo), float(hi))
        WEL[(a_name, b_name)] = row
        num(f"welch.{a_name} − {b_name}", row, "", f"{src_slt} (test_f1_S3·test_f1_L·ext_f1), {src_v4e} (ext_map)",
            "시드 1~3 Welch 95% CI (eval_v2.welch)")

    # ── 그림
    plt, fam = setup_mpl(C["fonts"])
    figs = {}
    names = [c["name"] for c in C["configs"]]
    markers = {c["name"]: c["marker"] for c in C["configs"]}
    est = {c["name"]: c["estimated_latency"] for c in C["configs"]}

    # 그림 1 — 정확도-에너지, 좌: 시험 데이터 / 우: 외부 데이터. 색 = 전처리 위치, 마커 = 구성, 점 옆 이름, CPU log-mel 은 에너지 순으로 선
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))
    allE = [LAT[n]["E"][k] for n in names for k in curve_src]
    ticks = [t for t in (20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000) if min(allE) * 0.85 <= t <= max(allE) * 1.15]
    cpu_names = sorted([n for n in names if DG[n] == "cpu"], key=lambda n: LAT[n]["E"]["datasheet"])
    PARETO = {}          # 패널(정답)별 파레토 앞면 — 점 추정(데이터시트 에너지, 시드 평균) 기준
    for ax, key, title in ((axes[0], "f1_S3", f"(a) {TR['testL_S3']}"), (axes[1], "ext_f1", f"(b) {TR['real31L']}")):
        xs_c = [LAT[n]["E"]["datasheet"] for n in cpu_names]
        ys_c = [mean_sd(ACC[n][key])[0] for n in cpu_names]
        ax.plot(xs_c, ys_c, "-", color=GC["cpu"], lw=1.0, alpha=0.6, zorder=1)
        for name in names:
            L = LAT[name]; col = GC[DG[name]]
            es = [L["E"][k] for k in curve_src]
            x = L["E"]["datasheet"]
            mu, sd = mean_sd(ACC[name][key])
            ax.errorbar([x], [mu], yerr=[[sd], [sd]], xerr=[[x - min(es)], [max(es) - x]], fmt=markers[name],
                        mfc="white" if est[name] else col, mec=col, ecolor=col, color=col, ms=7, capsize=2, lw=0.9, zorder=3)
            dsp = D["configs"][name]
            ax.annotate(DT[name] + ("*" if est[name] else ""), (x, mu), textcoords="offset points", xytext=dsp.get("offset", [7, -11]),
                        ha=dsp.get("ha", "left"), fontsize=8, color=col)
        # 파레토 앞면 (지배되지 않는 점): 큰 고리 + 점선. 점 추정 기준 (데이터시트 에너지, 시드 평균 F1)
        pts = {n: (LAT[n]["E"]["datasheet"], mean_sd(ACC[n][key])[0]) for n in names}
        front = pareto_front(pts)
        PARETO[key] = sorted(front, key=lambda n: pts[n][0])
        ax.plot([pts[n][0] for n in PARETO[key]], [pts[n][1] for n in PARETO[key]], ls=":", color="0.45", lw=0.9, zorder=2)
        for n in front:
            ax.plot([pts[n][0]], [pts[n][1]], marker="o", ms=14, mfc="none", mec="0.25", mew=0.8, ls="none", zorder=4)
        # (파레토 고리 설명은 캡션에 — 축 안 글은 왼쪽 아래 점 이름과 겹친다)
        # CPU log-mel 묶음 이름은 50 ms → 25 ms 선분 가운데의 바로 왼쪽 위에
        xm = math.sqrt(xs_c[1] * xs_c[2]); ym = (ys_c[1] + ys_c[2]) / 2
        ax.annotate(GL["cpu"] + "\n(숫자 = 프레임 간격)", (xm, ym), textcoords="offset points", xytext=(-5, 3), ha="right", va="bottom",
                    fontsize=7.5, color=GC["cpu"], linespacing=1.1)
        ax.set_xscale("log")
        ax.margins(y=0.12)
        ax.set_xlim(min(allE) * 0.6, max(allE) * 1.25)   # 왼쪽 여유: 가장 적은 점의 이름이 아래 가운데에 들어가게
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel("판단 1회 추정 에너지 (μJ, 로그축)")
        ax.set_ylabel("macro-F1")
        ax.set_title(title, fontsize=9, loc="left")
        ax.annotate("더 좋음 (정확하고 에너지가 작음)", xy=(0.025, 0.975), xytext=(0.075, 0.905), xycoords="axes fraction",
                    textcoords="axes fraction", fontsize=7, color="0.35", va="top", arrowprops=dict(arrowstyle="->", lw=0.8, color="0.35"))
        ax.grid(True, which="both", lw=0.3, color="0.85")
    fig.tight_layout()
    figs["fig1"] = save(fig, "fig1_acc_energy")
    num("pareto.front", {("test_S3" if k == "f1_S3" else "ext"): v for k, v in PARETO.items()}, "",
        "latency/energy 항목 (datasheet) × acc 항목 (시드 평균)", "다른 점에 지배되지 않는 구성 (에너지 ≤ 이고 정확도 ≥ 인 다른 점이 없다), 에너지 오름차순", "점 추정 기준")

    # 그림 2 — 판단 1회 지연 분해
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.4), gridspec_kw={"width_ratios": [5.6, 1.6]})   # 구성 7개 — 눈금 이름이 겹치지 않게
    segs = [("pre", "CPU 전처리", "white", "////"), ("load", "입력 적재(+출력)", "0.75", ""), ("infer", "NPU 추론", "0.35", "")]
    xs = np.arange(len(names))
    for ax, items, labels in ((ax1, [LAT[n] for n in names], [DS[n] + ("\n(추정)" if est[n] else "") for n in names]), (ax2, [LAT_M4], [M4S])):
        bottom = np.zeros(len(items))
        for key, lab, col, hatch in segs:
            vals = np.array([(L["load"] + L["out"]) if key == "load" else L[key] for L in items])
            if ax is ax2 and key == "infer":
                lab = "CPU 추론"
            ax.bar(np.arange(len(items)), vals, bottom=bottom, color=col, edgecolor="black", hatch=hatch, lw=0.8, width=0.6, label=lab)
            bottom += vals
        for i, L in enumerate(items):
            if ax is ax2:      # CPU 추론 막대 — 숫자는 막대 안 위쪽에 (판단 주기 점선과 겹치지 않게)
                ax.text(i, bottom[i] * 0.98, f"{L['total']:.2f}\n(NPU 대비 ×{NUMBERS['ratio.decision_m4cmsis_over_npu']['value']:.0f})",
                        ha="center", va="top", fontsize=8, color="white")
            else:
                ax.text(i, bottom[i], f"{L['total']:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(np.arange(len(items)))
        ax.set_xticklabels(labels, fontsize=6.8)
        ax.grid(True, axis="y", lw=0.3, color="0.85")
    ax1.set_ylabel("판단 1회 지연 (ms)")
    ax2.set_ylabel("판단 1회 지연 (ms)")
    ax2.axhline(hop_ms, color="black", ls="--", lw=0.8)
    ax2.text(0, hop_ms, f"판단 주기 {hop_ms:.0f} ms", ha="center", va="bottom", fontsize=7)
    ax2.set_ylim(0, hop_ms * 1.15)
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1, l1, loc="upper left", frameon=False)
    fig.tight_layout()
    figs["fig2"] = save(fig, "fig2_latency_breakdown")

    # 그림 3 — 클래스별 재현율 (시험 데이터 / 외부 데이터) + 대형 모델 점선. 논문판(3구성) 과 부록판(전체)
    def fig3(sel, fname):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
        width = 0.8 / len(sel)
        for ax, key, title, pr in ((axes[0], "rec_S3", TR["testL_S3"], pt["recall"]), (axes[1], "ext_rec", TR["real31L"], p_ext["recall"][:4])):
            for j, name in enumerate(sel):
                R = ACC[name][key]
                mu, sd = R.mean(axis=0), (R.std(axis=0, ddof=1) if len(R) > 1 else np.zeros(4))
                ax.bar(np.arange(4) + (j - (len(sel) - 1) / 2) * width, 100 * mu, width=width, yerr=100 * sd, capsize=1.5,
                       color=str(0.3 + 0.55 * j / max(1, len(sel) - 1)), edgecolor="black", hatch=HATCH[j % len(HATCH)], lw=0.6, label=DN[name])
            for k in range(4):
                ax.hlines(100 * pr[k], k - 0.45, k + 0.45, colors="black", linestyles="--", lw=1.2, label=PN if k == 0 else None)
            ax.set_xticks(np.arange(4))
            ax.set_xticklabels(CLS_KO)
            ax.set_title(title, fontsize=9)
            ax.grid(True, axis="y", lw=0.3, color="0.85")
            ax.set_ylim(0, 100)
        axes[0].set_ylabel("재현율 (%)")
        if len(sel) <= 3:
            axes[1].legend(loc="upper right", frameon=False, ncol=1, fontsize=7)
            fig.tight_layout()
        else:                                   # 부록 전체판 — 범례를 그림 아래에
            h, l = axes[1].get_legend_handles_labels()
            fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
            fig.tight_layout(rect=(0, 0.1, 1, 1))
        return save(fig, fname)

    figs["fig3"] = fig3(C["fig3_paper_configs"], "fig3_recall_by_class")
    figs["fig3_all"] = fig3(names, "fig3_recall_by_class_all")

    # 그림 4 — 데이터·정답 개선 단계별 외부 F1
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    eds = C["editions_for_fig4"]
    ED = {}
    for name, mk, ls in zip(C["fig4_configs"], ("o", "D"), ("-", "--")):
        key = next(c["v4eval"] for c in C["configs"] if c["name"] == name)
        mus, sds, ns = [], [], []
        for ed in eds:
            seeds = v4e["editions"][ed][key]["seeds"]
            mu, sd = mean_sd([s["ext_f1_real31L"] for s in seeds])
            mus.append(mu); sds.append(sd); ns.append(len(seeds))
        ED[name] = {ed: {"mean": m, "sd": s, "n": n} for ed, m, s, n in zip(eds, mus, sds, ns)}
        ax.errorbar(np.arange(len(eds)), mus, yerr=sds, fmt=mk + ls, color=GC[DG[name]], mfc="white" if mk == "D" else GC[DG[name]],
                    capsize=2, lw=1, label=DN[name])
    num("editions.ext_f1_real31L", ED, "", src_v4e, "editions[판][구성].seeds[*].ext_f1_real31L — 전부 real31L 로 다시 채점한 값")
    ax.set_xticks(np.arange(len(eds)))
    ax.set_xticklabels([EDN[e] for e in eds], fontsize=8)
    ax.set_xlabel("데이터 개선 단계 (누적)")
    ax.set_ylabel(f"F1 — {TR['real31L']}")
    ax.grid(True, lw=0.3, color="0.85")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    figs["fig4"] = save(fig, "fig4_edition_ext_f1")

    # ── 표
    tables = {}
    sr = MF.SR
    hdr1 = ["구성", "전처리 위치", "프레임 간격 (샘플)", "프레임 간격 (ms)", "n_fft", "멜 수", "프레임 수", "판단당 새 프레임", "모델", "모델 요약",
            "합성 층 수", "MAC", "가중치 (KB)", "가중치 메모리 비율 (%)", "NPU 사이클 (합성)", "출처"]
    rows1 = []
    for c in C["configs"]:
        L = LAT[c["name"]]; syn = L["synth"]
        if c["framing"]:
            hop, nfr, nmel = MF.INC_SPECS[c["framing"]]
            nfft = MF.inc_nfft(c["framing"])
            new_frames = int(round(hop_ms / 1000 * sr / hop))
            fr = (hop, round(hop / sr * 1e3, 2), nfft, nmel, nfr, new_frames)
        else:
            fr = ("—", "—", "—", "—", "—", "—")
        rows1.append([DN[c["name"]], c["preproc"], *fr, c["model"], c["model_desc"], syn["layers"], syn["macc"],
                      round(syn["weight_bytes"] / 1024, 1), round(100 * syn["weight_bytes"] / syn["weight_total"], 1), syn["cycles"],
                      f"{syn['dir']}/log.txt·cnn.h; datasets/melfeat.py INC_SPECS"])
    rows1.append([M4N, "없음", "—", "—", "—", "—", "—", "—", f"CMSIS-NN ({DN['④']} 가중치)", C["m4"]["desc"], LAT["④"]["synth"]["layers"],
                  LAT["④"]["synth"]["macc"], round(LAT["④"]["synth"]["weight_bytes"] / 1024, 1), "— (Flash)", "—", LAT_M4["src"]])
    tables["table1_configs"] = write_table("table1_configs", hdr1, rows1,
                                           ["표 1 구성 정의. 프레임 간격·n_fft·멜 수는 datasets/melfeat.py, MAC·가중치·사이클은 ai8x-synthesis 로그 (data/synth/out/*_v4d).",
                                            f"프레임 간격 ms = 샘플 ÷ {sr} Hz. 판단당 새 프레임 = hop {hop_ms:.0f} ms 동안 들어오는 프레임 수."])

    hdr2 = ["구성", "n", "CPU 전처리 (ms)", "입력 적재 (ms)", "추론 (ms)", "출력 (ms)", "판단 1회 (ms)", "판단 주기 대비 (%)", "지연 출처"] + \
           [f"에너지 {PS[k]['label']} (µJ, 추정)" for k in curve_src] + ["에너지 범위 (µJ, 추정)",
            f"macro-F1 {TR['testL_S3']} (주)", f"macro-F1 {TR['testL']}", f"F1 {TR['real31L']}", f"mAP {TR['real31L']}",
            f"재현율 {TR['testL_S3']} 사이렌/유리/비명/개 (%)", f"재현율 {TR['real31L']} 사이렌/유리/비명/개 (%)", "외부 배경 264창 발화 (창)", "배경+배경 발화율 (%)",
            "파레토 앞면 (시험 / 외부)"]
    rows2 = []

    def pareto_cell(name):
        return " / ".join("○" if name in PARETO[k] else "—" for k in ("f1_S3", "ext_f1"))

    def acc_cells(name):
        A = ACC[name]
        rs = " / ".join(f"{100 * v:.1f}" for v in A["rec_S3"].mean(axis=0))
        re_ = " / ".join(f"{100 * v:.1f}" for v in A["ext_rec"].mean(axis=0))
        return [fmt_ms(*mean_sd(A["f1_S3"])), fmt_ms(*mean_sd(A["f1_L"])), fmt_ms(*mean_sd(A["ext_f1"])), fmt_ms(*mean_sd(A["ext_map"])),
                rs, re_, f"{np.mean(A['ext_fire_n']):.1f}", f"{100 * np.mean(A['bgbg']):.2f}"]

    for c in C["configs"]:
        L = LAT[c["name"]]
        es = [L["E"][k] for k in curve_src]
        tag = " (추정)" if L["est"] else ""
        rows2.append([DN[c["name"]], ACC[c["name"]]["n"], f"{L['pre']:.3f}" if L["pre"] else "—", f"{L['load']:.3f}", f"{L['infer']:.3f}{tag}", f"{L['out']:.3f}",
                      f"{L['total']:.2f}{tag}", f"{100 * L['total'] / hop_ms:.1f}", L["src"]] + [f"{e:.0f}" for e in es] +
                     [f"{min(es):.0f} ~ {max(es):.0f}"] + acc_cells(c["name"]) + [pareto_cell(c["name"])])
    es = [LAT_M4["E"][k] for k in curve_src]
    same = f"{DN['④']} 와 같다"
    rows2.append([M4N, ACC["④"]["n"], "—", f"{LAT_M4['load']:.3f}", f"{LAT_M4['infer']:.2f}", "—", f"{LAT_M4['total']:.2f}",
                  f"{100 * LAT_M4['total'] / hop_ms:.1f}", LAT_M4["src"]] + [f"{e:.0f}" for e in es] + [f"{min(es):.0f} ~ {max(es):.0f}"] +
                 [same + " (같은 가중치·출력)"] * 4 + [same] * 4 + ["—"])
    tables["table2_main"] = write_table("table2_main", hdr2, rows2,
                                        [f"표 2 주 결과. 데이터 v4+D, 시드 1~3 평균 ± 표준편차. 정답: {TR['testL_S3']} (주; 청취 전 정답은 부록 열), {TR['real31L']}. 오경보 300회/h (시험 데이터 배경에서 잡은 문턱값, 외부에도 같은 문턱값).",
                                         "지연: MAX78000FTHR, v4+D 시드 1 체크포인트, POR 직후, 1000회 최악값, 코어 100 MHz / CNN 50 MHz, 바쁜 대기, 슬립·유휴·가중치 적재 미포함.",
                                         "에너지: " + C["energy_note"] + " 전력값·출처는 docs/paper/paper_config.json.",
                                         f"{DN['(a)']} 는 보드 미실측 — 적재·출력은 {DN['④']} 실측, 추론은 합성 사이클 ÷ CNN 클럭 (추정).",
                                         f"{DN['②′']} 는 두 네트워크(NPU 필터뱅크 p1 + 2D CNN p2)를 동시 적재한 보드 실측 — CPU 전처리 열 = p1 출력의 |cos|+|sin|·8위치 합·정수 로그 (NPU 전원 켜진 채 → 에너지는 P_load 로), "
                                         f"입력 적재 열 = 두 적재 + 네트워크 전환(층 레지스터 재설정), 추론 열 = p1 + p2. 판단 주기 256 ms (다른 구성 250 ms). 구간별 값은 numbers.json latency.②′.detail.",
                                         "파레토 앞면: 점 추정(데이터시트 에너지, 시드 평균) 기준으로 에너지 ≤ 이고 정확도 ≥ 인 다른 구성이 없는 점."])

    hdr3 = ["비교 (앞 − 뒤)", "n", f"macro-F1 {TR['testL_S3']} (주)", f"F1 {TR['real31L']}", f"mAP {TR['real31L']}", f"macro-F1 {TR['testL']}"]
    rows3 = []

    def fci(t, nd=3):
        d, lo, hi = t
        return f"{d:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"

    for (a_name, b_name), row in WEL.items():
        rows3.append([f"{DN[a_name]} − {DN[b_name]}", f"{ACC[a_name]['n']} / {ACC[b_name]['n']}", fci(row["test_f1_S3"]), fci(row["ext_f1"]), fci(row["ext_map"]), fci(row["test_f1_L"])])
    tables["table3_welch"] = write_table("table3_welch", hdr3, rows3,
                                         ["표 3 핵심 비교. 같은 시드 1~3, Welch 95% CI (tools/eval_v2.welch). CI 가 0 을 포함하면 '못 가른다'.",
                                          f"{DN['(a)']} 는 주 실행 (lr 0.0003 + 미세조정 스케줄) — 보조 실행은 판정에 쓰지 않는다 (STATUS.md)."])

    hdr4 = ["구간", f"NPU 추론 — {DN['④']} (ms)", f"{M4N} (ms)", "배율 (CPU / NPU)", "판단 주기 대비 NPU (%)", "판단 주기 대비 CPU (%)", "출처"]
    rows4 = [["추론만", f"{L4['infer']:.3f}", f"{LAT_M4['infer']:.2f}", f"{LAT_M4['infer'] / L4['infer']:.1f}", f"{100 * L4['infer'] / hop_ms:.1f}",
              f"{100 * LAT_M4['infer'] / hop_ms:.1f}", f"{L4['src']}, {LAT_M4['src']}"],
             ["판단 1회 (적재 + 추론 + 출력)", f"{L4['total']:.2f}", f"{LAT_M4['total']:.2f}", f"{LAT_M4['total'] / L4['total']:.1f}",
              f"{100 * L4['total'] / hop_ms:.1f}", f"{100 * LAT_M4['total'] / hop_ms:.1f}", "〃"]]
    for k in curve_src:
        rows4.append([f"판단 1회 에너지 {PS[k]['label']} (µJ, 추정)", f"{L4['E'][k]:.0f}", f"{LAT_M4['E'][k]:.0f}", f"{LAT_M4['E'][k] / L4['E'][k]:.0f}", "—", "—",
                      f"{rel(CFG_PATH)} power_sources.{k}"])
    tables["table4_npu_vs_cmsis"] = write_table("table4_npu_vs_cmsis", hdr4, rows4,
                                                [f"표 4 NPU 추론 대 {M4N}. 같은 가중치({DN['④']})·같은 출력 (보드 NPU 출력과 비트 일치, 1000회 불일치 0). 1000회 최악값.",
                                                 "CMSIS-NN: arm_convolve_wrapper_s8 (SINGLE_ROUNDING), -O2, 코어 100 MHz, NPU 꺼짐, TMR0 1 µs. 에너지는 추정(실측 지연 × 공개 전력)."])

    # 표 5 — 손익분기 전력비. E = C·P_cpu + N·P_npu (C = 전처리 + 적재 + 출력 시간, N = NPU 추론 시간; P_load ≈ P_cpu 로 둔 단순화).
    #   X 가 Y 보다 에너지가 적다 ⇔ (C_X − C_Y) + (N_X − N_Y)·r < 0,  r = P_npu / P_cpu.
    def cn(L, m4=False):
        return (L["load"] + L["infer"], 0.0) if m4 else (L["pre"] + L["load"] + L["out"], L["infer"])

    hdr5 = ["비교 (X 가 Y 보다 에너지가 적다)", "C_X (ms)", "N_X (ms)", "C_Y (ms)", "N_Y (ms)", "성립 조건 (r = P_npu / P_cpu)", "손익분기 r*"] + \
           [f"{PS[k]['label']}: r = {PS[k]['P_npu_mW']}/{PS[k]['P_cpu_mW']} → 성립?" for k in PS] + ["비고"]
    pairs5 = [(f"{DN['④']} (NPU 추론)", LAT["④"], False, M4N, LAT_M4, True, "NPU 추론이 CPU 추론보다 에너지가 적은 조건"),
              (DN["가 u1000"], LAT["가 u1000"], False, DN["④"], LAT["④"], False, "CPU 전처리 + 작은 NPU 가 원본 파형 NPU 보다 적은 조건"),
              (DN["다 u800"], LAT["다 u800"], False, DN["④"], LAT["④"], False, "〃"),
              (DN["간격 400"], LAT["간격 400"], False, DN["①′"], LAT["①′"], False, "전처리·추론 둘 다 짧다"),
              (DN["(a)"], LAT["(a)"], False, DN["④"], LAT["④"], False, f"{DN['(a)']} 지연은 추정 (보드 미실측)"),
              (DN["②′"], LAT["②′"], False, DN["④"], LAT["④"], False, "선형 전처리를 NPU 로 옮긴 구성이 원본 파형 NPU 보다 적은 조건 (보드 실측)"),
              (DN["②′"], LAT["②′"], False, DN["가 u1000"], LAT["가 u1000"], False, "선형 전처리를 NPU 로 옮긴 구성이 CPU log-mel 보다 적은 조건 (보드 실측)")]
    rows5 = []
    BRK = {}
    for xn, X, xm4, yn, Y, ym4, note in pairs5:
        cx, nx = cn(X, xm4); cy, ny = cn(Y, ym4)
        dc, dn = cx - cy, nx - ny
        if abs(dn) < 1e-12:
            cond, rstar = ("항상 성립" if dc < 0 else "성립하지 않음"), None
        elif dn < 0:
            rstar = dc / (-dn); cond = f"r > {rstar:.2f}" if rstar > 0 else "항상 성립 (r > 0 이면)"
        else:
            rstar = -dc / dn; cond = f"r < {rstar:.1f}" if rstar > 0 else "성립하지 않음 (r > 0 이면)"
        cells = []
        for k in PS:
            ex = energy(X, PS[k], m4=xm4); ey = energy(Y, PS[k], m4=ym4)
            cells.append(f"r = {PS[k]['P_npu_mW'] / PS[k]['P_cpu_mW']:.2f}; E_X {ex:.0f} / E_Y {ey:.0f} µJ → {'성립' if ex < ey else '불성립'}")
        rows5.append([f"{xn} < {yn}", f"{cx:.3f}", f"{nx:.3f}", f"{cy:.3f}", f"{ny:.3f}", cond,
                      f"{rstar:.2f}" if (rstar is not None and rstar > 0) else "— (r 와 무관)"] + cells + [note])
        BRK[f"{xn} < {yn}"] = {"C_X": cx, "N_X": nx, "C_Y": cy, "N_Y": ny, "condition": cond, "r_star": rstar,
                               "by_source": {k: {"r": PS[k]["P_npu_mW"] / PS[k]["P_cpu_mW"], "E_X": energy(X, PS[k], m4=xm4), "E_Y": energy(Y, PS[k], m4=ym4)} for k in PS}}
    num("breakeven", BRK, "", "latency 항목 × paper_config.json power_sources", "E = C·P_cpu + N·P_npu (P_load ≈ P_cpu 단순화); 성립 여부는 전력 상수 전부 넣은 값으로", "추정")
    tables["table5_breakeven"] = write_table("table5_breakeven", hdr5, rows5,
                                             ["표 5 손익분기 전력비. E = C·P_cpu + N·P_npu 로 두면 (C = CPU 쪽 시간 = 전처리 + 적재 + 출력, N = NPU 추론 시간, P_load ≈ P_cpu 단순화) 비교는 r = P_npu/P_cpu 하나로 정해진다.",
                                              "'성립?' 열은 단순화 없이 각 출처의 전력 상수(P_cpu / P_load / P_npu)를 전부 넣어 계산한 에너지(추정)로 판정했다. 전부 추정(실측 지연 × 공개 전력)."])

    # 표 6 — 청취 정답 요약
    hdr6 = ["세트", "그룹 (definition)", "대상 창", "O", "X", "?", "O 비율 (%)", "O 비율 Wilson 95% (%)", "대조 창", "대조 O", "대조 O 비율 (%)", "판정 파일", "비고"]
    rows6 = []
    LIS = {}

    def summarize(set_name, rows, vcol, path, note=""):
        tg = [r for r in rows if r["set"] == set_name and r["role"] == "target"]
        ct = [r for r in rows if r["set"] == set_name and r["role"] == "contrast"]
        cO = sum(r[vcol] == "O" for r in ct)
        groups = [("전체", tg)] + [(g, [r for r in tg if r["definition"] == g]) for g in sorted({r["definition"] for r in tg}) if g]
        for g, rs in groups:
            if g != "전체" and len(groups) == 2:
                continue
            o, x, q = (sum(r[vcol] == v for r in rs) for v in ("O", "X", "?"))
            lo, hi = wilson(o, len(rs))
            rows6.append([set_name, g, len(rs), o, x, q, f"{100 * o / len(rs):.1f}", f"[{100 * lo:.1f}, {100 * hi:.1f}]",
                          len(ct) if g == "전체" else "", cO if g == "전체" else "", f"{100 * cO / len(ct):.0f}" if (ct and g == "전체") else "", rel(path), note])
            LIS[f"{set_name}/{g}" + (f"/{vcol}" if vcol != "verdict" else "")] = {"n": len(rs), "O": o, "X": x, "?": q, "contrast_n": len(ct), "contrast_O": cO}

    pL, pS, pT = (os.path.join(REPO, n) for n in ("data_listened.csv", "data_listened_scream.csv", "data_listened_test_scream.csv"))
    L_rows, S_rows, T_rows = load_csv(pL), load_csv(pS), load_csv(pT)
    summarize("A_ext_scream", L_rows, "verdict", pL, "외부 비명 (정의별 그룹: v1만 / 실사용만 / v1+실사용 / v1=실사용)")
    summarize("B_ext_glass", L_rows, "verdict", pL, "외부 유리 — 재청취 판정 (주 정답)")
    summarize("B_ext_glass", L_rows, "verdict_first", pL, "외부 유리 — 첫 판정 (민감도 real31L_Bfirst)")
    summarize("C_test_glass", L_rows, "verdict", pL, "시험셋 유리 221창 → testL (O 만 양성)")
    summarize("S_train_scream", S_rows, "verdict", pS, "학습 비명 (yellshout_train·audioset_train·screaming_train) + 시험셋 Yell·Shout 만 66 (yellshout_test → testL-S2/S3)")
    summarize("T_test_scream", T_rows, "verdict", pT, "시험셋 Screaming 보유 159 (→ testL-S3) + 학습 Screaming·Yell/Shout 동반 30 (→ exclude_v5)")
    num("listening", LIS, "창", f"{rel(pL)}, {rel(pS)}, {rel(pT)}", "set/role/definition/verdict 열 집계 (대조 창 = role contrast, PANNs ≥ 0.50 학습 창, 일관성 점검용)")
    tables["table6_listening"] = write_table("table6_listening", hdr6, rows6,
                                             ["표 6 청취 정답 요약. 판정자 1인, O 비명(또는 유리 파손) / X 아님 / ? 판단 불가. O 만 양성, X·? 는 채점 제외 (배경으로 옮기지 않음). PANNs 점수는 정답에 쓰지 않았다.",
                                              "대조 창 = PANNs ≥ 0.50 인 학습 창 (세트마다 15, 무작위 번호로 섞음) — 청취자 일관성 점검용이며 정답·제외에 쓰지 않는다. 오디오는 저장소에 없다 (창 ID·판정만)."])

    xlsx = write_xlsx(tables)

    # ── 캡션 (그림 안에는 긴 설명을 두지 않는다)
    n_seed = ACC["④"]["n"]
    cond = (f"지연은 MAX78000FTHR 에서 v4+D 시드 1 가중치로 1000회 측정한 최악값 (전원 투입 직후, 코어 100 MHz / CNN 50 MHz, 바쁜 대기; 슬립·유휴·가중치 적재 미포함). "
            f"에너지는 전부 추정(실측 지연 × 공개 전력)이며 측정값이 아니다 — E = 전처리 시간·P_cpu + (적재 + 출력)·P_load + 추론·P_npu, 전력값과 출처는 paper_config.json.")
    acc_cond = (f"정확도는 데이터 v4+D, 시드 1~{n_seed} 평균 (막대 = 시드 표준편차), 오경보 300회/h 가 되는 문턱값을 {TR['testL_S3']} 배경에서 잡고 {TR['real31L']} 에도 같은 문턱값을 썼다. "
                f"{TR['testL_S3']} 정답은 사람 청취로 확정한 것(유리 70창·비명 134창 양성, 채점 창 {truth['testL_S3_n']:,}), {TR['real31L']} 정답도 청취 확정 (비명 양성 {truth['ext_scream_n']}, 배경 구간 264창).")
    cap = [
        "# 그림 캡션 (tools/make_paper_assets.py 가 생성 — 그림 안의 설명은 전부 여기로 옮겼다)", "",
        f"## 그림 1 — 판단 1회 추정 에너지와 정확도 (`{figs['fig1'][0]}`)", "",
        f"왼쪽은 {TR['testL_S3']} 의 macro-F1, 오른쪽은 {TR['real31L']} 의 F1 이다. 색은 전처리 위치 — 검정 {GL['none']}, 주황 {GL['npu']}, 파랑 {GL['cpu']} — 이고 "
        f"흑백 인쇄를 위해 마커 모양도 구성마다 다르다. 마커 위치는 {PS['datasheet']['label']} 전력 기준 에너지, 가로 막대는 공개 전력값 범위 "
        f"({' / '.join(PS[k]['label'] for k in curve_src)}), 세로 막대는 시드 표준편차 (n={n_seed}). CPU log-mel 네 점은 프레임 간격 (62.5 / 50 / 25 / 16 ms) 을 "
        f"에너지 순으로 이었다. 별표(*)가 붙은 {DN['(a)']} 는 보드에서 재지 않았고 지연을 합성 사이클로 추정한 점이다 (빈 마커). "
        f"{DN['②′']} 는 선형 전처리(512탭 Gabor 필터뱅크 + 프레임 안 에너지 적분)를 NPU 로 옮기고 정수 로그만 CPU 에 남긴 구성으로, 두 네트워크를 동시 적재한 보드 실측이다 "
        f"(CPU 구간은 NPU 전원이 켜진 채 돌아 P_load 로 계산). 큰 고리와 점선은 파레토 앞면 — 점 추정(데이터시트 에너지, 시드 평균) 기준으로 에너지가 같거나 작으면서 "
        f"정확도가 같거나 높은 다른 점이 없는 구성 (시험 데이터: {', '.join(DN[n] for n in PARETO['f1_S3'])}; 외부 데이터: {', '.join(DN[n] for n in PARETO['ext_f1'])}). {acc_cond} {cond}", "",
        f"## 그림 2 — 판단 1회 지연 분해 (`{figs['fig2'][0]}`)", "",
        f"왼쪽은 NPU 로 추론하는 {len(names)}개 구성의 판단 1회 지연을 CPU 전처리 / 입력 적재(+출력) / NPU 추론으로 나눈 것이다 (막대 위 숫자 = 합계 ms). "
        f"{DN['②′']} 의 CPU 전처리는 NPU 필터뱅크 출력의 |cos|+|sin|·8위치 합·정수 로그이고, 입력 적재에는 두 네트워크 전환(층 레지스터 재설정)이 들어 있다. "
        f"{DN['(a)']} 는 추정 (적재·출력은 {DN['④']} 실측, 추론은 합성 사이클 ÷ CNN 클럭). 오른쪽은 같은 가중치를 CMSIS-NN 으로 Cortex-M4F 에서 돌린 "
        f"{M4N} 으로, 축이 다르며 점선은 판단 주기 {hop_ms:.0f} ms 다 ({M4N} 은 주기의 {100 * LAT_M4['total'] / hop_ms:.0f}% 를 추론에 쓴다). {cond}", "",
        f"## 그림 3 — 클래스별 재현율 (`{figs['fig3'][0]}`, 부록 전체판 `{figs['fig3_all'][0]}`)", "",
        f"왼쪽 {TR['testL_S3']}, 오른쪽 {TR['real31L']}. 막대는 구성별 시드 평균 ± 표준편차, 점선은 {PN} (CNN14 zero-shot) 을 같은 정답·같은 문턱값 규칙으로 "
        f"채점한 기준선이다 (시험 F1 {pt['f1']:.3f}, 외부 F1 {p_ext['f1']:.3f}). 논문판은 {', '.join(DN[n] for n in C['fig3_paper_configs'])} 세 구성, "
        f"부록판은 {len(names)}개 구성 전부. {acc_cond}", "",
        f"## 그림 4 — 데이터 개선 단계별 외부 데이터 F1 (`{figs['fig4'][0]}`)", "",
        f"학습 데이터를 단계적으로 고치면서 ({' → '.join(EDN[e] for e in eds)}, 누적) {DN['④']} 과 {DN['간격 400']} 두 구성을 각 단계에서 3시드씩 학습하고, "
        f"모든 단계를 **같은** {TR['real31L']} 와 같은 청취 확정 정답으로 다시 채점한 F1 이다 (문턱값은 각 단계의 시험 데이터 배경 300회/h). "
        f"막대 = 시드 표준편차. 단계 설명:", "",
        "- 초기 데이터: FSD50K·UrbanSound8K·ESC-50을 칩 입력 규격(16 kHz, int8)으로 재가공",
        "- +배경 혼합: 학습 시 이벤트 창과 배경 창 모두에 다른 배경음을 SNR 0~20 dB로 섞는 증강",
        "- +AudioSet 추가: AudioSet 학습 분할의 실제 환경 녹음 구간 추가(비명은 Screaming 구간만)",
        "- +라벨 정정: 청취로 대부분 틀린 것이 확인된 유형의 학습 창 제외(유리·비명 중 대형 모델 점수가 매우 낮은 창)",
        "- +증류: 대형 모델(PANNs) 기반 교사의 출력을 라벨 없는 창 약 9.4만 개를 포함해 함께 학습", "",
    ]
    with open(os.path.join(OUT, "captions.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(cap))

    # ── numbers.json
    NUMBERS["_meta"] = {"generated_by": "tools/make_paper_assets.py", "basis": "데이터 v4+D · 정답 testL-S3 / real31L · 지연 v4+D 시드 1 보드 실측 · 에너지 추정(실측 지연 × 공개 전력)",
                        "config": rel(CFG_PATH), "display_names": {"configs": DN, "m4": M4N, "truth": TR, "panns": PN, "editions": EDN},
                        "figures": figs, "captions": "docs/paper/captions.md", "tables": {k: f"docs/paper/{k}.csv" for k in tables}, "xlsx": xlsx, "font": fam}
    with open(os.path.join(OUT, "numbers.json"), "w", encoding="utf-8") as f:
        json.dump(NUMBERS, f, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    # ── 요약 출력
    print(f"## 생성 완료 → {rel(OUT)}/  (글꼴 {fam})")
    for k, v in figs.items():
        print(f"  {k}: {v[0]}  {v[1]}")
    print(f"  표: {', '.join(f'{k}.csv' for k in tables)} + {xlsx}")
    print(f"  numbers.json 항목 {len(NUMBERS) - 1}")
    print(f"\n정답: testL-S3 {truth['testL_S3_n']}창 (비명 {truth['test_scream_n']}, 유리 {kcnt['glass_O']}), real31L 비명 {truth['ext_scream_n']}")
    print(f"NPU/CMSIS: 추론 {NUMBERS['ratio.infer_m4cmsis_over_npu']['value']:.1f}배, 판단 {NUMBERS['ratio.decision_m4cmsis_over_npu']['value']:.1f}배; "
          f"전처리 {first_ms:.2f} → {LAT['①′']['pre']:.2f} ms")
    print("| 구성 | 판단 1회 (ms) | 에너지 (가)/(나)/(나′) µJ | F1 testL-S3 | F1 testL | 외부 F1 | 외부 mAP |")
    for c in C["configs"]:
        L = LAT[c["name"]]; A = ACC[c["name"]]
        es_str = " / ".join(f"{L['E'][k]:.0f}" for k in curve_src)
        tag = " (추정)" if L["est"] else ""
        print(f"| {c['name']} | {L['total']:.2f}{tag} | {es_str} | "
              f"{fmt_ms(*mean_sd(A['f1_S3']))} | {fmt_ms(*mean_sd(A['f1_L']))} | {fmt_ms(*mean_sd(A['ext_f1']))} | {fmt_ms(*mean_sd(A['ext_map']))} |")
    es_str = " / ".join(f"{LAT_M4['E'][k]:.0f}" for k in curve_src)
    print(f"| {C['m4']['name']} | {LAT_M4['total']:.2f} | {es_str} | = ④ | = ④ | = ④ | = ④ |")
    print(f"PANNs testL-S3 F1 {pt['f1']:.3f} recall {[round(100 * v, 1) for v in pt['recall']]}; 외부 real31L(test300) F1 {p_ext['f1']:.3f}")


if __name__ == "__main__":
    main()
