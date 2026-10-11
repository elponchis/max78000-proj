#!/usr/bin/env python3
"""청취 판정 → 시험 정답 확정 (`eval-baseline-change.md` 2·4절, `v4-distill-design.md` 10.3).

입력  ~/safesound-external/audit_test/key.csv                       세트·파일 → 출처·창 (저장소 밖)
      ~/safesound-external/audit_test/verdict_<세트>.csv              사용자 판정 (file, verdict, memo)
출력  data_listened.csv (레포 루트, **커밋** — 창 ID 와 판정만, 오디오 없음)

규칙 (사전 등록): O → 양성, X → 그 클래스에서 제외 (배경으로 옮기지 않는다), ? → 평가 제외.
PANNs 점수는 정답에 쓰지 않는다. 대조 창(PANNs ≥ 0.50 인 학습 창 15/세트)은 판정 기준 점검용으로 기록만 한다.
정의별 양성 조건(v1eq / real31)은 유지하고 청취 판정은 **그 정의의 양성 창 위에만** 겹친다 — 정의상 음성(배경)인
창은 판정과 무관하게 음성으로 남는다 (2.2절 "군중 동반 비명"). 아기 울음 구간은 비명 양성에서 빼고 배경 (2.2절).

사용:  python tools/listened_labels.py --build            # 판정 파일 대조 → data_listened.csv + 분포 표
"""
import argparse
import collections
import csv
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "tools")]
AUDIT = os.path.expanduser("~/safesound-external/audit_test")
LISTENED = os.path.join(REPO, "data_listened.csv")
SETS = ("A_ext_scream", "B_ext_glass", "C_test_glass")
VERDICTS = ("O", "X", "?")
BABY_CRY_EXT = ("g9Qah25_yH0_14000",)          # 외부 세트 아기 울음 구간 → 배경 (eval-baseline-change.md 2.2)
RELISTENED = "B_ext_glass"                      # 재청취한 세트 — 첫 판정은 verdict_first 열에 민감도 분석용으로만 보관
COLUMNS = ["set", "file", "role", "source", "cls", "definition", "clip", "start_sample", "verdict", "verdict_first", "memo"]
HEADER = ["# 청취로 확정한 시험 정답 (2026-10-11). 규칙: O 양성 / X 그 클래스에서 제외(배경 아님) / ? 평가 제외. PANNs 점수는 정답에 쓰지 않았다.",
          "# 출처 ext = 외부 세트(AudioSet strong eval) 창, v1_test = 기존 시험셋 glass 창, v1_train = 대조 창(학습, 정답에 안 씀 — 청취자 일관성 점검용).",
          "# verdict = 주 정답. verdict_first = B_ext_glass 의 첫 청취 판정(00:41) — 재청취(13:11) 전 값, 민감도 분석에만 쓴다 (label-audit.md 7.3).",
          "# 근거: docs/results/eval-baseline-change.md 2절, docs/results/label-audit.md 7절. 오디오는 저장소에 없다."]


def _norm_verdict(v):
    v = v.strip().strip('"').strip().upper().replace("？", "?")
    return {"0": "O", "○": "O", "X": "X", "×": "X", "O": "O", "?": "?"}.get(v, v)


def read_verdicts(path):
    """file → (verdict, memo). BOM·CRLF·따옴표 허용."""
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            r = {k.strip().lower(): (v or "") for k, v in r.items() if k}
            fn = r.get("file", "").strip().strip('"')
            if not fn:
                continue
            if fn in out:
                sys.exit(f"[에러] 중복 파일 {fn} in {path}")
            out[fn] = (_norm_verdict(r.get("verdict", "")), r.get("memo", "").strip())
    return out


def build(audit_dir=AUDIT, out=LISTENED):
    with open(os.path.join(audit_dir, "key.csv"), encoding="utf-8") as f:
        key = list(csv.DictReader(f))
    rows = []
    for s in SETS:
        vd = read_verdicts(os.path.join(audit_dir, f"verdict_{s}.csv"))
        ks = [r for r in key if r["set"] == s]
        missing = [r["file"] for r in ks if r["file"] not in vd]
        extra = sorted(set(vd) - {r["file"] for r in ks})
        bad = [(fn, v) for fn, (v, _m) in vd.items() if v not in VERDICTS]
        if missing or extra or bad:
            sys.exit(f"[에러] {s}: 빠진 파일 {missing[:5]} ({len(missing)}), 모르는 파일 {extra[:5]}, 잘못된 판정 {bad[:5]}")
        first = {}
        if s == RELISTENED:
            first = read_verdicts(os.path.join(audit_dir, f"verdict_{s}_first.csv"))
            if set(first) != set(vd):
                sys.exit(f"[에러] {s}: 첫 판정 파일의 파일 목록이 다르다")
        for r in ks:
            v, m = vd[r["file"]]
            rows.append({c: r.get(c, "") for c in COLUMNS[:-3]} | {"verdict": v, "verdict_first": first.get(r["file"], ("",))[0], "memo": m})
    with open(out, "w", encoding="utf-8", newline="") as f:
        for h in HEADER:
            f.write(h + "\n")
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"→ {out}  ({len(rows)}행)")
    return rows


def load(path=LISTENED, b_first=False):
    """b_first=True 면 재청취 세트(B)의 verdict 를 첫 판정(verdict_first)으로 바꿔 돌려준다 — 민감도 분석용."""
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    rows = list(csv.DictReader(lines))
    if b_first:
        n = 0
        for r in rows:
            if r["set"] == RELISTENED and r.get("verdict_first"):
                r["verdict"] = r["verdict_first"]; n += 1
        assert n == sum(1 for r in rows if r["set"] == RELISTENED), "첫 판정이 비어 있는 B 창이 있다"
    return rows


def ext_meta():
    import eval_external as E
    with open(os.path.join(E.EXT, "meta.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summary(rows, M=None):
    """1) 대조 창 O 수, 2) 대상 창 판정 분포 (세트·출처·정의·하위 라벨). 표를 출력하고 dict 로 돌려준다."""
    M = M or ext_meta()
    meta = {r["segment_id"]: r for r in M}
    rep = {"contrast": {}, "target": {}}
    print("\n## 대조 창 (PANNs ≥ 0.50 학습 창) 판정 — 기준 점검")
    print("| 세트 | 대조 창 | O | X | ? | O 비율 |")
    print("|---|---:|---:|---:|---:|---:|")
    for s in SETS:
        c = collections.Counter(r["verdict"] for r in rows if r["set"] == s and r["role"] == "contrast")
        n = sum(c.values())
        rep["contrast"][s] = dict(n=n, O=c["O"], X=c["X"], q=c["?"])
        print(f"| {s} | {n} | {c['O']} | {c['X']} | {c['?']} | {100 * c['O'] / max(1, n):.0f}% |")

    def sub_scream(r):
        m = meta[r["clip"]]
        labs = set(m["labels_all"].split(";"))
        core = "Screaming" if "Screaming" in labs else "Yell·Shout 만"
        crowd = "Screaming" in labs and m["in_v1"] == "0"
        return r["definition"], core, "군중 동반" if crowd else "—"

    print("\n## 대상 창 판정 분포")
    print("| 세트 | 구분 | 창 | O | X | ? | O 비율 |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for s in SETS:
        tg = [r for r in rows if r["set"] == s and r["role"] == "target"]
        groups = collections.OrderedDict()
        if s == "A_ext_scream":
            for r in tg:
                d, core, crowd = sub_scream(r)
                groups.setdefault((f"{d} / {core} / {crowd}"), []).append(r)
            # 정의별 합계도
            for d in ("v1만", "실사용만", "v1+실사용"):
                groups[f"{d} 합계"] = [r for r in tg if r["definition"] == d]
            groups["Screaming 라벨 보유 합계"] = [r for r in tg if "Screaming" in meta[r["clip"]]["labels_all"].split(";")]
            groups["Yell·Shout 만 합계"] = [r for r in tg if "Screaming" not in meta[r["clip"]]["labels_all"].split(";")]
            groups["군중 동반 합계"] = [r for r in tg if sub_scream(r)[2] == "군중 동반"]
        else:
            groups["전체"] = tg
        rep["target"][s] = {}
        for g, rs in groups.items():
            c = collections.Counter(r["verdict"] for r in rs)
            n = len(rs)
            rep["target"][s][g] = dict(n=n, O=c["O"], X=c["X"], q=c["?"])
            print(f"| {s} | {g} | {n} | {c['O']} | {c['X']} | {c['?']} | {100 * c['O'] / max(1, n):.0f}% |")
    return rep


def ext_overlay(M, y, rows, definition_name=""):
    """정의별 정답 y (eval_external.labels) 위에 청취 판정을 겹친다. 양성 창만 바뀐다: O 유지, X·? → −1.
    아기 울음 구간은 양성이면 배경. 반환 (y_new, 통계)."""
    idx = {r["segment_id"]: i for i, r in enumerate(M)}
    y = y.copy()
    st = collections.Counter()
    for r in rows:
        if r["source"] != "ext" or r["role"] != "target":
            continue
        i = idx[r["clip"]]
        if r["clip"] in BABY_CRY_EXT:
            if y[i] >= 0 and y[i] != 4:
                y[i] = 4
                st["baby_cry→bg"] += 1
            continue
        if 0 <= y[i] < 4:
            if r["verdict"] == "O":
                st["pos_O"] += 1
            else:
                y[i] = -1
                st["pos_X_excl" if r["verdict"] == "X" else "pos_?_excl"] += 1
        elif y[i] == 4:
            st["neg_O_kept_bg" if r["verdict"] == "O" else "neg_kept_bg"] += 1
        else:
            st["not_in_def"] += 1
    return y, dict(st)


def test_keep(rows, clips, starts, y):
    """기존 시험셋 창 순서(clips, starts, y)에 대한 채점 포함 마스크. glass 창 중 X·? 는 제외."""
    pos = {}
    for i, (c, s, yy) in enumerate(zip(clips, starts, y)):
        if yy == 1:
            pos[(str(c), int(s))] = i
    keep = np.ones(len(y), dtype=bool)
    n = collections.Counter()
    for r in rows:
        if r["source"] != "v1_test" or r["role"] != "target":
            continue
        i = pos.get((r["clip"], int(r["start_sample"])))
        if i is None:
            sys.exit(f"[에러] 시험셋 glass 창을 찾지 못했다: {r['clip']} {r['start_sample']}")
        n[r["verdict"]] += 1
        if r["verdict"] != "O":
            keep[i] = False
    assert sum(n.values()) == int((y == 1).sum()), (n, int((y == 1).sum()))
    return keep, dict(n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--audit-dir", default=AUDIT)
    a = ap.parse_args()
    rows = build(a.audit_dir) if a.build else load()
    summary(rows)
