#!/usr/bin/env python3
"""시험용 정답 확정 청취 세트 (`label-audit.md` 7절) + 학습 정리 규칙의 제외 창 수 표.

세트 (파일명은 무작위 번호만, 정답·출처는 key.csv 에만):
    A_ext_scream   외부 세트 scream 그룹 중 v1 정의 또는 실사용 정의에서 양성인 창 전체 (+ 대조 15)
    B_ext_glass    외부 세트 glass 양성 창 전체 (+ 대조 15)
    C_test_glass   기존 시험셋 glass 창 전체 (+ 대조 15)
대조 창 = 같은 클래스에서 PANNs 점수 ≥ 0.50 인 **FSD50K 계열 학습** 창. 청취자가 "전부 의심스럽다" 고 가정하지
못하게 섞는다. 정답 판정(O/X/?)은 key.csv 의 verdict 열에 사용자가 적는다.

사용 (WSL2, ai8x venv):  python tools/build_audit_test.py [--out ~/safesound-external/audit_test]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "tools"), os.path.join(REPO, "scripts")]
import label_audit as LA                      # noqa: E402
import eval_external as E                     # noqa: E402

SEED = 78005
N_CONTRAST = 15
THR_EXCL = 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/safesound-external/audit_test"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    names = LA.label_names()
    M = LA.ext_meta()
    ext = LA.load("ext"); se = LA.class_scores(ext["P"], names)
    tr = LA.load("v1_train"); s_tr = LA.class_scores(tr["P"], names)
    as_ = LA.load("as_train"); s_as = LA.class_scores(as_["P"], names)
    zt = np.load(LA.TEST_CACHE, allow_pickle=True); s_te = LA.class_scores(zt["P"], names)
    y_eq, y_31 = E.labels(M, "v1eq"), E.labels(M, "real31")

    # ── 학습 정리 규칙: glass·scream 에서 PANNs 해당 클래스 점수 < 0.02 → 학습 제외 (출처별)
    print("\n## 학습 제외 창 수 (glass·scream, PANNs 점수 < 0.02)")
    print("| 출처 | 클래스 | 창 | 제외 | 남음 | 제외 비율 |")
    print("|---|---|---:|---:|---:|---:|")
    excl = {}
    for src, z, s in (("FSD50K 계열 학습", tr, s_tr), ("AudioSet 학습분(v3)", as_, s_as)):
        for c, cls in ((1, "glass"), (2, "scream")):
            sel = z["y"] == c
            if src.startswith("AudioSet") and cls == "scream":
                sel = sel & np.array([n.endswith("in_v31=1") for n in z["notes"]])   # v3.1 규칙 위에서
            n, k = int(sel.sum()), int((s[cls][sel] < THR_EXCL).sum())
            excl[(src, cls)] = (n, k)
            print(f"| {src}{' (v3.1 규칙 후)' if src.startswith('AudioSet') and cls == 'scream' else ''} | {cls} | {n} | {k} | {n - k} | {100 * k / n:.1f}% |")
    for c, cls in ((1, "glass"), (2, "scream")):
        sel = zt["y"] == c
        n, k = int(sel.sum()), int((s_te[cls][sel] < THR_EXCL).sum())
        print(f"| FSD50K 계열 시험 (참고 — 거르지 않는다) | {cls} | {n} | {k} | — | {100 * k / n:.1f}% |")

    # ── 아기 울음
    print("\n## 아기 울음(`Baby cry, infant cry`) 라벨 보유 구간")
    baby_ext = [r for r in M if "Baby cry" in r["labels_all"]]
    for r in baby_ext:
        print(f"  외부 {r['segment_id']} group={r['group']} v1={r['v1_class']} real={r['real_class']} primary={r['real_primary']}")
    with open(os.path.join(REPO, "tools", "external_set", "train_v2_list.csv"), encoding="utf-8") as f:
        tv2 = list(csv.DictReader(f))
    man = json.load(open(os.path.join(LA.V3, "MANIFEST.json"), encoding="utf-8"))
    used = {cls: set(v) for cls, v in man["audioset_segments"].items()}
    baby_tr = [r for r in tv2 if "Baby cry" in r["labels_all"]]
    for r in baby_tr:
        where = [cls for cls, v in used.items() if r["segment_id"] in v]
        print(f"  train_v2 {r['segment_id']} group={r['group']} v1={r['v1_class']} real={r['real_class']} v3에서={where}")
    print(f"  외부 {len(baby_ext)}구간, train_v2 {len(baby_tr)}구간")

    # ── 청취 세트
    rng = np.random.default_rng(SEED)
    Xext = np.load(os.path.join(LA.EXT, "windows_int8.npy"), mmap_mode="r")
    maps = {}
    v1_rows = {cls: {(r["clip_id"], int(r["start_sample"])): r for r in LA.read_index(LA.V1, "train", cls)} for cls in LA.EVENTS}
    te_rows = {cls: {(r["clip_id"], int(r["start_sample"])): r for r in LA.read_index(LA.V1, "test", cls)} for cls in LA.EVENTS}

    def contrast(cls):
        c = LA.EVENTS.index(cls)
        idx = np.flatnonzero((tr["y"] == c) & (s_tr[cls] >= LA.HIGH))
        pick = idx[rng.permutation(len(idx))[:N_CONTRAST]]
        return [dict(role="contrast", source="v1_train", cls=cls, clip=str(tr["clips"][i]), start=int(tr["starts"][i]),
                     score=float(s_tr[cls][i]), labels=str(tr["notes"][i]), definition="",
                     audio=lambda i=i: LA.shard_window(LA.V1, "train", cls, v1_rows[cls][(str(tr["clips"][i]), int(tr["starts"][i]))], maps))
                for i in pick]

    sets = {}
    # A: 외부 scream — v1eq 또는 real31 양성
    A = []
    for i, r in enumerate(M):
        if y_eq[i] == 2 or y_31[i] == 2:
            d = ("v1+실사용" if (y_eq[i] == 2 and y_31[i] == 2) else ("v1만" if y_eq[i] == 2 else "실사용만"))
            A.append(dict(role="target", source="ext", cls="scream", clip=r["segment_id"], start=int(r["win_start_sample"]),
                          score=float(se["scream"][i]), labels=r["labels_all"], definition=d, audio=lambda i=i: np.asarray(Xext[i])))
    sets["A_ext_scream"] = A + contrast("scream")
    B = [dict(role="target", source="ext", cls="glass", clip=r["segment_id"], start=int(r["win_start_sample"]),
              score=float(se["glass"][i]), labels=r["labels_all"], definition="v1=실사용", audio=lambda i=i: np.asarray(Xext[i]))
         for i, r in enumerate(M) if y_31[i] == 1]
    sets["B_ext_glass"] = B + contrast("glass")
    C = []
    for i in np.flatnonzero(zt["y"] == 1):
        clip, st = str(zt["clips"][i]), int(zt["starts"][i])
        C.append(dict(role="target", source="v1_test", cls="glass", clip=clip, start=st, score=float(s_te["glass"][i]),
                      labels=te_rows["glass"][(clip, st)]["note"], definition="v1",
                      audio=lambda clip=clip, st=st: LA.shard_window(LA.V1, "test", "glass", te_rows["glass"][(clip, st)], maps)))
    sets["C_test_glass"] = C + contrast("glass")

    print("\n## 청취 세트")
    tot = 0
    for k, v in sets.items():
        nt = sum(1 for d in v if d["role"] == "target")
        print(f"  {k}: 대상 {nt} + 대조 {len(v) - nt} = {len(v)}창  (예상 {len(v) * 8 / 60:.0f}분 @ 8초/창)")
        tot += len(v)
    print(f"  합계 {tot}창, 예상 {tot * 8 / 60:.0f}분")
    if a.dry_run:
        return
    import soundfile as sf
    os.makedirs(a.out, exist_ok=True)
    key = []
    for k, v in sets.items():
        d = os.path.join(a.out, k); os.makedirs(d, exist_ok=True)
        order = rng.permutation(len(v))
        for n, j in enumerate(order, 1):
            item = v[j]
            fn = f"{n:03d}.wav"
            sf.write(os.path.join(d, fn), (item["audio"]().astype(np.int16) * 256), LA.SR)
            key.append(dict(set=k, file=fn, verdict="", role=item["role"], source=item["source"], cls=item["cls"],
                            definition=item["definition"], clip=item["clip"], start_sample=item["start"],
                            panns_score=round(item["score"], 4), labels=item["labels"]))
    with open(os.path.join(a.out, "key.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0].keys())); w.writeheader(); w.writerows(key)
    # 청취자용 빈 판정표 (정답 없음)
    with open(os.path.join(a.out, "verdict_sheet.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["set", "file", "verdict(O/X/?)", "memo"])
        for r in key:
            w.writerow([r["set"], r["file"], "", ""])
    print("→", a.out, f"({len(key)} wav, key.csv, verdict_sheet.csv)")


if __name__ == "__main__":
    main()
