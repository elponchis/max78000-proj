#!/usr/bin/env python3
"""v5 학습 제외 목록 — `scream-plan.md` 9.1 (사용자 승인 2026-10-14).

exclude_v5.csv = exclude_v4.csv (558, PANNs < 0.02 glass·scream) + 청취 판정 X·? 인 **학습** 비명 창
  (2단계 `data_listened_scream.csv` 130 + 시험셋 정답 청취 세트의 학습 동반 30창 `data_listened_test_scream.csv` 중 X·? 12 — 9.7 항목 1).
role target, source v4_train / audioset 만. 시험셋 창(v4_test)과 대조 창은 넣지 않는다. 배경으로 옮기지 않는다 (제외만).
빌더(`build_v3_windows.py --exclude`)와 KAT(`kat_v3.py --edition v5`)는 (cls, clip_id, start_sample) 만 읽는다.

사용 (WSL2):  python3 tools/make_exclude_v5.py
"""
import csv
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import scream_listened as SL                     # noqa: E402
import scream_listened_test as ST                # noqa: E402

V4_LIST = os.path.join(REPO, "data", "label_audit", "exclude_v4.csv")
OUT = os.path.join(REPO, "data", "label_audit", "exclude_v5.csv")
EXPECT_NEW = 130 + 12                            # 2단계 X 86 + ? 44 (8.3) + 학습 동반 30창의 X 7 + ? 5 (10절)


def main():
    with open(V4_LIST, encoding="utf-8") as f:
        base = list(csv.DictReader(f))
    keys = {(r["cls"], r["clip_id"], int(r["start_sample"])) for r in base}
    new = []
    for r in SL.load() + ST.load():
        if r["role"] != "target" or r["source"] == "v4_test" or r["verdict"] == "O":
            continue
        k = ("scream", r["clip"], int(r["start_sample"]))
        if k in keys:
            sys.exit(f"[에러] 이미 v4 제외 목록에 있다: {k}")
        keys.add(k)
        new.append(dict(cls="scream", clip_id=r["clip"], start_sample=int(r["start_sample"]),
                        source=f"listened:{r['set']}:{r['definition']}:{r['verdict']}", score=""))
    assert len(new) == EXPECT_NEW, (len(new), EXPECT_NEW)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cls", "clip_id", "start_sample", "source", "score"])
        w.writeheader(); w.writerows(base); w.writerows(new)
    print(f"→ {OUT}: v4 {len(base)} + 청취 {len(new)} = {len(base) + len(new)}행")


if __name__ == "__main__":
    main()
