#!/usr/bin/env python3
"""다운로드 결과 집계: 클래스·정의별 유효 개수, 실패 사유, 스피커 부분집합 유효 개수.

사용: python3 summarize_download.py --list sampling_list.csv --state ~/safesound-external/download_state.csv
"""
import argparse
import collections
import csv
import os

CLASSES = ["siren", "glass", "scream", "dog_bark"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--state", required=True)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.list)))
    st = {}
    for r in csv.DictReader(open(os.path.expanduser(a.state))):
        st[r["segment_id"]] = r          # 마지막 기록이 최종
    ok = {s for s, r in st.items() if r["status"] == "ok"}
    pending = [r["segment_id"] for r in rows if r["segment_id"] not in st
               or st[r["segment_id"]]["status"] not in ("ok", "fail")]
    print(f"목록 {len(rows)}  시도됨 {len(st)}  ok {len(ok)}  미완료/차단 {len(pending)}")

    print("\n[실패 사유]")
    reasons = collections.Counter(r["reason"].split(":")[0] for r in st.values() if r["status"] == "fail")
    for k, v in reasons.most_common():
        print(f"  {k:16}{v:>5}")
    print("\n[그룹별 시도/유효/실패]")
    print(f"{'group':12}{'목록':>6}{'ok':>6}{'fail':>6}{'유효율':>8}")
    for g in CLASSES + ["background", "def_gap"]:
        gr = [r for r in rows if r["group"] == g]
        o = sum(r["segment_id"] in ok for r in gr)
        f = sum(st.get(r["segment_id"], {}).get("status") == "fail" for r in gr)
        print(f"{g:12}{len(gr):>6}{o:>6}{f:>6}{(o / len(gr) if gr else 0):>8.1%}")

    print("\n[정의별 유효 개수]  (클래스 그룹만. v1/real 은 그 정의가 해당 클래스를 양성으로 부르는 구간)")
    print(f"{'class':12}{'v1':>6}{'real':>6}{'real_primary':>14}{'union':>7}{'speaker':>9}")
    for c in CLASSES:
        gr = [r for r in rows if r["group"] == c and r["segment_id"] in ok]
        print(f"{c:12}{sum(r['v1_class'] == c for r in gr):>6}{sum(r['real_class'] == c for r in gr):>6}"
              f"{sum(r['real_primary'] == '1' for r in gr):>14}{len(gr):>7}"
              f"{sum(r['speaker'] == '1' for r in gr):>9}")
    gr = [r for r in rows if r["group"] == "background" and r["segment_id"] in ok]
    print(f"{'background':12}{len(gr):>6}{len(gr):>6}{'-':>14}{len(gr):>7}{sum(r['speaker'] == '1' for r in gr):>9}")
    gr = [r for r in rows if r["group"] == "def_gap" and r["segment_id"] in ok]
    print(f"{'def_gap':12}{'v1 양성':>6}{len(gr):>6}  v1 클래스별:",
          dict(collections.Counter(r["v1_class"] for r in gr)))


if __name__ == "__main__":
    main()
