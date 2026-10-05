#!/usr/bin/env python3
"""AudioSet strong **train** 분할에서 5클래스 구간 목록을 만든다 (다운로드 없음).

⚠️ 지금은 받아 두기만 한다 — **학습에 쓰지 않는다.** 오디오는 레포 밖(~/safesound-external/train_v2/).

선별 규칙 (결과 보기 전에 고정, 2026-10-05 사용자 승인: 클래스당 합집합 400 + 배경 1200, 두 정의 태그):
  - 정의 상수는 build_sampling.py 의 것을 그대로 import (v1 정의 / 실사용 정의). 둘 다 in_v1·in_real 로 태그.
  - 어느 정의에서든 대상 클래스가 2개 이상 겹치는 구간 제외.
  - **eval 분할·external_v1 과 영상 ID(ytid)가 겹치는 구간 제외** (겹침이 0 이어도 코드가 막는다).
  - 클래스마다 합집합 N_CLASS(400): 패스 1 = v1 후보에서 N_CLASS/2 개 균일 무작위,
    패스 2 = 실사용 후보에서 합집합이 N_CLASS 가 될 때까지 (패스 1 로 뽑힌 것은 제외). 컬럼 `pass` 에 기록.
    (한 정의를 먼저 몰아 뽑으면 포함 관계 때문에 다른 정의가 한쪽으로 쏠린다.)
  - background: 두 정의 모두 대상 mid 가 없는 구간에서 N_BG(1200) 균일 무작위. 하드 네거티브 유사 열 표시.
  - 고정 시드 SEED. 삭제 영상은 대체하지 않는다.

사용: python3 build_train_v2.py --meta ~/safesound-external/meta --out train_v2_list.csv
"""
import argparse
import collections
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_sampling import (CLASSES, CROWD_V1, DOG_REAL, DOG_V1, GLASS, HN, SCREAM_REAL,  # noqa: E402
                            SCREAM_V1, SIREN_SUB, SIREN_V1)

SEED = 78002
N_CLASS = 400
N_BG = 1200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--external-list", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                            "sampling_list.csv"))
    a = ap.parse_args()
    meta = os.path.expanduser(a.meta)

    nm = dict(l.rstrip("\n").split("\t") for l in open(f"{meta}/mid_to_display_name.tsv"))
    by = {v: k for k, v in nm.items()}
    M = lambda names: {by[n] for n in names if n in by}  # noqa: E731

    def load(p):
        seg = collections.defaultdict(list)
        for r in csv.DictReader(open(p), delimiter="\t"):
            seg[r["segment_id"]].append((float(r["start_time_seconds"]), float(r["end_time_seconds"]), r["label"]))
        return seg

    seg = load(f"{meta}/audioset_train_strong.tsv")
    yt = lambda s: s.rsplit("_", 1)[0]  # noqa: E731
    banned = {yt(s) for s in load(f"{meta}/audioset_eval_strong.tsv")}
    banned |= {r["ytid"] for r in csv.DictReader(open(a.external_list))}
    dropped = [s for s in seg if yt(s) in banned]
    print(f"train 구간 {len(seg)}, eval/external_v1 과 영상 ID 가 겹쳐 제외 {len(dropped)}")
    for s in dropped:
        del seg[s]

    sv1, ssub, gl = M(SIREN_V1), M(SIREN_SUB), M(GLASS)
    dv1, dre = M(DOG_V1), M(DOG_REAL)
    cv1, cre, crowd = M(SCREAM_V1), M(SCREAM_REAL), M(CROWD_V1)
    ALL = sv1 | ssub | gl | dv1 | cv1

    def k1(s):
        c = set()
        if s & sv1: c.add("siren")
        if s & gl: c.add("glass")
        if (s & cv1) and not (s & crowd): c.add("scream")
        if s & dv1: c.add("dog_bark")
        return c

    def k2(s):
        c = set()
        if s & (sv1 | ssub): c.add("siren")
        if s & gl: c.add("glass")
        if s & cre: c.add("scream")
        if s & dre: c.add("dog_bark")
        return c

    rows, multi = {}, 0
    for sid in sorted(seg):
        s = {l for _, _, l in seg[sid]}
        a1, a2 = k1(s), k2(s)
        if len(a1) > 1 or len(a2) > 1:
            multi += 1
            continue
        rows[sid] = dict(s=s, v1=next(iter(a1), ""), real=next(iter(a2), ""))
    print("다중 클래스 겹침 제외", multi)

    rng = random.Random(SEED)
    chosen, how = {}, {}
    for c in CLASSES:
        for key, quota in (("v1", N_CLASS // 2), ("real", N_CLASS)):
            pool = sorted(x for x, r in rows.items() if r[key] == c and x not in chosen)
            rng.shuffle(pool)
            have = sum(1 for g in chosen.values() if g == c)
            for x in pool[:max(0, quota - have)]:
                chosen[x] = c
                how[x] = key
    pool = sorted(x for x, r in rows.items() if not (r["s"] & ALL))
    print("배경 후보", len(pool))
    rng.shuffle(pool)
    for x in pool[:N_BG]:
        chosen[x] = "background"
        how[x] = "bg"

    def span(sid, r):
        c = r["v1"] or r["real"]
        if not c:
            return "", "", ""
        mids = {"siren": sv1 | ssub, "glass": gl, "scream": cv1, "dog_bark": dv1}[c]
        ev = [(s0, e0) for s0, e0, l in seg[sid] if l in mids]
        s0, e0 = max(ev, key=lambda e: e[1] - e[0])
        return f"{s0:.3f}", f"{e0:.3f}", f"{e0 - s0:.3f}"

    cols = ["segment_id", "ytid", "start_s", "group", "v1_class", "real_class", "in_v1", "in_real", "pass",
            "event_start", "event_end", "event_dur", "siren_sub", "scream_sub", "has_bark", "has_dog",
            "hardneg_like", "labels_all"]
    out = []
    for sid in sorted(chosen):
        r, s = rows[sid], rows[sid]["s"]
        y, st = sid.rsplit("_", 1)
        e0, e1, ed = span(sid, r)
        hn = [k for k, g in HN.items() if s & M(g)]
        if "dog_nonbark" in hn and by["Bark"] in s: hn.remove("dog_nonbark")
        if "glass_nonshatter" in hn and s & gl: hn.remove("glass_nonshatter")
        out.append(dict(segment_id=sid, ytid=y, start_s=int(st) / 1000, group=chosen[sid],
                        v1_class=r["v1"], real_class=r["real"], in_v1=int(bool(r["v1"])),
                        in_real=int(bool(r["real"])), **{"pass": how[sid]},
                        event_start=e0, event_end=e1, event_dur=ed,
                        siren_sub=";".join(sorted(nm[m] for m in s & (sv1 | ssub))),
                        scream_sub=";".join(sorted(nm[m] for m in s & (cv1 | crowd))),
                        has_bark=int(by["Bark"] in s), has_dog=int(by["Dog"] in s),
                        hardneg_like=";".join(hn) if chosen[sid] == "background" or hn else "",
                        labels_all=";".join(sorted(nm.get(m, m) for m in s))))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(out)

    print(f"\n총 구간 {len(out)}")
    print(f"{'class':11}{'합집합':>7}{'v1':>6}{'real':>6}{'패스1(v1)':>11}{'패스2(real)':>12}  후보(v1/real)")
    for c in CLASSES:
        g = [r for r in out if r["group"] == c]
        print(f"{c:11}{len(g):>7}{sum(r['v1_class'] == c for r in g):>6}{sum(r['real_class'] == c for r in g):>6}"
              f"{sum(r['pass'] == 'v1' for r in g):>11}{sum(r['pass'] == 'real' for r in g):>12}  "
              f"{sum(x['v1'] == c for x in rows.values())}/{sum(x['real'] == c for x in rows.values())}")
    print(f"{'background':11}{N_BG:>7}")
    g = [r for r in out if r["group"] == "background"]
    print("[배경 하드네거티브 유사]", dict(collections.Counter(
        k for r in g for k in (r["hardneg_like"].split(";") if r["hardneg_like"] else ["(없음)"]))))
    print("[siren 하위]", collections.Counter(r["siren_sub"].count(";") + 1 for r in out if r["group"] == "siren" and r["siren_sub"]))


if __name__ == "__main__":
    main()
