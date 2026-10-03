#!/usr/bin/env python3
"""AudioSet strong-eval 에서 외부 검증 세트 샘플링 목록을 만든다 (다운로드 없음).

두 정의를 모두 태그한다. 다운로드는 합집합으로 한 번, 채점은 정의별로 따로.
  (가) v1 정의   : 학습 데이터 정의 (scripts/build_class_manifest.py classify())
  (나) 실사용 정의 : glass=Shatter / siren=Siren+하위 4종 / dog=Bark만 / scream=Screaming만

⚠️ 학습 데이터 정의(v1)는 바꾸지 않았다. 이 스크립트는 평가용 목록만 만든다.
결과 보기 전에 규칙을 고정했다: 시드·그룹 크기·정의는 아래 상수가 전부다.

사용: python3 build_sampling.py --meta ~/safesound-external/meta --out sampling_list.csv
"""
import argparse
import collections
import csv
import random

SEED = 78000
N_CLASS = 100        # 정의별 클래스당 최대 (합집합은 겹치면 더 작다)
N_BG = 300
N_GAP = 100
N_SPK_CLASS = 30     # 스피커 재생용: 실사용 정의 기준 클래스당
N_SPK_BG = 60
CLASSES = ["siren", "glass", "scream", "dog_bark"]

# 표시 이름 (mid_to_display_name.tsv 기준). glass 는 `Glass shatter` = PANNs `Shatter` (/m/07rn7sz)
SIREN_V1 = ["Siren"]
SIREN_SUB = ["Civil defense siren", "Ambulance (siren)", "Police car (siren)",
             "Fire engine, fire truck (siren)"]
GLASS = ["Glass shatter"]
DOG_V1 = ["Bark", "Dog"]
DOG_REAL = ["Bark"]
SCREAM_V1 = ["Screaming", "Yell", "Shout"]
SCREAM_REAL = ["Screaming"]
# v1 군중 목록 (build_class_manifest.EX_CROWD). `Chatter` 는 strong 어휘에 없다.
CROWD_V1 = ["Cheering", "Applause", "Clapping", "Crowd", "Chatter", "Laughter",
            "Giggle", "Chuckle, chortle", "Music", "Singing"]
# 하드 네거티브 유사 여부 판정용
HN = {"dog_nonbark": ["Dog"], "glass_nonshatter": ["Glass", "Glass chink, clink"],
      "alarm": ["Alarm"], "speech": ["Speech"], "crying": ["Crying, sobbing"],
      "crowd": ["Crowd", "Cheering", "Applause", "Hubbub, speech noise, speech babble"],
      "laughter": ["Laughter"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    nm = dict(l.rstrip("\n").split("\t") for l in open(f"{a.meta}/mid_to_display_name.tsv"))
    by = {v: k for k, v in nm.items()}

    def M(names):
        return {by[n] for n in names if n in by}

    absent = sorted({n for g in (SIREN_V1, SIREN_SUB, GLASS, DOG_V1, SCREAM_V1, CROWD_V1,
                                 *HN.values()) for n in g if n not in by})
    print("strong 어휘에 없는 이름:", absent)

    seg = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{a.meta}/audioset_eval_strong.tsv"), delimiter="\t"):
        seg[r["segment_id"]].append((float(r["start_time_seconds"]),
                                     float(r["end_time_seconds"]), r["label"]))

    sv1, ssub, gl = M(SIREN_V1), M(SIREN_SUB), M(GLASS)
    dv1, dre = M(DOG_V1), M(DOG_REAL)
    cv1, cre, crowd = M(SCREAM_V1), M(SCREAM_REAL), M(CROWD_V1)
    ALL_TARGET = sv1 | ssub | gl | dv1 | cv1

    def classes_v1(s):
        c = set()
        if s & sv1: c.add("siren")
        if s & gl: c.add("glass")
        if (s & cv1) and not (s & crowd): c.add("scream")
        if s & dv1: c.add("dog_bark")
        return c

    def classes_real(s):
        c = set()
        if s & (sv1 | ssub): c.add("siren")
        if s & gl: c.add("glass")
        if s & cre: c.add("scream")
        if s & dre: c.add("dog_bark")
        return c

    rows = {}
    for sid, L in sorted(seg.items()):
        s = {l for _, _, l in L}
        k1, k2 = classes_v1(s), classes_real(s)
        if len(k1) > 1 or len(k2) > 1:
            continue        # 어느 정의에서든 대상 라벨이 2개 이상 겹치면 제외
        rows[sid] = dict(s=s, v1=next(iter(k1), ""), real=next(iter(k2), ""))

    rng = random.Random(SEED)
    chosen = {}             # sid -> group
    primary = set()         # 실사용 정의 패스에서 원래 뽑힌 구간 (real_primary)

    def pick(pool, n, group):
        pool = [x for x in sorted(pool) if x not in chosen]
        rng.shuffle(pool)
        for x in pool[:n]:
            chosen[x] = group

    for c in CLASSES:
        # 정의별로 최대 N_CLASS, 합집합. 겹침은 한 번만 (이미 뽑힌 것은 그 정의의 몫으로 센다).
        # 한 구간이 두 정의에 동시에 속하면 양쪽 개수에 다 들어가므로, 어느 쪽도 상한을 넘지 않게 센다.
        # 실사용 정의를 먼저 균일 무작위로 뽑는다 (스피커 부분집합 기준이고, siren 하위 종류가
        # 빠지지 않게). v1 은 그 위에 v1 에만 속하는 구간으로 채운다 — 포함 관계(Bark ⊂ Bark|Dog,
        # Siren ⊂ Siren+하위)라서 v1 이 100 에 못 미칠 수 있다. 보고에 그대로 적는다.
        cnt = {"v1": 0, "real": 0}
        for key in ("real", "v1"):
            other = "real" if key == "v1" else "v1"
            pool = [x for x, r in rows.items() if r[key] == c and x not in chosen]
            pool.sort()
            rng.shuffle(pool)
            for x in pool:
                if cnt[key] >= N_CLASS:
                    break
                # siren 만 예외: v1 정의 구간을 보충해 v1 100개를 맞추고 실사용은 100 을 넘겨도
                # 된다 (사용자 승인 2026-10-03). 채점은 real_primary=1 로 한정해 하위 종류 비율을 지킨다.
                if rows[x][other] == c and cnt[other] >= N_CLASS and c != "siren":
                    continue
                chosen[x] = c
                cnt[key] += 1
                if key == "real":
                    primary.add(x)
                if rows[x][other] == c:
                    cnt[other] += 1
    # def_gap: 실사용에서는 배경, v1 에서는 양성
    pick([x for x, r in rows.items() if r["v1"] and not r["real"]], N_GAP, "def_gap")
    # 배경: 두 정의 모두에서 대상 라벨이 없고, 어느 대상 mid 도 없는 구간
    pick([x for x, r in rows.items() if not (r["s"] & ALL_TARGET)], N_BG, "background")

    # 스피커 재생 부분집합: 실사용 정의 기준, 클래스당 30 / 배경 60
    spk = set()
    for c in CLASSES:
        pool = sorted(x for x, g in chosen.items() if g == c and rows[x]["real"] == c)
        random.Random(SEED + 1).shuffle(pool)
        spk |= set(pool[:N_SPK_CLASS])
    pool = sorted(x for x, g in chosen.items() if g == "background")
    random.Random(SEED + 1).shuffle(pool)
    spk |= set(pool[:N_SPK_BG])

    def event_span(sid, r):
        c = r["v1"] or r["real"]
        if not c:
            return "", "", ""
        mids = {"siren": sv1 | ssub, "glass": gl, "scream": cv1, "dog_bark": dv1}[c]
        # 가장 긴 단일 이벤트 인스턴스를 창 중심 기준으로 쓴다 (짖음처럼 흩어진 소리는
        # 전체 합집합 구간의 중심이 이벤트가 없는 곳일 수 있다).
        ev = [(s0, e0) for s0, e0, l in seg[sid] if l in mids]
        s0, e0 = max(ev, key=lambda e: e[1] - e[0])
        return f"{s0:.3f}", f"{e0:.3f}", f"{e0 - s0:.3f}"

    cols = ["segment_id", "ytid", "start_s", "group", "v1_class", "real_class", "in_v1",
            "in_real", "real_primary", "speaker", "event_start", "event_end", "event_dur", "siren_sub",
            "scream_sub", "has_bark", "has_dog", "hardneg_like", "labels_all"]
    out = []
    for sid in sorted(chosen):
        r, s = rows[sid], rows[sid]["s"]
        yt, st = sid.rsplit("_", 1)
        e0, e1, ed = event_span(sid, r)
        hn = [k for k, g in HN.items() if s & M(g)]
        if "dog_nonbark" in hn and by["Bark"] in s: hn.remove("dog_nonbark")
        if "glass_nonshatter" in hn and s & gl: hn.remove("glass_nonshatter")
        out.append(dict(
            segment_id=sid, ytid=yt, start_s=int(st) / 1000, group=chosen[sid],
            v1_class=r["v1"], real_class=r["real"], in_v1=int(bool(r["v1"])),
            in_real=int(bool(r["real"])), real_primary=int(sid in primary),
            speaker=int(sid in spk),
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

    # 요약표
    print(f"\n총 구간 {len(out)}  (스피커 {sum(r['speaker'] for r in out)})")
    print("\n[정의별 개수] 클래스 그룹만 (def_gap 제외)")
    print(f"{'class':10}{'v1':>6}{'real':>6}{'union':>7}{'spk':>5}  후보(v1/real)")
    for c in CLASSES:
        g = [r for r in out if r["group"] == c]
        pv = sum(1 for x in rows.values() if x["v1"] == c)
        pr = sum(1 for x in rows.values() if x["real"] == c)
        print(f"{c:10}{sum(r['v1_class'] == c for r in g):>6}"
              f"{sum(r['real_class'] == c for r in g):>6}{len(g):>7}"
              f"{sum(r['speaker'] for r in g):>5}  {pv}/{pr}")
    g = [r for r in out if r["group"] == "background"]
    print(f"{'background':10}{len(g):>6}{len(g):>6}{len(g):>7}"
          f"{sum(r['speaker'] for r in g):>5}")
    g = [r for r in out if r["group"] == "def_gap"]
    print(f"\n[def_gap] {len(g)}  v1 클래스별:",
          dict(collections.Counter(r["v1_class"] for r in g)))
    print("  siren_sub/scream/dog 내역:",
          dict(collections.Counter(("Dog-only" if r["v1_class"] == "dog_bark" else
                                    "Yell/Shout-only" if r["v1_class"] == "scream" else
                                    "siren-Siren-only" if r["v1_class"] == "siren" else "glass")
                                   for r in g)))
    g = [r for r in out if r["group"] == "background"]
    print("\n[배경 하드네거티브 유사]", dict(collections.Counter(
        k for r in g for k in (r["hardneg_like"].split(";") if r["hardneg_like"] else ["(없음)"]))))
    lab = collections.Counter(l for r in g for l in r["labels_all"].split(";"))
    print("[배경 라벨 상위 15]", lab.most_common(15))
    print("\n[이벤트 길이 중앙값]")
    for c in CLASSES:
        d = sorted(float(r["event_dur"]) for r in out if r["group"] == c and r["event_dur"])
        print(f"  {c:10} {d[len(d)//2]:.2f}s  (<=1s: {sum(x <= 1 for x in d)}/{len(d)})")


if __name__ == "__main__":
    main()
