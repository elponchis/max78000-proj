#!/usr/bin/env python3
"""조용한 배경음 창의 **자체 오탐률** — 라운드 6 (b) 의 기각 조건.

`docs/results/g8-round6-design.md` b.6:
> 조용한 배경음 쿼터의 자체 오탐률이 기준선 대비 **1.5배를 넘으면**,
> macro-F1 이 올라도 채택하지 않는다. G7(무인 8시간 구동)이 직접 깨진다.

**왜 필요한가**: 음량 정규화는 조용한 창의 잡음을 최대 GMAX 배까지 키운다.
그 창들이 이벤트처럼 보이면 무인 구동에서 오경보가 쏟아진다. macro-F1 은
이벤트 재현율에 끌려가므로 이 위험을 가리지 못한다.

**조용함의 정의**: 전처리에 `--bg-quiet-frac 0.15` 로 조용한 구간을 일부러
15% 넣었지만(CLAUDE.md 7장) **인덱스에 표식이 없다.** 그래서 int8 피크로
직접 정의한다 — 배경음 테스트 창을 피크 오름차순으로 세워 하위 q 분위를
"조용한 쿼터" 로 본다. 표식보다 이쪽이 오히려 조작적으로 명확하다.

⚠️ 오탐 판정은 **고정 오경보 문턱값**에서 한다. argmax 로 재면 모델마다
동작점이 달라 비교가 안 된다.

사용 (WSL2, numpy 만):
    python3 tools/quiet_bg_fa.py --npz data/logits/D-1_s1.npz \\
        data/logits/D-1nm_s1.npz --data ~/ai8x-training/data
"""

import argparse
import csv
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "tools")]

CLASSES = ["siren", "glass", "scream", "dog_bark", "background"]
BG = 4
WIN = 16384
MARGIN = 1600


def bg_peaks(data_root, split="test"):
    """배경음 테스트 창의 int8 피크를 **인덱스 순서 그대로** 돌려준다.

    ⚠️ 정규화 **전** 원본 샤드의 피크다. 정규화는 시프트라 순서를 바꾸지
    않으므로, 어느 구성이든 같은 창 집합을 고르게 된다 (비교가 성립한다).
    """
    d = os.path.join(data_root, "SafeSound", split, "background")
    maps, out = {}, []
    with open(os.path.join(d, "index.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sp = os.path.join(d, f"shard_{int(r['shard']):04d}.npy")
            m = maps.get(sp)
            if m is None:
                m = np.load(sp, mmap_mode="r")
                maps[sp] = m
            w = np.asarray(m[int(r["row"])][MARGIN:MARGIN + WIN])
            out.append(int(np.abs(w.astype(np.int16)).max()))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--data", default=os.path.expanduser("~/ai8x-training/data"))
    ap.add_argument("--quiet-frac", type=float, default=0.15,
                    help="하위 몇 분위를 '조용한 쿼터' 로 볼지 "
                         "(전처리의 --bg-quiet-frac 과 맞춘다)")
    ap.add_argument("--point", default="300/h", choices=("300/h", "1000/h"))
    a = ap.parse_args()

    import eval_confusion as EC
    # margin_of / margin_predict 는 eval_threshold 쪽에 있다
    from eval_threshold import margin_of, margin_predict

    peaks = bg_peaks(a.data)
    k = max(1, int(round(a.quiet_frac * len(peaks))))
    thr_peak = int(np.sort(peaks)[k - 1])
    quiet = peaks <= thr_peak
    print(f"배경음 테스트 창 {len(peaks):,}개 중 조용한 쿼터 "
          f"{int(quiet.sum()):,}개 (하위 {100*a.quiet_frac:.0f}%, "
          f"int8 피크 <= {thr_peak})")
    print(f"  피크 분위: p5={np.percentile(peaks,5):.0f} "
          f"p15={np.percentile(peaks,15):.0f} p50={np.percentile(peaks,50):.0f} "
          f"p85={np.percentile(peaks,85):.0f}")
    print()
    print(f"{'구성':<18}{'전체 배경 오탐':>14}{'조용한 쿼터':>14}"
          f"{'나머지 85%':>13}{'조용/전체 비':>13}")
    print("-" * 72)

    base = None
    rows = []
    for f in a.npz:
        z = np.load(f, allow_pickle=True)
        y, lg = z["y"], z["logits"]
        label = str(z["label"])
        fx = EC.fixed_fa_points(lg, y, CLASSES, hop_ms=250)
        t = (fx.get("targets") or {}).get(a.point)
        if not t:
            print(f"{label:<18}  {a.point} 지점 없음")
            continue
        thr = t["thr"]
        marg_ev, ev_best = margin_of(lg, BG)
        pred = margin_predict(marg_ev, ev_best, lg[:, BG], BG, thr)
        isbg = (y == BG)
        assert int(isbg.sum()) == len(peaks), \
            f"배경음 창 수 불일치: {int(isbg.sum())} vs {len(peaks)}"
        fa = (pred[isbg] != BG)
        r_all = float(fa.mean())
        r_q = float(fa[quiet].mean())
        r_r = float(fa[~quiet].mean())
        rows.append((label, r_all, r_q, r_r))
        if base is None:
            base = r_q
        print(f"{label:<18}{100*r_all:>13.3f}%{100*r_q:>13.3f}%"
              f"{100*r_r:>12.3f}%{(r_q/r_all if r_all else float('nan')):>13.2f}")

    print()
    if len(rows) >= 2:
        print(f"기각 조건 (b.6) — 기준선 {rows[0][0]} 의 조용한 쿼터 오탐률 "
              f"{100*rows[0][2]:.3f}%")
        for label, _ra, rq, _rr in rows[1:]:
            base_q = rows[0][2]
            if base_q == 0.0:
                # 기준선이 0 이면 비율이 정의되지 않는다. 늘지 않았으면 통과다.
                ok, note = (rq == 0.0), ("증가 없음" if rq == 0.0
                                         else "기준선 0 에서 증가")
                print(f"  {label:<18}{100*rq:>8.3f}%  비 정의불가 ({note})  "
                      f"{'통과' if ok else '**기각**'}")
                continue
            ratio = rq / base_q
            ok = ratio <= 1.5
            print(f"  {label:<18}{100*rq:>8.3f}%  비 {ratio:>5.2f}배  "
                  f"{'통과' if ok else '**기각**'}")
    print()
    print(f"⚠️ 문턱값은 각 구성의 {a.point} 지점이다 — 전체 배경 오탐률은")
    print("   설계상 거의 같아진다. **차이는 그 오탐이 어디에 몰리는가**다.")
    print("⚠️ '조용한 쿼터' 는 전처리의 --bg-quiet-frac 표식이 아니라 int8")
    print("   피크 하위 분위로 정의했다 (인덱스에 표식이 없다).")


if __name__ == "__main__":
    main()
