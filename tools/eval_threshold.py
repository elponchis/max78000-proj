#!/usr/bin/env python3
"""배경음 로짓 오프셋 스윕 — 오경보와 재현율의 맞교환 곡선.

기준선에서 배경음 오탐이 34%(시간당 4,896회)로 나왔다. 재학습 전에 **이미 있는
체크포인트로** 어디까지 내려갈 수 있는지부터 본다. 배경음 로짓에 상수 `b` 를 더하면
결정 경계가 배경음 쪽으로 밀리고, `b` 하나로 오경보-재현율 곡선 전체가 그려진다.

여기서 나오는 답은 두 갈래다.
  · **적당한 `b` 로 오경보를 쓸 만한 수준까지 낮출 수 있다** → 재학습이 아니라
    운용 임계값 문제다. 그 `b` 를 펌웨어 후처리에 넣는다
  · **어떤 `b` 에서도 재현율이 무너진다** → 모델·데이터 문제다. 재학습으로 간다

⚠️ 창 단위 수치다. `tools/eval_confusion.py` 의 N프레임 표와 같은 한계를 가지며
(창이 시간적으로 떨어져 있다), 논문에 실을 오경보는 `tools/stream_eval.py` 와
보드 구동에서 얻는다. 여기서는 **상대 비교**(어느 `b` 가 얼마나 낫나)만 쓴다.

사용법 (Colab 또는 WSL2):
    python3 tools/eval_threshold.py --checkpoint <qat_best.pth.tar> \\
        --data /content/ai8x-training/data --ai8x /content/ai8x-training \\
        --png /content/drive/MyDrive/max78000/threshold-sweep.png
"""

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "datasets"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def sweep(logits, y_true, k, bg, offsets, per_hour):
    """오프셋마다 (시간당 오경보, 클래스별 재현율)."""
    from eval_confusion import recalls

    rows = []
    sel = y_true == bg
    for b in offsets:
        adj = logits.copy()
        adj[:, bg] += b
        pred = adj.argmax(1)
        fa = float((pred[sel] != bg).mean()) if sel.sum() else float("nan")
        rows.append({"b": float(b), "fa_rate": fa, "per_hour": fa * per_hour,
                     "recall": recalls(y_true, pred, k)})
    return rows


def margin_of(logits, bg):
    """창마다 `max(이벤트 로짓)` 과 그때의 이벤트 클래스.

    `margin = max(이벤트 로짓) − 배경음 로짓` 에 문턱값 `thr` 을 걸면 배경음 로짓에
    `thr` 을 더하는 것과 **판정이 완전히 같다**. 다른 점은 훑는 방식이다 — b 를
    등간격으로 훑으면 분포가 퍼져 있을 때 목표 오경보 지점을 건너뛰지만, 마진
    **분위수** 위에서 훑으면 원하는 오경보 횟수를 정확히 집어낼 수 있다. 기준선
    곡선이 완만해서(b=8 에서도 2,623회/h) 이 방식이 필요해졌다.
    """
    ev = np.array([c for c in range(logits.shape[1]) if c != bg])
    sub = logits[:, ev]
    return sub.max(axis=1), ev[sub.argmax(axis=1)]


def margin_predict(marg_ev, ev_best, bg_logit, bg, thr):
    """마진 문턱값 `thr` 에서의 예측. thr=0 이면 원래 argmax 와 같다."""
    return np.where(marg_ev - bg_logit > thr, ev_best, bg)


def thr_for_target(marg_bg, n_bg, per_hour, target):
    """목표 오경보(시간당) 이하가 되는 가장 낮은 문턱값. 측정 불가면 None.

    창 하나가 곧 `per_hour/n_bg` 회다 — 목표가 그보다 작으면 **이 테스트셋으로는
    측정할 수 없다**(0회로 보여도 "관측 한계 아래"라는 뜻이지 0 이 아니다).
    """
    res = per_hour / n_bg
    if target < res:
        return None
    n_allow = int(np.floor(target / res))
    if n_allow >= n_bg:
        return float(marg_bg.min() - 1.0)
    return float(np.sort(marg_bg)[::-1][n_allow])


def margin_report(logits, y_true, clips, names, bg, per_hour, sources,
                  targets=(1000.0, 300.0, 100.0, 10.0, 1.0)):
    """마진 분포 요약 + 문턱값별 재현율 + 배경음 출처별 오경보 분해."""
    from eval_confusion import recalls

    marg_ev, ev_best = margin_of(logits, bg)
    marg = marg_ev - logits[:, bg]
    sel = y_true == bg
    n_bg = int(sel.sum())
    mb = marg[sel]
    res = per_hour / n_bg

    print("\n" + "=" * 72)
    print("배경음 마진 분포   margin = max(이벤트 로짓) − 배경음 로짓")
    print("=" * 72)
    qs = [1, 5, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]
    print("  " + "".join(f"{f'p{q:g}':>9}" for q in qs))
    print("  " + "".join(f"{np.percentile(mb, q):>9.2f}" for q in qs))
    print(f"  평균 {mb.mean():.2f}   표준편차 {mb.std():.2f}   "
          f"마진>0 (= b=0 에서 오경보인 창) {100*(mb > 0).mean():.1f}%")
    print(f"\n  ⚠️ 측정 해상도: 배경음 창 {n_bg:,}개 → **창 1개 = 시간당 "
          f"{res:.2f}회**")
    print(f"     시간당 {res:.2f}회보다 낮은 목표는 이 테스트셋으로 측정 불가다.")
    print("     0회로 나와도 '0' 이 아니라 '관측 한계 아래'라는 뜻이다. 그 영역을")
    print("     재려면 배경음 오디오 시간을 늘려야 한다 → tools/stream_eval.py 와")
    print("     보드 8시간 무인 구동(G7).")

    rows = []
    for t in targets:
        thr = thr_for_target(mb, n_bg, per_hour, t)
        if thr is None:
            rows.append({"target": t, "thr": None})
            continue
        pred = margin_predict(marg_ev, ev_best, logits[:, bg], bg, thr)
        got = float((pred[sel] != bg).mean()) * per_hour
        rows.append({"target": t, "thr": thr, "per_hour": got,
                     "recall": recalls(y_true, pred, len(names)), "pred": pred})

    print("\n=== 마진 문턱값별 — 목표 오경보와 그때의 클래스별 재현율 ===")
    print(f"  {'목표/h':>8}{'문턱값':>9}{'실제/h':>9}{'창':>7}   "
          + "".join(f"{n[:8]:>9}" for n in names))
    print("  " + "-" * (35 + 9 * len(names)))
    for r in rows:
        if r["thr"] is None:
            print(f"  {r['target']:>8.0f}{'—':>9}{'측정 불가':>11}{'—':>5}   "
                  f"← 해상도 {res:.2f}회/h 아래")
            continue
        print(f"  {r['target']:>8.0f}{r['thr']:>9.2f}{r['per_hour']:>9.1f}"
              f"{int(round(r['per_hour'] / res)):>7}   "
              + "".join(f"{100*v:>8.1f}%" for v in r["recall"]))
    print("  '창' 은 그 지점에서 오경보로 남은 배경음 창 수다.")

    # ── 출처별 분해
    if sources:
        tags = np.array([sources.get(c, "(미상)") for c in clips])
        hits = [r for r in rows if r["thr"] is not None]
        base = margin_predict(marg_ev, ev_best, logits[:, bg], bg, 0.0)
        cols = [("b=0", base)] + [(f"{r['target']:.0f}/h", r["pred"]) for r in hits]
        print("\n=== 배경음 오경보의 출처별 분해 (시간당 기여분) ===")
        print(f"  {'출처':<26}{'창':>7}" + "".join(f"{c[0]:>12}" for c in cols))
        print("  " + "-" * (33 + 12 * len(cols)))
        for tag in sorted(set(tags[sel])):
            m = sel & (tags == tag)
            n = int(m.sum())
            cells = ""
            for _, pred in cols:
                f = int((m & (pred != bg)).sum())
                cells += f"{per_hour * f / n_bg:>7.1f}({f:>3})"
            print(f"  {tag:<26}{n:>7,}{cells}")
        cells = "".join(f"{per_hour * int((sel & (p != bg)).sum()) / n_bg:>7.1f}"
                        f"({int((sel & (p != bg)).sum()):>3})" for _, p in cols)
        print(f"  {'합계':<26}{n_bg:>7,}{cells}")
        print("  괄호는 창 수. 시간당 값은 **전체 배경음 기준 기여분**이라 합이 총")
        print("  오경보와 같다 (출처별 자체 오탐률이 아니다). 자체 오탐률을 보려면")
        print("  창 수로 나눠 볼 것 — 창 수가 적은 출처는 분산이 크다.")

        print("\n=== 오경보가 울린 클래스 (b=0) ===")
        for c, n in enumerate(names):
            if c == bg:
                continue
            m = int((sel & (base == c)).sum())
            print(f"  → {n:<12}{m:>7,}창{per_hour * m / n_bg:>9.1f}회/h")

    return {"n_background": n_bg, "resolution_per_hour": round(res, 3),
            "margin_bg": mb,
            "quantiles": {f"p{q:g}": round(float(np.percentile(mb, q)), 3)
                          for q in qs},
            "targets": [{"target": r["target"], "thr": r["thr"],
                         "measurable": r["thr"] is not None,
                         "per_hour": (round(r["per_hour"], 2)
                                      if r["thr"] is not None else None),
                         "recall": ({n: round(float(v), 4)
                                     for n, v in zip(names, r["recall"])}
                                    if r["thr"] is not None else None)}
                        for r in rows]}


def load_bg_sources(root, split="test", manifest=None):
    """배경음 창의 출처 태그. 인덱스 CSV 의 note + (있으면) 매니페스트의 데이터셋.

    태그는 세 갈래로 나눈다 — 일반 배경음 / 하드 네거티브(종류별) / 조용한 창
    쿼터. 조용한 창 쿼터는 인덱스에 따로 적히지 않으므로 전처리와 **같은 해시**로
    되살린다 (`prepare_safesound.rank_id(clip_id + ":quiet")`, `--bg-quiet-frac`).
    """
    import csv as _csv

    idx = os.path.join(root, split, "background", "index.csv")
    if not os.path.isfile(idx):
        print(f"  (출처 분해 생략: {idx} 없음)")
        return None
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    try:
        from prepare_safesound import rank_id
    except Exception as e:                       # 전처리 스크립트가 없는 환경
        print(f"  (조용한 창 쿼터 표시 생략: {e})")
        rank_id = None

    ds = {}
    manifest = manifest or os.path.join(REPO, "data", "interim", "manifest.csv")
    if os.path.isfile(manifest):
        with open(manifest, encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                if r["cls"] == "background":
                    ds[r["clip_id"]] = r["source"]
    multi = len(set(ds.values())) > 1

    out = {}
    with open(idx, encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            note = (r.get("note") or "").split(" ")[0]
            if note.startswith("hard_neg:"):
                tag = "하드네거티브:" + note.split(":", 1)[1]
            elif note == "bg:general":
                tag = "일반 배경음"
            else:
                tag = f"기타({note or '빈칸'})"
            if rank_id is not None and rank_id(r["clip_id"] + ":quiet") % 1000 < 150:
                tag += " [조용]"
            if multi:
                tag += f" <{ds.get(r['clip_id'], '?')}>"
            out[r["clip_id"]] = tag
    if ds and not multi:
        print(f"  (데이터셋별 분해 생략: 배경음이 전량 "
              f"{next(iter(set(ds.values())))} 이다 — LibriSpeech 미투입)")
    return out


def half_split_by_origin(fsids, y_true, seed=0):
    """테스트셋을 **원본(fsid) 단위**로 반분한다. 클래스별로 번갈아 배정한다.

    창 단위로 나누면 같은 녹음의 창이 양쪽에 흩어져 두 반이 서로 독립이 아니다 —
    한쪽에서 고른 임계값이 다른 쪽에 이미 맞춰져 있는 셈이 된다. 원본 단위로
    갈라야 "본 적 없는 녹음에서 그 임계값이 통하나"를 볼 수 있다.

    클래스별로 시드 고정 순열을 만든 뒤 번갈아 배정해, 두 반의 클래스 구성이
    비슷하게 유지된다 (원본 하나는 한 클래스에만 속한다 — 규칙 1).
    """
    cls_of = {}
    for f, t in zip(fsids, y_true):
        cls_of[f] = int(t)
    rng = np.random.default_rng(seed)
    a, b = set(), set()
    for c in sorted(set(cls_of.values())):
        fs = sorted(f for f, t in cls_of.items() if t == c)
        for i, j in enumerate(rng.permutation(len(fs))):
            (a if i % 2 == 0 else b).add(fs[j])
    return a, b


def print_halves(fsids, y_true, a, b, names):
    """두 반의 클래스별 원본·창 수 — 구성이 치우쳤는지 눈으로 확인한다."""
    fs = np.asarray(fsids)
    ma = np.isin(fs, list(a))
    mb = np.isin(fs, list(b))
    print(f"  {'클래스':<12}{'A 원본':>8}{'B 원본':>8}{'A 창':>8}{'B 창':>8}")
    print("  " + "-" * 44)
    for c, n in enumerate(names):
        oa = len({f for f, t in zip(fsids, y_true) if t == c and f in a})
        ob = len({f for f, t in zip(fsids, y_true) if t == c and f in b})
        print(f"  {n:<12}{oa:>8}{ob:>8}{int((ma & (y_true == c)).sum()):>8}"
              f"{int((mb & (y_true == c)).sum()):>8}")
    print(f"  {'합계':<12}{len(a):>8}{len(b):>8}{int(ma.sum()):>8}{int(mb.sum()):>8}")
    return ma, mb


def at_targets(rows, names, targets=(100.0, 10.0, 1.0)):
    """목표 오경보(시간당) 이하가 되는 **첫 지점**을 고른다 (b 가 커질수록 감소)."""
    out = {}
    for t in targets:
        hit = next((r for r in rows if r["per_hour"] <= t), None)
        out[t] = hit
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--ai8x", default="/content/ai8x-training")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--bias", action="store_true")
    from eval_confusion import add_config_arg
    add_config_arg(ap)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--hop-ms", type=int, default=250)
    ap.add_argument("--lo", type=float, default=-2.0)
    ap.add_argument("--hi", type=float, default=8.0)
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--pick-on", choices=["test-half", "val", "test"],
                    default="test-half",
                    help="운용 지점(b)을 고를 데이터. test-half(기본)=테스트셋을 "
                         "**원본 단위**로 반분해 한쪽에서 고르고 다른 쪽에 적용. "
                         "val=ai8x 검증셋(창 단위 분리라 낙관적). "
                         "test=같은 데이터에서 고름(진단용, 낙관치)")
    ap.add_argument("--half-seed", type=int, default=0,
                    help="원본 단위 반분 시드 (실험 간 비교를 위해 고정한다)")
    ap.add_argument("--validation-split", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=0,
                    help="ai8x 가 검증셋을 떼는 시드. distiller 가 get_data_loaders "
                         "안에서 set_deterministic() 을 인자 없이 불러 0 이 된다")
    ap.add_argument("--no-margin", action="store_true",
                    help="마진 분위수 모드와 출처별 분해를 생략한다")
    ap.add_argument("--manifest",
                    help="배경음 출처(데이터셋) 조인용 매니페스트 CSV "
                         "(기본 data/interim/manifest.csv, 없으면 생략)")
    ap.add_argument("--png", help="곡선 PNG 저장 경로")
    ap.add_argument("--json", help="스윕 결과 JSON 저장 경로")
    a = ap.parse_args()

    # safesound 는 import 시점에 ai8x 를 부른다 → 경로를 먼저 넣는다
    if os.path.isdir(a.ai8x):
        sys.path.insert(0, a.ai8x)
    import safesound as S
    from eval_confusion import ai8x_valid_indices, collect_logits

    names = S.CLASSES
    k, bg = len(names), names.index("background")
    per_hour = 3600.0 / (a.hop_ms / 1000.0)
    print(f"체크포인트: {a.checkpoint}")

    # ── 운용 지점을 고를 데이터
    if a.pick_on == "val":
        # ai8x 가 train 에서 떼는 검증셋을 그대로 재현한다
        tr = S.SafeSound(os.path.join(a.data, "SafeSound"), "train",
                         transform=None, augment=False)
        vidx = ai8x_valid_indices(len(tr), a.validation_split, a.split_seed)
        print(f"검증셋 재현: train {len(tr):,} 중 {len(vidx):,}창 "
              f"(split {a.validation_split}, seed {a.split_seed})")
        del tr
        _f, _c, _s, y_pick, lg_pick = collect_logits(a, names, "train", vidx)
    else:
        y_pick = lg_pick = None

    # ── 테스트셋 (항상 본다)
    fsids, clips, _s, y_true, logits = collect_logits(a, names, "test")
    n_bg = int((y_true == bg).sum())
    print(f"테스트 창 {len(y_true):,} (배경음 {n_bg:,})  hop {a.hop_ms}ms "
          f"→ 시간당 {per_hour:,.0f}회 추론\n")

    # ── 마진 모드 — 분위수 위에서 훑는다 (등간격 b 스윕이 놓치는 지점을 집는다)
    mrep = None
    if not a.no_margin:
        src = load_bg_sources(os.path.join(a.data, "SafeSound"), "test", a.manifest)
        mrep = margin_report(logits, y_true, clips, names, bg, per_hour, src)

    offsets = np.arange(a.lo, a.hi + 1e-9, a.step)
    rows = sweep(logits, y_true, k, bg, offsets, per_hour)   # 곡선은 테스트 전체

    # ── 원본 단위 반분 (기본 모드)
    halves = None
    if a.pick_on == "test-half":
        A, B = half_split_by_origin(fsids, y_true, a.half_seed)
        print(f"=== 테스트셋 원본 단위 반분 (seed {a.half_seed}) ===")
        ma, mb = print_halves(fsids, y_true, A, B, names)
        halves = [("A→B", ma, mb), ("B→A", mb, ma)]
        y_pick = lg_pick = None
    elif y_pick is None:
        y_pick, lg_pick = y_true, logits

    if y_pick is not None:
        pick_rows = sweep(lg_pick, y_pick, k, bg, offsets, per_hour)
    else:
        pick_rows = rows                                     # 표 출력용 (아래서 재계산)

    print(f"{'b':>6}{'오경보/h':>11}{'오탐률':>9}   " +
          "".join(f"{n[:8]:>9}" for n in names))
    print("-" * (26 + 9 * k))
    for r in rows:
        if abs(r["b"] * 4 - round(r["b"] * 4)) < 1e-6 and round(r["b"] * 2) == r["b"] * 2:
            pass                                   # 0.5 간격만 표로 (나머지는 JSON)
        else:
            continue
        rec = "".join(f"{100*v:>8.1f}%" for v in r["recall"])
        print(f"{r['b']:>6.2f}{r['per_hour']:>11,.0f}{r['fa_rate']:>8.1%}   {rec}")

    # ── 운용 지점: 고르는 데이터와 보고하는 데이터를 분리한다
    applied = {}

    def one_direction(label, pick_mask, apply_mask):
        """pick_mask 에서 b 를 고르고 apply_mask 에 적용한 결과를 찍는다."""
        pr = sweep(lg_all[pick_mask], y_true[pick_mask], k, bg, offsets, per_hour)
        ar = {round(r["b"], 4): r
              for r in sweep(lg_all[apply_mask], y_true[apply_mask], k, bg,
                             offsets, per_hour)}
        print(f"\n  [{label}] b 는 앞쪽 반에서, 수치는 뒤쪽 반에서")
        print(f"  {'목표/h':>8}{'b':>7}{'고른 반/h':>11}{'적용 반/h':>11}   " +
              "".join(f"{n[:8]:>9}" for n in names))
        print("  " + "-" * (37 + 9 * k))
        for t, r in at_targets(pr, names).items():
            if r is None:
                print(f"  {t:>8.0f}{'—':>7}{'도달 불가':>11}{'—':>11}")
                continue
            ap_ = ar.get(round(r["b"], 4))
            rec = "".join(f"{100*v:>8.1f}%" for v in ap_["recall"])
            print(f"  {t:>8.0f}{r['b']:>7.2f}{r['per_hour']:>11,.0f}"
                  f"{ap_['per_hour']:>11,.0f}   {rec}")
            applied.setdefault(label, {})[t] = {
                "b": r["b"], "pick_per_hour": r["per_hour"],
                "apply_per_hour": ap_["per_hour"],
                "apply_recall": {n: round(float(v), 4)
                                 for n, v in zip(names, ap_["recall"])}}

    lg_all = logits
    if halves:
        print(f"\n=== 운용 지점 — **원본 단위로 분리**해 고르고 적용한다 ===")
        for label, pm, am in halves:
            one_direction(label, pm, am)
        print("\n  두 방향(A→B, B→A)이 크게 다르면 b 가 표본에 민감하다는 뜻이다 —")
        print("  그때는 어느 한 값을 운용 지점으로 확정하지 말 것.")
        print("  원본 단위로 갈랐으므로 같은 녹음이 양쪽에 걸치지 않는다. 다만 창")
        print("  단위 수치라 절대값이 아니라 상대 비교용이고, 논문 오경보는")
        print("  stream_eval.py 와 보드 8시간 구동에서 얻는다.")
    else:
        where = "ai8x 검증셋" if a.pick_on == "val" else "테스트셋(진단용)"
        hits = at_targets(pick_rows, names)
        by_b = {round(r["b"], 4): r for r in rows}
        print(f"\n=== 운용 지점 — b 는 **{where}**에서 고르고, 아래 수치는 "
              f"{'테스트셋' if a.pick_on == 'val' else '같은 데이터'} ===")
        print(f"{'목표/h':>8}{'b':>7}{'고른 곳/h':>11}{'테스트/h':>11}   " +
              "".join(f"{n[:8]:>9}" for n in names))
        print("-" * (37 + 9 * k))
        for t, r in hits.items():
            if r is None:
                print(f"{t:>8.0f}{'—':>7}{'도달 불가':>11}{'—':>11}   "
                      f"(b={a.hi} 까지 올려도 {pick_rows[-1]['per_hour']:,.0f}회)")
                continue
            te = by_b.get(round(r["b"], 4))
            rec = "".join(f"{100*v:>8.1f}%" for v in te["recall"])
            print(f"{t:>8.0f}{r['b']:>7.2f}{r['per_hour']:>11,.0f}"
                  f"{te['per_hour']:>11,.0f}   {rec}")
            applied[t] = {"b": r["b"], "pick_per_hour": r["per_hour"],
                          "test_per_hour": te["per_hour"],
                          "test_recall": {n: round(float(v), 4)
                                          for n, v in zip(names, te["recall"])}}

        if a.pick_on == "val":
            print("\n  ⚠️ **창 단위 분리라 낙관적이다.** ai8x 는 train 을 창 단위로")
            print("     쪼개므로, 검증 창의 대부분은 **같은 원본의 다른 창이 학습에")
            print("     들어가 있다.** 모델이 그 녹음을 이미 봤으니 여기서 고른 b 는")
            print("     처음 듣는 녹음에서보다 좋게 보인다.")
            print("     보고용으로는 `--pick-on test-half`(원본 단위 분리)를 쓸 것.")
        else:
            print("\n  ⚠️ **진단용, 낙관치다.** 같은 테스트셋에서 b 를 고르고 그 위에서")
            print("     보고했으므로 실제 운용 성능보다 좋게 나온다.")
            print("     보고하려면 `--pick-on test-half` 로 다시 돌릴 것.")

    base = rows[int(round((0.0 - a.lo) / a.step))] if a.lo <= 0 <= a.hi else None
    if base:
        print(f"\n  b=0 (원래 모델): 시간당 {base['per_hour']:,.0f}회, "
              + ", ".join(f"{n} {100*v:.1f}%" for n, v in zip(names, base['recall'])))
    print("\n  ⚠️ 창 단위 상대 비교용이다. 논문에 실을 오경보는 stream_eval.py 와")
    print("     보드 구동에서 얻는다. 여기서 고른 b 는 펌웨어 후처리에 그대로 넣는다.")

    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # ⚠️ **그림 안의 글자는 전부 영어다.** matplotlib 기본 폰트(DejaVu Sans)에
        # 한글 글리프가 없어 네모(tofu)로 깨진다. 논문 그림도 영어 캡션을 쓰므로
        # 폰트를 따로 설치하는 대신 라벨을 영어로 고정한다.
        # (한글이 꼭 필요하면 `apt-get install fonts-nanum` 후
        #  `matplotlib.rcParams["font.family"] = "NanumGothic"` 를 켤 것.)
        n_panel = 3 if mrep is not None else 2
        fig, ax = plt.subplots(1, n_panel, figsize=(6.5 * n_panel, 5))
        ph = [r["per_hour"] for r in rows]
        for c, n in enumerate(names):
            if c == bg:
                continue
            ax[0].plot(ph, [100 * r["recall"][c] for r in rows], marker=".", label=n)
        ax[0].set_xscale("symlog", linthresh=1)
        ax[0].invert_xaxis()
        ax[0].set_xlabel("False alarms per hour (window-level, log)")
        ax[0].set_ylabel("Recall (%)")
        ax[0].set_title("False alarms vs per-class recall")
        ax[0].grid(alpha=.3)
        ax[0].legend()
        ax[1].plot([r["b"] for r in rows], ph, marker=".")
        ax[1].set_yscale("log")
        ax[1].set_xlabel("Background logit offset b")
        ax[1].set_ylabel("False alarms per hour")
        ax[1].set_title("Offset vs false alarms")
        ax[1].grid(alpha=.3)
        for t in (100, 10, 1):
            ax[1].axhline(t, ls="--", lw=.8, color="gray")
        if mrep is not None:
            res = mrep["resolution_per_hour"]
            ax[1].axhline(res, ls=":", lw=1.2, color="crimson")
            ax[1].text(a.lo, res, f" measurement floor {res:.2f}/h",
                       color="crimson", va="bottom", fontsize=8)
            ax[2].hist(mrep["margin_bg"], bins=80, color="steelblue")
            ax[2].axvline(0, color="crimson", lw=1,
                          label="b=0 decision boundary")
            for t in (1000, 300, 100):
                thr = next((r["thr"] for r in mrep["targets"]
                            if r["target"] == t and r["thr"] is not None), None)
                if thr is not None:
                    ax[2].axvline(thr, ls="--", lw=.8, color="gray")
                    ax[2].text(thr, ax[2].get_ylim()[1] * .9, f"{t}/h",
                               rotation=90, fontsize=7, ha="right")
            ax[2].set_yscale("log")
            ax[2].set_xlabel("margin = max(event logit) - background logit")
            ax[2].set_ylabel("Background windows (log)")
            ax[2].set_title(f"Background margin distribution "
                            f"(n={mrep['n_background']:,})")
            ax[2].legend(fontsize=8)
            ax[2].grid(alpha=.3)
        fig.suptitle(f"Background logit offset sweep - "
                     f"{os.path.basename(a.checkpoint)}")
        fig.tight_layout()
        os.makedirs(os.path.dirname(a.png) or ".", exist_ok=True)
        fig.savefig(a.png, dpi=120)
        print(f"\n  그림: {a.png}  (그림 안 글자는 영어 — 한글 글리프 부재)")

    if a.json:
        out = {"checkpoint": a.checkpoint, "hop_ms": a.hop_ms,
               "pick_on": a.pick_on, "operating_points": applied,
               "half_seed": a.half_seed if a.pick_on == "test-half" else None,
               # val 은 창 단위 분리라 낙관적이다 (같은 원본의 다른 창이 학습에 있다)
               "diagnostic_only": a.pick_on in ("test", "val"),
               "classes": names, "n_windows": int(len(y_true)), "n_background": n_bg,
               # 마진 분위수 모드. `measurable: false` 는 "0회" 가 아니라
               # "이 테스트셋의 관측 한계 아래" 라는 뜻이다.
               "margin": ({k2: v for k2, v in mrep.items() if k2 != "margin_bg"}
                          if mrep is not None else None),
               "sweep": [{"b": r["b"], "per_hour": round(r["per_hour"], 1),
                          "fa_rate": round(r["fa_rate"], 5),
                          "recall": {n: round(float(v), 4)
                                     for n, v in zip(names, r["recall"])}}
                         for r in rows]}
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  JSON: {a.json}")


if __name__ == "__main__":
    main()
