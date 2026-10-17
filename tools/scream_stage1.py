#!/usr/bin/env python3
"""비명 1단계 (학습 없이) — `docs/results/scream-plan.md`, 사용자 지시 2026-10-13.

  (1) 평가 정의 일치 **testL-S**: testL 에서 `Yell`·`Shout` 만 보유(`Screaming` 없음)한 시험셋 비명 창 66 개를
      **평가에서 뺀다** (배경으로 옮기지 않는다). 외부 real31L 이 이미 그 규칙이다. 주 기준(testL)은 그대로 두고 나란히 낸다.
  (2) 비명 별도 문턱값: 문턱값은 시험셋 배경이 아니라 **학습 데이터의 검증용 창**(프로브 검증과 같은 원본 단위 10%,
      `teacher_cache.fsid_unit < 0.1`, 1,622창)에서 정한다. 전체 오경보 300회/h 를 검증 배경에서 유지한 채 비명 몫 b 를
      0.05~0.95 로 훑어 검증 macro-F1 이 최대인 b 를 고른다 (동률이면 작은 b). 비교용으로 같은 검증 창에서 정한 **단일**
      문턱값도 낸다. 시험셋·외부에는 그 문턱값을 그대로 적용하고 실제 오경보/h 를 함께 적는다.

규칙: 외부 세트는 채점만 한다 (문턱값·몫 선택에 쓰지 않는다). 시험셋 배경도 (2) 에서는 쓰지 않는다.
사용 (WSL2, ai8x venv, OPENBLAS_NUM_THREADS=1):  python tools/scream_stage1.py
출력: data/external_eval/scream_stage1.json, 콘솔 표. 검증 로짓 캐시 data/external_eval/logits/<실행>.<config>.val.npz
"""
import argparse
import contextlib
import csv
import glob
import io
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "tools"), os.path.join(REPO, "datasets")]
import eval_external as E                                      # noqa: E402
import eval_confusion as EC                                    # noqa: E402
import eval_v2 as V                                            # noqa: E402
import eval_v4 as Q                                            # noqa: E402
from eval_threshold import margin_of, thr_for_target           # noqa: E402
from teacher_cache import fsid_unit, VAL_FRAC                  # noqa: E402

V4 = os.path.join(REPO, "data", "processed", "safesound_v4")
V4_TRAIN_ROOT = os.path.join(E.AI8X, "data", "SafeSoundV4")    # 학습 루트 (safesound_v4 의 심볼릭 링크)
GT_DIR = os.path.join(REPO, "data", "raw", "FSD50K_meta", "FSD50K.ground_truth")
OUT = os.path.join(E.WORK, "scream_stage1.json")
SHARES = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
VAL_N_EXPECTED = 1622                                           # teacher.json probe.val_n
N_YELLSHOUT_TEST = 66                                           # scream-plan.md 2절
SCREAM = 2
# (표시 이름, 구성 키, 실행 접두사)
CONFIGS = [("④", "wave", "safesound-v4d-wave"), ("가 u1000", "mel_u1000", "safesound-v4d-melu1000"),
           ("다 u800", "mel_u800", "safesound-v4d-melu800"), ("간격 400", "mel_h400", "safesound-v4d-melh400"),
           ("①′", "mel_inc", "safesound-v4d-melinc"), ("(a) 주 lr 0.0003", "wave_logstage", "safesound-v4d-logstage"),
           ("(a) 보조 lr 0.001 (참고)", "wave_logstage", "safesound-v4d-logstage-lr1e3"),
           # ②′ NPU 에너지 적분 필터뱅크 (chain32, 2026-10-17, docs/results/fe-design.md) — 별도 행. 완주 전에는 seeds_of 가 건너뛴다
           ("②′ fe", "wave_fe", "safesound-v4d-fe"),
           # ②′ 곡선 변형 (chain33, 2026-10-17 저녁, docs/results/fe-curve-design.md)
           ("②′ b64", "wave_fe_b64", "safesound-v4d-fe-b64"),
           ("②′ f31", "wave_fe_f31", "safesound-v4d-fe-f31"),
           ("②′ b64f31", "wave_fe_b64f31", "safesound-v4d-fe-b64f31")]


def fsd_labels():
    gt = {}
    for f in ("dev.csv", "eval.csv"):
        with open(os.path.join(GT_DIR, f), encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                gt[r["fname"]] = set(r["labels"].split(","))
    return gt


def yellshout_only_test(gt):
    """시험셋 비명 창 중 `Screaming` 없이 `Yell`·`Shout` 만 보유한 (clip_id, start_sample)."""
    with open(os.path.join(V4, "test", "scream", "index.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    drop = set()
    for r in rows:
        L = gt.get(r["fsid"])
        if L is None:
            continue                       # FSD50K 가 아닌 출처 (시험셋 비명은 전부 FSD50K 다)
        if "Screaming" not in L and ({"Yell", "Shout"} & L):
            drop.add((r["clip_id"], int(r["start_sample"])))
    return drop


def test_keep_S(S, zt, drop):
    keep = S["keep_t"].copy()
    n = 0
    for i, (c, s, y) in enumerate(zip(zt["clips"], zt["starts"], zt["y"])):
        if y == SCREAM and (str(c), int(s)) in drop:
            keep[i] = False; n += 1
    assert n == N_YELLSHOUT_TEST, n
    return keep, n


def val_logits(config, ck, bias):
    """검증 창(원본 단위 10%, 무증강) 로짓 — 체크포인트 mtime 기준 캐시. (fsids, y, logits)."""
    import argparse as _ap
    import torch
    import ai8x
    os.makedirs(os.path.join(E.WORK, "logits"), exist_ok=True)
    key = os.path.basename(os.path.dirname(ck)).split("___")[0]
    path = os.path.join(E.WORK, "logits", f"{key}.{config}.val.npz")
    if os.path.isfile(path) and os.path.getmtime(path) >= os.path.getmtime(ck):
        z = np.load(path, allow_pickle=False)
        return z["fsids"], z["y"], z["logits"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = EC.load_model(ck, len(E.CLASSES), E.AI8X, False, bias, config=config)
    if "불일치" in buf.getvalue():
        sys.exit(f"[에러] state_dict 불일치 — {ck}\n{buf.getvalue()[-400:]}")
    _root, kw = EC.CONFIG_DATA.get(config, ("SafeSound", {}))
    ds = EC.dataset_class(config)(V4_TRAIN_ROOT, "train",
                                  transform=ai8x.normalize(args=_ap.Namespace(act_mode_8bit=False)), augment=False, **kw)
    idx = [j for j, m in enumerate(ds.meta) if fsid_unit(m[1]) < VAL_FRAC]
    assert len(idx) == VAL_N_EXPECTED, (len(idx), VAL_N_EXPECTED)
    fs, ys, lg = [], [], []
    with torch.no_grad():
        for i in range(0, len(idx), 256):
            xs = []
            for j in idx[i:i + 256]:
                x, t = ds[j]
                xs.append(x); ys.append(int(t)); fs.append(str(ds.meta[j][1]))
            lg.append(model(torch.stack(xs)).numpy())
    fs, y, lg = np.array(fs), np.array(ys), np.concatenate(lg)
    np.savez(path, fsids=fs, y=y, logits=lg)
    return fs, y, lg


def thr_pair(lv, yv, share):
    """검증 창에서 (비명 문턱값, 나머지 문턱값). 전체 300회/h 를 비명 share : 나머지 (1−share) 로 나눈다."""
    marg, best = margin_of(lv, E.BG)
    mb = marg - lv[:, E.BG]
    isbg = yv == E.BG
    n_bg = int(isbg.sum())
    ts = thr_for_target(mb[isbg & (best == SCREAM)], n_bg, E.PER_HOUR, share * E.TARGET)
    to = thr_for_target(mb[isbg & (best != SCREAM)], n_bg, E.PER_HOUR, (1 - share) * E.TARGET)
    return ts, to


def predict2(lg, ts, to):
    marg, best = margin_of(lg, E.BG)
    m = marg - lg[:, E.BG]
    thr = np.where(best == SCREAM, ts, to)
    return np.where(m > thr, best, E.BG)


def fa_per_hour(p, isbg):
    return float((p[isbg] != E.BG).sum() / isbg.sum() * E.PER_HOUR)


def evaluate(p_test, p_ext, S, keepL, keepS, yt, y_ext):
    isbg_e = S["isbg"]
    r = {}
    for tag, keep in (("testL", keepL), ("testL_S", keepS)):
        f1, rec, _ = E.metrics(yt[keep], p_test[keep])
        r[tag] = {"f1": f1, "scream_recall": float(rec[SCREAM]), "recall": rec.tolist()[:4]}
    r["test_fa_h"] = fa_per_hour(p_test, yt == E.BG)
    sel = y_ext >= 0
    f1, rec, _ = E.metrics(y_ext[sel], p_ext[sel])
    r["ext"] = {"f1": f1, "scream_recall": float(rec[SCREAM]), "recall": rec.tolist()[:4]}
    r["ext_fa_h"] = fa_per_hour(p_ext, isbg_e)
    r["ext_fire_n"] = int((p_ext[isbg_e] != E.BG).sum())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=None, help="표시 이름으로 일부만")
    a = ap.parse_args()
    gt = fsd_labels()
    drop = yellshout_only_test(gt)
    print(f"시험셋 비명 중 Yell·Shout 만 보유 (Screaming 없음): {len(drop)}창 → testL-S 에서 제외")
    S = Q.setup()
    zt = np.load(os.path.join(REPO, "data", "ceiling-panns-zs.npz"), allow_pickle=True)
    keepL = S["keep_t"]
    keepS, n_drop = test_keep_S(S, zt, drop)
    y_ext = S["Y"]["real31L"]
    print(f"시험셋 채점 창: testL {int(keepL.sum())} → testL-S {int(keepS.sum())} (비명 {int((zt['y'] == SCREAM).sum())} → "
          f"{int(((zt['y'] == SCREAM) & keepS).sum())})  외부 비명 양성 {int((y_ext == SCREAM).sum())}")
    out = {"n_test": {"testL": int(keepL.sum()), "testL_S": int(keepS.sum()), "scream_testL": int((zt["y"] == SCREAM).sum()),
                      "scream_testL_S": int(((zt["y"] == SCREAM) & keepS).sum()), "dropped": n_drop},
           "val_n": VAL_N_EXPECTED, "shares": SHARES, "configs": {}}
    R = {}
    for name, config, pre in CONFIGS:
        if a.configs and name not in a.configs:
            continue
        bias = config in Q.BIAS_CONFIGS
        cks = Q.seeds_of(pre)
        res = []
        for ck in cks:
            _c, yt, lt = E.logits_for(config, ck, Q.V.TEST, "test", bias=bias)
            assert np.array_equal(yt, zt["y"])
            _c2, _y2, le = E.logits_for(config, ck, S["ext_root"], "ext", bias=bias)
            fv, yv, lv = val_logits(config, ck, bias)
            isbg_v = yv == E.BG
            one = {"ckpt": os.path.basename(ck)}
            # ── 주 기준: 시험셋 배경 @300/h 단일 문턱값 (기존 그대로)
            thr = E.thr_at(lt, yt == E.BG)
            one["main"] = evaluate(E.predict(lt, thr), E.predict(le, thr), S, keepL, keepS, yt, y_ext)
            one["main"]["thr"] = thr
            # ── 검증 창 단일 문턱값 (@300/h, 검증 배경)
            thr_v = E.thr_at(lv, isbg_v)
            one["val_single"] = evaluate(E.predict(lt, thr_v), E.predict(le, thr_v), S, keepL, keepS, yt, y_ext)
            f1v, recv, _ = E.metrics(yv, E.predict(lv, thr_v))
            one["val_single"].update({"thr": thr_v, "val_f1": f1v, "val_scream_recall": float(recv[SCREAM])})
            # ── 검증 창 비명 별도 문턱값: 비명 몫 b 훑기 → 검증 macro-F1 최대
            sweep = []
            for b in SHARES:
                ts, to = thr_pair(lv, yv, b)
                if ts is None or to is None:
                    continue
                f1b, recb, _ = E.metrics(yv, predict2(lv, ts, to))
                sweep.append({"share": b, "thr_s": ts, "thr_o": to, "val_f1": f1b, "val_scream_recall": float(recb[SCREAM]),
                              "val_fa_h": fa_per_hour(predict2(lv, ts, to), isbg_v)})
            best = max(sweep, key=lambda d: (d["val_f1"], -d["share"]))
            one["val_split"] = evaluate(predict2(lt, best["thr_s"], best["thr_o"]), predict2(le, best["thr_s"], best["thr_o"]),
                                        S, keepL, keepS, yt, y_ext)
            one["val_split"].update({"share": best["share"], "thr_s": best["thr_s"], "thr_o": best["thr_o"],
                                     "val_f1": best["val_f1"], "val_scream_recall": best["val_scream_recall"], "sweep": sweep})
            res.append(one)
            print(f"  {name} {one['ckpt']}: testL F1 {one['main']['testL']['f1']:.4f} / testL-S {one['main']['testL_S']['f1']:.4f}"
                  f" | 검증 단일 thr {thr_v:.2f} (시험 FA {one['val_single']['test_fa_h']:.0f}/h) | 비명 몫 {best['share']:.2f}"
                  f" 검증 F1 {best['val_f1']:.4f} → testL-S 비명 {100 * one['val_split']['testL_S']['scream_recall']:.1f}%"
                  f" 외부 비명 {100 * one['val_split']['ext']['scream_recall']:.1f}% (외부 FA {one['val_split']['ext_fa_h']:.0f}/h)", flush=True)
        R[name] = res
        out["configs"][name] = {"n": len(res), "seeds": res}

    def ms(rs, path):
        v = []
        for r in rs:
            d = r
            for k in path:
                d = d[k]
            v.append(float(d))
        return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)

    def cell(rs, path, pct=False):
        m, s = ms(rs, path)
        return f"{100 * m:.1f} ± {100 * s:.1f}" if pct else f"{m:.3f} ± {s:.3f}"

    # ── 표 1: testL 대 testL-S (주 기준 문턱값 그대로)
    print("\n## (1) 평가 정의 일치 — testL 대 testL-S (시험셋 배경 @300/h 단일 문턱값, 시드 1~3 평균 ± sd)")
    print(f"testL-S = testL 에서 Yell·Shout 만 보유한 비명 {n_drop}창 제외 (비명 {out['n_test']['scream_testL']} → {out['n_test']['scream_testL_S']}창). 외부는 real31L 그대로.")
    print("| 구성 | n | macro-F1 testL | macro-F1 testL-S | Δ | 비명 recall testL | 비명 recall testL-S | 외부 비명 recall (참고) |")
    print("|---|---:|---|---|---:|---|---|---|")
    for name, rs in R.items():
        dL, dS = ms(rs, ("main", "testL", "f1"))[0], ms(rs, ("main", "testL_S", "f1"))[0]
        print(f"| {name} | {len(rs)} | {cell(rs, ('main', 'testL', 'f1'))} | {cell(rs, ('main', 'testL_S', 'f1'))} | {dS - dL:+.3f} | "
              f"{cell(rs, ('main', 'testL', 'scream_recall'), True)} | {cell(rs, ('main', 'testL_S', 'scream_recall'), True)} | "
              f"{cell(rs, ('main', 'ext', 'scream_recall'), True)} |")

    # ── 표 2: 비명 별도 문턱값 (검증 창에서 결정)
    print(f"\n## (2) 비명 별도 문턱값 — 검증 창({VAL_N_EXPECTED}창, 원본 단위 10%)에서 결정. 전체 300회/h 유지, 비명 몫 b 는 검증 macro-F1 최대")
    print("| 구성 | 문턱값 | 비명 몫 b (시드별) | 시험셋 FA/h (실현) | macro-F1 testL | 비명 recall testL | macro-F1 testL-S | 비명 recall testL-S | 외부 FA/h (실현) | 외부 macro-F1 | 외부 비명 recall |")
    print("|---|---|---|---:|---|---|---|---|---:|---|---|")
    for name, rs in R.items():
        for tag, lab in (("main", "주: 시험셋 배경 단일 (참고)"), ("val_single", "검증 단일"), ("val_split", "검증 비명 별도")):
            shares = ", ".join(f"{r['val_split']['share']:.2f}" for r in rs) if tag == "val_split" else "—"
            print(f"| {name} | {lab} | {shares} | {ms(rs, (tag, 'test_fa_h'))[0]:.0f} | {cell(rs, (tag, 'testL', 'f1'))} | "
                  f"{cell(rs, (tag, 'testL', 'scream_recall'), True)} | {cell(rs, (tag, 'testL_S', 'f1'))} | {cell(rs, (tag, 'testL_S', 'scream_recall'), True)} | "
                  f"{ms(rs, (tag, 'ext_fa_h'))[0]:.0f} | {cell(rs, (tag, 'ext', 'f1'))} | {cell(rs, (tag, 'ext', 'scream_recall'), True)} |")
    # ── 별도 − 단일 (같은 검증 문턱값 출처끼리, Welch 95% CI)
    print("\n| 구성 | 별도 − 단일: 비명 recall testL-S (pp) | macro-F1 testL-S | 외부 비명 recall (pp) | 외부 macro-F1 | 시험셋 FA/h 차 | 외부 FA/h 차 |")
    print("|---|---|---|---|---|---:|---:|")
    out["split_minus_single"] = {}
    for name, rs in R.items():
        if len(rs) < 2:
            continue
        d = {}
        for key, path, pct in (("scream_S", ("testL_S", "scream_recall"), True), ("f1_S", ("testL_S", "f1"), False),
                               ("scream_ext", ("ext", "scream_recall"), True), ("f1_ext", ("ext", "f1"), False)):
            w = V.welch([r["val_single"][path[0]][path[1]] for r in rs], [r["val_split"][path[0]][path[1]] for r in rs])
            d[key] = list(w)
        d["fa_test"] = ms(rs, ("val_split", "test_fa_h"))[0] - ms(rs, ("val_single", "test_fa_h"))[0]
        d["fa_ext"] = ms(rs, ("val_split", "ext_fa_h"))[0] - ms(rs, ("val_single", "ext_fa_h"))[0]
        out["split_minus_single"][name] = d
        pp = lambda t: f"{100 * t[0]:+.1f} [{100 * t[1]:+.1f}, {100 * t[2]:+.1f}]"
        print(f"| {name} | {pp(d['scream_S'])} | {Q.fmt(d['f1_S'])} | {pp(d['scream_ext'])} | {Q.fmt(d['f1_ext'])} | {d['fa_test']:+.0f} | {d['fa_ext']:+.0f} |")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("저장:", OUT)


if __name__ == "__main__":
    main()
