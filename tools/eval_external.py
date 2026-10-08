#!/usr/bin/env python3
"""외부 검증 세트(AudioSet strong eval) 평가 — 재학습 없음, 기존 체크포인트.

세트 설명: `docs/results/external-set.md` (보조 세션 작성). 이 도구는 **읽기만**
한다 — `~/safesound-external/` 과 `tools/external_set/` 은 건드리지 않는다.

평가 경로
    외부 창 (758, 16384) int8 을 **우리 샤드 형식**으로 한 번 감싼 뒤
    (`data/external_eval/SafeSound/test/background/`, git 제외) 기존
    `eval_confusion.collect_logits` 를 **그대로** 부른다. 즉 모델 적재
    (`load_model` → `ai8x.update_model`), 특징 계산(Dataset), 텐서 변환이 시험셋
    평가와 같은 코드다. `tools/kat_evalpath.py --external` 이 그 코드가
    `train.py --evaluate` 와 같은지 본다. 샤드로 감싼 창이 원본과 같은지는
    `--self-check` 가 본다.

사전 등록 (결과를 보기 전에 고정)
  · 문턱값은 **기존 시험셋의 @300/h 값**을 체크포인트마다 그대로 쓴다.
    외부 세트로 다시 맞추지 않는다. (외부 배경 기준 @300/h 지점은 병기만.)
  · 정의 두 개로 따로 채점한다 (external-set.md 6절):
      v1     양성 = 클래스 그룹이면서 v1_class == 그룹, 음성 = background 그룹
      실사용 양성 = 클래스 그룹이면서 real_primary == 1, 음성 = background 그룹
    그 정의에서 양성이 아닌 클래스 그룹 구간은 **채점에서 뺀다**.
  · 순위 일치 여부만 보고한다. 외부 세트로 구성을 고르지 않는다.
  · 신뢰구간: **구간(=영상) 단위** 부트스트랩 2000회, 95% 백분위. 시드 평균
    지표를 재표본마다 다시 계산한다 (구간은 시드 간에 공통으로 뽑는다).

사용 (WSL2, ai8x venv, OPENBLAS_NUM_THREADS=1):
    python tools/eval_external.py --self-check
    python tools/eval_external.py            # 추론(캐시) + 채점 + JSON
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
AI8X = os.environ.get("AI8X_DIR") or os.path.expanduser("~/ai8x-training")
sys.path[:0] = [AI8X, os.path.join(REPO, "datasets"), os.path.join(REPO, "tools")]

EXT = os.path.expanduser("~/safesound-external/external_v1")
WORK = os.path.join(REPO, "data", "external_eval")       # git 제외
CLASSES = ["siren", "glass", "scream", "dog_bark", "background"]
EVENTS = CLASSES[:4]
BG = 4
HOP_MS = 250
PER_HOUR = 3600.0 / (HOP_MS / 1000.0)
TARGET = 300.0

# (표시 이름, --config, 실행 이름 접두, 시드 수). 순서는 표 순서다.
RUNS = [
    ("④", "wave", "safesound-v1cpu", 5),
    ("④ 0.5×", "wave_w050", "safesound-w050-v1", 3),
    ("①′", "mel_inc", "safesound-melinc-v1", 3),
    ("간격 400", "mel_h400", "safesound-melh400-v1", 3),
    ("간격 500", "mel_h500", "safesound-melh500-v1", 3),
    ("간격 500+멜 32", "mel_h500m32", "safesound-melh500m32-v1", 3),
    ("가 u1000", "mel_u1000", "safesound-melu1000-v1", 3),
    ("나 u1000f1024", "mel_u1000f1024", "safesound-melu1000f1024-v1", 3),
    ("다 u800", "mel_u800", "safesound-melu800-v1", 3),
]


def checkpoint(prefix, seed):
    name = prefix if seed == 1 else f"{prefix}-s{seed}"
    hits = sorted(glob.glob(os.path.join(REPO, "data", "logs-local", name + "___*")),
                  key=os.path.getmtime)
    if not hits:
        sys.exit(f"[에러] 실행 폴더가 없다: {name}")
    ck = os.path.join(hits[-1], name + "_qat_best.pth.tar")
    if not os.path.isfile(ck):
        sys.exit(f"[에러] 체크포인트가 없다: {ck}")
    return ck


# ───────────────────────────────────────────────────────── 외부 창 → 샤드
def load_external():
    X = np.load(os.path.join(EXT, "windows_int8.npy"), mmap_mode="r")
    with open(os.path.join(EXT, "meta.csv"), encoding="utf-8") as f:
        M = list(csv.DictReader(f))
    if X.shape != (len(M), 16384) or X.dtype != np.int8:
        sys.exit(f"[에러] 외부 세트 모양이 다르다: {X.shape} {X.dtype} / meta {len(M)}")
    return X, M


def build_shards(X, M):
    """외부 창을 우리 샤드 형식으로 감싼다. 반환: `--data` 로 줄 루트.

    무증강 Dataset 은 행의 `[MARGIN:MARGIN+WIN]` 만 읽는다 (`SafeSound._crop`).
    양옆 여유는 0 으로 두고 `left/right_margin=0` 으로 적는다 — 읽히지 않는다.
    전부 `background` 디렉터리에 넣는다: 정답은 meta.csv 에서 따로 읽으므로
    Dataset 의 target 은 쓰지 않는다. 행 순서 = meta.csv 순서.
    """
    import safesound as S
    d = os.path.join(WORK, "SafeSound", "test", "background")
    os.makedirs(d, exist_ok=True)
    sp, ip = os.path.join(d, "shard_0000.npy"), os.path.join(d, "index.csv")
    src_m = os.path.getmtime(os.path.join(EXT, "windows_int8.npy"))
    if not (os.path.isfile(sp) and os.path.isfile(ip)
            and os.path.getmtime(sp) >= src_m):
        rows = np.zeros((len(M), S.STORE), dtype=np.int8)
        rows[:, S.MARGIN:S.MARGIN + S.WIN] = X
        np.save(sp, rows)
        with open(ip, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["shard", "row", "clip_id", "fsid", "start_sample",
                        "left_margin", "right_margin", "note"])
            for i, r in enumerate(M):
                w.writerow([0, i, r["segment_id"], r["ytid"], 0, 0, 0, "external_v1"])
        print(f"  외부 샤드 생성: {sp}  ({len(M)}행)")
    return WORK


def self_check(X, M):
    """샤드로 감싼 창이 원본과 **비트 단위로** 같은지, 순서가 meta 와 같은지."""
    import ai8x
    import eval_confusion as EC
    root = build_shards(X, M)
    ai8x.set_device(85, False, False)
    ds = EC.make_dataset("wave", root, "test", None)
    assert len(ds) == len(M), (len(ds), len(M))
    bad = 0
    for i in range(len(ds)):
        _t, shard, row, left, right = ds.index[i]
        w = ds._crop(np.asarray(ds._shard(shard)[row]), left, right, None)
        bad += int(not np.array_equal(w, X[i])) + int(ds.meta[i][0] != M[i]["segment_id"])
    # 텐서 변환까지: Dataset 이 내는 (128,128) 이 kws20 관례(전치)와 같은가
    x0, _ = ds[0]
    ref = (X[0].astype(np.float32) + 128) / 256.0
    ref = ref.reshape(-1, 128).T
    bad += int(not np.allclose(x0.numpy(), ref))
    print(f"자가 점검: 창 {len(ds)}개, 불일치 {bad} → {'PASS' if not bad else 'FAIL'}")
    return bad == 0


# ───────────────────────────────────────────────────────────────── 추론
def logits_for(config, ck, data_root, tag):
    """`collect_logits` 로 로짓을 얻는다 (체크포인트 mtime 기준 캐시)."""
    import eval_confusion as EC
    os.makedirs(os.path.join(WORK, "logits"), exist_ok=True)
    key = os.path.basename(os.path.dirname(ck)).split("___")[0]
    path = os.path.join(WORK, "logits", f"{key}.{config}.{tag}.npz")
    if os.path.isfile(path) and os.path.getmtime(path) >= os.path.getmtime(ck):
        z = np.load(path, allow_pickle=False)
        return z["clips"], z["y"], z["logits"]
    a = argparse.Namespace(checkpoint=ck, config=config, data=data_root, ai8x=AI8X,
                           simulate=False, bias=False, batch_size=128, split="test")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _fs, clips, _st, y, lg = EC.collect_logits(a, CLASSES, "test")
    if "불일치" in buf.getvalue():
        sys.exit(f"[에러] state_dict 불일치 — {ck}\n{buf.getvalue()[-400:]}")
    np.savez(path, clips=clips.astype(str), y=y, logits=lg)
    return clips.astype(str), y, lg


# ───────────────────────────────────────────────────────────────── 채점
def predict(lg, thr):
    from eval_threshold import margin_of, margin_predict
    marg_ev, ev_best = margin_of(lg, BG)
    return margin_predict(marg_ev, ev_best, lg[:, BG], BG, thr)


def thr_at(lg, is_bg, target=TARGET):
    from eval_threshold import margin_of, thr_for_target
    marg_ev, _ = margin_of(lg, BG)
    mb = (marg_ev - lg[:, BG])[is_bg]
    return thr_for_target(mb, int(is_bg.sum()), PER_HOUR, target)


def metrics(y, p):
    """(macro-F1, 클래스별 recall[5], 배경 오경보/h)."""
    f, rec = [], []
    for c in range(5):
        tp = float(((y == c) & (p == c)).sum())
        fp = float(((y != c) & (p == c)).sum())
        fn = float(((y == c) & (p != c)).sum())
        f.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan)
        rec.append(tp / (tp + fn) if (tp + fn) else np.nan)
    nb = (y == BG).sum()
    fa = float((p[y == BG] != BG).sum()) / nb * PER_HOUR if nb else np.nan
    return float(np.nanmean(f)), np.array(rec), fa


def labels(M, definition):
    """정의별 정답 (−1 = 채점 제외). external-set.md 6절.

    `v1eq` (2026-10-07, dataset-v3-design.md 1.2·사전 등록 1): v1 정의에서 **siren 만
    `Siren` + 하위 4종** 으로 넓힌 것. FSD50K 어휘에는 `Siren` 만 있어 v1 의 사이렌은 종류
    불문 전부 `Siren` 이었는데, AudioSet strong 은 조상 라벨을 붙이지 않아 하위만 있는
    구간 37개가 v1 문자열 규칙에서 빠져 있었다. 라벨 파일(meta.csv)만으로 정했다 —
    siren 그룹은 전부 `real_class == siren` 이므로 그 열을 쓴다. 다른 클래스는 v1 과 같다.
    """
    y = np.full(len(M), -1)
    for i, r in enumerate(M):
        g = r["group"]
        if g == "background":
            y[i] = BG
        elif g in EVENTS:
            if definition == "v1" and r["v1_class"] == g:
                y[i] = EVENTS.index(g)
            if definition == "v1eq" and (r["v1_class"] == g or (g == "siren" and r["real_class"] == g)):
                y[i] = EVENTS.index(g)
            if definition == "real" and r["real_primary"] == "1" and r["real_class"] == g:
                y[i] = EVENTS.index(g)
            if definition == "real31":
                # 실사용 정의 (v3.1 주 기준, dataset-v3.1-design.md 4절): 양성 = real_primary 이고 그 클래스,
                # **음성** = 배경 + "그 정의에서 배경인 구간" — scream 그룹의 Yell·Shout 만(Screaming 없음) 구간.
                # def_gap(Dog 만 / Yell·Shout 만) 도 아래에서 음성으로 넣는다.
                if r["real_primary"] == "1" and r["real_class"] == g:
                    y[i] = EVENTS.index(g)
                elif g == "scream" and r["real_class"] != "scream":
                    y[i] = BG
        elif g == "def_gap" and definition == "real31":
            y[i] = BG
    return y


def seed_mean(preds, y, sel):
    """시드별 예측 목록 → 시드 평균 (macro-F1, recall[5], 오경보/h)."""
    ms = [metrics(y[sel], p[sel]) for p in preds]
    return (float(np.mean([m[0] for m in ms])),
            np.nanmean([m[1] for m in ms], axis=0),
            float(np.mean([m[2] for m in ms])))


def bootstrap(preds, y, sel, n_boot, rng):
    """구간 단위 부트스트랩 — 시드 평균 (macro-F1, recall[5]) 의 95% 백분위."""
    idx = np.flatnonzero(sel)
    out = np.empty((n_boot, 6))
    for b in range(n_boot):
        s = idx[rng.integers(0, len(idx), len(idx))]
        ms = [metrics(y[s], p[s]) for p in preds]
        out[b, 0] = np.mean([m[0] for m in ms])
        out[b, 1:] = np.nanmean([m[1] for m in ms], axis=0)
    lo, hi = np.nanpercentile(out, [2.5, 97.5], axis=0)
    return lo, hi


def fire_rate(preds, sel, cls=None):
    """시드 평균 발화율 (cls 가 있으면 그 클래스로 발화한 비율)."""
    if not sel.sum():
        return float("nan")
    return float(np.mean([((p[sel] != BG) if cls is None else (p[sel] == cls)).mean()
                          for p in preds]))


def rank_of(vals):
    """높은 값이 1위. 반환: 순위 배열."""
    order = np.argsort(-np.asarray(vals), kind="stable")
    r = np.empty(len(vals), dtype=int)
    r[order] = np.arange(1, len(vals) + 1)
    return r


def spearman(a, b):
    ra, rb = rank_of(a).astype(float), rank_of(b).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-data", default=os.path.join(AI8X, "data"))
    ap.add_argument("--out", default=os.path.join(WORK, "external_eval.json"))
    a = ap.parse_args()

    X, M = load_external()
    if a.self_check:
        sys.exit(0 if self_check(X, M) else 1)
    ext_root = build_shards(X, M)
    seg = np.array([r["segment_id"] for r in M])
    grp = np.array([r["group"] for r in M])
    flag = {k: np.array([r[k] == "1" for r in M])
            for k in ("bg_near_silent", "v1_floor_excluded", "webm_origin")}
    Y = {d: labels(M, d) for d in ("v1", "real")}
    is_bg = grp == "background"
    rng = np.random.default_rng(a.seed)

    res = {"n_windows": len(M), "target_per_hour": TARGET,
           "resolution_ext_bg_per_hour": round(PER_HOUR / int(is_bg.sum()), 2),
           "population": {d: {c: int((Y[d] == i).sum()) for i, c in enumerate(CLASSES)}
                          for d in Y},
           "configs": []}
    for name, config, prefix, n_seed in RUNS:
        P_fix, P_ext, test_f1, test_rec, thrs, thrs_ext = [], [], [], [], [], []
        for s in range(1, n_seed + 1):
            ck = checkpoint(prefix, s)
            _c, yt, lt = logits_for(config, ck, a.test_data, "test")
            thr = thr_at(lt, yt == BG)
            f1, rec, _fa = metrics(yt, predict(lt, thr))
            test_f1.append(f1)
            test_rec.append(rec)
            clips, _y, le = logits_for(config, ck, ext_root, "ext")
            if not np.array_equal(clips, seg):
                sys.exit(f"[에러] 외부 창 순서가 meta 와 다르다 — {name} s{s}")
            thr_e = thr_at(le, is_bg)
            P_fix.append(predict(le, thr))
            P_ext.append(predict(le, thr_e))
            thrs.append(thr)
            thrs_ext.append(thr_e)
            print(f"  {name} s{s}: 시험셋 F1 {f1:.4f}  thr {thr:+.3f}  "
                  f"(외부 배경 기준 thr {thr_e:+.3f})", flush=True)
        c = {"name": name, "config": config, "n_seeds": n_seed,
             "test_f1": float(np.mean(test_f1)), "test_f1_seeds": test_f1,
             "test_recall": np.mean(test_rec, axis=0).tolist(),
             "thr_test": thrs, "thr_ext_bg": thrs_ext, "defs": {}, "sub": {}}
        for d in ("v1", "real"):
            y, sel = Y[d], Y[d] >= 0
            f1, rec, fa = seed_mean(P_fix, y, sel)
            lo, hi = bootstrap(P_fix, y, sel, a.n_boot, rng)
            f1e, rece, fae = seed_mean(P_ext, y, sel)
            c["defs"][d] = {
                "f1": f1, "f1_ci": [lo[0], hi[0]], "recall": rec.tolist(),
                "recall_ci": [lo[1:].tolist(), hi[1:].tolist()],
                "fa_per_hour": fa,
                "f1_seeds": [metrics(y[sel], p[sel])[0] for p in P_fix],
                "ext_bg_point": {"f1": f1e, "recall": rece.tolist(), "fa_per_hour": fae},
                "drop_vs_test": c["test_f1"] - f1}
            # 포함·제외 하위 분석 (시험셋 문턱값 고정)
            for k in ("bg_near_silent", "webm_origin"):
                s2 = sel & ~flag[k]
                f2, r2, fa2 = seed_mean(P_fix, y, s2)
                c["defs"][d]["excl_" + k] = {"f1": f2, "recall": r2.tolist(),
                                             "fa_per_hour": fa2, "n": int(s2.sum()),
                                             "n_removed": int((sel & flag[k]).sum())}
        # 배경 오경보: 거의 무음 창만 / 그 외
        c["sub"]["bg_fire"] = {
            "all": fire_rate(P_fix, is_bg), "n": int(is_bg.sum()),
            "near_silent": fire_rate(P_fix, is_bg & flag["bg_near_silent"]),
            "n_near_silent": int((is_bg & flag["bg_near_silent"]).sum()),
            "webm": fire_rate(P_fix, is_bg & flag["webm_origin"]),
            "n_webm": int((is_bg & flag["webm_origin"]).sum())}
        # siren 하위 종류 (실사용 siren 전체 121 — real_primary 밖도 포함)
        sir = (grp == "siren")
        subs = {}
        for i in np.flatnonzero(sir):
            subs.setdefault(M[i]["siren_sub"], []).append(i)
        c["sub"]["siren"] = {
            k: {"n": len(v), "recall": fire_rate(P_fix, np.isin(np.arange(len(M)), v), 0)}
            for k, v in subs.items()}
        has_siren = np.array([r["group"] == "siren" and "Siren" in r["siren_sub"].split(";")
                              for r in M])
        c["sub"]["siren_generic"] = {
            "with_Siren": {"n": int(has_siren.sum()), "recall": fire_rate(P_fix, has_siren, 0)},
            "subtype_only": {"n": int((sir & ~has_siren).sum()),
                             "recall": fire_rate(P_fix, sir & ~has_siren, 0)}}
        # scream 하위 라벨 (scream 그룹 130 + def_gap scream 37)
        def sc_key(r):
            labs = set(r["labels_all"].split(";"))
            core = "+".join(x for x in ("Screaming", "Yell", "Shout") if x in labs)
            if r["group"] == "def_gap":
                return "def_gap: " + core
            crowd = "Screaming" in labs and r["in_v1"] == "0"
            return core + (" (+군중·음악·웃음)" if crowd else "")
        scs = {}
        for i, r in enumerate(M):
            if r["group"] == "scream" or (r["group"] == "def_gap" and r["v1_class"] == "scream"):
                scs.setdefault(sc_key(r), []).append(i)
        c["sub"]["scream"] = {
            k: {"n": len(v), "recall": fire_rate(P_fix, np.isin(np.arange(len(M)), v), 2),
                "fire_any": fire_rate(P_fix, np.isin(np.arange(len(M)), v))}
            for k, v in scs.items()}
        # def_gap 발화율 (v1 양성 / 실사용 배경) — v1_floor_excluded 포함·제외
        dg = {}
        for cls in ("dog_bark", "scream"):
            m0 = (grp == "def_gap") & np.array([r["v1_class"] == cls for r in M])
            for tag, m in (("all", m0), ("excl_floor", m0 & ~flag["v1_floor_excluded"])):
                dg[f"{cls}/{tag}"] = {"n": int(m.sum()), "fire_any": fire_rate(P_fix, m),
                                      "fire_as_class": fire_rate(P_fix, m, EVENTS.index(cls))}
        c["sub"]["def_gap"] = dg
        res["configs"].append(c)

    # 순위 (사전 등록: 일치 여부만 본다)
    tf = [c["test_f1"] for c in res["configs"]]
    res["rank"] = {"names": [c["name"] for c in res["configs"]],
                   "test": rank_of(tf).tolist()}
    for d in ("v1", "real"):
        ef = [c["defs"][d]["f1"] for c in res["configs"]]
        res["rank"][d] = rank_of(ef).tolist()
        res["rank"]["spearman_" + d] = spearman(tf, ef)
        res["rank"]["identical_" + d] = res["rank"][d] == res["rank"]["test"]
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
