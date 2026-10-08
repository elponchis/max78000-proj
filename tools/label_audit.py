#!/usr/bin/env python3
"""라벨 감사 — PANNs 로 (1) 외부 세트 기준선, (2) 전 출처 양성 창의 클래스 점수 분포, (4) 청취 표본.
`docs/results/label-audit.md` 0절의 기준은 **실행 전에 고정**했다. 학습은 하지 않는다.

출처 (양성 창만; 외부 세트는 758창 전부):
    ext        ~/safesound-external/external_v1  (windows_int8.npy + meta.csv)
    v1_train   data/processed/safesound/train/{siren,glass,scream,dog_bark}   (FSD50K·US8K·ESC-50)
    v1_test    data/ceiling-panns-zs.npz 캐시 (ceiling_check.py 가 같은 조건으로 돌린 5,207창) 중 양성
    as_train   data/processed/safesound_v3/train/* 의 note 가 audioset 로 시작하는 행 (v3.1 포함 여부 표시)

PANNs 조건은 `tools/ceiling_check.py` 와 같다: Cnn14_16k_mAP=0.438.pth, 입력 int8/128, LABEL_SETS 그대로.

사용 (WSL2):
    python3 tools/label_audit.py --run             # 시스템 python3 (panns_inference) → data/label_audit/panns_*.npz
    ~/ai8x-training/venv/bin/python tools/label_audit.py --score   # 채점·표·청취 표본 (numpy 만)
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "tools"), os.path.join(REPO, "scripts")]
from prepare_safesound import LABEL_SETS          # noqa: E402  (복제하지 않는다)

SR, WIN, MARGIN = 16000, 16384, 1600
CLASSES = ["siren", "glass", "scream", "dog_bark", "background"]
EVENTS = CLASSES[:4]
BG = 4
V1 = os.path.join(REPO, "data", "processed", "safesound")
V3 = os.path.join(REPO, "data", "processed", "safesound_v3")
V31 = os.path.join(REPO, "data", "processed", "safesound_v31")
EXT = os.path.expanduser("~/safesound-external/external_v1")
AUDIT_DIR = os.path.expanduser("~/safesound-external/audit")
OUT = os.path.join(REPO, "data", "label_audit")
CKPT = os.path.expanduser("~/panns_data/Cnn14_16k_mAP=0.438.pth")
LABELS_CSV = os.path.expanduser("~/panns_data/class_labels_indices.csv")
TEST_CACHE = os.path.join(REPO, "data", "ceiling-panns-zs.npz")

# 0.2 절 기준값 (실행 전 고정)
VERY_LOW, LOW, HIGH = 0.02, 0.10, 0.50
N_LOW, N_HIGH, SEED = 15, 5, 78004


# ───────────────────────────────────────────────────────────── 창 읽기
def read_index(root, split, cls):
    with open(os.path.join(root, split, cls, "index.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def shard_window(root, split, cls, r, maps):
    sp = os.path.join(root, split, cls, f"shard_{int(r['shard']):04d}.npy")
    m = maps.get(sp)
    if m is None:
        m = np.load(sp, mmap_mode="r"); maps[sp] = m
    return np.asarray(m[int(r["row"])][MARGIN:MARGIN + WIN])


def ext_meta():
    with open(os.path.join(EXT, "meta.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def iter_source(src):
    """(cls_idx, clip, fsid, start, note, int8 창) 을 낸다."""
    maps = {}
    if src == "ext":
        X = np.load(os.path.join(EXT, "windows_int8.npy"), mmap_mode="r")
        for i, r in enumerate(ext_meta()):
            yield (-1, r["segment_id"], r["ytid"], int(r["win_start_sample"]), r["labels_all"], np.asarray(X[i]))
    elif src == "v1_train":
        for c, cls in enumerate(EVENTS):
            for r in read_index(V1, "train", cls):
                yield (c, r["clip_id"], r["fsid"], int(r["start_sample"]), r["note"], shard_window(V1, "train", cls, r, maps))
    elif src == "as_train":
        keep31 = {cls: {(r["clip_id"], r["start_sample"]) for r in read_index(V31, "train", cls)} for cls in EVENTS}
        for c, cls in enumerate(EVENTS):
            for r in read_index(V3, "train", cls):
                if not r["note"].startswith("audioset"):
                    continue
                note = r["note"] + (";in_v31=1" if (r["clip_id"], r["start_sample"]) in keep31[cls] else ";in_v31=0")
                yield (c, r["clip_id"], r["fsid"], int(r["start_sample"]), note, shard_window(V3, "train", cls, r, maps))
    else:
        raise ValueError(src)


# ───────────────────────────────────────────────────────────── PANNs 실행
def run(a):
    import torch
    from panns_inference.models import Cnn14
    torch.set_num_threads(a.threads)
    m = Cnn14(sample_rate=SR, window_size=512, hop_size=160, mel_bins=64, fmin=50, fmax=SR // 2, classes_num=527)
    m.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)["model"]); m.eval()
    os.makedirs(OUT, exist_ok=True)
    for src in a.sources:
        path = os.path.join(OUT, f"panns_{src}.npz")
        if os.path.isfile(path) and not a.force:
            print(f"[{src}] 캐시 있음 — 건너뜀 ({path})"); continue
        ys, clips, fsids, starts, notes, P, buf, t0 = [], [], [], [], [], [], [], time.time()

        def flush():
            if not buf:
                return
            X = torch.from_numpy(np.stack(buf).astype(np.float32) / 128.0)
            with torch.no_grad():
                P.append(m(X)["clipwise_output"].numpy().astype(np.float16))
            buf.clear()

        for c, clip, fsid, st, note, w in iter_source(src):
            ys.append(c); clips.append(clip); fsids.append(fsid); starts.append(st); notes.append(note); buf.append(w)
            if len(buf) >= a.batch_size:
                flush()
                if len(ys) % (a.batch_size * 10) == 0:
                    el = time.time() - t0
                    print(f"  [{src}] {len(ys)}창 {el:.0f}s ({1000 * el / len(ys):.0f} ms/창)", flush=True)
        flush()
        np.savez_compressed(path, y=np.array(ys), clips=np.array(clips), fsids=np.array(fsids), starts=np.array(starts),
                            notes=np.array(notes), P=np.concatenate(P) if P else np.zeros((0, 527), np.float16))
        print(f"[{src}] {len(ys)}창 {time.time() - t0:.0f}s → {path}", flush=True)


# ───────────────────────────────────────────────────────────── 채점
def label_names():
    with open(LABELS_CSV, encoding="utf-8") as f:
        return [r["display_name"] for r in csv.DictReader(f)]


def class_scores(P, names):
    """0.1 절: siren/glass/dog_bark 묶음 최대, scream = Screaming, shout = max(Yell, Shout), scream_v1 = 묶음 최대."""
    P = P.astype(np.float32)
    col = lambda l: P[:, names.index(l)]                                   # noqa: E731
    s = {c: P[:, [names.index(l) for l in LABEL_SETS[c]]].max(axis=1) for c in ("siren", "glass", "dog_bark")}
    s["scream"] = col("Screaming")
    s["shout"] = np.maximum(col("Yell"), col("Shout"))
    s["scream_v1"] = P[:, [names.index(l) for l in LABEL_SETS["scream"]]].max(axis=1)
    return s


def zeroshot(P, names):
    """ceiling_check.zeroshot_logits 와 같다: 이벤트 = 묶음 최대, 배경 = 1 − max(이벤트)."""
    P = P.astype(np.float32)
    ev = np.stack([P[:, [names.index(l) for l in LABEL_SETS[c]]].max(axis=1) for c in EVENTS], axis=1)
    return np.concatenate([ev, (1.0 - ev.max(axis=1))[:, None]], axis=1)


def load(src):
    z = np.load(os.path.join(OUT, f"panns_{src}.npz"), allow_pickle=False)
    return {k: z[k] for k in z.files}


def ap_of(score, pos):
    o = np.argsort(-score); p = pos[o]
    prec = np.cumsum(p) / np.arange(1, len(p) + 1)
    return float((prec * p).sum() / max(1.0, p.sum()))


def ext_eval(S, y, isbg, thr):
    """(macro-F1, recall[5], 오경보/h, mAP, AP[4]) — eval_external 과 같은 함수."""
    import eval_external as E
    sel = y >= 0
    f1, rec, fa = E.metrics(y[sel], E.predict(S, thr)[sel])
    aps = []
    for c in range(4):
        s = (y == c) | isbg
        aps.append(ap_of((S[:, c] - S[:, BG])[s], (y[s] == c).astype(float)))
    return f1, rec, fa, float(np.mean(aps)), aps


def our_logits(prefix, seeds):
    out = []
    for s in seeds:
        name = prefix if s == 1 else f"{prefix}-s{s}"
        z = np.load(os.path.join(REPO, "data", "external_eval", "logits", f"{name}.mel_h400.ext.npz"))
        out.append(z["logits"])
    return out


def our_test_thr(prefix, seeds):
    import eval_external as E
    out = []
    for s in seeds:
        name = prefix if s == 1 else f"{prefix}-s{s}"
        z = np.load(os.path.join(REPO, "data", "external_eval", "logits", f"{name}.mel_h400.test.npz"))
        out.append(E.thr_at(z["logits"], z["y"] == BG))
    return out


def baseline(names, M, report):
    """1절: PANNs vs 우리 모델(간격 400 v2.1·v3.1), 외부 @300/h 지점, 정의 real31·v1eq·v1."""
    import eval_external as E
    ext = load("ext")
    S_p = zeroshot(ext["P"], names)
    isbg = np.array([r["group"] == "background" for r in M])
    # PANNs 시험셋 문턱값(참고) + 시험셋 recall@300/h
    zt = np.load(TEST_CACHE, allow_pickle=True)
    S_t = zeroshot(zt["P"], names)
    thr_t = E.thr_at(S_t, zt["y"] == BG)
    f1_t, rec_t, _fa, _m, _a = ext_eval(S_t, zt["y"], zt["y"] == BG, thr_t)
    report["panns_test"] = {"f1_300": f1_t, "recall_300": rec_t.tolist(), "thr": thr_t}
    print(f"PANNs 시험셋 @300/h: F1 {f1_t:.4f} recall {np.round(100 * rec_t[:4], 1)}")

    models = {"v2.1 간격 400": ("safesound-v21-melh400", [1, 2, 3]), "v3.1 간격 400": ("safesound-v31-melh400", [1, 2, 3])}
    rows = []
    for d in ("real31", "v1eq", "v1"):
        y = E.labels(M, d)
        # PANNs — 외부 @300/h
        thr = E.thr_at(S_p, isbg)
        f1, rec, fa, mAP, aps = ext_eval(S_p, y, isbg, thr)
        rows.append(dict(model="PANNs zero-shot", definition=d, point="ext300", f1=f1, recall=rec.tolist(), fa=fa, map=mAP, ap=aps))
        f1b, recb, fab, _, _ = ext_eval(S_p, y, isbg, thr_t)
        rows.append(dict(model="PANNs zero-shot", definition=d, point="test300", f1=f1b, recall=recb.tolist(), fa=fab, map=mAP, ap=aps))
        for mname, (pre, seeds) in models.items():
            L = our_logits(pre, seeds)
            T = our_test_thr(pre, seeds)
            for point, thrs in (("ext300", [E.thr_at(lg, isbg) for lg in L]), ("test300", T)):
                res = [ext_eval(lg, y, isbg, t) for lg, t in zip(L, thrs)]
                rows.append(dict(model=mname, definition=d, point=point,
                                 f1=float(np.mean([r[0] for r in res])), recall=np.mean([r[1] for r in res], axis=0).tolist(),
                                 fa=float(np.mean([r[2] for r in res])), map=float(np.mean([r[3] for r in res])),
                                 ap=np.mean([r[4] for r in res], axis=0).tolist(),
                                 f1_seeds=[r[0] for r in res]))
    report["baseline"] = rows
    print("\n| 정의 | 지점 | 모델 | 외부 F1 | siren | glass | scream | dog_bark | 오경보/h | mAP |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        rc = r["recall"]
        print(f"| {r['definition']} | {r['point']} | {r['model']} | {r['f1']:.4f} | {100 * rc[0]:.1f} | {100 * rc[1]:.1f} | "
              f"{100 * rc[2]:.1f} | {100 * rc[3]:.1f} | {r['fa']:.0f} | {r['map']:.4f} |")


def audit(names, M, report):
    """2절: 출처 × 클래스 점수 분포. 외부는 real31·v1eq 양성, AudioSet scream 은 v3.1 포함/제외로 나눈다."""
    import eval_external as E
    groups = []            # (출처 표시, 클래스, 점수 dict 슬라이스, 인덱스)
    for src, label in (("v1_train", "FSD50K 계열 학습"), ("as_train", "AudioSet 학습분(v3)")):
        z = load(src); s = class_scores(z["P"], names)
        for c, cls in enumerate(EVENTS):
            sel = z["y"] == c
            groups.append((label, cls, s, sel, z))
            if src == "as_train" and cls == "scream":
                in31 = np.array([n.endswith("in_v31=1") for n in z["notes"]])
                groups.append((label + " · Screaming 보유(v3.1 포함)", cls, s, sel & in31, z))
                groups.append((label + " · Yell·Shout 만(v3.1 제외)", cls, s, sel & ~in31, z))
    zt = np.load(TEST_CACHE, allow_pickle=True)
    st = class_scores(zt["P"], names)
    for c, cls in enumerate(EVENTS):
        groups.append(("FSD50K 계열 시험", cls, st, zt["y"] == c, None))
    ext = load("ext"); se = class_scores(ext["P"], names)
    for d, label in (("real31", "외부(실사용 정의)"), ("v1eq", "외부(v1 정의)")):
        y = E.labels(M, d)
        for c, cls in enumerate(EVENTS):
            groups.append((label, cls, se, y == c, ext))
    # 외부의 Yell·Shout 만 구간(실사용 정의에서 음성)도 참고로
    y31 = E.labels(M, "real31")
    shout_only = np.array([r["group"] == "scream" and r["real_class"] != "scream" for r in M])
    groups.append(("외부 · Yell·Shout 만(실사용 음성)", "scream", se, shout_only, ext))

    rows = []
    print("\n| 출처 | 클래스 | n | p10 | p25 | 중앙 | p75 | p90 | <0.02 | <0.10 | ≥0.50 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, cls, s, sel, _z in groups:
        keys = [cls] if cls != "scream" else ["scream", "shout", "scream_v1"]
        for k in keys:
            v = s[k][sel]
            if not len(v):
                continue
            q = np.percentile(v, [10, 25, 50, 75, 90])
            row = dict(source=label, cls=cls, score=k, n=int(len(v)), q=q.tolist(),
                       very_low=float((v < VERY_LOW).mean()), low=float((v < LOW).mean()), high=float((v >= HIGH).mean()))
            rows.append(row)
            print(f"| {label} | {cls}{'' if k == cls else ' (' + k + ')'} | {len(v)} | " + " | ".join(f"{x:.3f}" for x in q)
                  + f" | {100 * row['very_low']:.1f}% | {100 * row['low']:.1f}% | {100 * row['high']:.1f}% |")
    report["audit"] = rows
    return groups


def listen(names, M, groups, report):
    """4절: 클래스마다 매우 낮음 15 + 높음 5, 전 출처 합쳐 무작위. 저장소 밖 (~/safesound-external/audit)."""
    import soundfile as sf
    rng = np.random.default_rng(SEED)
    os.makedirs(AUDIT_DIR, exist_ok=True)
    srcs = {"v1_train": load("v1_train"), "as_train": load("as_train"), "ext": load("ext")}
    zt = np.load(TEST_CACHE, allow_pickle=True)
    maps = {}
    # 창 하나를 다시 읽는 방법 (출처별)
    v1_rows = {cls: {(r["clip_id"], int(r["start_sample"])): r for r in read_index(V1, "train", cls)} for cls in EVENTS}
    v1t_rows = {cls: {(r["clip_id"], int(r["start_sample"])): r for r in read_index(V1, "test", cls)} for cls in EVENTS}
    v3_rows = {cls: {(r["clip_id"], int(r["start_sample"])): r for r in read_index(V3, "train", cls)} for cls in EVENTS}
    Xext = np.load(os.path.join(EXT, "windows_int8.npy"), mmap_mode="r")
    Mi = {r["segment_id"]: i for i, r in enumerate(M)}

    def window(src, cls, clip, start):
        if src == "ext":
            return np.asarray(Xext[Mi[clip]])
        if src == "v1_train":
            return shard_window(V1, "train", cls, v1_rows[cls][(clip, start)], maps)
        if src == "v1_test":
            return shard_window(V1, "test", cls, v1t_rows[cls][(clip, start)], maps)
        return shard_window(V3, "train", cls, v3_rows[cls][(clip, start)], maps)

    # 후보 풀: (src, cls, clip, start, score, shout, note)
    pool = {cls: [] for cls in EVENTS}
    import eval_external as E
    for src, z in (("v1_train", srcs["v1_train"]), ("as_train", srcs["as_train"])):
        s = class_scores(z["P"], names)
        for i in range(len(z["y"])):
            cls = EVENTS[int(z["y"][i])]
            pool[cls].append((src, cls, str(z["clips"][i]), int(z["starts"][i]), float(s[cls][i]), float(s["shout"][i]), str(z["notes"][i])))
    st = class_scores(zt["P"], names)
    for i in range(len(zt["y"])):
        if zt["y"][i] >= 4:
            continue
        cls = EVENTS[int(zt["y"][i])]
        pool[cls].append(("v1_test", cls, str(zt["clips"][i]), int(zt["starts"][i]), float(st[cls][i]), float(st["shout"][i]), ""))
    se = class_scores(srcs["ext"]["P"], names)
    y31 = E.labels(M, "real31")
    for i, r in enumerate(M):
        if y31[i] < 0 or y31[i] >= 4:
            continue
        cls = EVENTS[int(y31[i])]
        pool[cls].append(("ext", cls, r["segment_id"], int(r["win_start_sample"]), float(se[cls][i]), float(se["shout"][i]), r["labels_all"]))

    key_rows, summary = [], {}
    for cls in ["scream", "siren", "glass", "dog_bark"]:
        P_ = pool[cls]
        very = [p for p in P_ if p[4] < VERY_LOW]
        lowish = [p for p in P_ if VERY_LOW <= p[4] < LOW]
        high = [p for p in P_ if p[4] >= HIGH]
        pick_low = [very[i] for i in rng.permutation(len(very))[:N_LOW]]
        filled = 0
        if len(pick_low) < N_LOW:
            extra = [lowish[i] for i in rng.permutation(len(lowish))[:N_LOW - len(pick_low)]]
            filled = len(extra); pick_low += extra
        pick_high = [high[i] for i in rng.permutation(len(high))[:N_HIGH]]
        summary[cls] = dict(n_very_low=len(very), n_low=len(lowish), n_high=len(high), filled_from_low=filled)
        for bucket, picks in (("low", pick_low), ("high", pick_high)):
            for n, (src, _c, clip, start, sc, sh, note) in enumerate(picks, 1):
                fn = f"{cls}_{src}_{sc:.3f}_{bucket}{n:02d}.wav"
                w = window(src, cls, clip, start)
                sf.write(os.path.join(AUDIT_DIR, fn), (w.astype(np.int16) * 256), SR)
                key_rows.append(dict(file=fn, cls=cls, bucket=bucket, source=src, clip=clip, start_sample=start,
                                     score=round(sc, 4), shout_score=round(sh, 4), note=note))
    with open(os.path.join(AUDIT_DIR, "answer_key.csv"), "w", encoding="utf-8", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(key_rows[0].keys())); wtr.writeheader(); wtr.writerows(key_rows)
    report["listen"] = summary
    print("\n청취 표본:", json.dumps(summary, ensure_ascii=False), "→", AUDIT_DIR, f"({len(key_rows)} wav + answer_key.csv)")


def score(a):
    names = label_names()
    M = ext_meta()
    report = {"criteria": {"very_low": VERY_LOW, "low": LOW, "high": HIGH, "seed": SEED}}
    baseline(names, M, report)
    groups = audit(names, M, report)
    if not a.no_listen:
        listen(names, M, groups, report)
    path = os.path.join(OUT, "label_audit.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("저장:", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--sources", nargs="+", default=["ext", "v1_train", "as_train"])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-listen", action="store_true")
    a = ap.parse_args()
    if a.run:
        run(a)
    if a.score:
        score(a)
