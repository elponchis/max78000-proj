#!/usr/bin/env python3
"""외부 세트 낙폭 진단 — 시험셋을 변형해 낙폭이 재현되는지 (혼합음 / 코덱).

사전 등록: `docs/results/external-drop-diagnosis.md` 1절. 조건·기준 조건 선택·
판정 규칙은 거기에 고정돼 있다 — 여기서 바꾸지 말 것.

평가 경로는 `eval_external.py` 와 같다: 변형한 창을 샤드 형식으로 감싸
(`data/external_eval/diag/<조건>/`) `collect_logits` 를 그대로 부른다.

사용 (WSL2, ai8x venv, OPENBLAS_NUM_THREADS=1):
    python tools/diag_external_drop.py
"""

import csv
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "tools"), os.path.join(REPO, "datasets")]
import eval_external as E                                      # noqa: E402

TEST = os.path.join(E.AI8X, "data", "SafeSound", "test")
DIAG = os.path.join(E.WORK, "diag")
MARGIN, WIN = 1600, 16384
SNRS = (20, 10, 5, 0)
SEED = 78000
RUNS = [r for r in E.RUNS if r[1] in ("wave", "mel_inc", "mel_u800")]
CODECS = {   # 이름: (부호화 인자, 확장자)
    "resample": (["-ac", "2", "-ar", "44100", "-c:a", "pcm_f32le"], "wav"),
    "aac": (["-ac", "2", "-ar", "44100", "-c:a", "aac", "-b:a", "128k"], "m4a"),
    "opus": (["-ac", "2", "-ar", "48000", "-c:a", "libopus", "-b:a", "160k"], "webm"),
    "aac32": (["-ac", "1", "-ar", "44100", "-c:a", "aac", "-b:a", "32k"], "m4a"),
}


def load_test():
    """클래스별 (저장 행 (n, STORE) int8, index 행 목록). Dataset 과 같은 순서."""
    out = {}
    for c in E.CLASSES:
        d = os.path.join(TEST, c)
        with open(os.path.join(d, "index.csv"), encoding="utf-8") as f:
            idx = list(csv.DictReader(f))
        sh = {}
        rows = np.stack([np.asarray(sh.setdefault(
            r["shard"], np.load(os.path.join(d, f"shard_{int(r['shard']):04d}.npy"),
                                mmap_mode="r"))[int(r["row"])]) for r in idx])
        out[c] = (rows, idx)
    return out


def write_cond(name, data):
    """조건 하나를 샤드 루트로 쓴다. data: {클래스: (저장 행, index 행)}."""
    root = os.path.join(DIAG, name)
    for c, (rows, idx) in data.items():
        d = os.path.join(root, "SafeSound", "test", c)
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "shard_0000.npy"), rows.astype(np.int8))
        with open(os.path.join(d, "index.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["shard", "row", "clip_id", "fsid", "start_sample",
                        "left_margin", "right_margin", "note"])
            for i, r in enumerate(idx):
                w.writerow([0, i, r["clip_id"], r["fsid"], r["start_sample"], 0, 0, name])
    return root


def crop(rows):
    return rows[:, MARGIN:MARGIN + WIN]


def rms(w):
    return np.sqrt((w.astype(np.float64) ** 2).mean(axis=-1))


def zero_frac(w):
    return (w == 0).mean(axis=1)


# ─────────────────────────────────────────────────────────────── (가) 혼합
def mix(ev_rows, bg_rows, pair, snr_db):
    """이벤트 저장 행 + 스케일한 배경 저장 행. 게인은 **자른 창**의 RMS 로 정한다."""
    e = ev_rows.astype(np.float64)
    b = bg_rows[pair].astype(np.float64)
    re, rb = rms(crop(ev_rows)), rms(crop(bg_rows[pair]))
    g = re / (np.maximum(rb, 1e-9) * 10.0 ** (snr_db / 20.0))
    return np.clip(np.round(e + g[:, None] * b), -128, 127).astype(np.int8)


# ─────────────────────────────────────────────────────────────── (나) 코덱
def ffmpeg(args):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + args,
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"[에러] ffmpeg: {r.stderr[-400:]}")


def roundtrip(rows, enc, ext):
    """저장 행들을 이어 붙여 한 번에 왕복 변환. 반환: (int8 행, 보정한 지연 샘플)."""
    x = (rows.astype(np.float32) / 127.0).reshape(-1)
    with tempfile.TemporaryDirectory() as td:
        a, m, o = (os.path.join(td, n) for n in ("in.raw", "mid." + ext, "out.raw"))
        x.tofile(a)
        ffmpeg(["-f", "f32le", "-ar", "16000", "-ac", "1", "-i", a] + enc + [m])
        ffmpeg(["-i", m, "-ac", "1", "-ar", "16000", "-f", "f32le", o])
        y = np.fromfile(o, dtype=np.float32)
    # 지연: 앞 20초의 상호상관 최댓값 (±4000 샘플)
    n = min(len(x), len(y), 320000)
    L = 4000
    ref = x[L:n - L].astype(np.float64)
    best, lag = -np.inf, 0
    for d in range(-L, L + 1):
        v = float(np.dot(ref[::7], y[L + d:n - L + d:7]))    # 성긴 내적으로 훑는다
        if v > best:
            best, lag = v, d
    y = y[lag:] if lag >= 0 else np.concatenate([np.zeros(-lag, np.float32), y])
    y = np.pad(y, (0, max(0, len(x) - len(y))))[:len(x)]
    q = np.clip(np.round(y * 127.0), -128, 127).astype(np.int8)
    return q.reshape(rows.shape), lag


# ─────────────────────────────────────────────────────────────── 채점
def evaluate(conds, has_bg):
    """조건별·구성별 시드 평균 recall[4] (+ 배경이 있으면 오경보/h)."""
    res = {}
    for name, config, prefix, n_seed in RUNS:
        acc = {c: {"recall": [], "fa": []} for c in conds}
        base = {"recall": [], "f1": []}
        for s in range(1, n_seed + 1):
            ck = E.checkpoint(prefix, s)
            _c, yt, lt = E.logits_for(config, ck, os.path.join(E.AI8X, "data"), "test")
            thr = E.thr_at(lt, yt == E.BG)
            f1, rec, _fa = E.metrics(yt, E.predict(lt, thr))
            base["recall"].append(rec[:4])
            base["f1"].append(f1)
            for c, root in conds.items():
                _c2, y, lg = E.logits_for(config, ck, root, "diag-" + c)
                _f, r, fa = E.metrics(y, E.predict(lg, thr))
                acc[c]["recall"].append(r[:4])
                acc[c]["fa"].append(fa)
            print(f"  {name} s{s} 완료", flush=True)
        res[name] = {"clean": np.mean(base["recall"], axis=0).tolist(),
                     "clean_f1": float(np.mean(base["f1"])),
                     "conds": {c: {"recall": np.mean(v["recall"], axis=0).tolist(),
                                   "fa_per_hour": (float(np.mean(v["fa"]))
                                                   if has_bg.get(c) else None)}
                               for c, v in acc.items()}}
    return res


def main():
    os.makedirs(DIAG, exist_ok=True)
    T = load_test()
    rng = np.random.default_rng(SEED)
    out = {"snrs": SNRS, "n": {c: len(T[c][1]) for c in E.CLASSES}}

    # 무손실 16 kHz 왕복이 비트 단위로 같은가 (사전 등록 조건)
    probe = T["glass"][0][:40]
    q, lag = roundtrip(probe, ["-c:a", "pcm_f32le"], "wav")
    ok = bool(np.array_equal(q, probe)) and lag == 0
    print(f"무손실 왕복 점검: {'PASS' if ok else 'FAIL'} (지연 {lag})")
    if not ok:
        sys.exit(1)

    # 배경 풀
    bg_rows, bg_idx = T["background"]
    hard = np.array([r["note"].startswith("hard_neg:") for r in bg_idx])
    usable = zero_frac(crop(bg_rows)) <= 0.5
    pools = {"gen": np.flatnonzero(~hard & usable), "hard": np.flatnonzero(hard & usable)}
    out["pool"] = {k: int(len(v)) for k, v in pools.items()}
    out["pool"]["bg_total"] = int(len(bg_idx))
    out["pool"]["hard_total"] = int(hard.sum())
    pairs = {p: {c: rng.choice(pools[p], len(T[c][1])) for c in E.EVENTS} for p in pools}

    conds, has_bg, zf = {}, {}, {}
    zf["clean"] = {c: float(np.median(zero_frac(crop(T[c][0])))) for c in E.EVENTS}
    mixed = {}
    for p in pools:
        for snr in SNRS:
            nm = f"mix-{p}-{snr}"
            mixed[nm] = {c: (mix(T[c][0], bg_rows, pairs[p][c], snr), T[c][1])
                         for c in E.EVENTS}
            conds[nm] = write_cond(nm, mixed[nm])
            has_bg[nm] = False
            zf[nm] = {c: float(np.median(zero_frac(crop(mixed[nm][c][0]))))
                      for c in E.EVENTS}

    # 외부 세트의 클래스별 int8 0 비율 중앙값 (v1 양성) → 기준 조건
    X, M = E.load_external()
    yv = E.labels(M, "v1")
    zf["external"] = {c: float(np.median(zero_frac(np.asarray(X)[yv == i])))
                      for i, c in enumerate(E.EVENTS)}
    ref = {c: min(SNRS, key=lambda s: abs(zf[f"mix-gen-{s}"][c] - zf["external"][c]))
           for c in E.EVENTS}
    out["zero_frac"], out["ref_snr"] = zf, ref
    print("기준 SNR (일반 풀, 0 비율 기준):", ref, flush=True)

    # (나) 코덱 — 전 클래스
    out["codec_lag"] = {}
    coded = {}
    for nm, (enc, ext) in CODECS.items():
        coded[nm], lags = {}, {}
        for c in E.CLASSES:
            q, lag = roundtrip(T[c][0], enc, ext)
            coded[nm][c] = (q, T[c][1])
            lags[c] = lag
        out["codec_lag"][nm] = lags
        conds[nm] = write_cond(nm, coded[nm])
        has_bg[nm] = True
        zf[nm] = {c: float(np.median(zero_frac(crop(coded[nm][c][0])))) for c in E.EVENTS}
        print(f"  코덱 {nm}: 지연 {lags}", flush=True)

    # (다) 결합 — 기준 조건으로 섞은 뒤 aac. 이벤트만
    comb = {}
    for c in E.EVENTS:
        q, _lag = roundtrip(mixed[f"mix-gen-{ref[c]}"][c][0], *CODECS["aac"])
        comb[c] = (q, T[c][1])
    conds["comb"] = write_cond("comb", comb)
    has_bg["comb"] = False

    out["results"] = evaluate(conds, has_bg)

    # 외부 세트 recall (v1 정의) — external_eval.json 에서
    ext = json.load(open(os.path.join(E.WORK, "external_eval.json"), encoding="utf-8"))
    out["external_recall"] = {c["name"]: c["defs"]["v1"]["recall"][:4]
                              for c in ext["configs"] if c["name"] in out["results"]}

    def ratio(cfg, rec):
        cl, ex = out["results"][cfg]["clean"], out["external_recall"][cfg]
        return [(cl[k] - rec[k]) / (cl[k] - ex[k]) for k in range(4)]

    for cfg, r in out["results"].items():
        refrec = [r["conds"][f"mix-gen-{ref[c]}"]["recall"][k]
                  for k, c in enumerate(E.EVENTS)]
        r["ref_mix_recall"] = refrec
        r["ratio"] = {"ref_mix": ratio(cfg, refrec)}
        for nm in conds:
            r["ratio"][nm] = ratio(cfg, r["conds"][nm]["recall"])
    summ = {nm: float(np.mean([out["results"][cfg]["ratio"][nm]
                               for cfg in out["results"]]))
            for nm in out["results"][RUNS[0][0]]["ratio"]}
    out["summary_ratio"] = summ
    with open(os.path.join(E.WORK, "diag_drop.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("요약 재현 비:", {k: round(v, 3) for k, v in summ.items()})


if __name__ == "__main__":
    main()
