#!/usr/bin/env python3
"""v4+D 교사 — PANNs CNN14 임베딩(2048) + 선형 프로브(5) → 로짓 캐시 (v4-distill-design.md 3.1·8절).

  --embed   v4 학습 창(전 클래스, index.csv 순서)과 라벨 없는 창의 PANNs 임베딩을 캐시한다 (float16).
            v4: <v4>/teacher/emb_<cls>.npy, 라벨 없음: <v4u>/train/unlabeled/emb.npy. 입력 int8/128, 무증강 창.
  --probe   프로브 학습: v4 **라벨 있는 학습 창만** (외부 세트·시험셋 금지). 원본 단위(fsid blake2b < 0.1) 10% 검증으로
            L2·에폭 선택 (검증 macro-F1 argmax). 손실 = v4 클래스 가중치 CE. 결과: <v4>/teacher/probe.npz, teacher.json.
  --logits  프로브 로짓을 v4 라벨 창(teacher_<cls>.npy, (n,5) float32) 과 라벨 없는 창(teacher.npy) 에 쓴다.
            T=2 유효 클래스 수(exp 엔트로피 평균)·클래스별 평균 확률을 teacher.json 에 적는다.

사용 (WSL2, 시스템 python3 — panns_inference·torch):
    python3 tools/teacher_cache.py --embed --probe --logits
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "tools")]
import label_audit as LA                                    # noqa: E402

CLASSES = LA.CLASSES
V4 = os.path.join(REPO, "data", "processed", "safesound_v4")
V4U = os.path.join(REPO, "data", "processed", "safesound_v4u")
TDIR = os.path.join(V4, "teacher")
KD_T = 2.0
VAL_FRAC = 0.1


def fsid_unit(fsid):
    h = hashlib.blake2b(str(fsid).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def rows_of(root, split, cls):
    with open(os.path.join(root, split, cls, "index.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def panns():
    import torch
    from panns_inference.models import Cnn14
    torch.set_num_threads(6)
    m = Cnn14(sample_rate=LA.SR, window_size=512, hop_size=160, mel_bins=64, fmin=50, fmax=LA.SR // 2, classes_num=527)
    m.load_state_dict(torch.load(LA.CKPT, map_location="cpu", weights_only=False)["model"]); m.eval()
    return torch, m


def embed_rows(torch, m, root, split, cls, rows, out_path, bs=32):
    if os.path.isfile(out_path):
        z = np.load(out_path, mmap_mode="r")
        if z.shape[0] == len(rows):
            print(f"  캐시 있음 {out_path} {z.shape}"); return
    maps, E, buf, t0 = {}, [], [], time.time()

    def flush():
        if buf:
            X = torch.from_numpy(np.stack(buf).astype(np.float32) / 128.0)
            with torch.no_grad():
                E.append(m(X)["embedding"].numpy().astype(np.float16))
            buf.clear()
    for k, r in enumerate(rows):
        buf.append(LA.shard_window(root, split, cls, r, maps))
        if len(buf) >= bs:
            flush()
            if (k + 1) % (bs * 100) == 0:
                print(f"    {k + 1:,}/{len(rows):,} {time.time() - t0:.0f}s", flush=True)
    flush()
    np.save(out_path, np.concatenate(E))
    print(f"  {out_path} {len(rows):,}창 {time.time() - t0:.0f}s", flush=True)


def embed():
    torch, m = panns()
    os.makedirs(TDIR, exist_ok=True)
    for cls in CLASSES:
        rows = rows_of(V4, "train", cls)
        embed_rows(torch, m, V4, "train", cls, rows, os.path.join(TDIR, f"emb_{cls}.npy"))
    rows = rows_of(V4U, "train", "unlabeled")
    embed_rows(torch, m, V4U, "train", "unlabeled", rows, os.path.join(V4U, "train", "unlabeled", "emb.npy"))


def load_labeled():
    E, y, unit = [], [], []
    for c, cls in enumerate(CLASSES):
        rows = rows_of(V4, "train", cls)
        e = np.load(os.path.join(TDIR, f"emb_{cls}.npy")).astype(np.float32)
        assert e.shape[0] == len(rows), (cls, e.shape, len(rows))
        E.append(e); y += [c] * len(rows); unit += [fsid_unit(r["fsid"]) for r in rows]
    return np.concatenate(E), np.array(y), np.array(unit)


def class_weights(counts):
    raw = np.array([1.0 / c for c in counts]); s = (np.array(counts) * raw).sum() / sum(counts)
    return raw / s


def macro_f1(y, p, k=5):
    f = []
    for c in range(k):
        tp = ((y == c) & (p == c)).sum(); fp = ((y != c) & (p == c)).sum(); fn = ((y == c) & (p != c)).sum()
        f.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f))


def probe():
    import torch
    torch.manual_seed(1)
    E, y, unit = load_labeled()
    counts = [int((y == c).sum()) for c in range(5)]
    w = torch.tensor(class_weights(counts), dtype=torch.float32)
    val = unit < VAL_FRAC
    print(f"프로브 학습: 창 {len(y):,} (검증 {int(val.sum()):,}, 원본 단위 {VAL_FRAC:.0%}), 창 수 {counts}, 가중치 {np.round(w.numpy(), 3)}")
    mu, sd = E[~val].mean(0), E[~val].std(0) + 1e-6          # 학습 분할 통계로 표준화 (교사 내부 전처리; 캐시에 같이 저장)
    Xtr, Xva = torch.from_numpy((E[~val] - mu) / sd), torch.from_numpy((E[val] - mu) / sd)
    ytr, yva = torch.from_numpy(y[~val]), torch.from_numpy(y[val])
    best = None
    for wd in (1e-4, 1e-3, 1e-2, 1e-1):
        best_wd = None
        lin = torch.nn.Linear(E.shape[1], 5)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-3, weight_decay=wd)
        crit = torch.nn.CrossEntropyLoss(weight=w)
        for ep in range(1, 301):
            perm = torch.randperm(len(ytr))
            for i in range(0, len(perm), 1024):
                idx = perm[i:i + 1024]
                opt.zero_grad(); loss = crit(lin(Xtr[idx]), ytr[idx]); loss.backward(); opt.step()
            if ep % 10 == 0:
                with torch.no_grad():
                    f1 = macro_f1(yva.numpy(), lin(Xva).argmax(1).numpy())
                if best_wd is None or f1 > best_wd[0]:
                    best_wd = (f1, wd, ep, {k: v.clone() for k, v in lin.state_dict().items()})
        print(f"  wd {wd:g}: 최고 검증 macro-F1 (argmax) {best_wd[0]:.4f} @ epoch {best_wd[2]}", flush=True)
        if best is None or best_wd[0] > best[0]:
            best = best_wd
    f1, wd, ep, sd_ = best
    with torch.no_grad():
        lin = torch.nn.Linear(E.shape[1], 5); lin.load_state_dict(sd_)
        pv = lin(Xva).argmax(1).numpy(); ptr = lin(Xtr).argmax(1).numpy()
        rec = [float(((yva.numpy() == c) & (pv == c)).sum() / max(1, (yva.numpy() == c).sum())) for c in range(5)]
    print(f"선택: wd {wd:g}, epoch {ep}, 검증 macro-F1 {f1:.4f}, 학습 macro-F1 {macro_f1(y[~val], ptr):.4f}, 검증 recall {np.round(rec, 3)}")
    np.savez(os.path.join(TDIR, "probe.npz"), W=sd_["weight"].numpy(), b=sd_["bias"].numpy(), mu=mu, sd=sd)
    info = {"probe": {"wd": wd, "epoch": ep, "val_frac": VAL_FRAC, "val_n": int(val.sum()), "val_macro_f1_argmax": f1,
                      "train_macro_f1_argmax": macro_f1(y[~val], ptr), "val_recall": rec, "counts": counts,
                      "class_weights": [float(x) for x in w], "data": "v4 train labeled windows only (no external, no test)"}}
    with open(os.path.join(TDIR, "teacher.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print("저장:", os.path.join(TDIR, "probe.npz"))


def apply_probe(E):
    z = np.load(os.path.join(TDIR, "probe.npz"))
    return ((E.astype(np.float32) - z["mu"]) / z["sd"]) @ z["W"].T + z["b"]


def eff_classes(logits, T):
    l = logits / T; l = l - l.max(1, keepdims=True); p = np.exp(l); p /= p.sum(1, keepdims=True)
    ent = -(p * np.log(p + 1e-12)).sum(1)
    return float(np.exp(ent).mean()), p.mean(0).tolist()


def logits():
    info = json.load(open(os.path.join(TDIR, "teacher.json"), encoding="utf-8"))
    stats = {}
    allL = []
    for cls in CLASSES:
        L = apply_probe(np.load(os.path.join(TDIR, f"emb_{cls}.npy"))).astype(np.float32)
        np.save(os.path.join(TDIR, f"teacher_{cls}.npy"), L); allL.append(L)
        stats[cls] = {"n": int(L.shape[0]), "argmax_is_cls": float((L.argmax(1) == CLASSES.index(cls)).mean())}
    allL = np.concatenate(allL)
    ec, pm = eff_classes(allL, KD_T)
    stats["labeled"] = {"eff_classes_T2": ec, "mean_prob_T2": pm, "logit_std": float(allL.std())}
    Lu = apply_probe(np.load(os.path.join(V4U, "train", "unlabeled", "emb.npy"), mmap_mode="r")).astype(np.float32)
    np.save(os.path.join(V4U, "train", "unlabeled", "teacher.npy"), Lu)
    ecu, pmu = eff_classes(Lu, KD_T)
    stats["unlabeled"] = {"n": int(Lu.shape[0]), "eff_classes_T2": ecu, "mean_prob_T2": pmu,
                          "argmax_hist": [int((Lu.argmax(1) == c).sum()) for c in range(5)]}
    info["logits"] = stats; info["kd_T"] = KD_T
    with open(os.path.join(TDIR, "teacher.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true"); ap.add_argument("--probe", action="store_true"); ap.add_argument("--logits", action="store_true")
    a = ap.parse_args()
    if a.embed:
        embed()
    if a.probe:
        probe()
    if a.logits:
        logits()
