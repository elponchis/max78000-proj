#!/usr/bin/env python3
"""④(원본 파형)가 v4·v4+D 에서 학습을 따라가지 못한 것 — 출처별 **학습 재현율**로 본다 (사용자 지시 2026-10-11 5항).

무엇을 재나: 각 판(v3.1 / v4 / v4+D)의 ④ 와 간격 400 체크포인트(시드 1~3)를 **그 판의 학습 창**에 무증강으로
돌려 argmax 재현율을 클래스 × 출처(FSD50K / US8K / ESC-50 / AudioSet)로 낸다. 학습 창을 못 맞추면 "일반화" 가 아니라
"학습 자체를 못 따라간" 것이다. 함께 학습 로그의 마지막 에폭 Top1(학습·검증·시험)과 시험셋 @300/h 재현율(testL,
`v4_eval.json`)을 붙인다. 증강·문턱값은 끼지 않는다 — 학습 적합도만 본다.

사용 (WSL2, ai8x venv, OPENBLAS_NUM_THREADS=1):  python tools/wave_v4_diag.py   → data/external_eval/wave_v4_diag.json
"""
import argparse
import collections
import contextlib
import io
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "tools"), os.path.join(REPO, "datasets")]
import eval_external as E                                      # noqa: E402
import eval_confusion as EC                                    # noqa: E402
import label_audit as LA                                       # noqa: E402

# 판 → (④ 접두, 간격 400 접두, 학습 루트 이름)
EDITIONS = [("v3.1", "safesound-v31-wave", "safesound-v31-melh400", "SafeSoundV31"),
            ("v4", "safesound-v4-wave", "safesound-v4-melh400", "SafeSoundV4"),
            ("v4+D", "safesound-v4d-wave", "safesound-v4d-melh400", "SafeSoundV4")]
CONFIGS = [("④", "wave"), ("간격 400", "mel_h400")]
SOURCES = ("FSD50K", "US8K", "ESC-50", "AudioSet")
OUT = os.path.join(E.WORK, "wave_v4_diag.json")
LOGS = os.path.join(REPO, "data", "logs-local")
CLASSES = EC.CLASSES if hasattr(EC, "CLASSES") else ("siren", "glass", "scream", "dog_bark", "background")


def source_of(clip, note):
    if note.startswith("audioset"):
        return "AudioSet"
    if re.match(r"^\d+-\d+-\d+-\d+\.wav$", clip):
        return "US8K"
    if re.match(r"^\d-\d+-[A-Z]-\d+\.wav$", clip):
        return "ESC-50"
    return "FSD50K"


def train_logits(config, ck, root_name):
    """학습 창(무증강) 로짓 — 체크포인트 mtime 기준 캐시. 반환 (clips, starts, y, logits)."""
    import torch
    import ai8x
    os.makedirs(os.path.join(E.WORK, "logits"), exist_ok=True)
    key = os.path.basename(os.path.dirname(ck)).split("___")[0]
    path = os.path.join(E.WORK, "logits", f"{key}.{config}.train.npz")
    if os.path.isfile(path) and os.path.getmtime(path) >= os.path.getmtime(ck):
        z = np.load(path, allow_pickle=False)
        return z["clips"], z["starts"], z["y"], z["logits"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = EC.load_model(ck, len(CLASSES), E.AI8X, False, False, config=config)
    if "불일치" in buf.getvalue():
        sys.exit(f"[에러] state_dict 불일치 — {ck}\n{buf.getvalue()[-400:]}")
    _root, kw = EC.CONFIG_DATA.get(config, ("SafeSound", {}))
    ds = EC.dataset_class(config)(os.path.join(E.AI8X, "data", root_name), "train",
                                  transform=ai8x.normalize(args=argparse.Namespace(act_mode_8bit=False)),
                                  augment=False, **kw)
    clips, starts, ys, lg = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(ds), 256):
            xs, ts = [], []
            for j in range(i, min(i + 256, len(ds))):
                x, t = ds[j]
                xs.append(x); ts.append(int(t) if not hasattr(t, "shape") or t.ndim == 0 else int(t))
                c, _f, s = ds.meta[j]
                clips.append(str(c)); starts.append(int(s))
            lg.append(model(torch.stack(xs)).numpy()); ys += ts
    clips, starts, y, lg = np.array(clips), np.array(starts), np.array(ys), np.concatenate(lg)
    np.savez(path, clips=clips, starts=starts, y=y, logits=lg)
    return clips, starts, y, lg


def notes_for(root_name):
    root = os.path.join(REPO, "data", "processed", {"SafeSoundV31": "safesound_v31", "SafeSoundV4": "safesound_v4"}[root_name])
    out = {}
    for cls in CLASSES:
        for r in LA.read_index(root, "train", cls):
            out[(r["clip_id"], int(r["start_sample"]))] = r.get("note", "") or ""
    return out


def log_stats(name):
    """학습 로그 마지막 에폭: 학습 Top1(KD 판은 라벨 행만), 검증 Top1, 최종 시험 Top1."""
    p = os.path.join(LOGS, f"{name}.out")
    if not os.path.isfile(p):
        return {}
    tr = va = te = None
    for ln in open(p, encoding="utf-8", errors="replace"):
        if ln.startswith("Epoch: [149]"):
            m = re.search(r"Top1\s+([\d.]+)", ln)
            if m:
                if "Overall Loss" in ln:
                    tr = float(m.group(1))
                else:
                    va = float(m.group(1))
        elif ln.startswith("==> Top1:"):
            te = float(re.search(r"Top1:\s+([\d.]+)", ln).group(1))
    return {"train_top1": tr, "val_top1": va, "test_top1": te}


def main():
    out = {"editions": {}}
    v4eval = json.load(open(os.path.join(E.WORK, "v4_eval.json"), encoding="utf-8"))["editions"]
    for ed, pw, pm, root_name in EDITIONS:
        notes = notes_for(root_name)
        out["editions"][ed] = {}
        for (cname, config), pre in zip(CONFIGS, (pw, pm)):
            seeds = []
            for s in range(1, 4):
                try:
                    ck = E.checkpoint(pre, s)
                except SystemExit:
                    break
                clips, starts, y, lg = train_logits(config, ck, root_name)
                p = lg.argmax(1)
                src = np.array([source_of(c, notes.get((c, int(st)), "")) for c, st in zip(clips, starts)])
                rec = {}
                for c, cls in enumerate(CLASSES[:4]):
                    sel = y == c
                    rec[cls] = {"all": [int(((p == c) & sel).sum()), int(sel.sum())]}
                    for so in SOURCES:
                        s2 = sel & (src == so)
                        if s2.sum():
                            rec[cls][so] = [int(((p == c) & s2).sum()), int(s2.sum())]
                # 학습 창 중 배경이 아닌데 배경으로 간 비율, 배경이 이벤트로 간 비율
                ev = y < 4
                rec["event→bg"] = [int(((p == 4) & ev).sum()), int(ev.sum())]
                rec["bg→event"] = [int(((p != 4) & ~ev).sum()), int((~ev).sum())]
                name = pre if s == 1 else f"{pre}-s{s}"
                seeds.append({"name": name, "recall": rec, "log": log_stats(name)})
                print(f"  {ed} {cname} s{s}: " + "  ".join(
                    f"{cls} {100 * rec[cls]['all'][0] / rec[cls]['all'][1]:.1f}%" for cls in CLASSES[:4])
                      + f"  event→bg {100 * rec['event→bg'][0] / rec['event→bg'][1]:.1f}%  log {seeds[-1]['log']}", flush=True)
            if not seeds:
                continue
            ev4 = v4eval.get(ed, {}).get(cname, {}).get("seeds", [])
            test_rec = (np.mean([r["test_recall_L"] for r in ev4], axis=0).tolist() if ev4 else None)
            out["editions"][ed][cname] = {"seeds": seeds, "test_recall_L": test_rec}

    # ── 표: 클래스 × 출처 학습 재현율 (시드 평균) — ④ / 간격 400
    def mean_rec(S, cls, so):
        v = [s["recall"][cls][so][0] / s["recall"][cls][so][1] for s in S if so in s["recall"][cls]]
        return 100 * float(np.mean(v)) if v else None

    for cname, _cfg in CONFIGS:
        print(f"\n## {cname} — 학습 창 argmax 재현율 (시드 평균, %) / 창 수")
        print("| 클래스 | 출처 | 창 (v3.1 / v4) | v3.1 | v4 | v4+D |")
        print("|---|---|---|---:|---:|---:|")
        for cls in CLASSES[:4]:
            for so in ("all",) + SOURCES:
                cells, ns = [], []
                for ed, *_r in EDITIONS:
                    S = out["editions"][ed].get(cname, {}).get("seeds", [])
                    m = mean_rec(S, cls, so) if S else None
                    cells.append("—" if m is None else f"{m:.1f}")
                    ns.append(str(S[0]["recall"][cls][so][1]) if S and so in S[0]["recall"][cls] else "—")
                if all(c == "—" for c in cells):
                    continue
                print(f"| {cls} | {'전체' if so == 'all' else so} | {ns[0]} / {ns[1]} | " + " | ".join(cells) + " |")
        print("| (이벤트 창 → 배경 오예측) | | | " + " | ".join(
            f"{100 * np.mean([s['recall']['event→bg'][0] / s['recall']['event→bg'][1] for s in out['editions'][ed].get(cname, {}).get('seeds', [])]):.1f}"
            if out["editions"][ed].get(cname) else "—" for ed, *_r in EDITIONS) + " |")
        print("\n| 판 | 학습 Top1 (마지막 에폭, 라벨 행) | 검증 Top1 | 시험 Top1 (argmax) | 시험셋 @300/h recall testL siren/glass/scream/dog |")
        print("|---|---:|---:|---:|---|")
        for ed, *_r in EDITIONS:
            d = out["editions"][ed].get(cname)
            if not d:
                continue
            L = [s["log"] for s in d["seeds"] if s["log"]]
            f = lambda k: f"{np.mean([l[k] for l in L if l.get(k) is not None]):.1f}" if L else "—"
            tr = d["test_recall_L"]
            print(f"| {ed} | {f('train_top1')} | {f('val_top1')} | {f('test_top1')} | "
                  + ("/".join(f"{100 * x:.1f}" for x in tr[:4]) if tr else "—") + " |")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("저장:", OUT)


if __name__ == "__main__":
    main()
