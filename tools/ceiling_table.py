#!/usr/bin/env python3
"""상한선 점검 보고 표 — 여러 구성을 **같은 평가 경로**로 한 표에 낸다.

두 모드. 환경이 갈리기 때문이다 (`panns_inference` 는 시스템 python3 에만
있고, ai8x 모델은 ai8x venv 에서만 돈다).

  --dump : (ai8x venv) 체크포인트를 추론해 로짓 npz 를 남긴다
  기본    : (numpy 만) npz 들을 모아 표를 만든다

npz 규약: `y`(정답), `fsids`, `clips`, `starts`, `logits` (N,5).
PANNs 쪽은 `ceiling_check.py` 가 남긴 `P`(527확률)에서 즉석 변환한다.

사용 (WSL2):
    # 1) ①·D-1(√) 로짓 덤프
    ~/ai8x-training/venv/bin/python tools/ceiling_table.py --dump \\
        "① mel s1:mel:data/safesound-mel-v1_qat_best.pth.tar" \\
        --out data/logits/

    # 2) 표
    python3 tools/ceiling_table.py --npz data/logits/*.npz \\
        --panns data/ceiling-panns-zs.npz --md docs/results/ceiling-check.md
"""

import argparse
import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# --dump 는 ai8x venv 에서 돌므로 ai8x·datasets 경로가 필요하다.
# 표 만들기는 numpy 만 있으면 되지만, 경로를 넣어 두어도 해가 없다.
AI8X = os.environ.get("AI8X_DIR") or os.path.expanduser("~/ai8x-training")
sys.path[:0] = [AI8X, os.path.join(REPO, "datasets"),
                os.path.join(REPO, "tools"), os.path.join(REPO, "scripts")]

CLASSES = ["siren", "glass", "scream", "dog_bark", "background"]
BG = 4


# ────────────────────────────────────────────────────────── 덤프 (ai8x venv)
def dump(specs, out_dir, data, ai8x_dir, batch_size=128):
    import argparse as _ap
    import eval_confusion as EC
    os.makedirs(out_dir, exist_ok=True)
    for spec in specs:
        label, cfg, ck = spec.split(":", 2)
        args = _ap.Namespace(checkpoint=ck, config=cfg, data=data,
                             ai8x=ai8x_dir, simulate=False, bias=False,
                             batch_size=batch_size, split="test")
        print(f"추론: {label} ({cfg})")
        fs, cl, st, y, lg = EC.collect_logits(args, CLASSES, "test")
        p = os.path.join(out_dir, label.replace(" ", "_").replace("/", "_") + ".npz")
        np.savez_compressed(p, y=y, fsids=fs, clips=cl, starts=st, logits=lg,
                            label=label)
        print(f"  저장 {p}")


# ────────────────────────────────────────────────────────── 지표
def per_class_f1(y, p, k=5):
    out = []
    for c in range(k):
        tp = int(((y == c) & (p == c)).sum())
        fp = int(((y != c) & (p == c)).sum())
        fn = int(((y == c) & (p != c)).sum())
        out.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan"))
    return out


def row_for(label, y, fsids, logits):
    import eval_confusion as EC
    p = logits.argmax(1)
    fx = EC.fixed_fa_points(logits, y, CLASSES, hop_ms=250)
    t = fx.get("targets", {})
    of, ot, op = EC.majority_by_origin(fsids, y, p, len(CLASSES))
    return {
        "label": label,
        "f300": (t.get("300/h") or {}).get("macro_f1"),
        "f1000": (t.get("1000/h") or {}).get("macro_f1"),
        "argmax_macro_f1": EC.macro_f1(y, p, len(CLASSES)),
        "argmax_acc": float((y == p).mean()),
        "origin_macro_f1": EC.macro_f1(ot, op, len(CLASSES)),
        "n_origin": int(len(ot)),
        "class_f1": per_class_f1(y, p),
        "r300": (t.get("300/h") or {}).get("recall"),
    }


def fmt(v, w=8, d=4):
    return f"{'—':>{w}}" if v is None or (isinstance(v, float) and np.isnan(v)) \
        else f"{v:>{w}.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("specs", nargs="*", help="--dump 용 '이름:config:체크포인트'")
    ap.add_argument("--out", default="data/logits")
    ap.add_argument("--data", default=os.path.expanduser("~/ai8x-training/data"))
    ap.add_argument("--ai8x", default=os.path.expanduser("~/ai8x-training"))
    ap.add_argument("--npz", nargs="*", default=[])
    ap.add_argument("--panns", default=None, help="ceiling_check 의 npz")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()

    if a.dump:
        dump(a.specs, a.out, a.data, a.ai8x)
        return

    import ceiling_check as CC
    rows = []
    if a.panns and os.path.isfile(a.panns):
        z = np.load(a.panns, allow_pickle=True)
        lg = CC.zeroshot_logits(z["P"], z["idx"].item())
        rows.append(row_for("PANNs zero-shot", z["y"], z["fsids"], lg))
    files = []
    for pat in a.npz:
        files += sorted(glob.glob(pat))
    for f in files:
        z = np.load(f, allow_pickle=True)
        rows.append(row_for(str(z["label"]), z["y"], z["fsids"], z["logits"]))

    if not rows:
        sys.exit("[에러] 표에 넣을 것이 없다 (--npz / --panns)")

    out = []

    def say(s=""):
        print(s)
        out.append(s)

    say("| 구성 | @300/h | @1000/h | argmax macro-F1 | argmax 정확도 "
        "| 원본 다수결 macro-F1 |")
    say("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        say(f"| {r['label']} | {fmt(r['f300'],0)} | {fmt(r['f1000'],0)} "
            f"| {r['argmax_macro_f1']:.4f} | {100*r['argmax_acc']:.1f}% "
            f"| {r['origin_macro_f1']:.4f} ({r['n_origin']}원본) |")
    say()
    say("**클래스별 F1 (argmax)**")
    say()
    say("| 구성 | " + " | ".join(f"`{c}`" for c in CLASSES) + " |")
    say("|---|" + "---:|" * len(CLASSES))
    for r in rows:
        say(f"| {r['label']} | "
            + " | ".join(f"{v:.3f}" if v == v else "—" for v in r["class_f1"])
            + " |")
    say()
    say("**클래스별 재현율 @300/h** (고정 오경보 지점 — 모델 비교의 기준)")
    say()
    say("| 구성 | " + " | ".join(f"`{c}`" for c in CLASSES) + " |")
    say("|---|" + "---:|" * len(CLASSES))
    for r in rows:
        rc = r["r300"] or {}
        say(f"| {r['label']} | "
            + " | ".join(f"{100*rc[c]:.1f}%" if c in rc else "—"
                         for c in CLASSES) + " |")

    if a.md:
        with open(a.md, "a", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(out) + "\n")
        print(f"\n덧붙임: {a.md}")


if __name__ == "__main__":
    main()
