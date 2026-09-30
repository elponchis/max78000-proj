#!/usr/bin/env python3
"""청취 감사 세트 두 종 — **블라인드 감사**와 **의심 창 목록**.

`docs/results/data-quality.md` 의 2·3번.

## A. 블라인드 감사 (`--mode blind`)

테스트 창을 클래스당 N개 무작위로 뽑되 **원본 ID 중복 없이** 하나씩만
고른다. 라벨도 예측도 **파일명에 넣지 않는다** — 판정자가 알면 편향된다.

  · wav 는 `NNN.wav` 로만 이름 붙는다 (셔플된 일련번호)
  · 정답·예측은 **별도 키 파일**(`_key_DO_NOT_OPEN.csv`)에 넣고, 채점할 때만 연다
  · 판정자가 채울 CSV: `audit.csv` (`heard`, `confidence` 1~3, `memo`)

채운 뒤 `--score audit.csv` 로 클래스별 라벨 오류율과 **원본 단위 95% CI** 를 낸다.

## B. 의심 창 목록 (`--mode suspect`)

여러 모델(① 3시드 + D-1(√) 3시드)의 예측이 **전부 라벨과 다르고** 평균
마진이 큰 창을 뽑는다. 여러 모델이 한목소리로 틀렸다면 **라벨 쪽을 의심**할
근거가 된다.

사용 (WSL2):
    # 로짓 덤프가 먼저 필요하다 (tools/ceiling_table.py --dump)
    python3 tools/listen_audit.py --mode blind --n 40 --out data/audit/blind
    python3 tools/listen_audit.py --mode suspect --npz 'data/logits/*.npz' \\
        --n 30 --out data/audit/suspect
    python3 tools/listen_audit.py --score data/audit/blind/audit.csv
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "tools"), os.path.join(REPO, "scripts")]

CLASSES = ["siren", "glass", "scream", "dog_bark", "background"]
WIN, MARGIN, SR = 16384, 1600, 16000


def load_index(root, split="test"):
    """테스트 창 전부 — (클래스idx, clip_id, fsid, start, 샤드경로, 행)."""
    out = []
    for t, cls in enumerate(CLASSES):
        d = os.path.join(root, split, cls)
        ip = os.path.join(d, "index.csv")
        if not os.path.isfile(ip):
            continue
        with open(ip, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out.append((t, r["clip_id"], r["fsid"], int(r["start_sample"]),
                            os.path.join(d, f"shard_{int(r['shard']):04d}.npy"),
                            int(r["row"])))
    return out


def write_wav(path, w):
    import soundfile as sf
    sf.write(path, np.asarray(w, dtype=np.float32) / 128.0, SR)


def mode_blind(a):
    rng = np.random.default_rng(a.seed)
    rows = load_index(a.root)
    picks = []
    for t, cls in enumerate(CLASSES):
        cand = [r for r in rows if r[0] == t]
        # **원본 ID 중복 없이** — 같은 녹음에서 두 창을 뽑으면 판정이 종속된다
        by = {}
        for r in cand:
            by.setdefault(r[2], []).append(r)
        fs = list(by)
        rng.shuffle(fs)
        for f in fs[:a.n]:
            picks.append(rng.choice(len(by[f])) if False else by[f][0])
    rng.shuffle(picks)

    os.makedirs(a.out, exist_ok=True)
    maps = {}
    with open(os.path.join(a.out, "audit.csv"), "w", newline="",
              encoding="utf-8-sig") as fa, \
         open(os.path.join(a.out, "_key_DO_NOT_OPEN.csv"), "w", newline="",
              encoding="utf-8-sig") as fk:
        wa, wk = csv.writer(fa), csv.writer(fk)
        wa.writerow(["id", "wav", "heard(siren/glass/scream/dog_bark/"
                     "background/unknown)", "confidence(1~3)", "memo"])
        wk.writerow(["id", "true_class", "clip_id", "fsid", "start_sample"])
        for i, (t, clip, fsid, st, sp, row) in enumerate(picks, 1):
            if sp not in maps:
                maps[sp] = np.load(sp, mmap_mode="r")
            w = np.asarray(maps[sp][row][MARGIN:MARGIN + WIN])
            name = f"{i:03d}.wav"
            write_wav(os.path.join(a.out, name), w)
            wa.writerow([i, name, "", "", ""])
            wk.writerow([i, CLASSES[t], clip, fsid, st])
    print(f"블라인드 감사 {len(picks)}개 → {a.out}/")
    print("  · 라벨·예측을 파일명에 넣지 않았다. `_key_DO_NOT_OPEN.csv` 는")
    print("    **채점할 때만** 연다.")
    print("  · `audit.csv` 의 heard/confidence 를 채운 뒤")
    print(f"    python3 tools/listen_audit.py --score {a.out}/audit.csv")


def mode_suspect(a):
    files = []
    for pat in a.npz:
        files += sorted(glob.glob(pat))
    if len(files) < 2:
        sys.exit("[에러] 로짓 npz 가 2개 이상 필요하다 (ceiling_table.py --dump)")
    Ls, labels = [], []
    y = fs = clips = starts = None
    for f in files:
        z = np.load(f, allow_pickle=True)
        Ls.append(z["logits"])
        labels.append(str(z["label"]))
        if y is None:
            y, fs, clips, starts = z["y"], z["fsids"], z["clips"], z["starts"]
        elif not np.array_equal(y, z["y"]):
            sys.exit(f"[에러] {f} 의 정답 순서가 다르다")
    print(f"모델 {len(Ls)}개: {', '.join(labels)}")

    preds = np.stack([L.argmax(1) for L in Ls])          # (M, N)
    # 전부 라벨과 다른 창
    allwrong = np.all(preds != y[None, :], axis=0)
    # 평균 마진 (예측 클래스 − 정답 클래스). 클수록 "확신하며 틀렸다"
    marg = np.mean([L[np.arange(len(y)), L.argmax(1)] - L[np.arange(len(y)), y]
                    for L in Ls], axis=0)
    # 모델끼리 **같은** 오답을 냈는가 (한목소리면 라벨 쪽 의심이 커진다)
    agree = np.all(preds == preds[0][None, :], axis=0)

    cand = np.flatnonzero(allwrong)
    cand = cand[np.argsort(-marg[cand])][:a.n]
    print(f"전부 오답인 창 {int(allwrong.sum()):,}/{len(y):,} "
          f"(그중 전 모델 동일 오답 {int((allwrong & agree).sum()):,})")

    rows = load_index(a.root)
    key = {(r[1], r[3]): r for r in rows}                 # (clip, start) → 행
    os.makedirs(a.out, exist_ok=True)
    maps = {}
    import prepare_safesound as P
    with open(os.path.join(a.out, "suspect.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rank", "wav", "label", "pred(공통)", "모델일치",
                    "평균마진", "clip_id", "fsid", "start_sample",
                    "verdict(라벨맞음/라벨틀림/모호)", "memo"])
        for rnk, i in enumerate(cand, 1):
            r = key.get((str(clips[i]), int(starts[i])))
            if r is None:
                continue
            sp, row = r[4], r[5]
            if sp not in maps:
                maps[sp] = np.load(sp, mmap_mode="r")
            seg = np.asarray(maps[sp][row][MARGIN:MARGIN + WIN])
            name = f"{rnk:02d}_{CLASSES[y[i]]}.wav"
            write_wav(os.path.join(a.out, name), seg)
            w.writerow([rnk, name, CLASSES[y[i]],
                        CLASSES[preds[0][i]] if agree[i] else "(불일치)",
                        "예" if agree[i] else "아니오",
                        f"{marg[i]:.2f}", clips[i], fs[i], starts[i], "", ""])
    print(f"의심 창 {min(a.n, len(cand))}개 → {a.out}/")
    print("  verdict: 라벨맞음 / 라벨틀림 / 모호")


def mode_score(a):
    d = os.path.dirname(os.path.abspath(a.score))
    key = {}
    with open(os.path.join(d, "_key_DO_NOT_OPEN.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key[r["id"]] = r
    got = {}
    with open(a.score, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            h = (r.get("heard(siren/glass/scream/dog_bark/background/unknown)")
                 or "").strip()
            if h:
                got[r["id"]] = (h, (r.get("confidence(1~3)") or "").strip())
    if not got:
        sys.exit("[에러] audit.csv 의 heard 칸이 비어 있다")

    rng = np.random.default_rng(0)
    print(f"채점 {len(got)}/{len(key)}개")
    print(f"{'클래스':<12}{'n':>5}{'라벨 오류율':>12}{'95% CI (원본 부트스트랩)':>28}")
    print("-" * 58)
    for cls in CLASSES:
        ids = [i for i in got if key[i]["true_class"] == cls]
        if not ids:
            continue
        fsids = np.array([key[i]["fsid"] for i in ids])
        wrong = np.array([got[i][0] != cls for i in ids], dtype=float)
        uniq = np.unique(fsids)
        idx_by = {u: np.flatnonzero(fsids == u) for u in uniq}
        bs = []
        for _ in range(2000):
            pick = rng.choice(len(uniq), len(uniq), replace=True)
            sel = np.concatenate([idx_by[uniq[k]] for k in pick])
            bs.append(wrong[sel].mean())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{cls:<12}{len(ids):>5}{100*wrong.mean():>11.1f}%"
              f"{f'[{100*lo:.1f}, {100*hi:.1f}]':>28}")
    print()
    print("  ⚠️ CI 는 **원본 단위** 부트스트랩이다. 블라인드 세트는 원본")
    print("     중복 없이 뽑았으므로 창 수 = 원본 수다.")
    print("  ⚠️ 판정자 1인이다 — 일치도를 재지 않았다 (CLAUDE.md 8장 한계 2).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("blind", "suspect"), default=None)
    ap.add_argument("--root", default="data/processed/safesound")
    ap.add_argument("--npz", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--score", default=None)
    a = ap.parse_args()
    if a.score:
        mode_score(a)
    elif a.mode == "blind":
        a.out = a.out or "data/audit/blind"
        mode_blind(a)
    elif a.mode == "suspect":
        a.out = a.out or "data/audit/suspect"
        mode_suspect(a)
    else:
        ap.error("--mode blind|suspect 또는 --score 가 필요하다")


if __name__ == "__main__":
    main()
