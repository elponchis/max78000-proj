#!/usr/bin/env python3
"""시험셋 비명 정답 확정용 청취 세트 — `docs/results/scream-plan.md` 9.3 (사용자 결정 2026-10-14).

범위 (전수):
    screaming_test        시험셋 비명 창 중 FSD50K `Screaming` 보유 — 159 (Yell·Shout 만 66 은 S_train_scream 에서 이미 청취)
    screaming_plus_train  v4+D 학습 비명 창 중 `Screaming`+`Yell`/`Shout` 동반 — 30 (학습 미청취분 중 권고된 것)
    contrast              대조 15: FSD50K 계열 학습 비명 창 중 PANNs `Screaming` ≥ 0.50, 위 대상·이전 세트(S_train_scream)와 겹치지 않는 것
파일명은 무작위 번호(`NNN.wav`)만. 정답·출처·PANNs 점수는 key.csv 에만 둔다 (청취자는 verdict_sheet.csv 를 채운다).
판정 기준은 이전과 같다 (README.md): O = 겁·고통·놀람의 날카로운 외침, X = 고함·말·환호·군중·아기 울음 등, ? = 판단 불가.
**세트는 저장소에 올리지 않는다** (`~/safesound-external/audit_test_scream/`).
판정이 오면 testL-S3 (시험셋 비명 정답 = 청취 O 만; X·? 평가 제외) 을 만들어 v5 주 정답으로 학습 시작 전에 고정한다.

사용 (WSL2, ai8x venv):  python tools/build_audit_test_scream.py [--dry-run] [--out ~/safesound-external/audit_test_scream]
"""
import argparse
import csv
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "tools"), os.path.join(REPO, "scripts")]
import label_audit as LA                      # noqa: E402
from build_audit_train_scream import fsd_labels, combo, V4   # noqa: E402

SEED = 78007
N_CONTRAST = 15
SET = "T_test_scream"
PREV = os.path.expanduser("~/safesound-external/audit_train_scream/key.csv")     # 이전 세트 — 대조 창이 겹치지 않게
EXPECT = {"screaming_test": 159, "screaming_plus_train": 30, "contrast": N_CONTRAST}
README = """# 시험셋 비명 정답 확정 청취 세트 (T_test_scream) — 2026-10-14

- 파일은 무작위 번호뿐이다. 출처·라벨·PANNs 점수는 key.csv 에만 있다 — **청취 중에는 key.csv 를 열지 않는다.**
- 판정은 verdict_sheet.csv 의 `verdict(O/X/?)` 열에 적는다. 메모는 선택.
- 판정 기준 (이전 청취와 같다):
  - **O** = 비명: 겁·고통·놀람의 날카로운 외침 (사람 목소리, 1초 창 안에 들린다)
  - **X** = 비명 아님: 고함·외침(말), 말소리, 환호, 군중, 웃음, 노래, 아기 울음, 동물 소리, 효과음 등
  - **?** = 판단 불가 (너무 짧거나 작거나 겹쳐서 가를 수 없음)
- 창은 1초 (16 kHz mono, int8 → 16bit 스케일). 필요하면 반복 재생한다. 볼륨은 처음 몇 창으로 고정하고 바꾸지 않는다.
- 대조 창 15개가 섞여 있다 (PANNs 가 비명으로 확신하는 학습 창). 청취자 일관성 점검용이고 정답이 아니다.
- 총 204창, 1창 8초면 약 27분. 한 번에 다 듣지 않아도 되지만 **기준은 중간에 바꾸지 않는다.**
- 이 판정은 시험셋 비명 **정답**이 된다 (O 양성 / X·? 평가 제외). 학습 시작 전에 고정하고 그 뒤 바꾸지 않는다.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/safesound-external/audit_test_scream"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    gt = fsd_labels()
    names = LA.label_names()
    tr = LA.load("v1_train"); s_tr = LA.class_scores(tr["P"], names)["scream"]
    zt = np.load(LA.TEST_CACHE, allow_pickle=True); s_te = LA.class_scores(zt["P"], names)["scream"]

    def norm(c):
        return str(c)[:-4] if str(c).endswith(".wav") else str(c)
    score_tr = {(norm(c), int(s)): float(v) for c, s, v in zip(tr["clips"], tr["starts"], s_tr)}
    score_te = {(norm(c), int(s)): float(v) for c, s, v in zip(zt["clips"], zt["starts"], s_te)}
    with open(PREV, encoding="utf-8") as f:
        prev = {(r["clip"], int(r["start_sample"])) for r in csv.DictReader(f)}

    train = LA.read_index(V4, "train", "scream")
    test = LA.read_index(V4, "test", "scream")
    groups = {"screaming_test": [], "screaming_plus_train": []}
    for r in test:
        L = gt.get(r["fsid"])
        if L is None:
            sys.exit(f"[에러] FSD50K 라벨 없음 (test): {r['fsid']}")
        c = combo(L)
        if "Screaming" in c:
            groups["screaming_test"].append((r, c, score_te.get((r["clip_id"], int(r["start_sample"])))))
    for r in train:
        if "audioset" in r["note"]:
            continue
        c = combo(gt[r["fsid"]])
        if "Screaming" in c and ("Yell" in c or "Shout" in c):
            groups["screaming_plus_train"].append((r, c, score_tr.get((r["clip_id"], int(r["start_sample"])))))
    targets = {(r["clip_id"], int(r["start_sample"])) for g in groups.values() for r, _c, _s in g}
    cand = [(r, combo(gt[r["fsid"]]), score_tr.get((r["clip_id"], int(r["start_sample"]))))
            for r in train if "audioset" not in r["note"]
            and (r["clip_id"], int(r["start_sample"])) not in targets and (r["clip_id"], int(r["start_sample"])) not in prev]
    cand = [t for t in cand if t[2] is not None and t[2] >= LA.HIGH]
    groups["contrast"] = [cand[i] for i in sorted(rng.permutation(len(cand))[:N_CONTRAST])]

    print("## 청취 세트 구성")
    tot = 0
    for g, v in groups.items():
        miss = sum(1 for _r, _c, s in v if s is None)
        print(f"  {g}: {len(v)}창" + (f" (PANNs 점수 없음 {miss})" if miss else ""))
        tot += len(v)
    print(f"  합계 {tot}창 (예상 {tot * 8 / 60:.0f}분 @ 8초/창); 대조 후보 {len(cand)} (이전 세트 {len(prev)}창 제외)")
    for g, n in EXPECT.items():
        assert len(groups[g]) == n, (g, len(groups[g]), n)
    assert len(test) == 225 and len(groups["screaming_test"]) + 66 == 225
    if a.dry_run:
        return
    import soundfile as sf
    d = os.path.join(a.out, SET)
    os.makedirs(d, exist_ok=True)
    items = [(g, r, c, s) for g, v in groups.items() for r, c, s in v]
    order = rng.permutation(len(items))
    maps, key = {}, []
    for n, j in enumerate(order, 1):
        g, r, c, s = items[j]
        split = "test" if g == "screaming_test" else "train"
        x = LA.shard_window(V4, split, "scream", r, maps)
        fn = f"{n:03d}.wav"
        sf.write(os.path.join(d, fn), (x.astype(np.int16) * 256), LA.SR)
        key.append(dict(set=SET, file=fn, verdict="", role="contrast" if g == "contrast" else "target",
                        source=("v4_test" if split == "test" else "v4_train"), cls="scream", definition=g,
                        clip=r["clip_id"], start_sample=int(r["start_sample"]), panns_score=("" if s is None else round(s, 4)), labels=c))
    with open(os.path.join(a.out, "key.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0].keys())); w.writeheader(); w.writerows(key)
    with open(os.path.join(a.out, "verdict_sheet.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["set", "file", "verdict(O/X/?)", "memo"])
        for r in key:
            w.writerow([r["set"], r["file"], "", ""])
    with open(os.path.join(a.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(README)
    print("→", a.out, f"({len(key)} wav, key.csv, verdict_sheet.csv, README.md)")


if __name__ == "__main__":
    main()
