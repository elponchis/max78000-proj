#!/usr/bin/env python3
"""오경보 분석 — 청취 목록 + 출처별 분해 + 배포 시나리오 가중.

`tools/event_eval.py` 가 만든 **캐시된 로짓 위에서만** 돈다. 재학습도
재스캔도 없다 (프레임 피크는 오디오를 다시 읽어야 해서 따로 캐시한다).

세 가지를 낸다.

1. **청취 목록** (`--listen N`): 동작점에서 발생한 오경보 이벤트를 마진 순으로
   N개 뽑아 **wav 로 추출**하고, 원본 fsid·출처 태그·FSD50K 라벨을 CSV 로 낸다.
   분류 칸(`verdict`)은 비워 두고 사람이 채운다:
     a = 하드 네거티브 (알람·말소리·유리 부딪힘 등)
     b = 배경 클립 속 **실제 이벤트** (라벨 누락)
     c = 그 외
2. **출처별 분해**: 조용한 쿼터 / 하드 네거티브 / 일반 배경 각각의
   **재생시간과 회/h**. 어디서 오경보가 나는지 가른다.
3. **배포 시나리오 가중 회/h** (`--scenario`): 실배치의 시간 구성을 가정해
   가중 평균한다. ⚠️ **가정값이며 측정값이 아니다.** 표에 반드시 그렇게 적는다.

⚠️ "조용함" 은 `tools/quiet_bg_fa.py` 와 **같은 절대 기준**(int8 피크 ≤ 8,
테스트 배경 창 피크의 15퍼센타일)을 쓴다. 빈틈없는 프레임의 분포는 창
분포와 다르므로, 여기서 나오는 조용한 시간 비율은 15% 가 아니다.

사용 (WSL2):
    python3 tools/fa_analysis.py --tag mel-s1 --target 1 --listen 30 \\
        --outdir data/fa-listen/mel-1ph
"""

import argparse
import csv
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "scripts"), os.path.join(REPO, "tools")]

CLASSES = ["siren", "glass", "scream", "dog_bark", "background"]
BG = 4
QUIET_PEAK = 8            # quiet_bg_fa.py 와 같은 절대 기준
PEAKCACHE = os.path.join(REPO, "data", "eventcache", "peaks-{split}-hop{hop}.npz")


def frame_peaks(split, hop_ms, manifest, roots, limit=0):
    """배경음 클립의 **프레임별 int8 피크**. 모델을 돌리지 않는다."""
    p = PEAKCACHE.format(split=split, hop=hop_ms)
    if os.path.isfile(p):
        z = np.load(p, allow_pickle=True)
        off, out = 0, {}
        for c, n in zip(z["clip"], z["lens"]):
            out[str(c)] = z["peaks"][off:off + int(n)]
            off += int(n)
        return out
    import prepare_safesound as P
    hop = max(1, int(P.SR * hop_ms / 1000))
    rows = [r for r in csv.DictReader(open(manifest, encoding="utf-8"))
            if r["split"] == split and r["cls"] == "background"]
    if limit:
        rows = rows[:limit]
    out = {}
    for i, r in enumerate(rows, 1):
        path = P.clip_path(r, roots)
        if not path:
            continue
        x = P.load_audio(path)
        if x is None or len(x) < P.WIN:
            continue
        q = P.to_int8(x)
        starts = range(0, len(x) - P.WIN + 1, hop)
        out[r["clip_id"]] = np.array(
            [int(np.abs(q[s:s + P.WIN].astype(np.int16)).max()) for s in starts],
            dtype=np.int16)
        if i % 300 == 0:
            print(f"  피크 {i}/{len(rows)}", flush=True)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(
        p, clip=np.array(list(out)),
        lens=np.array([len(v) for v in out.values()]),
        peaks=np.concatenate(list(out.values())))
    print(f"저장 {p}")
    return out


def fsd_labels(meta_dir):
    """FSD50K clip_id -> 라벨 문자열."""
    lab = {}
    gt = os.path.join(meta_dir, "FSD50K.ground_truth")
    for fn in ("dev.csv", "eval.csv"):
        fp = os.path.join(gt, fn)
        if not os.path.isfile(fp):
            continue
        with open(fp, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                lab[r["fname"]] = r.get("labels", "")
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--hop-ms", type=int, default=250)
    ap.add_argument("--mk", default="3/4")
    ap.add_argument("--cooldown", type=float, default=2.0)
    ap.add_argument("--target", type=float, default=1.0, help="목표 오경보 회/h")
    ap.add_argument("--listen", type=int, default=0, help="wav 로 뽑을 상위 N개")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--manifest", default="data/interim/manifest.csv")
    ap.add_argument("--fsd-dir", default="data/raw/FSD50K_clips")
    ap.add_argument("--us8k-dir", default="data/raw/US8K_audio")
    ap.add_argument("--esc50-dir", default="data/raw/ESC50_audio")
    ap.add_argument("--fsd-meta", default="data/raw/FSD50K_meta")
    ap.add_argument("--scenario", default="quiet=0.80,general=0.18,hardneg=0.02",
                    help="배포 시나리오 시간 비중 (가정값)")
    a = ap.parse_args()

    import event_eval as EE
    roots = {"fsd": a.fsd_dir, "us8k": a.us8k_dir, "esc50": a.esc50_dir}
    hop_s = a.hop_ms / 1000.0
    m, k = (int(x) for x in a.mk.split("/"))

    per = EE.load_cache(a.tag, a.split, a.hop_ms)
    thr = EE.thr_for_fa(per, a.target, m, k, a.cooldown, hop_s)
    print(f"[{a.tag}] 목표 {a.target:g}회/h → 문턱값 {thr:+.3f} "
          f"(m/k {m}/{k}, cd {a.cooldown:g}s)")

    note = {r["clip_id"]: (r["note"] or "") for r in
            csv.DictReader(open(a.manifest, encoding="utf-8"))}
    peaks = frame_peaks(a.split, a.hop_ms, a.manifest, roots)

    # ── 버킷: 하드 네거티브 > 조용 > 일반 (프레임 단위) ────────────────
    def bucket(clip, fi):
        if note.get(clip, "").startswith("hard_neg"):
            return "hardneg"
        pk = peaks.get(clip)
        if pk is not None and fi < len(pk) and pk[fi] <= QUIET_PEAK:
            return "quiet"
        return "general"

    sec = {"quiet": 0.0, "general": 0.0, "hardneg": 0.0}
    cnt = {"quiet": 0, "general": 0, "hardneg": 0}
    fires = []
    for clip, fsid, cls, _dur, lg in per:
        if cls != "background":
            continue
        n = len(lg)
        for i in range(n):
            sec[bucket(clip, i)] += hop_s
        det = EE.detect(lg, thr)
        ev = lg[:, :BG]
        marg = ev.max(1) - lg[:, BG]
        for i, c in EE.fire_events(det, m, k, a.cooldown, hop_s):
            b = bucket(clip, i)
            cnt[b] += 1
            fires.append((float(marg[i]), clip, fsid, i, int(c), b))

    print()
    print(f"{'버킷':<12}{'재생시간':>10}{'비중':>8}{'오경보':>8}{'회/h':>10}")
    print("-" * 50)
    tot_sec = sum(sec.values())
    rate = {}
    for b in ("quiet", "general", "hardneg"):
        h = sec[b] / 3600.0
        rate[b] = cnt[b] / h if h else float("nan")
        print(f"{b:<12}{h:>9.2f}h{100*sec[b]/tot_sec:>7.1f}%{cnt[b]:>8}"
              f"{rate[b]:>10.3f}")
    print("-" * 50)
    print(f"{'전체':<12}{tot_sec/3600:>9.2f}h{100:>7.1f}%{sum(cnt.values()):>8}"
          f"{sum(cnt.values())/(tot_sec/3600):>10.3f}")
    print()
    print("  ⚠️ 0회 버킷의 95% 상한 = 3/재생시간. 재생시간이 짧은 버킷의")
    print("     0.000 은 '없다' 가 아니라 '못 쟀다' 다.")
    for b in ("quiet", "general", "hardneg"):
        if cnt[b] == 0 and sec[b] > 0:
            print(f"       {b}: 0회 관측, 95% 상한 {3/(sec[b]/3600):.2f}회/h")

    # ── 배포 시나리오 가중 (가정값) ───────────────────────────────────
    w = {}
    for part in a.scenario.split(","):
        kk, _, vv = part.partition("=")
        w[kk.strip()] = float(vv)
    tot_w = sum(w.values())
    wr = sum(w.get(b, 0.0) * rate[b] for b in rate if rate[b] == rate[b]) / tot_w
    print()
    print("=== 배포 시나리오 가중 (⚠️ **가정값**, 측정값이 아니다) ===")
    print("  시간 비중  " + "  ".join(f"{b} {100*w.get(b,0)/tot_w:.0f}%"
                                   for b in ("quiet", "general", "hardneg")))
    print(f"  가중 오경보 **{wr:.3f}회/h = {24*wr:.1f}회/일**")
    print("  이 값은 위 버킷별 실측률에 **가정한 시간 비중**을 곱한 것이다.")
    print("  실배치 구성이 다르면 값이 달라진다 — 측정값과 섞어 쓰지 말 것.")

    # ── 청취 목록 ────────────────────────────────────────────────────
    if a.listen:
        import prepare_safesound as P
        import soundfile as sf
        out = a.outdir or f"data/fa-listen/{a.tag}-{a.target:g}ph"
        os.makedirs(out, exist_ok=True)
        lab = fsd_labels(a.fsd_meta)
        rowmap = {r["clip_id"]: r for r in
                  csv.DictReader(open(a.manifest, encoding="utf-8"))}
        fires.sort(reverse=True)
        pick = fires[:a.listen]
        with open(os.path.join(out, "listen.csv"), "w", newline="",
                  encoding="utf-8-sig") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["rank", "wav", "margin", "pred_class", "bucket",
                           "clip_id", "fsid", "t_sec", "note", "fsd_labels",
                           "verdict(a/b/c)", "memo"])
            cache = {}
            for rnk, (mg, clip, fsid, fi, c, b) in enumerate(pick, 1):
                r = rowmap.get(clip)
                x = cache.get(clip)
                if x is None and r is not None:
                    pth = P.clip_path(r, roots)
                    x = P.load_audio(pth) if pth else None
                    cache[clip] = x
                name = f"{rnk:02d}_{CLASSES[c]}_{b}_{clip}_{fi}.wav"
                if x is not None:
                    s = fi * int(P.SR * a.hop_ms / 1000)
                    seg = x[s:s + P.WIN]
                    if len(seg) == P.WIN:
                        sf.write(os.path.join(out, name), seg, P.SR)
                wcsv.writerow([rnk, name, f"{mg:.2f}", CLASSES[c], b, clip,
                               fsid, f"{fi*hop_s:.2f}",
                               (r or {}).get("note", ""),
                               lab.get(clip, ""), "", ""])
        print()
        print(f"청취 목록 {len(pick)}개 → {out}/  (listen.csv 의 "
              f"verdict 칸을 채울 것: a=하드네거티브 / b=라벨 누락 실제 이벤트 "
              f"/ c=그 외)")


if __name__ == "__main__":
    main()
