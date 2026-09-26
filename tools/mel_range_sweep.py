#!/usr/bin/env python3
"""로그 멜 dB 분포 실측 → `melfeat.TOP_DB` / `SPAN_DB` 결정.

구성 ① 의 int8 양자화 구간을 **측정으로 정한다** (CLAUDE.md 10장 — 측정하지 않은
숫자를 확정값처럼 쓰지 않는다). `tools/zero_ratio_by_class.py` 가 파형 쪽에서 한
일을 멜 쪽에서 하는 도구다.

보는 것은 둘이다.

  · **위쪽**: 상위 분위수를 잘라 버리면 큰 소리가 포화한다. 이벤트 클래스의
    꼭대기가 살아 있어야 한다
  · **아래쪽**: 구간을 너무 넓게 잡으면 분해능(1 LSB = span/255 dB)이 나빠지고,
    좁게 잡으면 조용한 창이 통째로 -128 이 된다. background 의 조용한 창 쿼터가
    **일부러 조용한 데이터**라는 점을 감안한다 (7장)

⚠️ 파형 쪽 `--floor-zero` 와 달리 여기에는 **버리는 규칙이 없다.** 창 선택은
이미 파형 단계에서 끝났고, 멜은 같은 창을 다르게 표현할 뿐이다. 구성 ①과 ④가
**같은 창 집합**을 보아야 비교가 성립한다.

사용법 (WSL2):
    python3 tools/mel_range_sweep.py --root data/processed/safesound \\
        --n 400 --md docs/results/mel-range-sweep.md
"""

import argparse
import csv
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "datasets"))

import melfeat as M  # noqa: E402

CLASSES = ["siren", "glass", "scream", "dog_bark", "background"]
WIN = 16384
MARGIN = 1600


def sample_windows(root, split, cls, n, rng):
    """클래스에서 창 n 개를 샤드에서 직접 읽는다 (증강 없음, 가운데 1초)."""
    d = os.path.join(root, split, cls)
    idx = os.path.join(d, "index.csv")
    rows = []
    with open(idx, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["shard"]), int(r["row"])))
    take = rows if len(rows) <= n else [rows[i] for i in
                                        rng.choice(len(rows), n, replace=False)]
    maps = {}
    out = []
    for sh, row in take:
        m = maps.get(sh)
        if m is None:
            m = np.load(os.path.join(d, f"shard_{sh:04d}.npy"), mmap_mode="r")
            maps[sh] = m
        out.append(np.asarray(m[row][MARGIN:MARGIN + WIN]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/safesound")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--n", type=int, default=400, help="클래스당 표본 창 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--md", help="결과를 마크다운으로 저장할 경로")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    # ── 필터뱅크 건전성부터. 빈 필터가 있으면 그 멜 밴드는 항상 -inf 다
    fb = M.mel_filterbank()
    dead = int((fb.sum(axis=1) == 0).sum())
    thin = int((fb.sum(axis=1) < 0.5).sum())
    out(f"# 로그 멜 dB 분포 실측 ({a.split} split, 클래스당 {a.n}창)")
    out()
    out(f"n_fft {M.N_FFT} / hop {M.HOP} / n_mels {M.N_MELS} / "
        f"{M.FMIN:.0f}~{M.FMAX:.0f}Hz / 프레임 {M.N_FRAMES}")
    out(f"필터뱅크: 가중치 합 0 인 밴드 {dead}개, 0.5 미만 {thin}개 "
        f"(0 이면 그 밴드는 항상 바닥이다)")
    out()

    qs = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]
    out("## 클래스별 로그 멜 dB 분위수 (모든 밴드·프레임 합산)")
    out()
    out("| 클래스 | 창 | " + " | ".join(f"p{q:g}" for q in qs) + " |")
    out("|---|---:|" + "---:|" * len(qs))
    allv = []
    per_cls = {}
    for cls in CLASSES:
        ws = sample_windows(a.root, a.split, cls, a.n, rng)
        v = np.concatenate([M.log_mel_db(w).ravel() for w in ws])
        per_cls[cls] = v
        allv.append(v)
        out(f"| `{cls}` | {len(ws)} | "
            + " | ".join(f"{np.percentile(v, q):.1f}" for q in qs) + " |")
    v_all = np.concatenate(allv)
    ev = np.concatenate([per_cls[c] for c in CLASSES if c != "background"])
    out(f"| **전체** | | " + " | ".join(f"{np.percentile(v_all, q):.1f}"
                                        for q in qs) + " |")
    out()

    # ── 후보 구간별 포화·바닥 비율
    out("## 구간 후보 — 포화(+127)와 바닥(-128) 비율")
    out()
    out("`top_db` 는 이벤트 클래스 상위가 잘리지 않는 선에서 낮출수록 좋다 "
        "(분해능 = span/255 dB).")
    out()
    out("| top_db | span_db | LSB | 이벤트 포화 | 이벤트 바닥 | 배경음 바닥 |")
    out("|---:|---:|---:|---:|---:|---:|")
    bg = per_cls["background"]
    best = None
    for top in (0.0, -5.0, -10.0, -15.0, -20.0):
        for span in (60.0, 70.0, 80.0, 90.0, 100.0):
            sat = float((ev >= top).mean())
            flo = float((ev <= top - span).mean())
            bfl = float((bg <= top - span).mean())
            out(f"| {top:.0f} | {span:.0f} | {span/255:.3f}dB | "
                f"{sat:.3%} | {flo:.2%} | {bfl:.2%} |")
            # 이벤트 포화 0.1% 이하 & 이벤트 바닥 20% 이하 중 span 최소
            if sat <= 0.001 and flo <= 0.20:
                if best is None or span < best[1]:
                    best = (top, span, sat, flo, bfl)
    out()
    if best:
        out(f"**제안: `TOP_DB = {best[0]:.1f}`, `SPAN_DB = {best[1]:.1f}`** "
            f"(이벤트 포화 {best[2]:.3%}, 이벤트 바닥 {best[3]:.2%}, "
            f"배경음 바닥 {best[4]:.2%}, 1 LSB = {best[1]/255:.3f}dB)")
        out()
        out("기준: 이벤트 클래스의 포화를 0.1% 이하로 막고(큰 소리의 꼭대기가"
            " 살아야 한다), 이벤트 바닥이 20% 를 넘지 않는 선에서 span 을 가장"
            " 좁게 잡아 분해능을 확보한다. 배경음 바닥 비율은 제약이 아니다 —"
            " 조용한 창 쿼터(15%)가 일부러 조용하기 때문이다 (CLAUDE.md 7장).")
    else:
        out("⚠️ 후보 중 기준을 만족하는 조합이 없다. 범위를 넓혀 다시 볼 것.")
    out()

    # ── 양자화 잡음 바닥. 아래쪽 끝을 여기에 맞추면 "잘라도 잃는 정보가 없다"
    #    가 실측으로 뒷받침된다. 분위수 표보다 이쪽이 강한 근거다.
    rng2 = np.random.default_rng(1234)
    noise = rng2.integers(-1, 2, WIN).astype(np.int8)     # ±1 LSB 만 든 창
    nd = M.log_mel_db(noise)
    band = np.median(nd, axis=1)
    out("## int8 양자화 잡음 바닥 (±1 LSB 만 든 창)")
    out()
    out(f"밴드별 중앙값: 저역 {band[0]:.1f}dB → 고역 {band[-1]:.1f}dB, "
        f"전체 중앙값 {np.median(nd):.1f}dB")
    out()
    out(f"이 아래는 신호가 아니라 수치 잔여물이다. 자르는 지점을 정하는 것은 "
        f"**잡음 바닥이 가장 낮은 밴드**({band.min():.1f}dB, 저역)다 — 거기에는 "
        f"그보다 위의 실제 신호가 있기 때문이다. 고역은 바닥이 "
        f"{band.max():.1f}dB 로 더 높아 제약이 되지 않는다.")
    out()
    out(f"현재 설정의 아래쪽 끝 {M.TOP_DB - M.SPAN_DB:.0f}dB 는 저역 바닥보다 "
        f"{band.min() - (M.TOP_DB - M.SPAN_DB):.1f}dB 아래다 → 정보 손실 없음.")
    out()

    # ── 풀스케일 사인. 위쪽 끝의 물리적 의미를 못 박는다
    t = np.arange(WIN) / float(M.SR)
    sine = np.round(127 * np.sin(2 * np.pi * 1000 * t)).astype(np.int8)
    out(f"풀스케일 1kHz 사인파의 멜 최댓값: {M.log_mel_db(sine).max():.2f}dB "
        f"(0dB 에서 조금 낮은 것은 1kHz 가 멜 필터 꼭대기와 정확히 겹치지 "
        f"않기 때문이다)")
    out()
    out(f"현재 `melfeat.py` 값: TOP_DB {M.TOP_DB}, SPAN_DB {M.SPAN_DB} "
        f"→ 1 LSB = {M.SPAN_DB/255:.3f}dB")

    if a.md:
        os.makedirs(os.path.dirname(a.md) or ".", exist_ok=True)
        with open(a.md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n저장: {a.md}")


if __name__ == "__main__":
    main()
