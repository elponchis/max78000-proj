#!/usr/bin/env python3
"""선형 멜(①-lin)의 int8 양자화 구간을 **측정으로** 정한다.

`tools/mel_range_sweep.py` 가 로그 멜에서 한 일(TOP_DB/SPAN_DB 결정)을 로그를
뺀 판에서 한다. 로그를 빼면 양자화 법칙 자체가 바뀌기 때문에 상수를 그대로
쓸 수 없다 — 그래서 별도 도구다.

── 왜 이게 단순한 문제가 아닌가 ─────────────────────────────────────────
로그는 두 가지 일을 **동시에** 한다.

  (a) **표현**: 게인 g 를 곱하면 로그 멜은 `20·log10 g` 만큼 **평행이동**한다.
      선형에서는 전부 **곱해진다**. 우리는 음량 정규화를 하지 않고 ±12dB
      랜덤 게인을 쓰므로(CLAUDE.md 5장 규칙 6) 이 차이가 크다.
  (b) **압축(companding)**: 70dB 를 255단계에 담는다. 선형으로 같은 일을
      하려면 255단계로 담을 수 있는 범위가 정해져 있다 —
         파워 선형   1 LSB = 1/255      → 10·log10(1/255) = **-24.1dB**
         진폭 선형   1 LSB = 1/255      → 20·log10(1/255) = **-48.1dB**
         세제곱근    1 LSB = 1/255      → 30·log10(1/255) = **-72.2dB**
      즉 파워 선형은 **24dB 밖에 담지 못한다.** 70dB 신호를 넣으면 대부분이
      0 이 된다. "①-lin 이 나빴다" 가 나와도 그게 (a) 때문인지 (b) 때문인지
      구분되지 않으면 결론을 쓸 수 없다.

→ 그래서 이 도구는 **바닥에 깔리는 빈의 비율**을 후보 스케일마다 세고,
   ±12dB 게인을 걸었을 때 int8 표현이 얼마나 움직이는지도 함께 잰다.
   그 수치가 있어야 "선형이 나쁜 이유" 를 (a)/(b) 로 나눠 적을 수 있다.

사용법 (WSL2):
    python3 tools/lin_mel_range.py --root data/processed/safesound \\
        --n 120 --md docs/results/lin-mel-range.md
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
EVENTS = CLASSES[:-1]
WIN = 16384
MARGIN = 1600          # 샤드는 1.2초로 저장돼 있다 (시간축 shift 여유)

# 후보 양자화 법칙. (이름, 압축지수 alpha, 설명)
#   q = clip(round(((p**alpha) / top) * 255)) - 128,  p 는 정규화 멜 파워
#   alpha=1.0 파워 선형 / 0.5 진폭 선형 / 1/3 세제곱근
SCHEMES = [
    ("파워 선형", 1.0),
    ("진폭 선형", 0.5),
    ("세제곱근", 1.0 / 3.0),
]
GAINS_DB = (-12.0, 0.0, 12.0)


def sample_windows(root, split, cls, n, rng):
    """클래스에서 창 n 개를 샤드에서 직접 읽는다 (증강 없음, 가운데 1초)."""
    d = os.path.join(root, split, cls)
    rows = []
    with open(os.path.join(d, "index.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["shard"]), int(r["row"])))
    take = (rows if len(rows) <= n
            else [rows[i] for i in rng.choice(len(rows), n, replace=False)])
    maps, out = {}, []
    for sh, row in take:
        m = maps.get(sh)
        if m is None:
            m = np.load(os.path.join(d, f"shard_{sh:04d}.npy"), mmap_mode="r")
            maps[sh] = m
        out.append(np.asarray(m[row][MARGIN:MARGIN + WIN]))
    return out


def mel_power(w_int8, gain_db=0.0):
    """int8 파형 → 정규화 멜 파워 (mels, frames). 풀스케일 사인 = 1.0.

    게인은 **파형 단계**에서 건다. 학습 증강이 그 위치에서 걸리기 때문이다
    (CLAUDE.md 5장 규칙 6). 포화는 파형 int8 재양자화까지 포함해야 정확하지만,
    여기서 보려는 것은 멜 쪽 양자화라 파형은 float 로 둔다.
    """
    x = np.asarray(w_int8, dtype=np.float64) / 128.0
    if gain_db:
        x = np.clip(x * (10.0 ** (gain_db / 20.0)), -1.0, 1.0)
    return (M.stft_power(x) @ M._FB.T).T          # (mels, frames)


def quant(p, alpha, top):
    """압축 후 고정 아핀 양자화. top 은 **압축 전 파워** 기준 포화점이다."""
    c = np.power(np.maximum(p, 0.0), alpha)
    q = np.round(c / (top ** alpha) * 255.0) - 128.0
    return np.clip(q, -128.0, 127.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/safesound")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=120, help="클래스당 창 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--md", default=None, help="마크다운 표를 여기에 쓴다")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    pw = {}                       # 클래스 -> (mels, frames, n) 파워
    for cls in CLASSES:
        ws = sample_windows(a.root, a.split, cls, a.n, rng)
        pw[cls] = np.stack([mel_power(w) for w in ws])      # (n, mels, frames)
        print(f"  {cls:<11}{len(ws):>5}창  빈 {pw[cls].size:,}")

    ev = np.concatenate([pw[c].ravel() for c in EVENTS])
    bg = pw["background"].ravel()
    out = []

    def say(s=""):
        print(s)
        out.append(s)

    say()
    say("## 1. 멜 파워 분포 (dB, 풀스케일 사인 = 0dB)")
    say()
    say(f"| 클래스 | p50 | p90 | p99 | p99.9 | 최대 |")
    say("|---|---:|---:|---:|---:|---:|")
    for cls in CLASSES:
        v = 10 * np.log10(pw[cls].ravel() + M.EPS)
        q = np.percentile(v, [50, 90, 99, 99.9])
        say(f"| `{cls}` | {q[0]:.1f} | {q[1]:.1f} | {q[2]:.1f} | {q[3]:.1f} "
            f"| {v.max():.1f} |")
    evdb = 10 * np.log10(ev + M.EPS)
    p999 = float(np.percentile(evdb, 99.9))
    say()
    say(f"이벤트 4클래스 합산 p99.9 = **{p999:.1f}dB**, 최대 {evdb.max():.1f}dB.")

    say()
    say("## 2. 후보 양자화 법칙 — 바닥에 깔리는 빈의 비율")
    say()
    say("`top` 은 포화점(압축 전 파워, dB). 로그 판은 비교를 위해 함께 싣는다.")
    say()
    say("| 법칙 | top(dB) | 담기는 범위 | 이벤트 바닥% | 이벤트 포화% "
        "| 배경 바닥% |")
    say("|---|---:|---:|---:|---:|---:|")

    # 로그 판 (기준)
    lg_ev = M.db_to_int8(10 * np.log10(ev + M.EPS)).astype(np.float64)
    lg_bg = M.db_to_int8(10 * np.log10(bg + M.EPS)).astype(np.float64)
    say(f"| **로그 (현행 ①)** | {M.TOP_DB:.0f} | {M.SPAN_DB:.0f}dB "
        f"| {100*np.mean(lg_ev <= -128):.1f}% | {100*np.mean(lg_ev >= 127):.3f}% "
        f"| {100*np.mean(lg_bg <= -128):.1f}% |")

    rows = []
    for name, alpha in SCHEMES:
        for top_db in (0.0, p999):
            top = 10.0 ** (top_db / 10.0)
            span = -10.0 / alpha * np.log10(255.0)     # 1 LSB 까지의 dB
            qe = quant(ev, alpha, top)
            qb = quant(bg, alpha, top)
            tag = "FS" if top_db == 0.0 else "p99.9"
            rows.append((name, alpha, top_db, tag, span,
                         100 * np.mean(qe <= -128), 100 * np.mean(qe >= 127),
                         100 * np.mean(qb <= -128)))
            say(f"| {name} (top={tag}) | {top_db:.1f} | {-span:.1f}dB "
                f"| {rows[-1][5]:.1f}% | {rows[-1][6]:.3f}% | {rows[-1][7]:.1f}% |")

    say()
    say("**바닥%** = int8 로 -128 이 되어 정보가 사라진 빈의 비율이다. 로그 판의")
    say("바닥%와 비교해 늘어난 만큼이 **압축 법칙 때문에 잃는 몫**이다.")

    say()
    say("## 3. +-12dB 게인에서 int8 표현이 얼마나 움직이는가")
    say()
    say("학습 증강이 +-12dB 를 건다(CLAUDE.md 5장 규칙 6).")
    say()
    say("⚠️ **클리핑된 빈을 빼고 재야 한다.** 이미 바닥(-128)에 깔린 빈은 게인을")
    say("주어도 바닥에 그대로 있어 dq=0 이 된다 — 파워 선형은 이벤트 빈의")
    say("94%가 바닥이라, 그대로 재면 '게인에 가장 강건' 하다는 정반대 결론이")
    say("나온다. 아래 dq 는 **세 게인(-12/0/+12dB) 모두에서 포화·바닥이 아닌**")
    say("빈만 대상으로 한다. 그 빈이 몇 %인지도 함께 싣는다.")
    say()
    say("| 법칙 | 유효 빈% | dq 중앙값 (+12dB) | IQR | dq 중앙값 (-12dB) "
        "| IQR | -12dB 에서 새로 바닥% |")
    say("|---|---:|---:|---:|---:|---:|---:|")

    # 게인별 파워. **같은 창**에 게인만 다르게 걸어야 빈 대응이 맞는다.
    ws_ev = []
    for c in EVENTS:
        ws_ev += sample_windows(a.root, a.split, c, a.n,
                                np.random.default_rng(a.seed))
    gp = {g: np.stack([mel_power(w, g) for w in ws_ev]).ravel()
          for g in GAINS_DB}

    def iqr(v):
        return float(np.percentile(v, 75) - np.percentile(v, 25))

    def report(name, fq):
        q0, qp, qm = fq(gp[0.0]), fq(gp[12.0]), fq(gp[-12.0])
        # 세 조건 모두에서 포화·바닥이 아닌 빈만
        ok = ((q0 > -128) & (q0 < 127) & (qp > -128) & (qp < 127)
              & (qm > -128) & (qm < 127))
        frac = 100.0 * ok.mean()
        if ok.sum() < 100:
            say(f"| {name} | {frac:.1f}% | - | - | - | - | - |")
            return
        dp, dm = (qp - q0)[ok], (qm - q0)[ok]
        newfloor = 100 * np.mean((qm <= -128) & (q0 > -128))
        say(f"| {name} | {frac:.1f}% | {np.median(dp):+.1f} | {iqr(dp):.1f} "
            f"| {np.median(dm):+.1f} | {iqr(dm):.1f} | {newfloor:.1f}% |")

    report("**로그 (현행 1, top 0 / span 70)**",
           lambda p: M.db_to_int8(10 * np.log10(p + M.EPS)).astype(np.float64))
    # 세제곱근과 담는 범위를 맞춘 로그 — 범위를 같게 두면 무엇이 남는지 본다
    report("로그 (top=p99.9 / span 72.2)",
           lambda p: M.db_to_int8(10 * np.log10(p + M.EPS), p999, 72.2
                                  ).astype(np.float64))
    for name, alpha in SCHEMES:
        top = 10.0 ** (p999 / 10.0)
        report(f"{name} (top=p99.9)", lambda p, al=alpha, t=top: quant(p, al, t))

    say()
    say("**읽는 법.** 로그는 게인이 **평행이동**이므로 dq 중앙값이 이론값")
    say("`12 / (span/255)` 에 맞고 **IQR 이 0 에 가깝다** — 모든 빈이 같은 양만큼")
    say("움직인다. 선형·세제곱근은 빈마다 다르게 움직여 IQR 이 커진다. 그 차이가")
    say("음량 불변성의 정량이다 (우리는 음량 정규화를 하지 않으므로 중요하다).")
    say()
    say("`유효 빈%` 가 낮으면 그 법칙은 애초에 대부분의 정보를 못 담는다는 뜻이다.")

    if a.md:
        with open(a.md, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"# 선형 멜 양자화 구간 실측 (①-lin 설계)\n\n"
                    f"`tools/lin_mel_range.py --root {a.root} --split {a.split} "
                    f"--n {a.n} --seed {a.seed}`\n\n"
                    f"클래스당 {a.n}창, 증강 없음.\n")
            f.write("\n".join(out) + "\n")
        print(f"\n저장: {a.md}")


if __name__ == "__main__":
    main()
