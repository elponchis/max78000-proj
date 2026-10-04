#!/usr/bin/env python3
"""정확도 대 추정 에너지 그림 — `docs/results/curve-points.csv` 를 그린다.

  가로축  판단 1회 추정 에너지 (µJ, 로그 축). 마커는 데이터시트 전력 기준 추정,
          가는 선은 선행연구 전력 기준 추정까지의 범위다
  세로축  macro-F1 @ 300 오경보/h (시드 평균, 막대는 ±sd)
  채운 마커 = 지연을 보드에서 잰 구성, 빈 마커 = 지연이 추정인 구성

⚠️ 에너지는 전부 **추정(실측 지연 × 공개 전력)** 이다. 그림 아래에 그대로 적는다.
⚠️ 그림 안 글자는 영어다 — matplotlib 기본 폰트에 한글 글리프가 없다.

색은 3색으로 제한했다 (산점도는 모든 쌍이 인접하므로 3색까지만 색각 이상
검증을 통과한다). 색 = 추론이 도는 곳·입력 표현, 모양 = 뒷단 구조.

사용 (WSL2):  ~/ai8x-training/venv/bin/python tools/plot_curve.py
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "docs", "results", "curve-points.csv")
OUT = os.path.join(REPO, "docs", "figures", "curve-draft.png")

SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                         "#e1e0d9", "#c3c2b7")
GROUPS = {   # 색, 모양, 범례 이름
    "mel_npu": ("#2a78d6", "o", "log-mel on CPU + 2D CNN on NPU"),
    "mel_npu_1d": ("#2a78d6", "s", "log-mel on CPU + 1D CNN on NPU"),
    "raw_npu": ("#eb6834", "o", "raw waveform, 1D CNN on NPU"),
    "m4": ("#1baf7a", "o", "same 1D CNN in software on Cortex-M4"),
}
# 로그 멜 2D 계열 — 점에는 글자(a..h)만 찍고 설명은 표로 뺀다 (에너지 큰 순)
KEYED = ["melinc", "h400", "m32", "h500", "h500m32", "u1000f1024", "u800", "u1000"]
# 라벨 위치 (점 기준 오프셋, 포인트) — 겹치지 않게 손으로 잡았다
OFFS = {"wave": (8, -16), "w050": (9, -14), "melinc": (9, 13), "h400": (-6, 17),
        "h500": (2, -19), "h500m32": (-4, -19), "m32": (11, -14), "u800": (-9, 12),
        "u1000": (-9, 4), "u1000f1024": (-6, -19), "a1d": (9, -15),
        "m4cmsis": (0, 14), "m4plain": (0, 14)}
HALIGN = {"h400": "right", "h500": "center", "h500m32": "center", "u800": "right",
          "u1000": "right", "u1000f1024": "center", "m4cmsis": "center",
          "m4plain": "center"}


def main():
    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    key_lines = []
    for r in rows:
        color, marker, _ = GROUPS[r["group"]]
        x, x2 = float(r["e_datasheet_uJ"]), float(r["e_literature_uJ"])
        y, sd = float(r["f1"]), float(r["f1_sd"])
        measured = r["board"] == "1"
        ax.plot([x, x2], [y, y], color=color, lw=1.0, alpha=0.55, zorder=2,
                solid_capstyle="round")
        ax.plot([x, x], [y - sd, y + sd], color=color, lw=1.0, alpha=0.55, zorder=2)
        ax.plot(x, y, marker=marker, ms=8.5, zorder=3, mec=color, mew=1.8,
                mfc=color if measured else SURFACE, ls="none")
        t = float(r["time_ms"])
        tms = f"{'' if measured else '~'}{t:g} ms"
        if r["id"] in KEYED:
            # 로그 멜 2D 계열은 점이 몰려 있다 — 점에는 글자만, 설명은 옆의 표로
            letter = "abcdefgh"[KEYED.index(r["id"])]
            ax.annotate(letter, (x, y), xytext=(-9, 9), textcoords="offset points",
                        fontsize=8.2, color=INK, ha="center", va="center",
                        fontweight="bold")
            key_lines.append(f"{letter}   {r['label']}  ·  {tms}")
            continue
        lab = f"{r['label']}\n{tms}"
        ax.annotate(lab, (x, y), xytext=OFFS.get(r["id"], (8, 6)),
                    textcoords="offset points", fontsize=7.6, color=INK2,
                    ha=HALIGN.get(r["id"], "left"), va="center", linespacing=1.15)

    ax.text(0.435, 0.735, "log-mel + 2D CNN configurations",
            transform=ax.transAxes, fontsize=7.4, color=INK, va="top", fontweight="bold")
    ax.text(0.435, 0.700, "\n".join(sorted(key_lines)), transform=ax.transAxes,
            fontsize=7.2, color=INK2, va="top", linespacing=1.45)

    ax.set_xscale("log")
    ax.set_xlim(14, 40000)
    ax.set_ylim(0.465, 0.70)
    ax.set_xlabel("Estimated energy per decision (µJ, log scale)", color=INK2, fontsize=9)
    ax.set_ylabel("macro-F1 at 300 false alarms/h", color=INK2, fontsize=9)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.grid(False, which="minor")
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)

    handles = [plt.Line2D([], [], marker=m, ls="none", ms=7, mec=c, mfc=c, mew=1.6, label=n)
               for c, m, n in GROUPS.values()]
    handles += [plt.Line2D([], [], marker="o", ls="none", ms=7, mec=MUTED, mfc=MUTED, mew=1.6,
                           label="latency measured on board"),
                plt.Line2D([], [], marker="o", ls="none", ms=7, mec=MUTED, mfc=SURFACE, mew=1.6,
                           label="latency estimated (not yet measured)")]
    leg = ax.legend(handles=handles, loc="upper right", fontsize=7.4, frameon=False,
                    labelcolor=INK2, handletextpad=0.4, borderaxespad=0.6)
    leg.set_zorder(1)

    ax.set_title("Accuracy vs. estimated energy per decision (MAX78000FTHR, 1 s window, 250 ms hop)",
                 fontsize=9.5, color=INK, loc="left", pad=10)
    fig.text(0.012, 0.012,
             "Energy is an ESTIMATE (measured latency × published power), not a measurement. "
             "Marker: datasheet power; line: up to literature-derived power.\n"
             "Vertical bar: ±1 sd over seeds. Number under each label: time per decision "
             "(busy-wait, sleep excluded). DRAFT.",
             fontsize=6.6, color=MUTED, ha="left", va="bottom", linespacing=1.3)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.92, bottom=0.165)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, facecolor=SURFACE)
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
