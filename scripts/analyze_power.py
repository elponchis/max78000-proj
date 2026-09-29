#!/usr/bin/env python3
"""전력 트레이스 → 추론당 에너지. 모든 전력 실험이 이 도구를 지난다.

CLAUDE.md 7장의 측정 방법론을 코드로 옮긴 것이다. 보드 없이 작성·검증할 수
있고(`--self-test`), 장비가 들어오면 CSV 만 갈아 끼우면 된다.

── 두 가지 모드 ────────────────────────────────────────────────────────

**gated** — GPIO 토글로 추론 구간을 마킹한 경우 (MAX78000FTHR 1순위)
    구간마다 에너지를 적분하고, 구간 **밖**의 전력을 idle 로 보아 차감한다.
    추론 1000회의 분포(평균·표준편차·p99·최악)를 낸다.
    ⚠️ **최악값을 함께 보고한다** — G2 는 "DWT 1000회 최악값" 기준이다.

**stat** — 게이팅이 없거나 믿을 수 없는 경우 (OrangePi, OS 지터)
    추론 N회를 도는 구간의 평균 전력에서 idle 평균을 빼고 N 으로 나눈다.
    CLAUDE.md 7장 OrangePi 절의 "통계적 방식" 이다.

── 왜 idle 을 빼는가 ───────────────────────────────────────────────────
코어 레일을 분리할 수 없어 **M4 경로와 NPU 경로의 차분**으로 논증한다
(CLAUDE.md 7장). 베이스라인이 양쪽에 동일하게 깔리므로 차분이 오히려 깨끗하다.
그러려면 각 경로에서 idle 을 같은 방식으로 빼야 한다.

⚠️ **"시간 ↔ 전력 트레이드오프" 로 서술하지 말 것** (CLAUDE.md 2장). 이 도구는
에너지(= 전력 × 시간)를 낸다. 추론이 빨라지면 시간이 줄어 에너지도 함께 준다.

── 입력 CSV ────────────────────────────────────────────────────────────
헤더가 있는 CSV. 열 이름은 `--t-col` 등으로 지정한다. 기본 기대 형태:

    t,ch1,ch2
    0.000000,0.01234,0.02
    0.000010,0.01240,3.28

  · 시간: 초 단위 (`--t-unit ms|us` 로 바꿀 수 있다)
  · 전류원: `--i-col` 에 전류(A) 열, 또는 `--v-shunt-col` + `--shunt-ohm`
    (오실로스코프 + 1Ω 션트, CLAUDE.md 7장 1순위 구성)
  · 게이트: `--gate-col` 에 GPIO 전압 열. `--gate-thr` 위를 high 로 본다
  · 공급 전압: 고정값 `--vsupply`(기본 3.7, Li-Po) 또는 `--v-col` 로 실측 열

PPK2 는 전류를 µA 로 내보내므로 `--i-unit ua` 를 쓴다.

사용 (WSL2 / Windows 어디서든, numpy 만 필요):

    python scripts/analyze_power.py --self-test
    python scripts/analyze_power.py run.csv --i-col current --gate-col gpio
    python scripts/analyze_power.py run.csv --mode stat --n-inf 1000 \\
        --idle idle.csv
"""

import argparse
import csv
import sys

import numpy as np

# numpy 2.0 에서 trapz -> trapezoid 로 이름이 바뀌었다. 로컬(1.x)과 Colab(2.x)
# 양쪽에서 돌아야 하므로 있는 쪽을 쓴다.
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


# ────────────────────────────────────────────────────────────── 입력
def read_csv(path, cols):
    """필요한 열만 float 배열로 읽는다. 큰 트레이스를 통째로 올리지 않도록
    열을 골라 받는다 (오실로스코프 CSV 는 쉽게 수백 MB 다)."""
    out = {c: [] for c in cols if c}
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        missing = [c for c in out if c not in (r.fieldnames or [])]
        if missing:
            raise SystemExit(f"CSV 에 열이 없다: {missing}\n"
                             f"  있는 열: {r.fieldnames}")
        for row in r:
            for c in out:
                out[c].append(float(row[c]))
    return {c: np.asarray(v, dtype=np.float64) for c, v in out.items()}


T_SCALE = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
I_SCALE = {"a": 1.0, "ma": 1e-3, "ua": 1e-6, "na": 1e-9}


def to_power(d, a):
    """(t[s], p[W]) 로 정규화한다."""
    t = d[a.t_col] * T_SCALE[a.t_unit]
    if a.i_col:
        i = d[a.i_col] * I_SCALE[a.i_unit]
    else:
        # 션트 양단 전압 / 저항. 1Ω 이면 전압값이 곧 암페어다
        i = d[a.v_shunt_col] * I_SCALE[a.i_unit] / a.shunt_ohm
    v = d[a.v_col] if a.v_col else a.vsupply
    return t, i * v


# ────────────────────────────────────────────────────── 게이트 구간
def gate_spans(t, g, thr):
    """게이트가 high 인 구간의 (시작 인덱스, 끝 인덱스+1) 목록.

    ⚠️ 첫/마지막 구간이 트레이스 경계에서 잘렸으면 **버린다.** 잘린 구간은
    에너지가 작게 나와 분포의 왼쪽 꼬리를 오염시킨다.
    """
    hi = g > thr
    if hi.size == 0:
        return []
    edges = np.diff(hi.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if hi[0]:
        starts.insert(0, 0)          # 시작부터 high — 잘린 구간
    if hi[-1]:
        ends.append(hi.size)         # 끝까지 high — 잘린 구간
    spans = list(zip(starts, ends))
    trimmed = 0
    if spans and spans[0][0] == 0:
        spans.pop(0)
        trimmed += 1
    if spans and spans[-1][1] == hi.size:
        spans.pop()
        trimmed += 1
    return spans, trimmed


def integrate(t, p, i0, i1):
    """[i0, i1) 구간의 에너지(J)와 지속시간(s). 사다리꼴 적분."""
    if i1 - i0 < 2:
        return 0.0, 0.0
    return float(TRAPZ(p[i0:i1], t[i0:i1])), float(t[i1 - 1] - t[i0])


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


# ─────────────────────────────────────────────────────────── 보고
def report_gated(t, p, g, a):
    spans, trimmed = gate_spans(t, g, a.gate_thr)
    if not spans:
        raise SystemExit("게이트 구간이 없다. --gate-col / --gate-thr 확인.")

    # idle = 게이트 밖 구간의 평균 전력. 별도 idle 파일이 있으면 그쪽이 우선.
    mask = np.ones(t.size, dtype=bool)
    for i0, i1 in spans:
        mask[i0:i1] = False
    # 스위칭 과도 구간을 양옆으로 조금 빼고 센다 (--guard 샘플)
    if a.guard:
        for i0, i1 in spans:
            mask[max(0, i0 - a.guard):min(t.size, i1 + a.guard)] = False
    p_idle = float(p[mask].mean()) if mask.any() else 0.0
    idle_src = "게이트 밖"
    if a.idle:
        di = read_csv(a.idle, [a.t_col, a.i_col, a.v_shunt_col, a.v_col])
        _, pi = to_power(di, a)
        p_idle = float(pi.mean())
        idle_src = a.idle

    e_tot, dur = [], []
    for i0, i1 in spans:
        e, d = integrate(t, p, i0, i1)
        e_tot.append(e)
        dur.append(d)
    e_tot, dur = np.asarray(e_tot), np.asarray(dur)
    e_net = e_tot - p_idle * dur            # idle 차감 = 추론 순수 에너지

    print(f"게이트 구간           {len(spans):,}개"
          + (f"  (경계에서 잘린 {trimmed}개 제외)" if trimmed else ""))
    print(f"샘플링                {1.0/np.median(np.diff(t)):,.0f} Sa/s, "
          f"총 {t[-1]-t[0]:.3f}s")
    print(f"idle 평균 전력        {p_idle*1e3:.4f} mW   ({idle_src})")
    print()
    print(f"{'':<22}{'평균':>12}{'표준편차':>12}{'p99':>12}{'최악':>12}")
    print("-" * 70)
    print(f"{'추론 지속시간 (ms)':<22}{dur.mean()*1e3:>12.4f}"
          f"{dur.std()*1e3:>12.4f}{pct(dur,99)*1e3:>12.4f}{dur.max()*1e3:>12.4f}")
    print(f"{'구간 총 에너지 (uJ)':<22}{e_tot.mean()*1e6:>12.3f}"
          f"{e_tot.std()*1e6:>12.3f}{pct(e_tot,99)*1e6:>12.3f}"
          f"{e_tot.max()*1e6:>12.3f}")
    print(f"{'추론 순수 (uJ)':<22}{e_net.mean()*1e6:>12.3f}"
          f"{e_net.std()*1e6:>12.3f}{pct(e_net,99)*1e6:>12.3f}"
          f"{e_net.max()*1e6:>12.3f}")
    print(f"{'구간 평균 전력 (mW)':<22}"
          f"{(e_tot/np.maximum(dur,1e-12)).mean()*1e3:>12.4f}")
    print()
    # 하루 에너지 — 대표 그림의 가로축 (CLAUDE.md 9장)
    day = (p_idle + e_net.mean() * a.hop_hz) * 24.0
    print(f"hop {a.hop_hz:.1f}회/s 기준 하루 총 에너지  {day*1e3:.3f} mWh/day")
    print(f"  = (idle {p_idle*1e3:.4f}mW + 추론 "
          f"{e_net.mean()*a.hop_hz*1e3:.4f}mW) x 24h")
    print()
    print("⚠️ '시간 ↔ 전력 트레이드오프' 로 서술하지 말 것 (CLAUDE.md 2장).")
    print("   에너지 = 전력 x 시간이라 빨라지면 에너지도 함께 준다.")
    return {"n": len(spans), "p_idle_w": p_idle,
            "e_net_j_mean": float(e_net.mean()),
            "e_net_j_max": float(e_net.max()),
            "dur_s_mean": float(dur.mean()), "dur_s_max": float(dur.max()),
            "wh_per_day": day / 1000.0 * 1000.0}


def report_stat(t, p, a):
    """게이팅 없이 — 추론 N회 구간의 평균에서 idle 을 빼고 N 으로 나눈다."""
    if not a.n_inf:
        raise SystemExit("stat 모드는 --n-inf (구간 내 추론 횟수) 가 필요하다.")
    if not a.idle:
        raise SystemExit("stat 모드는 --idle (idle 트레이스) 가 필요하다.")
    di = read_csv(a.idle, [a.t_col, a.i_col, a.v_shunt_col, a.v_col])
    ti, pi = to_power(di, a)
    p_run, p_idle = float(p.mean()), float(pi.mean())
    span = float(t[-1] - t[0])
    e_inf = (p_run - p_idle) * span / a.n_inf

    print(f"구간 길이             {span:.3f}s, 추론 {a.n_inf:,}회")
    print(f"idle 평균 전력        {p_idle*1e3:.4f} mW  ({a.idle}, "
          f"{ti[-1]-ti[0]:.3f}s)")
    print(f"구동 중 평균 전력     {p_run*1e3:.4f} mW")
    print(f"증분 전력             {(p_run-p_idle)*1e3:.4f} mW")
    print(f"추론당 증분 에너지    {e_inf*1e6:.3f} uJ")
    print()
    print(f"시스템 평균 전력      {p_run*1e3:.4f} mW")
    print(f"하루 총 에너지        {p_run*24*1e3:.3f} mWh/day")
    print()
    print("⚠️ 이 모드는 **추론 시간을 모른다** — 증분 에너지만 나온다.")
    print("   지연은 DWT 사이클 카운터로 따로 잰다 (CLAUDE.md 7장).")
    print("⚠️ OrangePi 처럼 idle 이 지배적인 플랫폼은 세 지표를 모두 보고할 것:")
    print("   추론당 증분 에너지 / 시스템 평균 전력 / 하루 총 에너지.")
    return {"p_idle_w": p_idle, "p_run_w": p_run, "e_inf_j": e_inf,
            "wh_per_day": p_run * 24 / 1000.0}


# ───────────────────────────────────────────────────────── 자체 검증
def self_test():
    """알려진 에너지를 가진 합성 트레이스를 만들어 되찾는지 본다.

    장비가 없는 동안 이 도구가 맞게 동작하는지 확인할 유일한 방법이다.
    """
    import os
    import tempfile

    fs = 1_000_000.0                 # 1 MSa/s
    dur = 2.0
    n = int(fs * dur)
    t = np.arange(n) / fs
    vs = 3.7
    i_idle = 2.0e-3                  # 2 mA
    i_inf = 20.0e-3                  # 추론 중 20 mA
    t_inf = 0.8e-3                   # 0.8 ms
    hop = 0.25                       # 4회/s
    i = np.full(n, i_idle)
    g = np.zeros(n)
    starts = np.arange(0.1, dur - 0.05, hop)
    for s in starts:
        a0, a1 = int(s * fs), int((s + t_inf) * fs)
        i[a0:a1] = i_inf
        g[a0:a1] = 3.3
    rng = np.random.default_rng(0)
    i += rng.normal(0, 50e-6, n)     # 50 µA rms 잡음

    exp_net = (i_inf - i_idle) * vs * t_inf
    exp_idle = i_idle * vs

    path = os.path.join(tempfile.gettempdir(), "synthetic_power.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "current", "gpio"])
        for k in range(0, n, 1):
            w.writerow([f"{t[k]:.7f}", f"{i[k]:.9f}", f"{g[k]:.1f}"])

    print("합성 트레이스로 자체 검증")
    print(f"  설정: idle {i_idle*1e3:.1f}mA, 추론 {i_inf*1e3:.1f}mA x "
          f"{t_inf*1e3:.2f}ms, {len(starts)}회, {vs}V, 잡음 50uA rms")
    print(f"  기대: idle {exp_idle*1e3:.4f}mW, 추론 순수 {exp_net*1e6:.3f}uJ")
    print()

    a = argparse.Namespace(
        t_col="t", t_unit="s", i_col="current", i_unit="a",
        v_shunt_col=None, shunt_ohm=1.0, v_col=None, vsupply=vs,
        gate_col="gpio", gate_thr=1.65, guard=0, idle=None,
        hop_hz=4.0, mode="gated", n_inf=None)
    d = read_csv(path, ["t", "current", "gpio"])
    tt, pp = to_power(d, a)
    r = report_gated(tt, pp, d["gpio"], a)
    os.remove(path)

    print()
    ok = True
    for name, got, want, tol in (
            ("구간 수", r["n"], len(starts), 0),
            ("idle 전력 (mW)", r["p_idle_w"] * 1e3, exp_idle * 1e3, 0.02),
            ("추론 순수 (uJ)", r["e_net_j_mean"] * 1e6, exp_net * 1e6, 0.5),
            ("지속시간 (ms)", r["dur_s_mean"] * 1e3, t_inf * 1e3, 0.01)):
        good = abs(got - want) <= tol
        ok &= good
        print(f"  {'OK  ' if good else 'FAIL'}  {name:<18}"
              f"측정 {got:.4f}  기대 {want:.4f}  (허용 ±{tol})")
    print()
    print("전부 통과" if ok else "실패 — 적분·차감 경로를 볼 것")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="?", help="전력 트레이스 CSV")
    ap.add_argument("--mode", choices=("gated", "stat"), default="gated")
    ap.add_argument("--t-col", default="t")
    ap.add_argument("--t-unit", default="s", choices=tuple(T_SCALE))
    ap.add_argument("--i-col", default=None, help="전류 열")
    ap.add_argument("--i-unit", default="a", choices=tuple(I_SCALE))
    ap.add_argument("--v-shunt-col", default=None,
                    help="션트 양단 전압 열 (--i-col 대신)")
    ap.add_argument("--shunt-ohm", type=float, default=1.0)
    ap.add_argument("--v-col", default=None, help="공급전압 실측 열")
    ap.add_argument("--vsupply", type=float, default=3.7, help="Li-Po 공칭")
    ap.add_argument("--gate-col", default=None, help="GPIO 마킹 열")
    ap.add_argument("--gate-thr", type=float, default=1.65)
    ap.add_argument("--guard", type=int, default=0,
                    help="idle 계산에서 구간 양옆으로 뺄 샘플 수 (스위칭 과도)")
    ap.add_argument("--idle", default=None, help="별도 idle 트레이스 CSV")
    ap.add_argument("--n-inf", type=int, default=None, help="stat 모드 추론 횟수")
    ap.add_argument("--hop-hz", type=float, default=4.0,
                    help="상시 추론 주기 (하루 에너지 환산용)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        sys.exit(self_test())
    if not a.csv:
        ap.error("CSV 경로가 필요하다 (또는 --self-test)")
    if not a.i_col and not a.v_shunt_col:
        ap.error("--i-col 또는 --v-shunt-col 중 하나는 필요하다")

    cols = [a.t_col, a.i_col, a.v_shunt_col, a.v_col, a.gate_col]
    d = read_csv(a.csv, cols)
    t, p = to_power(d, a)

    if a.mode == "gated":
        if not a.gate_col:
            ap.error("gated 모드는 --gate-col 이 필요하다 "
                     "(게이팅이 없으면 --mode stat)")
        res = report_gated(t, p, d[a.gate_col], a)
    else:
        res = report_stat(t, p, a)

    if a.json:
        import json
        json.dump(res, open(a.json, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"저장: {a.json}")


if __name__ == "__main__":
    main()
