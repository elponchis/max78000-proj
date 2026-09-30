#!/usr/bin/env python3
"""여러 학습 실행을 **고정 오경보 지점**에서 나란히 비교한다 (로컬용).

Colab 셀 25 가 하는 일을 WSL2 에서 한다. 1D 실험이 CPU 에서 더 빠르다는 것이
확인된 뒤(④ CPU 22분 vs GPU 36분) 로컬 실행이 늘었기 때문이다.

**비교는 argmax 가 아니라 고정 오경보(300/h, 1000/h)에서 한다.** argmax 는 각
모델이 알아서 정한 동작점이라, 배경음을 더 버린 모델이 이벤트 재현율만 높게 보인다.

`--delta` 로 쌍을 주면 차이를 이름표와 함께 찍는다. 잡음 눈금(장치 차이·시드
차이)을 먼저 보고 나서 개선폭을 해석하기 위한 것이다.

사용법 (WSL2):
    python3 tools/compare_runs.py --ai8x ~/ai8x-training \\
        --data ~/ai8x-training/data \\
        --run "④ GPU s1:wave:<ckpt>" --run "④ CPU s1:wave:<ckpt>" \\
        --delta "장치 차이:④ CPU s1:④ GPU s1"
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POINTS = ("1000/h", "300/h")
EPOCH_RE = re.compile(r"Epoch:\s*\[(\d+)\]")


def welch(a, b):
    """Welch t 로 (평균차의 95% CI, 차이, A평균, A표준편차, B평균, B표준편차).

    등분산을 가정하지 않는다 — 계열마다 시드 변동이 크게 다르다
    (④ 0.055 vs ① 0.009). scipy 를 쓰지 않으려고 t 임계값은 표로 둔다.
    """
    import math
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return 0.0, 0.0, ma - mb, ma, math.sqrt(va), mb, math.sqrt(vb)
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    # 양측 95% t 임계값 (df 반내림). scipy 없이 쓰려고 표로 둔다.
    T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131,
         20: 2.086, 30: 2.042, 60: 2.000}
    k = max([x for x in T if x <= max(1, int(df))] or [1])
    t = T[k]
    d = ma - mb
    return (d - t * se, d + t * se, d, ma, math.sqrt(va), mb, math.sqrt(vb))


def run_finished(ckpt_path):
    """그 체크포인트를 낸 학습이 **끝났는가**. (끝났나, 사유) 를 돌려준다.

    ⚠️ QAT 는 에폭 60부터 시작하므로 `*_qat_best.pth.tar` 는 **학습 도중에도**
    생긴다. 그것을 확정값처럼 읽어 보고한 사고가 있었다 — D-1+C′ 를 83/150 에폭
    시점에 재서 0.4426 으로 보고했는데 완료 후 값은 0.4927 이었다.

    판정은 같은 폴더의 학습 로그에 **최종 테스트 단계**가 있는지로 한다.
    distiller 는 학습이 다 끝난 뒤에만 `--- test (ckpt) ---` / `Test: [` 를 찍는다.
    """
    d = os.path.dirname(os.path.abspath(ckpt_path))
    logs = [f for f in os.listdir(d) if f.endswith(".log")] if os.path.isdir(d) else []
    if not logs:
        return False, f"로그가 없어 완료를 확인할 수 없다 ({os.path.basename(d)})"
    path = os.path.join(d, logs[0])
    last_epoch, has_test = -1, False
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if "Test: [" in ln or "--- test (" in ln:
                has_test = True
            elif "Training epoch:" in ln:
                pass
            m = EPOCH_RE.search(ln)
            if m:
                last_epoch = max(last_epoch, int(m.group(1)))
    if has_test:
        return True, ""
    return False, f"로그에 최종 테스트 단계가 없다 (마지막 에폭 {last_epoch})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[],
                    metavar="이름:config:체크포인트",
                    help="비교할 실행. 여러 번 줄 수 있다")
    ap.add_argument("--delta", action="append", default=[],
                    metavar="설명:A:B", help="A − B 차이를 찍는다")
    ap.add_argument("--ai8x", default=os.path.expanduser("~/ai8x-training"))
    ap.add_argument("--data", default=os.path.expanduser("~/ai8x-training/data"))
    ap.add_argument("--hop-ms", type=int, default=250)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--json", help="모은 결과를 저장할 경로")
    ap.add_argument("--cache", default=os.path.join(tempfile.gettempdir(),
                                                    "compare_runs"),
                    help="실행별 평가 JSON 캐시 (같은 체크포인트는 재사용)")
    ap.add_argument("--seed-group", action="append", default=[],
                    metavar="이름:run1,run2,...",
                    help="같은 설정의 시드 반복 실행 묶음. **그 계열의 시드 "
                         "폭(max-min)** 을 계산해 잡음 눈금으로 쓴다. 계열마다 "
                         "안정성이 다르다 — ④는 0.052, ①은 0.001 수준이다")
    ap.add_argument("--seed-diff", action="append", default=[],
                    metavar="설명:계열A:계열B",
                    help="두 **시드 그룹**의 차이를 평균±표준편차와 "
                         "**Welch t 95% CI** 로 낸다. range x 3 규칙은 파형 "
                         "계열에서 검정력이 너무 낮아(폭 0.055 x 3 = 0.16) "
                         "이쪽을 주 판정으로 쓴다")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="**전역** 허용. 쓰지 말 것 — 학습 중인 실행까지 통째로 "
                         "들어온다. 실제로 이 플래그 때문에 82/150 에폭짜리를 "
                         "확정값으로 집었다. 대신 `--run` 이름 끝에 `!` 를 붙여 "
                         "**그 실행 하나만** 신뢰한다고 표시할 것")
    a = ap.parse_args()
    if not a.run:
        sys.exit("[에러] --run 이 필요하다")

    os.makedirs(a.cache, exist_ok=True)
    out = {}
    for spec in a.run:
        label, cfg, ck = spec.split(":", 2)
        # 이름 끝의 `!` = "완주를 사람이 확인했다". 옆에 로그가 없는 경우
        # (Colab 에서 받은 단독 체크포인트) 를 위한 것이고, **그 실행 하나에만**
        # 적용된다. 전역 --allow-incomplete 와 달리 다른 실행은 계속 걸러진다.
        trusted = label.endswith("!")
        if trusted:
            label = label[:-1].rstrip()
        if not os.path.isfile(ck):
            print(f"  [건너뜀] {label}: 체크포인트 없음 {ck}")
            continue
        done, why = run_finished(ck)
        if not done and trusted:
            print(f"  [신뢰] {label}: 로그가 없지만 완주 확인됨으로 표시됐다")
            done = True
        if not done and not a.allow_incomplete:
            print(f"  [건너뜀] {label}: **학습 미완료** — {why}")
            print("           (QAT 는 에폭 60부터라 qat_best 가 도중에도 생긴다. "
                  "굳이 보려면 --allow-incomplete)")
            continue
        if not done:
            print(f"  [주의] {label}: 학습 미완료인데 포함한다 — {why}")
        j = os.path.join(a.cache, label.replace(" ", "_").replace("/", "_")
                         + ".json")
        if not os.path.isfile(j) or os.path.getmtime(j) < os.path.getmtime(ck):
            print(f"  평가 중: {label} ({cfg})")
            r = subprocess.run(
                [os.path.join(a.ai8x, "venv", "bin", "python"),
                 os.path.join(REPO, "tools", "eval_confusion.py"),
                 "--checkpoint", ck, "--data", a.data, "--ai8x", a.ai8x,
                 "--config", cfg, "--hop-ms", str(a.hop_ms),
                 "--n-boot", str(a.n_boot), "--json", j],
                capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  [실패] {label}: {r.stdout[-400:]}{r.stderr[-400:]}")
                continue
        else:
            print(f"  캐시 사용: {label}")
        out[label] = json.load(open(j, encoding="utf-8"))

    if not out:
        sys.exit("[에러] 평가된 실행이 없다")
    names = list(next(iter(out.values()))["window"]["recall"])

    for pt in POINTS:
        print(f"\n=== 고정 오경보 {pt} ===")
        print(f"  {'실행':<16}{'macroF1':>9}   "
              + "".join(f"{n[:8]:>9}" for n in names))
        print("  " + "-" * (25 + 9 * len(names)))
        for lb, d in out.items():
            t = (d.get("fixed_fa") or {}).get("targets", {}).get(pt)
            if not t:
                print(f"  {lb:<16}{'—':>9}")
                continue
            print(f"  {lb:<16}{t['macro_f1']:>9.4f}   "
                  + "".join(f"{100*t['recall'][n]:>8.1f}%" for n in names))

    # ── 계열별 시드 폭 ────────────────────────────────────────────
    noise, vals = {}, {}
    if a.seed_group:
        print()
        print("=== 계열별 시드 잡음 (300/h macro-F1 의 max-min) ===")
        for g in a.seed_group:
            name, _, members = g.partition(":")
            vals_, got = [], []
            for m in [x.strip() for x in members.split(",") if x.strip()]:
                t = (out.get(m, {}).get("fixed_fa") or {}).get("targets", {}
                                                              ).get("300/h")
                if t:
                    vals_.append(t["macro_f1"])
                    got.append(m)
            if len(vals_) < 2:
                print(f"  {name:<12} 시드가 {len(vals_)}개뿐 — 폭을 낼 수 없다")
                continue
            w = max(vals_) - min(vals_)
            noise[name] = w
            vals[name] = list(vals_)
            mean = sum(vals_) / len(vals_)
            print(f"  {name:<12} n={len(vals)}  평균 {mean:.4f}  "
                  f"폭 {w:.4f}   " + " ".join(f"{v:.4f}" for v in vals_))
        print("  ⚠️ 폭은 **관측된 max-min** 이지 신뢰구간이 아니다. n 이 작으면")
        print("     실제 변동을 과소평가한다 — 보수적으로 쓸 것.")

    if a.delta:
        print("\n=== 차이 (macro-F1) ===")
        print(f"  {'설명':<22}" + "".join(f"{p:>12}" for p in POINTS)
              + "   판정")
        print("  " + "-" * (22 + 12 * len(POINTS) + 8))
        for spec in a.delta:
            parts = spec.split(":")
            # "설명:x:y" 또는 "설명:x:y:계열이름"
            if len(parts) >= 4:
                why, x, y, fam = parts[0], parts[1], parts[2], parts[3]
            else:
                why, x, y, fam = parts[0], parts[1], parts[2], None
            cells, verdict = "", ""
            d300 = None
            for pt in POINTS:
                tx = (out.get(x, {}).get("fixed_fa") or {}).get("targets", {}).get(pt)
                ty = (out.get(y, {}).get("fixed_fa") or {}).get("targets", {}).get(pt)
                if tx and ty:
                    d = tx["macro_f1"] - ty["macro_f1"]
                    cells += f"{d:>+12.4f}"
                    if pt == "300/h":
                        d300 = d
                else:
                    cells += f"{'—':>12}"
            # 계열을 `+` 로 여러 개 주면 **큰 자**를 쓴다. 비교 쌍의 두
            # 구성이 서로 다른 계열이면 안정성이 낮은 쪽이 판정을 지배해야
            # 한다 — 작은 자를 대면 잡음을 실재로 오인한다 (2026-09-30 규칙).
            fams = [f.strip() for f in fam.split("+")] if fam else []
            have = [f for f in fams if f in noise]
            if d300 is not None and have:
                n = max(noise[f] for f in have)
                fam = "+".join(have) + ("" if len(have) == 1 else
                                        f" 중 큰 자 {max(have, key=lambda f: noise[f])}")
                verdict = ("잡음 안" if abs(d300) <= n
                           else f"잡음 밖 ({abs(d300)/max(n,1e-9):.1f}배)")
                verdict += f" [{fam} {n:.4f}]"
            elif fams:
                verdict = f"[{fam} 기준 없음]"
            print(f"  {why:<22}{cells}   {verdict}")
        print("  ⚠️ 판정은 **그 계열의 시드 폭**을 자로 쓴다 — 계열마다 다르다.")
        print("     계열을 지정하지 않은 행은 판정이 비어 있다.")

    # ── 시드 그룹 간 Welch t (주 판정) ─────────────────────────────
    if a.seed_diff:
        print()
        print("=== 시드 그룹 차이 — 평균±표준편차와 Welch t 95% CI ===")
        print(f"  {'설명':<26}{'A 평균±sd(n)':>20}{'B 평균±sd(n)':>20}"
              f"{'차이':>9}{'95% CI':>20}  판정")
        print("  " + "-" * 100)
        for spec in a.seed_diff:
            why, ga, gb = spec.split(":", 2)
            va, vb = vals.get(ga, []), vals.get(gb, [])
            if len(va) < 2 or len(vb) < 2:
                print(f"  {why:<26}  시드가 모자란다 (A {len(va)}, B {len(vb)})")
                continue
            lo, hi, d, ma, sa, mb, sb = welch(va, vb)
            sig = "**차이 있음**" if (lo > 0 or hi < 0) else "0 을 포함 — 미확정"
            print(f"  {why:<26}{f'{ma:.4f}±{sa:.4f}({len(va)})':>20}"
                  f"{f'{mb:.4f}±{sb:.4f}({len(vb)})':>20}{d:>+9.4f}"
                  f"{f'[{lo:+.4f}, {hi:+.4f}]':>20}  {sig}")
        print("  ⚠️ n 이 3~5 라 t 분포가 넓다. **CI 가 0 을 포함한다고 '차이가")
        print("     없다' 가 아니라 '이 표본으로는 못 가른다' 다.**")
        print("  ⚠️ 시드는 독립이지만 **같은 데이터**를 쓴다 — 데이터 표본")
        print("     변동은 이 CI 에 들어 있지 않다.")

    print("\n  참고: argmax 오경보/h — "
          + ", ".join(f"{lb} {d['false_alarm']['per_hour']:,.0f}"
                      for lb, d in out.items()))
    print("  ⚠️ argmax 는 동작점이 모델마다 달라 비교 근거가 아니다.")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({lb: {"fixed_fa": d.get("fixed_fa"),
                            "window": d["window"], "origin": d["origin"],
                            "false_alarm": d["false_alarm"]}
                       for lb, d in out.items()}, f, ensure_ascii=False, indent=2)
        print(f"\n  저장: {a.json}")


if __name__ == "__main__":
    main()
