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
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="**학습이 끝나지 않은 실행도 포함**한다. 기본은 제외다 — "
                         "QAT 는 에폭 60부터라 `qat_best` 가 학습 도중에도 생기고, "
                         "그걸 집어 확정값처럼 보고한 사고가 있었다 (83/150 에폭 "
                         "시점 0.4426 → 완료 후 0.4927)")
    a = ap.parse_args()
    if not a.run:
        sys.exit("[에러] --run 이 필요하다")

    os.makedirs(a.cache, exist_ok=True)
    out = {}
    for spec in a.run:
        label, cfg, ck = spec.split(":", 2)
        if not os.path.isfile(ck):
            print(f"  [건너뜀] {label}: 체크포인트 없음 {ck}")
            continue
        done, why = run_finished(ck)
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

    if a.delta:
        print("\n=== 차이 (macro-F1) ===")
        print(f"  {'설명':<22}" + "".join(f"{p:>12}" for p in POINTS))
        print("  " + "-" * (22 + 12 * len(POINTS)))
        for spec in a.delta:
            why, x, y = spec.split(":", 2)
            cells = ""
            for pt in POINTS:
                tx = (out.get(x, {}).get("fixed_fa") or {}).get("targets", {}).get(pt)
                ty = (out.get(y, {}).get("fixed_fa") or {}).get("targets", {}).get(pt)
                cells += (f"{tx['macro_f1'] - ty['macro_f1']:>+12.4f}"
                          if tx and ty else f"{'—':>12}")
            print(f"  {why:<22}{cells}")
        print("  ⚠️ 개선폭은 **장치 차이·시드 차이보다 클 때만** 의미가 있다.")

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
