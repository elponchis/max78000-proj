#!/usr/bin/env python3
"""학습 실행의 **완주 여부**를 소급 감사한다.

왜 필요한가: 체인 스크립트가 찍던 `(exit 0)` 이 **종료 코드가 아니었다**
(2026-10-01). `echo "$(date) … (exit $?)"` 에서 명령 치환이 먼저 실행되어
`$?` 가 `date` 의 것으로 덮인다. 즉 **학습이 죽어도 0 으로 찍혔을 것**이고,
그 위에서 chain11·13·14 의 판정을 냈다.

그래서 체인 로그를 믿지 않고 **학습 로그에서 직접** 세 가지를 본다.

| 검사 | 통과 조건 | 왜 |
|---|---|---|
| 마지막 에폭 | `Epoch: [149]` 가 있다 | 150 에폭을 다 돌았나 |
| 최종 테스트 | `Test: [` / `--- test (` 가 있다 | distiller 는 다 끝난 뒤에만 찍는다 |
| 체크포인트 | `*_qat_best.pth.tar` 가 있다 | 평가가 실제로 쓴 파일 |

⚠️ 셋이 서로를 대신하지 못한다. `qat_best` 는 **에폭 60부터** 생기므로
파일이 있다고 완주가 아니고 (`compare_runs.run_finished` 의 주석 참조),
최종 테스트가 있어도 에폭 수는 따로 봐야 한다.

⚠️ **진행 중과 실패를 반드시 구분한다.** 체인이 도는 중에 감사하면 아직
시작도 안 한 실행이 "폴더 없음" 으로, 학습 중인 실행이 "최종 테스트 없음"
으로 찍혀 **멀쩡한 계열이 보류 대상이 된다.** `/proc` 에서 살아 있는
`train.py` 의 `--name` 을 읽어 가른다.

사용 (WSL2):
    ~/ai8x-training/venv/bin/python tools/audit_runs.py
    ~/ai8x-training/venv/bin/python tools/audit_runs.py --epochs 150
"""
import argparse
import glob
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
from compare_runs import run_finished            # noqa: E402

EPOCH_RE = re.compile(r"Epoch: \[\s*(\d+)\]")

# 판정에 쓴 실행을 **계열별로** 묶는다. 하나라도 미완주면 그 계열의 판정이
# 통째로 흔들리므로, 개별 실행이 아니라 계열 단위로 보고한다.
GROUPS = {
    "chain13 ④ (5시드)": ["safesound-v1cpu", "safesound-v1cpu-s2",
                         "safesound-v1cpu-s3", "safesound-v1cpu-s4",
                         "safesound-v1cpu-s5"],
    "chain13 (a) (5시드)": ["safesound-logstage-v1", "safesound-logstage-v1-s2",
                           "safesound-logstage-v1-s3", "safesound-logstage-v1-s4",
                           "safesound-logstage-v1-s5"],
    "chain11 학습 곡선": ["safesound-mel-siren25-v1", "safesound-mel-siren50-v1",
                        "safesound-mel-siren75-v1"],
    "chain14 mix50": ["safesound-mel-mix50-v1", "safesound-mix50-v1"],
    "chain15 mix50 (3시드)": ["safesound-mix50-v1-s2", "safesound-mix50-v1-s3",
                             "safesound-mel-mix50-v1-s2",
                             "safesound-mel-mix50-v1-s3"],
    "D-1 (3시드)": ["safesound-fbabs-v1", "safesound-fbabs-v1-s2",
                   "safesound-fbabs-v1-s3"],
    "D-1(√) (3시드)": ["safesound-fbw05-v1", "safesound-fbw05-v1-s2",
                      "safesound-fbw05-v1-s3"],
    "① (3시드)": ["safesound-mel-v1", "safesound-mel-v1-s2",
                 "safesound-mel-v1-s3"],
    "기타 비교": ["safesound-fbk2-v1", "safesound-fbk4-v1", "safesound-mellin-v1",
                "safesound-melcbrt-v1", "safesound-mellog72-v1",
                "safesound-w2d-v1", "safesound-v1ft", "safesound-v1ft-s2",
                "safesound-fbnorm2-v1"],
}

# 로컬에 학습 로그가 없는 실행. Colab 에서 체크포인트만 가져왔다.
# 완주는 **체크포인트의 epoch 필드**로 확인한다 — 로그만큼 강하지는 않다.
EXTERNAL = {
    "safesound-mel-v1": "data/safesound-mel-v1_qat_best.pth.tar",
}


def _cmdlines():
    """살아 있는 프로세스의 argv 목록."""
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open("/proc/%s/cmdline" % pid, "rb") as f:
                yield f.read().split(b"\0")
        except OSError:
            continue


def live_runs():
    """지금 돌고 있는 학습의 `--name` 집합."""
    out = set()
    for arg in _cmdlines():
        if not any(a.endswith(b"train.py") for a in arg):
            continue
        if b"--name" in arg:
            out.add(arg[arg.index(b"--name") + 1].decode())
    return out


def chain_alive():
    """체인 스크립트가 아직 도는가 — 그러면 '미시작' 은 실패가 아니다."""
    return any(any(b"run_chain" in a for a in arg) for arg in _cmdlines())


def last_epoch(run_dir):
    """로그에서 본 가장 큰 에폭 번호. 없으면 -1."""
    out = -1
    for lg in glob.glob(os.path.join(run_dir, "*.log")):
        with open(lg, encoding="utf-8", errors="replace") as f:
            for ln in f:
                m = EPOCH_RE.search(ln)
                if m:
                    out = max(out, int(m.group(1)))
    return out


def ckpt_epoch(path):
    """체크포인트에 기록된 에폭. 로그가 없는 실행의 차선책."""
    import torch                                   # noqa: PLC0415
    try:
        return torch.load(path, map_location="cpu").get("epoch", -1)
    except Exception:                              # noqa: BLE001
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150,
                    help="기대 에폭 수 (마지막 에폭 번호는 이보다 1 작다)")
    ap.add_argument("--logs", default=os.path.join(REPO, "data/logs-local"))
    a = ap.parse_args()
    want = a.epochs - 1
    live, alive = live_runs(), chain_alive()

    if live:
        print("진행 중: " + ", ".join(sorted(live)))
    elif alive:
        print("체인이 돌고 있다 (학습 사이 전환 중)")
    print()
    print("%-32s%7s%6s%10s%7s" % ("실행", "에폭", "완주", "최종테스트", "ckpt"))
    print("-" * 70)
    bad, weak, prog = [], [], []

    for g, runs in GROUPS.items():
        print("\n## " + g)
        for n in runs:
            ds = sorted(d for d in glob.glob(os.path.join(a.logs, n + "___*"))
                        if os.path.isdir(d))
            if not ds:
                if n in EXTERNAL:
                    p = os.path.join(REPO, EXTERNAL[n])
                    e = ckpt_epoch(p) if os.path.isfile(p) else -1
                    good = e >= want
                    print("%-32s%7s%6s%10s%7s" % (
                        n, e, "✔" if good else "✘", "(로그없음)",
                        "✔" if e >= 0 else "✘"))
                    (weak if good else bad).append(
                        (g, n, "로컬 로그 없음 — 체크포인트 epoch %s 로만 확인" % e))
                elif n in live or alive:
                    print("%-32s%7s%6s%10s%7s  대기 중" % (n, "—", "…", "—", "—"))
                    prog.append((g, n, "아직 시작하지 않았다"))
                else:
                    print("%-32s%7s%6s%10s%7s  폴더 없음" % (n, "—", "✘", "—", "—"))
                    bad.append((g, n, "로그 폴더 없음"))
                continue

            d = ds[-1]
            ck = os.path.join(d, n + "_qat_best.pth.tar")
            ok, why = run_finished(ck)
            e = last_epoch(d)
            hck = os.path.isfile(ck)
            full = e >= want
            running = n in live
            tag = "…" if (running and not full) else ("✔" if full else "✘")
            note = ""
            if not (full and ok and hck):
                note = ("  ← 학습 중 (%d/%d)" % (e + 1, a.epochs) if running
                        else "  ← " + (why or "에폭 부족"))
                (prog if running else bad).append(
                    (g, n, why or "마지막 에폭 %d (기대 %d)" % (e, want)))
            print("%-32s%7d%6s%10s%7s%s" % (
                n, e, tag, "✔" if ok else "✘", "✔" if hck else "✘", note))

    print("\n" + "=" * 66)
    if prog:
        print("… 진행 중 %d건 — **실패가 아니다**:" % len(prog))
        for g, n, w in prog:
            print("   [%s] %s — %s" % (g, n, w))
        print()
    if weak:
        print("⚠️ 약한 확인 %d건 — 로그가 없어 체크포인트로만 봤다:" % len(weak))
        for g, n, w in weak:
            print("   [%s] %s — %s" % (g, n, w))
        print()
    if bad:
        print("✘ 미완주/결손 %d건 — **아래 계열의 판정을 보류할 것**:" % len(bad))
        for g, n, w in bad:
            print("   [%s] %s — %s" % (g, n, w))
        return 1
    print("✔ 미완주 없음 — 판정에 쓴 체크포인트가 모두 최종 qat_best 다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
