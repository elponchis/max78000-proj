#!/usr/bin/env python3
"""②′ 1패스(p1) 합성 확인 — 사전 등록 1단계 (docs/results/fe-design.md 5절).

확인하는 것 셋:
  1. Conv1d(64→64, k=8, wide) 가 MAX78000 에 합성되는가 (출력 폭 32, 입력 (64, 64))
  2. 2패스(가, data/synth/out/safesound_melu1000_v4d) 와 가중치 메모리가 겹치지 않는가 (--weight-start)
  3. p1 의 NPU 사이클 — 지연·에너지 **추정**의 입력 (보드 실측 전)

체크포인트는 멜 Gabor 초기값(모델이 내장) 그대로를 quantize.py 로 int8 화한 것이다 —
사이클·메모리는 가중치 값과 무관하므로 학습 전에 확인할 수 있다 (curve_synth_check.py 와 같은 논리).

사용 (WSL2):
    ~/ai8x-training/venv/bin/python tools/fe_synth_check.py
"""
import importlib.util
import os
import re
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = os.path.expanduser("~/ai8x-training")
S = os.path.expanduser("~/ai8x-synthesis")
OUT = os.path.join(REPO, "data", "synth")
LOG = os.path.join(REPO, "data", "logs-local", "synth_fe")
P2_LOG = os.path.join(OUT, "out", "safesound_melu1000_v4d", "log.txt")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def p2_weight_end():
    """가 (2패스) 합성 로그의 Kernel offset/length 에서 가중치 끝 단위를 구한다."""
    txt = open(P2_LOG, encoding="utf-8").read()
    off = [int(v) for v in re.search(r"Kernel offset\s*=\s*\[([^\]]*)\]", txt).group(1).split(",")]
    ln = [int(v) for v in re.search(r"Kernel length\s*=\s*\[([^\]]*)\]", txt).group(1).split(",")]
    end = max(o + n for o, n in zip(off, ln))
    return off, ln, end


def main():
    sys.path.insert(0, T)
    import torch                                     # noqa: PLC0415
    import ai8x                                      # noqa: PLC0415
    ai8x.set_device(85, False, False)
    m = load(os.path.join(REPO, "models/ai85net-safesound-fe.py"), "fe")

    # ── 체크포인트 (float, 멜 Gabor 초기값) ─────────────────────────────
    net = m.ai85safesoundfe()                        # 초기값 포함
    p1 = m.ai85safesoundfe_p1()
    with torch.no_grad():
        p1.fe.op.weight.copy_(net.fe.op.weight)
    ck_f = os.path.join(OUT, "safesound-fe-p1.pth.tar")
    ck_q = os.path.join(OUT, "safesound-fe-p1-q8.pth.tar")
    torch.save({"epoch": 0, "state_dict": p1.state_dict(), "arch": "ai85safesoundfe_p1",
                "extras": {"init": "mel_gabor_fb"}}, ck_f)
    # 샘플 입력: 판단 1회 (64, 64) — 실제 학습 창 하나의 앞 4,096 샘플이면 좋지만 사이클과 무관하므로 난수
    rng = np.random.default_rng(0)
    np.save(os.path.join(S, "tests", "sample_safesoundfep1.npy"), rng.integers(-128, 128, (64, 64)).astype(np.int64))

    off, ln, end = p2_weight_end()
    ws = (end + 3) // 4 * 4
    print(f"2패스(가) 커널 구간: offset {off}, length {ln} → 끝 {end} 단위 → p1 --weight-start {ws}")

    # ── 양자화 ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    r = subprocess.run([os.path.join(S, "venv/bin/python"), "quantize.py", ck_f, ck_q, "--device", "MAX78000"],
                       cwd=S, capture_output=True, text=True)
    open(LOG + "-quantize.out", "w").write(r.stdout + r.stderr)
    print(f"quantize exit {r.returncode}")
    if r.returncode:
        print(r.stdout[-1500:], r.stderr[-1500:])
        return 1

    # ── 합성 ────────────────────────────────────────────────────────────
    cmd = [os.path.join(S, "venv/bin/python"), "ai8xize.py", "--verbose", "--test-dir", os.path.join(OUT, "out"),
           "--prefix", "safesound_fe_p1", "--checkpoint-file", ck_q,
           "--config-file", os.path.join(REPO, "synthesis", "safesound-fe-p1-hwc.yaml"),
           "--device", "MAX78000", "--compact-data", "--mexpress", "--timer", "0",
           "--display-checkpoint", "--overwrite", "--weight-start", str(ws)]
    r = subprocess.run(cmd, cwd=S, capture_output=True, text=True)
    open(LOG + "-ai8xize.out", "w").write(r.stdout + r.stderr)
    print(f"ai8xize exit {r.returncode}")
    err = [l for l in (r.stdout + r.stderr).splitlines() if "ERROR" in l or "WARNING" in l]
    for l in err[:8]:
        print("  ", l)
    log = os.path.join(OUT, "out", "safesound_fe_p1", "log.txt")
    if not os.path.isfile(log):
        print("log.txt 없음 — 합성 실패")
        return 1
    txt = open(log, encoding="utf-8").read()
    for pat in (r"Kernel offset\s*=.*", r"Kernel length\s*=.*", r"Total\s+[0-9,]+ cycles", r"Weight memory:.*", r"Bias memory:.*"):
        mm = re.search(pat, txt)
        print("  ", mm.group(0) if mm else f"({pat} 없음)")
    cyc = int(re.search(r"Total\s+([0-9,]+) cycles", txt).group(1).replace(",", ""))
    print(f"\np1 사이클 {cyc:,} → CNN 50 MHz 에서 {cyc/50e3:.3f} ms (추정; 보드 실측 전). "
          f"가 의 p2 26,337 사이클 (보드 실측 추론 0.531 ms) 과 합쳐 {(cyc+26337)/50e3:.3f} ms (추정)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
