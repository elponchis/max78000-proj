#!/usr/bin/env python3
"""NPU 내부 합산 안 A/B/C 의 **합성기 수용 여부**만 확인한다 (ai8xize, 보드 미접촉).

②′ p1 (시드 1 q8 가중치)에 층을 덧붙인 YAML 과 가짜 정수 체크포인트를 만들어 ai8xize 가 받아들이는지,
가중치·데이터 메모리에 들어가는지 본다. 학습값이 아니라 **층 구성 검증**이다 — 층 B 의 가중치는
0.5(=64) 고정이고 층 A 의 출력 시프트는 시뮬레이션(tools/npu_readback_sim.py)이 고른 값이다.

실행 (WSL2): ~/ai8x-training/venv/bin/python tools/npu_readback_synth_check.py
출력: synthesis/rb-*-hwc.yaml, data/synth/rb-*-q8.pth.tar, data/synth/out/rb_* (ai8xize 결과), 표준출력 요약
"""
import json
import os
import subprocess
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = os.path.expanduser("~/ai8x-synthesis")
SYN = os.path.join(REPO, "data", "synth")
OUT = os.path.join(SYN, "out")
PY = os.path.join(S, "venv", "bin", "python")
NB = 32
Q8 = os.path.join(SYN, "safesound-v4d-fe-p1-q8.pth.tar")

HDR = """---
# NPU 내부 합산 안 {name} — **합성기 수용 검사용** (tools/npu_readback_synth_check.py 가 만들었다, 2026-10-21).
#   설계: docs/results/npu-readback-design.md 1절. 층 A = ②′ p1 (Conv1d 64→64 k=8) + Abs, 8비트 (출력 시프트 {s}).
#   {desc}
#   입력 (64, 64) int8 — ②′ 와 같은 판단 1회 청크. 보드용이 아니다 (p2 와의 가중치 배치 --weight-start 없음).
arch: ai85safesoundrb_{tag}
dataset: SafeSoundFEP1

layers:
"""

LAYER_A = """  # 층 A{k}: Gabor conv + Abs → 8비트 (시프트는 체크포인트 output_shift)
  - pad: 0
    activate: Abs
    in_offset: 0x0000
{inseq}    out_offset: {out}
    processors: 0xffffffffffffffff
    data_format: HWC
    operation: Conv1d
    kernel_size: 8
"""
LAYER_SUM = """  # 층 B{k}: cos+sin 같은 대역 8탭 합 (k=8, 가중치 0.5) → wide int32, 프레임 위치만 CPU 가 읽는다
  - pad: 0
    activate: None
    in_offset: {inp}
    out_offset: {out}
    processors: 0xffffffffffffffff
    operation: Conv1d
    kernel_size: 8
    output_width: 32
    output_processors: 0x00000000ffffffff
    output: true
"""
LAYER_AVG = """  # 층 B: 8위치 평균 풀링(floor) → 1×1 cos+sin 합 (가중치 0.5){tail}
  - pad: 0
    activate: None
    in_offset: {inp}
    out_offset: {out}
    processors: 0xffffffffffffffff
    operation: Conv1d
    kernel_size: 1
    avg_pool: 8
    pool_stride: 8
{width}    output: true
"""


def w_sum(k):
    W = np.zeros((NB, 2 * NB, k), dtype=np.int64)
    for b in range(NB):
        W[b, 2 * b, :] = 64
        W[b, 2 * b + 1, :] = 64
    return W


def make(tag, name, desc, layers_yaml, state):
    yaml_path = os.path.join(REPO, "synthesis", f"rb-{tag}-hwc.yaml")
    ck_path = os.path.join(SYN, f"rb-{tag}-q8.pth.tar")
    open(yaml_path, "w", encoding="utf-8").write(HDR.format(name=name, tag=tag, s=state["l0.output_shift"].item(), desc=desc) + layers_yaml)
    torch.save({"epoch": 0, "arch": f"ai85safesoundrb_{tag}", "state_dict": state, "extras": {"note": "합성기 수용 검사용 가짜 체크포인트"}}, ck_path)
    return yaml_path, ck_path


def main():
    sim = json.load(open(os.path.join(REPO, "docs", "results", "npu-readback-sim", "fe_s1.json"), encoding="utf-8"))
    shifts = sim["shifts"]
    Wa = torch.load(Q8, map_location="cpu", weights_only=False)["state_dict"]["fe.op.weight"].to(torch.int64)
    assert tuple(Wa.shape) == (2 * NB, 64, 8)
    gain = torch.tensor(sim["gain"], dtype=torch.int64).repeat_interleave(2)
    Wa = Wa * gain[:, None, None]
    assert Wa.abs().max() <= 127, "대역 배율이 가중치 여유를 넘었다"

    def A(k, s, inseq=False, out="0x2000"):
        return LAYER_A.format(k=k, inseq="    in_sequences: -1\n" if inseq else "", out=out)

    def st(pairs):
        d = {}
        for i, (w, s) in enumerate(pairs):
            d[f"l{i}.op.weight"] = torch.as_tensor(w, dtype=torch.float32)
            d[f"l{i}.output_shift"] = torch.tensor([float(s)])
        return d

    cases = {}
    s0 = shifts["A-sum"]
    cases["asum"] = make("asum", "A-sum", "층 B = Conv1d(64→32, k=8) 가중치 0.5, wide int32 (8위치 합, 정확).",
                         A(1, s0) + LAYER_SUM.format(k=1, inp="0x2000", out="0x4000"), st([(Wa, s0), (w_sum(8), 0)]))
    s1 = shifts["A-avg"]
    cases["aavg"] = make("aavg", "A-avg", "층 B = AvgPool1d(8) → Conv1d(64→32, k=1) 가중치 0.5, wide int32 (평균 floor).",
                         A(1, s1) + LAYER_AVG.format(inp="0x2000", out="0x4000", tail=" → wide int32", width="    output_width: 32\n"),
                         st([(Wa, s1), (w_sum(1), 0)]))
    s2 = shifts["C"]
    cases["c"] = make("c", "C", "층 B = AvgPool1d(8) → Conv1d(64→32, k=1) 가중치 0.5, **8비트** 출력 (CPU 는 256칸 LUT).",
                      A(1, s2) + LAYER_AVG.format(inp="0x2000", out="0x4000", tail=" → 8비트", width=""),
                      st([(Wa, s2), (w_sum(1), 0)]))
    sb = shifts["B"]
    y = ""
    pairs = []
    for k, s in enumerate((sb + 4, sb, sb - 4)):
        y += A(k + 1, s, inseq=(k > 0), out=f"0x{0x2000 + 0x400 * k:04x}") + LAYER_SUM.format(k=k + 1, inp=f"0x{0x2000 + 0x400 * k:04x}", out=f"0x{0x4000 + 0x400 * k:04x}")
        pairs += [(Wa, s), (w_sum(8), 0)]
    cases["b"] = make("b", "B", "층 A 를 시프트 s+4 / s / s−4 로 3벌 (in_sequences: -1 로 같은 입력), 각각 A-sum 의 층 B.", y, st(pairs))

    common = ["--verbose", "--test-dir", OUT, "--device", "MAX78000", "--compact-data", "--mexpress", "--timer", "0", "--overwrite"]
    summary = {}
    for tag, (yaml_path, ck_path) in cases.items():
        pfx = f"rb_{tag}"
        r = subprocess.run([PY, "ai8xize.py", "--prefix", pfx, "--checkpoint-file", ck_path, "--config-file", yaml_path] + common,
                           cwd=S, capture_output=True, text=True)
        logp = os.path.join(SYN, f"rb-{tag}-ai8xize.out")
        open(logp, "w", encoding="utf-8").write(r.stdout + "\n--- stderr ---\n" + r.stderr)
        ok = r.returncode == 0 and os.path.isfile(os.path.join(OUT, pfx, "cnn.c"))
        err = [l for l in (r.stdout + r.stderr).splitlines() if "ERROR" in l or "error" in l.lower()][:3]
        info = {}
        lt = os.path.join(OUT, pfx, "log.txt")
        if os.path.isfile(lt):
            for l in open(lt, encoding="utf-8"):
                for key in ("Kernel length", "Kernel offset", "Total weight", "Memory", "Layers"):
                    if l.strip().startswith(key):
                        info[key] = l.strip()[:120]
        summary[tag] = {"ok": ok, "rc": r.returncode, "errors": err, "yaml": os.path.relpath(yaml_path, REPO), "info": info}
        print(f"{tag:5s} {'수용' if ok else '거부'} rc={r.returncode} {err[:1]} {info.get('Kernel length', '')}")
    json.dump(summary, open(os.path.join(REPO, "docs", "results", "npu-readback-sim", "synth_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.exit(main())
