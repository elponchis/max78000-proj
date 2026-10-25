#!/usr/bin/env python3
"""구성 ②′ (와 곡선 변형 b64 / f31 / b64f31, 후속 연구 v2 · fe8) 보드 실측 준비 — 시드 1 체크포인트를 p1·p2 로 갈라 합성하고, 두 네트워크 측정 펌웨어를 조립·빌드한다.

하는 일 (WSL2, ai8x-training venv):
  1. `safesound-v4d-fe[-<변형>]` 시드 1 QAT 체크포인트 → p1 (`fe.*`, arch ai85safesoundfe_p1[_b64]) / p2 (`cls.*`, arch ai85safesoundmelnet_p4)
  2. 샘플 창 = ④ 합성에 쓴 시험셋 창 (`data/synth/sample_safesound-v4d-wave.npy`, (128,128) 접기) — 같은 로더·같은 창
  3. quantize.py 두 번 → int8 가중치로 **정수 프론트엔드**(exact int64 conv → |cos|+|sin| → 8위치 합 → log2_q8 → int8)
     를 호스트에서 계산하고, 학습 그래프(`AI85SafeSoundFE.frontend` + `intlog`, QAT 모드)와 비트 대조
  4. ai8xize 두 번 (p2 먼저 → 커널 끝 단위 → p1 은 그 뒤에 `--weight-start`) → p1 기대 출력(청크 3)이 정수 conv 와 같은지,
     청크 입력 워드 포장이 생성 sampledata.h 와 같은지, p2 기대 로짓이 simulate 모드 PyTorch 와 같은지 확인
  5. firmware/npu/safesound_fe[_<변형>]_v4d/ 조립: p2 생성 파일 그대로 + p1 생성 cnn.c 를 접두어 p1_ 로 + fe_kat.h + fe_measure_main.c + log2_q8.c
  6. scripts/fw_build.sh 로 빌드

청크 (판단 1회 = 새 4,096 샘플 = 접기 64 위치):
  · 프레임 홉 16 위치 (fe, b64): p1 입력 64 위치 → 유효 출력 57 → 홉당 4 프레임 (16f .. 16f+7, f = 0..3)
  · 프레임 홉 8 위치 (f31, b64f31): p1 입력 71 위치 = 직전 8 위치 + 새 63 위치 (위치 64c−8 .. 64c+62) → 유효 출력 64 (64c−8 .. 64c+55)
    → 홉당 8 프레임, 청크 c 의 첫 프레임 번호 8c − 1 (학습 창의 8 위치 격자와 정확히 같다; 새 청크의 마지막 위치는 다음 홉으로 넘긴다 = 4 ms 지연).
    KAT 의 청크 0 은 앞 8 위치를 0 으로 채우고 프레임 −1 은 버린다. docs/results/fe-curve-design.md 5절 결과의 사용 4.

사용:
    OPENBLAS_NUM_THREADS=1 ~/ai8x-training/venv/bin/python tools/fe_board_prep.py [--variant b64|f31|b64f31] [--no-build] [--v2 | --fe8]
  --v2 (2026-10-24): chain34/35 의 v2 체크포인트(QAT 문턱 연결 끊음)를 같은 절차로 — 실행 접두 `safesound-v4d-fe-v2[-<변형>]`,
  펌웨어 `firmware/npu/safesound_fe_v2[_<변형>]_v4d`, 합성 접두 `safesound_fe_v2…`. 논문용 프로젝트는 건드리지 않는다.
  --fe8 (2026-10-25, docs/results/npu-readback-design.md 7절): chain34b 의 fe8 체크포인트 (`safesound-v4d-fe8[-<변형>]`, p1 8비트 + Abs + NPU 합산).
  p1 이 **2층**이 된다 — 층 A = Conv1d(64→2n, k=8) + Abs, 8비트 (출력 시프트 = quantize.py 가 정한 값 − 문턱) /
  층 B = Conv1d(2n→n, k=8), 가중치 64/128 고정, wide int32 (E′·2^14 = 64·Σa, 정수 정확; 체크포인트에 `sum.*` 로 덧붙인다, YAML
  `synthesis/safesound-fe8-p1[-b64]-hwc.yaml` 은 이 스크립트가 만든다). CPU 는 층 B 출력의 **프레임 시작 위치 한 워드**(n×F′ 개)만 읽어
  로그를 취한다 (LO_Q8 − 256). 호스트 정수 경로: a = sat(floor(y_int·2^(s−7) + 0.5)) → |·|·127 상한 (디바이스 순서) → E′ = 64·Σ_8위치 (a_cos + a_sin)
  → 학습 그래프 `fe()`·`frontend()`·`intlog()` 와 비트 대조. 펌웨어 `firmware/npu/safesound_fe8[_<변형>]_v4d`, 측정 main 은 FE_P1_BITS8 분기.
  보드에 올려 재면 ②′ 와 같은 KAT 셋(p1 = 층 B 기대 출력, CPU 특징, p2) 과 구간 지연이 나온다 — **지연·에너지는 보드 실측 전까지 전부 추정**이다.
"""
import argparse
import glob
import importlib.util
import os
import re
import shutil
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = os.path.expanduser("~/ai8x-training")
S = os.path.expanduser("~/ai8x-synthesis")
SYN = os.path.join(REPO, "data", "synth")
OUT = os.path.join(SYN, "out")
LOGD = os.path.join(REPO, "data", "logs-local")
P2_YAML = os.path.join(REPO, "synthesis", "safesound-fe-p2-hwc.yaml")
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, T)
from log2_q8 import log2_q8, to_int8                # noqa: E402

HOP = 64                                            # 판단 1회의 새 위치 수 (4,096 샘플 / 접기 64)
NCH = 4                                             # 창 256 위치 = 청크 4개
SUM_W = 64                                          # fe8 층 B 의 고정 가중치 (int8 64 = 0.5) — models/ai85net-safesound-fe.py E8_SCALE

FE8_P1_YAML = """---
# 후속 연구 fe8 의 1패스 (2층) — tools/fe_board_prep.py --fe8 가 만들었다 (docs/results/npu-readback-design.md 7.2, 합성기 수용 검사 1절 A-sum).
#   models/ai85net-safesound-fe.py `ai85safesoundfe8{arch_suf}` (p1_bits8=True) 의 p1 부분.
#   입력  (64, L) int8 — ②′ 와 같은 판단 1회 청크 (L = 64 / 71)
#   층 A  Conv1d(64 → {n_ch}, k=8) + Abs → 8비트 (출력 시프트는 체크포인트 fe.output_shift = quantize.py 값 − QAT 문턱)
#   층 B  Conv1d({n_ch} → {n_band}, k=8), 가중치 64/128 고정 (cos·sin 같은 대역 8탭) → wide int32 = 64·Σa.  CPU 는 프레임 시작 위치만 읽는다.
#   2패스(가 의 2D CNN)와 가중치 메모리를 나눠 쓴다 — 합성 시 --weight-start 로 2패스 뒤에 둔다.
arch: ai85safesoundfe8_p1{arch_suf}
dataset: SafeSoundFEP1

layers:
  # ── 층 A (fe): Conv1d k=8, 64 → {n_ch}, Abs, 8비트, 패딩 없음
  - pad: 0
    activate: Abs
    in_offset: 0x0000
    out_offset: 0x2000
    processors: 0xffffffffffffffff
    data_format: HWC
    operation: Conv1d
    kernel_size: 8
  # ── 층 B (sum): Conv1d k=8, {n_ch} → {n_band}, 가중치 0.5 고정, wide int32, 활성화 없음
  - pad: 0
    activate: None
    in_offset: 0x2000
    out_offset: 0x4000
    processors: 0xffffffffffffffff
    operation: Conv1d
    kernel_size: 8
    output_width: 32
    output_processors: 0x{out_mask:016x}
    output: true
"""


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run(cmd, cwd, log):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    with open(log, "w", encoding="utf-8") as f:
        f.write(r.stdout + r.stderr)
    if r.returncode:
        print(f"[에러] exit {r.returncode}: {' '.join(cmd)}\n{(r.stdout + r.stderr)[-2000:]}")
        sys.exit(1)
    return r.stdout + r.stderr


def parse_macro_words(path, macro):
    txt = open(path, encoding="utf-8").read()
    m = re.search(macro + r"\s*\{(.*?)\}", txt, re.S)
    assert m, (path, macro)
    return [int(w, 16) for w in re.findall(r"0x[0-9a-fA-F]+", m.group(1))]


def parse_sample_output(path, macro="SAMPLE_OUTPUT"):
    """[(addr, mask, [words])] — 합성기 check_output 형식."""
    w = parse_macro_words(path, macro)
    out, i = [], 0
    while w[i] != 0:
        addr, mask, n = w[i], w[i + 1], w[i + 2]
        out.append((addr, mask, w[i + 3:i + 3 + n]))
        i += 3 + n
    return out


def u32(v):
    return int(v) & 0xFFFFFFFF


def abs8(y, s):
    """디바이스의 8비트 출력 단계 (tools/npu_readback_sim.abs8, 배율 1): y int64 → floor(y·2^(s−7) + 0.5) → 포화 ±128 → |·| → 127 상한.
    izer/simulate.py 와 같은 순서 (반올림 → 포화 → Abs). fe8 학습 그래프(frontend)도 이 순서다."""
    if s < 7:
        r = np.floor_divide(y + (1 << (6 - s)), 1 << (7 - s))
    elif s == 7:
        r = y
    else:
        r = y << (s - 7)
    r = np.clip(r, -128, 127)
    return np.minimum(np.abs(r), 127)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["fe", "b64", "f31", "b64f31"], default="fe")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--v2", action="store_true",
                    help="chain34/35 의 v2 체크포인트 (QAT 문턱 연결 끊음, safesound-v4d-fe-v2[-<변형>]) — 별도 프로젝트 safesound_fe_v2[_<변형>]_v4d 로 (논문 펌웨어는 그대로)")
    ap.add_argument("--fe8", action="store_true",
                    help="chain34b 의 fe8 체크포인트 (p1 8비트 + Abs + NPU 합산, safesound-v4d-fe8[-<변형>]) — p1 2층 합성, 프로젝트 safesound_fe8[_<변형>]_v4d (모듈 설명)")
    a = ap.parse_args()
    assert not (a.v2 and a.fe8), "--v2 와 --fe8 은 함께 쓰지 않는다 (fe8 은 이미 연결을 끊고 학습했다)"
    import torch
    import ai8x
    ai8x.set_device(85, False, False)
    fe = load_mod(os.path.join(REPO, "models", "ai85net-safesound-fe.py"), "fe_mod")
    os.makedirs(LOGD, exist_ok=True)

    V = fe.VARIANTS[a.variant]
    n_band, n_frame, ppf, pool, lo_q8 = V["n_band"], V["n_frame"], V["pos_per_frame"], V["pool_pos"], V["lo_q8"]
    n_ch = 2 * n_band
    bits8 = a.fe8
    lo_eff = lo_q8 + (fe.LO8_SHIFT_Q8 if bits8 else 0)                                         # fe8: 로그 창을 log2 −1 (E′ = E/2)
    vsuf = "" if a.variant == "fe" else f"_{a.variant}"
    proj = ("safesound_fe8" if bits8 else "safesound_fe" + ("_v2" if a.v2 else "")) + vsuf      # 프로젝트 이름: safesound_fe / _fe_v2 / _fe8 [+ _변형]
    RUN = "safesound-v4d-" + ("fe8" if bits8 else "fe" + ("-v2" if a.v2 else "")) + vsuf.replace("_", "-")
    P1, P2 = f"{proj}_p1_v4d", f"{proj}_p2_v4d"
    FW = os.path.join(REPO, "firmware", "npu", f"{proj}_v4d")
    NET = f"{proj}_v4d"
    cfg = "wave_" + ("fe8" if bits8 else "fe" + ("_v2" if a.v2 else "")) + vsuf
    arch_suf = "_b64" if n_band == 64 else ""
    p1_arch = ("ai85safesoundfe8_p1" if bits8 else "ai85safesoundfe_p1") + arch_suf
    if bits8:
        P1_YAML = os.path.join(REPO, "synthesis", f"safesound-fe8-p1{arch_suf.replace('_', '-')}-hwc.yaml")
    else:
        P1_YAML = os.path.join(REPO, "synthesis", "safesound-fe-p1-hwc.yaml" if n_band == 32 else "safesound-fe-p1-b64-hwc.yaml")
    log_pfx = os.path.join(LOGD, f"synth_{proj[len('safesound_'):]}_v4d")
    # 청크 규약 (모듈 docstring)
    if ppf == 16:
        PRE, IN_POS, F0 = 0, HOP, 0
    elif ppf == 8:
        PRE, IN_POS, F0 = 8, HOP + 7, -1
    else:
        sys.exit(f"지원하지 않는 pos_per_frame {ppf}")
    OUT_POS = IN_POS - fe.KERNEL + 1                                                        # 층 A 유효 출력: 57 / 64
    FPH = OUT_POS // ppf if ppf == 8 else (OUT_POS - pool) // ppf + 1                        # 홉당 프레임: 8 / 4
    assert FPH * NCH + (F0 if F0 < 0 else 0) == n_frame, (FPH, n_frame)
    # CPU 가 읽는 p1 출력: ②′ 는 층 A 그대로 (2n 채널 × OUT_POS 위치), fe8 은 층 B (n 채널 × OUT_POS−7 위치)
    OUT_POS_RD = OUT_POS - fe.KERNEL + 1 if bits8 else OUT_POS                              # fe8: 50 / 57
    if bits8:
        assert ppf * (FPH - 1) <= OUT_POS_RD - 1, "마지막 프레임의 합 위치가 층 B 출력 밖이다"
    print(f"0) 변형 {a.variant}{' + fe8 (p1 8비트·NPU 합산)' if bits8 else (' v2' if a.v2 else '')}: 대역 {n_band} × 프레임 {n_frame} (홉 {ppf} 위치, 적분 {pool}), "
          f"LO_Q8 {lo_eff}{f' (= {lo_q8} − 256)' if bits8 else ''}; p1 {p1_arch} 입력 {IN_POS} 위치 → 층 A 출력 {OUT_POS}"
          f"{f' → 층 B 출력 {OUT_POS_RD}' if bits8 else ''}, 홉당 {FPH} 프레임 (첫 프레임 오프셋 {F0}); 실행 {RUN}, 펌웨어 {os.path.relpath(FW, REPO)}")

    # ── 1. 체크포인트 가르기 ─────────────────────────────────────────────────
    cands = sorted(glob.glob(os.path.join(LOGD, f"{RUN}___*", f"{RUN}_qat_best.pth.tar")))
    assert cands, f"{RUN} 의 시드 1 QAT 체크포인트가 없다 ({LOGD}/{RUN}___*/)"
    ck_path = cands[-1]
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    sd_p1 = {k: v for k, v in sd.items() if k.startswith("fe.")}
    sd_p2 = {k[len("cls."):]: v for k, v in sd.items() if k.startswith("cls.")}
    assert len(sd_p1) + len(sd_p2) == len(sd), "fe./cls. 밖의 키가 있다"
    p1_f, p1_q = os.path.join(SYN, f"{RUN}-p1.pth.tar"), os.path.join(SYN, f"{RUN}-p1-q8.pth.tar")
    p2_f, p2_q = os.path.join(SYN, f"{RUN}-p2.pth.tar"), os.path.join(SYN, f"{RUN}-p2-q8.pth.tar")
    torch.save({"epoch": ck["epoch"], "state_dict": sd_p1, "arch": p1_arch, "extras": {"from": os.path.relpath(ck_path, REPO)}}, p1_f)
    torch.save({"epoch": ck["epoch"], "state_dict": sd_p2, "arch": "ai85safesoundmelnet_p4", "extras": {"from": os.path.relpath(ck_path, REPO)}}, p2_f)
    print(f"1) 체크포인트 {os.path.relpath(ck_path, REPO)} (epoch {ck['epoch']}) → p1 {len(sd_p1)}키 / p2 {len(sd_p2)}키"
          + (f"; fe 문턱 {float(sd['fe.activation_threshold']):+.0f}, cls.conv1 문턱 {float(sd['cls.conv1.activation_threshold']):+.0f}" if bits8 or a.v2 else ""))

    # ── 2. 샘플 창 ───────────────────────────────────────────────────────────
    win = np.load(os.path.join(SYN, "sample_safesound-v4d-wave.npy")).astype(np.int64)     # (128, 128): x[r][c] = s[128c + r]
    assert win.shape == (128, 128) and win.min() >= -128 and win.max() <= 127
    s = win.T.reshape(-1)                                                                   # 16,384 샘플
    X = s.reshape(fe.NPOS, fe.FOLD).T                                                       # X[r][p] = s[64p + r]  (64, 256)
    Xpad = np.concatenate([np.zeros((fe.FOLD, PRE), dtype=np.int64), X], axis=1)            # 앞 PRE 위치는 0 (창 밖)
    chunks = [Xpad[:, c * HOP:c * HOP + IN_POS] for c in range(NCH)]                        # (64, IN_POS) ×4
    assert all(ch.shape == (fe.FOLD, IN_POS) for ch in chunks)

    # ── 3. 양자화 + 정수 프론트엔드 ─────────────────────────────────────────
    py = os.path.join(S, "venv", "bin", "python")
    run([py, "quantize.py", p1_f, p1_q, "--device", "MAX78000"], S, f"{log_pfx}-quantize-p1.out")
    run([py, "quantize.py", p2_f, p2_q, "--device", "MAX78000"], S, f"{log_pfx}-quantize-p2.out")
    q8_p1 = torch.load(p1_q, map_location="cpu", weights_only=False)
    Wq = q8_p1["state_dict"]["fe.op.weight"].numpy().astype(np.int64)                        # (n_ch, 64, 8) int8 값
    assert Wq.shape == (n_ch, fe.FOLD, fe.KERNEL) and Wq.min() >= -128 and Wq.max() <= 127 and np.all(Wq == np.round(Wq))
    s_out = int(q8_p1["state_dict"]["fe.output_shift"].item())                             # 디바이스 출력 시프트 (8비트 층에서만 뜻이 있다)
    # y[o][p] = Σ_j Σ_r W[o][r][j] X[r][p+j]  — 정확한 정수 (int64), 창 전체 유효 위치 0..248
    y = np.zeros((n_ch, fe.NVALID), dtype=np.int64)
    for j in range(fe.KERNEL):
        y += Wq[:, :, j] @ X[:, j:j + fe.NVALID]
    assert np.abs(y).max() < 2 ** 31
    idx = (np.arange(n_frame) * ppf)[:, None] + np.arange(pool)[None, :]
    if bits8:
        a8 = abs8(y, s_out)                                                                 # 층 A 디바이스 출력 |·| (n_ch, 249), 0..127
        mag = a8[0::2] + a8[1::2]                                                           # (n_band, 249)
        E = SUM_W * mag[:, idx].sum(-1)                                                     # 층 B 정수 출력 = E′·2^14 (n_band, n_frame) ≤ 64·16·127
    else:
        a8 = None
        mag = np.abs(y[0::2]) + np.abs(y[1::2])                                             # (n_band, 249)
        E = mag[:, idx].sum(-1)                                                             # (n_band, n_frame) 정수
    assert E.max() <= 0xFFFFFFFF
    q = to_int8(log2_q8(E.ravel()), lo_eff, fe.SPAN_Q8).astype(np.int64).reshape(n_band, n_frame)
    # 학습 그래프와 대조 (QAT 모드, 같은 체크포인트)
    E_mod = load_mod(os.path.join(REPO, "tools", "eval_confusion.py"), "eval_confusion")
    model = E_mod.load_model(ck_path, 5, T, simulate=False, config=cfg)
    assert model.lo_q8 == lo_eff, (model.lo_q8, lo_eff)
    assert bool(model.p1_bits8) == bits8
    with torch.no_grad():
        x = torch.tensor(win, dtype=torch.float32)[None] / 128.0
        e_t = model.frontend(x)
        q_t = model.intlog(e_t)[0].numpy().astype(np.int64)
        if bits8:
            x2 = x.transpose(1, 2).reshape(1, -1).reshape(1, fe.NPOS, fe.FOLD).transpose(1, 2)
            a_t = (model.fe(x2).abs().clamp(max=127.0 / 128.0) * 128).round()[0].numpy().astype(np.int64)   # 그래프의 8비트 단계 (디바이스 순서)
            w = model.fe.op.weight.detach()
            s_train = int((model.fe.calc_out_shift(torch.flatten(w), model.fe.output_shift.detach()) - model.fe.activation_threshold).clamp(-15., 15.))
    e_scaled = e_t[0].numpy().astype(np.float64) * (1 << fe.SCALE_SH)
    d_q = int((q_t != q).sum())
    if bits8:
        d_a = int((a_t != a8).sum())
        print(f"3) fe8 8비트 단계: 디바이스 출력 시프트 {s_out:+d} (quantize.py) / 학습 그래프 {s_train:+d}; 학습 그래프 a 와 정수 경로 불일치 {d_a} / {a8.size}; "
              f"포화(127) {100 * (a8 == 127).mean():.2f} %, 0 {100 * (a8 == 0).mean():.2f} % (창 전체 249 위치)")
        assert s_train == s_out, "quantize.py 의 출력 시프트가 학습 그래프와 다르다"
        assert d_a == 0, "8비트 단계가 학습 그래프와 다르다 — 반올림·포화·|·| 순서 또는 시프트를 확인할 것"
    print(f"3) 정수 프론트엔드: E 범위 [{E.min()}, {E.max()}], q 범위 [{q.min()}, {q.max()}]; "
          f"학습 그래프 q 와 불일치 {d_q} / {q.size}  (E·2^14 − 정수 E 최대 |차| {np.abs(e_scaled - E).max():.3f})")
    if d_q:
        bad = np.argwhere(q_t != q)[:5]
        for b, f in bad:
            print(f"   (b={b}, f={f}) 정수 E {E[b, f]}  float E·2^14 {e_scaled[b, f]:.3f}  q 정수 {q[b, f]}  q 학습 {q_t[b, f]}")
    assert d_q <= (0 if bits8 else 2), "정수 경로와 학습 그래프가 다르다 — 스케일 규약을 확인할 것"

    # ── 4. 합성 (p2 먼저, p1 은 p2 커널 뒤) ──────────────────────────────────
    np.save(os.path.join(S, "tests", "sample_safesoundfep2.npy"), q[None].astype(np.int64))
    np.save(os.path.join(S, "tests", "sample_safesoundfep1.npy"), chunks[NCH - 1].astype(np.int64))
    np.save(os.path.join(SYN, f"sample_{RUN}-p2.npy"), q[None].astype(np.int64))
    np.save(os.path.join(SYN, f"sample_{RUN}-p1-chunk{NCH - 1}.npy"), chunks[NCH - 1].astype(np.int64))
    if not os.path.isfile(P2_YAML):
        src = open(os.path.join(REPO, "synthesis", "safesound-melu1000-hwc.yaml"), encoding="utf-8").read()
        src = src.replace("dataset: SafeSoundMelU1000", "dataset: SafeSoundFEP2")
        hdr = ("# 구성 ②′ 2패스 — 가 (u1000) 의 2D CNN 그대로. 입력 (1, 32, 16) int8 은 CPU 정수 로그단이 만든다.\n"
               "#   safesound-melu1000-hwc.yaml 과 층 설정이 같고 dataset 이름만 다르다 (샘플 파일 sample_safesoundfep2.npy).\n"
               "#   tools/fe_board_prep.py 가 만들었다.\n")
        open(P2_YAML, "w", encoding="utf-8").write(hdr + src)
    if bits8:
        # 층 B (합산) 를 양자화된 p1 체크포인트에 덧붙인다 — 가중치 64 (= 0.5) 고정, 시프트 0. 키 순서 = 층 순서 (fe → sum).
        Wsum = np.zeros((n_band, n_ch, fe.KERNEL), dtype=np.float32)
        for b in range(n_band):
            Wsum[b, 2 * b, :] = SUM_W
            Wsum[b, 2 * b + 1, :] = SUM_W
        q8_p1["state_dict"]["sum.op.weight"] = torch.from_numpy(Wsum)
        q8_p1["state_dict"]["sum.output_shift"] = torch.tensor([0.0])
        torch.save(q8_p1, p1_q)
        if not os.path.isfile(P1_YAML):
            open(P1_YAML, "w", encoding="utf-8").write(FE8_P1_YAML.format(arch_suf=arch_suf, n_ch=n_ch, n_band=n_band, out_mask=(1 << n_band) - 1))
            print(f"   YAML 생성 {os.path.relpath(P1_YAML, REPO)}")
    common = ["--verbose", "--test-dir", OUT, "--device", "MAX78000", "--compact-data", "--mexpress", "--timer", "0",
              "--display-checkpoint", "--overwrite"]
    run([py, "ai8xize.py", "--prefix", P2, "--checkpoint-file", p2_q, "--config-file", P2_YAML, "--softmax"] + common,
        S, f"{log_pfx}-ai8xize-p2.out")
    log2 = open(os.path.join(OUT, P2, "log.txt"), encoding="utf-8").read()
    off = [int(v) for v in re.search(r"Kernel offset\s*=\s*\[([^\]]*)\]", log2).group(1).split(",")]
    ln = [int(v) for v in re.search(r"Kernel length\s*=\s*\[([^\]]*)\]", log2).group(1).split(",")]
    end = max(o + n for o, n in zip(off, ln))
    ws = (end + 3) // 4 * 4
    run([py, "ai8xize.py", "--prefix", P1, "--checkpoint-file", p1_q, "--config-file", P1_YAML, "--weight-start", str(ws)] + common,
        S, f"{log_pfx}-ai8xize-p1.out")
    log1 = open(os.path.join(OUT, P1, "log.txt"), encoding="utf-8").read()
    cyc = {k: int(re.search(r"^Total\s+([\d,]+) cycles", l, re.M).group(1).replace(",", "")) for k, l in (("p1", log1), ("p2", log2))}
    wb = {k: int(re.search(r"Weight memory:\s+([\d,]+) bytes", open(os.path.join(OUT, p, "cnn.h")).read()).group(1).replace(",", ""))
          for k, p in (("p1", P1), ("p2", P2))}
    off1 = [int(v) for v in re.search(r"Kernel offset\s*=\s*\[([^\]]*)\]", log1).group(1).split(",")]
    ln1 = [int(v) for v in re.search(r"Kernel length\s*=\s*\[([^\]]*)\]", log1).group(1).split(",")]
    assert min(off1) >= end, "p1 커널이 p2 와 겹친다"
    assert len(off1) == (2 if bits8 else 1), ("p1 층 수", off1)
    print(f"4) 합성: p2 커널 {off}+{ln} → 끝 {end}; p1 --weight-start {ws} → {off1}+{ln1}. "
          f"사이클 p1 {cyc['p1']:,} / p2 {cyc['p2']:,}; 가중치 p1 {wb['p1']:,} + p2 {wb['p2']:,} = {wb['p1'] + wb['p2']:,} B")

    # p1 기대 출력(청크 3) == 정수 경로 ?
    #   ②′: 층 A 출력 그대로 — 블록 16개(프로세서 4개씩), 위치 p 마다 STRIDE 워드:
    #     64 채널: 워드 4p + k → 채널 4g + k.  128 채널(프로세서당 2채널): 워드 8p + k → k<4 채널 4g+k, k≥4 채널 64+4g+k−4 (2026-10-19 합성기 출력으로 확인)
    #   fe8: 층 B 출력 (n 채널, 프로세서당 1채널) — 블록 n/4 개, 위치당 4 워드, 채널 4g + k. 값 = 64·Σ_{j<8} (a[2b][p+j] + a[2b+1][p+j]).
    so1 = parse_sample_output(os.path.join(OUT, P1, "sampleoutput.h"))
    c3 = NCH - 1
    p_lo = c3 * HOP - PRE
    if bits8:
        a3 = a8[:, p_lo:p_lo + OUT_POS]                                                     # 청크 3 의 층 A 출력 (n_ch, OUT_POS)
        y_rd = np.zeros((n_band, OUT_POS_RD), dtype=np.int64)
        for j in range(fe.KERNEL):
            y_rd += a3[0::2, j:j + OUT_POS_RD] + a3[1::2, j:j + OUT_POS_RD]
        y_rd *= SUM_W                                                                       # (n_band, OUT_POS_RD) 층 B 정수 출력
        for f in range(FPH):                                                                # 프레임 합 == 층 B 의 프레임 시작 위치 값
            dst = c3 * FPH + F0 + f
            if 0 <= dst < n_frame:
                assert np.array_equal(E[:, dst], y_rd[:, ppf * f]), (f, dst)
        n_grp, stride, rd_ch = n_band // 4, 4, n_band
        assert n_band % 4 == 0 and n_band <= 64

        def ch_of(g, k):
            return 4 * g + k
    else:
        y_rd = y[:, p_lo:p_lo + OUT_POS]                                                    # 청크 3 의 유효 위치 (n_ch, OUT_POS)
        n_grp, stride, rd_ch = 16, n_ch // 16, n_ch                                         # 4 / 8

        def ch_of(g, k):
            return 4 * g + k if k < 4 else 64 + 4 * g + (k - 4)
    assert y_rd.shape == (rd_ch, OUT_POS_RD)
    assert len(so1) == n_grp, (len(so1), n_grp)
    bad1 = 0
    for g, (addr, mask, words) in enumerate(so1):
        assert len(words) == stride * OUT_POS_RD, (g, len(words), stride, OUT_POS_RD)
        exp = [u32(y_rd[ch_of(g, i % stride), i // stride]) for i in range(len(words))]
        bad1 += sum(a_ != b for a_, b in zip(words, exp))
    print(f"   p1 기대 출력(합성기{', 층 B' if bits8 else ''}) vs 정수 경로: 불일치 {bad1} / {sum(len(w) for _, _, w in so1)} "
          f"({'OK' if bad1 == 0 else 'FAIL'}); 블록 {n_grp}개 × 위치당 {stride} 워드 × {OUT_POS_RD} 위치, 주소 {[hex(a_) for a_, _, _ in so1[:4]]} … {hex(so1[-1][0])}")
    assert bad1 == 0
    out_addr = [a_ for a_, _, _ in so1]

    # 청크 입력 워드 포장 == 생성 sampledata.h ?  (그룹 g 의 워드 p = 채널 4g..4g+3 의 바이트, LE)
    def pack_chunk(Xc):
        words = np.zeros((16, IN_POS), dtype=np.uint32)
        for g in range(16):
            for i in range(4):
                words[g] |= (Xc[4 * g + i] & 0xFF).astype(np.uint32) << (8 * i)
        return words
    main1 = open(os.path.join(OUT, P1, "main.c"), encoding="utf-8").read()
    in_addr = [int(m.group(1), 16) for m in re.finditer(r"memcpy32\(\(uint32_t \*\) (0x[0-9a-f]+), input_(\d+), (\d+)\)", main1)]
    in_words = {int(m.group(2)): int(m.group(3)) for m in re.finditer(r"memcpy32\(\(uint32_t \*\) (0x[0-9a-f]+), input_(\d+), (\d+)\)", main1)}
    assert len(in_addr) == 16 and set(in_words.values()) == {IN_POS}, (in_addr, in_words)
    sd_path = os.path.join(OUT, P1, "sampledata.h")
    pk3 = pack_chunk(chunks[c3])
    bad_in = sum(int(np.any(np.array(parse_macro_words(sd_path, f"SAMPLE_INPUT_{4 * g}"), dtype=np.uint32) != pk3[g])) for g in range(16))
    print(f"   청크 입력 포장 vs 생성 sampledata.h: 다른 그룹 {bad_in} / 16 ({'OK' if bad_in == 0 else 'FAIL'})")
    assert bad_in == 0
    out_offsets = [int(v, 16) for v in re.findall(r"0x[0-9a-f]+", re.search(r"Output offset\s*=\s*\[([^\]]*)\]", log1).group(1))]
    out_offset = out_offsets[-1]                                                            # CPU 가 읽는(마지막) 층의 출력 오프셋
    assert out_addr == [a_ + out_offset for a_ in in_addr[:n_grp]], (out_addr[:2], in_addr[:2], hex(out_offset))

    # p2: 입력 적재 주소·워드 수, 기대 로짓, simulate 모드 PyTorch 와 대조
    main2 = open(os.path.join(OUT, P2, "main.c"), encoding="utf-8").read()
    m2 = re.search(r"memcpy32\(\(uint32_t \*\) (0x[0-9a-f]+), input_0, (\d+)\)", main2)
    p2_in_addr, p2_in_words = int(m2.group(1), 16), int(m2.group(2))
    assert p2_in_words == n_band * n_frame, (p2_in_words, n_band * n_frame)
    sd2 = np.array(parse_macro_words(os.path.join(OUT, P2, "sampledata.h"), "SAMPLE_INPUT_0"), dtype=np.uint32)
    assert np.array_equal(sd2, (q.ravel() & 0xFF).astype(np.uint32)), "p2 입력은 워드당 1픽셀(하위 바이트), 행 우선이어야 한다"
    so2 = parse_sample_output(os.path.join(OUT, P2, "sampleoutput.h"))
    logits = [int(np.int32(np.uint32(w))) for _, _, ws_ in so2 for w in ws_]
    model_sim = E_mod.load_model(ck_path, 5, T, simulate=True, config=cfg)
    with torch.no_grad():
        lg_sim = model_sim.cls(torch.tensor(q, dtype=torch.float32)[None, None])[0].numpy()
    print(f"   p2 기대 로짓(합성기) {logits}  simulate 모드 PyTorch {lg_sim.astype(np.int64).tolist()} "
          f"({'일치' if np.array_equal(np.array(logits), lg_sim.astype(np.int64)) else '다름 — 참고'}); argmax {int(np.argmax(logits))}")
    # 두 네트워크가 **같은 층 슬롯**에 쓰는 레지스터(사분면 0 기준 오프셋)의 합집합 — 전환 때 먼저 0 으로 지운다.
    #   생성 코드는 0 이 아닌 레지스터만 쓰므로, 한쪽만 쓰는 레지스터(p1 의 1D·후처리, p2 의 Columns·TRAM)가 다른 쪽에 남는다 (2026-10-17 보드에서 p2 KAT 실패로 확인).
    #   레지스터 주소 = 사분면 기준 + 0x10 + 종류·0x80 + 층·4 → 층 L 은 ((addr − base − 0x10) % 0x80) // 4 == L.
    #   p1 이 쓰는 슬롯 수 = p1 층 수 (②′ 1 → 층 0 만, 옛 FE_L0_REGS 와 같다; fe8 2 → 층 0·1). p2 만 쓰는 슬롯(L ≥ p1 층 수)은 p1 이 건드리지 않아 그대로 남아도 된다.
    cnn1_c = open(os.path.join(OUT, P1, "cnn.c")).read()
    cnn2_c = open(os.path.join(OUT, P2, "cnn.c")).read()
    p2_layers = int(re.search(r"0x50100008\) = (0x[0-9a-f]+); // Layer count", cnn2_c).group(1), 16)
    m1 = re.search(r"0x50100008\) = (0x[0-9a-f]+); // Layer count", cnn1_c)
    p1_layers = int(m1.group(1), 16) if m1 else 0                                           # 레지스터 값 = 층 수 − 1 (1층이면 생성 코드가 쓰지 않는다 → 0)
    n_shared = p1_layers + 1
    assert n_shared == (2 if bits8 else 1), (n_shared, bits8)

    def shared_regs(cnn_c):
        body = re.search(r"int cnn_configure\(void\)\s*\{(.*?)\n\}", cnn_c, re.S).group(1)
        regs = set()
        for a_ in re.findall(r"\*\(\(volatile uint32_t \*\) (0x50[159d]00[0-9a-f]{3})\)", body):
            off_ = int(a_, 16) & 0xFFFF
            if ((off_ - 0x10) % 0x80) // 4 < n_shared:
                regs.add(off_)
        return regs
    l0 = sorted(shared_regs(cnn1_c) | shared_regs(cnn2_c))
    print(f"   공유 층 슬롯 {n_shared}개의 레지스터 합집합 {len(l0)}개: {[hex(o) for o in l0]}")

    # ── 5. 펌웨어 프로젝트 조립 ──────────────────────────────────────────────
    os.makedirs(FW, exist_ok=True)
    for f in ("cnn.c", "cnn.h", "weights.h", "softmax.c", "sampleoutput.h", "Makefile"):
        shutil.copy(os.path.join(OUT, P2, f), os.path.join(FW, f))
    with open(os.path.join(FW, "cnn.h"), "a", encoding="utf-8") as f:
        f.write("\n// ── tools/fe_board_prep.py 가 덧붙임 (sync_from_synth.sh 와 같다) ──\n"
                "#ifdef MEASURE_BUILD\n#undef CNN_START\n#undef CNN_COMPLETE\n#undef SYS_START\n#undef SYS_COMPLETE\n"
                "#undef CNN_INFERENCE_TIMER\n#define CNN_START\n#define CNN_COMPLETE\n#define SYS_START\n#define SYS_COMPLETE\n#endif\n")
    # p1 생성 cnn.c → cnn_p1.c (접두어 p1_; ISR·memcpy32 는 p2 쪽 것을 쓴다)
    c1 = cnn1_c
    c1 = re.sub(r"\bcnn_(enable|disable|init|configure|load_weights|verify_weights|load_bias|start|stop|continue|unload|boost_enable|boost_disable)\b",
                r"p1_cnn_\1", c1)
    c1 = c1.replace("void CNN_ISR(void)", "void p1_CNN_ISR_unused(void)").replace("MXC_NVIC_SetVector(CNN_IRQn, CNN_ISR)", "MXC_NVIC_SetVector(CNN_IRQn, p1_CNN_ISR_unused)")
    c1 = re.sub(r"\bmemcpy32(_const)?\b", r"p1_memcpy32\1", c1)
    c1 = c1.replace('#include "weights.h"', '#include "weights_p1.h"')
    c1 = ("// @generated — tools/fe_board_prep.py 가 data/synth/out/%s/cnn.c 에 접두어 p1_ 를 붙였다. 손으로 고치지 말 것.\n" % P1) + c1
    assert "p1_cnn_configure" in c1 and '#include "weights_p1.h"' in c1
    open(os.path.join(FW, "cnn_p1.c"), "w", encoding="utf-8").write(c1)
    shutil.copy(os.path.join(OUT, P1, "weights.h"), os.path.join(FW, "weights_p1.h"))
    so1_txt = open(os.path.join(OUT, P1, "sampleoutput.h"), encoding="utf-8").read().replace("SAMPLE_OUTPUT", "P1_SAMPLE_OUTPUT")
    open(os.path.join(FW, "sampleoutput_p1.h"), "w", encoding="utf-8").write(so1_txt)
    for f in ("fe_measure_main.c", "log2_q8.c", "log2_q8_lut.inc"):
        shutil.copy(os.path.join(REPO, "firmware", "common", f), os.path.join(FW, "main.c" if f == "fe_measure_main.c" else f))
    # fe_kat.h
    def carr(name, arr, ctype, per=8):
        flat = list(arr)
        rows = [", ".join(flat[i:i + per]) for i in range(0, len(flat), per)]
        return f"static const {ctype} {name}[{len(flat)}] = {{\n  " + ",\n  ".join(rows) + "\n};\n"
    lines = ["// @generated — tools/fe_board_prep.py. 손으로 고치지 말 것.",
             f"// 변형 {a.variant}{' + fe8' if bits8 else (' v2' if a.v2 else '')}: 대역 {n_band} × 프레임 {n_frame}, p1 입력 {IN_POS} 위치 (직전 {PRE} + 새 {IN_POS - PRE}) → "
             f"층 A 출력 {OUT_POS}{f' → 층 B 출력 {OUT_POS_RD} (CPU 가 읽는 것)' if bits8 else ''}, 홉당 {FPH} 프레임.",
             f"// 창: data/synth/sample_safesound-v4d-wave.npy (시험셋, ④ 합성과 같은 창). 체크포인트 {os.path.relpath(ck_path, REPO)}",
             "#include <stdint.h>",
             f"#define FE_NBAND {n_band}", f"#define FE_NFRAME {n_frame}", f"#define FE_POS_PER_FRAME {ppf}",
             f"#define FE_POOL_POS {pool}", f"#define FE_NCHUNK {NCH}", f"#define FE_FRAMES_PER_HOP {FPH}", f"#define FE_FRAME0_OFFSET ({F0})",
             f"#define FE_LO_Q8 {lo_eff}", f"#define FE_SPAN_Q8 {fe.SPAN_Q8}", f"#define FE_SCALE_SH {fe.SCALE_SH}",
             f"#define FE_P1_BITS8 {1 if bits8 else 0}" + ("   // fe8: p1 2층 (Abs 8비트 → 합산 wide), CPU 는 층 B 의 프레임 시작 워드만 읽는다; FE_LO_Q8 에 −256 포함" if bits8 else ""),
             f"#define FE_P1_OUT_SHIFT ({s_out})" + ("   // 층 A 디바이스 출력 시프트 (참고 — CPU 는 쓰지 않는다; 학습이 게인을 정했다)" if bits8 else "   // wide 층: 디바이스가 걸지 않는다 (참고)"),
             f"#define FE_P1_LAYER_COUNT {p1_layers}", f"#define FE_P2_LAYER_COUNT {p2_layers}", f"#define FE_N_SHARED_LAYERS {n_shared}",
             f"#define FE_P1_IN_GROUPS 16", f"#define FE_P1_IN_WORDS {IN_POS}", f"#define FE_P1_OUT_GROUPS {n_grp}", f"#define FE_P1_OUT_CH {rd_ch}",
             f"#define FE_P1_OUT_STRIDE {stride}", f"#define FE_P1_OUT_POS {OUT_POS_RD}",
             f"#define FE_P2_IN_ADDR 0x{p2_in_addr:08x}u", f"#define FE_P2_IN_WORDS {p2_in_words}",
             f"#define FE_P1_WEIGHT_BYTES {wb['p1']}", f"#define FE_P2_WEIGHT_BYTES {wb['p2']}", f"#define FE_P2_WEIGHT_END {end}",
             f"#define FE_N_L0_REGS {len(l0)}",
             f"// 두 네트워크가 공유하는 층 슬롯({n_shared}개)에 쓰는 레지스터 오프셋(사분면 기준) 합집합 — 전환 전에 0 으로 지운다 (fe_measure_main.c switch_to_*)",
             "static const uint32_t FE_L0_REGS[" + str(len(l0)) + "] = { " + ", ".join(f"0x{o:04x}u" for o in l0) + " };",
             "static const uint32_t FE_P1_IN_ADDR[16] = { " + ", ".join(f"0x{a_:08x}u" for a_ in in_addr) + " };",
             f"static const uint32_t FE_P1_OUT_ADDR[{n_grp}] = {{ " + ", ".join(f"0x{a_:08x}u" for a_ in out_addr) + " };",
             f"// 청크 c, 그룹 g(채널 4g..4g+3), 위치 p 의 입력 워드 (LE 바이트 = 연속 샘플 4개)",
             f"static const uint32_t fe_chunk_in[{NCH}][16][{IN_POS}] = {{"]
    for c in range(NCH):
        pk = pack_chunk(chunks[c])
        lines.append("  {")
        for g in range(16):
            ws_ = ", ".join(f"0x{int(w):08x}u" for w in pk[g])
            lines.append(f"    {{ {ws_} }},")
        lines.append("  },")
    lines.append("};")
    lines.append(f"// 호스트 정수 경로의 특징 q[b][f] (행 우선 b·{n_frame}+f) — 창 전체 {n_frame}프레임")
    lines.append(carr("fe_q_expected", [str(int(v)) for v in q.ravel()], "int8_t", 16))
    lines.append(carr("fe_logits_expected", [str(v) for v in logits], "int32_t", 8))
    open(os.path.join(FW, "fe_kat.h"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    open(os.path.join(FW, "project.mk"), "w", encoding="utf-8").write(
        "# @generated — tools/fe_board_prep.py\n# 측정 빌드: LED 마킹·소프트웨어 타이머를 뺀다 (CLAUDE.md 6장 MEASURE_BUILD).\n"
        f"BOARD = FTHR_RevA\nPROJ_CFLAGS += -DMEASURE_BUILD -DNET_NAME=\\\"{NET}\\\"\n")
    print(f"5) 펌웨어 프로젝트 {os.path.relpath(FW, REPO)}: {sorted(os.listdir(FW))}")
    if a.no_build:
        return 0

    # ── 6. 빌드 ─────────────────────────────────────────────────────────────
    r = subprocess.run(["bash", os.path.join(REPO, "scripts", "fw_build.sh"), os.path.relpath(FW, REPO)], cwd=REPO, capture_output=True, text=True)
    txt = r.stdout + r.stderr
    print("6) 빌드:", "\n   ".join(l for l in txt.splitlines() if re.search(r"성공|실패|Flash|SRAM|error|\.elf", l)))
    if r.returncode:
        print(txt[-3000:])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
