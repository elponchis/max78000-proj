#!/usr/bin/env python3
"""구성 ②′ 보드 실측 준비 — 시드 1 체크포인트를 p1·p2 로 갈라 합성하고, 두 네트워크 측정 펌웨어를 조립·빌드한다.

하는 일 (WSL2, ai8x-training venv):
  1. `safesound-v4d-fe` 시드 1 QAT 체크포인트 → p1 (`fe.*`, arch ai85safesoundfe_p1) / p2 (`cls.*`, arch ai85safesoundmelnet_p4)
  2. 샘플 창 = ④ 합성에 쓴 시험셋 창 (`data/synth/sample_safesound-v4d-wave.npy`, (128,128) 접기) — 같은 로더·같은 창
  3. quantize.py 두 번 → int8 가중치로 **정수 프론트엔드**(exact int64 conv → |cos|+|sin| → 8위치 합 → log2_q8 → int8)
     를 호스트에서 계산하고, 학습 그래프(`AI85SafeSoundFE.frontend` + `_IntLog`, QAT 모드)와 비트 대조
  4. ai8xize 두 번 (p2 먼저 → 커널 끝 단위 → p1 은 그 뒤에 `--weight-start`) → p1 기대 출력(청크 3)이 정수 conv 와 같은지,
     청크 입력 워드 포장이 생성 sampledata.h 와 같은지, p2 기대 로짓이 simulate 모드 PyTorch 와 같은지 확인
  5. firmware/npu/safesound_fe_v4d/ 조립: p2 생성 파일 그대로 + p1 생성 cnn.c 를 접두어 p1_ 로 + fe_kat.h + fe_measure_main.c + log2_q8.c
  6. scripts/fw_build.sh 로 빌드

사용:
    OPENBLAS_NUM_THREADS=1 ~/ai8x-training/venv/bin/python tools/fe_board_prep.py
"""
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
RUN = "safesound-v4d-fe"
P1, P2 = "safesound_fe_p1_v4d", "safesound_fe_p2_v4d"
FW = os.path.join(REPO, "firmware", "npu", "safesound_fe_v4d")
P2_YAML = os.path.join(REPO, "synthesis", "safesound-fe-p2-hwc.yaml")
P1_YAML = os.path.join(REPO, "synthesis", "safesound-fe-p1-hwc.yaml")
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, T)
from log2_q8 import log2_q8, to_int8                # noqa: E402


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


def main():
    import torch
    import ai8x
    ai8x.set_device(85, False, False)
    fe = load_mod(os.path.join(REPO, "models", "ai85net-safesound-fe.py"), "fe_mod")
    os.makedirs(LOGD, exist_ok=True)

    # ── 1. 체크포인트 가르기 ─────────────────────────────────────────────────
    ck_path = sorted(glob.glob(os.path.join(LOGD, f"{RUN}___*", f"{RUN}_qat_best.pth.tar")))[-1]
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    sd_p1 = {k: v for k, v in sd.items() if k.startswith("fe.")}
    sd_p2 = {k[len("cls."):]: v for k, v in sd.items() if k.startswith("cls.")}
    assert len(sd_p1) + len(sd_p2) == len(sd), "fe./cls. 밖의 키가 있다"
    p1_f, p1_q = os.path.join(SYN, f"{RUN}-p1.pth.tar"), os.path.join(SYN, f"{RUN}-p1-q8.pth.tar")
    p2_f, p2_q = os.path.join(SYN, f"{RUN}-p2.pth.tar"), os.path.join(SYN, f"{RUN}-p2-q8.pth.tar")
    torch.save({"epoch": ck["epoch"], "state_dict": sd_p1, "arch": "ai85safesoundfe_p1", "extras": {"from": os.path.relpath(ck_path, REPO)}}, p1_f)
    torch.save({"epoch": ck["epoch"], "state_dict": sd_p2, "arch": "ai85safesoundmelnet_p4", "extras": {"from": os.path.relpath(ck_path, REPO)}}, p2_f)
    print(f"1) 체크포인트 {os.path.relpath(ck_path, REPO)} (epoch {ck['epoch']}) → p1 {len(sd_p1)}키 / p2 {len(sd_p2)}키")

    # ── 2. 샘플 창 ───────────────────────────────────────────────────────────
    win = np.load(os.path.join(SYN, "sample_safesound-v4d-wave.npy")).astype(np.int64)     # (128, 128): x[r][c] = s[128c + r]
    assert win.shape == (128, 128) and win.min() >= -128 and win.max() <= 127
    s = win.T.reshape(-1)                                                                   # 16,384 샘플
    X = s.reshape(fe.NPOS, fe.FOLD).T                                                       # X[r][p] = s[64p + r]  (64, 256)
    CH = 64                                                                                 # 청크 위치 수 (판단 1회 = 4,096 샘플)
    NCH = fe.NPOS // CH                                                                     # 4
    chunks = [X[:, c * CH:(c + 1) * CH] for c in range(NCH)]                                # (64, 64) ×4

    # ── 3. 양자화 + 정수 프론트엔드 ─────────────────────────────────────────
    py = os.path.join(S, "venv", "bin", "python")
    run([py, "quantize.py", p1_f, p1_q, "--device", "MAX78000"], S, os.path.join(LOGD, "synth_fe_v4d-quantize-p1.out"))
    run([py, "quantize.py", p2_f, p2_q, "--device", "MAX78000"], S, os.path.join(LOGD, "synth_fe_v4d-quantize-p2.out"))
    Wq = torch.load(p1_q, map_location="cpu", weights_only=False)["state_dict"]["fe.op.weight"].numpy().astype(np.int64)   # (64, 64, 8) int8 값
    assert Wq.min() >= -128 and Wq.max() <= 127 and np.all(Wq == np.round(Wq))
    # y[o][p] = Σ_j Σ_r W[o][r][j] X[r][p+j]  — 정확한 정수 (int64)
    y = np.zeros((fe.NCH, fe.NVALID), dtype=np.int64)
    for j in range(fe.KERNEL):
        y += Wq[:, :, j] @ X[:, j:j + fe.NVALID]
    assert np.abs(y).max() < 2 ** 31
    mag = np.abs(y[0::2]) + np.abs(y[1::2])                                                 # (32, 249)
    idx = (np.arange(fe.NFRAME) * fe.POS_PER_FRAME)[:, None] + np.arange(fe.POOL_POS)[None, :]
    E = mag[:, idx].sum(-1)                                                                  # (32, 16) 정수
    assert E.max() <= 0xFFFFFFFF
    q = to_int8(log2_q8(E.ravel()), fe.LO_Q8, fe.SPAN_Q8).astype(np.int64).reshape(fe.NBAND, fe.NFRAME)
    # 학습 그래프와 대조 (QAT 모드, 같은 체크포인트)
    E_mod = load_mod(os.path.join(REPO, "tools", "eval_confusion.py"), "eval_confusion")
    model = E_mod.load_model(ck_path, 5, T, simulate=False, config="wave_fe")
    with torch.no_grad():
        x = torch.tensor(win, dtype=torch.float32)[None] / 128.0
        e_t = model.frontend(x)
        q_t = fe._IntLog.apply(e_t)[0].numpy().astype(np.int64)
    e_scaled = e_t[0].numpy().astype(np.float64) * (1 << fe.SCALE_SH)
    d_q = int((q_t != q).sum())
    print(f"3) 정수 프론트엔드: E 범위 [{E.min()}, {E.max()}], q 범위 [{q.min()}, {q.max()}]; "
          f"학습 그래프 q 와 불일치 {d_q} / {q.size}  (E·2^14 − 정수 E 최대 |차| {np.abs(e_scaled - E).max():.3f})")
    if d_q:
        bad = np.argwhere(q_t != q)[:5]
        for b, f in bad:
            print(f"   (b={b}, f={f}) 정수 E {E[b, f]}  float E·2^14 {e_scaled[b, f]:.3f}  q 정수 {q[b, f]}  q 학습 {q_t[b, f]}")
    assert d_q <= 2, "정수 경로와 학습 그래프가 다르다 — 스케일 규약을 확인할 것"

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
    common = ["--verbose", "--test-dir", OUT, "--device", "MAX78000", "--compact-data", "--mexpress", "--timer", "0",
              "--display-checkpoint", "--overwrite"]
    run([py, "ai8xize.py", "--prefix", P2, "--checkpoint-file", p2_q, "--config-file", P2_YAML, "--softmax"] + common,
        S, os.path.join(LOGD, "synth_fe_v4d-ai8xize-p2.out"))
    log2 = open(os.path.join(OUT, P2, "log.txt"), encoding="utf-8").read()
    off = [int(v) for v in re.search(r"Kernel offset\s*=\s*\[([^\]]*)\]", log2).group(1).split(",")]
    ln = [int(v) for v in re.search(r"Kernel length\s*=\s*\[([^\]]*)\]", log2).group(1).split(",")]
    end = max(o + n for o, n in zip(off, ln))
    ws = (end + 3) // 4 * 4
    run([py, "ai8xize.py", "--prefix", P1, "--checkpoint-file", p1_q, "--config-file", P1_YAML, "--weight-start", str(ws)] + common,
        S, os.path.join(LOGD, "synth_fe_v4d-ai8xize-p1.out"))
    log1 = open(os.path.join(OUT, P1, "log.txt"), encoding="utf-8").read()
    cyc = {k: int(re.search(r"^Total\s+([\d,]+) cycles", l, re.M).group(1).replace(",", "")) for k, l in (("p1", log1), ("p2", log2))}
    wb = {k: int(re.search(r"Weight memory:\s+([\d,]+) bytes", open(os.path.join(OUT, p, "cnn.h")).read()).group(1).replace(",", ""))
          for k, p in (("p1", P1), ("p2", P2))}
    off1 = [int(v) for v in re.search(r"Kernel offset\s*=\s*\[([^\]]*)\]", log1).group(1).split(",")]
    ln1 = [int(v) for v in re.search(r"Kernel length\s*=\s*\[([^\]]*)\]", log1).group(1).split(",")]
    assert min(off1) >= end, "p1 커널이 p2 와 겹친다"
    print(f"4) 합성: p2 커널 {off}+{ln} → 끝 {end}; p1 --weight-start {ws} → {off1}+{ln1}. "
          f"사이클 p1 {cyc['p1']:,} / p2 {cyc['p2']:,}; 가중치 p1 {wb['p1']:,} + p2 {wb['p2']:,} = {wb['p1'] + wb['p2']:,} B")

    # p1 기대 출력(청크 3) == 정수 conv ?  채널 c, 위치 p → 그룹 c>>2 블록의 워드 4p + (c&3)
    so1 = parse_sample_output(os.path.join(OUT, P1, "sampleoutput.h"))
    c3 = NCH - 1
    y3 = y[:, c3 * CH:c3 * CH + (CH - fe.KERNEL + 1)]                                        # (64, 57) — 청크 3 의 유효 위치
    assert len(so1) == 16, len(so1)
    bad1 = 0
    for g, (addr, mask, words) in enumerate(so1):
        exp = [u32(y3[4 * g + (i & 3), i >> 2]) for i in range(len(words))]
        bad1 += sum(a != b for a, b in zip(words, exp))
    print(f"   p1 기대 출력(합성기) vs 정수 conv: 불일치 {bad1} / {sum(len(w) for _, _, w in so1)} "
          f"({'OK' if bad1 == 0 else 'FAIL'}); 블록 주소 {[hex(a) for a, _, _ in so1[:4]]} …")
    assert bad1 == 0
    out_addr = [a for a, _, _ in so1]

    # 청크 입력 워드 포장 == 생성 sampledata.h ?  (그룹 g 의 워드 p = 채널 4g..4g+3 의 바이트, LE)
    def pack_chunk(Xc):
        words = np.zeros((16, CH), dtype=np.uint32)
        for g in range(16):
            for i in range(4):
                words[g] |= (Xc[4 * g + i] & 0xFF).astype(np.uint32) << (8 * i)
        return words
    main1 = open(os.path.join(OUT, P1, "main.c"), encoding="utf-8").read()
    in_addr = [int(a, 16) for a in re.findall(r"memcpy32\(\(uint32_t \*\) (0x[0-9a-f]+), input_\d+, (\d+)\)", main1) for a in [a] ] if False else \
              [int(m.group(1), 16) for m in re.finditer(r"memcpy32\(\(uint32_t \*\) (0x[0-9a-f]+), input_(\d+), (\d+)\)", main1)]
    in_words = {int(m.group(2)): int(m.group(3)) for m in re.finditer(r"memcpy32\(\(uint32_t \*\) (0x[0-9a-f]+), input_(\d+), (\d+)\)", main1)}
    assert len(in_addr) == 16 and set(in_words.values()) == {CH}, (in_addr, in_words)
    sd_path = os.path.join(OUT, P1, "sampledata.h")
    pk3 = pack_chunk(chunks[c3])
    bad_in = sum(int(np.any(np.array(parse_macro_words(sd_path, f"SAMPLE_INPUT_{4 * g}"), dtype=np.uint32) != pk3[g])) for g in range(16))
    print(f"   청크 입력 포장 vs 생성 sampledata.h: 다른 그룹 {bad_in} / 16 ({'OK' if bad_in == 0 else 'FAIL'})")
    assert bad_in == 0
    out_offset = int(re.search(r"Output offset\s*=\s*\[(0x[0-9a-f]+)\]", log1).group(1), 16)
    assert out_addr == [a + out_offset for a in in_addr], (out_addr[:2], in_addr[:2], hex(out_offset))

    # p2: 입력 적재 주소·워드 수, 기대 로짓, simulate 모드 PyTorch 와 대조
    main2 = open(os.path.join(OUT, P2, "main.c"), encoding="utf-8").read()
    m2 = re.search(r"memcpy32\(\(uint32_t \*\) (0x[0-9a-f]+), input_0, (\d+)\)", main2)
    p2_in_addr, p2_in_words = int(m2.group(1), 16), int(m2.group(2))
    assert p2_in_words == fe.NBAND * fe.NFRAME
    sd2 = np.array(parse_macro_words(os.path.join(OUT, P2, "sampledata.h"), "SAMPLE_INPUT_0"), dtype=np.uint32)
    assert np.array_equal(sd2, (q.ravel() & 0xFF).astype(np.uint32)), "p2 입력은 워드당 1픽셀(하위 바이트), 행 우선이어야 한다"
    so2 = parse_sample_output(os.path.join(OUT, P2, "sampleoutput.h"))
    logits = [int(np.int32(np.uint32(w))) for _, _, ws_ in so2 for w in ws_]
    model_sim = E_mod.load_model(ck_path, 5, T, simulate=True, config="wave_fe")
    with torch.no_grad():
        lg_sim = model_sim.cls(torch.tensor(q, dtype=torch.float32)[None, None])[0].numpy()
    print(f"   p2 기대 로짓(합성기) {logits}  simulate 모드 PyTorch {lg_sim.astype(np.int64).tolist()} "
          f"({'일치' if np.array_equal(np.array(logits), lg_sim.astype(np.int64)) else '다름 — 참고'}); argmax {int(np.argmax(logits))}")
    p2_layers = int(re.search(r"0x50100008\) = (0x[0-9a-f]+); // Layer count", open(os.path.join(OUT, P2, "cnn.c")).read()).group(1), 16)
    m1 = re.search(r"0x50100008\) = (0x[0-9a-f]+); // Layer count", open(os.path.join(OUT, P1, "cnn.c")).read())
    p1_layers = int(m1.group(1), 16) if m1 else 0

    # ── 5. 펌웨어 프로젝트 조립 ──────────────────────────────────────────────
    os.makedirs(FW, exist_ok=True)
    for f in ("cnn.c", "cnn.h", "weights.h", "softmax.c", "sampleoutput.h", "Makefile"):
        shutil.copy(os.path.join(OUT, P2, f), os.path.join(FW, f))
    with open(os.path.join(FW, "cnn.h"), "a", encoding="utf-8") as f:
        f.write("\n// ── tools/fe_board_prep.py 가 덧붙임 (sync_from_synth.sh 와 같다) ──\n"
                "#ifdef MEASURE_BUILD\n#undef CNN_START\n#undef CNN_COMPLETE\n#undef SYS_START\n#undef SYS_COMPLETE\n"
                "#undef CNN_INFERENCE_TIMER\n#define CNN_START\n#define CNN_COMPLETE\n#define SYS_START\n#define SYS_COMPLETE\n#endif\n")
    # p1 생성 cnn.c → cnn_p1.c (접두어 p1_; ISR·memcpy32 는 p2 쪽 것을 쓴다)
    c1 = open(os.path.join(OUT, P1, "cnn.c"), encoding="utf-8").read()
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
             f"// 창: data/synth/sample_safesound-v4d-wave.npy (시험셋, ④ 합성과 같은 창). 체크포인트 {os.path.relpath(ck_path, REPO)}",
             "#include <stdint.h>",
             f"#define FE_NBAND {fe.NBAND}", f"#define FE_NFRAME {fe.NFRAME}", f"#define FE_POS_PER_FRAME {fe.POS_PER_FRAME}",
             f"#define FE_POOL_POS {fe.POOL_POS}", f"#define FE_NCHUNK {NCH}", f"#define FE_FRAMES_PER_HOP {CH // fe.POS_PER_FRAME}",
             f"#define FE_LO_Q8 {fe.LO_Q8}", f"#define FE_SPAN_Q8 {fe.SPAN_Q8}", f"#define FE_SCALE_SH {fe.SCALE_SH}",
             f"#define FE_P1_LAYER_COUNT {p1_layers}", f"#define FE_P2_LAYER_COUNT {p2_layers}",
             f"#define FE_P1_IN_GROUPS 16", f"#define FE_P1_IN_WORDS {CH}",
             f"#define FE_P2_IN_ADDR 0x{p2_in_addr:08x}u", f"#define FE_P2_IN_WORDS {p2_in_words}",
             f"#define FE_P1_WEIGHT_BYTES {wb['p1']}", f"#define FE_P2_WEIGHT_BYTES {wb['p2']}", f"#define FE_P2_WEIGHT_END {end}",
             "static const uint32_t FE_P1_IN_ADDR[16] = { " + ", ".join(f"0x{a:08x}u" for a in in_addr) + " };",
             "static const uint32_t FE_P1_OUT_ADDR[16] = { " + ", ".join(f"0x{a:08x}u" for a in out_addr) + " };",
             f"// 청크 c, 그룹 g(채널 4g..4g+3), 위치 p 의 입력 워드 (LE 바이트 = 연속 샘플 4개)",
             f"static const uint32_t fe_chunk_in[{NCH}][16][{CH}] = {{"]
    for c in range(NCH):
        pk = pack_chunk(chunks[c])
        lines.append("  {")
        for g in range(16):
            ws_ = ", ".join(f"0x{int(w):08x}u" for w in pk[g])
            lines.append(f"    {{ {ws_} }},")
        lines.append("  },")
    lines.append("};")
    lines.append("// 호스트 정수 경로의 특징 q[b][f] (행 우선 b·16+f) — 창 전체 16프레임")
    lines.append(carr("fe_q_expected", [str(int(v)) for v in q.ravel()], "int8_t", 16))
    lines.append(carr("fe_logits_expected", [str(v) for v in logits], "int32_t", 8))
    open(os.path.join(FW, "fe_kat.h"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    open(os.path.join(FW, "project.mk"), "w", encoding="utf-8").write(
        "# @generated — tools/fe_board_prep.py\n# 측정 빌드: LED 마킹·소프트웨어 타이머를 뺀다 (CLAUDE.md 6장 MEASURE_BUILD).\n"
        f"BOARD = FTHR_RevA\nPROJ_CFLAGS += -DMEASURE_BUILD -DNET_NAME=\\\"safesound_fe_v4d\\\"\n")
    print(f"5) 펌웨어 프로젝트 {os.path.relpath(FW, REPO)}: {sorted(os.listdir(FW))}")

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
