#!/usr/bin/env python3
"""구성 B 의 NPU STFT 층 — **합성용 체크포인트·샘플·yaml 을 만든다** (학습 없음).

B 는 구현하기 전에 **비용만** 본다: 고정 Conv1d(250 → 510, k=3, 4bit 가중치,
32bit 출력)를 `ai8xize.py` 에 넣어 NPU 사이클 수와 가중치 메모리를 읽는다.
가중치는 학습한 것이 아니라 **DFT 기저 × Hann 을 4bit 로 반올림한 상수**다
(`datasets/melfeat.py` `_stft_basis_q` 와 같은 값).

  입력 배치  x[c][t] = sample[122 + 250*t + c]   (채널 = hop 한 칸의 샘플 250개)
  출력       y[2k'][f], y[2k'+1][f] = 빈 k'+1 의 Re, Im   (k' = 0..254)
  프레임 f 는 시간 칸 f, f+1, f+2 를 본다 (k=3). 세 번째 칸은 12샘플만 유효하다

두 경우를 만든다: 64프레임 전부(T=66) / 판단당 새 16프레임(T=18).

사용 (WSL2, ai8x-training venv):
    ~/ai8x-training/venv/bin/python tools/b_stft_synth.py
그다음 `bash data/logs-local/synth_b_stft.sh` 가 합성한다.
"""

import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "datasets"))
import melfeat as MF                                         # noqa: E402

BITS = 4
HOP, NFFT = MF.HOP_INC, MF.N_FFT
K = 3
N_BINS = NFFT // 2 - 1                    # 빈 1..255
SYN = os.path.expanduser("~/ai8x-synthesis")
OUT = os.path.join(REPO, "data", "synth")
YAML = """# 구성 B — NPU 고정 STFT 층 ({frames}프레임). tools/b_stft_synth.py 생성.
# 비용 확인용이다 (학습·구현 전). 가중치는 DFT 기저 x Hann 의 {bits}bit 상수.
arch: stftconv
dataset: {ds}

layers:
  # Conv1d k=3, 250 -> 510 (Re/Im x 255빈), 활성화 없음, 32bit 출력
  - pad: 0
    activate: None
    out_offset: 0x4000
    processors: 0xffffffffffffffff
    data_format: HWC
    operation: Conv1d
    kernel_size: 3
    quantization: {bits}
    output_width: 32
"""


def main():
    wq, s = MF._stft_basis_q(BITS)                           # noqa: SLF001
    # wq[0|1][bin][n], n = 0..511  →  Conv1d 가중치 w[out][c][j], n = 250*j + c
    w = np.zeros((2 * N_BINS, HOP, K), dtype=np.float32)
    for j in range(K):
        n0 = j * HOP
        n1 = min(NFFT, n0 + HOP)
        w[0::2, :n1 - n0, j] = wq[0][1:N_BINS + 1, n0:n1]
        w[1::2, :n1 - n0, j] = wq[1][1:N_BINS + 1, n0:n1]
    assert np.abs(w).max() <= 2 ** (BITS - 1) - 1
    nz = int((w != 0).sum())
    print(f"가중치 {w.shape} = {w.size:,}개 (0 아닌 것 {nz:,}), 값 범위 "
          f"[{int(w.min())}, {int(w.max())}], 스케일 S={s:g}")

    sd = {
        "stft.output_shift": torch.tensor([0.0]),
        "stft.weight_bits": torch.tensor([BITS]),
        "stft.bias_bits": torch.tensor([8]),
        "stft.quantize_activation": torch.tensor([1.0]),
        "stft.adjust_output_shift": torch.tensor([0.0]),
        "stft.shift_quantile": torch.tensor([1.0]),
        "stft.op.weight": torch.from_numpy(w),
    }
    ck = os.path.join(OUT, "b-stft-q4.pth.tar")
    torch.save({"epoch": 0, "state_dict": sd, "arch": "stftconv"}, ck)
    print(f"체크포인트: {ck}")

    win = np.load(os.path.join(REPO, "tools", "kat_vectors", "melkat_window_int8.npy"))
    stream = np.concatenate([win[MF.OFF_INC:], np.zeros(HOP * K, dtype=np.int8)])
    for frames in (64, 16):
        t = frames + K - 1
        x = stream[:HOP * t].reshape(t, HOP).T.astype(np.int64)      # (250, T)
        ds = f"SafeSoundSTFT{frames}"
        np.save(os.path.join(SYN, "tests", f"sample_{ds.lower()}.npy"), x)
        with open(os.path.join(REPO, "synthesis", f"b-stft-{frames}.yaml"), "w",
                  encoding="utf-8") as f:
            f.write(YAML.format(frames=frames, bits=BITS, ds=ds))
        print(f"  {frames}프레임: 입력 {x.shape}, MAC {w.shape[0] * HOP * K * frames:,} "
              f"(0 아닌 가중치만 {nz * frames:,})")


if __name__ == "__main__":
    main()
