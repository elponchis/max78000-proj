#!/usr/bin/env python3
"""②′ (models/ai85net-safesound-fe.py) 의 **착수 전 검산 셋**과 **로그단 창 결정**.

1. 접기·탭 배치 검산 — p1 출력이 512탭 FIR 을 샘플에 직접 건 것과 같은가 (부동소수 오차 안).
2. 톤 검산 — 단일 톤이 기대 멜 밴드를 켜는가 (저역 분리도 포함).
3. 창 결정 — **학습셋** 표본(시드 0, 400창)에서 적분 에너지 E 의 분포를 보고
   LO_Q8 = round(p99·256) − SPAN_Q8 를 권고한다 (SPAN 70 dB 고정). 시험셋은 보지 않는다.
4. --verify — 모델의 bucketize 경로와 정수 경로(log2_q8 → to_int8)가 실제 E 에서 같은가.

사용 (WSL2):
    OPENBLAS_NUM_THREADS=1 ~/ai8x-training/venv/bin/python tools/init_fe_fb.py            # 1·2·3
    OPENBLAS_NUM_THREADS=1 ~/ai8x-training/venv/bin/python tools/init_fe_fb.py --verify   # + 4 (경계 파일 필요)
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "datasets"))
from log2_q8 import log2_q8, to_int8                # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai8x", default=os.path.expanduser("~/ai8x-training"))
    ap.add_argument("--data", default=os.path.expanduser("~/ai8x-training/data/SafeSoundV4"))
    ap.add_argument("--n", type=int, default=400, help="창 결정에 쓸 학습셋 표본 수")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--variant", default="fe", help="VARIANTS 의 행 (fe / b64 / f31 / b64f31)")
    a = ap.parse_args()

    sys.path.insert(0, a.ai8x)
    import torch                                    # noqa: PLC0415
    import ai8x                                     # noqa: PLC0415
    ai8x.set_device(85, False, False)
    import melfeat as MF                            # noqa: PLC0415

    m = load(os.path.join(REPO, "models/ai85net-safesound-fe.py"), "fe")
    torch.manual_seed(0)
    net = m.AI85SafeSoundFE(variant=a.variant)
    net.eval()
    NCH, NBAND = net.n_ch, net.n_band
    Wc, centers, lengths = m.filterbank_init(n_band=NBAND)
    W = Wc.transpose(0, 2, 1).reshape(NCH, m.TAPS)            # (2n, 512) 시간순 탭
    print(f"[{a.variant}] p1: Conv1d({m.FOLD}→{NCH}, k={m.KERNEL}) = {m.TAPS}탭 FIR, 홉 {m.FOLD} 샘플 ({1000*m.FOLD/MF.SR:.0f} ms); "
          f"밴드 {NBAND} ({centers[0]:.0f}~{centers[-1]:.0f} Hz), 창 길이 {lengths.min()}~{lengths.max()}탭; "
          f"프레임 {net.n_frame}개, 홉 {net.pos_per_frame*m.FOLD} 샘플, 적분 {net.pool_pos} 위치 = {(net.pool_pos-1)*m.FOLD+m.TAPS} 샘플")

    # ── 1. 접기·탭 배치 검산 ───────────────────────────────────────────
    rng = np.random.default_rng(1)
    s = rng.integers(-128, 128, 16384).astype(np.float32) / 128.0
    x = torch.from_numpy(s).reshape(-1, 128).transpose(1, 0).unsqueeze(0)      # ④ 접기 (1, 128, 128)
    with torch.no_grad():
        B = x.size(0)
        ss = x.transpose(1, 2).reshape(B, -1)
        x2 = ss.reshape(B, m.NPOS, m.FOLD).transpose(1, 2)
        y = net.fe(x2)[0].numpy()                                               # (2n, 249)
    ref = np.stack([[np.dot(W[o], s[m.FOLD*p:m.FOLD*p + m.TAPS]) for p in range(m.NVALID)] for o in range(NCH)])
    err = np.abs(y - ref).max() / (np.abs(ref).max() + 1e-12)
    print(f"1) 접기 검산: p1 출력 대 직접 FIR 최대 상대 오차 {err:.2e} → {'OK' if err < 1e-4 else '불일치!'}")
    assert err < 1e-4

    # ── 2. 톤 검산 ─────────────────────────────────────────────────────
    print("2) 톤 검산 (단일 톤 → 최대 밴드, 인접비 dB)")
    ok = True
    for f_hz in (60.0, 125.0, 300.0, 1000.0, 3000.0, 6000.0):
        tt = np.arange(16384) / MF.SR
        xs = np.clip(np.round(100 * np.sin(2 * np.pi * f_hz * tt)), -128, 127).astype(np.float32) / 128.0
        xt = torch.from_numpy(xs).reshape(-1, 128).transpose(1, 0).unsqueeze(0)
        with torch.no_grad():
            e = net.frontend(xt)[0].mean(dim=1).numpy()                        # (n_band,)
        band = int(np.argmax(e))
        near = int(np.argmin(np.abs(centers - f_hz)))
        other = np.delete(e, band)
        sep = 20 * np.log10(max(e[band], 1e-12) / max(other.max(), 1e-12))
        hit = band == near
        ok &= hit
        print(f"   {f_hz:6.0f} Hz → 밴드 {band:2d} ({centers[band]:6.0f} Hz), 기대 {near:2d} ({centers[near]:6.0f} Hz), "
              f"인접비 {sep:5.1f} dB  {'OK' if hit else '불일치'}")
    print("   →", "전부 OK" if ok else "⚠️ 불일치 있음 (저역은 512탭 분해능 한계일 수 있다)")

    # ── 3. 창 결정 (학습셋) ──────────────────────────────────────────────
    import safesound as S                            # noqa: PLC0415
    ds = S.SafeSound(a.data, d_type="train", augment=False)
    idx = np.random.default_rng(0).choice(len(ds), min(a.n, len(ds)), replace=False)
    xs = torch.stack([ds[int(i)][0] for i in idx])
    with torch.no_grad():
        E = net.frontend(xs).numpy()                                            # (n, n_band, n_frame)
    v = np.round(E.ravel() * (1 << m.SCALE_SH))
    print(f"3) 학습셋 {len(idx)}창 E 분포 (SCALE_SH={m.SCALE_SH}): 0 비율 {100*(v == 0).mean():.2f}%, "
          f"max v {v.max():.3g} (< 2^32 {'OK' if v.max() < 2**32 else '초과!'})")
    lg = np.log2(np.maximum(v[v > 0], 1.0))
    p1, p50, p99 = (np.percentile(lg, q) for q in (1, 50, 99))
    print(f"   log2 v: p1 {p1:.2f} / p50 {p50:.2f} / p99 {p99:.2f}  (p1~p99 = {(p99-p1)*6.0206:.1f} dB)")
    lo_rec = int(round(p99 * 256)) - m.SPAN_Q8
    print(f"   권고: SPAN_Q8 = {m.SPAN_Q8} (70.0 dB 고정), LO_Q8 = {lo_rec}  (창 = [{lo_rec/256:.2f}, {p99:.2f}] log2)")
    LO = net.lo_q8
    if LO is None:
        print(f"   현재 모델 상수: LO_Q8 미정 → VARIANTS['{a.variant}']['lo_q8'] = {lo_rec} 로 적고 경계를 만들 것")
        return 0
    print(f"   현재 모델 상수: LO_Q8 = {LO} → {'일치' if LO == lo_rec else '⚠️ 모델 파일의 LO_Q8 을 권고값으로 바꾸고 경계를 다시 만들 것'}")
    q_lo = (lg * 256 < LO).mean()
    q_hi = (lg * 256 >= LO + m.SPAN_Q8).mean()
    print(f"   현재 창에서 포화: 아래 {100*q_lo:.2f}% / 위 {100*q_hi:.2f}%")

    # ── 4. 경로 동일성 (경계 파일 필요) ──────────────────────────────────
    if a.verify:
        q_model = net.intlog(torch.from_numpy(E)).numpy().astype(np.int64)
        ref = to_int8(log2_q8(v.astype(np.int64).clip(0, 0xFFFFFFFF)), LO, m.SPAN_Q8).astype(np.int64).reshape(E.shape)
        bad = int((q_model != ref).sum())
        print(f"4) 모델 bucketize 대 정수 경로: 불일치 {bad} / {ref.size:,} "
              f"({'OK' if bad == 0 else '정확히 x.5 인 입력만이어야 한다 — 확인할 것'})")
        print(f"   int8 분포: min {ref.min()} / p1 {np.percentile(ref, 1):.0f} / p50 {np.percentile(ref, 50):.0f} / "
              f"p99 {np.percentile(ref, 99):.0f} / max {ref.max()}; −128 비율 {100*(ref == -128).mean():.2f}%, 127 비율 {100*(ref == 127).mean():.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
