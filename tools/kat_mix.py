#!/usr/bin/env python3
"""배경 혼합 증강(v2) 점검 — 사전 등록(`dataset-v2-design.md` 8절)대로 도는가.

  1. `safesound.mix_int8` == 진단 도구의 `mix` (같은 짝·같은 SNR, 비트 단위)
  2. 포화만 쓴다: 넘치는 짝에서 이벤트 성분이 줄지 않는다 (창 전체 rescale 없음)
  3. 실현 SNR 이 요청값과 같다 (포화가 없고 스케일한 배경이 4 LSB 이상인 짝, ±0.5 dB)
  4. `mix_prob=0` 이면 입력을 그대로 돌려준다 (v1 경로)
  5. `mix_prob=1` 이어도 background 창은 그대로다
  6. `mix_prob=0.5` 에서 이벤트 창의 약 절반이 바뀐다
  7. 배경 풀이 **train split 의 background** 뿐이고 0 비율 상한을 지킨다
  8. test Dataset 은 증강이 꺼져 있다 (v2 로더로 만들어도)

사용 (WSL2, ai8x venv):  python tools/kat_mix.py
"""

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI8X = os.environ.get("AI8X_DIR") or os.path.expanduser("~/ai8x-training")
sys.path[:0] = [AI8X, os.path.join(REPO, "datasets"), os.path.join(REPO, "tools")]


def main():
    import ai8x
    import safesound as S
    import diag_external_drop as D

    ai8x.set_device(85, False, False)
    root = os.path.join(AI8X, "data", "SafeSound")
    ds = S.SafeSound(root, "train", mix_prob=1.0)
    bg = len(S.CLASSES) - 1
    rng = np.random.default_rng(1)
    bad = []

    def win(i):
        _t, shard, row, _l, _r = ds.index[i]
        return np.asarray(ds._shard(shard)[row])

    ev = [i for i, x in enumerate(ds.index) if x[0] != bg]
    pool = ds._bg_pool()

    # 1. 진단 도구의 식과 비트 단위로 같은가
    n1 = 0
    for _ in range(300):
        i, j = int(rng.choice(ev)), int(rng.choice(pool))
        snr = float(rng.uniform(0, 20))
        a = S.mix_int8(win(i)[S.MARGIN:S.MARGIN + S.WIN],
                       win(j)[S.MARGIN:S.MARGIN + S.WIN], snr)
        b = D.mix(win(i)[None], np.stack([win(j)]), np.array([0]), snr)
        n1 += int(not np.array_equal(a, D.crop(b)[0]))
    print(f"1. mix_int8 == 진단 mix: 불일치 {n1}/300")
    bad += ["1"] * bool(n1)

    # 2·3. 포화만 / 실현 SNR
    e = np.full(S.WIN, 100, dtype=np.int8)
    e[::2] = -100
    b = np.full(S.WIN, 60, dtype=np.int8)
    y = S.mix_int8(e, b, 0.0)                      # g = 100/60 → 배경 +100
    ok2 = int(y.max()) == 127 and int(y.min()) == 0   # +200 → 127 포화, −100+100 = 0
    print(f"2. 포화만 (창 전체 rescale 없음): max {y.max()} min {y.min()} → {'OK' if ok2 else 'FAIL'}")
    bad += ["2"] * (not ok2)
    errs = []
    for _ in range(200):
        i, j = int(rng.choice(ev)), int(rng.choice(pool))
        ew, bw = (win(k)[S.MARGIN:S.MARGIN + S.WIN] for k in (i, j))
        snr = float(rng.uniform(0, 20))
        y = S.mix_int8(ew, bw, snr).astype(np.float64)
        re = np.sqrt((ew.astype(float) ** 2).mean())
        # 포화한 짝과, 스케일한 배경이 4 LSB 미만인 짝은 뺀다 — 뒤쪽은 반올림이
        # 배경을 지배해 SNR 을 잴 수 없다 (조용한 이벤트 + 높은 SNR. 실기기에서도
        # 그 배경은 int8 에서 사라진다)
        if np.abs(y).max() >= 127 or re / 10 ** (snr / 20) < 4:
            continue
        resid = y - ew
        got = 20 * np.log10(np.sqrt((ew.astype(float) ** 2).mean())
                            / max(np.sqrt((resid ** 2).mean()), 1e-9))
        errs.append(abs(got - snr))
    e3 = float(np.max(errs))
    print(f"3. 실현 SNR 오차: 최대 {e3:.2f} dB (n={len(errs)}) → {'OK' if e3 < 0.5 else 'FAIL'}")
    bad += ["3"] * (e3 >= 0.5)

    # 4. mix_prob=0 → 그대로
    ds0 = S.SafeSound(root, "train")
    w = win(ev[0])[S.MARGIN:S.MARGIN + S.WIN]
    ok4 = ds0.mix_prob == 0.0 and ds0._mix_noise(w, rng, 0) is w
    print(f"4. mix_prob=0 (v1 경로) 무변화: {'OK' if ok4 else 'FAIL'}")
    bad += ["4"] * (not ok4)

    # 5. background 는 그대로
    wb = win(int(pool[0]))[S.MARGIN:S.MARGIN + S.WIN]
    ok5 = all(ds._mix_noise(wb, rng, bg) is wb for _ in range(50))
    print(f"5. background 창 무변화 (mix_prob=1): {'OK' if ok5 else 'FAIL'}")
    bad += ["5"] * (not ok5)

    # 6. 확률 0.5
    ds5 = S.SafeSound(root, "train", mix_prob=S.MIX_PROB)
    ch = sum(not np.array_equal(ds5._mix_noise(w, rng, 0), w) for _ in range(2000))
    ok6 = 900 <= ch <= 1100
    print(f"6. mix_prob={S.MIX_PROB}: 2000회 중 {ch}회 변화 → {'OK' if ok6 else 'FAIL'}")
    bad += ["6"] * (not ok6)

    # 7. 배경 풀
    tg = {ds.index[int(i)][0] for i in pool}
    paths = {ds.index[int(i)][1] for i in pool}
    zmax = max(float((win(int(i))[S.MARGIN:S.MARGIN + S.WIN] == 0).mean())
               for i in pool[:: max(1, len(pool) // 500)])
    n_bg = sum(1 for x in ds.index if x[0] == bg)
    ok7 = (tg == {bg} and all(os.sep + "train" + os.sep in p for p in paths)
           and zmax <= S.MIX_POOL_MAX_ZERO)
    print(f"7. 배경 풀: {len(pool)}/{n_bg}창, 전부 train·background, 0 비율 최대 "
          f"{zmax:.3f} → {'OK' if ok7 else 'FAIL'}")
    bad += ["7"] * (not ok7)

    # 8. v2 로더의 test 는 무증강, train 은 사전 등록값
    args = argparse.Namespace(act_mode_8bit=False)
    tr, te = S.safesound_v2_get_datasets((os.path.join(AI8X, "data"), args))
    import safesound_mel as SM
    by = {d["name"]: d for d in SM.datasets}
    mtr, mte = by["SafeSoundMelU800V2"]["loader"]((os.path.join(AI8X, "data"), args))
    ok8 = (not te.augment and te.mix_prob == 0.0 and tr.mix_prob == S.MIX_PROB
           and tr.mix_snr_db == S.MIX_SNR_DB and not mte.augment and mte.mix_prob == 0.0
           and mtr.mix_prob == S.MIX_PROB)
    # 멜 경로: 이벤트 창의 특징이 혼합으로 실제로 바뀌는가
    mtr1 = SM.SafeSoundMel(os.path.join(AI8X, "data", "SafeSound"), "train",
                           scheme="log_u800", mix_prob=1.0, gain_db=0.0)
    mtr0 = SM.SafeSoundMel(os.path.join(AI8X, "data", "SafeSound"), "train",
                           scheme="log_u800", gain_db=0.0)
    i0 = ev[0]
    diff = sum(not np.array_equal(mtr1._features(i0), mtr0._features(i0)) for _ in range(5))
    ok8 = ok8 and diff >= 4
    print(f"8. 로더: test 무증강, train mix_prob={tr.mix_prob} snr={tr.mix_snr_db}; "
          f"멜 경로 혼합 반영 {diff}/5 → {'OK' if ok8 else 'FAIL'}")
    bad += ["8"] * (not ok8)

    print("\n" + ("전부 통과" if not bad else f"실패: {bad}"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
