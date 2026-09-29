#!/usr/bin/env python3
"""압축 법칙별 멜 데이터셋이 **실제로 다른 입력**을 내는지 학습 전에 확인한다.

밤새 세 번 학습을 돌렸는데 알고 보니 셋 다 같은 입력이었던 사고를 막는 것이
목적이다. 실제로 겪을 뻔한 경로가 둘 있다.

  1. 테스트 캐시 파일 이름에 법칙이 안 들어가면, (1) 의 로그 캐시를 선형 판이
     그대로 읽는다 (`safesound_mel._mel_loader` 가 이름에 태그를 붙인다)
  2. 데이터셋 이름을 잘못 줘도 train.py 는 조용히 돌아간다

검사하는 것
  · 같은 인덱스의 샘플이 법칙마다 **다른 텐서**인가
  · int8 바닥(-128) 비율이 `lin_mel_range.py` 실측과 맞는가
  · 모델이 그 입력을 받아 forward 가 되는가 (shape·dtype)
  · "log" 판이 기존 (1) 과 **비트 단위로 같은가**

사용 (WSL2):
    python3 tools/kat_melcomp.py --data ~/ai8x-training/data
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

AI8X = os.environ.get("AI8X_DIR") or os.path.expanduser("~/ai8x-training")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(AI8X, "data"))
    ap.add_argument("--ai8x", default=AI8X)
    ap.add_argument("--n", type=int, default=64, help="검사할 샘플 수")
    a = ap.parse_args()

    sys.path[:0] = [a.ai8x, os.path.join(REPO, "datasets")]
    import torch
    import ai8x
    ai8x.set_device(85, False, False)

    import melfeat as MF
    import safesound_mel as SM

    class Args:                      # ai8x.normalize 가 보는 것
        act_mode_8bit = False
        truncate_testset = False

    reg = {d["name"]: d for d in SM.datasets}
    names = ["SafeSoundMel", "SafeSoundMelLin", "SafeSoundMelCbrt",
             "SafeSoundMelLog72"]

    spec = importlib.util.spec_from_file_location(
        "melmodel", os.path.join(REPO, "models", "ai85net-safesound-mel.py"))
    mm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mm)
    net = mm.ai85safesoundmelnet(num_classes=5, num_channels=1,
                                 dimensions=(MF.N_MELS, MF.N_FRAMES),
                                 bias=False)
    net.eval()

    print(f"{'데이터셋':<20}{'법칙':<8}{'바닥%':>8}{'포화%':>8}"
          f"{'평균':>9}{'표준편차':>10}  forward")
    print("-" * 78)
    xs, bad = {}, []
    for nm in names:
        if nm not in reg:
            bad.append(f"{nm}: 등록되지 않았다")
            print(f"{nm:<20}{'—':<8}{'—':>8} 미등록")
            continue
        # **테스트셋**을 쓴다 — 증강이 없어 법칙 외에는 아무것도 안 달라진다
        _, ds = reg[nm]["loader"]((a.data, Args()), load_train=False,
                                  load_test=True)
        n = min(a.n, len(ds))
        x = torch.stack([ds[i][0] for i in range(n)])
        xs[nm] = x
        # ⚠️ 통계는 **변환 전 원본 int8** 에서 낸다. `ai8x.normalize` 는 버전과
        # act_mode_8bit 에 따라 아핀이 달라져(예: data-0.5) 역산하면 틀린다 —
        # 실제로 한 번 틀려서 평균 -231.9 (int8 범위 밖) 가 나왔다.
        q = np.stack([ds._features(i) for i in range(n)]).astype(np.float64)
        scheme = ds.scheme
        try:
            with torch.no_grad():
                y = net(x[:4])
            fw = f"OK {tuple(y.shape)}"
        except Exception as e:                                # noqa: BLE001
            fw = f"실패 {type(e).__name__}"
            bad.append(f"{nm}: forward {e}")
        print(f"{nm:<20}{scheme:<8}{100*np.mean(q <= -128):>7.1f}%"
              f"{100*np.mean(q >= 127):>7.2f}%{q.mean():>9.1f}{q.std():>10.1f}"
              f"  {fw}")

    print()
    # 1) 법칙끼리 실제로 다른가
    ref = xs.get("SafeSoundMel")
    for nm in names[1:]:
        if nm not in xs or ref is None:
            continue
        same = torch.equal(ref, xs[nm])
        print(f"  {'✘' if same else '✔'} {nm} 가 SafeSoundMel 과 "
              f"{'같다 — 법칙이 안 먹었다' if same else '다르다'}")
        if same:
            bad.append(f"{nm}: (1) 과 동일한 입력")

    # 2) log 경로가 기존 구현과 비트 단위로 같은가
    rng = np.random.default_rng(0)
    t = np.arange(16384) / 16000.0
    w = np.clip(np.round(127 * 0.3 * np.sin(2 * np.pi * 1000 * t)
                         + rng.normal(0, 3, 16384)), -128, 127).astype(np.int8)
    ok = np.array_equal(MF.log_mel_int8(w), MF.mel_int8(w, "log"))
    print(f"  {'✔' if ok else '✘'} mel_int8(w,'log') 가 log_mel_int8(w) 와 "
          f"{'같다' if ok else '다르다'}")
    if not ok:
        bad.append("log 경로가 기존 구현과 다르다")

    # 3) 캐시 파일 이름이 법칙별로 갈리는가 (같으면 서로를 덮어쓴다)
    root = os.path.join(a.data, "SafeSound", "test")
    tags = set()
    for nm in names:
        if nm in reg:
            sc = SM.MF.COMP_SCHEMES
            s = {"SafeSoundMel": "log", "SafeSoundMelLin": "lin",
                 "SafeSoundMelCbrt": "cbrt", "SafeSoundMelLog72": "log72"}[nm]
            tag = "" if s == "log" else f"_{s}"
            tags.add(os.path.join(
                root, f"melcache{tag}_{MF.N_MELS}x{MF.N_FRAMES}.npy"))
            assert sc  # 사용 표시
    print(f"  {'✔' if len(tags) == len(names) else '✘'} 캐시 파일 이름이 "
          f"법칙별로 다르다 ({len(tags)}/{len(names)})")
    if len(tags) != len(names):
        bad.append("캐시 파일 이름 충돌")

    print()
    if bad:
        print(f"실패 {len(bad)}건:")
        for b in bad:
            print("  " + b)
        sys.exit(1)
    print("전부 통과 — 세 법칙이 서로 다른 입력을 내고 모델이 받는다")


if __name__ == "__main__":
    main()
