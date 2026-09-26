#!/usr/bin/env python3
"""모델이 실제로 받는 int8 입력의 통계 — 두 구성을 나란히.

**학습 없이 "입력이 무정보인가" 를 가르는 도구다.** 한 클래스로 붕괴한 학습을
보면 원인이 둘인데 손실 곡선으로는 갈리지 않는다.

  · **입력이 무정보다** — 포화·상수·전부 0 이면 모델이 배울 것이 없고, 역빈도
    가중 손실에서는 가장 가중치가 큰 클래스로 붕괴한다
  · **최적화가 아직 안 풀렸다** — 입력은 정상인데 학습률·에폭이 문제다

여기서 보는 것 (모델 입력 직전, `ai8x.normalize` 를 통과한 int8 값):
  - min / max / 평균 / 표준편차
  - **−128, +127 에 붙은 비율** — 포화가 심하면 표현이 망가진 것이다
  - **샘플별 표준편차가 거의 0 인 샘플 비율** — 그 샘플은 상수 그림이라
    어떤 CNN 도 구분할 수 없다
  - 클래스별 같은 통계 — 특정 클래스만 비어 있는지 본다

⚠️ 증강을 켠 **학습 경로**를 기본으로 본다. 학습이 붕괴했다면 학습이 본 것을
봐야 한다. `--split test` 로 무증강 경로도 같이 확인할 수 있다.

사용법 (WSL2):
    ~/ai8x-training/venv/bin/python tools/input_stats.py \\
        --root data/processed/safesound --ai8x ~/ai8x-training --batches 20
"""

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "datasets"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FLAT_STD = 1.0      # 샘플 표준편차가 이 값(int8 LSB) 미만이면 "거의 상수"로 본다


def collect(ds, norm, n, batch, rng):
    """무작위 n*batch 창의 int8 입력을 모은다. (X (N, ...), y (N,))"""
    import torch

    idx = rng.choice(len(ds), size=min(n * batch, len(ds)), replace=False)
    xs, ys = [], []
    for i in idx:
        x, t = ds[int(i)]
        xs.append(norm(x))
        ys.append(t)
    return torch.stack(xs).numpy(), np.array(ys)


def report(label, X, y, names):
    flat = X.reshape(len(X), -1)
    per_std = flat.std(axis=1)
    dead = per_std == 0.0                       # 창 전체가 한 값 — 완전 무정보
    print(f"\n=== {label}  (창 {len(X):,}개, 입력 {tuple(X.shape[1:])}) ===")
    print(f"  {'':<12}{'min':>7}{'max':>7}{'평균':>9}{'표준편차':>10}"
          f"{'-128%':>8}{'+127%':>8}{'std<1':>7}{'전부동일':>9}")
    print("  " + "-" * 77)

    def line(nm, m):
        f = flat[m]
        if not len(f):
            print(f"  {nm:<12}{'(없음)':>7}")
            return
        print(f"  {nm:<12}{f.min():>7.0f}{f.max():>7.0f}{f.mean():>9.2f}"
              f"{f.std():>10.2f}{100*(f == -128).mean():>7.2f}%"
              f"{100*(f == 127).mean():>7.2f}%"
              f"{100*(per_std[m] < FLAT_STD).mean():>6.1f}%"
              f"{100*dead[m].mean():>8.1f}%")

    line("전체", np.ones(len(X), bool))
    for c, nm in enumerate(names):
        line(nm, y == c)
    print(f"  'std<1'   창 내부 표준편차가 {FLAT_STD} LSB 미만 — 거의 평평한 창")
    print("  '전부동일' 창의 모든 값이 같음 — **완전 무정보**. 여기가 0 이 아니면")
    print("            그 창들은 어떤 모델도 구분할 수 없다")
    return {"n": int(len(X)), "min": float(flat.min()), "max": float(flat.max()),
            "mean": float(flat.mean()), "std": float(flat.std()),
            "sat_lo": float((flat == -128).mean()),
            "sat_hi": float((flat == 127).mean()),
            "flat_frac": float((per_std < FLAT_STD).mean()),
            "dead_frac": float(dead.mean())}


def valid_split_shares(root, names):
    """ai8x 검증셋의 클래스 구성 → **한 클래스로 붕괴했을 때의 val Top1** 표.

    붕괴한 학습의 val Top1 은 곧 "그 클래스가 검증셋에서 차지하는 비율" 이다.
    그 값을 미리 표로 두면, 로그의 Top1 숫자만 보고 어느 클래스로 붕괴했는지
    바로 읽을 수 있다.
    """
    import safesound as S
    from eval_confusion import ai8x_valid_indices

    tr = S.SafeSound(root, "train", transform=None, augment=False)
    vidx = ai8x_valid_indices(len(tr), 0.1, 0)
    y = np.array([tr.index[i][0] for i in vidx])
    print(f"\n=== ai8x 검증셋 구성 (train {len(tr):,} 중 {len(y):,}창, "
          f"split 0.1 / seed 0) ===")
    print("  한 클래스로 붕괴하면 val Top1 이 그 클래스의 비율과 같아진다 —")
    print("  로그의 Top1 숫자로 어느 클래스로 붕괴했는지 바로 읽을 수 있다.")
    print(f"\n  {'클래스':<12}{'창':>7}{'붕괴 시 val Top1':>18}")
    print("  " + "-" * 38)
    for c, nm in enumerate(names):
        k = int((y == c).sum())
        print(f"  {nm:<12}{k:>7,}{100*k/len(y):>17.3f}%")
    return {nm: round(100 * float((y == c).mean()), 3)
            for c, nm in enumerate(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/safesound")
    ap.add_argument("--ai8x", default=os.path.expanduser("~/ai8x-training"))
    ap.add_argument("--split", default="train", choices=["train", "test"],
                    help="train(기본)은 **증강을 켠다** — 학습이 본 것을 본다")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--float-scale", action="store_true",
                    help="float 학습이 실제로 받는 [−1, 0.992) 스케일로 출력한다. "
                         "기본은 int8 정수 단위 — 아래 설명 참조")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-valid-split", action="store_true")
    a = ap.parse_args()

    if os.path.isdir(a.ai8x):
        sys.path.insert(0, a.ai8x)
    import argparse as _ap

    import ai8x
    import safesound as S
    import safesound_mel as SM

    # ⚠️ **int8 정수 단위로 재는 것이 기본이다.** `ai8x.normalize` (ai8x.py:37)는
    #    act_mode_8bit 이면 [−128,127] 정수를, 아니면 **같은 정수를 128로 나눈**
    #    [−1, 0.992) 를 내놓는다. 값은 동일하고 스케일만 다르므로, 포화 비율과
    #    '상수 창' 판정을 LSB 단위로 읽으려면 정수 쪽이 맞다.
    #    (처음에 float 스케일로 재서 표준편차 0.13, 상수 창 100% 라는 엉뚱한
    #     결과를 봤다 — 임계값 1.0 LSB 가 그 스케일에서는 전 구간을 삼킨다.)
    int8 = not a.float_scale
    ai8x.set_device(85, int8, False)
    norm = ai8x.normalize(args=_ap.Namespace(act_mode_8bit=int8))
    names = S.CLASSES
    augment = (a.split == "train")
    print(f"split={a.split}  증강={'켬' if augment else '끔'}  "
          f"단위={'int8 정수 [−128,127]' if int8 else 'float [−1,0.992)'}  "
          f"창 {a.batches * a.batch_size:,}개 표본")
    if int8:
        print("  (float 학습은 같은 정수를 128로 나눈 값을 받는다 — 값은 동일)")

    out = {}
    for label, cls in (("④ wave  raw 파형 (128,128)", S.SafeSound),
                       ("① mel   로그 멜 (1,64,64)", SM.SafeSoundMel)):
        # 같은 시드 → 두 구성이 **같은 창 집합**을 본다 (비교의 전제)
        rng = np.random.default_rng(a.seed)
        ds = cls(a.root, a.split, transform=None, augment=augment)
        X, y = collect(ds, norm, a.batches, a.batch_size, rng)
        out[label.split()[0]] = report(label, X, y, names)

    if not a.no_valid_split:
        out["valid_split"] = valid_split_shares(a.root, names)

    print("\n판정 기준")
    print("  · 포화(−128/+127)가 수십 %면 표현이 망가진 것이다")
    print("  · '상수 창' 이 몇 %를 넘으면 그 창들은 어떤 모델도 구분할 수 없다")
    print("  · 표준편차가 한 자리 LSB 수준이면 입력의 동적범위가 거의 없는 것이다")
    print("  · 위 셋이 정상인데 학습이 붕괴했다면 **입력 문제가 아니다** —")
    print("    학습률·에폭·초기화 쪽을 볼 것 (역빈도 가중 손실은 학습 초기에")
    print("    가중치가 큰 클래스로 쏠렸다가 풀리는 것이 정상 경로다)")


if __name__ == "__main__":
    main()
