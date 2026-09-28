#!/usr/bin/env python3
"""교사 로짓의 **전역 스케일 상수**를 학습셋에서 구한다 (C′ 준비).

## 왜 필요한가

C(첫 증류)가 ④-ft 대비 −0.085 로 나빠졌다. 원인은 ① 의 로짓이 |값| 평균 1435
규모라(QAT 체크포인트를 `act_mode_8bit` 없이 추론하면 `output_shift` 스케일링이
빠진다) **T=4 에서 최대확률이 0.998** — 소프트 타깃이 사실상 one-hot 이었던 것이다
(`docs/results/g8-c-distillation.md`).

교사 로짓을 상수 `s` 로 나누면 유효 온도가 `T·s` 가 된다. 적당한 `s` 를 고르면
통상적인 `T=4` 로 교과서적인 증류가 된다.

## ⚠️ 샘플별 정규화를 하지 않는다

**전역 상수 하나**로만 나눈다. 샘플 간 확신도 차이 — 쉬운 창은 뾰족하고 어려운
창은 평평하다 — 가 증류가 전달하려는 정보의 핵심이다. 샘플마다 정규화하면 모든
창을 같은 뾰족함으로 만들어 그 정보를 지운다.

## 어떻게 고르나

로짓의 **표준편차**를 목표값 `--target-std` 로 맞추는 상수를 쓴다
(`s = std / target_std`). KD 문헌의 통상적인 로짓 규모에서 `T=4` 가 잘 동작하므로,
우리 로짓을 그 규모로 옮기는 것이다. 여러 후보에 대해 **T=4 에서의 유효 클래스
수(exp(엔트로피))** 를 표로 내고 고른 값을 체크포인트에 써 넣는다.

판정 기준은 `tools/kat_teacher.py` 와 같다 — **유효 클래스 수 ≥ 1.5**
(1.0=one-hot, 5.0=균등). 너무 크면(≈5) 교사 정보가 사라지므로 2~3 을 노린다.

사용법 (WSL2):
    python3 tools/calib_teacher.py --teacher data/teacher-mel.pth.tar \\
        --ai8x ~/ai8x-training --data ~/ai8x-training/data --write
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "datasets"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MIN_EFF = 1.5           # kat_teacher.py 와 같은 하한
TARGET_EFF = (2.0, 3.5)  # 노리는 구간 — 너무 크면 교사 정보가 사라진다


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=os.path.join(REPO, "data",
                                                      "teacher-mel.pth.tar"))
    ap.add_argument("--ai8x", default=os.path.expanduser("~/ai8x-training"))
    ap.add_argument("--data", default=os.path.expanduser("~/ai8x-training/data"))
    ap.add_argument("--split", default="train",
                    help="스케일을 구할 split. **학습셋이 기본**이다 — "
                         "테스트셋에서 구하면 평가 정보가 학습에 새어든다")
    ap.add_argument("--temp", type=float, default=4.0, help="학습에 쓸 온도")
    ap.add_argument("--target-std", type=float, default=2.0,
                    help="정규화 후 로짓 표준편차 목표")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="0 이면 전체")
    ap.add_argument("--write", action="store_true",
                    help="고른 상수를 교사 체크포인트에 써 넣는다")
    a = ap.parse_args()

    sys.path.insert(0, a.ai8x)
    import torch

    import ai8x
    import safesound as S

    ai8x.set_device(85, False, False)
    spec = importlib.util.spec_from_file_location(
        "melmod", os.path.join(REPO, "models", "ai85net-safesound-mel.py"))
    mm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mm)

    teacher = mm.ai85safesoundmelteacher(num_classes=len(S.CLASSES))
    ck = torch.load(a.teacher, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    missing, unexpected = teacher.load_state_dict(sd, strict=False)
    print(f"교사: {a.teacher}  (누락 {len(missing)} / 초과 {len(unexpected)})")
    if any("net.op" in k or "net.voice" in k or "net.conv" in k for k in missing):
        sys.exit("[에러] 교사 가중치가 얹히지 않았다 — wrap_teacher.py 먼저 실행")
    # 스케일을 구할 때는 **원본 로짓**이 필요하다 (이미 스케일이 들어 있으면 되돌린다)
    prev = float(teacher.logit_scale)
    teacher.logit_scale.fill_(1.0)
    teacher.eval()

    # ⚠️ 학습셋, **증강 없이**. 증강을 켜면 매번 다른 값이 나와 상수가 재현되지 않는다
    ds = S.SafeSound(os.path.join(a.data, "SafeSound"), a.split,
                     transform=ai8x.normalize(args=argparse.Namespace(
                         act_mode_8bit=True)), augment=False)
    n = len(ds) if a.limit <= 0 else min(a.limit, len(ds))
    print(f"{a.split} split {n:,}창에서 로짓 분포를 잰다 (증강 없음)\n")

    logits = []
    with torch.no_grad():
        for i in range(0, n, a.batch_size):
            xs = torch.stack([ds[j][0] for j in range(i, min(i + a.batch_size, n))])
            logits.append(teacher(xs))
            if (i // a.batch_size) % 20 == 0:
                print(f"  {min(i + a.batch_size, n):,}/{n:,}", flush=True)
    lg = torch.cat(logits)
    teacher.logit_scale.fill_(prev)

    std = float(lg.std())
    gap = float((lg.topk(2, 1).values[:, 0] - lg.topk(2, 1).values[:, 1]).mean())
    print(f"\n로짓 분포 ({len(lg):,}창)")
    print(f"  표준편차 {std:.1f}   |값| 평균 {float(lg.abs().mean()):.1f}   "
          f"최대 {float(lg.abs().max()):.1f}")
    print(f"  최대−차순 간격 평균 {gap:.1f}")

    def eff_classes(scale, T):
        p = torch.softmax(lg / scale / T, 1)
        ent = -(p * p.clamp_min(1e-12).log()).sum(1)
        return float(ent.exp().mean()), float(p.max(1).values.mean())

    print(f"\n후보 상수별 T={a.temp} 에서의 소프트 타깃 "
          f"(유효 클래스 1.0=one-hot / {len(S.CLASSES)}.0=균등)")
    print(f"  {'target_std':>11}{'상수 s':>10}{'유효 클래스':>12}"
          f"{'최대확률':>10}   판정")
    print("  " + "-" * 56)
    best = None
    for tgt in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        s = std / tgt
        e, mx = eff_classes(s, a.temp)
        ok = TARGET_EFF[0] <= e <= TARGET_EFF[1]
        # 라벨은 세 갈래다 — 하한 미달 / 노리는 구간보다 뾰족 / 너무 평평
        mark = ("★ 노리는 구간" if ok else
                "한계 미달" if e < MIN_EFF else
                "너무 뾰족" if e < TARGET_EFF[0] else "너무 평평")
        print(f"  {tgt:>11.1f}{s:>10.1f}{e:>12.2f}{mx:>10.4f}   {mark}")
        if ok and (best is None or abs(e - 2.5) < abs(best[2] - 2.5)):
            best = (tgt, s, e, mx)

    if best is None:
        print("\n⚠️ 노리는 구간에 드는 후보가 없다. --target-std 범위를 넓힐 것")
        sys.exit(1)
    tgt, s, e, mx = best
    print(f"\n선택: **target_std {tgt} → 상수 s = {s:.1f}**")
    print(f"  T={a.temp} 에서 유효 클래스 {e:.2f}, 최대확률 {mx:.4f}")
    print(f"  (유효 온도는 원본 로짓 기준 T·s = {a.temp * s:.0f} 에 해당한다)")

    if a.write:
        teacher.logit_scale.fill_(s)
        ck["state_dict"] = teacher.state_dict()
        ck.setdefault("extras", {}).update(
            logit_scale=round(s, 4), logit_std=round(std, 2),
            calib_split=a.split, calib_n=int(len(lg)), kd_temp=a.temp,
            eff_classes=round(e, 3))
        torch.save(ck, a.teacher)
        print(f"\n체크포인트에 기록: {a.teacher}")
        print(f"  logit_scale={s:.4f}  (extras 에 측정값도 함께 저장)")
        print("  다음: tools/kat_teacher.py 로 유효 클래스 검사를 통과하는지 확인")
    else:
        print("\n(--write 를 주면 체크포인트에 기록한다)")


if __name__ == "__main__":
    main()
