#!/usr/bin/env python3
"""모델 forward known-answer 테스트 — 배치 크기·메모리 레이아웃·컴파일 회귀.

**왜 필요한가.** 구성 ① 학습이 Colab 에서 에폭 0 의 **마지막 배치(59샘플)** 에서
죽었다.

    models/ai85net-safesound-mel.py:98  x = x.view(x.size(0), -1)
    RuntimeError: view size is not compatible with input tensor's size and stride

원인은 `view` 다. torch.compile 이 마지막 배치 크기 변화로 재컴파일하면서 conv
출력을 **channels_last** 로 배치했고, 그때 (C,H,W) 가 메모리에서 연속이 아니라
`view` 가 불가능해진다. 에폭 0 을 다 돌고 나서야 터지므로 **학습 시간을 통째로
버린다** — 그래서 학습 전에 CPU 에서 몇 초 만에 잡는다.

⚠️ **핵심 검사는 `channels_last` 쪽이다.** 모델과 입력을 channels_last 로 바꾸면
CPU·eager 에서도 같은 예외가 재현된다. 반면 CPU 의 `torch.compile` 은 이 버그를
잡지 못한다 — 실측으로 확인했다(`view` 로 되돌려도 CPU 컴파일 경로는 통과한다).
CPU inductor 가 channels_last 를 고르지 않기 때문이고, CUDA 쪽이 고른다.
그래서 컴파일 검사만 믿으면 안 되고 channels_last 검사가 반드시 있어야 한다.

확인하는 것
  1. 마지막 배치처럼 **어중간한 배치 크기**(59)에서 forward 가 통과하는가
  2. **channels_last** 입력에서 통과하는가 ← 실제로 죽었던 자리
  3. 평탄화가 **논리 순서를 보존**하는가 — channels_last 결과가 contiguous 결과와
     원소 단위로 같아야 한다. 같으면 FC 가중치 대응이 바뀌지 않았다는 뜻이고,
     합성(`ai8xize.py`)의 FC 평탄화 순서도 그대로다
  4. `torch.compile(dynamic=True)` 로 배치 크기를 바꿔 가며 통과하는가
     (컴파일러가 없는 환경에서는 건너뛰고 그 사실을 적는다)

사용법 (WSL2 / Colab):
    python3 tools/kat_models.py --ai8x ~/ai8x-training
"""

import argparse
import importlib.util
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (파일, 클래스, 입력 모양 — 배치 제외, 추가 생성 인자)
MODELS = [
    ("ai85net-safesound.py", "AI85SafeSoundNet", (128, 128), {}),
    ("ai85net-safesound-mel.py", "AI85SafeSoundMelNet", (1, 64, 64), {}),
    # wave2D 대조 — ①과 같은 구조에 raw 파형. 앞 풀링이 하나 더 붙는다
    ("ai85net-safesound-mel.py", "AI85SafeSoundMelNet", (1, 128, 128),
     {"pool_first": True, "dimensions": (128, 128)}),
]

# 실패했던 마지막 배치 크기. 11707 % 128 = 59 (구성 ① 학습 에폭 0).
ODD_BATCH = 59

OK, FAIL = "  [통과]", "  [실패]"
fails = []


def check(name, cond, detail=""):
    print(f"{OK if cond else FAIL} {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def load(fname, cname):
    spec = importlib.util.spec_from_file_location(
        "katmodel_" + cname, os.path.join(REPO, "models", fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return getattr(m, cname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai8x", default=os.path.expanduser("~/ai8x-training"))
    ap.add_argument("--classes", type=int, default=5)
    ap.add_argument("--no-compile", action="store_true",
                    help="torch.compile 검사를 건너뛴다")
    a = ap.parse_args()

    if not os.path.isdir(a.ai8x):
        sys.exit(f"[에러] ai8x 경로가 없다: {a.ai8x} (--ai8x 로 줄 것)")
    sys.path.insert(0, a.ai8x)
    import torch
    import ai8x

    ai8x.set_device(85, False, False)
    print(f"torch {torch.__version__}  배치 {ODD_BATCH} (실패했던 마지막 배치)\n")

    for fname, cname, shape, extra in MODELS:
        tag = cname + (" (wave2D)" if extra else "")
        print(f"── {tag}  입력 (B, {', '.join(str(s) for s in shape)})")
        net = load(fname, cname)(num_classes=a.classes, bias=False,
                                 **extra).eval()
        x = torch.randn(ODD_BATCH, *shape)

        with torch.no_grad():
            try:
                y = net(x)
                ref = y.clone()
                ok = tuple(y.shape) == (ODD_BATCH, a.classes)
            except Exception as e:                       # noqa: BLE001
                ref, ok = None, False
                print(f"    {type(e).__name__}: {str(e)[:120]}")
            check(f"{tag}: 배치 {ODD_BATCH} forward", ok,
                  "" if ref is None else str(tuple(ref.shape)))

        # ── channels_last — 실제로 죽었던 자리. 4D 입력에만 있는 개념이다
        if len(shape) == 3:
            netl = load(fname, cname)(num_classes=a.classes, bias=False,
                                      **extra).eval()
            netl.load_state_dict(net.state_dict())
            netl = netl.to(memory_format=torch.channels_last)
            xl = x.to(memory_format=torch.channels_last)
            with torch.no_grad():
                try:
                    yl = netl(xl)
                    check(f"{tag}: channels_last forward",
                          tuple(yl.shape) == (ODD_BATCH, a.classes),
                          str(tuple(yl.shape)))
                    # 3. 평탄화가 논리 순서를 보존하는가 = 같은 값이 나오는가
                    same = ref is not None and torch.allclose(yl, ref, atol=1e-4)
                    check(f"{tag}: channels_last 결과가 contiguous 와 동일 "
                          f"(평탄화 순서 보존 → FC·합성 영향 없음)", same,
                          "" if ref is None else
                          f"최대 차이 {float((yl - ref).abs().max()):.2e}")
                except Exception as e:                   # noqa: BLE001
                    check(f"{tag}: channels_last forward", False,
                          f"{type(e).__name__}: {str(e)[:110]} "
                          f"← `view` 를 `flatten` 으로 바꿀 것")
        else:
            print(f"    (1D 경로라 channels_last 개념이 없다 — 건너뜀)")

        # ── torch.compile(dynamic=True): 배치 크기를 바꿔 재컴파일을 유발한다
        if a.no_compile or not hasattr(torch, "compile"):
            print("    (torch.compile 검사 생략)")
            continue
        try:
            comp = torch.compile(net, dynamic=True)
            with torch.no_grad():
                comp(torch.randn(128, *shape))           # 정상 배치
                yc = comp(torch.randn(ODD_BATCH, *shape))  # 마지막 배치 → 재컴파일
            check(f"{tag}: torch.compile(dynamic=True) 배치 128→{ODD_BATCH}",
                  tuple(yc.shape) == (ODD_BATCH, a.classes), str(tuple(yc.shape)))
        except Exception as e:                           # noqa: BLE001
            # 컴파일러·triton 이 없는 환경은 흔하다. 그건 모델 결함이 아니다.
            msg = f"{type(e).__name__}: {str(e)[:100]}"
            if any(k in str(e).lower() for k in
                   ("compiler", "gcc", "cl.exe", "triton", "backend",
                    "not supported", "c++")):
                print(f"    (torch.compile 불가한 환경 — 건너뜀: {msg})")
            else:
                check(f"{tag}: torch.compile(dynamic=True)", False, msg)
        print()

    if fails:
        print(f"실패 {len(fails)}건: " + ", ".join(fails))
        sys.exit(1)
    print("전부 통과")


if __name__ == "__main__":
    main()
