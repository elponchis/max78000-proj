#!/usr/bin/env python3
"""KWS20 v3 사전학습 가중치를 ④ 구조에 얹어 초기 체크포인트를 만든다 (TASKS.md B).

④는 KWS20 v3 파생이라 conv 8층의 가중치 모양이 **전부 같다**. 마지막 FC 만
21클래스 → 5클래스로 바뀐다. 그 FC 만 새로 초기화하고 나머지를 물려받으면,
음성 21단어로 배운 시간축 필터를 출발점으로 쓸 수 있다.

**무엇이 얹혔는지 반드시 표로 출력한다.** `strict=False` 로 조용히 실패하면
사전학습 효과가 0 인데도 학습은 돌아가고, 결과를 보고 "사전학습이 안 듣는다" 로
잘못 결론 내린다. 그 사고를 막는 것이 이 스크립트의 존재 이유다.

## 실측 조사 (2026-09-27)

| 항목 | 값 |
|---|---|
| 체크포인트 | `~/ai8x-synthesis/trained/ai85-kws20_v3-qat8.pth.tar` |
| arch / epoch / top1 | `ai85kws20netv3` / 192 / 88.91 |
| 모양까지 일치하는 키 | 62 / 90 |
| 교체 대상 | `fc.op.weight` (5,256) vs (21,256) |
| 체크포인트에만 있음 | conv 8층 **bias** → `--model-bias` 로 받는다 |
| 우리에만 있음 | QAT 버퍼 27개 (`activation_threshold` / `final_scale` / `clamp_activation`) — ai8x 버전 차이, 기본값 사용 |

## ⚠️ 입력 스케일이 다르다 — 이 실험의 가장 큰 위험

`kws20.py:392` 가 `mx = np.amax(abs(data))` 로 **파일 단위 피크 정규화**를 한다.
우리는 하지 않는다 (CLAUDE.md 7장 — 실기기 경로에 정규화가 없어서다). 즉
사전학습된 1층 필터는 "피크가 대략 일정한" 입력에 맞춰져 있고 우리 입력은 절대
레벨이 제각각이다. 랜덤 게인 ±12dB 가 그 분포를 넓게 덮으므로 파인튜닝으로
흡수되길 기대하지만 **안 될 수도 있다.**

→ **정규화를 도입해서 맞추는 것은 금지다.** train/serve 불일치가 되고 G4(온디바이스
vs PC 괴리)의 원인을 스스로 만든다. 맞지 않으면 이 실험을 버리는 것이 맞다.

사용법 (WSL2 또는 Colab):
    python3 tools/init_from_kws20.py \\
        --src ~/ai8x-synthesis/trained/ai85-kws20_v3-qat8.pth.tar \\
        --out data/kws20v3-init-5cls.pth.tar --ai8x ~/ai8x-training
그 다음 학습에서 `--exp-load-weights-from <out>` 로 쓴다 (`--resume-from` 이
아니다 — 에폭·옵티마이저 상태를 물려받으면 파인튜닝이 아니게 된다).
"""

import argparse
import importlib.util
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# FC 는 클래스 수가 달라 반드시 새로 초기화한다. 접두사로 판정한다.
SKIP_PREFIX = ("fc.",)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.expanduser(
        "~/ai8x-synthesis/trained/ai85-kws20_v3-qat8.pth.tar"),
        help="KWS20 v3 사전학습 체크포인트 (ADI 공식 배포분)")
    ap.add_argument("--out", default=os.path.join(REPO, "data",
                                                  "kws20v3-init-5cls.pth.tar"))
    ap.add_argument("--ai8x", default=os.path.expanduser("~/ai8x-training"))
    ap.add_argument("--classes", type=int, default=5)
    ap.add_argument("--model-bias", action="store_true", default=True,
                    help="bias=True 로 모델을 만들어 체크포인트의 conv bias 를 "
                         "받는다 (기본 켬). 끄면 bias 텐서가 버려진다")
    ap.add_argument("--no-model-bias", dest="model_bias", action="store_false")
    a = ap.parse_args()

    if not os.path.isfile(a.src):
        sys.exit(f"[에러] 체크포인트가 없다: {a.src}\n"
                 "  ADI 공식 배포분은 ai8x-synthesis 레포의 trained/ 에 들어 있다.")
    sys.path.insert(0, a.ai8x)
    import torch

    import ai8x

    ai8x.set_device(85, False, False)

    spec = importlib.util.spec_from_file_location(
        "m4", os.path.join(REPO, "models", "ai85net-safesound.py"))
    mm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mm)
    model = mm.AI85SafeSoundNet(num_classes=a.classes, bias=a.model_bias)
    ours = model.state_dict()

    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    src = ck.get("state_dict", ck)
    src = {k.replace("module.", ""): v for k, v in src.items()}

    print(f"원본: {a.src}")
    for k in ("arch", "epoch"):
        if k in ck:
            print(f"  {k}: {ck[k]}")
    if isinstance(ck.get("extras"), dict) and "best_top1" in ck["extras"]:
        print(f"  best_top1: {ck['extras']['best_top1']:.2f}")
    print(f"우리 모델: AI85SafeSoundNet(num_classes={a.classes}, "
          f"bias={a.model_bias})  파라미터 "
          f"{sum(p.numel() for p in model.parameters()):,}")

    loaded, newinit, mismatch, unused = [], [], [], []
    merged = dict(ours)
    for k, v in ours.items():
        if k.startswith(SKIP_PREFIX):
            newinit.append((k, tuple(v.shape), "클래스 수가 달라 교체"))
            continue
        if k not in src:
            newinit.append((k, tuple(v.shape), "체크포인트에 없음"))
            continue
        if tuple(src[k].shape) != tuple(v.shape):
            mismatch.append((k, tuple(v.shape), tuple(src[k].shape)))
            continue
        merged[k] = src[k].clone()
        loaded.append((k, tuple(v.shape)))
    unused = [k for k in src if k not in ours]

    print(f"\n=== 얹은 층 ({len(loaded)}) ===")
    for k, sh in loaded:
        print(f"  ✔ {k:<34}{str(sh)}")
    print(f"\n=== 새로 초기화된 층 ({len(newinit)}) ===")
    for k, sh, why in newinit:
        print(f"  ✘ {k:<34}{str(sh):<18}{why}")
    if mismatch:
        print(f"\n=== ⚠️ 모양 불일치 — 얹지 못했다 ({len(mismatch)}) ===")
        for k, mine, theirs in mismatch:
            print(f"  ! {k:<34}우리 {mine}  원본 {theirs}")
    if unused:
        print(f"\n=== 원본에만 있어 버린 키 ({len(unused)}) ===")
        for k in unused[:12]:
            print(f"  - {k}")
        if len(unused) > 12:
            print(f"  ... 외 {len(unused) - 12}개")

    # 실제로 값이 바뀌었는지 검산 — "얹혔다" 는 출력만 믿지 않는다
    changed = sum(1 for k, _ in loaded if not torch.equal(merged[k], ours[k]))
    conv_loaded = [k for k, _ in loaded if "conv" in k and k.endswith("weight")]
    print(f"\n검산: 얹은 {len(loaded)}개 중 값이 실제로 달라진 것 {changed}개")
    print(f"      그중 conv 가중치 {len(conv_loaded)}개 "
          f"(8층이면 8, 그보다 적으면 무언가 빠진 것이다)")
    if not a.model_bias:
        nb = [k for k in src if k.endswith("op.bias")]
        print(f"  ⚠️ --no-model-bias 라 원본의 bias {len(nb)}개를 버렸다. "
              f"사전학습의 일부를 잃는다")

    model.load_state_dict(merged, strict=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    # ⚠️ `epoch` 키가 **반드시 있어야 한다.** `--exp-load-weights-from` 경로에서
    #    train.py:401 이 `checkpoint.get('epoch', None) >= qat_policy['start_epoch']`
    #    를 평가하므로, 없으면 `TypeError: '>=' not supported between NoneType
    #    and int` 로 죽는다 (실제로 겪었다).
    #
    #    값은 **0** 이다. 그 비교의 뜻은 "이 체크포인트가 이미 QAT 구간에서 온
    #    것인가" 이고, True 면 train.py 가 즉시 BN fold + QAT 초기화를 한다.
    #    원본 KWS20 체크포인트는 실제로 QAT 학습분(epoch 192)이지만, 여기서는
    #    **float 파인튜닝으로 시작해 기준선과 같은 지점(에폭 60)에서 QAT 로
    #    들어가야** 비교가 성립한다. QAT 가중치는 float 초기값으로도 유효하다.
    #    (즉시 QAT 로 들어가는 변형을 보고 싶으면 이 값을 192 로 두면 된다 —
    #     그때는 기준선과 QAT 시점이 달라지므로 비교표에 적어야 한다.)
    torch.save({"state_dict": model.state_dict(),
                "epoch": 0,
                "arch": "ai85safesoundnet_bias" if a.model_bias
                        else "ai85safesoundnet",
                "extras": {"init_from": os.path.basename(a.src),
                           "src_epoch": ck.get("epoch"),
                           "loaded_keys": len(loaded),
                           "new_keys": len(newinit)}}, a.out)
    # ⚠️ 저장한 키 집합이 **train.py 가 만들 모델**과 맞는지 못 박는다.
    #    train.py:797 은 `model_args["bias"] = args.use_bias` 로 bias 를 항상
    #    명시해 넘기고 `--use-bias` 기본값이 False 다. 그래서 학습 명령에
    #    `--use-bias` 를 주지 않으면 bias 텐서 9개가 버려지고
    #    "contains 9 unexpected state keys" 경고만 남는다 (실제로 겪었다).
    nbias = sum(1 for k in model.state_dict() if k.endswith("op.bias"))
    if a.model_bias:
        assert nbias == 9, f"bias 텐서가 9개여야 한다 (실제 {nbias})"
        print(f"\n⚠️ 학습 명령에 **`--use-bias` 를 반드시 포함**할 것. "
              f"이 체크포인트에는 bias 텐서 {nbias}개가 들어 있고, 빠뜨리면 "
              f"조용히 버려진다 (\"unexpected state keys\" 경고만 남는다).")
    print(f"\n저장: {a.out}")
    print("학습에서 `--exp-load-weights-from` 으로 쓸 것 "
          "(`--resume-from` 이 아니다 — 에폭·옵티마이저 상태를 물려받으면 "
          "파인튜닝이 아니게 된다).")
    print("\n⚠️ 입력 스케일이 다르다 (kws20 은 파일 단위 피크 정규화, 우리는 없음). "
          "30에폭 안에 기준선을 못 넘으면 그 때문으로 보고 중단할 것 — "
          "정규화를 도입해 맞추는 것은 금지다 (CLAUDE.md 7장).")


if __name__ == "__main__":
    main()
