#!/usr/bin/env python3
"""모든 모델 진입점을 **train.py 가 부르는 방식 그대로** 생성해 본다.

`train.py:792-800` 이 model_args 를 이렇게 만든다:
    pretrained / num_classes / num_channels=dimensions[0] /
    dimensions=(dimensions[1], dimensions[2]) / bias=args.use_bias /
    weight_bits / bias_bits / quantize_activation

진입점이 이 중 하나를 **하드코딩**하면 `got multiple values for keyword argument`
로 죽는다. 실제로 `bias`(무력화)와 `dimensions`(크래시) 두 번 당했으므로,
데이터셋의 `input` 과 짝지어 전수 확인한다.
"""
import importlib.util
import os
import sys

AI8X = os.environ.get("AI8X_DIR") or os.path.expanduser("~/ai8x-training")
for _i, _a in enumerate(sys.argv):
    if _a == "--ai8x" and _i + 1 < len(sys.argv):
        AI8X = sys.argv[_i + 1]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [AI8X, os.path.join(REPO, "datasets")]

import ai8x  # noqa: E402

ai8x.set_device(85, False, False)


def load(fname):
    spec = importlib.util.spec_from_file_location(
        "m_" + fname.replace("-", "_").replace(".py", ""),
        os.path.join(REPO, "models", fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# (모델 파일, 진입점, 그 모델을 쓰는 데이터셋의 input)
CASES = [
    ("ai85net-safesound.py", "ai85safesoundnet", (128, 128)),
    ("ai85net-safesound.py", "ai85safesoundnet_bias", (128, 128)),
    ("ai85net-safesound.py", "ai85safesoundnet_fb", (128, 128)),
    ("ai85net-safesound.py", "ai85safesoundnet_fb_relu", (128, 128)),
    ("ai85net-safesound.py", "ai85safesoundnet_fb_k2", (128, 128)),
    ("ai85net-safesound.py", "ai85safesoundnet_fb_k4", (128, 128)),
    # 구성 A — 입력이 (멜 64, 프레임 64) 다
    ("ai85net-safesound.py", "ai85safesoundnet_mel1d", (64, 64)),
    ("ai85net-safesound.py", "ai85safesoundnet_w025", (128, 128)),
    ("ai85net-safesound.py", "ai85safesoundnet_w050", (128, 128)),
    ("ai85net-safesound.py", "ai85safesoundnet_w150", (128, 128)),
    ("ai85net-safesound-mel.py", "ai85safesoundmelnet", (1, 64, 64)),
    ("ai85net-safesound-mel.py", "ai85safesoundmelnet_w050", (1, 64, 64)),
    ("ai85net-safesound-mel.py", "ai85safesoundmelnet_w150", (1, 64, 64)),
    ("ai85net-safesound-mel.py", "ai85safesoundwave2dnet", (1, 128, 128)),
    ("ai85net-safesound-mel.py", "ai85safesoundwave2dfoldnet", (4, 64, 64)),
    ("ai85net-safesound-mel.py", "ai85safesoundmelteacher", (128, 128)),
    # (a) NPU 로그 근사 — 5시드를 돌도록 검사 대상이 아니었다 (2026-10-01)
    ("ai85net-safesound-logstage.py", "ai85safesoundlogstage", (128, 128)),
    ("ai85net-safesound-logstage.py", "ai85safesoundlogstage_k2", (128, 128)),
    # (c) 학습용 단일 모델 (중간에 CPU 로그가 낀 한 덩어리)
    ("ai85net-safesound-cstage-train.py", "ai85safesoundcstage", (128, 128)),
    # (c) 2패스 — 합성 확인용 조각
    ("ai85net-safesound-cstage.py", "ai85safesoundcstage_p1", (128, 128)),
    ("ai85net-safesound-cstage.py", "ai85safesoundcstage_p2", (100, 128)),
]

# ── 자동 발견 — **CASES 에 빠진 진입점을 잡는다** ───────────────────
# 하드코딩 목록만 검사하면 **새 모델 파일이 조용히 빠진다** (2026-10-01,
# ai85net-safesound-cstage.py 가 그랬다). 검사 도구가 빠뜨리는 것을 모르면
# 검사가 아니므로, 레포의 models/*.py 전부에서 `models` 목록을 긁어
# CASES 와 대조한다.
_MODELS_DIR = os.path.join(REPO, "models")
_declared = {}
for _f in sorted(os.listdir(_MODELS_DIR)):
    if not _f.endswith(".py"):
        continue
    try:
        _m = load(_f)
    except Exception as e:                                  # noqa: BLE001
        print(f"⚠️ {_f}: 로드 실패 — {type(e).__name__}: {e}")
        continue
    for _d in getattr(_m, "models", []):
        _declared[_d["name"]] = _f
_covered = {ep for _, ep, _ in CASES}
_missing = sorted(set(_declared) - _covered)
if _missing:
    print("⚠️ CASES 에 없는 진입점 — 아래를 추가할 것:")
    for _ep in _missing:
        print(f"     (\"{_declared[_ep]}\", \"{_ep}\", (?, ?)),")
    print()

# ── ★ train.py 가 models/ **전체**를 import 할 수 있는가 ───────────
# `train.py` 는 `pydoc.locate` 로 models/ 의 모든 파일을 import 한다.
# 따라서 **파일 하나가 import 에서 터지면 그 모델과 무관한 학습까지
# 전부 죽는다.** 실제로 그렇게 chain15 의 마지막 시드와 chain16 의 첫
# 시드가 죽었다 (2026-10-01, 경계 파일 경로를 `abspath` 로 잡아 심링크를
# 못 푼 것). **import 시점 부작용을 두지 말 것.**
#
# ⚠️ 반드시 **ai8x-training 쪽 경로**(심링크)로 import 해야 한다.
#    레포 경로로만 보면 심링크 때문에 생기는 문제를 놓친다.
_AI8X = os.path.expanduser("~/ai8x-training")
if os.path.isdir(os.path.join(_AI8X, "models")):
    import pydoc                                    # noqa: PLC0415
    _cwd = os.getcwd()
    _sp = list(sys.path)
    os.chdir(_AI8X)
    sys.path.insert(0, _AI8X)
    _bad = []
    for _f in sorted(os.listdir(os.path.join(_AI8X, "models"))):
        if not _f.endswith(".py") or _f == "__init__.py":
            continue
        try:
            pydoc.safeimport("models." + _f[:-3])
        except Exception as e:                      # noqa: BLE001
            _bad.append((_f, type(e).__name__, str(e)[:100]))
    os.chdir(_cwd)
    sys.path[:] = _sp
    if _bad:
        print(f"✘ models/ import 실패 {len(_bad)}건 — "
              "**이 파일들이 모든 학습을 죽인다**:")
        for _f, _t, _m in _bad:
            print(f"     {_f}: {_t} {_m}")
        sys.exit(1)
    print("✔ models/ 전체 import 가능 (심링크 경로 기준)")
    print()

print(f"{'진입점':<30}{'입력':<16}{'params':>10}  결과")
print("-" * 72)
bad = []
for fname, ep, dims in CASES:
    m = load(fname)
    registered = {d["name"] for d in getattr(m, "models", [])}
    if ep not in registered:
        bad.append(f"{ep}: models 리스트 미등록 (train.py --model 선택지에 없다)")
        print(f"{ep:<30}{str(dims):<16}{'—':>10}  ✘ models 미등록")
        continue
    fn = getattr(m, ep, None)
    if fn is None:
        bad.append(f"{ep}: 진입점 없음")
        print(f"{ep:<30}{str(dims):<16}{'—':>10}  ✘ 진입점 없음")
        continue
    # train.py 와 **똑같이** 만든다
    if len(dims) == 2:                      # 1D: input 이 (채널, 길이)
        num_channels, dimensions = dims[0], (dims[1], 1)
    else:                                   # 2D: (채널, 높이, 너비)
        num_channels, dimensions = dims[0], (dims[1], dims[2])
    args = dict(pretrained=False, num_classes=5, num_channels=num_channels,
                dimensions=dimensions, bias=False,
                weight_bits=None, bias_bits=None, quantize_activation=False)
    try:
        net = fn(**args)
        n = sum(p.numel() for p in net.parameters())
        print(f"{ep:<30}{str(dims):<16}{n:>10,}  ✔")
    except Exception as e:                  # noqa: BLE001
        bad.append(f"{ep}: {type(e).__name__}: {e}")
        print(f"{ep:<30}{str(dims):<16}{'—':>10}  ✘ {type(e).__name__}")
        print(f"    {str(e)[:110]}")

print()
if bad:
    print(f"실패 {len(bad)}건:")
    for b in bad:
        print("  " + b)
    sys.exit(1)
print("전부 통과 — train.py 인자 규약과 충돌하는 진입점 없음")
