#!/usr/bin/env python3
"""평가 경로 회귀 테스트 — **우리 로더 == `train.py --evaluate`** 인가.

2026-09-30 사고: `tools/eval_confusion.py` 의 `load_model()` 이
`ai8x.update_model()` 을 부르지 않아, QAT 체크포인트를 **fake-quantization
없이 순수 float 로** 평가했다. 고정 오경보 macro-F1 이 ④에서 0.385 → 0.521
로 바뀔 만큼 큰 오류였는데 **단위 테스트로는 잡히지 않는다** — 값이
그럴듯하고 시드 간 일관성도 있었다.

잡을 방법은 하나뿐이다: **독립 경로(train.py)와 대조**.

이 테스트는 고정 체크포인트에 대해
  (1) `train.py --evaluate` 를 돌려 혼동행렬을 얻고
  (2) 거기서 argmax macro-F1 을 계산한 뒤
  (3) 우리 `collect_logits` 경로의 값과 비교한다.
소수점 4자리까지 같아야 통과다.

사용 (WSL2, ai8x venv):
    ~/ai8x-training/venv/bin/python tools/kat_evalpath.py
    ~/ai8x-training/venv/bin/python tools/kat_evalpath.py --quick   # ④만
"""

import argparse
import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI8X = os.environ.get("AI8X_DIR") or os.path.expanduser("~/ai8x-training")
sys.path[:0] = [AI8X, os.path.join(REPO, "datasets"), os.path.join(REPO, "tools")]

CLASSES = ["siren", "glass", "scream", "dog_bark", "background"]

# (이름, --model, --dataset, 우리 --config, 체크포인트)
# 체크포인트는 **고정**이다. 새 실험으로 갈아 끼우지 말 것 — 기준점이 움직이면
# 회귀 테스트가 아니다.
CASES = [
    ("④ 기준선", "ai85safesoundnet", "SafeSound", "wave",
     "data/logs-local/safesound-v1cpu___2026.09.27-223329/"
     "safesound-v1cpu_qat_best.pth.tar"),
    ("D-1", "ai85safesoundnet_fb", "SafeSound", "wave_fb",
     "data/logs-local/safesound-fbabs-v1___2026.09.27-204346/"
     "safesound-fbabs-v1_qat_best.pth.tar"),
    ("① 로그 멜", "ai85safesoundmelnet", "SafeSoundMel", "mel",
     "data/safesound-mel-v1_qat_best.pth.tar"),
    # 전처리가 **데이터셋 인자**로 들어가는 구성 (2026-10-02 추가). 위 셋은
    # 전부 기본 로더라, 우리 평가 경로가 정규화를 빼먹어도 잡지 못했다.
    ("D-1 정규화", "ai85safesoundnet_fb", "SafeSoundNorm2", "wave_fb_norm2",
     "data/logs-local/safesound-fbnorm2-v1___2026.09.30-001903/"
     "safesound-fbnorm2-v1_qat_best.pth.tar"),
]

CONF_RE = re.compile(r"==> Confusion:\s*\n((?:\s*\[.*\]\s*\n?)+)")


def macro_f1_from_conf(m):
    m = np.asarray(m, dtype=np.float64)
    f = []
    for c in range(m.shape[0]):
        tp = m[c, c]
        fp = m[:, c].sum() - tp
        fn = m[c, :].sum() - tp
        f.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan)
    return float(np.nanmean(f))


def from_train_py(model, dataset, ckpt, threads=3):
    """train.py --evaluate 를 돌려 혼동행렬을 뽑는다."""
    out = os.path.join(tempfile.gettempdir(), "kat_evalpath")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    cmd = [os.path.join(AI8X, "venv", "bin", "python"),
           os.path.join(AI8X, "train.py"),
           "--model", model, "--dataset", dataset, "--device", "MAX78000",
           "--cpu", "--workers", "2", "--batch-size", "128", "--confusion",
           "--evaluate", "--exp-load-weights-from",
           os.path.join(REPO, ckpt),
           "--qat-policy", os.path.join(REPO, "colab",
                                        "qat_policy_safesound.yaml"),
           "--out-dir", out, "--name", "katpath"]
    # ⚠️ distiller 의 로거가 stderr 로도 쓴다 — 합쳐서 봐야 한다
    r = subprocess.run(cmd, cwd=AI8X, env=env, text=True,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    m = CONF_RE.search(r.stdout)
    if not m:
        print(r.stdout[-2500:])
        raise SystemExit("[에러] train.py 출력에서 혼동행렬을 못 찾았다")
    rows = []
    for ln in m.group(1).strip().splitlines():
        rows.append([int(x) for x in re.findall(r"-?\d+", ln)])
    return rows


def from_our_loader(ckpt, config):
    import eval_confusion as EC
    a = argparse.Namespace(
        checkpoint=os.path.join(REPO, ckpt), config=config,
        data=os.path.join(AI8X, "data"), ai8x=AI8X, simulate=False,
        bias=False, batch_size=128, split="test")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _fs, _cl, _st, y, lg = EC.collect_logits(a, CLASSES, "test")
    return EC.macro_f1(y, lg.argmax(1), len(CLASSES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="④만")
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()

    cases = CASES[:1] if a.quick else CASES
    print("평가 경로 회귀 테스트 — 우리 로더 vs train.py --evaluate")
    print(f"{'계열':<14}{'train.py':>11}{'우리 로더':>12}{'차이':>10}  결과")
    print("-" * 56)
    bad = []
    for name, model, dataset, config, ckpt in cases:
        if not os.path.isfile(os.path.join(REPO, ckpt)):
            print(f"{name:<14}  체크포인트 없음 — 건너뜀 ({ckpt})")
            continue
        ref = macro_f1_from_conf(from_train_py(model, dataset, ckpt))
        got = from_our_loader(ckpt, config)
        ok = abs(ref - got) <= a.tol
        bad += [] if ok else [name]
        print(f"{name:<14}{ref:>11.4f}{got:>12.4f}{abs(ref-got):>10.5f}"
              f"  {'OK' if ok else '**불일치**'}")

    print()
    if bad:
        print(f"실패 {len(bad)}건: {', '.join(bad)}")
        print("  → load_model() 이 train.py 의 로드 절차를 따르는지 볼 것.")
        print("     특히 `ai8x.update_model(model)` (train.py:415).")
        sys.exit(1)
    print("전부 통과 — 평가 경로가 학습 경로와 같다")


if __name__ == "__main__":
    main()
