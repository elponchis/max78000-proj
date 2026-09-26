#!/usr/bin/env python3
"""학습 로그 → 에폭별 곡선(PNG) + QAT 전환 전후 요약.

distiller 의 `PythonLogger` 가 남기는 줄을 읽는다. 형식은 둘이다.

    Epoch: [12][  100/  102]    Overall Loss 0.412    Objective Loss 0.412    Top1 78.9 ...
    Epoch: [12][    0/    1]    Loss 0.501    Top1 71.2

앞의 것은 **학습** 진행(에폭 내 러닝 평균이라 그 에폭의 마지막 줄이 에폭 평균),
뒤의 `[0/1]` 은 **검증** 요약이다. 이 구분이 파서의 전부다.

QAT 가 시작되는 에폭에서 정확도가 한 번 떨어졌다가 회복하는 것이 정상이다.
회복하지 못하면 QAT 정책(시작 에폭·학습률 감쇠 시점)을 다시 봐야 하므로,
전환 전후 값을 따로 요약한다.

사용법 (Colab 또는 WSL2):
    python3 tools/parse_trainlog.py --log logs/safesound-v1___*/safesound-v1*.log \\
        --qat-epoch 60 --png /content/drive/MyDrive/max78000/curves.png
    python3 tools/parse_trainlog.py --self-test
"""

import argparse
import glob
import os
import re
import sys

EPOCH_RE = re.compile(r"Epoch:\s*\[(\d+)\]\[\s*(\d+)/\s*(\d+)\]\s+(.*)")
# 키는 공백을 포함할 수 있다 (Overall Loss, Objective Loss)
KV_RE = re.compile(r"([A-Za-z][A-Za-z0-9 ]*?)\s+(-?\d+\.?\d*(?:e[-+]?\d+)?)(?=\s{2,}|\s*$)")


def parse(path):
    """로그 → {epoch: {train_top1, train_loss, val_top1, val_loss}}"""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if not m:
                continue
            ep, done, total = int(m.group(1)), int(m.group(2)), int(m.group(3))
            kv = {k.strip(): float(v) for k, v in KV_RE.findall(m.group(4))}
            rec = out.setdefault(ep, {})
            if total == 1:                                   # 검증 요약 줄
                if "Top1" in kv:
                    rec["val_top1"] = kv["Top1"]
                if "Loss" in kv:
                    rec["val_loss"] = kv["Loss"]
            else:                                            # 학습 진행 줄
                if "Top1" in kv:
                    rec["train_top1"] = kv["Top1"]           # 마지막 줄이 에폭 평균
                for key in ("Objective Loss", "Overall Loss", "Loss"):
                    if key in kv:
                        rec["train_loss"] = kv[key]
                        break
    return out


def summarize(ep, qat, span=3):
    """QAT 시작 전후 요약."""
    before = [e for e in ep if e < qat]
    after = [e for e in ep if e >= qat]
    lines = []
    if not after:
        lines.append(f"  QAT 시작 에폭({qat})에 도달하지 않았다 — 마지막 에폭 {max(ep)}")
        return lines
    b = sorted(before)[-span:]
    a = sorted(after)[:span]
    lines.append(f"  {'구간':<22}{'train Top1':>12}{'val Top1':>11}{'val Loss':>11}")
    lines.append("  " + "-" * 55)
    for label, eps in ((f"QAT 직전 {span}에폭", b), (f"QAT 직후 {span}에폭", a)):
        if not eps:
            continue
        def avg(key):
            v = [ep[e][key] for e in eps if key in ep[e]]
            return sum(v) / len(v) if v else float("nan")
        lines.append(f"  {label + ' ' + str(eps):<22}"
                     f"{avg('train_top1'):>11.2f}%{avg('val_top1'):>10.2f}%"
                     f"{avg('val_loss'):>11.4f}")
    best_pre = max((ep[e].get("val_top1", -1) for e in before), default=-1)
    best_post = max((ep[e].get("val_top1", -1) for e in after), default=-1)
    lines.append("")
    lines.append(f"  최고 val Top1 — QAT 전 {best_pre:.2f}% / QAT 후 {best_post:.2f}%")
    if best_post < best_pre - 1.0:
        lines.append("  ⚠️ QAT 후에도 회복하지 못했다. 시작 에폭을 뒤로 미루거나")
        lines.append("     전환 직후 학습률 감쇠 시점을 조정할 것 (colab/*.yaml).")
    return lines


def self_test():
    """합성 로그로 파서를 검증한다 — 실제 로그 없이 형식 대응을 확인한다."""
    import tempfile
    sample = []
    for e in range(4):
        sample.append(f"2026-09-26 10:00:00 - Epoch: [{e}][   50/  102]    "
                      f"Overall Loss {1.0-e*0.1:.6f}    Objective Loss {1.0-e*0.1:.6f}    "
                      f"Top1 {50+e*5:.6f}    LR 0.001000    ")
        sample.append(f"2026-09-26 10:00:10 - Epoch: [{e}][  102/  102]    "
                      f"Overall Loss {0.9-e*0.1:.6f}    Objective Loss {0.9-e*0.1:.6f}    "
                      f"Top1 {52+e*5:.6f}    LR 0.001000    ")
        sample.append(f"2026-09-26 10:00:11 - Epoch: [{e}][    0/    1]    "
                      f"Loss {0.8-e*0.05:.6f}    Top1 {60+e*4:.6f}    ")
    p = os.path.join(tempfile.mkdtemp(), "t.log")
    open(p, "w", encoding="utf-8").write("\n".join(sample))
    ep = parse(p)
    ok = True
    checks = [
        ("에폭 4개 파싱", len(ep) == 4),
        ("train Top1 은 에폭 마지막 줄", abs(ep[0]["train_top1"] - 52.0) < 1e-6),
        ("val Top1 분리", abs(ep[0]["val_top1"] - 60.0) < 1e-6),
        ("train/val loss 분리", abs(ep[0]["train_loss"] - 0.9) < 1e-6
         and abs(ep[0]["val_loss"] - 0.8) < 1e-6),
    ]
    for name, cond in checks:
        print(f"  [{'통과' if cond else '실패'}] {name}")
        ok &= cond
    print("\n" + "\n".join(summarize(ep, qat=2, span=2)))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", help="로그 파일 또는 글롭 (가장 최근 것을 쓴다)")
    ap.add_argument("--qat-epoch", type=int, default=60)
    ap.add_argument("--png")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        self_test()
    if not a.log:
        sys.exit("[에러] --log 가 필요하다 (또는 --self-test)")

    paths = sorted(glob.glob(a.log), key=os.path.getmtime)
    if not paths:
        sys.exit(f"[에러] 로그가 없다: {a.log}")
    path = paths[-1]
    print(f"로그: {path}")

    ep = parse(path)
    if not ep:
        sys.exit("[에러] 에폭을 하나도 읽지 못했다 — 로그 형식이 바뀌었을 수 있다. "
                 "`--self-test` 로 파서를 먼저 확인할 것.")
    print(f"에폭 {len(ep)}개 ({min(ep)}~{max(ep)})\n")

    print(f"{'epoch':>6}{'train Top1':>12}{'val Top1':>11}{'train Loss':>12}{'val Loss':>11}")
    print("-" * 52)
    for e in sorted(ep):
        r = ep[e]
        mark = "  ← QAT 시작" if e == a.qat_epoch else ""
        print(f"{e:>6}{r.get('train_top1', float('nan')):>11.2f}%"
              f"{r.get('val_top1', float('nan')):>10.2f}%"
              f"{r.get('train_loss', float('nan')):>12.4f}"
              f"{r.get('val_loss', float('nan')):>11.4f}{mark}")

    print(f"\n=== QAT 전환 전후 (시작 에폭 {a.qat_epoch}) ===")
    print("\n".join(summarize(ep, a.qat_epoch)))

    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        es = sorted(ep)
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].plot(es, [ep[e].get("train_top1", float("nan")) for e in es],
                   label="train Top1")
        ax[0].plot(es, [ep[e].get("val_top1", float("nan")) for e in es],
                   label="val Top1")
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("Top1 (%)")
        ax[0].set_title("정확도")
        ax[1].plot(es, [ep[e].get("train_loss", float("nan")) for e in es],
                   label="train loss")
        ax[1].plot(es, [ep[e].get("val_loss", float("nan")) for e in es],
                   label="val loss")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("loss")
        ax[1].set_title("손실")
        for x in ax:
            x.axvline(a.qat_epoch, ls="--", color="gray", lw=1)
            x.grid(alpha=.3)
            x.legend()
        fig.suptitle(f"{os.path.basename(path)} — 점선은 QAT 시작({a.qat_epoch})")
        fig.tight_layout()
        os.makedirs(os.path.dirname(a.png) or ".", exist_ok=True)
        fig.savefig(a.png, dpi=120)
        print(f"\n그림: {a.png}")


if __name__ == "__main__":
    main()
