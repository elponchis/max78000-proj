#!/usr/bin/env python3
"""학습 로그 → 에폭별 곡선(PNG) + QAT 전환 전후 요약.

distiller 의 `PythonLogger` 가 남기는 줄을 읽는다. 형식은 둘이다.

    Epoch: [0][   92/   92]    Overall Loss 1.507    Objective Loss 1.507    Top1 15.5    LR ...
    Epoch: [0][   11/   11]    Loss 1.419    Top1 11.769
    ==> Top1: 11.769    Loss: 1.420
    Test:  [   41/   41]    Loss 1.316    Top1 13.693

1행이 **학습**(에폭 내 러닝 평균이라 그 에폭의 마지막 줄이 에폭 값), 2행이
**검증**, 3행이 그 에폭의 검증 요약, 4행은 학습이 끝난 뒤의 **최종 테스트**다.

⚠️ 학습과 검증을 가르는 것은 **`Objective Loss` 키의 유무**다. 분모(`/11`)는
검증 배치 수이지 1이 아니다 — 처음에 `[0/1]` 로 짐작해 파서를 썼다가 val 이
전부 nan 으로 나왔다. 실제 로그로 확인하고 고친 것이다.

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


SUMMARY_RE = re.compile(r"==>\s*Top1:\s*(-?[\d.]+)\s+Loss:\s*(-?[\d.]+)")


def parse(path):
    """로그 → ({epoch: {train_top1, train_loss, val_top1, val_loss}}, 최종 테스트)."""
    out = {}
    final_test = {}
    ctx = None          # 직전에 본 줄의 성격: train / val / test
    cur_ep = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "Test: [" in line:
                ctx = "test"
                kv = {k.strip(): float(v)
                      for k, v in KV_RE.findall(line.split("Test:", 1)[1])}
                if "Top1" in kv:
                    final_test["top1"] = kv["Top1"]      # 마지막 줄이 최종값
                if "Loss" in kv:
                    final_test["loss"] = kv["Loss"]
                continue

            m = EPOCH_RE.search(line)
            if m:
                cur_ep = int(m.group(1))
                kv = {k.strip(): float(v) for k, v in KV_RE.findall(m.group(4))}
                rec = out.setdefault(cur_ep, {})
                # 학습 줄에만 Objective/Overall Loss 가 있다 — 이것이 판별자다
                if "Objective Loss" in kv or "Overall Loss" in kv:
                    ctx = "train"
                    if "Top1" in kv:
                        rec["train_top1"] = kv["Top1"]   # 마지막 줄 = 에폭 평균
                    rec["train_loss"] = kv.get("Objective Loss",
                                               kv.get("Overall Loss"))
                    if "LR" in kv:
                        rec["lr"] = kv["LR"]
                else:
                    ctx = "val"
                    if "Top1" in kv:
                        rec["val_top1"] = kv["Top1"]
                    if "Loss" in kv:
                        rec["val_loss"] = kv["Loss"]
                continue

            s = SUMMARY_RE.search(line)
            if s and ctx == "val" and cur_ep is not None:
                # `==> Top1: ...` 는 직전 구간의 요약이다. 검증 직후일 때만 받는다
                out[cur_ep]["val_top1"] = float(s.group(1))
                out[cur_ep]["val_loss"] = float(s.group(2))
            elif s and ctx == "test":
                final_test["top1"] = float(s.group(1))
                final_test["loss"] = float(s.group(2))
    return out, final_test


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
    # 실제 ai8x 로그 형식 그대로 (검증 줄의 분모는 배치 수다)
    sample = []
    for e in range(4):
        sample.append(f"2026-09-26 10:00:00 - Epoch: [{e}][   50/   92]    "
                      f"Overall Loss {1.0-e*0.1:.6f}    Objective Loss {1.0-e*0.1:.6f}    "
                      f"Top1 {50+e*5:.6f}    LR 0.001000    Time 0.066453    ")
        sample.append(f"2026-09-26 10:00:10 - Epoch: [{e}][   92/   92]    "
                      f"Overall Loss {0.9-e*0.1:.6f}    Objective Loss {0.9-e*0.1:.6f}    "
                      f"Top1 {52+e*5:.6f}    LR 0.001000    Time 0.066453    ")
        sample.append(f"2026-09-26 10:00:11 - Epoch: [{e}][   10/   11]    "
                      f"Loss {0.85-e*0.05:.6f}    Top1 {59+e*4:.6f}    ")
        sample.append(f"2026-09-26 10:00:11 - Epoch: [{e}][   11/   11]    "
                      f"Loss {0.8-e*0.05:.6f}    Top1 {60+e*4:.6f}    ")
        sample.append(f"2026-09-26 10:00:11 - ==> Top1: {60+e*4:.3f}    "
                      f"Loss: {0.8-e*0.05:.3f}")
    sample.append("2026-09-26 10:01:00 - Test: [   41/   41]    "
                  "Loss 1.316636    Top1 13.693105    ")
    sample.append("2026-09-26 10:01:00 - ==> Top1: 13.693    Loss: 1.317")
    p = os.path.join(tempfile.mkdtemp(), "t.log")
    open(p, "w", encoding="utf-8").write("\n".join(sample))
    ep, final = parse(p)
    ok = True
    checks = [
        ("에폭 4개 파싱", len(ep) == 4),
        ("train Top1 은 에폭 마지막 줄", abs(ep[0]["train_top1"] - 52.0) < 1e-6),
        ("val Top1 분리 (분모가 1이 아니어도)", abs(ep[0]["val_top1"] - 60.0) < 1e-6),
        ("train/val loss 분리", abs(ep[0]["train_loss"] - 0.9) < 1e-6
         and abs(ep[0]["val_loss"] - 0.8) < 1e-6),
        ("최종 Test 를 검증과 섞지 않음",
         abs(final.get("top1", 0) - 13.693) < 1e-3
         and abs(ep[3]["val_top1"] - 72.0) < 1e-6),
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
    ap.add_argument("--every", type=int, default=10,
                    help="에폭 표를 이 간격으로만 찍는다 (경계 에폭은 항상 표시). "
                         "1 이면 전량")
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

    ep, final_test = parse(path)
    if not ep:
        sys.exit("[에러] 에폭을 하나도 읽지 못했다 — 로그 형식이 바뀌었을 수 있다. "
                 "`--self-test` 로 파서를 먼저 확인할 것.")
    print(f"에폭 {len(ep)}개 ({min(ep)}~{max(ep)})\n")

    es = sorted(ep)
    # 150에폭을 전부 찍으면 읽히지 않는다. `--every` 간격 + 경계 에폭만 남긴다
    # (QAT 시작 전후, 최고 val, 처음·마지막). `--every 1` 로 전량 출력.
    best_e = max(es, key=lambda e: ep[e].get("val_top1", -1))
    keep = {es[0], es[-1], best_e, a.qat_epoch, a.qat_epoch - 1,
            a.qat_epoch + 1}
    show = [e for e in es if e % max(a.every, 1) == 0 or e in keep]
    print(f"{'epoch':>6}{'train Top1':>12}{'val Top1':>11}{'차이':>8}"
          f"{'train Loss':>12}{'val Loss':>11}")
    print("-" * 60)
    prev = None
    for e in show:
        r = ep[e]
        if prev is not None and e - prev > 1:
            print(f"{'  ⋮':>6}")
        prev = e
        gap = r.get("train_top1", float("nan")) - r.get("val_top1", float("nan"))
        mark = ""
        if e == a.qat_epoch:
            mark = "  ← QAT 시작"
        elif e == best_e:
            mark = "  ← 최고 val"
        print(f"{e:>6}{r.get('train_top1', float('nan')):>11.2f}%"
              f"{r.get('val_top1', float('nan')):>10.2f}%{gap:>+8.1f}"
              f"{r.get('train_loss', float('nan')):>12.4f}"
              f"{r.get('val_loss', float('nan')):>11.4f}{mark}")
    if len(show) < len(es):
        print(f"  ({len(es)}에폭 중 {len(show)}개만 표시 — 전량은 `--every 1`)")

    # ── train/val Top1 요약.
    #
    # ⚠️ **이 값으로 과소적합을 판정하지 않는다.** 두 가지 때문에 낮게 나오는 것이
    #    정상이다.
    #      · 손실이 **역빈도 가중**이다 (background 0.27). 모델이 배경음을 의도적으로
    #        희생하도록 학습되는데, Top1 은 **비가중** 정확도라 전체 창의 71%인
    #        배경음에서 잃은 만큼 그대로 낮아진다. 설정대로 동작해도 낮다
    #      · train Top1 은 **증강이 걸린 데이터** 기준이다 (shift·게인 ±12dB).
    #        평가와 같은 조건이 아니다
    #      · 게다가 **마지막 2배치에서만** 잰 값이다. ai8x 의
    #        `--show-train-accuracy` 기본값이 `last_batch` 라
    #        (`parsecmd.py:187`, `train.py:940`), 에폭 평균이 아니라 256샘플
    #        표본이다. 에폭마다 크게 흔들리는 것이 정상이다
    #    판정은 `tools/eval_confusion.py --split train`(무증강)의 **클래스별
    #    재현율**을 test 와 나란히 놓고 한다. train 재현율도 낮으면 용량·최적화
    #    문제이고, train 만 높으면 일반화 문제다.
    tt = [ep[e]["train_top1"] for e in es if "train_top1" in ep[e]]
    vv = [ep[e]["val_top1"] for e in es if "val_top1" in ep[e]]
    if tt:
        print("\n=== Top1 요약 (비가중·train 은 증강 포함) ===")
        print(f"  train Top1  최고 {max(tt):.2f}%  마지막 {tt[-1]:.2f}%")
        if vv:
            print(f"  val   Top1  최고 {max(vv):.2f}%  마지막 {vv[-1]:.2f}%")
            print(f"  마지막 에폭의 train−val 차이 {tt[-1] - vv[-1]:+.1f}p")
        print("  ⚠️ 이 수치로 과소적합을 판정하지 말 것 — 손실이 역빈도 가중이고,")
        print("     train 은 증강이 걸린 마지막 2배치(256샘플)만 잰 값이다.")
        print("     판정: `eval_confusion.py --split train` 의 클래스별 재현율을")
        print("     test 와 비교한다 (노트북 셀 16).")

    n_val = sum(1 for e in ep if "val_top1" in ep[e])
    if n_val == 0:
        print("\n⚠️ 검증 값을 하나도 읽지 못했다. 로그에 `Epoch: [n][a/b]` 형식의")
        print("   검증 줄이 있는지 확인할 것 — 학습 줄과의 차이는 `Objective Loss`")
        print("   키의 유무다. (검증이 실제로 안 돌았다면 --validation-split 0 인지 볼 것)")
    else:
        print(f"\n검증 값을 읽은 에폭: {n_val}/{len(ep)}")
    if final_test:
        print(f"최종 테스트 (학습 후): Top1 {final_test.get('top1', float('nan')):.3f}%  "
              f"Loss {final_test.get('loss', float('nan')):.4f}")

    print(f"\n=== QAT 전환 전후 (시작 에폭 {a.qat_epoch}) ===")
    print("\n".join(summarize(ep, a.qat_epoch)))

    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].plot(es, [ep[e].get("train_top1", float("nan")) for e in es],
                   label="train Top1")
        ax[0].plot(es, [ep[e].get("val_top1", float("nan")) for e in es],
                   label="val Top1")
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("Top1 (%)")
        # ⚠️ 그림 안 글자는 영어다 — matplotlib 기본 폰트(DejaVu Sans)에 한글
        # 글리프가 없어 네모(tofu)로 깨진다. 한글이 필요하면
        # `apt-get install fonts-nanum` 후 rcParams["font.family"]="NanumGothic".
        ax[0].set_title("Accuracy")
        ax[1].plot(es, [ep[e].get("train_loss", float("nan")) for e in es],
                   label="train loss")
        ax[1].plot(es, [ep[e].get("val_loss", float("nan")) for e in es],
                   label="val loss")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("loss")
        ax[1].set_title("Loss")
        for x in ax:
            x.axvline(a.qat_epoch, ls="--", color="gray", lw=1)
            x.grid(alpha=.3)
            x.legend()
        fig.suptitle(f"{os.path.basename(path)} - dashed line = QAT start "
                     f"(epoch {a.qat_epoch})")
        fig.tight_layout()
        os.makedirs(os.path.dirname(a.png) or ".", exist_ok=True)
        fig.savefig(a.png, dpi=120)
        print(f"\n그림: {a.png}")


if __name__ == "__main__":
    main()
