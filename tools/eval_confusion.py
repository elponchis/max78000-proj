#!/usr/bin/env python3
"""테스트셋 평가 — 창 단위 / 원본 단위 혼동행렬 + 원본 단위 부트스트랩 신뢰구간.

ai8x 의 `train.py --confusion` 은 **창 단위 혼동행렬만** 낸다. 그 수치는 두 가지
이유로 논문에 그대로 쓸 수 없다.

1. **창은 독립 표본이 아니다.** 한 원본에서 여러 창이 나오고(US8K 는 한 녹음에서
   슬라이스를, 거기서 또 창을 뜬다), 같은 녹음의 창들은 성공·실패가 함께 간다.
   창 수로 신뢰구간을 내면 실제보다 좁게 나온다 (CLAUDE.md 5장 규칙 1).
2. **배치 단위가 창이 아니다.** 실기기는 초당 4회 추론하고 이벤트 병합
   상태머신이 연속 추론을 하나로 묶는다(6장). 사용자가 겪는 단위는 "이 소리를
   맞혔나"이지 "이 창을 맞혔나"가 아니다.

그래서 두 가지를 함께 낸다.
  · **창 단위** — 모델 출력 그대로. ai8x 로그와 대조 가능
  · **원본 단위** — 같은 `fsid` 의 창들을 다수결로 묶어 원본 하나 = 표본 하나.
    동점은 그 원본의 정답 클래스를 **택하지 않는 쪽**으로 깨어(보수적) 성능을
    과대평가하지 않는다
재현율 95% 신뢰구간은 **원본을 복원추출**하는 부트스트랩으로 낸다 — 창을
재표집하면 같은 이유로 구간이 좁아진다.

사용법 (Colab 또는 WSL2):
    python3 tools/eval_confusion.py --checkpoint logs/safesound-v1/best.pth.tar \\
        --data /content/ai8x-training/data
    python3 tools/eval_confusion.py --self-test      # 통계 경로만 검증 (체크포인트 불필요)
"""

import argparse
import collections
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "datasets"))


# ──────────────────────────────────────────────────────────── 집계
def confusion(y_true, y_pred, k):
    m = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def majority_by_origin(fsids, y_true, y_pred, k):
    """같은 원본의 창들을 다수결로 묶는다.

    동점이면 **정답 클래스를 피해서** 고른다(보수적). 이렇게 하지 않으면 2:2
    동점이 우연히 정답으로 떨어져 원본 단위 성능이 부풀려진다.
    """
    by = collections.defaultdict(list)
    truth = {}
    for f, t, p in zip(fsids, y_true, y_pred):
        by[f].append(p)
        truth[f] = t                      # 한 원본의 정답은 하나다
    out_f, out_t, out_p = [], [], []
    for f, preds in by.items():
        cnt = collections.Counter(preds)
        top = max(cnt.values())
        tied = sorted(c for c, n in cnt.items() if n == top)
        pick = next((c for c in tied if c != truth[f]), tied[0])
        out_f.append(f)
        out_t.append(truth[f])
        out_p.append(pick)
    return np.array(out_f), np.array(out_t), np.array(out_p)


def recalls(y_true, y_pred, k):
    """클래스별 재현율. 표본이 없는 클래스는 nan."""
    out = np.full(k, np.nan)
    for c in range(k):
        sel = y_true == c
        if sel.sum():
            out[c] = float((y_pred[sel] == c).mean())
    return out


def macro_f1(y_true, y_pred, k):
    f1 = []
    for c in range(k):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1)) if f1 else float("nan")


def bootstrap_recall_ci(fsids, y_true, y_pred, k, n_boot=2000, seed=0):
    """원본을 복원추출해 클래스별 재현율의 95% 구간을 낸다.

    **재표집 단위가 원본**인 것이 핵심이다. 창을 재표집하면 같은 녹음의 창들이
    독립인 것처럼 계산되어 구간이 실제보다 좁아진다.
    원본 단위로 묶은 뒤(`majority_by_origin`) 원본을 뽑는다.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(fsids)
    idx_of = {f: np.where(fsids == f)[0] for f in uniq}
    boots = np.full((n_boot, k), np.nan)
    for b in range(n_boot):
        pick = rng.choice(len(uniq), len(uniq), replace=True)
        sel = np.concatenate([idx_of[uniq[i]] for i in pick])
        boots[b] = recalls(y_true[sel], y_pred[sel], k)
    lo = np.nanpercentile(boots, 2.5, axis=0)
    hi = np.nanpercentile(boots, 97.5, axis=0)
    return lo, hi


# ──────────────────────────────────────────────────────────── 출력
def false_alarm_report(clips, starts, y_true, y_pred, names, hop_ms, ns=(1, 2, 3)):
    """배경음 창의 오탐률과 **시간당 오경보 횟수**.

    상시 동작 시스템에서 사용자가 겪는 것은 정확도가 아니라 **오경보 빈도**다.
    hop 250ms 면 초당 4회, 시간당 14,400회 추론하므로 창 단위 오탐률 0.1% 라도
    시간당 14회가 울린다. G7(무인 구동)의 실질 기준이라 따로 낸다.

    N프레임 다수결은 펌웨어의 이벤트 병합 상태머신(CLAUDE.md 6장)을 근사한다.
    **엄격 다수**(> N/2)를 요구한다 — N=2 는 2창 모두, N=3 은 2창 이상이
    같은 이벤트여야 울린다.

    ⚠️ **근사의 한계를 같이 출력한다.** 테스트셋의 창은 클립당 최대 3개를
    점수 순으로 고른 것이라 시간적으로 인접하지 않는다(실제 hop 은 4,000샘플,
    여기 간격은 그보다 훨씬 크다). 실기기에서 연속 프레임은 0.75초를 겹쳐 서로
    강하게 상관되므로, 여기서 구한 N프레임 감소폭은 **낙관적으로도 비관적으로도**
    실기기와 다를 수 있다. 확정 수치는 보드에서 8시간 무인 구동으로 잰다.
    """
    bg = len(names) - 1                       # background 는 마지막 클래스
    sel = y_true == bg
    n_bg = int(sel.sum())
    if not n_bg:
        print("\n[오탐률] 배경음 창이 없다 — 건너뜀")
        return
    per_hour = 3600.0 / (hop_ms / 1000.0)     # hop 250ms → 14,400
    fp = y_pred[sel] != bg

    print(f"\n=== 배경음 오탐률 / 시간당 오경보 (hop {hop_ms}ms → 시간당 "
          f"{per_hour:,.0f}회 추론) ===")
    print(f"  배경음 창 {n_bg}개 중 이벤트로 분류 {int(fp.sum())}개 "
          f"= **{fp.mean():.3%}**  →  시간당 **{fp.mean() * per_hour:,.1f}회**")
    print(f"\n  {'오경보 클래스':<14}{'창':>7}{'비율':>9}{'시간당':>10}")
    print("  " + "-" * 40)
    for c in range(bg):
        n_c = int((y_pred[sel] == c).sum())
        if n_c:
            r = n_c / n_bg
            print(f"  {names[c]:<14}{n_c:>7}{r:>8.3%}{r * per_hour:>9.1f}회")

    # ── N프레임 다수결
    order = collections.defaultdict(list)
    for i in np.where(sel)[0]:
        order[clips[i]].append((starts[i], int(y_pred[i])))
    gaps = []
    for v in order.values():
        v.sort()
        gaps += [b[0] - a[0] for a, b in zip(v, v[1:])]

    print(f"\n  {'N':>3}{'판정 지점':>10}{'오경보':>8}{'비율':>10}{'시간당':>10}")
    print("  " + "-" * 42)
    for n in ns:
        alarms = points = 0
        for v in order.values():
            seq = [p for _s, p in sorted(v)]
            for i in range(len(seq) - n + 1):
                points += 1
                cnt = collections.Counter(seq[i:i + n])
                lab, top = cnt.most_common(1)[0]
                if top > n / 2 and lab != bg:
                    alarms += 1
        if points:
            r = alarms / points
            print(f"  {n:>3}{points:>10}{alarms:>8}{r:>9.3%}{r * per_hour:>9.1f}회")
        else:
            print(f"  {n:>3}{points:>10}{'—':>8}{'—':>10}{'—':>10}  (창 부족)")

    if gaps:
        g = np.array(gaps)
        adjacent = float((g == int(16000 * hop_ms / 1000)).mean())
        print(f"\n  ⚠️ **이 N프레임 수치는 하한(낙관치)이며 논문에 쓰지 않는다.**"
              f"\n     같은 클립에서 고른 창들의 간격이 중앙값"
              f" {np.median(g):,.0f}샘플({np.median(g)/16000:.1f}초)이고, 실제 hop"
              f" {int(16000*hop_ms/1000):,}샘플과"
              f"\n     붙어 있는 쌍은 {adjacent:.1%} 뿐이다. 떨어진 창들의 오류는 서로"
              f" 독립에 가까워"
              f"\n     다수결이 실제보다 잘 듣는다 — 실기기의 연속 프레임은 0.75초를"
              f" 겹쳐 오류가 함께"
              f"\n     가므로 감소폭이 반드시 더 작다. **항상 낙관적으로 틀린다.**"
              f"\n     논문에 쓸 수치는 `tools/stream_eval.py`(빈틈없는 스트리밍)와"
              f"\n     보드 8시간 무인 구동에서 얻는다."
              f"\n     N=3 이 N=2 보다 나쁜 식의 **역전이 보이면 그 자체가 증상**이다 —"
              f"\n     연속 프레임이라면 N 이 커질수록 단조 감소해야 한다.")


def print_matrix(m, names, title):
    print(f"\n{title}")
    w = max(9, max(len(n) for n in names) + 1)
    print(" " * w + "".join(f"{n[:8]:>9}" for n in names) + f"{'합계':>8}{'재현율':>9}")
    for i, n in enumerate(names):
        row = m[i]
        tot = row.sum()
        rec = row[i] / tot if tot else float("nan")
        print(f"{n:<{w}}" + "".join(f"{v:>9d}" for v in row)
              + f"{tot:>8d}{rec:>8.1%}")
    print(" " * w + "".join(f"{m[:, j].sum():>9d}" for j in range(len(names))))
    print(" " * w + "(열 = 예측)")


def summary_json(fsids, y_true, y_pred, names, hop_ms=250):
    """체크포인트 간 비교용 요약 — float best vs qat_best 를 손으로 대조하지 않게."""
    k = len(names)
    bg = k - 1
    of, ot, op = majority_by_origin(fsids, y_true, y_pred, k)
    sel = y_true == bg
    fa = float((y_pred[sel] != bg).mean()) if sel.sum() else float("nan")
    per_hour = 3600.0 / (hop_ms / 1000.0)
    return {
        "window": {"recall": dict(zip(names, [round(float(v), 4)
                                              for v in recalls(y_true, y_pred, k)])),
                   "macro_f1": round(macro_f1(y_true, y_pred, k), 4),
                   "n": int(len(y_true))},
        "origin": {"recall": dict(zip(names, [round(float(v), 4)
                                              for v in recalls(ot, op, k)])),
                   "macro_f1": round(macro_f1(ot, op, k), 4),
                   "n": int(len(ot))},
        "false_alarm": {"window_rate": round(fa, 5),
                        "per_hour": round(fa * per_hour, 1), "hop_ms": hop_ms},
    }


def report(fsids, y_true, y_pred, names, n_boot, seed,
           clips=None, starts=None, hop_ms=250):
    k = len(names)
    print_matrix(confusion(y_true, y_pred, k), names, "=== 창 단위 혼동행렬 ===")
    print(f"  macro-F1 {macro_f1(y_true, y_pred, k):.4f}   창 {len(y_true)}개")

    of, ot, op = majority_by_origin(fsids, y_true, y_pred, k)
    print_matrix(confusion(ot, op, k), names,
                 "=== 원본 단위 혼동행렬 (같은 fsid 다수결) ===")
    print(f"  macro-F1 {macro_f1(ot, op, k):.4f}   원본 {len(ot)}개")

    lo, hi = bootstrap_recall_ci(of, ot, op, k, n_boot, seed)
    rec = recalls(ot, op, k)
    print(f"\n=== 클래스별 재현율 95% 신뢰구간 (원본 부트스트랩 {n_boot}회) ===")
    print(f"{'클래스':<12}{'원본':>7}{'재현율':>9}{'95% CI':>20}")
    print("-" * 48)
    for c, n in enumerate(names):
        cnt = int((ot == c).sum())
        print(f"{n:<12}{cnt:>7}{rec[c]:>8.1%}   [{lo[c]:>6.1%}, {hi[c]:>6.1%}]")
    print("\n  ⚠️ 구간은 **원본 수**로 계산했다. 창 수로 내면 같은 녹음의 창들을")
    print("     독립 표본으로 세어 실제보다 좁게 나온다 (CLAUDE.md 5장 규칙 1).")

    if clips is not None and starts is not None:
        false_alarm_report(clips, starts, y_true, y_pred, names, hop_ms)


# ──────────────────────────────────────────────────────────── 추론
def load_model(checkpoint, n_classes, ai8x_dir, simulate=False, bias=False,
               random_init=False):
    """ai8x 모델을 만들고 체크포인트를 얹는다. `tools/stream_eval.py` 도 쓴다.

    QAT 체크포인트는 키가 어긋날 수 있어 `strict=False` 로 얹고 불일치 수를
    보고한다 — 조용히 무작위 가중치로 평가하는 것을 막기 위해서다.
    """
    import importlib.util

    import torch
    import ai8x

    sys.path.insert(0, ai8x_dir)
    ai8x.set_device(85, simulate, False)

    spec = importlib.util.spec_from_file_location(
        "ai85net_safesound", os.path.join(REPO, "models", "ai85net-safesound.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.AI85SafeSoundNet(num_classes=n_classes, bias=bias)

    if random_init:
        print("  [주의] --random-init — 배선 검증용이다. 수치에 의미 없음")
    else:
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"  [주의] state_dict 불일치 — 누락 {len(missing)} / "
                  f"초과 {len(unexpected)}")
            print("         QAT 체크포인트면 ai8x train.py --evaluate 로 교차 확인할 것")
    model.eval()
    return model


def ai8x_valid_indices(n_train, validation_split=0.1, seed=0):
    """ai8x(distiller) 가 train 에서 떼어내는 **검증셋 인덱스**를 재현한다.

    `distiller/apputils/data_loaders.py:get_data_loaders` 경로:
        if deterministic: distiller.set_deterministic()   # ← 인자 없음 = seed 0
        ...
        np.random.shuffle(indices)
        valid_indices, train_indices = __split_list(indices, validation_split)
        # __split_list: l[:floor(ratio*len)], l[floor(ratio*len):]

    ⚠️ **시드는 `--seed 1` 이 아니라 0 이다.** train.py 가 먼저
    `set_deterministic(args.seed)` 로 1 을 심지만, `get_data_loaders` 안에서
    `set_deterministic()` 을 **인자 없이** 다시 불러 0 으로 덮어쓴다
    (`set_deterministic(seed=0)`). 그 재시드와 shuffle 사이에 전역 numpy 난수를
    쓰는 코드가 없어 이 재현이 성립한다. 학습 명령의 `--seed` 를 바꿔도 분리는
    그대로다 — 바뀌는 건 가중치 초기화·증강 쪽이다.
    """
    idx = list(range(n_train))
    rng_state = np.random.get_state()
    try:
        np.random.seed(seed)
        np.random.shuffle(idx)
    finally:
        np.random.set_state(rng_state)             # 전역 상태를 건드리지 않는다
    n_val = int(np.floor(validation_split * n_train))
    return sorted(idx[:n_val])


def collect_logits(args, names, split="test", indices=None):
    """테스트셋 전체를 추론해 **로짓까지** 돌려준다.

    `tools/eval_threshold.py` 가 임계값을 쓸어 보려면 argmax 가 아니라 로짓이
    필요하다. 추론은 한 번만 하고 스윕은 로짓 위에서 한다.

    반환: (fsid, clip, start, 정답, 로짓 (N,K))
    """
    import torch
    import ai8x
    import safesound as S

    model = load_model(args.checkpoint, len(names), args.ai8x, args.simulate,
                       args.bias)

    ds = S.SafeSound(os.path.join(args.data, "SafeSound"), split,
                     transform=ai8x.normalize(args=argparse.Namespace(
                         act_mode_8bit=args.simulate)), augment=False)
    # 검증셋 평가처럼 일부만 볼 때 쓴다. 증강은 꺼 둔 채로다 (평가 조건).
    order = list(range(len(ds))) if indices is None else list(indices)
    fsids, clips, starts, y_true, logits = [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(order), args.batch_size):
            xs, ts = [], []
            for j in order[i:i + args.batch_size]:
                x, t = ds[j]
                xs.append(x)
                ts.append(t)
                clip_id, fsid, start = ds.meta[j]
                fsids.append(fsid)
                clips.append(clip_id)
                starts.append(start)
            out = model(torch.stack(xs))
            logits.append(out.detach().numpy())
            y_true += ts
            if i % (args.batch_size * 20) == 0:
                print(f"  추론 {i}/{len(order)}", flush=True)
    return (np.array(fsids), np.array(clips), np.array(starts),
            np.array(y_true), np.concatenate(logits))


def run_inference(args, names):
    """체크포인트로 테스트셋을 추론해 (fsid, clip, start, 정답, 예측)."""
    fsids, clips, starts, y_true, logits = collect_logits(args, names)
    return fsids, clips, starts, y_true, logits.argmax(1)


def self_test(names, seed=0):
    """통계 경로 검증 — 체크포인트 없이 합성 예측으로 돌린다.

    원본당 창 3개, 그중 1개만 틀리게 만들어 **원본 단위 다수결이 창 단위보다
    재현율이 높아야** 한다는 것을 확인한다. 동점 처리도 함께 본다.
    """
    rng = np.random.default_rng(seed)
    fsids, clips, starts, y_true, y_pred = [], [], [], [], []
    for c in range(len(names)):
        for o in range(40):
            f = f"c{c}o{o}"
            for w in range(3):
                fsids.append(f)
                clips.append(f)
                starts.append(16000 * (w + 1))       # 창 순서 (간격은 1초)
                y_true.append(c)
                # 창 하나는 틀리게 (다수결이면 살아난다)
                y_pred.append(c if w != 2 else int(rng.integers(len(names))))
    fsids, y_true, y_pred = np.array(fsids), np.array(y_true), np.array(y_pred)
    clips, starts = np.array(clips), np.array(starts)
    report(fsids, y_true, y_pred, names, 500, seed,
           clips=clips, starts=starts, hop_ms=250)

    k = len(names)
    win = np.mean(recalls(y_true, y_pred, k))
    of, ot, op = majority_by_origin(fsids, y_true, y_pred, k)
    org = np.mean(recalls(ot, op, k))
    print(f"\n[self-test] 창 단위 평균 재현율 {win:.3f} < 원본 단위 {org:.3f} : "
          f"{'통과' if org > win else '실패'}")
    print(f"[self-test] 원본 수 {len(ot)} (기대 {40 * k}) : "
          f"{'통과' if len(ot) == 40 * k else '실패'}")
    # 동점(2:2)에서 정답을 고르지 않는지
    tie_t, tie_p = np.array([0, 0, 0, 0]), np.array([0, 0, 1, 1])
    _f, _t, p = majority_by_origin(np.array(["x"] * 4), tie_t, tie_p, k)
    print(f"[self-test] 2:2 동점에서 정답(0) 회피 → 예측 {p[0]} : "
          f"{'통과' if p[0] != 0 else '실패'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--data", default="data", help="`{data}/SafeSound/test/...`")
    ap.add_argument("--ai8x", default="/content/ai8x-training")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--bias", action="store_true")
    ap.add_argument("--simulate", action="store_true",
                    help="act_mode_8bit — 양자화 체크포인트 평가 시")
    ap.add_argument("--hop-ms", type=int, default=250,
                    help="상시 추론 주기 — 시간당 오경보 환산에 쓴다 (250ms → 14,400회/h)")
    ap.add_argument("--json", help="요약을 JSON 으로 저장 (체크포인트 간 비교용)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true",
                    help="합성 예측으로 통계 경로만 검증 (체크포인트 불필요)")
    a = ap.parse_args()

    # ⚠️ `safesound` 는 **import 시점에** `ai8x` 를 부른다. 그래서 인자를 파싱하자마자,
    # safesound 를 부르기 전에 경로를 넣어야 한다 — 늦으면 ModuleNotFoundError: ai8x.
    if os.path.isdir(a.ai8x):
        sys.path.insert(0, a.ai8x)
    import safesound as S                              # noqa: PLC0415
    names = S.CLASSES

    if a.self_test:
        self_test(names, a.seed)
        return
    if not a.checkpoint:
        sys.exit("[에러] --checkpoint 가 필요하다 (또는 --self-test)")
    fsids, clips, starts, y_true, y_pred = run_inference(a, names)
    report(fsids, y_true, y_pred, names, a.n_boot, a.seed,
           clips=clips, starts=starts, hop_ms=a.hop_ms)
    if a.json:
        import json
        out = summary_json(fsids, y_true, y_pred, names, a.hop_ms)
        out["checkpoint"] = a.checkpoint
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n요약 JSON: {a.json}")


if __name__ == "__main__":
    main()
