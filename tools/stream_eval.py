#!/usr/bin/env python3
"""빈틈없는 스트리밍 오탐 평가 — 실기기 오경보 빈도의 PC 대조군.

`tools/eval_confusion.py` 의 N프레임 수치는 **하한(낙관치)** 이다. 테스트셋의 창은
클립당 최대 3개를 점수 순으로 고른 것이라 시간적으로 떨어져 있고, 떨어진 창들의
오류는 서로 독립에 가까워 다수결이 실제보다 잘 듣는다. 실기기의 연속 프레임은
0.75초를 겹쳐 오류가 함께 가므로 감소폭이 반드시 더 작다.

그래서 여기서는 **배경음 원본 오디오를 처음부터 끝까지 hop 간격으로 빈틈없이**
훑는다. 실기기가 상시 구동에서 보는 것과 같은 프레임 열이고, N프레임 다수결도
진짜 연속 프레임 위에서 계산된다. 이 수치가 논문에 쓰는 PC 측 값이고, 보드
8시간 무인 구동 결과와 **같은 형식**으로 출력해 직접 대조한다 (G7).

── 전처리 (학습 경로와 동일해야 한다) ────────────────────────────────────
  16kHz mono 리샘플 → 고정 스케일 ×127 int8 → (128,128) → ai8x.normalize
  **정규화 없음, 증강 없음.** 배경음이므로 에너지 하한(`--floor-zero`)도, 무음
  베드 채움도 적용하지 않는다 — 실기기는 그런 전처리를 하지 않고 들어오는 모든
  프레임을 추론한다. 조용한 구간을 걸러내면 오탐률이 낙관적으로 나온다.

사용법 (WSL2 또는 Colab, CPU 로 충분):
    python3 tools/stream_eval.py --checkpoint logs/safesound-v1/best.pth.tar
    python3 tools/stream_eval.py --checkpoint ... --hop-ms 500 --json out.json
    python3 tools/stream_eval.py --random-init --limit 20   # 배선 검증
"""

import argparse
import collections
import csv
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "datasets"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prepare_safesound as P                              # noqa: E402


def frames_of(x, hop):
    """오디오 → 빈틈없는 창 시작점. 꼬리(1초 미만)는 버린다 — 실기기도 그렇다."""
    return range(0, max(0, len(x) - P.WIN + 1), hop)


def majority_alarms(preds, bg, n):
    """연속 N프레임 엄격 다수결. 반환 (판정 지점 수, 오경보 수, 클래스별 오경보).

    엄격 다수(> N/2)를 요구한다 — N=2 는 2프레임 모두, N=3 은 2프레임 이상이
    같은 이벤트여야 울린다. 펌웨어의 이벤트 병합 상태머신과 같은 규칙이다
    (CLAUDE.md 6장).
    """
    points = alarms = 0
    by_cls = collections.Counter()
    for i in range(len(preds) - n + 1):
        points += 1
        cnt = collections.Counter(preds[i:i + n])
        lab, top = cnt.most_common(1)[0]
        if top > n / 2 and lab != bg:
            alarms += 1
            by_cls[lab] += 1
    return points, alarms, by_cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--manifest", default="data/interim/manifest.csv")
    ap.add_argument("--fsd-dir", default="data/raw/FSD50K_clips")
    ap.add_argument("--us8k-dir", default="data/raw/US8K_audio")
    ap.add_argument("--esc50-dir", default="data/raw/ESC50_audio")
    ap.add_argument("--ai8x", default="/content/ai8x-training")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--hop-ms", type=int, default=250,
                    help="상시 추론 주기. 실기기와 같은 값을 쓸 것")
    ap.add_argument("--ns", type=int, nargs="+", default=[1, 2, 3],
                    help="연속 N프레임 다수결")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="처음 N클립만 (디버깅)")
    ap.add_argument("--bias", action="store_true")
    ap.add_argument("--simulate", action="store_true", help="act_mode_8bit")
    ap.add_argument("--random-init", action="store_true",
                    help="체크포인트 없이 배선만 검증 (수치 무의미)")
    ap.add_argument("--json", help="결과를 JSON 으로도 저장 (보드 결과와 대조용)")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    # ai8x 는 레포 안에 있어 import 전에 경로를 넣어야 한다
    sys.path.insert(0, a.ai8x)
    import torch
    import ai8x
    import safesound as S
    from eval_confusion import load_model

    torch.set_num_threads(a.threads)
    if not a.checkpoint and not a.random_init:
        sys.exit("[에러] --checkpoint 가 필요하다 (또는 --random-init)")

    names = S.CLASSES
    bg = names.index("background")
    hop = max(1, int(P.SR * a.hop_ms / 1000))

    roots = {"fsd": a.fsd_dir, "us8k": a.us8k_dir, "esc50": a.esc50_dir}
    rows = [r for r in csv.DictReader(open(a.manifest, encoding="utf-8"))
            if r["cls"] == "background" and r["split"] == a.split]
    rows = [(r, P.clip_path(r, roots)) for r in rows]
    rows = [(r, p) for r, p in rows if p]
    if a.limit:
        rows = rows[:a.limit]
    if not rows:
        sys.exit(f"[에러] {a.split} 배경음 오디오가 없다.")

    model = load_model(a.checkpoint, len(names), a.ai8x, a.simulate, a.bias,
                       a.random_init)
    norm = ai8x.normalize(args=argparse.Namespace(act_mode_8bit=a.simulate))

    print(f"배경음 원본 {len(rows)}개를 hop {a.hop_ms}ms 로 빈틈없이 훑는다 "
          f"(split={a.split})")
    per_clip = {}                  # clip_id -> 예측 열
    n_samples = 0
    t0 = time.time()
    for i, (r, path) in enumerate(rows, 1):
        x = P.load_audio(path)
        if x is None:
            continue
        starts = list(frames_of(x, hop))
        if not starts:
            continue               # 1초 미만 클립은 프레임이 없다
        n_samples += len(x)
        preds = []
        for b in range(0, len(starts), a.batch_size):
            chunk = starts[b:b + a.batch_size]
            q = np.stack([P.to_int8(x[s:s + P.WIN]) for s in chunk])
            t = (torch.from_numpy(q.astype(np.int16)) + 128).float() / 256.0
            t = torch.transpose(t.reshape(len(chunk), -1, 128), 2, 1)
            with torch.no_grad():
                preds += model(norm(t)).argmax(1).tolist()
        per_clip[r["clip_id"]] = preds
        if i % 100 == 0:
            done = n_samples / P.SR / 3600
            print(f"  {i}/{len(rows)}  누적 {done:.2f}시간  "
                  f"{time.time() - t0:.0f}s", flush=True)

    hours = n_samples / P.SR / 3600
    total_frames = sum(len(v) for v in per_clip.values())
    per_hour = 3600.0 / (a.hop_ms / 1000.0)

    print(f"\n=== 스트리밍 오탐 평가 — PC (hop {a.hop_ms}ms) ===")
    print(f"  오디오 {hours:.2f}시간 / 원본 {len(per_clip)}개 / 프레임 "
          f"{total_frames:,}개 (시간당 {per_hour:,.0f})")
    print(f"  추론 소요 {time.time() - t0:.0f}s (CPU {a.threads}스레드)")

    result = {"source": "pc-stream", "split": a.split, "hop_ms": a.hop_ms,
              "hours": round(hours, 3), "clips": len(per_clip),
              "frames": total_frames, "by_n": {}}

    print(f"\n  {'N':>3}{'판정 지점':>12}{'오경보':>9}{'비율':>10}{'시간당':>11}")
    print("  " + "-" * 46)
    for n in a.ns:
        pts = al = 0
        by_cls = collections.Counter()
        for preds in per_clip.values():
            p, c, b = majority_alarms(preds, bg, n)
            pts += p
            al += c
            by_cls += b
        rate = al / pts if pts else float("nan")
        print(f"  {n:>3}{pts:>12,}{al:>9,}{rate:>9.3%}{rate * per_hour:>10.1f}회")
        result["by_n"][n] = {"points": pts, "alarms": al,
                             "rate": rate, "per_hour": rate * per_hour,
                             "by_class": {names[c]: v for c, v in by_cls.items()}}

    n1 = result["by_n"].get(1)
    if n1 and n1["alarms"]:
        print(f"\n  프레임 단위(N=1) 오경보 클래스 분해")
        print(f"  {'클래스':<14}{'오경보':>9}{'비율':>10}{'시간당':>11}")
        print("  " + "-" * 45)
        for cls, v in sorted(n1["by_class"].items(), key=lambda kv: -kv[1]):
            r = v / n1["points"]
            print(f"  {cls:<14}{v:>9,}{r:>9.3%}{r * per_hour:>10.1f}회")

    # 오경보가 몰린 원본 — 청취로 확인할 대상
    worst = sorted(((sum(1 for p in v if p != bg), len(v), c)
                    for c, v in per_clip.items()), reverse=True)[:10]
    print(f"\n  오경보가 많은 원본 상위 10개 (청취 확인용)")
    print(f"  {'클립':<16}{'오경보/프레임':>16}{'비율':>10}")
    print("  " + "-" * 44)
    for n_fa, n_fr, clip in worst:
        if n_fa:
            print(f"  {clip:<16}{n_fa:>7,}/{n_fr:<8,}{n_fa / n_fr:>9.1%}")
    result["worst_clips"] = [{"clip_id": c, "alarms": n_fa, "frames": n_fr}
                             for n_fa, n_fr, c in worst if n_fa]

    print("\n  ⚠️ 이 수치는 **연속 프레임** 위에서 계산했다 — `eval_confusion.py` 의")
    print("     N프레임 하한과 달리 논문에 쓸 수 있다. 다만 여전히 PC 측 값이다:")
    print("     실기기는 마이크 잡음·AGC 없음·클리핑 랩어라운드(7장) 때문에 다를 수")
    print("     있으므로, 보드 8시간 무인 구동 결과와 **같은 형식으로 나란히** 싣는다.")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n  JSON: {a.json}  (보드 결과를 같은 스키마로 내면 바로 대조된다)")


if __name__ == "__main__":
    main()
