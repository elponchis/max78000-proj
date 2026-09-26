#!/usr/bin/env python3
"""오분류 창을 wav 로 내보낸다 — 귀로 원인을 가르기 위해.

기준선에서 scream 이 siren 40 / dog_bark 44 로 샜다. 이 혼동의 원인은 둘 중
하나인데 숫자로는 갈리지 않는다.

  · **라벨이 틀렸다** — 태거 임계값 0.025 로 내리며 약 45% 오염을 감수한 자리다
    (`docs/results/listening-verification.md` 4.2). 그렇다면 임계값을 되돌리는
    재학습이 답이다
  · **라벨은 맞는데 모델이 못 가른다** — 그렇다면 임계값을 올려도 소용없고
    모델·데이터 쪽을 봐야 한다

들어보면 갈린다. 파일명에 **원본 ID·예측 클래스·확률**을 넣어, 듣는 즉시 어느
쪽인지 적을 수 있게 한다.

저장 형식은 학습 입력과 같은 1초 창이다 (샤드의 1.2초에서 가운데를 자른 것 —
증강 없이 평가에 실제로 쓰인 구간).

사용법 (Colab 또는 WSL2):
    python3 tools/export_errors.py --checkpoint <qat_best.pth.tar> \\
        --data /content/ai8x-training/data --ai8x /content/ai8x-training \\
        --true scream --pred siren dog_bark --n 15 \\
        --out /content/drive/MyDrive/max78000/listen_scream_errors
"""

import argparse
import csv
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "datasets"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


SLUG = {"일반 배경음": "general", "하드네거티브": "hardneg", "[조용]": "-quiet",
        "기타": "other", "(미상)": "unknown"}


def slug(tag):
    """출처 태그를 파일명에 넣을 ASCII 로 바꾼다 (파일명 한글은 OS 간에 깨진다)."""
    out = tag
    for k, v in SLUG.items():
        out = out.replace(k, v)
    return (out.replace(":", "-").replace(" ", "").replace("<", "-")
            .replace(">", "").strip("-_") or "unknown")


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--ai8x", default="/content/ai8x-training")
    ap.add_argument("--out", required=True, help="wav 를 저장할 디렉터리")
    ap.add_argument("--true", default="scream", help="정답 클래스")
    ap.add_argument("--pred", nargs="+", default=["siren", "dog_bark"],
                    help="이 클래스로 잘못 간 창들을 뽑는다")
    ap.add_argument("--n", type=int, default=15, help="예측 클래스별 표본 수")
    ap.add_argument("--pick", choices=["random", "top"], default="random",
                    help="random(기본)=무작위 표본 — 오류의 **분포**를 본다. "
                         "top=예측 확률 상위 — 모델이 **가장 확신한** 오류를 본다. "
                         "배경음 오경보 청취는 top 을 쓴다 (확신한 것일수록 라벨에 "
                         "실제 이벤트가 섞여 있을 확률이 높다)")
    ap.add_argument("--tag-source", action="store_true",
                    help="파일명에 배경음 출처 태그(일반/하드네거티브 종류/조용한 "
                         "창)를 넣는다. `--true background` 일 때 쓴다")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--bias", action="store_true")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if os.path.isdir(a.ai8x):
        sys.path.insert(0, a.ai8x)
    import soundfile as sf
    import safesound as S
    from eval_confusion import collect_logits

    names = S.CLASSES
    t_idx = names.index(a.true)
    p_idx = [names.index(p) for p in a.pred]

    print(f"체크포인트: {a.checkpoint}")
    fsids, clips, starts, y_true, logits = collect_logits(a, names, "test")
    prob = softmax(logits)
    pred = logits.argmax(1)
    print(f"테스트 창 {len(y_true):,} / {a.true} {int((y_true == t_idx).sum()):,}창\n")

    # 원본 int8 창을 그대로 꺼내려면 샤드를 다시 연다 (transform 을 거치지 않은 값)
    raw = S.SafeSound(os.path.join(a.data, "SafeSound"), "test",
                      transform=None, augment=False)

    # 배경음 출처 태그 — 파일명에 넣어 듣는 즉시 어느 풀에서 온 소리인지 알게 한다
    src = None
    if a.tag_source:
        from eval_threshold import load_bg_sources
        src = load_bg_sources(os.path.join(a.data, "SafeSound"), "test") or {}

    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    rows = []
    for c in p_idx:
        sel = np.where((y_true == t_idx) & (pred == c))[0]
        if len(sel) <= a.n:
            take = sel
        elif a.pick == "top":
            take = sel[np.argsort(-prob[sel, c])[:a.n]]     # 확률 상위
        else:
            take = rng.choice(sel, a.n, replace=False)
        take = sorted((int(i) for i in take), key=lambda i: -prob[i, c])
        how = "확률 상위" if a.pick == "top" else "무작위"
        print(f"  {a.true} → {names[c]}: 오분류 {len(sel)}창 중 {len(take)}개 "
              f"({how})")
        for i in take:
            stored = np.asarray(raw._shard(raw.index[i][1])[raw.index[i][2]])  # noqa: SLF001
            w = stored[S.MARGIN:S.MARGIN + S.WIN].astype(np.float32) / 127.0
            p = float(prob[i, c])
            tag = ""
            if src is not None:
                tag = "_" + slug(src.get(clips[i], "unknown"))
            name = (f"{a.true}_to_{names[c]}_p{round(100*p):02d}{tag}_"
                    f"{clips[i]}_{starts[i]}.wav")
            sf.write(os.path.join(a.out, name), w, 16000)
            rows.append([name, a.true, names[c], f"{p:.4f}",
                         f"{float(prob[i, t_idx]):.4f}", clips[i], fsids[i],
                         int(starts[i]), src.get(clips[i], "") if src else ""])

    with open(os.path.join(a.out, "errors.csv"), "w", newline="",
              encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "true", "pred", "p_pred", "p_true", "clip_id",
                     "fsid", "start_sample", "bg_source"])
        wr.writerows(sorted(rows))

    print(f"\n{len(rows)}개 → {a.out}")
    print("  파일명: {정답}_to_{예측}_p{확률×100}"
          + ("_{출처}" if src is not None else "") + "_{원본}_{시작샘플}.wav")
    print("\n듣고 판정할 것:")
    if a.true == "background":
        print("  · 진짜 사이렌/개 짖음이 배경음 라벨에 섞였다 → **배경음 풀 오염**이다.")
        print("    판정 순서(4장)가 그 클립을 배경음으로 보낸 이유를 되짚어야 한다.")
        print("    하드네거티브 태그가 붙어 있으면 의도된 것이고(경보음≠사이렌),")
        print("    '일반 배경음' 인데 사이렌이면 라벨 판정이 새는 것이다.")
        print("  · 사이렌도 개도 아닌데 모델이 확신했다 → **모델이 약한 것**이다.")
        print("    지금 train Top1 61~64% 과소적합과 맞물리는 쪽이다.")
        print("  · [quiet] 태그가 붙은 조용한 창에서 울린다면 조용한 창 쿼터(15%)가")
        print("    '무음 탐지기' 를 만드는 대신 오히려 오탐원이 된 것이다 → 비율 재검토.")
    else:
        print(f"  · 진짜 {a.true} 인데 모델이 못 가른 것  → 임계값이 아니라 모델 문제")
        print(f"  · {a.true} 가 아닌데 라벨이 잘못 붙은 것 → 태거 임계값 복귀(재학습) 근거")
    print("  결과는 data_overrides.csv 에 기록하면 다음 집계부터 반영된다.")


if __name__ == "__main__":
    main()
