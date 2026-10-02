#!/usr/bin/env python3
"""게인 스윕 — **음량 단서 의존**을 직접 잰다. 재학습 없음, 평가만.

여러 실험에서 `siren` 만 반복해 무너졌다.

| 실험 | siren | 다른 클래스 |
|---|---:|---|
| 제곱근 가중치 (파형) | −4.7pp | glass·dog_bark 상승 |
| ~~음량 정규화 (b1)~~ | ~~−8~−11pp~~ | **정정 (2026-10-02)**: 평가 버그였다. 정정 후 +3.1pp |
| **(a) NPU 로그 근사** | **−15.6pp** | glass·scream·dog_bark **+9~15pp** |

세 변경의 공통점: **레벨 정보를 지우거나 압축한다.**
(⚠️ 정규화 행은 근거에서 빠진다 — `docs/STATUS.md` 5.7. 남는 것은 두 변경이다.)

> **가설**: `siren` 은 고유 원본이 202개뿐이라 모델이 소리의 모양이 아니라
> **음량(레벨)** 을 단서로 쓰고 있다.

**이 도구가 답하는 것**: 테스트 입력의 게인을 바꿨을 때 클래스별 재현율이
얼마나 흔들리는가. `siren` 만 크게 흔들리면 가설을 지지한다.

⚠️ **학습 곡선(`siren_curve.py`)과는 다른 질문이다.**
- 학습 곡선 → **양**: 원본을 더 넣으면 좋아지는가
- 게인 스윕 → **단서**: 지금 모델이 무엇에 의존하는가
둘은 독립이다 — 양이 부족해도 단서 의존이 없을 수 있고, 그 반대도 가능하다.

⚠️ 게인은 **학습 때와 같은 규칙**으로 건다 (`SafeSound._rand_gain`):
피크가 풀스케일을 넘으면 **넘는 만큼만 되돌린다**(인위적 클리핑 금지).
`--hard-clip` 으로 포화 판을 따로 볼 수 있다 — 실기기의 큰 소리는 포화이므로
둘을 비교하면 그 차이도 드러난다.

사용 (WSL2):
    ~/ai8x-training/venv/bin/python tools/gain_sweep.py \\
        --run "D-1:wave_fb:<ckpt>" --run "(a):wave_logstage:<ckpt>" \\
        --run "①:mel:data/safesound-mel-v1_qat_best.pth.tar"
"""

import argparse
import contextlib
import io as _io
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI8X = os.environ.get("AI8X_DIR") or os.path.expanduser("~/ai8x-training")
sys.path[:0] = [AI8X, os.path.join(REPO, "datasets"), os.path.join(REPO, "tools")]

CLASSES = ["siren", "glass", "scream", "dog_bark", "background"]
BG = 4
GAINS = (-12.0, -6.0, 0.0, 6.0, 12.0)


class GainDataset:
    """테스트셋을 읽되 int8 파형에 게인을 걸어 내놓는다.

    `SafeSound` 를 감싸고 `_crop` 직후에 게인을 끼운다. 창 선택·잘라내기·
    표현 변환은 전부 원래 경로를 그대로 탄다 — 게인 하나만 다르다.
    """

    def __init__(self, base, gain_db, hard_clip=False):
        self.b = base
        self.g = 10.0 ** (gain_db / 20.0)
        self.hard = hard_clip
        self.meta = base.meta
        self.index = base.index

    def __len__(self):
        return len(self.b)

    def _apply16(self, w):
        """int16 샤드용 — 게인을 **정규화 전** int16 위에서 건다.

        실기기에서 레벨 변동은 마이크 쪽(int16)에서 생기고 정규화는 그 뒤다.
        int16 범위(마이크 풀스케일)를 넘을 때만 되돌리기/포화가 갈린다.
        """
        y = w.astype(np.float32) * self.g
        if not self.hard:
            peak = float(np.abs(y).max())
            if peak > 32767.0:
                y *= 32767.0 / peak
        return np.clip(np.round(y), -32768, 32767).astype(np.int16)

    def _apply(self, w):
        y = w.astype(np.float32) * self.g
        if self.hard:
            return np.clip(np.round(y), -128, 127).astype(np.int8)
        peak = float(np.abs(y).max())
        if peak > 127.0:            # 학습과 같은 규칙 — 넘는 만큼만 되돌린다
            y *= 127.0 / peak
        return np.clip(np.round(y), -128, 127).astype(np.int8)

    def __getitem__(self, i):
        # 원래 __getitem__ 의 앞부분을 그대로 재현하고 게인만 끼운다
        import torch
        b = self.b
        target, shard, row, left, right = b.index[i]
        stored = np.asarray(b._shard(shard)[row])
        rng = np.random.default_rng()
        w = b._crop(stored, left, right, rng)       # augment=False → 가운데
        # ⚠️ 전처리가 데이터셋 인자로 들어가는 구성은 **게인 뒤에** 그 전처리를
        # 태운다 (2026-10-02). 이 메서드가 base.__getitem__ 을 우회하므로
        # 빠뜨리면 정규화로 학습한 모델을 정규화 없는 입력으로 재게 된다.
        if hasattr(b, "gmax"):                      # int16 샤드 + 변환 전 정규화
            import safesound as S
            _, w = S.norm16(self._apply16(w), b.gmax)
        else:
            w = self._apply(w)
            if getattr(b, "norm_pow2", False):      # 1단계 — int8 위 시프트
                _, w = b._norm_pow2(w)
        if hasattr(b, "scheme"):                    # 멜 구성
            import melfeat as MF
            f = MF.mel_int8(w, b.scheme)
            x = torch.from_numpy(np.ascontiguousarray(f).astype(np.int16)) + 128
            x = (x.float() / 256.0).unsqueeze(0)
        else:                                       # 파형 구성
            x = torch.from_numpy(np.ascontiguousarray(w).astype(np.int16)) + 128
            x = x.float() / 256.0
            x = torch.transpose(x.reshape((-1, 128)), 1, 0)
        if b.transform is not None:
            x = b.transform(x)
        return x, target


def logits_for(ck, config, data, gain_db, hard, batch=128):
    import argparse as _ap
    import torch
    import ai8x
    import eval_confusion as EC
    model = EC.load_model(os.path.join(REPO, ck), len(CLASSES), AI8X, False,
                          False, config=config)
    cls = EC.dataset_class(config)
    kw = dict(EC.CONFIG_KWARGS.get(config, {}))
    kw.pop("num_channels", None)
    kw.pop("dimensions", None)
    kw.pop("abs_first", None)
    kw.pop("first_kernel", None)
    kw.pop("pool_first", None)
    kw.pop("bias", None)
    # 샤드 루트·전처리 인자는 평가 경로와 **같은 표**에서 읽는다
    root, dkw = EC.CONFIG_DATA.get(config, ("SafeSound", {}))
    kw.update(dkw)
    base = cls(os.path.join(data, root), "test",
               transform=ai8x.normalize(args=_ap.Namespace(act_mode_8bit=False)),
               augment=False, **kw)
    ds = GainDataset(base, gain_db, hard)
    ys, lg = [], []
    with torch.no_grad():
        for b in range(0, len(ds), batch):
            xs, ts = [], []
            for j in range(b, min(b + batch, len(ds))):
                x, t = ds[j]
                xs.append(x)
                ts.append(t)
            lg.append(model(torch.stack(xs)).numpy())
            ys += ts
    fs = np.array([m[1] for m in base.meta])
    return np.array(ys), np.concatenate(lg), fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    metavar="이름:config:체크포인트")
    ap.add_argument("--data", default=os.path.join(AI8X, "data"))
    ap.add_argument("--gains", default="-12,-6,0,6,12")
    ap.add_argument("--refix-fa", action="store_true",
                    help="게인마다 오경보를 300/h 로 다시 맞춘다 "
                         "(동작점 이동을 걷어내고 변별력만 본다)")
    ap.add_argument("--hard-clip", action="store_true",
                    help="포화 판 (기본은 학습과 같은 '되돌리기')")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import eval_confusion as EC
    from eval_threshold import margin_of, margin_predict
    gains = [float(x) for x in a.gains.split(",")]

    out = {}
    for spec in a.run:
        name, config, ck = spec.split(":", 2)
        print(f"\n=== {name}  ({config})"
              + ("  [포화판]" if a.hard_clip else ""))
        print(f"{'게인':>7}" + "".join(f"{c:>11}" for c in CLASSES)
              + f"{'macroF1':>10}")
        print("-" * (7 + 11 * len(CLASSES) + 10))
        rows = {}
        base_thr = None
        for g in gains:
            with contextlib.redirect_stdout(_io.StringIO()):
                y, lg, _fs = logits_for(ck, config, a.data, g, a.hard_clip)
            # 두 읽기가 있다.
            #  · 고정 문턱값 (기본) — **배포 현실**. 실기기의 문턱값은 빌드
            #    상수라 입력 레벨을 따라 움직이지 않는다. 다만 게인이 사실상
            #    동작점 이동으로 작동해 '변별력 저하' 와 섞인다.
            #  · --refix-fa — 게인마다 오경보를 300/h 로 **다시 맞춘다**.
            #    동작점 이동을 걷어내고 **변별력 자체**만 본다.
            if a.refix_fa:
                fx = EC.fixed_fa_points(lg, y, CLASSES, hop_ms=250)
                thr = fx["targets"]["300/h"]["thr"]
            else:
                if g == 0.0 or base_thr is None:
                    fx = EC.fixed_fa_points(lg, y, CLASSES, hop_ms=250)
                    base_thr = fx["targets"]["300/h"]["thr"]
                thr = base_thr
            marg_ev, ev_best = margin_of(lg, BG)
            pred = margin_predict(marg_ev, ev_best, lg[:, BG], BG, thr)
            # EC.recalls 는 **클래스 인덱스 배열**을 돌려준다 (dict 아님)
            rv = EC.recalls(y, pred, len(CLASSES))
            rec = {c: float(rv[i]) for i, c in enumerate(CLASSES)}
            mf = EC.macro_f1(y, pred, len(CLASSES))
            rows[g] = {"recall": rec, "macro_f1": mf, "thr": thr}
            print(f"{g:>+6.0f}dB"
                  + "".join(f"{100*rec[c]:>10.1f}%" for c in CLASSES)
                  + f"{mf:>10.4f}")
        out[name] = rows
        r0 = rows[0.0]["recall"]
        print("  0dB 대비 최대 낙폭(pp): " + "  ".join(
            f"{c} {min(100*(rows[g]['recall'][c]-r0[c]) for g in gains):+.1f}"
            for c in CLASSES[:4]))

    print()
    print("⚠️ 문턱값은 **0dB 에서 정한 값으로 고정**했다. 실기기의 문턱값은")
    print("   빌드 상수라 입력 레벨을 따라 움직이지 않는다.")
    print("⚠️ 게인은 학습과 같은 규칙(피크 초과 시 되돌리기)이다. --hard-clip")
    print("   으로 포화 판을 따로 보면 큰 소리에서의 차이가 드러난다.")
    print("⚠️ 이것은 **단서 의존**을 재는 것이고, 원본 **양** 문제와는 다른")
    print("   질문이다 (그쪽은 tools/siren_curve.py).")

    if a.json:
        json.dump(out, open(a.json, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"\n저장: {a.json}")


if __name__ == "__main__":
    main()
