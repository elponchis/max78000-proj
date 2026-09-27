#!/usr/bin/env python3
"""증류 교사 어댑터 known-answer 테스트 — torch 멜이 numpy 멜과 얼마나 다른가.

C(지식 증류)의 교사는 `AI85SafeSoundMelTeacher` 다. 학생(④)과 같은 파형 입력을
받아 **forward 안에서** 로그 멜을 만들고 멜 CNN 에 넣는다 (distiller 의 KD 정책이
교사에게 학생과 같은 입력을 주기 때문이다 — `knowledge_distillation.py:118`).

그 변환은 torch 로 다시 쓴 것이므로 `datasets/melfeat.py`(numpy, 학습·평가·펌웨어의
유일한 기준)와 **똑같을 수 없다.** "차이가 작아서 괜찮다" 를 측정 없이 주장하지
않기 위해 이 도구가 있다.

확인하는 것
  1. 학생 입력에서 int8 파형을 **정확히** 복원하는가 (역변환이 무손실인가)
  2. torch 로그 멜이 numpy 로그 멜과 몇 LSB 다른가 (float dB 와 int8 양쪽)
  3. 교사 로짓이 ① 원본 모델의 로짓과 얼마나 다른가 ← 최종적으로 이것이 중요하다
     (소프트 타깃으로 쓰이는 값이다)
  4. float 모드와 act_mode_8bit 모드 양쪽에서 동작하는가

⚠️ 교사는 `torch.no_grad()` 로만 돌고 학습되지 않으므로(`--kd-teacher-wt 0`)
1 LSB 수준 차이는 소프트 타깃에 거의 영향이 없다. 그러나 **로짓 차이가 크면**
증류가 전달하는 것이 ①의 결정 경계가 아니게 되므로, 3번이 판정 기준이다.

사용법 (WSL2 또는 Colab):
    python3 tools/kat_teacher.py --ai8x ~/ai8x-training \\
        --checkpoint <① qat_best.pth.tar>      # 체크포인트는 선택
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "datasets"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import melfeat as MF  # noqa: E402

# torch 멜과 numpy 멜의 허용 오차. int8 1 LSB = 0.275dB 이므로 1 LSB 안이면
# 모델이 받는 값이 사실상 같다. **측정 전 가안이 아니라 판정 기준이다** —
# 넘으면 구현이 갈린 것이므로 고쳐야 한다.
TOL_LSB = 1
TOL_LOGIT = 0.05        # 교사 로짓과 ① 로짓의 최대 절대 차이 상한

OK, FAIL = "  [통과]", "  [실패]"
fails = []


def check(name, cond, detail=""):
    print(f"{OK if cond else FAIL} {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai8x", default=os.path.expanduser("~/ai8x-training"))
    ap.add_argument("--data", default=os.path.join(REPO, "data", "processed",
                                                   "safesound"))
    ap.add_argument("--checkpoint", help="① qat_best — 주면 로짓까지 비교한다")
    ap.add_argument("--n", type=int, default=16, help="비교할 창 수")
    a = ap.parse_args()

    if not os.path.isdir(a.ai8x):
        sys.exit(f"[에러] ai8x 경로가 없다: {a.ai8x}")
    sys.path.insert(0, a.ai8x)
    import torch

    import ai8x
    import safesound as S

    ai8x.set_device(85, False, False)
    spec = importlib.util.spec_from_file_location(
        "melmod", os.path.join(REPO, "models", "ai85net-safesound-mel.py"))
    mm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mm)

    if not os.path.isdir(os.path.join(a.data, "test")):
        sys.exit(f"[에러] 샤드가 없다: {a.data} (--data 로 줄 것)")
    ds = S.SafeSound(a.data, "test", transform=None, augment=False)
    print(f"테스트 창 {len(ds):,} 중 {a.n}개로 비교\n")

    teacher = mm.ai85safesoundmelteacher(num_classes=len(S.CLASSES)).eval()
    norm8 = ai8x.normalize(args=argparse.Namespace(act_mode_8bit=True))
    normf = ai8x.normalize(args=argparse.Namespace(act_mode_8bit=False))

    # ── 1. 역변환이 무손실인가
    xs = torch.stack([ds[i][0] for i in range(a.n)])          # (N,128,128) [0,1)
    raw = []
    for i in range(a.n):
        row = np.asarray(ds._shard(ds.index[i][1])[ds.index[i][2]])  # noqa: SLF001
        raw.append(row[S.MARGIN:S.MARGIN + S.WIN])
    raw = np.stack(raw).astype(np.int16)

    x8 = norm8(xs)
    back = torch.transpose(x8, 1, 2).reshape(a.n, -1).numpy().astype(np.int16)
    check("학생 입력 → int8 파형 역변환이 무손실",
          bool((back == raw).all()),
          f"최대 차이 {int(np.abs(back - raw).max())}")

    # ── 2. torch 멜 vs numpy 멜
    with torch.no_grad():
        if not teacher._mel_ready:                             # noqa: SLF001
            teacher._build_mel(torch.device("cpu"), torch.float32)  # noqa: SLF001
        # 교사 내부와 같은 순서로 직접 계산해 비교한다
        w = torch.transpose(x8, 1, 2).reshape(a.n, -1)
        xp = torch.nn.functional.pad(w.unsqueeze(1) / 128.0,
                                     (MF.PAD, MF.PAD), mode="reflect").squeeze(1)
        fr = xp.unfold(1, MF.N_FFT, MF.HOP)[:, :MF.N_FRAMES] * teacher.win
        sp = torch.fft.rfft(fr, n=MF.N_FFT, dim=2)
        pw = (sp.real ** 2 + sp.imag ** 2) / teacher._ref      # noqa: SLF001
        db_t = (10.0 * torch.log10(torch.matmul(pw, teacher.fb.t())
                                   + MF.EPS)).transpose(1, 2).numpy()
        q_t = np.clip(np.round((db_t - MF.TOP_DB) * (255.0 / MF.SPAN_DB)) + 127.0,
                      -128, 127).astype(np.int16)

    db_n = np.stack([MF.log_mel_db(r.astype(np.int8)) for r in raw])
    q_n = np.stack([MF.log_mel_int8(r.astype(np.int8)) for r in raw]).astype(np.int16)

    lsb = MF.SPAN_DB / 255.0
    d_db = np.abs(db_t - db_n)
    d_q = np.abs(q_t - q_n)
    print(f"\n  float dB 차이   최대 {d_db.max():.6f}dB "
          f"(= {d_db.max()/lsb:.3f} LSB), 평균 {d_db.mean():.2e}dB")
    print(f"  int8 차이       최대 {int(d_q.max())} LSB, "
          f"불일치 {100*(d_q != 0).mean():.3f}% "
          f"({int((d_q != 0).sum()):,}/{d_q.size:,})")
    check(f"torch 멜이 numpy 멜과 {TOL_LSB} LSB 안", int(d_q.max()) <= TOL_LSB,
          f"최대 {int(d_q.max())} LSB")

    # ── 3. 교사 로짓 vs ① 원본 로짓 — 최종 판정 기준
    if a.checkpoint and os.path.isfile(a.checkpoint):
        from eval_confusion import load_model

        ref = load_model(a.checkpoint, len(S.CLASSES), a.ai8x, config="mel")
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        sd = {k.replace("module.", ""): v
              for k, v in ck.get("state_dict", ck).items()}
        miss, unexp = teacher.net.load_state_dict(sd, strict=False)
        print(f"\n  교사 내부에 ① 가중치 적재: 누락 {len(miss)} / 초과 {len(unexp)}")
        teacher.eval()
        with torch.no_grad():
            lt = teacher(x8)
            mel_in = torch.from_numpy(q_n.astype(np.float32)).unsqueeze(1)
            lr = ref(mel_in)
        d = float((lt - lr).abs().max())
        print(f"  교사 로짓 vs ① 로짓  최대 차이 {d:.6f}")
        check(f"교사 로짓이 ① 로짓과 {TOL_LOGIT} 안", d <= TOL_LOGIT, f"{d:.6f}")
    else:
        print("\n  (--checkpoint 를 주면 교사 로짓과 ① 로짓을 직접 비교한다 — "
              "이것이 최종 판정 기준이다)")

    # ── 4. 두 스케일 모드에서 동작하는가
    with torch.no_grad():
        for nm, t in (("act_mode_8bit", norm8(xs)), ("float", normf(xs))):
            y = teacher(t)
            check(f"{nm} 입력에서 forward", tuple(y.shape) == (a.n, len(S.CLASSES)),
                  str(tuple(y.shape)))

    print()
    if fails:
        print(f"실패 {len(fails)}건: {fails}")
        sys.exit(1)
    print("전부 통과")


if __name__ == "__main__":
    main()
