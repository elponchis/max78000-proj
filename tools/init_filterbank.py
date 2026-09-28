#!/usr/bin/env python3
"""④의 첫 층을 **멜 대역통과 필터뱅크로 초기화**한다 — G8 구성 ③의 구현.

## 왜 이것이 가능한가 (핵심)

④의 첫 층은 `voice_conv1 = Conv1d(128 → 100, kernel_size=1)` 이다.
입력은 `(128채널, 길이 128)` 이고, `kws20.py:558-560` 의 접기에 따라

    x[r][c] = sample[c*128 + r]        r = 프레임 내 잔 시간축, c = 프레임 번호

이므로, `k=1` 인 이 층의 출력은

    out[o][c] = act( Σ_r W[o][r] · x[r][c] )
              = act( W[o] 와 c번째 **128샘플 프레임**의 내적 )

**즉 이 층은 수학적으로 길이 128 FIR 필터 100개를, 128샘플마다 한 번씩(겹침
없이) 적용하는 필터뱅크와 완전히 같다.** 그래서 여기에 멜 대역통과 필터의
시간영역 응답을 넣으면 "전처리를 네트워크 안에 심은" 것이 된다 — G8 표의
구성 ③(학습 프론트엔드)이 바로 이것이다.

프레임률 = 16000/128 = **125Hz (8ms)**. 참고로 구성 ①의 멜 프론트엔드는 창 512 /
홉 256 이라 62.5Hz 다. ④의 접기는 **시간 해상도가 2배 좋고 주파수 해상도가 4배
나쁘다** — 128탭의 주파수 분해능은 Hann 기준 대략 500Hz 다. 500Hz 아래 대역들은
서로 분리되지 않는다 (아래 `--fmin` 기본값 125Hz 의 근거).

## 실수 FIR 만으로 크기를 얻는 법 — cos/sin 쌍과 활성화

FIR 하나는 **부호 있는 사영**이라 입력 위상에 따라 값이 출렁인다. 크기(에너지)를
얻으려면 해석 신호 쌍이 필요하다.

    re = ⟨w·cos(2πf t), frame⟩        im = ⟨w·sin(2πf t), frame⟩
    크기 ≈ |re| + |im|   (L1 근사. 참값의 1~√2 배 안)

여기서 **활성화가 결정적**이다.

| 활성화 | 밴드당 필터 | 밴드 수(100채널) | 문제 |
|---|---:|---:|---|
| **`Abs`** | **2** (cos, sin) | **50** | 없음. `|re|`, `|im|` 이 바로 나온다 |
| `ReLU` | 4 (±cos, ±sin) | 25 | 2개만 쓰면 위상이 3사분면일 때 둘 다 0 이 되어 **밴드가 사라진다** |

MAX78000 은 `Abs` 활성화를 지원하고 ai8x 에 `FusedConv1dAbs` 가 있다
(`ai8x.py:1727`). 그래서 **`Abs` + 밴드 50개**를 기본으로 한다. 다음 층
`voice_conv2(100→96, k=3)` 가 `|re|+|im|` 을 선형결합으로 만들 수 있다.

⚠️ `Abs` 를 쓰면 **초기화와 활성화 두 가지가 함께 바뀐다.** 떼어 낼 수 없다 —
`Abs` 가 있어야 밴드당 2필터로 크기가 나온다. 초기화만 바꾸는 순수 대조가
필요하면 `--relu` (밴드 25개, 활성화는 ReLU 유지) 를 쓴다.

## 학습 가능 / 고정

`--freeze-conv1` 로 첫 층을 고정할 수 있다. **학습 가능(기본)을 먼저 한다.**
  · 기준선 대비 바뀌는 것이 **초기값뿐**이라 원인이 깨끗하다
  · 구성 ③의 정의가 "**학습** 프론트엔드" 다. 고정은 사실상 고정 변환이라
    구성 ②(CPU FFT)에 가깝다
  · 학습 가능 판이 기준선과 다르지 않게 나오면 "초기화가 씻겨 나갔다" 는 뜻이고,
    그때 고정 판이 "고정 필터뱅크 자체는 도움이 되는가" 에 답한다

사용법 (WSL2 또는 Colab):
    python3 tools/init_filterbank.py --out data/fb-init.pth.tar \\
        --ai8x ~/ai8x-training
    # 학습: --model ai85safesoundnet_fb --exp-load-weights-from data/fb-init.pth.tar
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "datasets"))

import melfeat as MF  # noqa: E402

TAPS = 128          # voice_conv1 의 입력 채널 수 = FIR 길이
OUT_CH = 100        # voice_conv1 의 출력 채널 수


def mel_centers(n_bands, fmin, fmax):
    """멜 등간격 중심 주파수. `melfeat` 과 같은 HTK 식을 쓴다."""
    return MF.mel_to_hz(np.linspace(MF.hz_to_mel(fmin), MF.hz_to_mel(fmax),
                                    n_bands))


def to_conv_weight(fir, kernel):
    """길이 `kernel*128` FIR 들을 `(out_ch, 128, kernel)` 가중치로 배치한다.

    ⚠️ **배치 순서가 핵심이다.** 입력이 `x[r][c] = sample[c*128 + r]` 이므로

        out[o][c] = Σ_r Σ_j W[o][r][j] · x[r][c+j]
                  = Σ_r Σ_j W[o][r][j] · sample[c*128 + j*128 + r]

    즉 탭 인덱스는 **t = j*128 + r** 다. 따라서 길이 `k*128` 의 FIR 을
    `(k, 128)` 로 끊어(앞 128탭이 j=0) **전치**하면 `(128, k)` 가 된다.
    뒤집어 넣으면 필터가 시간축으로 뒤섞여 엉뚱한 응답이 나오는데, 학습이
    돌아가 버려서 눈에 띄지 않는다 — 그래서 톤 검산으로 확인한다.
    """
    n = fir.shape[0]
    return fir.reshape(n, kernel, TAPS).transpose(0, 2, 1)   # (out, 128, k)


def filterbank_weights(n_bands, fmin, fmax, relu=False, taps=TAPS,
                       out_ch=OUT_CH, sr=MF.SR):
    """(out_ch, taps) 가중치와 밴드 설명을 만든다.

    각 필터는 **L2 노름 1** 로 맞춘다. 정규직교 사영에 가까워져 출력 크기가
    입력 RMS 와 같은 자릿수가 되고, 이후 층의 초기 스케일 가정을 깨지 않는다.
    (윈도잉된 코사인을 그대로 넣으면 진폭이 커서 QAT 전 활성화가 포화한다.)
    """
    w = np.hanning(taps + 1)[:taps]            # 주기형 Hann
    t = (np.arange(taps) - (taps - 1) / 2.0) / sr
    centers = mel_centers(n_bands, fmin, fmax)
    rows, desc = [], []
    for f in centers:
        c = w * np.cos(2 * np.pi * f * t)
        s = w * np.sin(2 * np.pi * f * t)
        pairs = [(c, "cos"), (s, "sin")]
        if relu:
            # ReLU 는 음의 사영을 0 으로 만든다 → 부호 쌍을 함께 넣어야
            # |re|, |im| 이 두 채널에 나뉘어 살아남는다
            pairs = [(c, "+cos"), (-c, "-cos"), (s, "+sin"), (-s, "-sin")]
        for v, nm in pairs:
            n = np.linalg.norm(v)
            rows.append(v / (n if n else 1.0))
            desc.append((float(f), nm))
    W = np.array(rows, dtype=np.float32)
    if len(W) > out_ch:
        W, desc = W[:out_ch], desc[:out_ch]
    return W, desc, centers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "data",
                                                  "fb-init.pth.tar"))
    ap.add_argument("--ai8x", default=os.path.expanduser("~/ai8x-training"))
    ap.add_argument("--classes", type=int, default=5)
    ap.add_argument("--relu", action="store_true",
                    help="활성화를 ReLU 로 유지한다 (밴드당 4필터 → 25밴드). "
                         "초기화만 바꾸는 순수 대조용")
    ap.add_argument("--kernel", type=int, default=1, choices=(1, 2, 4),
                    help="첫 층 커널. **FIR 길이 = kernel × 128탭**이다 "
                         "(1→128탭 8ms, 2→256탭 16ms, 4→512탭 32ms). "
                         "4 는 ① 멜의 n_fft 512 와 창 길이가 같다")
    ap.add_argument("--fmin", type=float, default=125.0,
                    help="최저 중심 주파수. 128탭의 분해능이 약 500Hz 라 그 아래는 "
                         "서로 분리되지 않는다 — 너무 낮게 잡아도 밴드만 낭비된다")
    ap.add_argument("--fmax", type=float, default=7000.0)
    ap.add_argument("--freeze-conv1", action="store_true",
                    help="첫 층을 고정한다. **두 번째 실험용** — 먼저 학습 가능 "
                         "판을 돌릴 것 (모듈 설명 참조)")
    a = ap.parse_args()

    n_taps = a.kernel * TAPS
    res_hz = 4.0 * MF.SR / n_taps          # Hann 메인로브 폭 (근사)
    n_bands = OUT_CH // (4 if a.relu else 2)
    W, desc, centers = filterbank_weights(n_bands, a.fmin, a.fmax, a.relu,
                                          taps=n_taps)
    act = "ReLU" if a.relu else "Abs"
    print(f"활성화 {act} / 밴드 {n_bands}개 / 밴드당 {4 if a.relu else 2}필터 "
          f"/ 채널 {len(W)}/{OUT_CH}")
    print(f"중심 주파수 {centers[0]:.0f} ~ {centers[-1]:.0f}Hz (멜 등간격)")
    print(f"커널 {a.kernel} → 필터 길이 **{n_taps}탭 = {1000*n_taps/MF.SR:.1f}ms**, "
          f"홉 {TAPS}샘플({1000*TAPS/MF.SR:.1f}ms), 프레임률 {MF.SR/TAPS:.1f}Hz")
    print(f"분해능 약 {res_hz:.0f}Hz"
          + (f" (① 멜 n_fft {MF.N_FFT} 와 창 길이 동일)" if n_taps == MF.N_FFT else ""))
    low = int((centers < res_hz).sum())
    if low:
        print(f"  ⚠️ {res_hz:.0f}Hz 아래 밴드 {low}개는 {n_taps}탭으로 서로 "
              f"분리되지 않는다 (겹쳐서 초기화된다)")
    else:
        print(f"  ✔ 모든 밴드가 분해능({res_hz:.0f}Hz) 위에 있다")

    sys.path.insert(0, a.ai8x)
    import torch

    import ai8x

    ai8x.set_device(85, False, False)
    spec = importlib.util.spec_from_file_location(
        "m4", os.path.join(REPO, "models", "ai85net-safesound.py"))
    mm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mm)
    model = mm.AI85SafeSoundNet(num_classes=a.classes, abs_first=not a.relu,
                                first_kernel=a.kernel)

    sd = model.state_dict()
    key = "voice_conv1.op.weight"
    want = tuple(sd[key].shape)                 # (100, 128, kernel)
    print(f"\n대상 {key} {want}")
    assert want == (OUT_CH, TAPS, a.kernel), f"모양이 예상과 다르다: {want}"
    Wc = to_conv_weight(W, a.kernel)            # (out, 128, kernel)
    assert Wc.shape == want, f"배치 후 모양 불일치: {Wc.shape}"
    sd[key] = torch.from_numpy(Wc.copy())
    model.load_state_dict(sd, strict=True)
    print(f"  FIR {n_taps}탭을 커널 위치 {a.kernel}개로 나눠 배치했다 "
          f"(탭 t = j*{TAPS} + r)")

    # ── 검산: 톤을 넣으면 그 주파수 밴드가 최대여야 한다
    print("\n검산 — 단일 톤이 기대 밴드를 켜는가 (저역이 갈라지는지가 관건)")
    print(f"  {'톤':>8}{'최대 채널':>10}{'중심':>9}{'기대':>9}{'인접비(dB)':>11}  판정")
    print("  " + "-" * 56)
    ok = True
    for f_hz in (125.0, 200.0, 300.0, 400.0, 1000.0, 3000.0, 6000.0):
        tt = np.arange(16384) / MF.SR
        x = np.clip(np.round(100 * np.sin(2 * np.pi * f_hz * tt)), -128, 127)
        xt = torch.from_numpy(x.astype(np.float32) / 128.0)
        xt = torch.transpose(xt.reshape(-1, 128), 1, 0).unsqueeze(0)
        with torch.no_grad():
            y = model.voice_conv1(xt)[0]        # (100, 128)
        resp = y.mean(dim=1).numpy()
        band = int(np.argmax(resp))
        got = desc[band][0]
        near = min(range(len(centers)), key=lambda i: abs(centers[i] - f_hz))
        hit = abs(got - centers[near]) < 1e-6
        ok &= hit
        # 인접 밴드 분리도 — 최대 밴드와 **다른 중심 주파수** 중 최대의 비.
        # 같은 밴드의 cos/sin 짝은 제외한다 (한 밴드이지 이웃이 아니다).
        other = [resp[i] for i in range(len(resp))
                 if abs(desc[i][0] - got) > 1e-6]
        sep = (20 * np.log10(max(resp[band], 1e-12) / max(max(other), 1e-12))
               if other else float("inf"))
        print(f"  {f_hz:>6.0f}Hz{band:>10}{got:>8.0f}Hz{centers[near]:>8.0f}Hz"
              f"{sep:>11.1f}  {'OK' if hit else '불일치'}")
    if not ok:
        print("  ⚠️ 불일치가 있다. 접기 방향이나 샘플레이트를 다시 볼 것")

    if a.freeze_conv1:
        model.voice_conv1.op.weight.requires_grad_(False)
        print("\n첫 층을 고정했다 (requires_grad=False). "
              "옵티마이저는 grad 가 None 인 파라미터를 건너뛴다")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    # ⚠️ `epoch` 키가 **반드시 있어야 한다** — `--exp-load-weights-from` 경로의
    #    train.py:401 이 `checkpoint.get('epoch', None) >= start_epoch` 를
    #    평가하므로 없으면 TypeError 로 죽는다. 0 이면 float 으로 시작해
    #    기준선과 같은 에폭(60)에서 QAT 로 들어간다.
    torch.save({"state_dict": model.state_dict(),
                "epoch": 0,
                # 진입점 이름과 맞춘다 — 체크포인트의 arch 가 실제 모델과
                # 다르면 로더가 경고 없이 넘어가는 경로가 생긴다.
                "arch": "ai85safesoundnet_fb"
                        + ("" if a.kernel == 1 else f"_k{a.kernel}")
                        + ("_relu" if a.relu else ""),
                "extras": {"init": "mel_filterbank", "activation": act,
                           "kernel": a.kernel, "taps": n_taps,
                           "resolution_hz": round(res_hz, 1),
                           "bands": n_bands, "fmin": a.fmin, "fmax": a.fmax,
                           "freeze_conv1": a.freeze_conv1}}, a.out)
    print(f"\n저장: {a.out}")
    print(f"  분리도(인접비)는 클수록 좋다 — 0dB 근처면 이웃 밴드와 구분이 없다")
    print("학습: --model ai85safesoundnet_fb"
          + (" --relu 판이면 relu_first=True 진입점" if a.relu else "")
          + f" --exp-load-weights-from {a.out}")


if __name__ == "__main__":
    main()
