#!/usr/bin/env python3
"""`datasets/melfeat.py` + `safesound_mel.py` known-answer 테스트 — 구성 ①.

구성 ④의 `kat_safesound.py` 와 같은 목적이되, 검사 대상이 **전처리 체인 자체**다.
④는 파형을 옮겨 담기만 하므로 축만 맞으면 끝이지만, ①은 STFT·멜·로그·양자화를
거치므로 **M4 고정소수점 구현이 PC float 과 비트 단위로 같을 수 없다.** 그래서
여기서는 "같은가" 가 아니라 **"얼마나 다른가를 정해 두고 지키는가"** 를 본다.

확인하는 것
  1. 필터뱅크가 건전한가 — 빈 밴드가 없고, 주파수 정렬이 맞는가
     (단일 톤을 넣어 기대한 밴드가 최대가 되는지 자리별로 확인)
  2. dB ↔ int8 왕복이 정의대로인가 — 위/아래 끝이 **포화**하는가
     (MSDK 데모의 랩어라운드 버그를 전처리에서 반복하지 않는다, CLAUDE.md 7장)
  3. 절대 레벨 정책이 지켜지는가 — **입력을 6dB 줄이면 출력 dB 도 6dB 준다.**
     어딘가에 정규화가 숨어 있으면 이 검사가 깨진다
  4. 구성 ④와 **같은 창**을 보는가 — 같은 샤드·같은 meta·같은 길이
  5. 증강이 ④와 같은 코드를 타는가 (상속 구조가 끊기지 않았는가)
  6. lazy 인가 — 테스트 캐시가 mmap 으로 열리는가

기준 벡터는 `tools/kat_vectors/` 에 `melkat_*` 로 저장한다. **보드 도착 후
M4 전처리에 같은 입력을 넣고 이 출력과 비교한다** (허용 오차는 아래 TOL).

사용법 (WSL2):
    python3 tools/kat_safesound_mel.py            # 검증 (벡터 없으면 생성)
    python3 tools/kat_safesound_mel.py --save     # 벡터 다시 쓰기

⚠️ 데이터로더 검사에 `--data` 로 샤드 루트를 줘야 한다. 기본값은 WSL 경로다.
   Colab: `--data /content/ai8x-training/data/SafeSound`
   샤드를 못 찾으면 **생략이 아니라 실패**다 — 기준 벡터도 실샤드 창에서만
   만든다 (예전의 합성 톤 대체 경로가 Colab 회귀 검사 실패의 진짜 원인이었다).
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "datasets"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import melfeat as MF  # noqa: E402

# ai8x 가 없어도 돌도록 최소 스텁을 넣는다 (`kat_safesound.py` 와 같은 방식).
# `safesound.py` 가 import 시점에 ai8x 를 부르기 때문이다. 검사 대상은 정규화
# 규약뿐이라 이것으로 충분하다.
try:
    import ai8x                                            # noqa: F401
except ImportError:
    import types as _t

    class _Norm:                                           # ai8x.py:29 와 동일
        def __init__(self, args=None):
            self.args = args

        def __call__(self, img):
            return img.sub(0.5).mul(256.).round().clamp(min=-128, max=127)

    sys.modules["ai8x"] = _t.SimpleNamespace(normalize=_Norm)

VEC_DIR = os.path.join(REPO, "tools", "kat_vectors")

# 펌웨어 허용 오차. **측정 전 가안이다** (CLAUDE.md 10장) — 보드에서 M4
# 고정소수점 구현을 돌려 실제 분포를 본 뒤 확정한다. int8 1 LSB = 0.275dB 이므로
# ±2 LSB 는 약 0.55dB 이고, 그 정도면 로그 멜 특징으로서 유의미한 차이가 아니다.
TOL_LSB = 2

# 회귀 검사(PC↔PC)의 허용 오차. 펌웨어(TOL_LSB)보다 **엄격하다** — 같은 파이썬
# 구현을 돌리므로 차이가 나올 이유가 numpy/FFT 반올림뿐이고, 그건 int8 경계에
# 걸친 소수의 원소에서 1 LSB 로만 나타난다. 이보다 크거나 넓게 퍼지면 상수나
# 구현이 바뀐 것이므로 실패로 본다.
REG_TOL_LSB = 1          # 최대 절대 차이 상한
REG_TOL_FRAC = 0.001     # 불일치 원소 비율 상한 (0.1%)

OK, FAIL = "  [통과]", "  [실패]"
fails = []


def check(name, cond, detail=""):
    print(f"{OK if cond else FAIL} {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def tone(freq, amp=100.0, n=16384):
    """int8 사인파. 자리 검사용 — 어느 멜 밴드가 최대가 되어야 하는지 계산 가능."""
    t = np.arange(n) / float(MF.SR)
    return np.clip(np.round(amp * np.sin(2 * np.pi * freq * t)), -128, 127) \
             .astype(np.int8)


def expected_band(freq):
    """그 주파수가 속해야 할 멜 밴드 — 필터 중심이 가장 가까운 것."""
    edges = MF.mel_to_hz(np.linspace(MF.hz_to_mel(MF.FMIN),
                                     MF.hz_to_mel(MF.FMAX), MF.N_MELS + 2))
    return int(np.argmin(np.abs(edges[1:-1] - freq)))


def c_array(name, arr, per_line=16):
    flat = arr.reshape(-1).astype(np.int8)
    body = ""
    for i in range(0, len(flat), per_line):
        body += "    " + ", ".join(f"{int(v):4d}"
                                   for v in flat[i:i + per_line]) + ",\n"
    return f"static const int8_t {name}[{len(flat)}] = {{\n{body}}};\n"


def compare_vectors(ref, got, got_db=None, d=VEC_DIR):
    """회귀 검사 — 자리별 차이를 수치로 보고하고 허용 오차로 판정한다.

    단순 `array_equal` 로 두면 **원인을 알 수 없다.** 실패가
      · 상수·구현이 바뀐 것인지 (차이가 크고 넓다)
      · 플랫폼 간 numpy/FFT 반올림이 int8 경계에서 1 LSB 튄 것인지 (작고 드물다)
      · 아예 **다른 입력**을 비교한 것인지 (거의 전부 불일치)
    구별되지 않아서다. 실제로 Colab 에서 세 번째가 일어났고(샤드를 못 찾아
    합성 톤으로 대체됐다) 두 번째로 오진할 뻔했다.

    `got_db` 를 주면 저장된 float dB 기준과도 비교한다. int8 차이가 1 LSB 인데
    float 차이가 1 LSB(0.275dB)의 절반에도 못 미치면 **반올림 경계 문제**임이
    확정된다 — 구현이 같다는 증거다.
    """
    if ref.shape != got.shape:
        check("저장된 기준 벡터와 일치 (회귀 검사)", False,
              f"모양이 다르다 {ref.shape} vs {got.shape} — 상수가 바뀌었다")
        return
    diff = got.astype(np.int32) - ref.astype(np.int32)
    n_mis = int((diff != 0).sum())
    frac = n_mis / diff.size
    mx = int(np.abs(diff).max())
    print(f"\n  회귀 검사 상세: 불일치 {n_mis:,}/{diff.size:,} "
          f"({frac:.4%})  최대 절대 차이 {mx} LSB "
          f"(1 LSB = {MF.SPAN_DB/255:.3f}dB)")
    if n_mis:
        vals, cnt = np.unique(diff[diff != 0], return_counts=True)
        print("  차이 분포: "
              + ", ".join(f"{int(v):+d}×{int(c)}" for v, c in zip(vals, cnt)))

    # float 단계 비교 — 있으면 반올림 문제인지 확정할 수 있다
    dbp = os.path.join(d, "melkat_logmel_db.npy")
    if got_db is not None and os.path.isfile(dbp):
        rdb = np.load(dbp).astype(np.float64)
        if rdb.shape == np.asarray(got_db).shape:
            ddb = np.abs(np.asarray(got_db, dtype=np.float64) - rdb)
            lsb = MF.SPAN_DB / 255.0
            print(f"  float dB 단계: 최대 {ddb.max():.6f}dB "
                  f"(= {ddb.max()/lsb:.3f} LSB), 평균 {ddb.mean():.2e}dB")
            if ddb.max() < 0.5 * lsb:
                if n_mis:
                    print("  → float 차이가 1 LSB 의 절반 미만이다. int8 불일치는")
                    print("     **반올림 경계 문제**이고 구현은 같다.")
                else:
                    print("  → float 단계까지 동일하다 (float32 저장 왕복 오차만).")
            else:
                print("  → float 차이가 이미 크다. 반올림이 아니라 계산이 다르다")
                print("     (numpy/FFT 버전, 필터뱅크 상수, 창 함수를 볼 것).")
    elif got_db is not None:
        print("  (float dB 기준이 없다 — `--save` 로 다시 만들면 다음부터 비교된다)")

    ok = mx <= REG_TOL_LSB and frac <= REG_TOL_FRAC
    detail = (f"허용: 최대 ≤{REG_TOL_LSB} LSB, 불일치 ≤{REG_TOL_FRAC:.1%}")
    if ok and n_mis:
        detail += " — 플랫폼 간 반올림 차이로 본다 (경고)"
    elif not ok:
        detail += " — 상수·구현이 바뀐 것이다. 의도한 변경이면 --save"
    check("저장된 기준 벡터와 일치 (회귀 검사)", ok, detail)


def save_vectors(win, feat, feat_db=None, d=VEC_DIR):
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "melkat_window_int8.npy"), win)
    np.save(os.path.join(d, "melkat_logmel_int8.npy"), feat)
    if feat_db is not None:
        # float 기준도 남긴다 — int8 불일치가 반올림인지 계산 차이인지 가른다
        np.save(os.path.join(d, "melkat_logmel_db.npy"),
                np.asarray(feat_db, dtype=np.float32))
    with open(os.path.join(d, "melkat_vectors.h"), "w", encoding="utf-8") as f:
        f.write(
            "/* SafeSound 구성 (1) 로그 멜 known-answer 벡터.\n"
            " * tools/kat_safesound_mel.py 생성. 손으로 고치지 말 것.\n"
            " *\n"
            f" * melkat_window : 1초 int8 창 ({len(win)} 샘플).\n"
            f" * melkat_logmel : 그 창의 int8 로그 멜 "
            f"({MF.N_MELS} mel x {MF.N_FRAMES} frame, mel-major).\n"
            " *\n"
            " * 펌웨어(M4 + CMSIS-DSP)에서 melkat_window 를 전처리에 넣고 결과를\n"
            f" * melkat_logmel 과 비교할 것. 허용 오차 +-{TOL_LSB} LSB\n"
            f" * (1 LSB = {MF.SPAN_DB/255:.3f}dB). 이보다 크게 어긋나면 STFT 창\n"
            " * 함수, 멜 필터 계수, dB 기준점(_REF) 중 하나가 다른 것이다.\n"
            " *\n"
            f" * 상수: n_fft {MF.N_FFT}, hop {MF.HOP}, pad {MF.PAD}(reflect),\n"
            f" *       n_mels {MF.N_MELS}, {MF.FMIN:.0f}-{MF.FMAX:.0f}Hz HTK,\n"
            f" *       TOP_DB {MF.TOP_DB}, SPAN_DB {MF.SPAN_DB} */\n"
            "#ifndef MELKAT_VECTORS_H\n#define MELKAT_VECTORS_H\n"
            "#include <stdint.h>\n\n")
        f.write(c_array("melkat_window", win))
        f.write("\n")
        f.write(c_array("melkat_logmel", feat))
        f.write("\n#endif /* MELKAT_VECTORS_H */\n")
    # ④ 쪽 README.md 를 덮지 않도록 파일을 따로 쓴다 (두 도구가 같은 폴더를 쓴다)
    with open(os.path.join(d, "README-mel.md"), "w", encoding="utf-8") as f:
        f.write(
            "# KAT 벡터 (SafeSound 구성 ① — 로그 멜)\n\n"
            "`tools/kat_safesound_mel.py` 가 생성한다. **손으로 고치지 말 것.**\n\n"
            "| 파일 | 내용 |\n|---|---|\n"
            f"| `melkat_window_int8.npy` | 1초 int8 창 ({len(win)}) — ④와 같은 "
            "지점의 값이다 |\n"
            f"| `melkat_logmel_int8.npy` | int8 로그 멜 "
            f"({MF.N_MELS}×{MF.N_FRAMES}, mel-major) |\n"
            "| `melkat_logmel_db.npy` | 양자화 **전** float dB (float32) — "
            "int8 불일치가 반올림인지 계산 차이인지 가른다 |\n"
            "| `melkat_vectors.h` | int8 둘의 C 배열 — 펌웨어용 |\n\n"
            "## 회귀 검사 허용 오차 (PC↔PC)\n\n"
            f"최대 절대 차이 ≤ **{REG_TOL_LSB} LSB** 이고 불일치 원소 비율 ≤ "
            f"**{REG_TOL_FRAC:.1%}** 면 통과(경고)다. 같은 파이썬 구현이라 차이가\n"
            "날 이유는 numpy/FFT 반올림뿐이고, 그건 int8 경계에 걸친 소수의\n"
            "원소에서 1 LSB 로만 나타난다. float dB 차이가 1 LSB 의 절반 미만이면\n"
            "반올림 문제임이 확정된다.\n\n"
            "⚠️ 이 벡터는 **실제 샤드의 test 창 0번**에서 만든다. 샤드를 못 찾으면\n"
            "검사는 생략이 아니라 **실패**다 — Colab 에서 경로가 달라 합성 톤으로\n"
            "대체됐고, 그 때문에 회귀 검사만 실패해 수치 오차로 오진할 뻔했다.\n"
            "`--data /content/ai8x-training/data/SafeSound` 로 넘길 것.\n\n"
            "## 상수 (바꾸면 학습·KAT·펌웨어를 함께 재생성할 것)\n\n"
            f"n_fft {MF.N_FFT} / hop {MF.HOP} / reflect pad {MF.PAD} / "
            f"n_mels {MF.N_MELS} / {MF.FMIN:.0f}~{MF.FMAX:.0f}Hz HTK / "
            f"TOP_DB {MF.TOP_DB} / SPAN_DB {MF.SPAN_DB} "
            f"(1 LSB = {MF.SPAN_DB/255:.3f}dB)\n\n"
            "근거는 `docs/results/mel-range-sweep.md` 다.\n\n"
            "## 쓰는 곳\n\n"
            "1. PC: `python3 tools/kat_safesound_mel.py` — 전처리 회귀 검사\n"
            "2. 보드: `melkat_window` 를 M4 전처리(CMSIS-DSP)에 넣고 결과를\n"
            f"   `melkat_logmel` 과 비교. **허용 오차 ±{TOL_LSB} LSB**"
            f"(≈{TOL_LSB*MF.SPAN_DB/255:.2f}dB).\n"
            "   넘으면 창 함수·멜 계수·dB 기준점 중 하나가 다른 것이다.\n\n"
            "⚠️ 구성 ④(raw 파형)에는 이 오차 항목이 **없다**. 전처리가 옮겨 담기뿐\n"
            "이기 때문이다. 이 차이가 G4(온디바이스 vs PC 괴리)에서 ①이 ④보다\n"
            "크게 어긋날 것으로 보는 근거이고, 그 자체가 G8 의 결과 중 하나다.\n")
    return d


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO, "data", "processed",
                                                   "safesound"),
                    help="샤드 루트 (`{루트}/test/<클래스>/index.csv`). "
                         "Colab 은 /content/ai8x-training/data/SafeSound. "
                         "**없으면 생략이 아니라 실패다**")
    ap.add_argument("--save", action="store_true", help="기준 벡터를 다시 쓴다")
    args = ap.parse_args()

    print(f"멜 프론트엔드: n_fft {MF.N_FFT} hop {MF.HOP} pad {MF.PAD} "
          f"mels {MF.N_MELS} frames {MF.N_FRAMES}")
    print(f"양자화: TOP_DB {MF.TOP_DB} SPAN_DB {MF.SPAN_DB} "
          f"(1 LSB = {MF.SPAN_DB/255:.3f}dB)\n")

    # 1. 필터뱅크 건전성 + 주파수 자리
    fb = MF.mel_filterbank()
    check("필터뱅크 모양", fb.shape == (MF.N_MELS, MF.N_FFT // 2 + 1),
          str(fb.shape))
    check("빈 밴드 없음", bool((fb.sum(axis=1) > 0).all()),
          f"최소 가중치 합 {fb.sum(axis=1).min():.3f}")
    # ⚠️ "꼭대기가 정확히 1.0" 을 검사하면 안 된다 — 저역 필터는 폭이 좁아
    #    삼각형 꼭대기가 FFT 빈 사이에 떨어지고, 빈에서 표본한 최댓값은 1보다
    #    작다(실측 0.49까지). 검사할 것은 **면적 정규화를 하지 않았다**는 사실,
    #    즉 가중치가 1을 넘지 않고 밴드가 넓어질수록 합이 커진다는 것이다.
    check("가중치가 1을 넘지 않음", bool(fb.max() <= 1.0 + 1e-12),
          f"최대 {fb.max():.4f}")
    rs = fb.sum(axis=1)
    check("면적 정규화 안 함 (고역 밴드 합이 저역보다 큼)",
          bool(rs[-8:].mean() > 3.0 * rs[:8].mean()),
          f"저역 평균 {rs[:8].mean():.2f} → 고역 평균 {rs[-8:].mean():.2f}")
    for f_hz in (250.0, 1000.0, 4000.0):
        d = MF.log_mel_db(tone(f_hz))
        got = int(np.argmax(d.mean(axis=1)))
        want = expected_band(f_hz)
        check(f"{f_hz:.0f}Hz 톤이 밴드 {want} 부근에서 최대",
              abs(got - want) <= 1, f"실제 최대 밴드 {got}")

    # 2. dB ↔ int8 왕복과 **포화**
    grid = np.array([MF.TOP_DB + 20, MF.TOP_DB, MF.TOP_DB - MF.SPAN_DB / 2,
                     MF.TOP_DB - MF.SPAN_DB, MF.TOP_DB - MF.SPAN_DB - 20])
    q = MF.db_to_int8(grid)
    check("위쪽 포화 (+127)", int(q[0]) == 127 and int(q[1]) == 127, str(q))
    check("아래쪽 포화 (-128)", int(q[3]) == -128 and int(q[4]) == -128, str(q))
    check("가운데가 선형", abs(int(q[2]) - 0) <= 1, f"기대 0 실제 {int(q[2])}")
    back = (q[2].astype(np.float64) - 127.0) * (MF.SPAN_DB / 255.0) + MF.TOP_DB
    check("역변환 오차 1 LSB 이내",
          abs(back - grid[2]) <= MF.SPAN_DB / 255.0 + 1e-9,
          f"{back:.3f} vs {grid[2]:.3f}")

    # 3. 절대 레벨 — 정규화가 숨어 있으면 여기서 깨진다
    loud = tone(1000.0, amp=100.0)
    quiet = np.round(loud.astype(np.float64) / 2.0).astype(np.int8)  # -6dB
    dl = MF.log_mel_db(loud)
    dq = MF.log_mel_db(quiet)
    b = expected_band(1000.0)
    delta = float(dl[b].mean() - dq[b].mean())
    check("입력 -6dB → 출력 -6dB (정규화 없음)", abs(delta - 6.02) < 0.3,
          f"실제 {delta:.2f}dB")
    check("풀스케일 사인 ≈ 0dB", abs(float(MF.log_mel_db(tone(1000.0, 127.0))
                                          .max())) < 1.5,
          f"{float(MF.log_mel_db(tone(1000.0, 127.0)).max()):.2f}dB")

    # 4~6. 데이터로더 — 실제 샤드가 **있어야 한다**.
    #
    # ⚠️ 예전에는 샤드가 없으면 "생략" 하고 합성 톤으로 기준 벡터를 만들었다.
    #    그 결과 Colab 에서 경로가 달라 샤드를 못 찾자 로더 검사가 **한 번도
    #    돌지 않은 채** 통과로 보였고, 저장된(실샤드) 벡터와 그 자리에서 만든
    #    (합성 톤) 벡터를 비교해 회귀 검사만 실패했다. 원인이 수치 오차처럼
    #    보이지만 사실은 **입력 자체가 다른 것**이었다. 생략은 없앤다.
    root = args.data
    if not os.path.isdir(os.path.join(root, "test")):
        check(f"샤드를 찾았다 ({root})", False,
              "데이터로더 검사가 하나도 못 돌았다. --data 로 샤드 루트를 줄 것 "
              "(Colab: /content/ai8x-training/data/SafeSound)")
    else:
        import safesound as S
        import safesound_mel as SM

        ws = S.SafeSound(root, "test", transform=None, augment=False)
        ms = SM.SafeSoundMel(root, "test", transform=None, augment=False)
        check("구성 ④와 같은 창 집합", len(ws) == len(ms) and ws.meta == ms.meta,
              f"{len(ms):,}창")
        x, _t = ms[0]
        check("입력 모양 (1, mels, frames)",
              tuple(x.shape) == (1, MF.N_MELS, MF.N_FRAMES), str(tuple(x.shape)))
        check("입력 범위 [0,1) — ai8x.normalize 앞단",
              0.0 <= float(x.min()) and float(x.max()) < 1.0,
              f"[{float(x.min()):.4f}, {float(x.max()):.4f}]")
        # 데이터로더 출력과 프론트엔드 직접 호출이 같은가 (경로가 하나인가)
        w0 = np.asarray(ws._shard(ws.index[0][1])[ws.index[0][2]])  # noqa: SLF001
        direct = MF.log_mel_int8(w0[S.MARGIN:S.MARGIN + S.WIN])
        via = np.round(x.numpy()[0] * 256.0).astype(np.int16) - 128
        check("데이터로더 = 프론트엔드 직접 호출",
              bool((direct.astype(np.int16) == via).all()),
              f"최대 차이 {int(np.abs(direct.astype(np.int16) - via).max())}")

        tr = SM.SafeSoundMel(root, "train", transform=None, augment=True)
        a1, _ = tr[3]
        a2, _ = tr[3]
        check("증강이 살아 있다 (같은 인덱스 두 번이 다름)",
              not bool(np.array_equal(a1.numpy(), a2.numpy())))
        check("증강 split 에 캐시를 못 붙인다",
              _raises(lambda: SM.SafeSoundMel(root, "train", cache="x.npy")))
        check("샤드는 mmap 으로만 연다",
              isinstance(ms._shard(ms.index[0][1]), np.memmap))  # noqa: SLF001

        win = w0[S.MARGIN:S.MARGIN + S.WIN]
        feat = direct
        feat_db = MF.log_mel_db(win)

        # 기준 벡터 — 실샤드 창 하나로 만든다 (합성 톤 대체 경로는 없다)
        wpath = os.path.join(VEC_DIR, "melkat_logmel_int8.npy")
        if args.save or not os.path.isfile(wpath):
            save_vectors(win, feat, feat_db)
            print(f"\n  기준 벡터 저장: {VEC_DIR}/melkat_*")
        else:
            compare_vectors(np.load(wpath), feat, feat_db)

    print()
    if fails:
        print(f"실패 {len(fails)}건: {fails}")
        sys.exit(1)
    print("전부 통과")


def _raises(fn):
    try:
        fn()
    except Exception:       # noqa: BLE001 — 예외 종류가 아니라 발생 여부를 본다
        return True
    return False


if __name__ == "__main__":
    main()
