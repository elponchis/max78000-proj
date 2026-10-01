"""(c) 의 CPU 정수 로그 — **파이썬 기준 구현.**

이 파일과 `firmware/common/log2_q8.c` 는 **같은 LUT·같은 식**을 쓴다.
한쪽만 고치면 train/serve 가 어긋나고, 그걸 잡는 것이 `tools/kat_logstage.py`
(전수 비교)다.

## 왜 CLZ + LUT 인가

Cortex-M4 에 FPU 는 있지만 `logf()` 는 수백 사이클이다. 12,800 값에 걸면
전처리가 통째로 날아간다. 정수 로그는

    v = 2^e × m,  m ∈ [1, 2)
    log2(v) = e + log2(m)

로 쪼갠다. `e` 는 `__CLZ` 한 번(1 사이클), `log2(m)` 은 가수 상위 8bit 로
찾는 256엔트리 LUT 한 번이다.

## 고정소수점 규약 (양쪽이 **반드시** 같아야 한다)

| 항목 | 값 |
|---|---|
| 출력 | Q8 — `log2(v) × 256`, int32 |
| 입력 | `uint32_t` (NPU int32 출력의 절댓값) |
| `v == 0` | **0 을 돌려준다** (−∞ 대신). 아래 참조 |
| LUT | 256엔트리 int16, `round(log2(1 + i/256) × 256)` |
| 가수 추출 | `e >= 8 ? v >> (e-8) : v << (8-e)` 의 하위 8bit |

`v == 0` 을 0 으로 두는 이유: 로그의 참값은 −∞ 이고 어떤 유한값을 넣어도
임의 선택이다. **0 은 int8 양자화 후 하한에 붙으므로**(아래 `to_int8`),
"가장 작은 값" 이라는 뜻을 유지하면서 분기 하나로 끝난다.
⚠️ 이 선택은 `v == 1` (log2 = 0) 과 구분되지 않는다. 1 LSB 짜리 신호와
완전 무음이 같은 값이 된다는 뜻인데, 둘 다 int8 바닥이라 영향이 없다.
"""
import numpy as np

# Q8 고정소수점: 1.0 == 256
Q = 8
QONE = 1 << Q

# log2(1 + i/256) × 256, i = 0..255. **C 와 같은 반올림(round-half-away)** 이다.
#   numpy 의 np.round 는 half-to-even 이라 쓰지 않는다 — C 의 `lroundf` 와
#   달라지는 엔트리가 생긴다.
_frac = np.arange(256, dtype=np.float64) / 256.0
_exact = np.log2(1.0 + _frac) * QONE
LUT = np.floor(_exact + 0.5).astype(np.int16)


def log2_q8(v):
    """`log2(v) × 256` 을 int32 로. `v` 는 uint32 배열 또는 스칼라.

    C 구현과 **비트 단위로 같다** (`tools/kat_logstage.py` 가 전수 확인).
    """
    v = np.asarray(v, dtype=np.int64)
    if np.any(v < 0) or np.any(v > 0xFFFFFFFF):
        raise ValueError("입력은 uint32 범위여야 한다 (|NPU 출력| 을 넣는다)")
    out = np.zeros(v.shape, dtype=np.int64)
    nz = v > 0
    if not np.any(nz):
        return out.astype(np.int32)
    vv = v[nz]
    # e = 31 - CLZ(v)  →  2^e <= v < 2^(e+1)
    e = (np.floor(np.log2(vv.astype(np.float64)))).astype(np.int64)
    # ⚠️ float log2 는 2 의 거듭제곱 경계에서 흔들린다. 정수로 보정한다 —
    #    이 한 줄이 없으면 v = 2^k 부근에서 C 와 갈린다.
    e = np.clip(e, 0, 31)
    e = np.where((1 << np.minimum(e + 1, 63)) <= vv, e + 1, e)
    e = np.where((1 << e) > vv, e - 1, e)
    # 가수 상위 8bit
    m = np.where(e >= Q, vv >> np.maximum(e - Q, 0), vv << np.maximum(Q - e, 0))
    out[nz] = (e << Q) + LUT[(m & 0xFF).astype(np.int64)]
    return out.astype(np.int32)


def to_int8(lg, lo_q8, span_q8):
    """Q8 로그값을 int8 로. **포화**한다 (랩어라운드 금지, CLAUDE.md 7장).

    `lo_q8` 아래는 −128, `lo_q8 + span_q8` 위는 127 에 붙는다.
    """
    lg = np.asarray(lg, dtype=np.int64)
    y = ((lg - lo_q8) * 255) // max(int(span_q8), 1) - 128
    return np.clip(y, -128, 127).astype(np.int8)
