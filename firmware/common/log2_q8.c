/* (c) 의 CPU 정수 로그 — **C 구현.**
 *
 * `tools/log2_q8.py` 와 같은 LUT·같은 식이다. 한쪽만 고치면 train/serve 가
 * 어긋난다 — `tools/kat_logstage.py` 가 전수 비교로 막는다.
 *
 * 규약은 파이썬 쪽 docstring 과 같다:
 *   · 출력 Q8 (log2(v) x 256), int32
 *   · v == 0 이면 0
 *   · LUT 256엔트리 int16, round(log2(1 + i/256) x 256), half-away-from-zero
 *
 * ⚠️ 이 파일은 **x86 에서도 빌드해 KAT 를 돈다.** Cortex-M 전용 내장함수를
 *    직접 쓰지 않고 __CLZ 를 매크로로 감싼다.
 */
#include <stdint.h>

#if defined(__ARM_ARCH) && (__ARM_ARCH >= 7)
#include <arm_acle.h>
#define CLZ32(x) ((int)__clz((uint32_t)(x)))
#elif defined(__GNUC__) || defined(__clang__)
/* __builtin_clz 는 0 에서 정의되지 않는다. 호출 전에 v != 0 을 보장한다. */
#define CLZ32(x) (__builtin_clz((uint32_t)(x)))
#else
#error "CLZ 를 제공하는 컴파일러가 필요하다"
#endif

#define LOG2Q8_Q 8

/* round(log2(1 + i/256) * 256), i = 0..255.
 * tools/gen_log2_lut.py 가 생성한다 — **손으로 고치지 말 것.** */
const int16_t log2_q8_lut[256] = {
#include "log2_q8_lut.inc"
};

/* |v| 의 Q8 로그. v == 0 이면 0. */
int32_t log2_q8(uint32_t v)
{
    if (v == 0u) {
        return 0;
    }
    const int32_t e = 31 - CLZ32(v);
    /* 가수 상위 8bit: 1.xxxxxxxx 의 소수부 */
    const uint32_t m = (e >= LOG2Q8_Q) ? (v >> (e - LOG2Q8_Q))
                                       : (v << (LOG2Q8_Q - e));
    return (e << LOG2Q8_Q) + (int32_t)log2_q8_lut[m & 0xFFu];
}

/* Q8 로그값을 int8 로. **포화**한다 (랩어라운드 금지 — CLAUDE.md 7장).
 *
 * ⚠️ MSDK kws20_demo 는 int8 대입에서 포화시키지 않아 큰 소리의 부호가
 *    뒤집힌다. 우리는 반드시 포화한다. 나눗셈은 **바닥 나눗셈**이라
 *    파이썬의 `//` 와 같다 — C 의 `/` 는 0 방향 절단이므로 음수에서
 *    갈린다. 그래서 명시적으로 처리한다.
 */
int8_t log2_q8_to_int8(int32_t lg, int32_t lo_q8, int32_t span_q8)
{
    if (span_q8 < 1) {
        span_q8 = 1;
    }
    const int64_t num = (int64_t)(lg - lo_q8) * 255;
    int64_t q = num / span_q8;
    if ((num % span_q8) != 0 && ((num < 0) != (span_q8 < 0))) {
        q -= 1;                      /* 바닥 나눗셈 보정 */
    }
    q -= 128;
    if (q < -128) {
        return (int8_t)-128;
    }
    if (q > 127) {
        return (int8_t)127;
    }
    return (int8_t)q;
}
