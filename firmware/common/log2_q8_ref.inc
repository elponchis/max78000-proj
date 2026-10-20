/* 옛 log2_q8_to_int8 (64비트 나눗셈, 2026-10-21 이전) — -DFE_LOG_AB 전·후 비교용. 비교가 끝나면 쓰지 않는다. */
#include <stdint.h>
__attribute__((noinline)) int8_t log2_q8_to_int8_ref(int32_t lg, int32_t lo_q8, int32_t span_q8)
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
