/* feat/log32 — log2_q8_to_int8 의 32비트 구현이 옛 64비트 구현과 **전 범위**에서 같은지 (x86, PC).
 *   gcc -O2 -I firmware/common -o /tmp/log32_check tools/log32_check.c && /tmp/log32_check
 * 1) lg 전 범위 int32 (2^32 개) × lo ∈ {1279, 1627} × span 2976   2) E 전 범위 uint32 → log2_q8 → 둘 비교 */
#include <stdio.h>
#include <stdint.h>
#include "log2_q8.c"
#include "log32_ref.inc"
int main(void)
{
    const int32_t los[2] = {1279, 1627}, span = 2976;
    unsigned long long bad = 0, n = 0;
    for (int k = 0; k < 2; k++) {
        for (int64_t lg = INT32_MIN; lg <= INT32_MAX; lg++) {
            int8_t a = log2_q8_to_int8_ref((int32_t)lg, los[k], span), b = log2_q8_to_int8((int32_t)lg, los[k], span);
            n++;
            if (a != b) { if (bad < 5) printf("  DIFF lg=%lld lo=%d ref=%d new=%d\n", (long long)lg, los[k], a, b); bad++; }
        }
        printf("lg int32 전 범위 lo=%d: %llu 비교, 불일치 %llu\n", los[k], n, bad); n = 0;
        for (int64_t v = 0; v <= 0xFFFFFFFFLL; v++) {
            int32_t lg = log2_q8((uint32_t)v);
            int8_t a = log2_q8_to_int8_ref(lg, los[k], span), b = log2_q8_to_int8(lg, los[k], span);
            n++;
            if (a != b) { if (bad < 5) printf("  DIFF E=%llu lo=%d ref=%d new=%d\n", (unsigned long long)v, los[k], a, b); bad++; }
        }
        printf("E uint32 전 범위 lo=%d: %llu 비교, 누적 불일치 %llu\n", los[k], n, bad); n = 0;
    }
    printf("RESULT %s (불일치 %llu)\n", bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
}
