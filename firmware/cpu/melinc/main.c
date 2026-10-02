// 구성 ①′ 의 CPU 전처리(증분 로그 멜) 측정 펌웨어 (MEASURE_BUILD) — MAX78000FTHR
//
// 하는 일
//   1. KAT (a): 일괄 계산(`melfeat_inc_batch`) vs melfeat.py "loginc" 기준 — ±2 LSB
//      KAT (b): 증분 4회 push 뒤의 링 버퍼 == 일괄 계산 — **비트 일치**
//   2. 판단 1회의 전처리 지연을 N_ITER 회 잰다. 구간 둘:
//        push — 새 4,000 샘플 → 프레임 16개 (창·rfft·멜·로그)
//        get  — 링 버퍼 64열을 시간 순으로 풀어 64×64 int8 로 (NPU 적재 직전)
//   3. 참고: 일괄 계산(64프레임, 패딩 없음) 1회 — 증분과의 비
//
// 조건: 코어 100 MHz (IPO), **NPU 꺼짐**, 카운터 SysTick, UART 는 측정 구간 밖.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "mxc.h"
#include "melfeat.h"
#include "mel_tables.h"
#include "melkat_vectors.h"
#include "melinc_kat.h"

#ifndef N_ITER
#define N_ITER 1000
#endif
#define TICK_MASK 0x00FFFFFFu
#define N_FEAT (MEL_N_MELS * MEL_N_FRAMES)

static int8_t feat[N_FEAT], feat_inc[N_FEAT];
static uint32_t t_push[N_ITER], t_get[N_ITER];

static uint32_t tick(void)
{
    return 0x00FFFFFFu - SysTick->VAL;
}

static int cmp_u32(const void *a, const void *b)
{
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static void report(const char *name, uint32_t *v, uint32_t per)
{
    uint64_t sum = 0;
    for (int i = 0; i < N_ITER; i++) sum += v[i];
    qsort(v, N_ITER, sizeof(v[0]), cmp_u32);
    printf("%-5s cycles min %lu  med %lu  mean %lu  max %lu   (us: med %lu  max %lu)\n", name,
           (unsigned long)v[0], (unsigned long)v[N_ITER / 2], (unsigned long)(sum / N_ITER),
           (unsigned long)v[N_ITER - 1], (unsigned long)(v[N_ITER / 2] / per),
           (unsigned long)(v[N_ITER - 1] / per));
}

int main(void)
{
    MXC_ICC_Enable(MXC_ICC0);
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO); // 100 MHz
    SystemCoreClockUpdate();

    printf("Waiting...\n");
    MXC_Delay(SEC(2)); // 디버거가 끼어들 틈 (지우지 말 것)

    melfeat_init();
    // SysTick 을 인터럽트 없이 켠다. ⚠️ 이 뒤로는 MXC_Delay 를 부르지 않는다 (멈춘다).
    SysTick->CTRL = 0;
    SysTick->LOAD = 0x00FFFFFFu;
    SysTick->VAL = 0;
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk | SysTick_CTRL_ENABLE_Msk;
    printf("measuring melfeat (incremental) ...\n");

    // ── 1. KAT ────────────────────────────────────────────────────────
    const int8_t *w = melkat_window;
    const int tail0 = MEL_OFF_INC + MEL_INC_TAIL; // 122 + 262 = 384
    melfeat_inc_batch(w, feat);
    uint32_t n_diff = 0, n_over = 0;
    int max_diff = 0;
    for (int i = 0; i < N_FEAT; i++) {
        int d = (int)feat[i] - (int)melinc_logmel[i];
        if (d < 0) d = -d;
        if (d) n_diff++;
        if (d > 2) n_over++;
        if (d > max_diff) max_diff = d;
    }
    melfeat_inc_reset(&w[MEL_OFF_INC]);
    for (int k = 0; k < 4; k++) melfeat_inc_push(&w[tail0 + MEL_INC_STEP * k]);
    melfeat_inc_get(feat_inc);
    uint32_t n_inc = 0;
    for (int i = 0; i < N_FEAT; i++)
        if (feat_inc[i] != feat[i]) n_inc++;

    // ── 2. N_ITER 회 — 판단마다 다른 4,000 샘플 블록을 돌려 가며 넣는다 ──
    for (int i = 0; i < N_ITER; i++) {
        const int8_t *blk = &w[tail0 + MEL_INC_STEP * (i & 3)];
        uint32_t c0 = tick();
        melfeat_inc_push(blk);
        uint32_t c1 = tick();
        melfeat_inc_get(feat_inc);
        uint32_t c2 = tick();
        t_push[i] = (c1 - c0) & TICK_MASK;
        t_get[i] = (c2 - c1) & TICK_MASK;
    }

    // ── 3. 참고: 일괄 계산 1회 ────────────────────────────────────────
    uint32_t b0 = tick();
    melfeat_inc_batch(w, feat);
    uint32_t t_batch = (tick() - b0) & TICK_MASK;

    SysTick->CTRL = 0;
    uint32_t per = SystemCoreClock / 1000000;
    while (1) {
        printf("\n=== SafeSound CPU preprocess measure: log-mel incremental (hop 250) ===\n");
        printf("build %s %s  N_ITER %d\n", __DATE__, __TIME__, N_ITER);
        printf("core clock %lu Hz  CNN: disabled  counter: SysTick\n",
               (unsigned long)SystemCoreClock);
        printf("KAT batch vs melfeat.py loginc: mismatches %lu/%d  max diff %d LSB  "
               "over +-2 LSB: %lu  -> %s\n",
               (unsigned long)n_diff, N_FEAT, max_diff, (unsigned long)n_over,
               n_over == 0 ? "PASS" : "FAIL");
        printf("KAT incremental (4 pushes) vs batch: differing %lu/%d  -> %s\n",
               (unsigned long)n_inc, N_FEAT, n_inc == 0 ? "PASS (bit-exact)" : "FAIL");
        report("push", t_push, per);
        report("get", t_get, per);
        printf("batch (64 frames, 1 pass): %lu cycles  %lu us\n", (unsigned long)t_batch,
               (unsigned long)(t_batch / per));
        printf("=== END ===\n");
        for (volatile uint32_t v = 0; v < 40000000u; v++) {}
    }
}
