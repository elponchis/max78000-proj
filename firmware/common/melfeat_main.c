// 구성 ①의 CPU 전처리(로그 멜) 측정 펌웨어 (MEASURE_BUILD) — MAX78000FTHR
//
// 하는 일
//   1. KAT: `melkat_window` 를 전처리에 넣고 `melkat_logmel`(melfeat.py 기준)과
//      비교한다 — 불일치 수, 최대 차이, ±2 LSB 초과 수
//   2. 전처리 한 번(1초 창 → 64×64 int8)의 지연을 N_ITER 회 잰다
//   3. 단계별 내역(프레임 구성 / rfft / 파워+멜 / 로그) — 별도 1회 통과
//
// 조건: 코어 100 MHz (IPO), **NPU 꺼짐** (cnn_enable 을 부르지 않는다),
// 카운터 SysTick (이 칩에는 DWT CYCCNT 가 없다), UART 는 측정 구간 밖.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "mxc.h"
#include "melfeat.h"
#include "mel_tables.h"
#include "melkat_vectors.h"

#ifndef N_ITER
#define N_ITER 1000
#endif
#define TICK_MASK 0x00FFFFFFu

static int8_t feat[MEL_N_MELS * MEL_N_FRAMES];
static uint32_t t_pre[N_ITER];

static uint32_t tick(void)
{
    return 0x00FFFFFFu - SysTick->VAL;
}

static int cmp_u32(const void *a, const void *b)
{
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
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
    printf("measuring melfeat ...\n");

    // ── 1. KAT ────────────────────────────────────────────────────────
    melfeat_int8(melkat_window, feat);
    uint32_t n_diff = 0, n_over = 0;
    int max_diff = 0;
    for (int i = 0; i < MEL_N_MELS * MEL_N_FRAMES; i++) {
        int d = (int)feat[i] - (int)melkat_logmel[i];
        if (d < 0) d = -d;
        if (d) n_diff++;
        if (d > 2) n_over++;
        if (d > max_diff) max_diff = d;
    }

    // ── 2. N_ITER 회 ──────────────────────────────────────────────────
    for (int i = 0; i < N_ITER; i++) {
        uint32_t c0 = tick();
        melfeat_int8(melkat_window, feat);
        uint32_t c1 = tick();
        t_pre[i] = (c1 - c0) & TICK_MASK;
    }
    uint64_t sum = 0;
    for (int i = 0; i < N_ITER; i++) sum += t_pre[i];
    qsort(t_pre, N_ITER, sizeof(t_pre[0]), cmp_u32);

    // ── 3. 단계별 내역 (1회) ──────────────────────────────────────────
    uint32_t prof[4] = { 0, 0, 0, 0 };
    melfeat_int8_prof(melkat_window, feat, prof, tick, TICK_MASK);

    SysTick->CTRL = 0;
    uint32_t per = SystemCoreClock / 1000000;
    while (1) {
        printf("\n=== SafeSound CPU preprocess measure: log-mel (melfeat) ===\n");
        printf("build %s %s  N_ITER %d\n", __DATE__, __TIME__, N_ITER);
        printf("core clock %lu Hz  CNN: disabled  counter: SysTick\n",
               (unsigned long)SystemCoreClock);
        printf("KAT vs melfeat.py: mismatches %lu/%d  max diff %d LSB  over +-2 LSB: %lu  -> %s\n",
               (unsigned long)n_diff, MEL_N_MELS * MEL_N_FRAMES, max_diff,
               (unsigned long)n_over, n_over == 0 ? "PASS" : "FAIL");
        printf("preproc cycles min %lu  med %lu  mean %lu  max %lu\n",
               (unsigned long)t_pre[0], (unsigned long)t_pre[N_ITER / 2],
               (unsigned long)(sum / N_ITER), (unsigned long)t_pre[N_ITER - 1]);
        printf("preproc us     min %lu  med %lu  max %lu\n",
               (unsigned long)(t_pre[0] / per), (unsigned long)(t_pre[N_ITER / 2] / per),
               (unsigned long)(t_pre[N_ITER - 1] / per));
        printf("stages (1 pass, cycles): frame %lu  rfft %lu  power+mel %lu  log+quant %lu\n",
               (unsigned long)prof[0], (unsigned long)prof[1], (unsigned long)prof[2],
               (unsigned long)prof[3]);
        printf("=== END ===\n");
        for (volatile uint32_t w = 0; w < 40000000u; w++) {}
    }
}
