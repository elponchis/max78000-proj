// M4 대조군 측정 펌웨어 (MEASURE_BUILD) — ④ 가중치의 소프트웨어 추론.
//
// ⚠️ **단순 C 참조 구현(-O2)** 이다. CMSIS-NN 이 아니다. 표에 그렇게 적는다.
//
// 하는 일
//   1. KAT: 합성 KAT 입력 → 로짓이 **보드 NPU 가 낸 값**(m4_kat_output)과 같은가
//   2. N_ITER 회 지연. 구간: 입력 적재(memcpy 16,384 B) / 추론
//
// 조건: 코어 100 MHz (IPO), **NPU 꺼짐**, UART 는 측정 구간 밖.
// 시간은 하드웨어 타이머 TMR0 (MSDK `MXC_TMR_SW_Start/Stop`, 1 µs 단위)로 잰다 —
// 추론이 SysTick 24bit 한계(167 ms)를 넘기 때문이다.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mxc.h"
#include "m4ref.h"
#include "m4_weights.h"

#ifndef N_ITER
#define N_ITER 1000
#endif

// 구현 표기 — 표에 그대로 옮긴다. -DM4_CMSIS 면 m4cmsis.c (CMSIS-NN) 를 링크한다.
#ifdef M4_CMSIS
#define M4_IMPL_NAME "CMSIS-NN (arm_convolve_wrapper_s8, SINGLE_ROUNDING), -O2"
extern int m4cmsis_status;
#else
#define M4_IMPL_NAME "plain C reference (-O2), NOT CMSIS-NN"
#endif

static int8_t win[M4_WIN];
static int32_t logits[M4_NUM_OUTPUTS];
static uint32_t t_load[N_ITER], t_infer[N_ITER];

static int cmp_u32(const void *a, const void *b)
{
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static void report(const char *name, uint32_t *v)
{
    uint64_t sum = 0;
    for (int i = 0; i < N_ITER; i++) sum += v[i];
    qsort(v, N_ITER, sizeof(v[0]), cmp_u32);
    printf("%-6s us  min %lu  med %lu  mean %lu  max %lu\n", name, (unsigned long)v[0],
           (unsigned long)v[N_ITER / 2], (unsigned long)(sum / N_ITER),
           (unsigned long)v[N_ITER - 1]);
}

int main(void)
{
    MXC_ICC_Enable(MXC_ICC0);
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO); // 100 MHz
    SystemCoreClockUpdate();

    printf("Waiting...\n");
    MXC_Delay(SEC(2)); // 디버거가 끼어들 틈 (지우지 말 것)
    printf("measuring M4 software inference: %s (%d iterations) ...\n", M4_IMPL_NAME, N_ITER);

    // ── 1. KAT ────────────────────────────────────────────────────────
    m4ref_infer(m4_kat_input, logits);
    int kat = 1;
    static int32_t kat_raw[M4_NUM_OUTPUTS];
    for (int i = 0; i < M4_NUM_OUTPUTS; i++) {
        kat_raw[i] = logits[i];
        if (logits[i] != m4_kat_output[i]) kat = 0;
    }
    printf("KAT %s\n", kat ? "PASS" : "FAIL");

    // ── 2. N_ITER 회 ──────────────────────────────────────────────────
    uint32_t bad = 0;
    for (int i = 0; i < N_ITER; i++) {
        MXC_TMR_SW_Start(MXC_TMR0);
        memcpy(win, m4_kat_input, M4_WIN);
        t_load[i] = MXC_TMR_SW_Stop(MXC_TMR0);

        MXC_TMR_SW_Start(MXC_TMR0);
        m4ref_infer(win, logits);
        t_infer[i] = MXC_TMR_SW_Stop(MXC_TMR0);

        for (int k = 0; k < M4_NUM_OUTPUTS; k++)
            if (logits[k] != m4_kat_output[k]) { bad++; break; }
        if ((i + 1) % 100 == 0) printf("  %d/%d\n", i + 1, N_ITER);
    }

    while (1) {
        printf("\n=== SafeSound M4 software inference: safesound_wave weights ===\n");
        printf("implementation: %s\n", M4_IMPL_NAME);
#ifdef M4_CMSIS
        printf("CMSIS-NN status flags: %d (0 = ok)\n", m4cmsis_status);
#endif
        printf("build %s %s  N_ITER %d\n", __DATE__, __TIME__, N_ITER);
        printf("core clock %lu Hz  CNN: disabled  timer: TMR0 (1 us)\n",
               (unsigned long)SystemCoreClock);
        printf("KAT vs board NPU output: %s\n", kat ? "PASS (bit-exact)" : "FAIL");
        for (int i = 0; i < M4_NUM_OUTPUTS; i++)
            printf("  class %d: raw %ld\n", i, (long)kat_raw[i]);
        printf("%d iterations, output mismatches: %lu\n", N_ITER, (unsigned long)bad);
        report("load", t_load);
        report("infer", t_infer);
        printf("=== END ===\n");
        MXC_Delay(SEC(5));
    }
}
