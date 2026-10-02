// ①′ 전처리 **최적화 판 비교** 측정 펌웨어 (MEASURE_BUILD) — MAX78000FTHR
//
// 변형 넷을 한 펌웨어에서 같은 조건으로 잰다 (POR 한 번):
//   V0  float32 rfft + logf            (기준 구현과 같은 계산 — 최적화 전)
//   V1  float32 rfft + LUT 로그
//   V2  q15 rfft     + LUT 로그
//   V3  q31 rfft     + LUT 로그
// 변형마다
//   · KAT: 일괄 계산 vs melfeat.py "loginc" 기준 — 불일치 수, 최대 차이, ±2 LSB 초과 수
//   · 판단 1회 전처리(push, 새 프레임 16개) 지연 N_ITER 회
//   · 단계별 내역 (별도 1회 통과, 16프레임 합): 프레임 구성 / rfft / 파워+멜 / 로그
//
// 조건: 코어 100 MHz (IPO), NPU 꺼짐, 카운터 SysTick, UART 는 측정 구간 밖.
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
#define N_VAR 4

static const int variants[N_VAR] = { 0, MELV_LUTLOG, MELV_Q15 | MELV_LUTLOG,
                                     MELV_Q31 | MELV_LUTLOG };
static const char *names[N_VAR] = { "V0 f32+logf", "V1 f32+LUT", "V2 q15+LUT", "V3 q31+LUT" };

static int8_t feat[N_FEAT];
static uint32_t t_push[N_ITER];
static struct {
    uint32_t n_diff, n_over;
    int max_diff;
    uint32_t mn, md, mean, mx;
    uint32_t prof[4];
} res[N_VAR];

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
    SysTick->CTRL = 0; // ⚠️ 이 뒤로는 MXC_Delay 를 부르지 않는다 (멈춘다)
    SysTick->LOAD = 0x00FFFFFFu;
    SysTick->VAL = 0;
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk | SysTick_CTRL_ENABLE_Msk;
    printf("measuring melfeat variants ...\n");

    const int8_t *w = melkat_window;
    const int tail0 = MEL_OFF_INC + MEL_INC_TAIL;
    for (int v = 0; v < N_VAR; v++) {
        const int var = variants[v];
        // KAT
        melfeat_inc_batch_v(w, feat, var);
        for (int i = 0; i < N_FEAT; i++) {
            int d = (int)feat[i] - (int)melinc_logmel[i];
            if (d < 0) d = -d;
            if (d) res[v].n_diff++;
            if (d > 2) res[v].n_over++;
            if (d > res[v].max_diff) res[v].max_diff = d;
        }
        // 지연
        melfeat_inc_reset(&w[MEL_OFF_INC]);
        for (int i = 0; i < N_ITER; i++) {
            const int8_t *blk = &w[tail0 + MEL_INC_STEP * (i & 3)];
            uint32_t c0 = tick();
            melfeat_inc_push_v(blk, var, 0, 0, 0);
            t_push[i] = (tick() - c0) & TICK_MASK;
        }
        uint64_t sum = 0;
        for (int i = 0; i < N_ITER; i++) sum += t_push[i];
        qsort(t_push, N_ITER, sizeof(t_push[0]), cmp_u32);
        res[v].mn = t_push[0];
        res[v].md = t_push[N_ITER / 2];
        res[v].mx = t_push[N_ITER - 1];
        res[v].mean = (uint32_t)(sum / N_ITER);
        // 단계별 (1회)
        melfeat_inc_push_v(&w[tail0], var, res[v].prof, tick, TICK_MASK);
    }

    SysTick->CTRL = 0;
    while (1) {
        printf("\n=== SafeSound CPU preprocess: log-mel incremental, variants ===\n");
        printf("build %s %s  N_ITER %d  (push = 16 new frames per decision)\n", __DATE__,
               __TIME__, N_ITER);
        printf("core clock %lu Hz  CNN: disabled  counter: SysTick\n",
               (unsigned long)SystemCoreClock);
        for (int v = 0; v < N_VAR; v++) {
            printf("%s | KAT mismatches %lu/%d  max %d LSB  over+-2: %lu -> %s\n", names[v],
                   (unsigned long)res[v].n_diff, N_FEAT, res[v].max_diff,
                   (unsigned long)res[v].n_over, res[v].n_over == 0 ? "PASS" : "FAIL");
            printf("%s | push cycles min %lu  med %lu  mean %lu  max %lu\n", names[v],
                   (unsigned long)res[v].mn, (unsigned long)res[v].md,
                   (unsigned long)res[v].mean, (unsigned long)res[v].mx);
            printf("%s | stages (16 frames): frame %lu  rfft %lu  power+mel %lu  log %lu\n",
                   names[v], (unsigned long)res[v].prof[0], (unsigned long)res[v].prof[1],
                   (unsigned long)res[v].prof[2], (unsigned long)res[v].prof[3]);
        }
        printf("=== END ===\n");
        for (volatile uint32_t d = 0; d < 40000000u; d++) {}
    }
}
