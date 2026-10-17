// 구성 ②′ (NPU 에너지 적분 필터뱅크) 측정 펌웨어 (MEASURE_BUILD) — MAX78000FTHR
//
// 두 네트워크가 한 가속기에 **동시 적재**돼 있다 (가중치 메모리를 나눠 쓴다):
//   p1  Conv1d(64→64, k=8, wide int32)  — cnn_p1.c  (접두어 p1_, 생성 코드)
//   p2  가 의 2D CNN (1×32×16 → 로짓 5)  — cnn.c     (생성 코드 그대로)
// 판단 1회 (hop 256 ms = 새 4,096 샘플) 의 경로와 측정 구간:
//   load1  새 청크(접기 64×64 HWC) 를 p1 입력 메모리에 적재
//   cfg1   네트워크 전환 → p1 (층 수 레지스터 + p1 층 설정)
//   infer1 p1 시작 ~ 완료 인터럽트
//   cpu    CPU: |cos|+|sin| → 8위치 합 → log2_q8 → int8 (4프레임 × 32밴드) + 32×16 버퍼 밀기 + 입력 워드 포장
//   load2  512 워드를 p2 입력 메모리에 적재
//   cfg2   네트워크 전환 → p2
//   infer2 p2 시작 ~ 완료
//   unload p2 출력 언로드 + softmax
// KAT 셋 (전부 비트 단위): p1 출력 = 합성기 기대값(청크 3), CPU 특징 = 호스트 정수 경로(fe_kat.h), p2 출력 = 합성기 기대값.
// 조건은 다른 구성과 같다: 코어 100 MHz, CNN 50 MHz (PCLK÷1), 바쁜 대기, SysTick, UART 는 측정 구간 밖, 1000회.
// ⚠️ CPU 구간은 NPU 전원이 켜진(유휴) 상태에서 돈다 — 에너지 추정에서 이 구간은 '적재' 전력(P_load)으로 친다 (보수적).
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mxc.h"
#include "cnn.h"             // p2 (생성)
#include "fe_kat.h"          // 생성 (tools/fe_board_prep.py): 청크 입력·기대 특징·주소·상수
#include "sampleoutput.h"    // p2 기대 출력 SAMPLE_OUTPUT (생성)
#include "sampleoutput_p1.h" // p1 기대 출력 P1_SAMPLE_OUTPUT (청크 3, 생성)

#ifndef N_ITER
#define N_ITER 1000
#endif
#ifndef NET_NAME
#define NET_NAME "?"
#endif
// -DFE_DEBUG: KAT 단계마다 UART 로 표식을 찍는다 (측정 밖, 디버그 전용 — 측정 빌드에서는 끈다)
#ifdef FE_DEBUG
#define DBG(...) do { printf(__VA_ARGS__); } while (0)
#else
#define DBG(...) do {} while (0)
#endif

volatile uint32_t cnn_time; // 완료 인터럽트가 세운다 (cnn.c 의 CNN_ISR — 두 네트워크가 같이 쓴다)

// firmware/common/log2_q8.c
int32_t log2_q8(uint32_t v);
int8_t log2_q8_to_int8(int32_t lg, int32_t lo_q8, int32_t span_q8);
// cnn_p1.c (생성 코드에 접두어 p1_ 를 붙인 것)
int p1_cnn_configure(void);
int p1_cnn_load_weights(void);
int p1_cnn_start(void);
int p1_cnn_stop(void);

static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];
static int8_t fb[FE_NBAND][FE_NFRAME];                 // 32×16 특징 버퍼 (b, f)
static uint32_t p2_in[FE_P2_IN_WORDS];                 // p2 입력 워드 (1채널 HWC: 워드당 1픽셀)

enum { S_LOAD1, S_CFG1, S_INFER1, S_CPU, S_LOAD2, S_CFG2, S_INFER2, S_UNLOAD, N_SEG };
static const char *seg_name[N_SEG] = { "load1", "cfg1", "infer1", "cpu", "load2", "cfg2", "infer2", "unload" };
static uint32_t t_seg[N_SEG][N_ITER];
static uint32_t t_total[N_ITER];

// ── 사이클 카운터: measure_main.c 와 같다 (MAX78000 M4 는 DWT CYCCNT 가 없어 SysTick) ──
static int use_systick;
static uint32_t tick_mask = 0xFFFFFFFFu;

static inline uint32_t tick(void)
{
    return use_systick ? (0x00FFFFFFu - SysTick->VAL) : DWT->CYCCNT;
}

static void dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    uint32_t a = DWT->CYCCNT;
    for (volatile int i = 0; i < 1000; i++) {}
    if (DWT->CYCCNT == a) {
        use_systick = 1;
        tick_mask = 0x00FFFFFFu;
        SysTick->CTRL = 0;
        SysTick->LOAD = 0x00FFFFFFu;
        SysTick->VAL = 0;
        SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk | SysTick_CTRL_ENABLE_Msk;
    }
}

static int cmp_u32(const void *a, const void *b)
{
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static void report(const char *name, uint32_t *v, uint32_t hz)
{
    uint64_t sum = 0;
    for (int i = 0; i < N_ITER; i++) sum += v[i];
    qsort(v, N_ITER, sizeof(v[0]), cmp_u32);
    uint32_t mn = v[0], md = v[N_ITER / 2], mx = v[N_ITER - 1];
    uint32_t mean = (uint32_t)(sum / N_ITER);
    uint32_t per = hz / 1000000;
    printf("%-7s cycles min %lu  med %lu  mean %lu  max %lu\n", name,
           (unsigned long)mn, (unsigned long)md, (unsigned long)mean, (unsigned long)mx);
    printf("%-7s us     min %lu.%02lu  med %lu.%02lu  max %lu.%02lu\n", name,
           (unsigned long)(mn / per), (unsigned long)((mn % per) * 100 / per),
           (unsigned long)(md / per), (unsigned long)((md % per) * 100 / per),
           (unsigned long)(mx / per), (unsigned long)((mx % per) * 100 / per));
}

// ── 네트워크 전환: 층 수 레지스터(4 사분면) + 그 네트워크의 층 설정 ──────────
static inline void set_layer_count(uint32_t n)
{
    *((volatile uint32_t *)0x50100008) = n;
    *((volatile uint32_t *)0x50500008) = n;
    *((volatile uint32_t *)0x50900008) = n;
    *((volatile uint32_t *)0x50d00008) = n;
}

// 생성 cnn_configure 는 0 이 아닌 레지스터만 쓴다 (리셋 상태를 가정). 두 네트워크가 같은 층 0 슬롯을 쓰므로
// 한쪽만 쓰는 레지스터(p1 의 1D·후처리, p2 의 Columns·TRAM ptr)가 남는다 → 전환 전에 합집합을 0 으로 (2026-10-17 보드 KAT 로 확인).
static inline void clear_l0_regs(void)
{
    static const uint32_t qbase[4] = { 0x50100000u, 0x50500000u, 0x50900000u, 0x50d00000u };
    for (int q = 0; q < 4; q++)
        for (int i = 0; i < FE_N_L0_REGS; i++)
            *((volatile uint32_t *)(qbase[q] + FE_L0_REGS[i])) = 0;
}

static inline void switch_to_p1(void)
{
    set_layer_count(FE_P1_LAYER_COUNT);
    clear_l0_regs();
    p1_cnn_configure();
}

static inline void switch_to_p2(void)
{
    set_layer_count(FE_P2_LAYER_COUNT);
    clear_l0_regs();
    cnn_configure();
}

// ── 입력 적재 (생성 load_input 과 같은 memcpy32 — 청크는 호스트가 HWC 워드로 포장) ──
static inline void load_chunk(int c)
{
    for (int g = 0; g < FE_P1_IN_GROUPS; g++)
        memcpy32((uint32_t *)FE_P1_IN_ADDR[g], fe_chunk_in[c][g], FE_P1_IN_WORDS);
}

// ── 기대 출력 대조 (합성기 check_output 과 같은 형식: addr, mask, len, 값…, 0) ──
static int check_words(const uint32_t *ptr)
{
    uint32_t mask, len;
    volatile uint32_t *addr;
    while ((addr = (volatile uint32_t *)*ptr++) != 0) {
        mask = *ptr++;
        len = *ptr++;
        for (uint32_t i = 0; i < len; i++)
            if ((*addr++ & mask) != *ptr++) return CNN_FAIL;
    }
    return CNN_OK;
}

static const uint32_t p1_sample_output[] = P1_SAMPLE_OUTPUT;
static const uint32_t p2_sample_output[] = SAMPLE_OUTPUT;

// ── CPU 단: p1 출력(int32, 메모리에 그대로)을 읽어 프레임 4개의 밴드 에너지 → 정수 로그 → int8 ──
//   채널 c 의 위치 p 워드: 그룹 g = c>>2 의 기준 주소 + (4p + (c&3))·4.  밴드 b 는 채널 2b(cos)·2b+1(sin) — 같은 그룹.
static void fe_cpu_frames(int first_frame)
{
    for (int b = 0; b < FE_NBAND; b++) {
        const int g = (2 * b) >> 2, slot = (2 * b) & 3;
        const volatile int32_t *base = (const volatile int32_t *)FE_P1_OUT_ADDR[g];
        for (int f = 0; f < FE_FRAMES_PER_HOP; f++) {
            const volatile int32_t *p = base + (FE_POS_PER_FRAME * f) * 4 + slot;
            uint32_t e = 0;
            for (int k = 0; k < FE_POOL_POS; k++) {
                int32_t y0 = p[0], y1 = p[1];
                e += (uint32_t)(y0 < 0 ? -y0 : y0) + (uint32_t)(y1 < 0 ? -y1 : y1);
                p += 4;
            }
            fb[b][first_frame + f] = log2_q8_to_int8(log2_q8(e), FE_LO_Q8, FE_SPAN_Q8);
        }
    }
}

// 판단 1회의 CPU 몫: 버퍼 밀기(4프레임) + 새 4프레임 + p2 입력 워드 포장
static void fe_cpu_hop(void)
{
    for (int b = 0; b < FE_NBAND; b++)
        memmove(fb[b], fb[b] + FE_FRAMES_PER_HOP, FE_NFRAME - FE_FRAMES_PER_HOP);
    fe_cpu_frames(FE_NFRAME - FE_FRAMES_PER_HOP);
    const int8_t *q = &fb[0][0];
    for (int i = 0; i < FE_P2_IN_WORDS; i++) p2_in[i] = (uint8_t)q[i];
}

static inline void run_and_wait_p1(void)
{
    p1_cnn_start();
    while (cnn_time == 0) {}
}

static inline void run_and_wait_p2(void)
{
    cnn_start();
    while (cnn_time == 0) {}
}

int main(void)
{
    MXC_ICC_Enable(MXC_ICC0);
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO); // 100 MHz
    SystemCoreClockUpdate();

    printf("Waiting...\n");
    MXC_Delay(SEC(2)); // 디버거가 끼어들 틈 (지우지 말 것)

    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1); // CNN 50 MHz
    cnn_init();
    cnn_load_weights();     // p2 (커널 0 ~ FE_P2_WEIGHT_END)
    DBG("[dbg] p2 weights loaded\n");
    p1_cnn_load_weights();  // p1 (그 뒤) — 겹치지 않는다 (fe-design.md 3절)
    DBG("[dbg] p1 weights loaded\n");
    cnn_load_bias();
    dwt_init();
    printf("measuring %s ...\n", NET_NAME);

    // ── 1. KAT: 창 전체 (청크 4개 → 16프레임) → p2 ────────────────────────
    int kat_p1 = CNN_FAIL, kat_p2;
    for (int c = 0; c < FE_NCHUNK; c++) {
        DBG("[dbg] chunk %d load\n", c);
        load_chunk(c);
        DBG("[dbg] switch p1\n");
        switch_to_p1();
        DBG("[dbg] p1 start\n");
        run_and_wait_p1();
        DBG("[dbg] p1 done (cnn_time %lu)\n", (unsigned long)cnn_time);
        if (c == FE_NCHUNK - 1) kat_p1 = check_words(p1_sample_output);
        DBG("[dbg] p1 check ok, cpu frames\n");
        fe_cpu_frames(c * FE_FRAMES_PER_HOP);
        DBG("[dbg] cpu done, stop\n");
        p1_cnn_stop();
    }
    uint32_t feat_bad = 0;
    for (int i = 0; i < FE_NBAND * FE_NFRAME; i++)
        if ((&fb[0][0])[i] != fe_q_expected[i]) feat_bad++;
    for (int i = 0; i < FE_P2_IN_WORDS; i++) p2_in[i] = (uint8_t)(&fb[0][0])[i];
    DBG("[dbg] feat_bad %lu, load p2\n", (unsigned long)feat_bad);
    memcpy32((uint32_t *)FE_P2_IN_ADDR, p2_in, FE_P2_IN_WORDS);
    DBG("[dbg] switch p2\n");
    switch_to_p2();
    DBG("[dbg] p2 start\n");
    run_and_wait_p2();
    DBG("[dbg] p2 done\n");
    kat_p2 = check_words(p2_sample_output);
    DBG("[dbg] p2 check %d\n", kat_p2);
    cnn_unload((uint32_t *)ml_data);
    DBG("[dbg] unload: %ld %ld %ld %ld %ld (expected %ld %ld %ld %ld %ld)\n", (long)ml_data[0], (long)ml_data[1], (long)ml_data[2],
        (long)ml_data[3], (long)ml_data[4], (long)fe_logits_expected[0], (long)fe_logits_expected[1], (long)fe_logits_expected[2],
        (long)fe_logits_expected[3], (long)fe_logits_expected[4]);
    softmax_q17p14_q15((const q31_t *)ml_data, CNN_NUM_OUTPUTS, ml_softmax);
    DBG("[dbg] softmax done\n");
    cnn_stop();
    static int32_t kat_raw[CNN_NUM_OUTPUTS];
    for (int i = 0; i < CNN_NUM_OUTPUTS; i++) kat_raw[i] = ml_data[i];
    DBG("[dbg] loop start\n");

    // ── 2. 판단 1회 × N_ITER ───────────────────────────────────────────────
    //   매 회 "직전 12프레임이 버퍼에 있고 새 청크(청크 3)가 들어온다" 는 상태를 재현한다 —
    //   측정 구간 밖에서 버퍼의 슬롯 4..15 에 기대 특징의 프레임 0..11 을 넣어, 4프레임 밀기 뒤 새 4프레임을 더하면
    //   버퍼가 기대 특징(창 전체)과 같아지게 한다 → p2 입력이 매 회 같고 비트 대조가 된다.
    uint32_t bad_p1 = 0, bad_feat = 0, bad_p2 = 0;
    for (int i = 0; i < N_ITER; i++) {
        for (int b = 0; b < FE_NBAND; b++)                                     // 상태 복원 (측정 밖)
            memcpy(fb[b] + FE_FRAMES_PER_HOP, fe_q_expected + b * FE_NFRAME, FE_NFRAME - FE_FRAMES_PER_HOP);
        uint32_t c0 = tick();
        load_chunk(FE_NCHUNK - 1);
        uint32_t c1 = tick();
        switch_to_p1();
        uint32_t c2 = tick();
        run_and_wait_p1();
        uint32_t c3 = tick();
        if (check_words(p1_sample_output) != CNN_OK) bad_p1++; // 측정 밖
        uint32_t c4 = tick();
        fe_cpu_hop();
        uint32_t c5 = tick();
        if (memcmp(fb, fe_q_expected, sizeof(fb)) != 0) bad_feat++; // 측정 밖
        p1_cnn_stop();
        uint32_t c6 = tick();
        memcpy32((uint32_t *)FE_P2_IN_ADDR, p2_in, FE_P2_IN_WORDS);
        uint32_t c7 = tick();
        switch_to_p2();
        uint32_t c8 = tick();
        run_and_wait_p2();
        uint32_t c9 = tick();
        if (check_words(p2_sample_output) != CNN_OK) bad_p2++; // 측정 밖
        uint32_t c10 = tick();
        cnn_unload((uint32_t *)ml_data);
        softmax_q17p14_q15((const q31_t *)ml_data, CNN_NUM_OUTPUTS, ml_softmax);
        uint32_t c11 = tick();
        cnn_stop();
        t_seg[S_LOAD1][i] = (c1 - c0) & tick_mask;
        t_seg[S_CFG1][i] = (c2 - c1) & tick_mask;
        t_seg[S_INFER1][i] = (c3 - c2) & tick_mask;
        t_seg[S_CPU][i] = (c5 - c4) & tick_mask;
        t_seg[S_LOAD2][i] = (c7 - c6) & tick_mask;
        t_seg[S_CFG2][i] = (c8 - c7) & tick_mask;
        t_seg[S_INFER2][i] = (c9 - c8) & tick_mask;
        t_seg[S_UNLOAD][i] = (c11 - c10) & tick_mask;
        uint32_t tot = 0;
        for (int s = 0; s < N_SEG; s++) tot += t_seg[s][i];
        t_total[i] = tot;
        if (i == 0 || i == N_ITER - 1) DBG("[dbg] iter %d total %lu cycles (bad p1 %lu feat %lu p2 %lu)\n", i, (unsigned long)tot,
                                           (unsigned long)bad_p1, (unsigned long)bad_feat, (unsigned long)bad_p2);
    }
    SysTick->CTRL = 0; // 이후 MXC_Delay 금지 (measure_main.c 참조)
    cnn_disable();

    while (1) {
        printf("\n=== SafeSound NPU measure: %s ===\n", NET_NAME);
        printf("build %s %s  SystemCoreClock %lu Hz  N_ITER %d\n", __DATE__, __TIME__,
               (unsigned long)SystemCoreClock, N_ITER);
        printf("two networks co-resident: p1 Conv1d 64->64 k8 wide (weights %lu B) + p2 (weights %lu B)\n",
               (unsigned long)FE_P1_WEIGHT_BYTES, (unsigned long)FE_P2_WEIGHT_BYTES);
        printf("KAT p1 sampleoutput (chunk %d): %s\n", FE_NCHUNK - 1, kat_p1 == CNN_OK ? "PASS (bit-exact)" : "FAIL");
        printf("KAT cpu feature vs host integer path: %s (%lu / %d mismatches)\n",
               feat_bad == 0 ? "PASS (bit-exact)" : "FAIL", (unsigned long)feat_bad, FE_NBAND * FE_NFRAME);
        printf("KAT sampleoutput: %s\n", kat_p2 == CNN_OK ? "PASS (bit-exact)" : "FAIL");
        for (int i = 0; i < CNN_NUM_OUTPUTS; i++)
            printf("  class %d: raw %ld\n", i, (long)kat_raw[i]);
        printf("core clock %lu Hz  CNN clock PCLK %lu Hz / 1  wait: busy (no sleep)\n",
               (unsigned long)SystemCoreClock, (unsigned long)PeripheralClock);
        printf("counter: %s\n", use_systick ? "SysTick (DWT CYCCNT not running)" : "DWT CYCCNT");
        printf("%d iterations, output mismatches: %lu  (p1 %lu, feature %lu, p2 %lu)\n", N_ITER,
               (unsigned long)(bad_p1 + bad_feat + bad_p2), (unsigned long)bad_p1,
               (unsigned long)bad_feat, (unsigned long)bad_p2);
        for (int s = 0; s < N_SEG; s++) report(seg_name[s], t_seg[s], SystemCoreClock);
        report("total", t_total, SystemCoreClock);
        printf("=== END ===\n");
        for (volatile uint32_t w = 0; w < 40000000u; w++) {}
    }
}
