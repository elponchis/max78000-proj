// SafeSound NPU 측정 펌웨어 (MEASURE_BUILD) — MAX78000FTHR
//
// 하는 일
//   1. KAT: 합성기가 낸 샘플 입력을 넣고 NPU 출력이 `sampleoutput.h` 와
//      **비트 단위로 같은지** 본다 (G4 의 전제 — 보드가 PC 와 같은 계산을 한다)
//   2. DWT 사이클 카운터로 N_ITER 회 추론 지연을 잰다 (CLAUDE.md 7장:
//      1000회, 최악값 기준 보고). 매 회 출력 대조도 한다
//
// 구간 (각각 최소·중앙·최악):
//   load   — 입력을 NPU 데이터 메모리에 적재 (`load_input`)
//   infer  — `cnn_start()` 직전 ~ 완료 인터럽트가 플래그를 세운 직후
//   unload — `cnn_unload` + softmax
//
// ⚠️ 추론 대기는 **바쁜 대기**다. 슬립(WFI)에 들어가면 코어 클럭이 멎어
//    DWT CYCCNT 도 멎으므로 이 방법으로는 잴 수 없다. 슬립 판의 지연·전력은
//    외부 계측(GPIO 마킹)으로 따로 잰다.
// ⚠️ UART 출력은 **측정 구간 밖**에서만 한다. LED·부저·SD 는 쓰지 않는다.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "mxc.h"
#include "cnn.h"

#ifndef N_ITER
#define N_ITER 1000
#endif
#ifndef NET_NAME
#define NET_NAME "?"
#endif

volatile uint32_t cnn_time; // 완료 인터럽트가 세운다 (cnn.c)

void load_input(void);   // kat_io.c (합성기 생성)
int check_output(void);  // kat_io.c (합성기 생성)

static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

static uint32_t t_load[N_ITER], t_infer[N_ITER], t_unload[N_ITER];

// 사이클 카운터. DWT CYCCNT 를 먼저 시도하고, **돌지 않으면 SysTick** 을 쓴다.
// 2026-10-02 실측: **MAX78000 의 M4 에는 CYCCNT 가 없다** — `DWT->CTRL` 이
// 0x4f000001 로 NOCYCCNT(bit 25)=1 이고 CYCCNT 는 항상 0 이다. 따라서 실제로는
// 늘 SysTick 이다. SysTick 도 코어 클럭(100 MHz)을 세므로 단위는 같다 —
// 24bit 라 167 ms 에서 감기지만 우리가 재는 구간은 그보다 훨씬 짧다.
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
        SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk | SysTick_CTRL_ENABLE_Msk; // 인터럽트 없음
    }
}

static int cmp_u32(const void *a, const void *b)
{
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

// 정렬해서 최소·중앙·최악을 찍는다. us 는 소수 둘째 자리까지 정수 연산으로.
static void report(const char *name, uint32_t *v, uint32_t hz)
{
    uint64_t sum = 0;
    for (int i = 0; i < N_ITER; i++) sum += v[i];
    qsort(v, N_ITER, sizeof(v[0]), cmp_u32);
    uint32_t mn = v[0], md = v[N_ITER / 2], mx = v[N_ITER - 1];
    uint32_t mean = (uint32_t)(sum / N_ITER);
    uint32_t per = hz / 1000000; // 사이클/us
    printf("%-7s cycles min %lu  med %lu  mean %lu  max %lu\n", name,
           (unsigned long)mn, (unsigned long)md, (unsigned long)mean, (unsigned long)mx);
    printf("%-7s us     min %lu.%02lu  med %lu.%02lu  max %lu.%02lu\n", name,
           (unsigned long)(mn / per), (unsigned long)((mn % per) * 100 / per),
           (unsigned long)(md / per), (unsigned long)((md % per) * 100 / per),
           (unsigned long)(mx / per), (unsigned long)((mx % per) * 100 / per));
}

int main(void)
{
    MXC_ICC_Enable(MXC_ICC0); // 캐시
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO); // 100 MHz
    SystemCoreClockUpdate();
    // ⚠️ 플래싱 직후 시리얼이 `*` 만 찍히면 **POR**(USB 재인가)이 필요하다
    //    (docs/board-bringup.md 2′절). 여기서 `Console_Init()` 을 다시 불러
    //    보았으나 **고쳐지지 않았다** (2026-10-02) — 펌웨어로 우회되지 않는다.

    printf("Waiting...\n");
    MXC_Delay(SEC(2)); // 디버거가 끼어들 틈 (지우지 말 것)

    // CNN 클럭: PCLK(50 MHz) / 1
    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_init();
    cnn_load_weights();
    cnn_load_bias();
    cnn_configure();
    dwt_init();
    printf("measuring %s ...\n", NET_NAME);

    // ── 1. KAT ────────────────────────────────────────────────────────
    load_input();
    cnn_start();
    while (cnn_time == 0) {}
    int kat = check_output();
    cnn_unload((uint32_t *)ml_data);
    softmax_q17p14_q15((const q31_t *)ml_data, CNN_NUM_OUTPUTS, ml_softmax);
    cnn_stop();
    static int32_t kat_raw[CNN_NUM_OUTPUTS];
    for (int i = 0; i < CNN_NUM_OUTPUTS; i++) kat_raw[i] = ml_data[i];

    // ── 2. DWT N_ITER 회 ──────────────────────────────────────────────
    uint32_t bad = 0;
    for (int i = 0; i < N_ITER; i++) {
        uint32_t c0 = tick();
        load_input();
        uint32_t c1 = tick();
        cnn_start();
        while (cnn_time == 0) {}
        uint32_t c2 = tick();
        if (check_output() != CNN_OK) bad++; // 측정 구간 밖
        uint32_t c3 = tick();
        cnn_unload((uint32_t *)ml_data);
        softmax_q17p14_q15((const q31_t *)ml_data, CNN_NUM_OUTPUTS, ml_softmax);
        uint32_t c4 = tick();
        cnn_stop();
        t_load[i] = (c1 - c0) & tick_mask;
        t_infer[i] = (c2 - c1) & tick_mask;
        t_unload[i] = (c4 - c3) & tick_mask;
    }
    // ⚠️ 측정 뒤에는 **MXC_Delay 를 쓰지 않는다.** 우리가 SysTick 을 인터럽트
    //    없이 켜 둔 상태에서 MXC_Delay 를 부르면 오버플로 인터럽트를 기다리며
    //    영원히 멈춘다 (2026-10-02, 디버거로 pc 가 MXC_Delay 안인 것을 확인).
    SysTick->CTRL = 0;

    cnn_disable();

    // 결과는 **5초마다 통째로 다시 찍는다** — 시리얼을 늦게 열어도 읽힌다.
    // (측정은 이미 끝났다. 다시 재려면 리셋)
    while (1) {
        printf("\n=== SafeSound NPU measure: %s ===\n", NET_NAME);
        printf("build %s %s  SystemCoreClock %lu Hz  N_ITER %d\n", __DATE__, __TIME__,
               (unsigned long)SystemCoreClock, N_ITER);
        printf("KAT sampleoutput: %s\n", kat == CNN_OK ? "PASS (bit-exact)" : "FAIL");
        for (int i = 0; i < CNN_NUM_OUTPUTS; i++)
            printf("  class %d: raw %ld\n", i, (long)kat_raw[i]);
        printf("counter: %s\n", use_systick ? "SysTick (DWT CYCCNT not running)" : "DWT CYCCNT");
        printf("%d iterations, output mismatches: %lu\n", N_ITER, (unsigned long)bad);
        report("load", t_load, SystemCoreClock);
        report("infer", t_infer, SystemCoreClock);
        report("unload", t_unload, SystemCoreClock);
        printf("=== END ===\n");
        for (volatile uint32_t w = 0; w < 40000000u; w++) {} // 수 초 (바쁜 대기)
    }
}
