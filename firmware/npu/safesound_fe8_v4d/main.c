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
//          (fe8, FE_P1_BITS8: |·|·합은 NPU 층 A·B 가 했다 → 프레임 시작 위치의 int32 한 워드씩 n×F′ 개만 읽어 log2_q8 → int8)
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
// ②′ 변형 (fe-curve-design.md): b64 는 p1 출력 128 채널(32 그룹), f31 계열은 p1 입력 71 위치(직전 8 + 새 63)·홉당 8 프레임이고
// 청크 c 의 첫 프레임 번호가 8c − 1 이다 (FE_FRAME0_OFFSET −1; KAT 의 청크 0 에서 프레임 −1 은 버린다). 옛 fe_kat.h 와의 호환 기본값.
#ifndef FE_P1_OUT_GROUPS
#define FE_P1_OUT_GROUPS FE_P1_IN_GROUPS
#endif
#ifndef FE_P1_OUT_CH
#define FE_P1_OUT_CH (4 * FE_P1_OUT_GROUPS)
#endif
#ifndef FE_FRAME0_OFFSET
#define FE_FRAME0_OFFSET 0
#endif
// p1 출력 배치 (합성기): 그룹 g(프로세서 4g..4g+3) 블록에 위치 p 의 워드가 FE_P1_OUT_STRIDE 개 — 64 채널은 4 (채널 4g+k),
// 128 채널은 8 (k<4: 채널 4g+k, k≥4: 채널 64+4g+k−4; 프로세서당 2채널). tools/fe_board_prep.py 가 기대 출력으로 검증한다.
#ifndef FE_P1_OUT_STRIDE
#define FE_P1_OUT_STRIDE 4
#endif
// fe8 (후속 연구, docs/results/npu-readback-design.md 7절, tools/fe_board_prep.py --fe8): p1 이 **2층**이다 — 층 A Conv1d(64→2n, k=8) + Abs (8비트)
// → 층 B Conv1d(2n→n, k=8, 가중치 64/128 고정, wide int32) 가 cos+sin 의 8위치 합 E′·2^14 = 64·Σa 를 NPU 안에서 낸다. CPU 는 프레임 시작 위치
// (FE_POS_PER_FRAME·f) 의 int32 **한 워드**만 읽어 로그를 취한다 (판단당 n×F′ 워드; ②′ 는 2n×8F′ 워드를 읽어 |·|·합을 했다). 이 빌드에서
// FE_P1_OUT_ADDR/GROUPS/STRIDE 는 층 B 출력(n 채널 → 그룹 n/4, 위치당 4 워드, 채널 b = 그룹 b>>2 슬롯 b&3)이고 FE_LO_Q8 에 −256 이 이미 들어 있다.
// 두 네트워크가 공유하는 층 슬롯이 2개가 되므로 FE_L0_REGS 는 층 0·1 의 합집합이다 (fe_kat.h 가 만든다). 지연·에너지는 보드 실측 전까지 추정.
#ifndef FE_P1_BITS8
#define FE_P1_BITS8 0
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
#ifdef FE_LOG_AB
static uint32_t t_seg_a[N_SEG][N_ITER], t_total_a[N_ITER], bad_a[3];   // 패스 A(옛 로그) 보관 — 부팅 직후 출력은 시리얼이 놓치므로 while(1) 에서 되풀이한다
#endif

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
#ifdef FE_DIAG
static uint32_t dg_lc_max[4], dg_lc_rewrite[4], dg_lc_to[4];   // 층 수 레지스터 확인: 최대 읽기 횟수 / 다시 쓴 횟수 / 상한 도달
#endif
static const uint32_t QBASE_LC[4] = { 0x50100008u, 0x50500008u, 0x50900008u, 0x50d00008u };
static inline void set_layer_count(uint32_t n)
{
    for (int q = 0; q < 4; q++) *((volatile uint32_t *)QBASE_LC[q]) = n;
#ifdef FE_LC_VERIFY
    // (진단용, 기본 꺼짐 — 2026-10-20 b64 50,000회에서 한 번도 어긋나지 않았다) 쓴 값이 읽힐 때까지 기다린다 — 완료 직후 한동안 사분면이 APB 쓰기를 바로 받지 않는 창이 있다 (2026-10-20 b64, fe_cnn_isr 참조).
    // 읽기 64회마다 한 번만 다시 쓴다 (그 창에서 쓰기를 거듭하면 해롭다). 상한 512.
    for (int q = 0; q < 4; q++) {
        uint32_t k = 0, rw = 0;
        while (*((volatile uint32_t *)QBASE_LC[q]) != n && k < 512u) { if ((k & 63u) == 63u) { *((volatile uint32_t *)QBASE_LC[q]) = n; rw++; } k++; }
#ifdef FE_DIAG
        if (k > dg_lc_max[q]) dg_lc_max[q] = k; dg_lc_rewrite[q] += rw; if (k >= 512u) dg_lc_to[q]++;
#endif
    }
#endif
}

// 생성 cnn_configure 는 0 이 아닌 레지스터만 쓴다 (리셋 상태를 가정). 두 네트워크가 같은 층 0 슬롯을 쓰므로
// 한쪽만 쓰는 레지스터(p1 의 1D·후처리, p2 의 Columns·TRAM ptr)가 남는다 → 전환 전에 합집합을 0 으로 (2026-10-17 보드 KAT 로 확인).
// fe8 은 p1 이 2층이라 합집합이 층 0·1 (FE_N_SHARED_LAYERS) 에 걸친다 — fe_kat.h 의 FE_L0_REGS 가 그 합집합이다.
static inline void clear_l0_regs(void)
{
    static const uint32_t qbase[4] = { 0x50100000u, 0x50500000u, 0x50900000u, 0x50d00000u };
    for (int q = 0; q < 4; q++)
        for (int i = 0; i < FE_N_L0_REGS; i++)
            *((volatile uint32_t *)(qbase[q] + FE_L0_REGS[i])) = 0;
}

// ── 제어 레지스터를 읽지 않고 쓴다 (2026-10-19, b64 변형에서 발견) ───────────────
// 생성 코드의 ISR·cnn_stop 은 사분면 제어 레지스터(0x50x00000)를 읽어서-고쳐-쓴다(RMW). b64 변형에서 1000회 중 1~2할의 회차에
// p1 뒤 제어 레지스터가 0 으로 읽혔고(클럭·마스터 비트까지 꺼짐) 그 회차는 사분면 0 이 p1 출력을 쓰지 않았으며 p2 도 짧게 끝났다.
// 전환 전에 네 사분면 제어 레지스터를 알려진 값 0x00100008(생성 cnn_init 의 "Stop SM": 클럭 켜짐·실행 꺼짐)으로 **직접 써서**
// 어떤 읽힘값에도 의존하지 않게 한다. ISR 도 같은 값을 직접 쓰는 것으로 바꾼다 (fe_cnn_isr). ②′ (fe) 실측(2026-10-17)은 이 코드 전의
// 펌웨어로 쟀고 RMW 에서도 1000회 불일치 0 이었다 — 변형 실측부터 적용. -DFE_NO_CTRL_FIX 로 끌 수 있다.
static const uint32_t QBASE[4] = { 0x50100000u, 0x50500000u, 0x50900000u, 0x50d00000u };
// ★ 기본 정지 방식 = 지연 정지 (2026-10-20 확정). 완료 인터럽트에서는 가속기 레지스터를 쓰지 않고 FE_DELAY_STOP 사이클(기본 500 회 루프,
//   실측 약 49 µs — ②′ 옛/새 펌웨어 infer 차이, fe-curve-design.md 8.6절) 뒤에 네 사분면을 한 번만 정지시킨다. 100 회(약 4 µs)는 b64·f31 에서 충분했으나 p1 출력이 가장 큰 b64f31 에서는 멈춤(슬레이브가 꺼진 채 p2 의 층 동기를 영원히 기다림)이 났고 500 회에서 15,000회 멈춤·불일치 0. b64 변형 진단: 생성 코드처럼 인터럽트 직후 정지하면 50,000회 중 17회(0.03 %) 한
//   사분면(슬레이브)의 제어 레지스터가 0 으로 읽히며 다음 실행을 통째로 건너뛰었고, 같은 창에서 정지 쓰기를 거듭하면 18배 악화됐다.
//   지연 정지로 50,000회 0회. -DFE_IMMEDIATE_STOP 이면 예전 방식(인터럽트에서 직접 정지), -DFE_LATE_STOP 은 CPU 단 뒤 정지 실험.
#if !defined(FE_DELAY_STOP) && !defined(FE_LATE_STOP) && !defined(FE_IMMEDIATE_STOP)
#define FE_DELAY_STOP 500
#endif
#ifndef FE_RDY_SEL
#define FE_RDY_SEL 0            // CNNx16 제어 레지스터 RDY_SEL(비트 1~2) — 합성기 --ready-sel 과 같은 뜻 (APB 대기 상태). 생성 코드 기본 0.
#endif
#define CTL_IDLE   (0x00100008u | ((uint32_t)FE_RDY_SEL << 1))   // 클럭 켜짐·실행 꺼짐 (생성 cnn_init "Stop SM")
#define CTL_MASTER_PREP (0x00100808u | ((uint32_t)FE_RDY_SEL << 1))
#define CTL_SLAVE_RUN   (0x00100809u | ((uint32_t)FE_RDY_SEL << 1))
#define CTL_MASTER_RUN  (0x00100009u | ((uint32_t)FE_RDY_SEL << 1))

static inline void ctrl_stop_all(void)
{
#ifndef FE_NO_CTRL_FIX
    for (int q = 0; q < 4; q++) *((volatile uint32_t *)QBASE[q]) = CTL_IDLE;
#endif
}

#ifdef FE_DIAG
static uint32_t dg_stop_retry[4], dg_stop_max[4], dg_stop_to[4];   // ISR 정지 쓰기 재시도 통계 (사분면별)
static uint32_t dg_qfail[4], dg_qwhole[4];
static uint32_t dg_regs[4][3];                                      // 첫 실패 회차의 사분면별 제어/층 수/층 0 첫 레지스터                          // 실패 회차 누적: 사분면에 불일치가 있던 횟수 / 사분면 전체가 틀린 횟수
#endif
#ifndef FE_NO_CTRL_FIX
static void fe_cnn_isr(void)
{
#if !defined(FE_LATE_STOP) && !defined(FE_DELAY_STOP)
    // 생성 CNN_ISR 과 같은 결과(인터럽트·실행 비트 해제)를 네 사분면에 RMW 없이 직접 쓴다.
    //   ⚠️ 실행 비트가 켜진 채로는 APB 로 읽는 데이터 메모리가 아직 이전 값일 수 있다 (2026-10-19 실험: 완료 직후 정지하지 않으면
    //   매 회 앞 4 워드가 옛 값) — 정지가 먼저여야 메모리를 읽을 수 있다.
    for (int q = 0; q < 4; q++) *((volatile uint32_t *)QBASE[q]) = CTL_IDLE;
#ifdef FE_STOP_VERIFY
    // (실험, 기본 꺼짐) 정지값이 읽힐 때까지 다시 쓴다. 2026-10-20 b64: 쓴 직후 읽힘값은 거의 매번 쓴 값이 아니었고(사분면 0 은 최대
    // 24회 재시도) 재시도를 넣자 실패가 54,000회 중 17회 → 50,000회 중 287회로 **18배 늘었다** (사분면 1 전체). 완료 직후 한동안
    // 사분면이 APB 쓰기를 바로 받지 않으며, 그 창에서 쓰기를 거듭하면 오히려 해롭다.
    for (int q = 0; q < 4; q++) {
        uint32_t k = 0;
        while (*((volatile uint32_t *)QBASE[q]) != CTL_IDLE && k < 256u) { *((volatile uint32_t *)QBASE[q]) = CTL_IDLE; k++; }
#ifdef FE_DIAG
        if (k) { dg_stop_retry[q]++; if (k > dg_stop_max[q]) dg_stop_max[q] = k; if (k >= 256u) dg_stop_to[q]++; }
#endif
    }
#endif
#else
    // 완료 직후에는 가속기 레지스터를 건드리지 않는다 — 인터럽트만 막고(레벨 인터럽트 재진입 방지) 깃발만 세운다.
    // 네 사분면의 정지(인터럽트 비트 해제 포함)는 CPU 단계가 끝난 뒤 net_stop() 에서 한다 (2026-10-19 b64 실험: 완료 직후
    // 슬레이브를 끄는 생성 코드 방식에서 88,000회 중 30회 한 사분면이 다음 p1 을 통째로 건너뛰었다).
    NVIC_DisableIRQ(CNN_IRQn);
#endif
    cnn_time = 1;
}

// 시작: 생성 cnn_start 와 같은 쓰기 순서(마스터 준비 → 슬레이브 활성(EXT_SYNC) → 마스터 활성)에, 슬레이브 활성 뒤
// **읽기 3회 + 짧은 대기**를 넣어 슬레이브의 활성이 CNN 클럭 도메인에 자리잡은 뒤 마스터의 동기 펄스를 내보낸다.
// 2026-10-19 b64: 생성 순서 그대로는 88,000회 중 30회 한 사분면이 통째로 실행을 건너뛰었다(출력 전체가 이전 값) —
// 슬레이브가 동기 펄스를 놓친 것으로 해석. 비용 약 1 µs, 추론 구간에 포함된다.
static inline void fe_cnn_start(void)
{
    cnn_time = 0;
    *((volatile uint32_t *)QBASE[0]) = CTL_MASTER_PREP;
    for (int q = 1; q < 4; q++) *((volatile uint32_t *)QBASE[q]) = CTL_SLAVE_RUN;
    for (int q = 1; q < 4; q++) (void)*((volatile uint32_t *)QBASE[q]);
    for (volatile int w = 0; w < 25; w++) {}
    *((volatile uint32_t *)QBASE[0]) = CTL_MASTER_RUN;
}
#endif

// 네 사분면이 모두 정지 상태(0x00100008)로 읽힐 때까지 기다린 뒤 설정한다 — 쓰기 완료·유휴 확인 (보통 즉시 통과)
#ifdef FE_DIAG
static uint32_t dg_wait_timeout = 0, dg_wait_val = 0, dg_wait_q = 0;
#endif
static inline void ctrl_wait_idle(void)
{
#ifndef FE_NO_CTRL_FIX
    for (int q = 0; q < 4; q++) {
        int k;
        for (k = 0; k < 64; k++)                             // 유한 대기 — 읽힘값이 쓴 값과 다를 수 있어 무한 대기는 금지 (2026-10-19 멈춤)
            if (*((volatile uint32_t *)QBASE[q]) == CTL_IDLE) break;
#ifdef FE_DIAG
        if (k == 64) { if (!dg_wait_timeout) { dg_wait_val = *((volatile uint32_t *)QBASE[q]); dg_wait_q = q; } dg_wait_timeout++; }
#endif
    }
#endif
}

// 네트워크 정지: 생성 cnn_stop(사분면 0 RMW) 대신 네 사분면을 알려진 값으로 직접 쓴다 (FE_NO_CTRL_FIX 면 생성 함수)
static inline void net_stop(void)
{
#ifndef FE_NO_CTRL_FIX
    ctrl_stop_all();
#ifdef FE_LATE_STOP
    NVIC_ClearPendingIRQ(CNN_IRQn);
    NVIC_EnableIRQ(CNN_IRQn);
#endif
#else
    cnn_stop();
#endif
}

int p1_cnn_init(void);
// 전환 = 생성 *_cnn_init (AON 제어·Stop SM·SRAM control·층 수 — 단일 네트워크 예제가 설정 전에 부르는 것과 같다)
//        → 층 0 레지스터 합집합 지우기 → 생성 *_cnn_configure.  (p1 은 1층이라 생성 init 이 층 수를 쓰지 않는다 → 직접 0)
static inline void switch_to_p1(void)
{
    p1_cnn_init();
    set_layer_count(FE_P1_LAYER_COUNT);
    clear_l0_regs();
    p1_cnn_configure();
}

static inline void switch_to_p2(void)
{
    cnn_init();
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

#ifdef FE_DIAG
// -DFE_DIAG: 측정 루프에서 실패 회차의 상태를 기록해 끝에 찍는다 (진단 전용 — 측정 빌드에서는 끈다)
static uint32_t dg_bad_idx[24], dg_nbad = 0;
static uint32_t dg_first_i = 0xFFFFFFFFu, dg_p1_word, dg_p1_got, dg_p1_exp, dg_p1_nbad, dg_inf1, dg_inf2, dg_ctrl_p1, dg_ctrl_p2, dg_cnt_short2 = 0;
static uint32_t dg_ctrl_before_p1, dg_ctrl_before_p2;
static uint32_t dg_odd_after_p1 = 0, dg_odd_after_p2 = 0, dg_odd_val_p1 = 0, dg_odd_val_p2 = 0;   // 완료 뒤 제어 레지스터가 0x00100008 이 아닌 회차
static uint32_t dg_blk[16];                                                                       // 첫 실패 회차의 블록별 불일치 수

static void diag_blocks(const uint32_t *ptr, int first)
{
    uint32_t mask, len, b = 0, qbad[4] = {0, 0, 0, 0}, qlen[4] = {0, 0, 0, 0};
    volatile uint32_t *addr;
    while ((addr = (volatile uint32_t *)*ptr++) != 0 && b < 16) {
        mask = *ptr++;
        len = *ptr++;
        qlen[b >> 2] += len;
        for (uint32_t i = 0; i < len; i++)
            if ((*addr++ & mask) != *ptr++) { if (first) dg_blk[b]++; qbad[b >> 2]++; }
        b++;
    }
    for (int q = 0; q < 4; q++) { if (qbad[q]) dg_qfail[q]++; if (qbad[q] && qbad[q] == qlen[q]) dg_qwhole[q]++; }
}

static uint32_t diag_first_bad(const uint32_t *ptr, uint32_t *word, uint32_t *got, uint32_t *exp)
{
    uint32_t mask, len, n = 0, idx = 0, found = 0;
    volatile uint32_t *addr;
    while ((addr = (volatile uint32_t *)*ptr++) != 0) {
        mask = *ptr++;
        len = *ptr++;
        for (uint32_t i = 0; i < len; i++, idx++) {
            uint32_t g = *addr++ & mask, e = *ptr++;
            if (g != e) {
                if (!found) { *word = idx; *got = g; *exp = e; found = 1; }
                n++;
            }
        }
    }
    return n;
}
#endif

// ── CPU 단: p1 출력(int32, 메모리에 그대로)을 읽어 프레임 4개의 밴드 에너지 → 정수 로그 → int8 ──
//   채널 c 의 위치 p 워드: 그룹 g = c>>2 의 기준 주소 + (4p + (c&3))·4.  밴드 b 는 채널 2b(cos)·2b+1(sin) — 같은 그룹.
static inline __attribute__((always_inline)) void fe_cpu_frames_impl(int first_frame, int8_t (*lq)(int32_t, int32_t, int32_t))
{
#if FE_P1_BITS8
    // fe8: 층 B 출력 (대역 b = 채널 b, 위치 p 워드 = 그룹 b>>2 기준 + (4p + (b&3))·4) 의 프레임 시작 위치 값이 곧 E′·2^14 (≥ 0)
    for (int b = 0; b < FE_NBAND; b++) {
        const volatile uint32_t *base = (const volatile uint32_t *)FE_P1_OUT_ADDR[b >> 2] + (b & 3);
        for (int f = 0; f < FE_FRAMES_PER_HOP; f++) {
            const int dst = first_frame + f;
            if (dst < 0 || dst >= FE_NFRAME) continue;       // f31 계열 KAT 의 청크 0 프레임 −1
            fb[b][dst] = lq(log2_q8(base[(FE_POS_PER_FRAME * f) * FE_P1_OUT_STRIDE]), FE_LO_Q8, FE_SPAN_Q8);
        }
    }
#else
    for (int b = 0; b < FE_NBAND; b++) {
        const int c = 2 * b;                                   // cos 채널 (sin 은 c+1, 같은 블록의 다음 워드)
        const int g = (c & 63) >> 2, slot = (c & 3) + (c >= 64 ? 4 : 0);
        const volatile int32_t *base = (const volatile int32_t *)FE_P1_OUT_ADDR[g];
        for (int f = 0; f < FE_FRAMES_PER_HOP; f++) {
            const int dst = first_frame + f;
            if (dst < 0 || dst >= FE_NFRAME) continue;       // f31 계열 KAT 의 청크 0 프레임 −1 (측정 루프에서는 항상 범위 안)
            const volatile int32_t *p = base + (FE_POS_PER_FRAME * f) * FE_P1_OUT_STRIDE + slot;
            uint32_t e = 0;
            for (int k = 0; k < FE_POOL_POS; k++) {
                int32_t y0 = p[0], y1 = p[1];
                e += (uint32_t)(y0 < 0 ? -y0 : y0) + (uint32_t)(y1 < 0 ? -y1 : y1);
                p += FE_P1_OUT_STRIDE;
            }
            fb[b][dst] = lq(log2_q8(e), FE_LO_Q8, FE_SPAN_Q8);
        }
    }
#endif
}
#ifdef FE_LOG_AB
// -DFE_LOG_AB (feat/log32, fe-curve-design.md 10절): 옛 64비트 log2_q8_to_int8 (_ref) 과 새 32비트 구현을 한 바이너리에 넣고
// 같은 POR 세션에서 옛 → 새 순서로 N_ITER 회씩 잰다. 함수 포인터는 always_inline 본체에서 상수로 접혀 직접 호출(bl)이 된다 —
// 빌드 뒤 objdump 로 blx 가 없는지 확인한다. fe_log_sel 분기 한 번(수 사이클)은 두 패스에 같이 든다.
#include "log2_q8_ref.inc"
static int fe_log_sel = 0;                                 // 0 = 옛(64비트 ref), 1 = 새(32비트)
static void fe_cpu_frames_ref(int ff) { fe_cpu_frames_impl(ff, log2_q8_to_int8_ref); }
static void fe_cpu_frames_new(int ff) { fe_cpu_frames_impl(ff, log2_q8_to_int8); }
static void fe_cpu_frames(int ff) { if (fe_log_sel) fe_cpu_frames_new(ff); else fe_cpu_frames_ref(ff); }
#else
static void fe_cpu_frames(int ff) { fe_cpu_frames_impl(ff, log2_q8_to_int8); }
#endif

// 판단 1회의 CPU 몫: 버퍼 밀기(4프레임) + 새 4프레임 + p2 입력 워드 포장
#ifndef FE_CPU_SPLIT
static void fe_cpu_hop(void)
{
    for (int b = 0; b < FE_NBAND; b++)
        memmove(fb[b], fb[b] + FE_FRAMES_PER_HOP, FE_NFRAME - FE_FRAMES_PER_HOP);
    fe_cpu_frames(FE_NFRAME - FE_FRAMES_PER_HOP);
    const int8_t *q = &fb[0][0];
    for (int i = 0; i < FE_P2_IN_WORDS; i++) p2_in[i] = (uint8_t)q[i];
}
#else
// -DFE_CPU_SPLIT (2026-10-21, fe-curve-design.md 9절 실험 II): CPU 몫을 두 단계로 나눠 잰다.
//   (a) cpu_rd — 가속기 메모리에서 int32 2,048 워드 읽기 + |·| + 8위치 합 (밴드·프레임별 e 까지)
//   (b) cpu_lg — 정수 로그 128회 + 버퍼 밀기 + p2 입력 포장
// 계산은 같고(각 e 가 독립이라 순서만 바뀜 — 결과 비트 동일, KAT 로 확인) tick 두 번이 더 든다.
// 주 값(옛 펌웨어 1.982 ms)은 바꾸지 않고 비율만 쓴다.
static uint32_t e_hop[FE_NBAND][FE_FRAMES_PER_HOP];
static uint32_t t_cpu_rd[N_ITER], t_cpu_lg[N_ITER];
#ifdef FE_LOG_AB
static uint32_t t_cpu_rd_a[N_ITER], t_cpu_lg_a[N_ITER];
#endif
static inline __attribute__((always_inline)) void split_phase_b(int8_t (*lq)(int32_t, int32_t, int32_t))
{
    for (int b = 0; b < FE_NBAND; b++) {                       // (b) 로그 + 밀기
        memmove(fb[b], fb[b] + FE_FRAMES_PER_HOP, FE_NFRAME - FE_FRAMES_PER_HOP);
        for (int f = 0; f < FE_FRAMES_PER_HOP; f++)
            fb[b][FE_NFRAME - FE_FRAMES_PER_HOP + f] = lq(log2_q8(e_hop[b][f]), FE_LO_Q8, FE_SPAN_Q8);
    }
}
#ifdef FE_LOG_AB
static void split_phase_b_ref(void) { split_phase_b(log2_q8_to_int8_ref); }
static void split_phase_b_new(void) { split_phase_b(log2_q8_to_int8); }
static inline void split_phase_b_sel(void) { if (fe_log_sel) split_phase_b_new(); else split_phase_b_ref(); }
#else
static inline void split_phase_b_sel(void) { split_phase_b(log2_q8_to_int8); }
#endif
static uint32_t split_rd, split_lg;
static void fe_cpu_hop(void)
{
    uint32_t s0 = tick();
#if FE_P1_BITS8
    for (int b = 0; b < FE_NBAND; b++) {                       // (a) fe8: 층 B 출력 읽기만 (n×F′ 워드)
        const volatile uint32_t *base = (const volatile uint32_t *)FE_P1_OUT_ADDR[b >> 2] + (b & 3);
        for (int f = 0; f < FE_FRAMES_PER_HOP; f++)
            e_hop[b][f] = base[(FE_POS_PER_FRAME * f) * FE_P1_OUT_STRIDE];
    }
#else
    for (int b = 0; b < FE_NBAND; b++) {                       // (a) 읽기 + |·| + 합
        const int c = 2 * b;
        const int g = (c & 63) >> 2, slot = (c & 3) + (c >= 64 ? 4 : 0);
        const volatile int32_t *base = (const volatile int32_t *)FE_P1_OUT_ADDR[g];
        for (int f = 0; f < FE_FRAMES_PER_HOP; f++) {
            const volatile int32_t *p = base + (FE_POS_PER_FRAME * f) * FE_P1_OUT_STRIDE + slot;
            uint32_t e = 0;
            for (int k = 0; k < FE_POOL_POS; k++) {
                int32_t y0 = p[0], y1 = p[1];
                e += (uint32_t)(y0 < 0 ? -y0 : y0) + (uint32_t)(y1 < 0 ? -y1 : y1);
                p += FE_P1_OUT_STRIDE;
            }
            e_hop[b][f] = e;
        }
    }
#endif
    uint32_t s1 = tick();
    split_phase_b_sel();                                       // (b) 로그 + 밀기 (+ 아래 포장)
    const int8_t *q = &fb[0][0];
    for (int i = 0; i < FE_P2_IN_WORDS; i++) p2_in[i] = (uint8_t)q[i];
    uint32_t s2 = tick();
    split_rd = (s1 - s0) & tick_mask;
    split_lg = (s2 - s1) & tick_mask;
}
#endif

#ifdef FE_DIAG
static uint32_t dg_done_max[4], dg_done_to = 0, dg_done_sum[4];
#endif
// 완료 대기: 마스터 인터럽트(cnn_time) 뒤에 **네 사분면 각각의 완료 비트(12)** 를 기다린다 — 마스터의 인터럽트는 마지막 메모리
// 쓰기보다 먼저 올 수 있고(2026-10-19 b64: 인터럽트 직후 읽으면 4 워드가 아직 이전 값), 슬레이브가 늦게 끝날 수 있다.
static inline void wait_all_done(void)
{
    while (cnn_time == 0) {}
#ifdef FE_DELAY_STOP
    // 완료 인터럽트 직후의 "마무리 창" 을 피한다 (2026-10-20 b64 진단: 이 창에 제어 레지스터를 쓰면 드물게(0.03 %) 한 사분면의 제어
    // 레지스터가 0 으로 읽히며(클럭 꺼짐) 그 사분면이 다음 실행을 통째로 건너뛴다; 같은 창에서 쓰기를 거듭하면 18배 악화, 생성 코드의
    // RMW 는 이 창에서 0 을 읽는다). FE_DELAY_STOP 사이클만큼 기다린 뒤 네 사분면을 한 번만 정지시키고 인터럽트를 다시 연다.
    for (volatile int w = 0; w < FE_DELAY_STOP; w++) {}
    ctrl_stop_all();
    NVIC_ClearPendingIRQ(CNN_IRQn);
    NVIC_EnableIRQ(CNN_IRQn);
#endif
#ifdef FE_LATE_STOP
    for (int q = 0; q < 4; q++) {
        uint32_t k = 0;
        while (!(*((volatile uint32_t *)QBASE[q]) & (1u << 12)) && k < 4000000u) k++;
#ifdef FE_DIAG
        if (k > dg_done_max[q]) dg_done_max[q] = k;
        dg_done_sum[q] += k;
        if (k >= 4000000u) dg_done_to++;
#endif
    }
#endif
}

static inline void run_and_wait_p1(void)
{
#ifdef FE_START_SYNC
    fe_cnn_start();
#else
    p1_cnn_start();
#endif
    wait_all_done();
}

static inline void run_and_wait_p2(void)
{
#ifdef FE_START_SYNC
    fe_cnn_start();
#else
    cnn_start();
#endif
    wait_all_done();
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
#ifndef FE_NO_CTRL_FIX
    MXC_NVIC_SetVector(CNN_IRQn, fe_cnn_isr);   // 생성 CNN_ISR(RMW) 대신 — 위 설명
#endif
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
        fe_cpu_frames(c * FE_FRAMES_PER_HOP + FE_FRAME0_OFFSET);
        DBG("[dbg] cpu done, stop\n");
        net_stop();
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
    net_stop();
    static int32_t kat_raw[CNN_NUM_OUTPUTS];
    for (int i = 0; i < CNN_NUM_OUTPUTS; i++) kat_raw[i] = ml_data[i];
    DBG("[dbg] loop start\n");

    // ── 2. 판단 1회 × N_ITER ───────────────────────────────────────────────
    //   매 회 "직전 12프레임이 버퍼에 있고 새 청크(청크 3)가 들어온다" 는 상태를 재현한다 —
    //   측정 구간 밖에서 버퍼의 슬롯 4..15 에 기대 특징의 프레임 0..11 을 넣어, 4프레임 밀기 뒤 새 4프레임을 더하면
    //   버퍼가 기대 특징(창 전체)과 같아지게 한다 → p2 입력이 매 회 같고 비트 대조가 된다.
    uint32_t bad_p1 = 0, bad_feat = 0, bad_p2 = 0;
#ifdef FE_LOG_AB
#define FE_N_PASS 2
#else
#define FE_N_PASS 1
#endif
    for (int fe_pass = 0; fe_pass < FE_N_PASS; fe_pass++) {
#ifdef FE_LOG_AB
    fe_log_sel = fe_pass;
    bad_p1 = bad_feat = bad_p2 = 0;
#endif
#ifdef FE_DIAG_REPEAT
    // -DFE_DIAG_REPEAT (FE_DIAG 와 함께): 리셋 없이 N_ITER 회 루프를 무한 반복하며 라운드마다 누적 불일치를 찍는다 — 드문 실패의 빈도 추정용
    uint32_t round = 0, cum_p1 = 0, cum_feat = 0, cum_p2 = 0;
repeat_round:
    bad_p1 = bad_feat = bad_p2 = 0;
#endif
    for (int i = 0; i < N_ITER; i++) {
        for (int b = 0; b < FE_NBAND; b++)                                     // 상태 복원 (측정 밖)
            memcpy(fb[b] + FE_FRAMES_PER_HOP, fe_q_expected + b * FE_NFRAME, FE_NFRAME - FE_FRAMES_PER_HOP);
        uint32_t c0 = tick();
        load_chunk(FE_NCHUNK - 1);
        uint32_t c1 = tick();
#ifdef FE_DIAG
        uint32_t ctrl_b1 = *((volatile uint32_t *)0x50100000);
#endif
        switch_to_p1();
        uint32_t c2 = tick();
#ifdef FE_DEBUG_ITER
        printf("[%d] p1 start\n", i);
#endif
        run_and_wait_p1();
        uint32_t c3 = tick();
#ifdef FE_DEBUG_ITER
        printf("[%d] p1 done\n", i);
#endif
#ifdef FE_DIAG
        uint32_t ctrl_a1 = *((volatile uint32_t *)0x50100000);
        if (ctrl_a1 != CTL_IDLE) { if (!dg_odd_after_p1) dg_odd_val_p1 = ctrl_a1; dg_odd_after_p1++; }
        int p1_ok = check_words(p1_sample_output) == CNN_OK;
        if (!p1_ok) {
            bad_p1++;
            if (dg_nbad < 24) dg_bad_idx[dg_nbad] = i;
            dg_nbad++;
            if (dg_first_i == 0xFFFFFFFFu) {
                dg_first_i = i; dg_inf1 = (c3 - c2) & tick_mask; dg_ctrl_p1 = ctrl_a1; dg_ctrl_before_p1 = ctrl_b1;
                dg_p1_nbad = diag_first_bad(p1_sample_output, &dg_p1_word, &dg_p1_got, &dg_p1_exp);
                diag_blocks(p1_sample_output, 1);
                for (int q = 0; q < 4; q++) {
                    dg_regs[q][0] = *((volatile uint32_t *)QBASE[q]);
                    dg_regs[q][1] = *((volatile uint32_t *)QBASE_LC[q]);
                    dg_regs[q][2] = *((volatile uint32_t *)(QBASE[q] + FE_L0_REGS[0]));
                }
            } else diag_blocks(p1_sample_output, 0);
        }
#else
        if (check_words(p1_sample_output) != CNN_OK) bad_p1++; // 측정 밖
#endif
        uint32_t c4 = tick();
        fe_cpu_hop();
        uint32_t c5 = tick();
        if (memcmp(fb, fe_q_expected, sizeof(fb)) != 0) bad_feat++; // 측정 밖
        net_stop();
        uint32_t c6 = tick();
        memcpy32((uint32_t *)FE_P2_IN_ADDR, p2_in, FE_P2_IN_WORDS);
        uint32_t c7 = tick();
#ifdef FE_DIAG
        uint32_t ctrl_b2 = *((volatile uint32_t *)0x50100000);
#endif
        switch_to_p2();
        uint32_t c8 = tick();
#ifdef FE_DEBUG_ITER
        printf("[%d] p2 start\n", i);
#endif
        run_and_wait_p2();
        uint32_t c9 = tick();
#ifdef FE_DEBUG_ITER
        printf("[%d] p2 done\n", i);
#endif
#ifdef FE_DIAG
        if (((c9 - c8) & tick_mask) < 50000u) dg_cnt_short2++;
        {
            uint32_t ctrl_a2 = *((volatile uint32_t *)0x50100000);
            if (ctrl_a2 != CTL_IDLE) { if (!dg_odd_after_p2) dg_odd_val_p2 = ctrl_a2; dg_odd_after_p2++; }
            if (dg_first_i == i) { dg_inf2 = (c9 - c8) & tick_mask; dg_ctrl_p2 = ctrl_a2; dg_ctrl_before_p2 = ctrl_b2; }
        }
#endif
        if (check_words(p2_sample_output) != CNN_OK) bad_p2++; // 측정 밖
        uint32_t c10 = tick();
        cnn_unload((uint32_t *)ml_data);
        softmax_q17p14_q15((const q31_t *)ml_data, CNN_NUM_OUTPUTS, ml_softmax);
        uint32_t c11 = tick();
        net_stop();
        t_seg[S_LOAD1][i] = (c1 - c0) & tick_mask;
        t_seg[S_CFG1][i] = (c2 - c1) & tick_mask;
        t_seg[S_INFER1][i] = (c3 - c2) & tick_mask;
        t_seg[S_CPU][i] = (c5 - c4) & tick_mask;
#ifdef FE_CPU_SPLIT
        t_cpu_rd[i] = split_rd;
        t_cpu_lg[i] = split_lg;
#endif
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
#ifdef FE_DIAG_REPEAT
    round++; cum_p1 += bad_p1; cum_feat += bad_feat; cum_p2 += bad_p2;
    {
        uint32_t mx1 = 0, mx2 = 0, mn2 = 0xFFFFFFFFu;
        for (int i = 0; i < N_ITER; i++) {
            if (t_seg[S_INFER1][i] > mx1) mx1 = t_seg[S_INFER1][i];
            if (t_seg[S_INFER2][i] > mx2) mx2 = t_seg[S_INFER2][i];
            if (t_seg[S_INFER2][i] < mn2) mn2 = t_seg[S_INFER2][i];
        }
        printf("ROUND %lu: this round mismatches p1 %lu feat %lu p2 %lu; cumulative %lu iters: p1 %lu feat %lu p2 %lu; infer1 max %lu infer2 min %lu max %lu; "
               "ctrl0 odd after p1 %lu (0x%08lx) after p2 %lu (0x%08lx); first bad iter %lu words %lu first word %lu got 0x%08lx exp 0x%08lx; blocks",
               (unsigned long)round, (unsigned long)bad_p1, (unsigned long)bad_feat, (unsigned long)bad_p2, (unsigned long)(round * N_ITER),
               (unsigned long)cum_p1, (unsigned long)cum_feat, (unsigned long)cum_p2, (unsigned long)mx1, (unsigned long)mn2, (unsigned long)mx2,
               (unsigned long)dg_odd_after_p1, (unsigned long)dg_odd_val_p1, (unsigned long)dg_odd_after_p2, (unsigned long)dg_odd_val_p2,
               (unsigned long)dg_first_i, (unsigned long)dg_p1_nbad, (unsigned long)dg_p1_word, (unsigned long)dg_p1_got, (unsigned long)dg_p1_exp);
        for (int b = 0; b < 16; b++) printf(" %lu", (unsigned long)dg_blk[b]);
        printf(" | done-wait max q0..3 %lu %lu %lu %lu sum %lu %lu %lu %lu timeouts %lu", (unsigned long)dg_done_max[0], (unsigned long)dg_done_max[1],
               (unsigned long)dg_done_max[2], (unsigned long)dg_done_max[3], (unsigned long)dg_done_sum[0], (unsigned long)dg_done_sum[1],
               (unsigned long)dg_done_sum[2], (unsigned long)dg_done_sum[3], (unsigned long)dg_done_to);
        printf(" | qfail q0..3 %lu %lu %lu %lu whole %lu %lu %lu %lu", (unsigned long)dg_qfail[0], (unsigned long)dg_qfail[1], (unsigned long)dg_qfail[2],
               (unsigned long)dg_qfail[3], (unsigned long)dg_qwhole[0], (unsigned long)dg_qwhole[1], (unsigned long)dg_qwhole[2], (unsigned long)dg_qwhole[3]);
        printf(" | stop-retry q0..3 %lu %lu %lu %lu max %lu %lu %lu %lu timeouts %lu %lu %lu %lu", (unsigned long)dg_stop_retry[0], (unsigned long)dg_stop_retry[1],
               (unsigned long)dg_stop_retry[2], (unsigned long)dg_stop_retry[3], (unsigned long)dg_stop_max[0], (unsigned long)dg_stop_max[1],
               (unsigned long)dg_stop_max[2], (unsigned long)dg_stop_max[3], (unsigned long)dg_stop_to[0], (unsigned long)dg_stop_to[1],
               (unsigned long)dg_stop_to[2], (unsigned long)dg_stop_to[3]);
        printf(" | lc-verify max q0..3 %lu %lu %lu %lu rewrite %lu %lu %lu %lu timeouts %lu %lu %lu %lu", (unsigned long)dg_lc_max[0], (unsigned long)dg_lc_max[1],
               (unsigned long)dg_lc_max[2], (unsigned long)dg_lc_max[3], (unsigned long)dg_lc_rewrite[0], (unsigned long)dg_lc_rewrite[1],
               (unsigned long)dg_lc_rewrite[2], (unsigned long)dg_lc_rewrite[3], (unsigned long)dg_lc_to[0], (unsigned long)dg_lc_to[1],
               (unsigned long)dg_lc_to[2], (unsigned long)dg_lc_to[3]);
        printf(" | first-fail regs");
        for (int q = 0; q < 4; q++) printf(" q%d %08lx/%08lx/%08lx", q, (unsigned long)dg_regs[q][0], (unsigned long)dg_regs[q][1], (unsigned long)dg_regs[q][2]);
        printf(" | wait-idle timeouts %lu (q%lu read 0x%08lx)\n=== END ===\n", (unsigned long)dg_wait_timeout, (unsigned long)dg_wait_q, (unsigned long)dg_wait_val);
    }
    goto repeat_round;
#endif
#ifdef FE_LOG_AB
    if (fe_pass == 0) {
        memcpy(t_seg_a, t_seg, sizeof(t_seg)); memcpy(t_total_a, t_total, sizeof(t_total));
        bad_a[0] = bad_p1; bad_a[1] = bad_feat; bad_a[2] = bad_p2;
#ifdef FE_CPU_SPLIT
        memcpy(t_cpu_rd_a, t_cpu_rd, sizeof(t_cpu_rd)); memcpy(t_cpu_lg_a, t_cpu_lg, sizeof(t_cpu_lg));
#endif
    }
#endif
    }   // fe_pass
    SysTick->CTRL = 0; // 이후 MXC_Delay 금지 (measure_main.c 참조)
    cnn_disable();

    while (1) {
        printf("\n=== SafeSound NPU measure: %s ===\n", NET_NAME);
        printf("build %s %s  SystemCoreClock %lu Hz  N_ITER %d\n", __DATE__, __TIME__,
               (unsigned long)SystemCoreClock, N_ITER);
#if FE_P1_BITS8
        printf("two networks co-resident: p1 fe8 = Conv1d 64->%d k8 Abs 8bit (out shift %d) + sum Conv1d ->%d k8 wide, in %d pos (weights %lu B) + p2 1x%dx%d (weights %lu B)\n",
               2 * FE_NBAND, (int)FE_P1_OUT_SHIFT, FE_P1_OUT_CH, FE_P1_IN_WORDS, (unsigned long)FE_P1_WEIGHT_BYTES, FE_NBAND, FE_NFRAME, (unsigned long)FE_P2_WEIGHT_BYTES);
        printf("cpu stage reads %d int32 words per decision (p1 layer-B frame positions), LO_Q8 %d (includes -256)\n",
               FE_NBAND * FE_FRAMES_PER_HOP, FE_LO_Q8);
#else
        printf("two networks co-resident: p1 Conv1d 64->%d k8 wide, in %d pos (weights %lu B) + p2 1x%dx%d (weights %lu B)\n",
               FE_P1_OUT_CH, FE_P1_IN_WORDS, (unsigned long)FE_P1_WEIGHT_BYTES, FE_NBAND, FE_NFRAME, (unsigned long)FE_P2_WEIGHT_BYTES);
#endif
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
#ifdef FE_CPU_SPLIT
        report("cpu_rd", t_cpu_rd, SystemCoreClock);
        report("cpu_lg", t_cpu_lg, SystemCoreClock);
#endif
#ifdef FE_DIAG
        printf("DIAG p1-bad iterations %lu, first %lu: infer1 %lu cyc, p1 mismatching words %lu, first word %lu got 0x%08lx exp 0x%08lx; "
               "ctrl0 before p1 0x%08lx after p1 0x%08lx; infer2 %lu cyc, ctrl0 before p2 0x%08lx after p2 0x%08lx; short-infer2 count %lu\n",
               (unsigned long)dg_nbad, (unsigned long)dg_first_i, (unsigned long)dg_inf1, (unsigned long)dg_p1_nbad, (unsigned long)dg_p1_word,
               (unsigned long)dg_p1_got, (unsigned long)dg_p1_exp, (unsigned long)dg_ctrl_before_p1, (unsigned long)dg_ctrl_p1, (unsigned long)dg_inf2,
               (unsigned long)dg_ctrl_before_p2, (unsigned long)dg_ctrl_p2, (unsigned long)dg_cnt_short2);
        printf("DIAG bad idx:");
        for (uint32_t k = 0; k < dg_nbad && k < 24; k++) printf(" %lu", (unsigned long)dg_bad_idx[k]);
        printf("\nDIAG ctrl0 != 0x00100008 after p1: %lu iters (first 0x%08lx), after p2: %lu iters (first 0x%08lx); first-bad per-block mismatches:",
               (unsigned long)dg_odd_after_p1, (unsigned long)dg_odd_val_p1, (unsigned long)dg_odd_after_p2, (unsigned long)dg_odd_val_p2);
        for (int b = 0; b < 16; b++) printf(" %lu", (unsigned long)dg_blk[b]);
        printf("\n");
#endif
#ifdef FE_LOG_AB
        printf("=== PASS A (log64 ref): %d iterations, output mismatches: %lu  (p1 %lu, feature %lu, p2 %lu)\n", N_ITER,
               (unsigned long)(bad_a[0] + bad_a[1] + bad_a[2]), (unsigned long)bad_a[0], (unsigned long)bad_a[1], (unsigned long)bad_a[2]);
        for (int s = 0; s < N_SEG; s++) report(seg_name[s], t_seg_a[s], SystemCoreClock);
        report("total", t_total_a, SystemCoreClock);
#ifdef FE_CPU_SPLIT
        report("cpu_rd", t_cpu_rd_a, SystemCoreClock);
        report("cpu_lg", t_cpu_lg_a, SystemCoreClock);
#endif
        printf("=== PASS END ===\n");
        printf("=== PASS B (log32 new): %d iterations, output mismatches: %lu  (p1 %lu, feature %lu, p2 %lu)\n", N_ITER,
               (unsigned long)(bad_p1 + bad_feat + bad_p2), (unsigned long)bad_p1, (unsigned long)bad_feat, (unsigned long)bad_p2);
        for (int s = 0; s < N_SEG; s++) report(seg_name[s], t_seg[s], SystemCoreClock);
        report("total", t_total, SystemCoreClock);
#ifdef FE_CPU_SPLIT
        report("cpu_rd", t_cpu_rd, SystemCoreClock);
        report("cpu_lg", t_cpu_lg, SystemCoreClock);
#endif
        printf("=== PASS END ===\n");
#endif
        printf("=== END ===\n");
        for (volatile uint32_t w = 0; w < 40000000u; w++) {}
    }
}
