// 로그 멜 전처리 — datasets/melfeat.py 의 C 이식 (float32).
//
// 프레임 정의가 둘이다.
//
//   ①  (melfeat_int8)      hop 256 + 창 단위 reflect 패딩 128. 판단마다 64프레임
//                          **전부** 다시 계산한다 — hop 이 판단 주기(4,000 샘플)의
//                          약수가 아니고 프레임이 창 끝 패딩에 묶여 있다
//   ①′ (melfeat_inc_*)     hop 250, 패딩 없음, 끝 정렬. 판단 주기 = 정확히 16프레임
//                          이라 **새 프레임 16개만** 계산하고 48개는 링 버퍼에서
//                          재사용한다. 프레임 k 는 스트림의 [250k, 250k+512) 에만
//                          의존한다
//
// 프레임 하나의 계산(창 → rfft → 파워 → 멜 → 로그 → int8)은 두 정의가 **같은
// 코드**를 탄다. 로그 스케일은 고정 기준(TOP_DB 0 / SPAN 70)이라 창 통계에
// 의존하지 않는다 — 그래서 증분 결과가 일괄 계산과 비트 단위로 같다.
//
// 보드에서는 CMSIS-DSP `arm_rfft_fast_f32`, PC 대조용(-DMELFEAT_HOST)에서는
// 아래의 단순 radix-2 FFT 를 쓴다. 나머지 코드는 **한 벌**이다.
#include "melfeat.h"
#include "mel_tables.h"
#include <math.h>
#include <string.h>

#ifdef MELFEAT_HOST
// ── PC 대조용 FFT. 출력 배치만 CMSIS 와 맞춘다: spec[2k]=Re, spec[2k+1]=Im ──
static float tw_re[MEL_N_FFT / 2], tw_im[MEL_N_FFT / 2];
static float w_re[MEL_N_FFT], w_im[MEL_N_FFT];
void melfeat_init(void)
{
    for (int k = 0; k < MEL_N_FFT / 2; k++) {
        tw_re[k] = cosf(-2.0f * 3.14159265358979f * (float)k / MEL_N_FFT);
        tw_im[k] = sinf(-2.0f * 3.14159265358979f * (float)k / MEL_N_FFT);
    }
}
static void rfft512(float *in, float *spec)
{
    for (int i = 0, j = 0; i < MEL_N_FFT; i++) { // 비트 반전
        w_re[j] = in[i];
        w_im[j] = 0.0f;
        int bit = MEL_N_FFT >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
    }
    for (int len = 2; len <= MEL_N_FFT; len <<= 1) {
        int step = MEL_N_FFT / len;
        for (int i = 0; i < MEL_N_FFT; i += len)
            for (int k = 0; k < len / 2; k++) {
                float cr = tw_re[k * step], ci = tw_im[k * step];
                float xr = w_re[i + k + len / 2], xi = w_im[i + k + len / 2];
                float tr = xr * cr - xi * ci, ti = xr * ci + xi * cr;
                w_re[i + k + len / 2] = w_re[i + k] - tr;
                w_im[i + k + len / 2] = w_im[i + k] - ti;
                w_re[i + k] += tr;
                w_im[i + k] += ti;
            }
    }
    for (int k = 0; k < MEL_N_FFT / 2; k++) {
        spec[2 * k] = w_re[k];
        spec[2 * k + 1] = w_im[k];
    }
}
#else
#include "arm_math.h"
static arm_rfft_fast_instance_f32 rfft_inst;
void melfeat_init(void)
{
    arm_rfft_fast_init_f32(&rfft_inst, MEL_N_FFT);
}
static inline void rfft512(float *in, float *spec)
{
    arm_rfft_fast_f32(&rfft_inst, in, spec, 0); // in 을 망가뜨린다
}
#endif

static float buf[MEL_N_FFT];
static float spec[MEL_N_FFT];
static float pw[MEL_N_FFT / 2]; // 빈 1..255 만 쓴다 (0·256 은 어느 필터에도 안 걸린다)
static float mel[MEL_N_MELS];

// ── 프레임 채우기 ───────────────────────────────────────────────────────
// ① : numpy `mode="reflect"` (가장자리 샘플을 반복하지 않는다)
static inline __attribute__((always_inline)) void fill_reflect(const int8_t *x, int f)
{
    int base = f * MEL_HOP - MEL_PAD;
    for (int n = 0; n < MEL_N_FFT; n++) {
        int i = base + n;
        if (i < 0) i = -i;
        else if (i >= MEL_WIN) i = 2 * (MEL_WIN - 1) - i;
        buf[n] = (float)x[i] * mel_win[n];
    }
}
// ①′ : 연속한 512 샘플 그대로
static inline __attribute__((always_inline)) void fill_plain(const int8_t *src)
{
    for (int n = 0; n < MEL_N_FFT; n++) buf[n] = (float)src[n] * mel_win[n];
}

// ── 프레임 계산의 뒤 세 단계 (두 정의가 공유한다) ───────────────────────
static inline __attribute__((always_inline)) void power_mel(void)
{
    for (int k = 1; k < MEL_N_FFT / 2; k++)
        pw[k] = spec[2 * k] * spec[2 * k] + spec[2 * k + 1] * spec[2 * k + 1];
    const float *w = mel_fb_w;
    for (int m = 0; m < MEL_N_MELS; m++) {
        const float *p = &pw[mel_fb_start[m]];
        float s = 0.0f;
        for (int j = 0; j < mel_fb_len[m]; j++) s += p[j] * (*w++);
        mel[m] = s * MEL_INV_REF;
    }
}
static inline __attribute__((always_inline)) void log_quant(int8_t *out, int stride)
{
    for (int m = 0; m < MEL_N_MELS; m++) {
        float q = floorf(MEL_K_LN * logf(mel[m] + MEL_EPS) + MEL_Q_OFF + 0.5f);
        if (q > 127.0f) q = 127.0f;
        if (q < -128.0f) q = -128.0f;
        out[m * stride] = (int8_t)q; // 포화 (랩어라운드 아님)
    }
}

// ───────────────────────────────────────────────────────────── ① (전량)
static inline __attribute__((always_inline)) void core(
    const int8_t *x, int8_t *out, const int prof, uint32_t *acc,
    uint32_t (*tick)(void), uint32_t mask)
{
    uint32_t t0 = 0, t1;
    for (int f = 0; f < MEL_N_FRAMES; f++) {
        if (prof) t0 = tick();
        fill_reflect(x, f);
        if (prof) { t1 = tick(); acc[0] += (t1 - t0) & mask; t0 = t1; }
        rfft512(buf, spec);
        if (prof) { t1 = tick(); acc[1] += (t1 - t0) & mask; t0 = t1; }
        power_mel();
        if (prof) { t1 = tick(); acc[2] += (t1 - t0) & mask; t0 = t1; }
        log_quant(&out[f], MEL_N_FRAMES); // out[m*64 + f]
        if (prof) { t1 = tick(); acc[3] += (t1 - t0) & mask; }
    }
}

void melfeat_int8(const int8_t *x, int8_t *out)
{
    core(x, out, 0, 0, 0, 0);
}

void melfeat_int8_prof(const int8_t *x, int8_t *out, uint32_t prof[4],
                       uint32_t (*tick)(void), uint32_t mask)
{
    core(x, out, 1, prof, tick, mask);
}

// ───────────────────────────────────────────────────────────── ①′ (증분)
// 일괄 계산 — 기준. 1초 창의 샘플 MEL_OFF_INC.. 를 쓴다 (끝 정렬).
void melfeat_inc_batch(const int8_t *x, int8_t *out)
{
    for (int f = 0; f < MEL_N_FRAMES; f++) {
        fill_plain(&x[MEL_OFF_INC + f * MEL_HOP_INC]);
        rfft512(buf, spec);
        power_mel();
        log_quant(&out[f], MEL_N_FRAMES);
    }
}

// 링 버퍼: 열(프레임) 64개 × 멜 64. ring[col][mel] — 한 프레임이 연속 64바이트다.
static int8_t ring[MEL_N_FRAMES][MEL_N_MELS];
static int ring_head; // 다음에 쓸 열 = 가장 오래된 열
// 직전 판단의 꼬리 262 샘플 + 새 4,000 샘플. 새 16프레임이 이 안에 다 들어온다
// (마지막 프레임: 3,750 + 512 = 4,262).
static int8_t hist[MEL_INC_TAIL + MEL_INC_STEP];

void melfeat_inc_reset(const int8_t *tail)
{
    memset(ring, -128, sizeof(ring));
    ring_head = 0;
    if (tail) memcpy(&hist[MEL_INC_STEP], tail, MEL_INC_TAIL);
    else memset(&hist[MEL_INC_STEP], 0, MEL_INC_TAIL);
}

void melfeat_inc_push(const int8_t *x_new)
{
    // 꼬리를 앞으로, 새 샘플을 뒤에
    memmove(hist, &hist[MEL_INC_STEP], MEL_INC_TAIL);
    memcpy(&hist[MEL_INC_TAIL], x_new, MEL_INC_STEP);
    for (int j = 0; j < MEL_INC_FRAMES; j++) {
        fill_plain(&hist[j * MEL_HOP_INC]);
        rfft512(buf, spec);
        power_mel();
        log_quant(ring[ring_head], 1);
        ring_head = (ring_head + 1) % MEL_N_FRAMES;
    }
}

void melfeat_inc_get(int8_t *out)
{
    // 시간 순(가장 오래된 열부터)으로 풀어 mel-major 로 쓴다: out[m*64 + f]
    for (int f = 0; f < MEL_N_FRAMES; f++) {
        const int8_t *col = ring[(ring_head + f) % MEL_N_FRAMES];
        for (int m = 0; m < MEL_N_MELS; m++) out[m * MEL_N_FRAMES + f] = col[m];
    }
}
