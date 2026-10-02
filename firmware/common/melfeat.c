// 로그 멜 전처리 — datasets/melfeat.py 의 C 이식 (float32).
//
//   1초 int8 창 16,384 샘플 → reflect 패딩 128 → 512점 Hann, hop 256, 64프레임
//   → rfft → 파워 → 멜 64대역(HTK 삼각, 20~8000 Hz) → 10·log10 → int8 (0dB..−70dB)
//
// ⚠️ 프레임이 **창 단위 reflect 패딩**에 묶여 있고 hop(256)이 판단 주기
//    (4,000 샘플)의 약수가 아니다. 그래서 이웃한 판단끼리 프레임을 재사용할 수
//    없다 — 판단마다 64프레임을 전부 다시 계산한다. 이것이 구성 ①의 CPU 비용이다.
//
// 보드에서는 CMSIS-DSP `arm_rfft_fast_f32`, PC 대조용(-DMELFEAT_HOST)에서는
// 아래의 단순 radix-2 FFT 를 쓴다. 나머지 코드는 **한 벌**이다.
#include "melfeat.h"
#include "mel_tables.h"
#include <math.h>

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

static inline __attribute__((always_inline)) void core(
    const int8_t *x, int8_t *out, const int prof, uint32_t *acc,
    uint32_t (*tick)(void), uint32_t mask)
{
    uint32_t t0 = 0, t1;
    for (int f = 0; f < MEL_N_FRAMES; f++) {
        // 1) 프레임 — numpy `mode="reflect"` (가장자리 샘플을 반복하지 않는다)
        if (prof) t0 = tick();
        int base = f * MEL_HOP - MEL_PAD;
        for (int n = 0; n < MEL_N_FFT; n++) {
            int i = base + n;
            if (i < 0) i = -i;
            else if (i >= MEL_WIN) i = 2 * (MEL_WIN - 1) - i;
            buf[n] = (float)x[i] * mel_win[n];
        }
        if (prof) { t1 = tick(); acc[0] += (t1 - t0) & mask; t0 = t1; }

        // 2) rfft
        rfft512(buf, spec);
        if (prof) { t1 = tick(); acc[1] += (t1 - t0) & mask; t0 = t1; }

        // 3) 파워 + 멜 (희소 삼각 필터)
        for (int k = 1; k < MEL_N_FFT / 2; k++)
            pw[k] = spec[2 * k] * spec[2 * k] + spec[2 * k + 1] * spec[2 * k + 1];
        const float *w = mel_fb_w;
        float mel[MEL_N_MELS];
        for (int m = 0; m < MEL_N_MELS; m++) {
            const float *p = &pw[mel_fb_start[m]];
            float s = 0.0f;
            for (int j = 0; j < mel_fb_len[m]; j++) s += p[j] * (*w++);
            mel[m] = s * MEL_INV_REF;
        }
        if (prof) { t1 = tick(); acc[2] += (t1 - t0) & mask; t0 = t1; }

        // 4) 로그 + int8 (포화)
        for (int m = 0; m < MEL_N_MELS; m++) {
            float q = floorf(MEL_K_LN * logf(mel[m] + MEL_EPS) + MEL_Q_OFF + 0.5f);
            if (q > 127.0f) q = 127.0f;
            if (q < -128.0f) q = -128.0f;
            out[m * MEL_N_FRAMES + f] = (int8_t)q;
        }
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
