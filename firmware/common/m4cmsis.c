// ④ 의 M4 소프트웨어 추론 — **CMSIS-NN 구현** (M4 쪽 최선 구현).
//
// m4ref.c (단순 C 참조 구현)와 **같은 가중치·같은 출력**이고 합성곱만
// CMSIS-NN `arm_convolve_wrapper_s8` 로 바꿨다 (Cortex-M4 DSP 확장: SMLAD 등).
//
// ── 비트 일치 조건 ──────────────────────────────────────────────────────
// ai8x 의 층 출력은  out = clip( floor(0.5 + acc / 2^sh), 0, 127 ),  sh = 7 - shift.
// CMSIS-NN 의 재양자화는 **`CMSIS_NN_USE_SINGLE_ROUNDING` 을 정의하면**
//     out = floor( acc * M / 2^(31 - s) + 0.5 )
// 이므로  M = 2^30,  s = 1 - sh  로 두면 **식이 정확히 같다** (M 이 2의
// 거듭제곱이라 근사가 없다). 기본값(이중 반올림)으로 빌드하면 음수 중간값에서
// 1 LSB 어긋날 수 있으므로 **반드시 그 매크로를 정의해 빌드한다.**
// 입력·출력 오프셋 0, 활성화 범위 [0, 127] (ReLU + 포화), bias 없음.
//
// 풀링·FC 는 m4ref.c 와 같은 단순 C 다 (전체의 1% 미만 — 풀링은 ai8x 의
// 버림 규약을 그대로 지키려는 것이고, FC 는 32bit 출력이라 재양자화가 없다).
//
// 배치: 활성값 [시간][채널] = CMSIS-NN 의 NHWC (H=1, W=시간).
//       가중치 w[co][k][ci] = CMSIS-NN 의 [out][kh=1][kw][in].
#include "m4ref.h"
#include "m4_weights.h"
#include "arm_nnfunctions.h"

static int8_t buf_a[128 * 128];
static int8_t buf_b[128 * 128];
static int16_t scratch[2 * 100 * 6 + 64]; // arm_convolve_s8: 2 * ci * kw * sizeof(int16)
static int32_t q_mult[100], q_shift[100];

int m4cmsis_status; // 0 이 아니면 CMSIS-NN 이 인자를 거부했다

static int conv1d_relu(const int8_t *in, int n_in, int ci, const int8_t *w, int co, int k,
                       int pad, int shift, int8_t *out)
{
    const int n_out = n_in + 2 * pad - k + 1;
    const int sh = 7 - shift;
    for (int o = 0; o < co; o++) {
        q_mult[o] = 1 << 30;
        q_shift[o] = 1 - sh;
    }
    const cmsis_nn_context ctx = { .buf = scratch, .size = sizeof(scratch) };
    const cmsis_nn_conv_params cp = {
        .input_offset = 0, .output_offset = 0,
        .stride = { .w = 1, .h = 1 }, .padding = { .w = pad, .h = 0 },
        .dilation = { .w = 1, .h = 1 }, .activation = { .min = 0, .max = 127 },
    };
    const cmsis_nn_per_channel_quant_params qp = { .multiplier = q_mult, .shift = q_shift };
    const cmsis_nn_dims in_d = { .n = 1, .h = 1, .w = n_in, .c = ci };
    const cmsis_nn_dims f_d = { .n = co, .h = 1, .w = k, .c = ci };
    const cmsis_nn_dims b_d = { .n = 1, .h = 1, .w = 1, .c = co };
    const cmsis_nn_dims out_d = { .n = 1, .h = 1, .w = n_out, .c = co };
    if (arm_convolve_wrapper_s8_get_buffer_size(&cp, &in_d, &f_d, &out_d) > (int32_t)sizeof(scratch))
        m4cmsis_status |= 2;
    if (arm_convolve_wrapper_s8(&ctx, &cp, &qp, &in_d, in, &f_d, w, &b_d, 0, &out_d, out) !=
        ARM_CMSIS_NN_SUCCESS)
        m4cmsis_status |= 1;
    return n_out;
}

// in[n_in][c] → out[n_in/2][c]. 입력은 ReLU 뒤라 비음수다 (ai8x: 0 쪽으로 버림).
static int pool2(const int8_t *in, int n_in, int c, int avg, int8_t *out)
{
    const int n_out = n_in / 2;
    for (int t = 0; t < n_out; t++) {
        const int8_t *a = &in[(2 * t) * c], *b = &in[(2 * t + 1) * c];
        for (int i = 0; i < c; i++) {
            if (avg) out[t * c + i] = (int8_t)(((int)a[i] + (int)b[i]) / 2);
            else out[t * c + i] = a[i] > b[i] ? a[i] : b[i];
        }
    }
    return n_out;
}

void m4ref_infer(const int8_t *x, int32_t out[M4_NUM_OUTPUTS])
{
    int n;
    n = conv1d_relu(x, 128, 128, w_voice_conv1, 100, 1, 0, m4_shift[0], buf_a);
    n = conv1d_relu(buf_a, n, 100, w_voice_conv2, 96, 3, 0, m4_shift[1], buf_b);
    n = pool2(buf_b, n, 96, 0, buf_a);
    n = conv1d_relu(buf_a, n, 96, w_voice_conv3, 64, 3, 1, m4_shift[2], buf_b);
    n = conv1d_relu(buf_b, n, 64, w_voice_conv4, 48, 3, 0, m4_shift[3], buf_a);
    n = pool2(buf_a, n, 48, 0, buf_b);
    n = conv1d_relu(buf_b, n, 48, w_kws_conv1, 64, 3, 1, m4_shift[4], buf_a);
    n = conv1d_relu(buf_a, n, 64, w_kws_conv2, 96, 3, 0, m4_shift[5], buf_b);
    n = pool2(buf_b, n, 96, 1, buf_a);
    n = conv1d_relu(buf_a, n, 96, w_kws_conv3, 100, 3, 1, m4_shift[6], buf_b);
    n = pool2(buf_b, n, 100, 0, buf_a);
    n = conv1d_relu(buf_a, n, 100, w_kws_conv4, 64, 6, 1, m4_shift[7], buf_b);
    for (int o = 0; o < M4_NUM_OUTPUTS; o++) {
        int32_t acc = 0;
        for (int c = 0; c < 64; c++)
            for (int t = 0; t < n; t++)
                acc += (int32_t)w_fc[o * 256 + c * 4 + t] * (int32_t)buf_b[t * 64 + c];
        out[o] = acc * (1 << M4_FC_SHIFT);
    }
}
