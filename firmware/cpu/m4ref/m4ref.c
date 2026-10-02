// ④ 의 M4 소프트웨어 추론 — **단순 C 참조 구현** (CMSIS-NN 아님).
//
// 대조군의 전제는 "같은 가중치" 다 (CLAUDE.md 6장). 가중치는 NPU 에 올린
// 양자화 체크포인트에서 그대로 뽑았고(tools/export_m4_weights.py), 정수 규약은
// ai8x-synthesis/izer/simulate.py 와 같다:
//     conv : out = clip( floor(0.5 + acc * 2^shift / 128), -128, 127 ), ReLU → 0..127
//     pool : max 2 / avg 2 (ReLU 뒤라 비음수 — 버림)
//     FC   : 32bit 출력, 스케일·포화 없음 (보드 값 = 누산값 << M4_FC_SHIFT)
//
// ⚠️ 이것은 **최적화하지 않은 참조 구현**이다 (SIMD·루프 전개·CMSIS-NN 없음,
//    컴파일러 -O2 만). 표에 "단순 C 참조 구현(-O2)" 로 적는다. CMSIS-NN 값은
//    이보다 빠를 것이므로, 이 값으로 "M4 는 불가능하다" 를 단정하지 않는다.
//
// 활성값 배치는 [시간][채널] 이다 — 한 출력의 내적이 연속 메모리를 훑는다.
// ④의 입력 x[채널 r][시간 c] = sample[c*128 + r] 은 이 배치에서 원본 파형 순서다.
#include "m4ref.h"
#include "m4_weights.h"

static int8_t buf_a[128 * 128];
static int8_t buf_b[128 * 128];

// in[n_in][ci] → out[n_out][co], w[co][k][ci]. 반환: n_out
static int conv1d_relu(const int8_t *in, int n_in, int ci, const int8_t *w, int co,
                       int k, int pad, int shift, int8_t *out)
{
    const int n_out = n_in + 2 * pad - k + 1;
    const int sh = 7 - shift;            // shift <= 0
    const int32_t rnd = 1 << (sh - 1);
    for (int t = 0; t < n_out; t++) {
        for (int o = 0; o < co; o++) {
            int32_t acc = 0;
            for (int j = 0; j < k; j++) {
                int ti = t + j - pad;
                if (ti < 0 || ti >= n_in) continue; // 0 패딩
                const int8_t *xp = &in[ti * ci];
                const int8_t *wp = &w[(o * k + j) * ci];
                for (int c = 0; c < ci; c++) acc += (int32_t)wp[c] * (int32_t)xp[c];
            }
            int32_t v = (acc + rnd) >> sh;   // 산술 시프트 = floor
            if (v < 0) v = 0;                // ReLU (아래쪽 포화를 겸한다)
            if (v > 127) v = 127;
            out[t * co + o] = (int8_t)v;
        }
    }
    return n_out;
}

// in[n_in][c] → out[n_in/2][c]. 입력은 ReLU 뒤라 비음수다.
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
    // voice_conv1 : k1, 128→100  (입력은 x 를 그대로 읽는다)
    n = conv1d_relu(x, 128, 128, w_voice_conv1, 100, 1, 0, m4_shift[0], buf_a);
    // voice_conv2 : k3, 100→96
    n = conv1d_relu(buf_a, n, 100, w_voice_conv2, 96, 3, 0, m4_shift[1], buf_b);
    // voice_conv3 : max2 + k3 pad1, 96→64
    n = pool2(buf_b, n, 96, 0, buf_a);
    n = conv1d_relu(buf_a, n, 96, w_voice_conv3, 64, 3, 1, m4_shift[2], buf_b);
    // voice_conv4 : k3, 64→48
    n = conv1d_relu(buf_b, n, 64, w_voice_conv4, 48, 3, 0, m4_shift[3], buf_a);
    // kws_conv1 : max2 + k3 pad1, 48→64
    n = pool2(buf_a, n, 48, 0, buf_b);
    n = conv1d_relu(buf_b, n, 48, w_kws_conv1, 64, 3, 1, m4_shift[4], buf_a);
    // kws_conv2 : k3, 64→96
    n = conv1d_relu(buf_a, n, 64, w_kws_conv2, 96, 3, 0, m4_shift[5], buf_b);
    // kws_conv3 : avg2 + k3 pad1, 96→100
    n = pool2(buf_b, n, 96, 1, buf_a);
    n = conv1d_relu(buf_a, n, 96, w_kws_conv3, 100, 3, 1, m4_shift[6], buf_b);
    // kws_conv4 : max2 + k6 pad1, 100→64
    n = pool2(buf_b, n, 100, 0, buf_a);
    n = conv1d_relu(buf_a, n, 100, w_kws_conv4, 64, 6, 1, m4_shift[7], buf_b);
    // fc : 64ch × 4 = 256 → 5. 평탄화는 채널 우선(c*4 + t)이다
    for (int o = 0; o < M4_NUM_OUTPUTS; o++) {
        int32_t acc = 0;
        for (int c = 0; c < 64; c++)
            for (int t = 0; t < n; t++)
                acc += (int32_t)w_fc[o * 256 + c * 4 + t] * (int32_t)buf_b[t * 64 + c];
        out[o] = acc * (1 << M4_FC_SHIFT);
    }
}
