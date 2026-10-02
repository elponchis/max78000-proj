// 로그 멜 전처리 (G8 구성 ①의 CPU 몫) — datasets/melfeat.py 의 C 이식.
#ifndef MELFEAT_H
#define MELFEAT_H
#include <stdint.h>

// 한 번만 부른다 (FFT 인스턴스 초기화).
void melfeat_init(void);

// int8 파형 x[16384] → int8 로그 멜 out[64*64] (mel-major: out[m*64 + f]).
// melfeat.py 의 `log_mel_int8` 과 같은 값을 낸다 (float32 라 ±1 LSB 안).
void melfeat_int8(const int8_t *x, int8_t *out);

// 같은 계산을 단계별로 재면서 한다. prof[0..3] 에 **코어 사이클 합**을 더한다:
//   [0] 프레임 구성(reflect + 창)  [1] rfft  [2] 파워 + 멜  [3] 로그 + 양자화
// 단계마다 tick 을 읽는 비용이 섞이므로 **전체 지연은 melfeat_int8 로 잰다.**
void melfeat_int8_prof(const int8_t *x, int8_t *out, uint32_t prof[4],
                       uint32_t (*tick)(void), uint32_t mask);
#endif
