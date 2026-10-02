// ④ 의 M4 소프트웨어 추론 — **단순 C 참조 구현** (CMSIS-NN 아님).
#ifndef M4REF_H
#define M4REF_H
#include <stdint.h>

#define M4_WIN 16384
#define M4_NUM_OUTPUTS 5

// 1초 int8 파형 x[16384] → 로짓 5개 (NPU 의 32bit 출력과 **같은 값**).
// NPU 에 올린 것과 같은 가중치·같은 정수 규약(시프트·반올림·포화)이다.
void m4ref_infer(const int8_t *x, int32_t out[M4_NUM_OUTPUTS]);
#endif
