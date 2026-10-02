// 로그 멜 전처리 (G8 구성 ①의 CPU 몫) — datasets/melfeat.py 의 C 이식.
#ifndef MELFEAT_H
#define MELFEAT_H
#include <stdint.h>

// 한 번만 부른다 (FFT 인스턴스 초기화).
void melfeat_init(void);

// ── ① : hop 256 + reflect 패딩. 판단마다 64프레임 전부 ──────────────────
// int8 파형 x[16384] → int8 로그 멜 out[64*64] (mel-major: out[m*64 + f]).
// melfeat.py 의 `log_mel_int8` 과 같은 값을 낸다 (float32 라 ±1 LSB 안).
void melfeat_int8(const int8_t *x, int8_t *out);

// 같은 계산을 단계별로 재면서 한다. prof[0..3] 에 **코어 사이클 합**을 더한다:
//   [0] 프레임 구성(reflect + 창)  [1] rfft  [2] 파워 + 멜  [3] 로그 + 양자화
// 단계마다 tick 을 읽는 비용이 섞이므로 **전체 지연은 melfeat_int8 로 잰다.**
void melfeat_int8_prof(const int8_t *x, int8_t *out, uint32_t prof[4],
                       uint32_t (*tick)(void), uint32_t mask);

// ── ①′ : hop 250, 패딩 없음, 끝 정렬. 판단마다 새 프레임 16개만 ──────────
// 일괄 계산 (기준). x[16384] 의 샘플 122.. 를 쓴다. melfeat.py "loginc" 와 같다.
void melfeat_inc_batch(const int8_t *x, int8_t *out);

// 증분 계산. 스트림을 4,000 샘플(판단 주기)씩 넣는다.
//   reset(tail) : 링 버퍼를 비우고 직전 262 샘플을 준다 (없으면 NULL → 0)
//   push(x_new) : 새 4,000 샘플 → 프레임 16개를 계산해 링 버퍼에 넣는다
//   get(out)    : 최근 64프레임을 시간 순으로 out[m*64 + f] 에 쓴다
// push 를 4번 한 뒤의 get 은 같은 16,262 샘플에 대한 melfeat_inc_batch 와
// **비트 단위로 같다.**
void melfeat_inc_reset(const int8_t *tail);
void melfeat_inc_push(const int8_t *x_new);
void melfeat_inc_get(int8_t *out);
#endif
