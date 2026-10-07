// @generated — firmware/npu/sync_from_synth.sh 가 safesound_melu800_v21/main.c 에서 떼어 냈다.
// 손으로 고치지 말 것. 합성을 다시 하면 이 스크립트를 다시 돌린다.
#include <stdint.h>
#include <stdio.h>
#include "mxc.h"
#include "cnn.h"
#include "sampledata.h"
#include "sampleoutput.h"

// 1-channel 32x20 data input (640 bytes / 160 32-bit words):
// HWC 32x20, channels 0 to 0
static const uint32_t input_0[] = SAMPLE_INPUT_0;

void load_input(void)
{
  // This function loads the sample data input -- replace with actual data

  memcpy32((uint32_t *) 0x50400000, input_0, 640);
}

// Expected output of layer 6 for safesound_melu800_v21 given the sample input (known-answer test)
// Delete this function for production code
static const uint32_t sample_output[] = SAMPLE_OUTPUT;
int check_output(void)
{
  int i;
  uint32_t mask, len;
  volatile uint32_t *addr;
  const uint32_t *ptr = sample_output;

  while ((addr = (volatile uint32_t *) *ptr++) != 0) {
    mask = *ptr++;
    len = *ptr++;
    for (i = 0; i < len; i++)
      if ((*addr++ & mask) != *ptr++) {
        printf("Data mismatch (%d/%d) at address 0x%08x: Expected 0x%08x, read 0x%08x.\n",
               i + 1, len, addr - 1, *(ptr - 1), *(addr - 1) & mask);
        return CNN_FAIL;
      }
  }

  return CNN_OK;
}

