// @generated — firmware/npu/sync_from_synth.sh 가 safesound_wave_v21/main.c 에서 떼어 냈다.
// 손으로 고치지 말 것. 합성을 다시 하면 이 스크립트를 다시 돌린다.
#include <stdint.h>
#include <stdio.h>
#include "mxc.h"
#include "cnn.h"
#include "sampledata.h"
#include "sampleoutput.h"

// 128-channel 128x1 data input (16384 bytes total / 128 bytes per channel):
// HWC 128x1, channels 0 to 3
// HWC 128x1, channels 64 to 67
static const uint32_t input_0[] = SAMPLE_INPUT_0;

// HWC 128x1, channels 4 to 7
// HWC 128x1, channels 68 to 71
static const uint32_t input_4[] = SAMPLE_INPUT_4;

// HWC 128x1, channels 8 to 11
// HWC 128x1, channels 72 to 75
static const uint32_t input_8[] = SAMPLE_INPUT_8;

// HWC 128x1, channels 12 to 15
// HWC 128x1, channels 76 to 79
static const uint32_t input_12[] = SAMPLE_INPUT_12;

// HWC 128x1, channels 16 to 19
// HWC 128x1, channels 80 to 83
static const uint32_t input_16[] = SAMPLE_INPUT_16;

// HWC 128x1, channels 20 to 23
// HWC 128x1, channels 84 to 87
static const uint32_t input_20[] = SAMPLE_INPUT_20;

// HWC 128x1, channels 24 to 27
// HWC 128x1, channels 88 to 91
static const uint32_t input_24[] = SAMPLE_INPUT_24;

// HWC 128x1, channels 28 to 31
// HWC 128x1, channels 92 to 95
static const uint32_t input_28[] = SAMPLE_INPUT_28;

// HWC 128x1, channels 32 to 35
// HWC 128x1, channels 96 to 99
static const uint32_t input_32[] = SAMPLE_INPUT_32;

// HWC 128x1, channels 36 to 39
// HWC 128x1, channels 100 to 103
static const uint32_t input_36[] = SAMPLE_INPUT_36;

// HWC 128x1, channels 40 to 43
// HWC 128x1, channels 104 to 107
static const uint32_t input_40[] = SAMPLE_INPUT_40;

// HWC 128x1, channels 44 to 47
// HWC 128x1, channels 108 to 111
static const uint32_t input_44[] = SAMPLE_INPUT_44;

// HWC 128x1, channels 48 to 51
// HWC 128x1, channels 112 to 115
static const uint32_t input_48[] = SAMPLE_INPUT_48;

// HWC 128x1, channels 52 to 55
// HWC 128x1, channels 116 to 119
static const uint32_t input_52[] = SAMPLE_INPUT_52;

// HWC 128x1, channels 56 to 59
// HWC 128x1, channels 120 to 123
static const uint32_t input_56[] = SAMPLE_INPUT_56;

// HWC 128x1, channels 60 to 63
// HWC 128x1, channels 124 to 127
static const uint32_t input_60[] = SAMPLE_INPUT_60;

void load_input(void)
{
  // This function loads the sample data input -- replace with actual data

  memcpy32((uint32_t *) 0x50400000, input_0, 256);
  memcpy32((uint32_t *) 0x50408000, input_4, 256);
  memcpy32((uint32_t *) 0x50410000, input_8, 256);
  memcpy32((uint32_t *) 0x50418000, input_12, 256);
  memcpy32((uint32_t *) 0x50800000, input_16, 256);
  memcpy32((uint32_t *) 0x50808000, input_20, 256);
  memcpy32((uint32_t *) 0x50810000, input_24, 256);
  memcpy32((uint32_t *) 0x50818000, input_28, 256);
  memcpy32((uint32_t *) 0x50c00000, input_32, 256);
  memcpy32((uint32_t *) 0x50c08000, input_36, 256);
  memcpy32((uint32_t *) 0x50c10000, input_40, 256);
  memcpy32((uint32_t *) 0x50c18000, input_44, 256);
  memcpy32((uint32_t *) 0x51000000, input_48, 256);
  memcpy32((uint32_t *) 0x51008000, input_52, 256);
  memcpy32((uint32_t *) 0x51010000, input_56, 256);
  memcpy32((uint32_t *) 0x51018000, input_60, 256);
}

// Expected output of layer 8 for safesound_wave_v21 given the sample input (known-answer test)
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

