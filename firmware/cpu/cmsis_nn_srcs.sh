#!/bin/bash
# m4cmsis.c 가 쓰는 CMSIS-NN 소스 목록을 낸다 (PC 대조·펌웨어 빌드 공용).
#   CMSIS_NN=~/CMSIS-NN bash firmware/cpu/cmsis_nn_srcs.sh
# CMSIS-NN 은 레포에 넣지 않는다 (벤더 코드). `git clone https://github.com/ARM-software/CMSIS-NN`
# 측정에 쓴 커밋: 71cbe3d
C=${CMSIS_NN:-$HOME/CMSIS-NN}
ls "$C"/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c \
   "$C"/Source/ConvolutionFunctions/arm_convolve_s8.c \
   "$C"/Source/ConvolutionFunctions/arm_convolve_1x1_s8.c \
   "$C"/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.c \
   "$C"/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.c \
   "$C"/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.c \
   "$C"/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c \
   "$C"/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_row_offset_s8_s16.c \
   "$C"/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.c \
   "$C"/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c \
   "$C"/Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s8.c \
   "$C"/Source/NNSupportFunctions/arm_nn_mat_mul_core_4x_s8.c \
   "$C"/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.c \
   "$C"/Source/NNSupportFunctions/arm_s8_to_s16_unordered_with_offset.c
