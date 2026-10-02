# @generated — firmware/cpu/make_m4ref.sh
BOARD = FTHR_RevA
PROJ_CFLAGS += -DMEASURE_BUILD
# CMSIS-NN (/home/max78000/CMSIS-NN, 커밋 71cbe3d)
# ⚠️ SINGLE_ROUNDING 이 NPU 와의 비트 일치 조건이다 (m4cmsis.c 머리말)
PROJ_CFLAGS += -DM4_CMSIS -DCMSIS_NN_USE_SINGLE_ROUNDING
AUTOSEARCH = 0
SRCS += main.c m4cmsis.c
IPATH += /home/max78000/CMSIS-NN/Include
VPATH += /home/max78000/CMSIS-NN/Source/ConvolutionFunctions
VPATH += /home/max78000/CMSIS-NN/Source/NNSupportFunctions
SRCS += arm_convolve_1_x_n_s8.c
SRCS += arm_convolve_1x1_s8.c
SRCS += arm_convolve_1x1_s8_fast.c
SRCS += arm_convolve_get_buffer_sizes_s8.c
SRCS += arm_convolve_s8.c
SRCS += arm_convolve_wrapper_s8.c
SRCS += arm_nn_mat_mult_kernel_row_offset_s8_s16.c
SRCS += arm_nn_mat_mult_kernel_s8_s16.c
SRCS += arm_nn_mat_mult_s8.c
SRCS += arm_nn_mat_mul_core_1x_s8.c
SRCS += arm_nn_mat_mul_core_4x_s8.c
SRCS += arm_nn_mat_mult_nt_t_s8.c
SRCS += arm_q7_to_q15_with_offset.c
SRCS += arm_s8_to_s16_unordered_with_offset.c
