# @generated — tools/fe_board_prep.py
# 측정 빌드: LED 마킹·소프트웨어 타이머를 뺀다 (CLAUDE.md 6장 MEASURE_BUILD).
BOARD = FTHR_RevA
PROJ_CFLAGS += -DMEASURE_BUILD -DNET_NAME=\"safesound_fe_v4d\"
PROJ_CFLAGS += -DFE_LOG_AB -DFE_CPU_SPLIT   # feat/log32: CPU 몫 분해(읽기/로그) 옛·새 같은 세션 비교 (실험 II 조건)
