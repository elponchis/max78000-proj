# @generated — tools/fe_board_prep.py
# 측정 빌드: LED 마킹·소프트웨어 타이머를 뺀다 (CLAUDE.md 6장 MEASURE_BUILD).
BOARD = FTHR_RevA
PROJ_CFLAGS += -DMEASURE_BUILD -DNET_NAME=\"safesound_fe_v4d\"
PROJ_CFLAGS += -DFE_LOG_AB   # feat/log32: 10-17 주 값 펌웨어 원본 + 옛/새 로그 같은 세션 비교
