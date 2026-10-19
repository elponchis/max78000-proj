# @generated — tools/fe_board_prep.py
# 측정 빌드: LED 마킹·소프트웨어 타이머를 뺀다 (CLAUDE.md 6장 MEASURE_BUILD).
BOARD = FTHR_RevA
PROJ_CFLAGS += -DMEASURE_BUILD -DNET_NAME=\"safesound_fe_b64_v4d\"
