# docs/paper — 논문용 표·그림 (자동 생성)

`tools/make_paper_assets.py` 가 **실측 결과 파일에서** 만든다. 이 폴더의 PNG·SVG·CSV·xlsx·`numbers.json` 은
손으로 고치지 않는다 — 원천이 바뀌면 다시 돌린다.

```bash
# WSL2, ai8x venv
OPENBLAS_NUM_THREADS=1 ~/ai8x-training/venv/bin/python tools/make_paper_assets.py
```

**기준 (2026-10-16 결정)**: 데이터 **v4+D**, 정답 **testL-S3** (시험셋, 청취 확정 비명 134창·유리 70창; 채점 창 4,965) ·
**real31L** (외부), 지연은 **v4+D 시드 1 보드 실측** (1000회 최악), 에너지는 전부 **추정(실측 지연 × 공개 전력)**.
testL 값은 부록 열에 나란히 둔다. 공개 전력값과 출처는 `paper_config.json` 에만 있다.

| 산출물 | 내용 |
|---|---|
| `fig1_acc_energy_test.*` / `fig1_acc_energy_ext.*` | 정확도–에너지 곡선 (x = 판단 1회 추정 에너지 μJ 로그축, y = 시험셋 F1 testL-S3 / 외부 F1). 마커 = (가) 데이터시트, 가로 막대 = 공개 전력값 범위, 세로 막대 = 시드 sd. (a) 는 빈 마커(지연 추정) |
| `fig2_latency_breakdown.*` | 판단 1회 지연 분해 (CPU 전처리 / 입력 적재(+출력) / NPU 추론), M4 CMSIS-NN 은 별도 축 |
| `fig3_recall_by_class.*` | 클래스별 재현율 묶음 막대 (testL-S3, real31L), PANNs CNN14 zero-shot 점선 (같은 정답·같은 문턱값 규칙) |
| `fig4_edition_ext_f1.*` | 데이터·정답 개선 단계별 외부 F1 (v1 → v2.1 → v3.1 → v4 → v4+D, 전부 real31L 로 채점), ④ 와 간격 400 |
| `table1_configs.csv` | 구성 정의 (프레임 간격, n_fft, 멜 수, 모델, 가중치 KB, NPU 사이클) |
| `table2_main.csv` | 주 결과 (구간별 지연, 추정 에너지 범위, 시험셋 F1 testL-S3 + 부록 testL, 외부 F1·mAP, 평균 ± sd, n) |
| `table3_welch.csv` | 핵심 비교 Welch 95% CI (가 − ④, 다 − ④, 간격 400 − ①′, 가 − (a), (a) − ④) |
| `table4_npu_vs_cmsis.csv` | NPU 대 CMSIS-NN (추론·판단 1회 시간과 배율, 추정 에너지) |
| `table5_breakeven.csv` | 손익분기 전력비 r* = P_npu/P_cpu 와 공개 전력값별 성립 여부 |
| `table6_listening.csv` | 청취 정답 요약 (세트별 창 수, O/X/? 비율과 Wilson 95%, 대조 창 일치율) |
| `paper_tables.xlsx` | 위 여섯 표를 시트로 |
| `numbers.json` | 본문 수치 — 값마다 원천 파일과 열 |

그림은 PNG 300 dpi + SVG(글자는 텍스트로 남김), 한글 글꼴 NanumGothic (없으면 Malgun Gothic), 흑백 인쇄용으로
마커 모양·해치를 다르게 했다.

**원천 파일**: `data/external_eval/scream_listened_test.json` (v4+D 7구성 × 시드, testL/testL-S3/외부),
`scream_stage3.json` (교차 검증·정답 창 수), `v4_eval.json` (판별 외부 F1·mAP), `v4_panns.json`,
`data/ceiling-panns-zs.npz` (PANNs 시험셋 → testL-S3 재채점), `data/logs-local/board_v4d_*.log` (보드 실측),
`docs/results/synthesis-check.md` §14.1·§12.1b·§10 (CPU 전처리 사이클), `data/synth/out/*_v4d/{log.txt,cnn.h}`,
`datasets/melfeat.py`, `data_listened*.csv`. `docs/results/curve-points-v4d.csv` 와 지연·에너지를 교차 검증한다
(0.02 ms / 1 μJ 넘게 다르면 멈춘다).
