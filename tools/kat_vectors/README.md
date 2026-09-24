# KAT 벡터 (SafeSound)

`tools/kat_safesound.py` 가 생성한다. **손으로 고치지 말 것.**

| 파일 | 내용 |
|---|---|
| `kat_window_int8.npy` | 전처리 출력 1초 창 (int8, 16384) |
| `kat_input2d_int8.npy` | (128,128) NPU 입력 |
| `kat_vectors.h` | 위 둘의 C 배열 — 펌웨어용 |

입력은 `(i % 255) - 127` 램프다. 자리마다 값이 달라 축이 뒤바뀌면
바로 드러난다.

## 쓰는 곳

1. PC: `python3 tools/kat_safesound.py` — 데이터로더 회귀 검사
2. 보드(관문 C): `kat_window` 를 펌웨어 전처리에 넣고 CNN 입력
   버퍼가 `kat_input2d` 와 일치하는지 확인. 어긋나면 전처리 축이나
   스케일이 PC 와 다른 것이다 (CLAUDE.md 7장).
