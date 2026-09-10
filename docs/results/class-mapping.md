# 클래스 매핑 확정안 — 5클래스

- 확정일: 2026-09-10
- 생성: `scripts/build_class_manifest.py` → `data/interim/manifest.csv`
- 관련: [`vocabulary-check.md`](vocabulary-check.md) · [`alarm-class-analysis.md`](alarm-class-analysis.md)

이 문서는 **확정 사항**과 **잠정 사항**을 구분한다. 잠정 항목을 확정처럼
인용하지 말 것 (CLAUDE.md 10장).

---

## A. 확정

### A-1. 클래스 (5종)

| ID | 클래스 | 알림 등급 | 주 출처 |
|---|---|---|---|
| 0 | `siren` | 즉시 | FSD50K `Siren`, US8K `siren`, ESC-50 `siren` |
| 1 | `glass` | 즉시 | FSD50K `Shatter`/`Glass`, ESC-50 `glass_breaking` |
| 2 | `scream` | 즉시 | FSD50K `Screaming`/`Yell`/`Shout` |
| 3 | `dog_bark` | 기록 | FSD50K `Bark`/`Dog`, US8K `dog_bark`, ESC-50 `dog` |
| 4 | `background` | — | 나머지 전부 + 하드 네거티브 + LibriSpeech + MSnoise |

**제외한 클래스**
- `alarm` — 상위 라벨 `Alarm` 만 존재하고 실제 내용이 주차미터기·전철 출입문·
  찻물 타이머다. 화재경보는 586클립 중 24개. → [alarm-class-analysis.md](alarm-class-analysis.md)
- `baby_cry` — 전 소스 합쳐 176클립으로 다른 클래스의 1/5. 보강 소스 없음.

### A-2. 라벨 판정 규칙 (순서가 규칙이다)

```python
if "Siren" in L:                                    → siren
elif "Shatter" in L or ("Glass" in L and 식기류∉L):  → glass
elif {"Screaming","Yell","Shout"}∩L and 군중류∉L:   → scream
elif {"Bark","Dog"}∩L:                              → dog_bark
else:                                               → background 후보
```

FSD50K는 AudioSet 조상 라벨을 함께 부여한다 (`Siren` 132/132 = 100%가 `Alarm`
동시 보유). 다중라벨 필터를 문자열에 그대로 적용하면 siren이 전멸하므로,
**구체 클래스가 먼저 판정되도록 순서를 고정**한다.

배제 집합:
- **식기류**: `Chink_and_clink`, `Dishes_and_pots_and_pans`, `Cutlery_and_silverware`, `Coin_(dropping)`, `Tap`, `Liquid`, `Pour`
- **군중류**: `Cheering`, `Applause`, `Clapping`, `Crowd`, `Chatter`, `Laughter`, `Giggle`, `Chuckle_and_chortle`, `Music`, `Singing`

### A-3. 분할 — Freesound 원본 ID 단위 전역 분할

FSD50K·UrbanSound8K·ESC-50은 **모두 Freesound 파생**이며 ID 체계가 같다
(`fname` / `fsID` / `src_file`). 각 데이터셋의 공식 분할을 그대로 쓰면
**1,002개 원본이 2개 이상 데이터셋에 등장하고 337개가 train/test로 갈라진다**
(dog_bark 197, siren 62, glass 24, scream 3). CLAUDE.md 5장 규칙 1 위반이다.

→ 분할은 **원본 ID 단위로 전역 1회** 결정한다.

**FSD50K 공식 dev/eval을 그대로 쓰지 않은 이유**: 공식 eval은 클래스에 따라
test 비율이 15~30%로 들쭉날쭉해 평가 신뢰구간이 불필요하게 넓어진다. 본 연구는
FSD50K 벤치마크에 참가하지 않으므로 타 논문과의 수치 비교 이득이 없다.
대신 **공식 eval 원본은 전량 test에 유지**하고(전수 검증된 고품질 라벨이므로
train으로 내리지 않는다), 부족분만 dev에서 결정적 해시로 추가 이관해
클래스별 test 비율을 30%에 맞춘다. 논문 데이터셋 절에 이 사유를 명시할 것.

### A-4. 고유 원본 수 — 신뢰구간의 기준 단위

| 클래스 | train | test | test% | 95% CI | 출처별 원본 |
|---|---:|---:|---:|---:|---|
| siren | 137 | 65 | 32% | **±7.4%p** | FSD50K 132 / US8K 54 / ESC-50 19 |
| glass | 518 | 223 | 30% | ±4.0%p | FSD50K 739 / ESC-50 3 |
| scream | 412 | 191 | 32% | ±4.3%p | FSD50K 603 |
| dog_bark | 788 | 339 | 30% | ±3.2%p | FSD50K 932 / US8K 195 / ESC-50 19 |
| **합계** | **1,855** | **818** | | | |

CI는 재현율 90% 가정 시 95% Wilson 신뢰구간 반폭이다. **실측값이 아니라 표본
크기의 성질**이다. 한 원본에서 나온 슬라이스·윈도우는 독립 표본이 아니므로
CI는 반드시 **원본 수**로 계산한다. 윈도우 수로 계산하면 신뢰도가 과장된다.

### A-5. 오디오 파일 수 (데이터 양)

UrbanSound8K는 원본 하나에서 여러 슬라이스를 뜬다. **siren 929슬라이스는 고유
원본이 74개뿐**이다 (원본당 중앙값 9, 최대 100). 데이터 양과 표본 독립성은 다르다.

| 클래스 | train 파일 | test 파일 | 원본당 파일 |
|---|---:|---:|---:|
| siren | 860 | 126 | 4.88 |
| glass | 518 | 224 | 1.00 |
| scream | 412 | 191 | 1.00 |
| dog_bark | 1,170 | 448 | 1.44 |

원본이 FSD50K에 이미 있는 US8K/ESC-50 사본은 오디오 중복이므로 폐기했다
(siren 115, dog_bark 354, glass 37).

### A-6. 배경음 하드 네거티브 — 2,478클립

배제한 클립을 버리지 않고 배경음으로 편입한다. 오탐률에 직접 기여하며 G7의 근거다.

| 사유 | 클립 |
|---|---:|
| `alarm` 전량 (클래스 제외) | 1,724 |
| glass 배제 (식기·유리잔) | 500 |
| `baby_cry` (클래스 제외) | 138 |
| scream 배제 (군중·환호) | 116 |
| **합계** | **2,478** |

여기에 FSD50K 잔여 배경음 후보(약 46,000클립), US8K 나머지 6,803슬라이스,
LibriSpeech, MSnoise가 더해진다. 이벤트 윈도우 합의 2~3배로 샘플링한다.

---

## B. 잠정 — 실측 후 재판정

### B-1. 윈도우 수량

아래는 **클립 길이 기반 상한**이다. CLAUDE.md 5장 규칙 2의 RMS 에너지 임계값
필터를 적용하면 줄어든다. 특히 `glass`는 유리 파손이 0.3초 내외 과도음인데
클립 중앙값이 3.0초라, 에너지 필터를 제대로 걸면 **클립당 1개만 살아남을
가능성이 높다**. 그 경우 train 상한 1,059 → 518로 반토막 난다.

| 클래스 | train 상한 | test 상한 |
|---|---:|---:|
| siren | 2,549 | 372 |
| glass | 1,059 | 483 |
| scream | 867 | 433 |
| dog_bark | 2,947 | 1,141 |

정책: 비중첩 1초 윈도우, 클립당 최대 3개.

### B-2. 목표 달성 여부

CLAUDE.md 5장의 목표(클래스당 학습 800~1500 윈도우, 테스트 200 이상)를
**달성했다고 판정하지 않는다.** `prepare_safesound.py` 실행 후 실측으로 재판정한다.

---

## C. 남은 약점 (논문 한계에 기술)

1. **siren 원본 다양성이 낮다.** 고유 원본 202개(train 137)뿐이다. 오디오 파일은
   986개지만 US8K 슬라이스가 소수 원본에서 나온 것이라 실제 다양성은 낮다.
   특정 사이렌 녹음에 과적합할 위험이 있고, test 65원본이라 CI가 ±7.4%p로 넓다.
2. **US8K siren은 미국식 사이렌**이다. 한국 배치 환경과 음향 특성이 다르다.
3. **화재경보를 감지하지 않는다.** 데이터가 없다. 범위 제외로 명시한다.
