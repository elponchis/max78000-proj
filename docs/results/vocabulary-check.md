# Phase 2.1 — FSD50K 라벨 실재 확인 결과

- 실행일: 2026-09-10
- 스크립트: `scripts/check_vocabulary.py`
- 입력: FSD50K ground_truth (vocabulary 200 라벨 / dev 40,966 + eval 10,231 = 51,197 클립)
- 재현:
  ```bash
  # WSL2 (max78000)
  cd ~/max78000-proj
  python3 scripts/check_vocabulary.py \
      --gt-dir data/raw/FSD50K_meta/FSD50K.ground_truth --show-contamination
  ```

> 취득 경로 주의: Zenodo 접속 불가로 HuggingFace 미러(`Fhrozen/FSD50k`)에서 받았다.
> 상세는 `CLAUDE.md` 12장.

---

## 1. 라벨 실재 여부

CLAUDE.md 4장이 상정한 라벨 중 **5개가 vocabulary 200개 안에 없다.**

| 클래스 | 존재하는 라벨 | 부재 라벨 |
|---|---|---|
| siren | `Siren` | `Ambulance_(siren)`, `Police_car_(siren)`, `Fire_engine_and_fire_truck_(siren)` |
| alarm | `Alarm` | **`Alarm_clock`**, **`Smoke_detector_and_smoke_alarm`** |
| glass | `Glass`, `Shatter` | — |
| scream | `Screaming`, `Yell`, `Shout` | — |
| baby_cry | `Crying_and_sobbing` | — |
| dog_bark | `Bark`, `Dog` | — |

CLAUDE.md 4장이 예상한 대로 `Smoke_detector_and_smoke_alarm`은 부재하며,
`Alarm_clock`까지 함께 없어 **alarm은 상위 라벨 `Alarm` 하나로만 정의 가능**하다.

---

## 2. AudioSet 온톨로지 조상 라벨 문제

FSD50K는 조상 라벨을 함께 부여한다. 실측 동시 등장률:

| 라벨 | 동시 등장 라벨 | 비율 |
|---|---|---|
| `Siren` (132) | `Alarm` | **100.0%** |
| `Shatter` (510) | `Glass` | 100.0% |
| `Screaming` (377) | `Human_voice` | 100.0% |
| `Crying_and_sobbing` (151) | `Human_voice` | 100.0% |
| `Bark` / `Dog` (930) | `Animal`, `Domestic_animals_and_pets` | 100.0% |

`Siren ⊂ Alarm`이 클래스 간 포함관계이므로, 다중라벨 필터를 라벨 문자열에
그대로 적용하면 **siren이 0개로 전멸한다** (첫 실행에서 실제로 발생).

**해결**: `CLASS_PRIORITY = [siren, glass, scream, baby_cry, dog_bark, alarm]`.
구체 클래스가 상위 클래스를 이기고, `SUBSUMED_PAIRS = {(siren, alarm)}`에
해당하지 않는 다중 히트만 폐기한다.

---

## 3. 클래스별 클립 수 (dev/eval 분할 준수)

`raw` = 라벨 포함 클립 / `final` = 온톨로지 우선순위 + 다중라벨 필터 통과

| 클래스 | dev raw | dev final | eval raw | eval final | **final 합** | 판정 |
|---|---:|---:|---:|---:|---:|---|
| siren | 77 | 75 | 55 | 53 | **128** | 심각 부족 |
| alarm | 1,280 | 1,202 | 584 | 517 | **1,719** | 수량 충분 / 오염 |
| glass | 974 | 973 | 267 | 263 | **1,236** | 수량 충분 / 오염 |
| scream | 441 | 432 | 287 | 271 | **703** | 문제없음 |
| baby_cry | 109 | 101 | 42 | 35 | **136** | 심각 부족 |
| dog_bark | 752 | 749 | 178 | 170 | **919** | 문제없음 |
| background 후보 | 37,333 | — | 8,818 | — | **46,151** | 충분 |
| | | **3,532** | | **1,309** | **4,841** | |

다중 클래스 충돌 폐기: 총 36클립 (최다 `baby_cry+scream` 13, `alarm+scream` 6).

**목표 대비**: 클래스당 800윈도우, 클립당 1.5윈도우 가정 시 최소 533클립 필요.

| 클래스 | 보유 | 부족분 |
|---|---:|---:|
| siren | 128 | **-405** |
| baby_cry | 136 | **-397** |

---

## 4. 클래스 오염도 (`--show-contamination`)

### alarm — 오염 심각
`Alarm`은 AudioSet에서 광범위한 상위 카테고리다.

| 동시 등장 라벨 | 비율 |
|---|---:|
| `Telephone` | 36.7% |
| `Vehicle` | 28.1% |
| `Motor_vehicle_(road)` | 17.7% |
| `Car` | 11.9% |
| `Vehicle_horn_and_car_horn_and_honking` | 9.8% |
| `Ringtone` | 9.1% |
| `Doorbell` | 7.7% |

전화벨·차량 경적·초인종이 대량 포함되어 있어, **"화재/보안 경보"** 의미로
쓰려면 하위 라벨 배제 규칙이 필요하다.

### glass — 오염 중간

| 동시 등장 라벨 | 비율 |
|---|---:|
| `Chink_and_clink` (식기·유리잔 부딪힘) | 34.9% |
| `Dishes_and_pots_and_pans` | 8.8% |

`Shatter`(510클립)만 쓰면 깨끗하나 수량이 1,236 → 510으로 준다.

### siren — 차량 소음 동반

`Motor_vehicle_(road)` 58.3%, `Vehicle` 58.3%. 실배치 환경과 유사하므로
반드시 나쁘지는 않으나, 배경음 클래스에 차량 소음이 충분히 들어가야
siren을 "차량 소리"로 학습하지 않는다.

---

## 5. 결정 필요 사항 (미해결)

CLAUDE.md 4장 갱신 전 확정해야 한다.

1. **alarm 재정의** — `Alarm`에서 무엇을 배제할지.
   (예: `Telephone`, `Ringtone`, `Vehicle_horn_*`, `Doorbell` 제외)
2. **siren 보강** — FSD50K 128클립만으로 불가. UrbanSound8K `siren`(929클립) 필수.
3. **baby_cry 존치 여부** — FSD50K 136 + ESC-50 `crying_baby` 40 ≈ 176클립.
   목표 533에 크게 미달하며 보강 소스가 없다. **클래스 제외(6클래스) 검토 대상.**
4. **glass 범위** — `Glass`+`Shatter`(1,236, 오염) vs `Shatter`만(510, 깨끗).

> CLAUDE.md 4장은 `scream`을 최대 리스크로 지목했으나, **실측 결과는 반대**다.
> scream은 703클립으로 안전하고 실제 위험은 siren·baby_cry다.

## 6. 주의

위 수치는 **메타데이터 기반 상한**이다. RMS 에너지 필터(CLAUDE.md 5장 규칙 2)를
통과한 실제 윈도우 수는 더 적다. 논문 표에 넣을 확정 수치는
`prepare_safesound.py` 실행 후 `dataset_stats.py`로 산출한다.
