# `alarm` 클래스 제외 근거 — 온톨로지 라벨과 응용 의미의 괴리

- 조사일: 2026-09-10
- 데이터: FSD50K ground_truth + `dev/eval_clips_info_FSD50K.json` (Freesound 제목·태그)
- 결론: **`alarm` 을 클래스에서 제외한다.**

> 논문 데이터셋 절의 클래스 선정 근거로 쓸 것. AudioSet 온톨로지의 상위 라벨이
> 실제 응용 의미와 어긋나는 구체적 사례다.

---

## 1. 배경

CLAUDE.md 4장은 `alarm` 을 FSD50K `Alarm` / `Alarm_clock` /
`Smoke_detector_and_smoke_alarm` 으로 정의했다. 그러나 vocabulary 200개를
확인한 결과 **`Alarm_clock` 과 `Smoke_detector_and_smoke_alarm` 이 모두 부재**하며,
`Fire_alarm`, `Buzzer` 도 없다. 상위 라벨 `Alarm` 하나만 사용 가능하다.

`Alarm` 보유 클립에서 전화·벨·경적 계열(`Telephone`, `Ringtone`, `Doorbell`,
`Bell`, `Church_bell`, `Bicycle_bell`, `Cowbell`, `Chime`, `Wind_chime`,
`Vehicle_horn_and_car_horn_and_honking`)을 배제하고 `Siren` 보유 클립을
siren 으로 넘기면 **586클립**이 남는다.

## 2. 남은 586클립의 실제 내용

Freesound 제목·태그·설명을 키워드로 분류한 결과다 (중복 집계).

| 실제 내용 | 클립 | 비율 | 실제 제목 예시 |
|---|---:|---:|---|
| 일반 비프/전자음 | 226 | 38.6% | `parkingmeter.wav`, `pressing a light toggle switch` |
| 알람시계/타이머 | 86 | 14.7% | `Alarm Clock- E.mp3`, `old alarm clock ringing` |
| 철도/전철 | 75 | 12.8% | `London Underground- train doors closing`, `Train Door Closing.wav` |
| 보안/도난 경보 | 43 | 7.3% | `Car Alarm, Distant, A.wav` |
| **화재/연기 경보** | **24** | **4.1%** | `firealarm.wav`, `alarm_fatal.wav` |
| 차량 후진음 | 16 | 2.7% | `backup truck.flac` |
| 가전 비프음 | 6 | 1.0% | `microwave alarm.wav`, `ovenalarm.WAV` |
| 의료기기 | 6 | 1.0% | `intensive care ventilator alarm.wav` |
| 미분류 | 211 | 36.0% | 아래 참조 |

미분류 클립의 실제 제목:
`Bus Money FS.wav` · `Steam Whistle.mp3` · `sfx nebelhorn.mp3`(무적 경적) ·
`TAI_cay_saati_alarm_normal_reverb.wav`(터키어 "찻물 시간 알람") ·
`enemy detected.wav`(게임 효과음) · `Interior Prius door open close with key minder.wav`

**안전 관련 신호가 있는 클립은 21.3%뿐이고, 이 장치의 canonical 용도인
화재경보는 586개 중 24개(4.1%)다.**

## 3. 판단

`Alarm` 은 AudioSet 온톨로지에서 "경보 기능을 하는 소리"라는 **기능적 상위 범주**다.
주차미터기·전등 스위치·전철 출입문·찻물 타이머가 같은 라벨을 공유한다.
반면 본 프로젝트의 문제 정의는 "안전 관련 음향 이벤트 상시 감지"이고,
`alarm` 은 **즉시 알림** 등급으로 설계되어 있었다.

이 클래스를 유지하면:
- 즉시 알림이 주차미터기·전철 출입문에 발화한다 → G7(8시간 무인 구동 오탐률) 붕괴
- 화재경보를 감지한다고 주장할 근거가 24클립뿐이다 → 검증 불가
- `Siren ⊂ Alarm` 이라 siren 과 음향적으로도 인접해 혼동행렬이 지저분해진다

이름만 "경보·비프음"으로 바꿔 유지하는 방안도 검토했으나, 그 경우 클래스가
문제 정의와 무관해진다. **범위에서 제외하는 것이 정직하다.**

## 4. 처리

`Alarm` 보유 클립 **1,724개 전량을 배경음 하드 네거티브로 편입**한다.
버리지 않는다 — 초인종·주차미터기·전철 경고음을 배경음으로 학습시키면
실배치에서 그 소리들에 오탐하지 않는다. G7 에 직접 기여한다.

논문 "범위 제외" 절에 화재경보 감지를 명시하고, 향후 과제로
"화재경보 전용 데이터셋 구축 시 클래스 추가 가능"을 기술한다.

## 5. 재현

```bash
# WSL2 (max78000)
cd ~/max78000-proj
python3 scripts/check_vocabulary.py \
    --gt-dir data/raw/FSD50K_meta/FSD50K.ground_truth --show-contamination
```

제목·태그 분류는 `scripts/analyze_alarm_identity.py` 로 수행했다.
