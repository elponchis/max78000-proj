# 라벨 노이즈 필터 — 전후 수량

`scripts/prepare_safesound.py --filter-report` 자동 생성. 손으로 고치지 말 것.

- 입력: 오디오가 있는 클립 5782개, 읽기 실패 0
- 에너지 필터: floor -60dBFS, rel 0.30, 클립당 최대 3윈도우, hop 250ms
- 윈도우 선택 정책: `--policy tag`
- 태거: PANNs Cnn14 16kHz (`Cnn14_16k_mAP=0.438.pth`), AudioSet 527클래스, 1초 윈도우 단위
- 태거 일치 규칙: 이벤트는 자기 라벨 묶음 최대 확률 ≥ 임계값, background 는 이벤트 라벨 묶음 전체 최대 확률 < 배경 임계값
- 임계값: siren 0.100, **glass 태거 미사용**, scream 0.025, dog_bark 0.100, background < 0.50
- 값은 **윈도우 수**(클립 수 명시한 곳 제외). 신뢰구간은 원본 수로 계산할 것

## 태거 라벨 묶음

| 클래스 | AudioSet 라벨 |
|---:|---:|
| siren | Siren, Civil defense siren, Police car (siren), Ambulance (siren), Fire engine, fire truck (siren), Emergency vehicle |
| glass | Shatter, Breaking, Smash, crash |
| scream | Screaming, Yell, Shout, Battle cry, Children shouting |
| dog_bark | Bark, Dog, Yip, Bow-wow |

## A. glass 하위유형 게이트 (onset strength)

클립 최대 onset strength 분포 (librosa `onset_strength`, hop 10ms, mel 64). Shatter 보유 클립은 무조건 채택, Glass-only 만 임계값 판정.

| 하위유형 | 클립 | p5 | p10 | p25 | p50 | p75 | p90 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Shatter 보유 (+ESC-50 glass_breaking) | 513 | 3.7 | 4.7 | 8.4 | 18.7 | 31.1 | 38.6 |
| Glass-only | 228 | 0.7 | 1.4 | 5.2 | 9.0 | 16.5 | 25.8 |

Glass-only 클립의 임계값별 잔존 (클립 수, train / test). 탈락분은 배경음 하드 네거티브.

| onset 임계값 | 채택 train | 채택 test | 탈락(→배경) |
|---:|---:|---:|---:|
| 2 | 155 | 46 | 27 |
| 4 | 142 | 41 | 45 |
| 6 | 117 | 39 | 72 |
| 8 ← 현재 | 96 | 35 | 97 |
| 10 | 72 | 30 | 126 |
| 12 | 59 | 26 | 143 |
| 15 | 42 | 23 | 163 |

## B. 단계별 수량 (현재 임계값)

클래스는 **실효 클래스**다(glass 게이트 탈락분은 background 에 포함). `최종` 은 클립당 상한 적용 후.
임계값 0 인 클래스(glass)는 태거를 쓰지 않으므로 `+태거 일치` 가 에너지 통과와 같다 — 차이는 청취 override 뿐이다.

| 클래스/split | 클립 | 후보 | 에너지 통과 | +태거 일치 | 최종(태거 전) | 최종(태거 후) | 태거 제거율 | 클립(태거 전) | 클립(태거 후) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| siren/train | 274 | 6489 | 4754 | 3598 | 659 | 539 | 18.2% | 270 | 243 |
| siren/test | 75 | 3173 | 2108 | 1510 | 187 | 151 | 19.3% | 74 | 61 |
| glass/train | 436 | 5583 | 2589 | 2581 | 633 | 633 | 0.0% | 436 | 436 |
| glass/test | 208 | 3373 | 1748 | 1742 | 326 | 326 | 0.0% | 208 | 208 |
| scream/train | 412 | 6632 | 4554 | 1712 | 740 | 490 | 33.8% | 412 | 317 |
| scream/test | 191 | 3987 | 2902 | 870 | 378 | 228 | 39.7% | 191 | 142 |
| dog_bark/train | 1165 | 21742 | 15327 | 9493 | 2427 | 1938 | 20.1% | 1165 | 1029 |
| dog_bark/test | 447 | 12638 | 8688 | 5000 | 985 | 782 | 20.6% | 447 | 378 |
| background/train | 1758 | 47301 | 30859 | 30365 | 3487 | 3466 | 0.6% | 1755 | 1747 |
| background/test | 816 | 30237 | 19741 | 19590 | 1863 | 1863 | 0.0% | 815 | 815 |

background 중 glass 게이트 탈락 재배정분: 클립 train 82 / test 15

## C. 이벤트 태거 임계값 민감도 (최종 윈도우 수, 전 이벤트 클래스 동일 임계값)

| 임계값 | siren/train | siren/test | glass/train | glass/test | scream/train | scream/test | dog_bark/train | dog_bark/test |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 필터 전 | 659 | 187 | 633 | 326 | 740 | 378 | 2427 | 985 |
| 0.020 | 581 | 159 | 395 | 211 | 499 | 235 | 2098 | 848 |
| 0.025 | 577 | 157 | 387 | 201 | 490 | 228 | 2074 | 841 |
| 0.030 | 571 | 155 | 377 | 198 | 478 | 223 | 2054 | 834 |
| 0.050 | 553 | 154 | 362 | 181 | 429 | 206 | 2012 | 818 |
| 0.100 | 539 | 151 | 325 | 173 | 370 | 183 | 1938 | 782 |
| 0.200 | 521 | 147 | 285 | 153 | 308 | 138 | 1842 | 745 |
| 0.300 | 506 | 143 | 258 | 132 | 254 | 123 | 1752 | 712 |
| 0.500 | 432 | 129 | 202 | 104 | 151 | 77 | 1452 | 622 |

## D. 배경음 태거 임계값 민감도 (최종 윈도우 수)

배경음 윈도우 중 이벤트 라벨 확률이 임계값 이상인 것을 제거한다.

| 배경 임계값 | background/train | background/test |
|---:|---:|---:|
| 필터 전 | 3487 | 1863 |
| 0.20 | 3442 | 1858 |
| 0.30 | 3453 | 1861 |
| 0.50 | 3466 | 1863 |
| 0.70 | 3481 | 1863 |

## E. 태거 불일치 윈도우의 AudioSet 최상위 라벨 (에너지 통과분, 상위 8)

우리 라벨과 불일치로 판정된 윈도우에 실제로 무엇이 들어 있었는지.

| 클래스 | 불일치 윈도우 | 최상위 라벨 (빈도) |
|---:|---:|---:|
| siren | 1751 | Music 549, Vehicle 326, Speech 171, Alarm 87, Train 51, Animal 45, Buzzer 38, Bird 38 |
| glass | 태거 미사용 | — |
| scream | 4866 | Speech 2587, Music 529, Vehicle 325, Animal 160, Fireworks 120, Siren 107, Whistle 55, Sigh 52 |
| dog_bark | 9522 | Animal 1540, Music 1328, Speech 894, Vehicle 825, Humming 317, Bird 285, Owl 158, Siren 132 |
| background | 645 | Siren 405, Children shouting 44, Cheering 37, Civil defense siren 31, Breaking 25, Animal 22, Screaming 21, Crowd 16 |

## F. 상위 라벨이 최상위인 불일치 윈도우의 자식 라벨 확률

AudioSet 온톨로지에서 `Animal`·`Alarm` 은 `Dog`·`Siren` 의 조상이다. 태거가 조상 라벨을 최상위로 내놓았다면 실제로는 우리 클래스인데 구체 라벨 확률이 낮아 걸러졌을 수 있다. 자식 확률이 임계값 근처에 몰려 있으면 임계값을 내리거나 상위 라벨을 라벨 묶음에 넣어야 한다는 신호다.

| 클래스 | 상위 라벨 | 윈도우 | 상위 확률 p50 | 자식최대 p50 | 자식최대 p90 | ≥0.02 | ≥0.05 | ≥0.10 | 현재 임계값 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dog_bark | Animal | 1540 | 0.164 | 0.034 | 0.082 | 1001 | 544 | 0 | 0.10 |
| dog_bark | Domestic animals, pets | 2 | 0.311 | 0.075 | 0.078 | 2 | 2 | 0 | 0.10 |
| dog_bark | Canidae, dogs, wolves | 1 | 0.489 | 0.086 | 0.086 | 1 | 1 | 0 | 0.10 |
| siren | Alarm | 87 | 0.517 | 0.033 | 0.042 | 83 | 0 | 0 | 0.10 |
| siren | Vehicle | 326 | 0.373 | 0.008 | 0.046 | 79 | 30 | 0 | 0.10 |

같은 윈도우들의 **자식 라벨별** 확률 중앙값. 확률이 여러 자식 라벨로 쪼개졌는지(→ 최대만 보면 과소평가) 확인용.

| 클래스 | 상위 라벨 | 윈도우 | Bark | Dog | Yip | Bow-wow |
|---:|---:|---:|---:|---:|---:|---:|
| dog_bark | Animal | 1540 | 0.002 | 0.034 | 0.004 | 0.006 |
| dog_bark | Domestic animals, pets | 2 | 0.006 | 0.075 | 0.007 | 0.006 |
| dog_bark | Canidae, dogs, wolves | 1 | 0.002 | 0.086 | 0.003 | 0.001 |

| 클래스 | 상위 라벨 | 윈도우 | Siren | Civil defense siren | Police car (siren) | Ambulance (siren) | Fire engine, fire truck (siren) | Emergency vehicle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| siren | Alarm | 87 | 0.033 | 0.000 | 0.011 | 0.001 | 0.001 | 0.003 |
| siren | Vehicle | 326 | 0.008 | 0.001 | 0.002 | 0.002 | 0.002 | 0.005 |

## G. 청취 판정 override

`data_overrides.csv` 의 48행 중 **48행이 현재 후보 윈도우와 매칭**됐다. 근거는 `docs/results/listening-verification.md`.

| action | 클래스 | 행 |
|---:|---:|---:|
| drop | glass | 14 |
| drop | scream | 10 |
| drop | siren | 3 |
| keep | glass | 11 |
| keep | scream | 10 |

