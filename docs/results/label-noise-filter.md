# 라벨 노이즈 필터 — 전후 수량

`scripts/prepare_safesound.py --filter-report` 자동 생성. 손으로 고치지 말 것.

- 입력: 오디오가 있는 클립 9202개, 읽기 실패 0
- 에너지 필터: int8 0비율 ≤ 0.95, rel 0.30, 클립당 최대 3윈도우, hop 250ms
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
| Shatter 보유 (+ESC-50 glass_breaking) | 513 | 6.0 | 8.0 | 12.0 | 19.6 | 30.4 | 37.9 |

Glass-only 클립의 임계값별 잔존 (클립 수, train / test). 탈락분은 배경음 하드 네거티브.

| onset 임계값 | 채택 train | 채택 test | 탈락(→배경) |
|---:|---:|---:|---:|
| 2 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 |
| 6 | 0 | 0 | 0 |
| 8 ← 현재 | 0 | 0 | 0 |
| 10 | 0 | 0 | 0 |
| 12 | 0 | 0 | 0 |
| 15 | 0 | 0 | 0 |

## B. 단계별 수량 (현재 임계값)

클래스는 **실효 클래스**다(glass 게이트 탈락분은 background 에 포함). `최종` 은 클립당 상한 적용 후.
임계값 0 인 클래스(glass)는 태거를 쓰지 않으므로 `+태거 일치` 가 에너지 통과와 같다 — 차이는 청취 override 뿐이다.

| 클래스/split | 클립 | 후보 | 에너지 통과 | +태거 일치 | 최종(태거 전) | 최종(태거 후) | 태거 제거율 | 클립(태거 전) | 클립(태거 후) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| siren/train | 274 | 6489 | 4684 | 3549 | 646 | 528 | 18.3% | 267 | 238 |
| siren/test | 75 | 3173 | 2095 | 1496 | 184 | 147 | 20.1% | 72 | 58 |
| glass/train | 359 | 4429 | 1896 | 1889 | 477 | 477 | 0.0% | 336 | 336 |
| glass/test | 154 | 2437 | 1141 | 1137 | 221 | 221 | 0.0% | 147 | 147 |
| scream/train | 412 | 6632 | 4564 | 1709 | 742 | 491 | 33.8% | 412 | 317 |
| scream/test | 191 | 3987 | 2809 | 861 | 375 | 225 | 40.0% | 191 | 142 |
| dog_bark/train | 1165 | 21742 | 15055 | 9441 | 2409 | 1934 | 19.7% | 1159 | 1030 |
| dog_bark/test | 447 | 12638 | 8478 | 4947 | 974 | 776 | 20.3% | 444 | 375 |
| background/train | 4520 | 134572 | 98706 | 97917 | 9604 | 9577 | 0.3% | 4519 | 4509 |
| background/test | 1605 | 61955 | 45611 | 45411 | 3839 | 3838 | 0.0% | 1605 | 1604 |

background 중 glass 게이트 탈락 재배정분: 클립 train 0 / test 0

## C. 이벤트 태거 임계값 민감도 (최종 윈도우 수, 전 이벤트 클래스 동일 임계값)

| 임계값 | siren/train | siren/test | glass/train | glass/test | scream/train | scream/test | dog_bark/train | dog_bark/test |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 필터 전 | 646 | 184 | 477 | 221 | 742 | 375 | 2409 | 974 |
| 0.020 | 569 | 155 | 394 | 179 | 503 | 234 | 2087 | 837 |
| 0.025 | 565 | 153 | 386 | 173 | 491 | 225 | 2060 | 830 |
| 0.030 | 559 | 151 | 383 | 170 | 479 | 220 | 2042 | 822 |
| 0.050 | 541 | 150 | 372 | 156 | 435 | 200 | 2002 | 806 |
| 0.100 | 528 | 147 | 342 | 149 | 378 | 178 | 1934 | 776 |
| 0.200 | 512 | 144 | 309 | 132 | 313 | 140 | 1842 | 735 |
| 0.300 | 499 | 140 | 280 | 117 | 262 | 121 | 1753 | 706 |
| 0.500 | 426 | 127 | 220 | 95 | 152 | 73 | 1478 | 615 |

## D. 배경음 태거 임계값 민감도 (최종 윈도우 수)

배경음 윈도우 중 이벤트 라벨 확률이 임계값 이상인 것을 제거한다.

| 배경 임계값 | background/train | background/test |
|---:|---:|---:|
| 필터 전 | 9604 | 3839 |
| 0.20 | 9540 | 3828 |
| 0.30 | 9560 | 3834 |
| 0.50 | 9577 | 3838 |
| 0.70 | 9596 | 3838 |

## E. 태거 불일치 윈도우의 AudioSet 최상위 라벨 (에너지 통과분, 상위 8)

우리 라벨과 불일치로 판정된 윈도우에 실제로 무엇이 들어 있었는지.

| 클래스 | 불일치 윈도우 | 최상위 라벨 (빈도) |
|---:|---:|---:|
| siren | 1731 | Music 526, Vehicle 325, Speech 181, Alarm 87, Train 51, Animal 45, Buzzer 38, Bird 38 |
| glass | 태거 미사용 | — |
| scream | 4797 | Speech 2588, Music 531, Vehicle 326, Animal 157, Siren 105, Whistle 55, Sigh 53, Groan 48 |
| dog_bark | 9145 | Animal 1444, Music 1287, Speech 878, Vehicle 828, Humming 317, Bird 278, Owl 145, Siren 131 |
| background | 951 | Siren 532, Animal 108, Breaking 50, Children shouting 44, Cheering 38, Civil defense siren 36, Crowd 26, Screaming 24 |

## F. 상위 라벨이 최상위인 불일치 윈도우의 자식 라벨 확률

AudioSet 온톨로지에서 `Animal`·`Alarm` 은 `Dog`·`Siren` 의 조상이다. 태거가 조상 라벨을 최상위로 내놓았다면 실제로는 우리 클래스인데 구체 라벨 확률이 낮아 걸러졌을 수 있다. 자식 확률이 임계값 근처에 몰려 있으면 임계값을 내리거나 상위 라벨을 라벨 묶음에 넣어야 한다는 신호다.

| 클래스 | 상위 라벨 | 윈도우 | 상위 확률 p50 | 자식최대 p50 | 자식최대 p90 | ≥0.02 | ≥0.05 | ≥0.10 | 현재 임계값 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dog_bark | Animal | 1444 | 0.165 | 0.034 | 0.083 | 950 | 509 | 0 | 0.10 |
| dog_bark | Domestic animals, pets | 1 | 0.392 | 0.079 | 0.079 | 1 | 1 | 0 | 0.10 |
| dog_bark | Canidae, dogs, wolves | 1 | 0.489 | 0.086 | 0.086 | 1 | 1 | 0 | 0.10 |
| siren | Alarm | 87 | 0.517 | 0.033 | 0.042 | 83 | 0 | 0 | 0.10 |
| siren | Vehicle | 325 | 0.374 | 0.009 | 0.046 | 79 | 30 | 0 | 0.10 |

같은 윈도우들의 **자식 라벨별** 확률 중앙값. 확률이 여러 자식 라벨로 쪼개졌는지(→ 최대만 보면 과소평가) 확인용.

| 클래스 | 상위 라벨 | 윈도우 | Bark | Dog | Yip | Bow-wow |
|---:|---:|---:|---:|---:|---:|---:|
| dog_bark | Animal | 1444 | 0.002 | 0.034 | 0.005 | 0.006 |
| dog_bark | Domestic animals, pets | 1 | 0.011 | 0.079 | 0.009 | 0.010 |
| dog_bark | Canidae, dogs, wolves | 1 | 0.002 | 0.086 | 0.003 | 0.001 |

| 클래스 | 상위 라벨 | 윈도우 | Siren | Civil defense siren | Police car (siren) | Ambulance (siren) | Fire engine, fire truck (siren) | Emergency vehicle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| siren | Alarm | 87 | 0.033 | 0.000 | 0.011 | 0.001 | 0.001 | 0.003 |
| siren | Vehicle | 325 | 0.008 | 0.001 | 0.002 | 0.002 | 0.002 | 0.005 |

## G. 청취 판정 override

`data_overrides.csv` 의 49행 중 **49행이 현재 후보 윈도우와 매칭**됐다. 근거는 `docs/results/listening-verification.md`.

| action | 클래스 | 행 |
|---:|---:|---:|
| drop | background | 26 |
| drop | glass | 11 |
| drop | scream | 10 |
| drop | siren | 3 |
| keep | glass | 11 |
| keep | scream | 10 |

