# 합성 검증 — ④ 기준선과 D-1 (2026-09-29, WSL2)

CLAUDE.md 3장 "모델 수정 시 442KB / 512KB 제약 초과 여부를 `ai8xize.py` 합성으로
조기 검증할 것" 에 대한 실행 기록이다.

확인하려던 것 두 가지:
1. ④ 와 D-1 이 **442KB 가중치 / 512KB 데이터** 안에 들어가는가
2. **D-1 의 `Abs` 활성화가 실제로 합성되는가** (학습에서만 되고 합성에서 막히면
   D-1 은 논문에 못 쓴다)

**둘 다 통과했다.**

---

## 1. 입력

| | ④ 기준선 | D-1 |
|---|---|---|
| 학습 로그 | `safesound-v1cpu___2026.09.27-223329` | `safesound-fbabs-v1___2026.09.27-204346` |
| 체크포인트 | `safesound-v1cpu_qat_best.pth.tar` (epoch 137) | `safesound-fbabs-v1_qat_best.pth.tar` (epoch 129) |
| 진입점 | `ai85safesoundnet` | `ai85safesoundnet_fb` |
| 첫 층 활성화 | ReLU | **Abs** |
| 합성 설정 | `synthesis/safesound-wave-hwc.yaml` | `synthesis/safesound-fb-hwc.yaml` |

두 yaml 은 `ai8x-synthesis/networks/kws20-v3-hwc.yaml` 에서 파생했다. 우리 백본이
KWS20 v3 와 **층 구성·채널 수가 동일**하고 출력 클래스 수만 21 → 5 라서 층 서술과
프로세서 마스크가 그대로 성립한다(클래스 수는 yaml 에 없다).

두 yaml 의 차이는 **정확히 두 줄**이다 — 그래야 비교가 오염되지 않는다.

```diff
-arch: ai85safesoundnet
+arch: ai85safesoundnet_fb
-    activate: ReLU      # 첫 층
+    activate: Abs
```

---

## 2. 절차 (WSL2, `~/ai8x-synthesis/venv`)

```bash
# 1) 양자화
cd ~/ai8x-synthesis
venv/bin/python quantize.py <qat_best>.pth.tar <out>-q8.pth.tar --device MAX78000

# 2) 샘플 입력 생성 — ai8x-training 쪽에서, **양자화 체크포인트로**
cd ~/ai8x-training
venv/bin/python train.py --model ai85safesoundnet --dataset SafeSound \
  --device MAX78000 --cpu --evaluate -8 --save-sample 10 --confusion \
  --exp-load-weights-from ~/max78000-proj/data/synth/safesound-wave-q8.pth.tar
cp sample_safesound.npy ~/ai8x-synthesis/tests/

# 3) 합성
cd ~/ai8x-synthesis
venv/bin/python ai8xize.py --verbose --test-dir <out> --prefix safesound_wave \
  --checkpoint-file ...-q8.pth.tar --config-file .../safesound-wave-hwc.yaml \
  --softmax --device MAX78000 --compact-data --mexpress --timer 0 \
  --display-checkpoint --overwrite
```

⚠️ **함정 하나 (2026-09-29 겪음).** 2번 단계에서 `-8` 평가에 **QAT 체크포인트를
그대로 주면 안 된다.** `qat_best.pth.tar` 를 주면 전 배치 손실이 정확히
`ln 5 = 1.609438`, Top1 4.3% 로 나온다 — 모든 로짓이 같은 값이라는 뜻이다.
`-8` 모드는 가중치가 **정수**라고 가정하므로 `quantize.py` **출력**을 줘야 한다.
바꿔 주면 Top1 72.38% 가 나온다. 손실이 `ln(클래스수)` 와 정확히 같으면 이 함정을
의심할 것.

---

## 3. 결과 — 자원 사용량

**두 모델이 완전히 같다.** 층 구성과 채널 수가 같고 활성화만 다르므로 당연하지만,
Abs 가 추가 자원을 먹지 않는다는 확인이기도 하다.

```
TOTAL: 9 parameter layers, 165,376 parameters, 165,376 bytes

RESOURCE USAGE
Weight memory: 165,376 bytes out of 442,368 bytes total (37.4%)
Bias memory:   0 bytes out of 2,048 bytes total (0.0%)

Total 71,718 cycles
Hardware: 8,398,432 ops (8,341,248 macc; 54,496 comp; 2,688 add)
```

| 자원 | 사용 | 한계 | 비율 |
|---|---:|---:|---:|
| 가중치 메모리 | 165,376 B | 442,368 B | **37.4%** |
| 바이어스 메모리 | 0 B | 2,048 B | 0.0% |
| 데이터 메모리 (동시 최대) | 29,184 B | 524,288 B | **5.6%** |
| 채널당 데이터 (HWC) | 128 px | 8,192 px | **1.6%** |

> `bias=False` 라 바이어스 메모리가 0 이다. KWS20 체크포인트의 bias 를 받는
> `ai85safesoundnet_bias` 를 쓰게 되면 이 칸이 채워진다 (9층 × 출력채널 ≈ 637 B).

### 데이터 메모리 산출 근거

합성 도구는 데이터 메모리 총량을 따로 찍지 않는다. 로그의 층별 차원에서 센다.

| 층 | 출력 | 바이트 |
|---|---|---:|
| 입력 | 128 ch × 128 | 16,384 |
| L0 voice_conv1 | 100 × 128 | 12,800 |
| L1 voice_conv2 | 96 × 126 | 12,096 |
| L2 voice_conv3 | 64 × 63 | 4,032 |
| L3 voice_conv4 | 48 × 61 | 2,928 |
| L4 kws_conv1 | 64 × 30 | 1,920 |
| L5 kws_conv2 | 96 × 28 | 2,688 |
| L6 kws_conv3 | 100 × 14 | 1,400 |
| L7 kws_conv4 | 64 × 4 | 256 |

버퍼는 `0x0000` ↔ `0x2000` 을 **핑퐁**한다(로그의 Input/Output offset 교대).
동시에 살아 있는 것은 인접한 두 개뿐이므로 최대는 **입력 + L0 = 29,184 B** 다.

채널당 제약이 더 빡빡한 쪽인데(HWC 는 채널당 8,192 px), 가장 긴 것이 128 px 라
**여유가 64배**다. 모델을 1.5× 로 키워도 채널 수만 늘고 길이는 그대로라 이쪽은
문제가 되지 않는다 — **병목은 가중치 메모리 하나**다.

### 여유

가중치 37.4% 는 `models/ai85net-safesound.py` 헤더의 파라미터 수 기반 추정
36.6% 와 거의 일치한다(합성은 정렬 때문에 조금 더 쓴다). 그 표의 1.5× = 80.6%
추정도 그대로 믿을 만하다 → **8bit 상한 약 1.6×** 라는 결론이 합성으로 확인됐다.

---

## 4. Abs 활성화 — 합성된다 ✔

D-1 합성 로그:

```
Activation          = [Abs, ReLU, ReLU, ReLU, ReLU, ReLU, ReLU, ReLU, no]
100x128 OUTPUT BEFORE ACTIVATION:
100x128 ACTIVATED OUTPUT (ABS):
96x126 OUTPUT BEFORE ACTIVATION:
96x126 ACTIVATED OUTPUT (RELU):
```

첫 층만 ABS, 나머지 ReLU 로 정확히 들어갔다. 파서 쪽 근거는
`ai8x-synthesis/izer/op.py:28` 의 `ACT_ABS: 'Abs'` 와
`izer/yamlcfg.py:305` 의 `elif ll[key].lower() == 'abs':` 다.

→ **D-1 을 논문 구성으로 쓰는 데 하드웨어 제약상 걸림돌이 없다.**

---

## 5. 생성물

```
data/synth/
├── safesound-wave-q8.pth.tar          # ④ 양자화
├── safesound-fb-q8.pth.tar            # D-1 양자화
├── sample_safesound-wave.npy          # 합성 KAT 입력 (테스트셋 index 10)
└── out/
    ├── safesound_wave/{cnn.c,cnn.h,weights.h,sampledata.h,sampleoutput.h,main.c,Makefile,...}
    └── safesound_fb/{...}
```

`data/` 는 git 제외라 생성물은 레포에 없다. 재생성 명령은 2절 그대로다.
yaml 두 개는 레포에 있다 (`synthesis/`).

---

## 6. 아직 확인하지 않은 것

- **온디바이스 KAT.** `sampleoutput.h` 는 합성 도구의 시뮬레이션 값이다.
  실제 하드웨어가 같은 값을 내는지는 보드에서 이 프로젝트를 빌드·실행해야
  안다 (`docs/board-bringup.md` 를 먼저 끝낼 것).
- **C 빌드.** MSDK 가 이 PC 에 아직 없어서 `cnn.c` 를 컴파일해 보지 못했다.
  MSDK 설치 후 `data/synth/out/safesound_wave/` 에서
  `make BOARD=FTHR_RevA` 로 확인한다.
- **D-1 k=2 / k=4 의 합성.** 첫 층 `kernel_size` 만 1 → 2 / 4 로 바꾸면 되지만,
  **가중치가 12,800 → 25,600 / 51,200 바이트로 는다**(+2.9%p / +8.7%p).
  학습 결과를 보고 채택이 정해지면 그때 합성한다.
- **CMSIS-NN 대조군.** 같은 `weights.h` 를 `arm_convolve_*_s8` 에 매핑한다
  (CLAUDE.md 6장). 보드 대기.

---

## 7. 구성 ① (로그 멜 2D CNN) 합성 — **통과** (2026-10-02)

| | ① 로그 멜 | (비교) ④ |
|---|---:|---:|
| 체크포인트 | `data/safesound-mel-v1_qat_best.pth.tar` (s1) | — |
| 합성 설정 | `synthesis/safesound-mel-hwc.yaml` | `safesound-wave-hwc.yaml` |
| 가중치 | **157,472 B (35.6%)** | 165,376 B (37.4%) |
| 바이어스 | 0 B | 0 B |
| **NPU 사이클** | **204,029** | 71,718 (**①이 2.84배**) |
| 연산 | 19,315,712 ops (18.9M macc) | 8,398,432 ops |
| 데이터 메모리 (동시 최대) | **135,168 B (25.8%)** — 입력 4,096 + L0 출력 32×64×64 | 29,184 B (5.6%) |
| 채널당 픽셀 (HWC 한도 8,192) | **4,096 (50%)** | 128 |
| `-8` 평가 Top1 (양자화 체크포인트) | 87.08% | — |

- 층별 연산은 L1(32→32, 64×64 입력)이 9.6M ops 로 절반이다.
- ⚠️ **사이클 수는 합성 로그 값**이다. 지연은 보드 DWT 로 잰다. ①은 여기에
  **CPU 전처리(STFT+멜+로그)가 더** 붙는다 — NPU 만으로도 ④의 2.84배라는 것은
  "①이 정확도를 얻는 대신 NPU 쪽에서도 비용을 더 낸다" 는 뜻이다.
- 재생성: `bash data/logs-local/synth_mel.sh` (양자화 → `-8` 샘플 → `ai8xize.py`).
  산출물 `data/synth/out/safesound_mel/`.

## 8. ④ C 빌드와 측정 펌웨어 (2026-10-02)

6절의 "C 빌드" 는 해소됐다 — MSDK 는 WSL 에 있었다 (`~/msdk`).

- `firmware/npu/sync_from_synth.sh <prefix>` 가 합성 산출물에서 벤더 생성
  파일을 가져오고 main 을 `firmware/common/measure_main.c` 로 바꾼다
  (KAT 비트 대조 + DWT 1000회, LED·타이머 없음).
- `bash scripts/fw_build.sh firmware/npu/safesound_wave` →
  **Flash 236,140 B (45.0%) / SRAM 16,852 B (12.9%)**.
- 플래싱 `** Verified OK **`.

### 온디바이스 KAT — **통과** (2026-10-02, POR 후 시리얼)

```
KAT sampleoutput: PASS (bit-exact)
  class 0..4 raw: -259664 / -248192 / 89872 / -34416 / 31632
1000 iterations, output mismatches: 0
```

보드의 NPU 출력이 합성기의 `sampleoutput.h` 와 **비트 단위로 같고**, 1000회
반복에서도 불일치가 없다. (6절 "온디바이스 KAT" 해소.)

### 지연 — ④, 1000회 (2026-10-02)

| 구간 | 최소 | 중앙 | 최악 | 100 MHz 환산 (최악) |
|---|---:|---:|---:|---:|
| load (입력 16,384 B 적재) | 107,832 | 107,832 | 108,740 | **1.087 ms** |
| **infer** (`cnn_start` ~ 완료 플래그) | 143,926 | 143,926 | 143,926 | **1.4393 ms** |
| unload + softmax | 801 | 801 | 801 | 8.0 µs |

단위는 코어 사이클 (`SystemCoreClock 100000000 Hz` 시리얼 표시 확인). 합계
최악 **253,467 사이클 ≈ 2.53 ms** = hop 250 ms 의 **1.0%**.

**이 표는 POR 직후 시리얼로 읽은 값이다** (펌웨어 빌드 `Oct 2 2026 11:58:07`).

- **infer 는 1000회 전부 같은 값**이다 (결정적). 합성 로그의 71,718 CNN
  사이클 ÷ 50 MHz = 1.434 ms 와 맞는다 (+5 µs 는 시작·인터럽트 진입).
- 같은 펌웨어를 **POR 없이**(SWD 리셋 직후) 돌려 RAM 에서 읽었을 때는 infer 가
  144,950 (1.4495 ms)로 **1,024 사이클(10 µs) 더 길었다.** load·unload 는
  같았다. 원인은 모른다 — **보고 값은 POR 조건 것만 쓴다.**
- **load 가 infer 의 75%** 다. `memcpy32` 16회로 4,096 워드를 옮기는 비용이다 —
  실제 펌웨어에서는 마이크 버퍼에서 NPU 로 옮기는 단계가 이 자리에 온다.
- ⚠️ **카운터는 DWT 가 아니라 SysTick 이다.** MAX78000 의 M4 에는 CYCCNT 가
  없다 (`DWT->CTRL = 0x4f000001`, NOCYCCNT=1, CYCCNT 항상 0). SysTick 도 코어
  클럭을 세므로 단위는 같다.
- 시리얼이 안 나올 때는 `scripts/fw_peek.sh` 로 결과 배열을 SWD 로 RAM 에서
  읽을 수 있다 (측정이 끝난 뒤 읽으므로 측정 구간에 끼어들지 않는다). 다만
  위 항목대로 **POR 없는 상태의 값은 보고에 쓰지 않는다.**
- ⚠️ 추론 대기는 바쁜 대기다. 슬립 대기 판의 지연·전력은 외부 계측 몫이다.
- 함정 기록: 측정용으로 SysTick 을 직접 켠 뒤 `MXC_Delay` 를 부르면 멈춘다
  (오버플로 인터럽트를 기다린다). 측정 뒤에는 바쁜 대기를 쓴다.

## 9. ① 온디바이스 KAT·지연 — NPU 구간 (2026-10-02)

`firmware/npu/safesound_mel` (Flash 221,692 B = 42.3% / SRAM 16,972 B = 12.9%).
**POR 직후 시리얼**, 1000회, 코어 100 MHz / CNN 50 MHz (PCLK÷1), **바쁜 대기**,
카운터 SysTick.

```
KAT sampleoutput: PASS (bit-exact)      1000 iterations, output mismatches: 0
```

| 구간 (코어 사이클) | 최소 | 중앙 | 최악 | 시간 (최악) |
|---|---:|---:|---:|---:|
| load (입력 1ch 64×64 = 4,096 워드) | 107,922 | 107,922 | 108,512 | 1.085 ms |
| **infer** | 408,512 | 408,512 | 408,522 | **4.085 ms** |
| unload + softmax | 765 | 765 | 809 | 8.1 µs |

- 합계 최악 **517,843 사이클 ≈ 5.18 ms** = hop 의 2.1%. **CPU 멜 전처리는
  아직 포함되지 않았다** (`TBD`).
- infer 는 합성 로그 204,029 CNN 사이클 ÷ 50 MHz = 4.081 ms 와 맞는다.
  **④(1.439 ms)의 2.84배** — 합성 사이클 비와 같다.
- load 가 ④와 거의 같다(1.085 vs 1.087 ms). 바이트 수는 ①이 1/4 이지만
  1채널 입력은 워드당 1픽셀이라 **옮기는 워드 수가 4,096 으로 같다.**

## 10. ① CPU 전처리 (로그 멜) — 온디바이스 KAT·지연 (2026-10-02)

`firmware/cpu/melfeat` (Flash 151,484 B = 28.9% / SRAM 17,300 B = 13.2%, `-O2`,
float32, CMSIS-DSP 1.16.2 `arm_rfft_fast_f32`). **POR 직후 시리얼**, 1000회,
코어 100 MHz, **NPU 꺼짐**, 카운터 SysTick.

```
KAT vs melfeat.py: mismatches 0/4096  max diff 0 LSB  -> PASS
```

보드(CMSIS FFT 포함)의 int8 로그 멜이 `melfeat.py` 기준 벡터와 **전부 같다.**

| 1초 창 → 64×64 int8 | 최소 | 중앙 | 최악 | 시간 (최악) |
|---|---:|---:|---:|---:|
| **전처리 전체** (코어 사이클) | 4,783,509 | 4,783,526 | 4,787,112 | **47.87 ms** |

단계별 (별도 1회 통과, 64프레임 합):

| 단계 | 사이클 | 비중 | 프레임당 |
|---|---:|---:|---:|
| 프레임 구성 (reflect + 창 512) | 812,825 | 17% | 12,700 |
| **rfft 512** | **2,423,774** | **51%** | 37,871 |
| 파워 + 멜 (희소 499 가중치) | 644,159 | 13% | 10,065 |
| 로그 + 양자화 (`logf` 64회) | 900,363 | 19% | 14,068 |

### ① 종단간 (전처리 + NPU)

| | ④ 원본 파형 | ① 로그 멜 |
|---|---:|---:|
| CPU 전처리 | — | 47.87 ms |
| 입력 적재 | 1.087 ms | 1.085 ms |
| NPU 추론 | 1.439 ms | 4.085 ms |
| 출력 | 0.008 ms | 0.008 ms |
| **합계 (최악값의 합)** | **2.53 ms** | **53.05 ms** |
| hop 250 ms 대비 | 1.0% | **21.2%** |

- **①은 ④의 20.9배**이고 그 **90%가 CPU 전처리**다. NPU 추론(4.085 ms)은
  ① 전체의 7.7% 뿐이다 — 가속기를 달아도 앞단이 지배한다.
- 그래도 hop 의 21% 라 **실시간 제약은 충족한다.** "①은 성립하지 않는다" 가
  아니라 **"정확도 0.138 을 얻는 대가가 지연 20.9배"** 다.
- ⚠️ **최적화 전 첫 구현이다.** 대조군에는 최선의 구현을 줘야 한다 —
  줄일 여지: 프레임 구성의 분기 제거, `logf` 근사(이미 KAT 가 있는
  `log2_q8` 류), rfft 는 CMSIS 라 그대로. 줄인 값도 함께 보고한다.
- ⚠️ **증분 계산 불가**: hop 256 이 판단 주기 4,000 샘플의 약수가 아니고
  프레임이 창 단위 reflect 패딩에 묶여 있다. 판단마다 64프레임 전부다.
