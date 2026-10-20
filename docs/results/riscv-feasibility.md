# RISC-V 전처리 실현 가능성 검토 (2026-10-20)

범위: **문헌·SDK 조사와 추정만.** 보드 작업·원고 수정 없음. 모든 에너지 값은
**추정(실측 지연 × 공개 전력)** 이고 측정이 아니다. 지시 순서(전력값 → 코어 사양 →
이식 견적 → 손익 → 일정)대로 적는다.

## 0. 결론

1. **관문(공개 전력값) 통과.** 데이터시트에 RV32 가 켜진 조건의 전류 행이 둘 있다
   (CM4 SLEEP + RV32 ACTIVE 22.9 µA/MHz; LOW POWER 모드 18.3 µA/MHz + 0.64 mA).
   그러나 **선행연구 셋(Moss, Ulkar & Okman, Moosmann)에는 RV32 값이 없다** —
   Moss 는 "RISC-V is disabled" 로 명시. 따라서 세 전력 출처 방식 중
   **데이터시트 한 줄만 성립**하고 문헌 두 줄은 "추정 불가" 다.
2. **손익은 데이터시트 기준으로도 성립하지 않는다.** RV32 전력은 M4 활성의
   0.59~0.97 배이므로, 가(CPU log-mel) 보다 에너지가 적으려면 RV32 전처리가
   **2.2~4.2 ms 안**에 끝나야 한다. RV32 는 60 MHz·FPU 없음(rv32imc)·SIMD 없음
   이라 사이클 수가 M4F 의 0.59~1.11 배 이하여야 한다는 뜻인데, 정수 이식의
   낙관 가정으로도 사이클 2배·시간 7.6 ms 다 → 가 대비 **1.4~2.1 배** 에너지.
   소프트 float 그대로면 5~16 배.
3. **일정: 26일(월)까지 보드 실측·원고 반영은 권하지 않는다.** 툴체인 미설치,
   정수 FFT 자작, 듀얼코어 펌웨어, RV 디버거 부재가 겹쳐 5~8 일 작업이고,
   기대 결과가 "손해" 다. **졸업 후속**으로 넘기는 것을 권고한다.

## 1. 전력값 (가장 중요 — 관문)

### 1.1 MAX78000 데이터시트 (19-100868, Rev 1, 5/21 — 로컬 PDF 47쪽)

Electrical Characteristics 표에는 표 번호가 없다. **쪽·기호·조건**으로 인용한다.
전부 TYP(T_A = +25 °C), V_REGI = 3.0 V, V_COREA = V_COREB = 1.1 V, CNN 전 쿼드런트·
메모리 꺼짐, ECC 꺼짐. 2026-10-20 원문 재확인 (`pdftotext`).

| 쪽 | 기호 | 조건 (요약) | TYP | 뜻 |
|---:|---|---|---:|---|
| 9 | I_REGI_DACT | IPO, f_SYS 100 MHz, CM4 ACTIVE CoreMark, **RV32 ACTIVE while(1)** | 41.9 µA/MHz | 두 코어 다 켬 (연산 부하) |
| 9 | I_REGI_DACT | IPO, f_SYS 100 MHz, CM4 ACTIVE while(1), **RV32 ACTIVE while(1)** | 38 µA/MHz | 두 코어 다 켬 |
| 9 | I_REGI_DACT | IPO, f_SYS 100 MHz, CM4 ACTIVE while(1), **RV32 SLEEP** | **22.2 µA/MHz** | 현재 P_cpu 의 근거 (`energy-estimate.md` 3(가)) |
| 9 | I_REGI_DACT | **CM4 SLEEP, RV32 ACTIVE, ISO 에서 동작** | **22.9 µA/MHz** | ★ RV32 전처리의 동적 전류 |
| 9 | I_REGI_FACT | 고정분. IPO·ISO 켜짐, CM4·RV32 ACTIVE 0 MHz | 744 µA | ACTIVE 고정 |
| 10 | I_REGI_DSLP | SLEEP: CM4 SLEEP, RV32 SLEEP, 표준 DMA 2채널 | 22.5 µA/MHz | M4 잠든 상태 (동적) |
| 10 | I_REGI_FSLP | SLEEP 고정분 | 1.5 mA | M4 잠든 상태 (고정) |
| 10 | I_REGI_DLP | **LOW POWER: CM4 전원 차단, RV32 ACTIVE, f_SYS(MAX) 60 MHz** | **18.3 µA/MHz** | ★ RV32 단독 동작 (동적) |
| 10 | I_REGI_FLP | LOW POWER 고정분. RV32 ACTIVE 0 MHz | 0.64 mA | RV32 단독 (고정) |
| 11 | t_SLP_ON | SLEEP 복귀 | 0.67 µs | M4 기상 비용 — 무시 가능 |
| 11 | t_LP_ON | LOW POWER 복귀 | 9.5 µs | 〃 |

같은 표 13~14쪽의 CNN 행(I_REGI_CNNIA 790.4 µA, I_COREA_CNN_MNISTB 29 mA)은
`energy-estimate.md` 3(가) 와 같다.

**해석상 가정 (명시).**
- "µA/MHz" 는 f_SYS_CLK 당 값이다. RV32 행은 ISO(60 MHz) 조건이라 **× 60 MHz** 로
  환산했다. RV32 는 ISO 60 MHz 를 넘지 못한다 (1쪽 "up to 60MHz").
- "CM4 SLEEP + RV32 ACTIVE" 조합의 **고정분 행은 없다.** ACTIVE 고정 744 µA 와
  SLEEP 고정 1.5 mA 둘을 다 넣어 **범위**로 보고한다 (SLEEP 고정분이 더 큰 것은
  DMA 2채널 조건 때문으로 보이나 원문이 분리해 주지 않는다).
- LOW POWER 모드는 고정분 행(0.64 mA)이 있어 가정이 적다. 단 **CM4 전원 차단**이라
  판단 결과 처리를 RV32 가 다 해야 한다 (출력·로그·UART).

RV32 가 전처리를 맡을 때의 모드 전력 (레일 전력의 합, SIMO 손실 미포함 — 기존 식과 같은 방식):

| 경우 | 계산 | P_rv | M4 활성 8.89 mW 대비 |
|---|---|---:|---:|
| A. CM4 SLEEP + RV32 ACTIVE, 고정 744 µA | (22.9 × 60 + 744) µA × 3.0 V | **6.35 mW** | 0.71 |
| B. CM4 SLEEP + RV32 ACTIVE, 고정 1.5 mA | (22.9 × 60 + 1500) µA × 3.0 V | **8.62 mW** | 0.97 |
| C. LOW POWER (CM4 꺼짐) + RV32 ACTIVE | (18.3 × 60 + 640) µA × 3.0 V | **5.21 mW** | 0.59 |

### 1.2 User Guide UG7456

**원문을 받지 못했다.** analog.com 과 CDN(`www-cce.cldnet.analog.com`) 모두
WSL 의 `curl` 과 페이지 가져오기가 시간 초과·연결 거부로 끝났다 (2026-10-20, 3회).
검색 요약으로 확인된 것은 UG7456 Rev 1 (3/2024), 420쪽, 레지스터 중심 문서라는
것뿐이다. **전류값은 UG 가 아니라 데이터시트에 있으므로 관문에는 영향이 없다.**
RV32 ISA·메모리 접근은 아래 2절처럼 데이터시트 37쪽과 MSDK 소스로 대신 확인했다.

### 1.3 ADI 응용 노트 "Developing Power-optimized Applications on the MAX78000" (2021-06)

원문 역시 받지 못했다 (같은 호스트, 3회 시간 초과). 검색 스니펫에 드러난 내용:
"In this test, RISC-V and peripherals are off and Arm is tested in different
operation modes. ACTIVE and SLEEP power were measured using ISO oscillator at
60MHz" / "LPM: RISC-V is expected to collect and move data from enabled
peripherals while Arm is in deepsleep" / "CNN can only wake up RISC-V but not Arm
in LPM". 즉 **이 노트의 측정은 RISC-V 를 꺼 둔 조건**으로 보이며, RV32 활성
전력의 2차 출처가 되지 못한다. ⚠️ 우리가 가진 PDF 는 MAX78002 용이라 쓰지 않는다
(`energy-estimate.md` 3(가) 각주).

### 1.4 선행연구

| 출처 | RV32 측정 | 근거 |
|---|---|---|
| Moss et al., SenSys '22 AIChallengeIoT (`alessandro-montanari.github.io/papers/AIChallengeIoT2022.pdf`, 2026-10-20 원문 확인) | **없음** | 3절: "only the ARM core and the CNN accelerator are active ... **while the RISC-V is disabled**". 2절에 "Other operations can be executed on the ARM or RISC-V cores but will incur latency penalties" 한 줄만 |
| Ulkar & Okman, arXiv:2111.04988 | 없음 | `energy-estimate.md` 3(나) — M4F 만 |
| Moosmann et al., TinyissimoYOLO, arXiv:2306.00001 | 없음 | 〃 |
| GitHub `BoonR3alKua/EdgeAI-Benchmarking-and-Application---Thesis` (학위논문 저장소, 저자·기관 미표기) | 전력값 없음 ("Energy: measured via power profiler (not in logs)"). RISC-V 로 CNN 구동 지연만 (AlexNet 13.39 ms 등, 다른 모델) | 출처로 쓸 수 없다 |

→ **문헌 출처 (나)·(다) 에는 RV32 전력이 없다.** 4절의 손익은 데이터시트 한 줄로만
계산하고, 문헌 줄은 "추정 불가" 로 둔다. 비율 환산(데이터시트 P_rv/P_m4 × 문헌 P_cpu)은
근거 없는 가정이라 본문에 쓰지 않는다.

## 2. 코어 사양

| 항목 | 값 | 근거 |
|---|---|---|
| ISA | **rv32imc** (I 기본 + M 정수 곱셈·나눗셈 + C 압축). **F 없음 → float 는 소프트웨어 에뮬레이션** | MSDK `Libraries/CMSIS/Device/Maxim/GCC/gcc_riscv.mk`: `MARCH ?= rv32imc` (`riscv-none-elf` 는 `rv32imc_zicsr_zifencei`), `MABI ?= ilp32`, `-mfloat-abi` 없음. ai8x-synthesis `assets/embedded-riscv-ai85/templateMakefile.RISCV` 도 `#MARCH=rv32imc` |
| 최대 클럭 | **60 MHz** (ISO) | 데이터시트 1쪽 "32-Bit RISC-V Coprocessor up to 60MHz", 10쪽 I_REGI_DLP 조건 |
| 메모리 | RV32 는 **SRAM2(48 KB, 데이터)·SRAM3(16 KB, 코드 또는 I-캐시)** 만 접근. 플래시에서 CM4 와 동시 실행 가능. SRAM0/1 은 CM4 전용 | 데이터시트 37쪽 ACTIVE/LPM 설명; MSDK `common.ld` (SRAM2 0x20010000 48 kB "RV Data", SRAM3 0x2001C000 16 kB "RV Code") |
| 캐시 | SRAM3 를 RV32 명령 캐시로 구성 가능. 데이터 캐시 없음 | 데이터시트 37쪽; `ai8xize --riscv-cache` (기본 true) |
| CMSIS-DSP | **Arm 전용 프리빌트만** (`libarm_cortexM4lf_math.a` 등). RV32 용 빌드 없음 | MSDK `Libraries/CMSIS/5.9.0/DSP/1.16.2/Lib/`, `Libraries/CMSIS/Lib/GCC/` |
| 기동 | CM4 가 `MXC_SYS_RISCVRun()`: CPU1 클럭 → `urvbootaddr` 설정 → RST1.CPU1. 공유 메모리는 `.shared`/mailbox (SRAM2) + 세마포어 인터럽트 | MSDK `sys_ai85.c` 525행, `max78000_riscv.ld` |
| CNN 제어 | **가능.** `ai8xize.py --riscv` (`--riscv-flash`·`--riscv-cache` 기본 true, `--riscv-exclusive`) 가 `main_riscv.c` 를 생성. 완료 인터럽트를 RV32 가 받는다 (`while (cnn_time == 0) asm volatile("wfi")`). Fast FIFO 는 RV32 전용 | ai8x-synthesis `izer/commandline.py` 209~225행, README 755~761·3057행; MSDK `Examples/MAX78000/CNN/mnist-riscv/main_riscv.c` 107~118행 |
| MSDK 예제 | `CNN/mnist-riscv`, `CNN/kws20_demo-riscv` (RV32: I2S 마이크 + CNN, CM4: `__WFI()` 로 대기 후 TFT 표시), `RV_ARM_Loader` (CM4 가 RV 코드 적재·기동). 빌드 산출물은 `max78000-combined.elf` | MSDK `Examples/MAX78000/` 디렉터리 (로컬 `~/msdk`) |
| 지연 측정 | RV32 에는 SysTick 이 없다. 예제는 `MXC_TMR` 로 µs 를 잰다 | `mnist-riscv/main_riscv.c` 35·118행 |
| 디버깅 | RV32 JTAG 는 I2S 와 핀 공유. FTHR 의 DAPLink 는 SWD(CM4) 만 → **RV32 쪽은 printf 디버깅만** | `kws20_demo-riscv/README.md`, MSDK USERGUIDE 976~1016행 (별도 JTAG 디버거 전제) |
| 툴체인 | **WSL 에 RISC-V GCC 미설치** (`riscv-none-elf-gcc` 없음, `~/msdk/Tools` 에 xPack 없음) | 2026-10-20 확인 |
| 설계 의도 | 데이터시트 37쪽 LPM: "suitable for running the RV32 processor to **collect and move data** from enabled peripherals" — 신호처리가 아니라 데이터 이동용 코어 | 데이터시트 32쪽 "ultra low power consumption signal processing" 과 병존 |

## 3. 이식 범위 견적 — 가(CPU log-mel 62.5 ms) 전처리를 RV32 로

### 3.1 지금 M4 에서 하는 일 (`firmware/common/melfeat.c`, 변형 V1)

판단 1회 = 새 프레임 4개(`melfeat_inc_push`; u1000 은 프레임 간격 1000 샘플이라
4,000 샘플마다 4프레임) + 링 버퍼 풀기. **전부 float32** 다.

| 단계 | 구현 | M4 사이클 (판단 1회, `synthesis-check.md` 14.1절) |
|---|---|---:|
| 프레임 (창 곱) | float `x[i] * mel_win[n]` | 30,266 |
| rfft 512 | **CMSIS-DSP `arm_rfft_fast_f32`** (Arm 전용 프리빌트) | 148,216 (프레임당 37,054) |
| 파워 + 멜 합 | float, 가중치 499개 (`mel_fb_w`) | 34,846 |
| 로그 | 지수 + 256칸 LUT — **float 비트를 읽는** 구현 (`v.u >> 23`) | 10,262 |
| 합계 (push + get) | | **226,409 = 2.264 ms @100 MHz** |

### 3.2 RV32 에서의 선택지

**(a) 그대로 컴파일 (소프트 float).** `arm_rfft_fast_f32` 는 쓸 수 없고 PC 대조용
`rfft512` (plain C, `-DMELFEAT_HOST` 경로)로 바꾸면 컴파일은 된다. 그러나 rv32imc 는
float 곱·합이 전부 `__mulsf3`/`__addsf3` 라이브러리 호출이다 — 단순 정수 코어에서
연산당 수십 사이클. FPU 1~3 사이클 대비 **×10~20** 을 가정하면 사이클 2.3~4.5 M,
60 MHz 에서 **38~75 ms**. 판단 주기 250 ms 안에는 들지만 4절처럼 에너지는 크게 손해.

**(b) 정수 이식.** rfft 를 q31 로, 멜 합을 정수 누산으로, 로그를 정수 로그로.
- rfft: CMSIS-DSP 의 `arm_rfft_q31` 은 Arm 전용이라 **자작**해야 한다 (radix-2/4 q31,
  트위들 표). M4 실측에서 q31(V3) 은 float(V1) 의 **1.35 배** 사이클이었고 KAT
  1 LSB 로 통과, q15(V2) 는 KAT 실패(15 LSB) 라 쓸 수 없다 (`synthesis-check.md`
  356~359행). RV32IM 은 SIMD(SMLAD 등)·듀얼 MAC 이 없고 플래시 실행이므로
  **M4 q31 대비 사이클 ×1.5~3** 을 가정한다.
- 로그: ②′ 의 정수 `log2_q8_to_int8` (`firmware/common/log2_q8.c`) 이 있으나 입력
  스케일(int32 합)이 달라 상수를 다시 맞춰야 한다. KAT 1 LSB 조건은 같다.
- 추정 지연: 2.264 × 1.35 × (1.5~3) × (100/60) = **7.6~15.3 ms**.

**메모리.** 링 4,096 B + 이력 ~4.3 KB + FFT 버퍼 ~6 KB + 정수 테이블 ~4.5 KB
→ SRAM2 48 KB 에 넉넉히 든다. 코드는 플래시(또는 SRAM3 16 KB). 마이크 I2S 는
RV32 가 접근 가능(LPM 에서도).

**근거 수준.** ×1.5~3, ×10~20 은 **가정**이다. 보드 실측 전에는 TBD 이고,
논문·코드 주석에 확정값처럼 쓰지 않는다.

## 4. 손익 추정 — "RV32 전처리 + NPU 추론, M4 수면" 대 가

식 (기존 `make_paper_assets.py` 와 같은 형태):
`E = t_pre · P_rv + (t_load + t_out) · (P_rv + P_cnn대기) + t_infer · (P_cnn + P_rv)`.
가 의 실측 지연 pre 2.264 / load 0.134 / infer 0.531 / out 0.005 ms. RV32 는
60 MHz 라 적재·출력도 × 100/60 (사이클 같다고 가정). 추론 중 RV32 는 바쁜 대기
(`wfi` 를 쓰면 더 내려가지만 기존 식의 "바쁜 대기" 가정과 맞춘다).
**가 = 43.3 µJ** (데이터시트, `numbers.json` `energy.가 u1000`).

| 경우 (1.1절) | P_rv | 적재·출력·추론 몫 | **손익분기 t_pre*** | = M4 2.264 ms 의 | 필요한 사이클 비 (60 MHz) |
|---|---:|---:|---:|---:|---:|
| A. SLEEP, 고정 744 µA | 6.35 mW | 22.3 µJ | **3.31 ms** | 1.46× | **0.88×** |
| B. SLEEP, 고정 1.5 mA | 8.62 mW | 24.1 µJ | **2.24 ms** | 0.99× | **0.59×** |
| C. LOW POWER | 5.21 mW | 21.5 µJ | **4.20 ms** | 1.85× | **1.11×** |

즉 RV32 가 **M4F(FPU·SIMD·CMSIS-DSP, 100 MHz) 와 같거나 더 적은 사이클**로
전처리를 끝내야 겨우 같아진다. 3절의 추정으로 채우면:

| RV32 전처리 가정 | t_pre | E (A) | E (B) | E (C) | 가 대비 |
|---|---:|---:|---:|---:|---:|
| 정수 이식 낙관 (사이클 ×1.35×1.5) | 7.6 ms | 70.9 µJ | 89.9 | 61.3 | **1.4~2.1×** |
| 정수 이식 비관 (×1.35×3) | 15.3 ms | 119.4 | 155.8 | 101.1 | 2.3~3.6× |
| 소프트 float (×10) | 37.7 ms | 262 | 349 | 218 | 5.0~8.1× |
| 소프트 float (×20) | 75.5 ms | 502 | 675 | 415 | 9.6~15.6× |

계산 스크립트는 보관하지 않았다 — 식과 상수가 위에 전부 있어 재현된다.

**판정.** 데이터시트 출처에서 **어느 경우도 가 보다 적어지지 않는다.** 전력이
0.59~0.97 배밖에 안 내려가는데 시간은 클럭만으로 1.67 배 늘고 FPU 부재가 그 위에
얹힌다 — "race to sleep" 과 같은 논리로, 느린 저전력 코어는 에너지에서 지지 않는다
(CLAUDE.md 2장 프레이밍과 일치: 시간 ↔ 전력 트레이드오프가 아니다).
문헌 출처 (나)·(다) 는 RV32 값이 없어 **추정 불가**.

**이 검토가 다루지 않은 것.** RV32 를 **상시 감시 게이트**(CLAUDE.md 도전 2순위:
M4 deep sleep, RV32 가 I2S DMA + 에너지 임계값만)로 쓰는 안은 **유휴 전력**을 줄이는
것이라 축이 다르다. 우리의 판단당 에너지 식은 슬립을 포함하지 않으므로 그 이득은 이
문서의 손익에 나타나지 않는다. 그 안은 별도 과제다.

## 5. 일정 권고

오늘 10-20(화), 목표 10-26(월) → **6일**, 투고 마감 10-28(수).

| 작업 | 견적 | 위험 |
|---|---:|---|
| xPack RISC-V GCC 설치 + 듀얼코어 빌드 경로 (`fw_build.sh` 확장, combined elf 플래싱) | 0.5~1 일 | OpenOCD 로 combined elf 플래싱 경로 미검증 |
| q31 rfft 자작 + 정수 멜 + 정수 로그 재스케일, PC KAT (1 LSB) | 2~3 일 | KAT 실패 시 재설계 (q15 전례) |
| `ai8xize --riscv` 로 가 u1000 네트워크 재생성, RV32 측정 펌웨어 (TMR 기반), CM4 수면·기상 | 1~2 일 | RV32 디버거 없음 → printf 만; mailbox 동기 |
| 보드 실측 (POR 여러 번) + 결과 문서 | 0.5~1 일 | — |
| 원고 반영 (3쪽 유지, 그림 1 점 추가, 수치 훑기) | 0.5 일 | 3쪽 초과 위험 |
| **합계** | **5~8 일** | 기대 결과는 "가 대비 1.4 배 이상 손해" |

**권고 하나: 졸업 후속으로 넘긴다.** 이유는 (1) 마감 전 6일을 전부 쓰고도 끝나지
않을 수 있고, (2) 실측 전 추정으로도 성립하지 않으며 (손익분기가 사이클 0.6~1.1 배라
결과가 뒤집힐 여지가 거의 없다), (3) 그 결과를 넣으려면 3쪽 원고에서 다른 것을
빼야 한다. 원고에는 손대지 않았다 — 향후 과제에 한 문장 넣을지는 사용자 결정이다.

## 6. 출처

- MAX78000 Data Sheet, 19-100868 Rev 1 (5/21), `/mnt/c/Users/1124j/Downloads/MAX78000_data_sheet.pdf` — 9~11쪽 Electrical Characteristics, 32쪽 Detailed Description, 37~38쪽 Power Management
- MSDK (로컬 `~/msdk`): `Libraries/CMSIS/Device/Maxim/GCC/gcc_riscv.mk`, `Libraries/CMSIS/Device/Maxim/MAX78000/Source/GCC/{common.ld,max78000_riscv.ld}`, `Libraries/PeriphDrivers/Source/SYS/sys_ai85.c`, `Examples/MAX78000/CNN/{mnist-riscv,kws20_demo-riscv}`, `Examples/MAX78000/RV_ARM_Loader`, `USERGUIDE.md`
- ai8x-synthesis (로컬 `~/ai8x-synthesis`): `izer/commandline.py` RISC-V 그룹, `README.md` 755~761·2169~2172·3043~3057행, `assets/embedded-riscv-ai85/templateMakefile.RISCV`
- A. Moss et al., "Ultra-low Power DNN Accelerators for IoT: Resource Characterization of the MAX78000", SenSys '22 AIChallengeIoT, doi:10.1145/3560905.3568300 — 저자 공개 PDF 원문 확인
- UG7456 (MAX78000 User Guide, Rev 1 3/2024), ADI 응용 노트 "Developing Power-optimized Applications on the MAX78000" (2021-06) — **원문 미취득** (1.2·1.3절)
- 본 레포: `docs/results/energy-estimate.md` 3(가)·(나), `docs/results/synthesis-check.md` 14.1절·356~359행, `firmware/common/melfeat.c`, `docs/paper/numbers.json`
