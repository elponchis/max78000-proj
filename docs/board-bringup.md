# MAX78000FTHR 보드 브링업 — Windows 네이티브

보드를 처음 연결해서 우리 펌웨어를 올릴 수 있는 상태까지 만드는 절차다.
**전부 Windows 네이티브에서 한다.** WSL2 는 USB 를 보지 못하므로 플래싱·시리얼
작업을 WSL 에서 시도하지 말 것 (CLAUDE.md 6장).

각 단계 끝에 **"성공이면 이렇게 보인다"** 가 있다. 그 모습이 아니면 다음
단계로 넘어가지 말고 5절(흔한 실패)로 간다.

작성 시점 확인: 2026-09-29 기준 이 PC 에 **MSDK 가 설치돼 있지 않다**
(`C:\MaximSDK`, `C:\analog\MSDK`, `%USERPROFILE%\MaximSDK` 전부 부재,
`%MAXIM_PATH%` 비어 있음). 그래서 1절이 설치부터 시작한다.

> 표기: `PS>` 는 PowerShell, `$` 는 MSYS2(MinGW) 셸이다. 둘을 섞지 말 것 —
> `make` 는 MSYS2 에서만 돈다.

---

## 0. 준비물 확인

| 항목 | 확인 |
|---|---|
| MAX78000FTHR 보드 | 있음 |
| USB Micro-B 케이블 | **데이터 케이블**이어야 한다. 충전 전용이면 아무것도 안 잡힌다 |
| 디스크 여유 | MSDK 약 4~6GB |

보드에는 USB 커넥터가 **하나**다(보드 끝 Micro-B). 이 하나가 전원 + DAPLink
디버거 + 시리얼을 전부 겸한다.

**성공이면 이렇게 보인다** — 보드를 꽂으면 보드 위 LED 가 켜지고(공장 펌웨어가
들어 있다면 깜빡일 수도 있다), 잠시 뒤 탐색기에 `DAPLINK` 라는 이름의
이동식 디스크가 뜬다.

---

## 1′. ★ 실제로 쓰는 경로 — **WSL 빌드 + Windows 드래그 플래싱** (2026-10-01 확정)

아래 1·2절은 **Windows 네이티브 MSDK** 기준으로 쓴 것이다. 실제 환경을
확인해 보니 **WSL2 에 이미 전부 있었다** (이전 `max78000-auth` 작업의 유산):

| | 위치 | 버전 |
|---|---|---|
| MSDK | `~/msdk` | Examples/Libraries 전부 |
| arm 툴체인 | `/usr/bin/arm-none-eabi-gcc` | **10.3.1** |
| OpenOCD | `/usr/local/bin/openocd` | **0.12.0** (소스 빌드) |

→ **Windows 에 MSDK 를 또 깔지 않는다.** 역할을 이렇게 나눈다:

| 작업 | 환경 | 이유 |
|---|---|---|
| **빌드** | **WSL2** | 툴체인이 이미 있다 |
| **플래싱** | **Windows** — DAPLink 드라이브에 `.bin` 끌어다 놓기 | WSL2 는 USB 미지원. usbipd 는 **디버거가 필요할 때만** |
| **시리얼** | **Windows** — Tera Term / PuTTY, **115200** | 〃 |

⚠️ CLAUDE.md 6장은 "펌웨어 명령을 WSL 기준으로 안내하지 말 것" 이라고
적었다. **빌드만 예외**다 — 산출물을 Windows 로 넘기고, 플래싱·시리얼·
디버깅은 전부 Windows 절차다.

### 빌드 명령 (WSL2)

```bash
cd ~/max78000-proj
bash scripts/fw_build.sh Hello_World        # 또는 CNN/kws20_demo
bash scripts/fw_build.sh Hello_World clean
```

`scripts/fw_build.sh` 가 하는 일:
- `MAXIM_PATH=~/msdk`, `BOARD=FTHR_RevA` 로 `make`
- `.bin` 이 없으면 `.elf` 에서 `objcopy`
- **Flash 512KB / SRAM 128KB 대비 사용률**을 함께 찍는다 (CLAUDE.md 3장)
- 산출물을 **Windows 에서 보이는 폴더**로 복사:
  `C:/Users/1124j/Downloads/max78000-fw/<이름>.bin`
  (탐색기 주소창에는 `C:\Users\1124j\Downloads\max78000-fw` 로 넣는다)

### 빌드 실측 (2026-10-01, 보드 없이 확인)

| 예제 | text | data | bss | Flash | SRAM |
|---|---:|---:|---:|---:|---:|
| `Hello_World` | 36,432 | 2,596 | 1,532 | **39,028 B (7.4%)** | 4,128 B (3.1%) |
| `CNN/kws20_demo` | 389,820 | 2,604 | 35,636 | **392,424 B (74.8%)** | 38,240 B (29.2%) |

⚠️ `kws20_demo` 가 Flash 의 **74.8%** 를 쓴다. 가중치가 코드에 박혀 있어서다.
우리 펌웨어도 같은 자리를 쓰므로, **DEMO_BUILD 에 SD·GUI·부저를 다 넣으면
512KB 가 빠듯할 수 있다.** 측정 빌드(`MEASURE_BUILD`)를 따로 두는 이유가
하나 더 생겼다 (CLAUDE.md 6장).

---

## 1. MSDK 설치 확인과 설치

### 1.1 이미 설치돼 있는지 확인

PowerShell 을 열고:

```powershell
# 설치 경로 후보
"C:\MaximSDK","C:\analog\MSDK","$env:USERPROFILE\MaximSDK" |
    ForEach-Object { "{0,-40} {1}" -f $_, (Test-Path $_) }

# 환경변수
"MAXIM_PATH = $env:MAXIM_PATH"

# 핵심 도구 3종
"C:\MaximSDK\Tools\OpenOCD\openocd.exe",
"C:\MaximSDK\Tools\MSYS2\msys2_shell.cmd",
"C:\MaximSDK\Examples\MAX78000\CNN\kws20_demo\main.c" |
    ForEach-Object { "{0,-70} {1}" -f $_, (Test-Path $_) }
```

셋 다 `True` 면 1.3 으로 건너뛴다.

### 1.2 없으면 설치

Analog Devices 웹사이트에서 **MSDK (MaximSDK) Windows 인스톨러**를 받는다.
검색어: `Analog Devices MSDK installer` 또는 `MaximSDK Windows`.
(구 Maxim Integrated 페이지가 ADI 로 통합됐다. 파일명은 대략
`MaximMicrosSDK_win.exe` 형태다.)

설치 옵션 — **반드시 아래를 포함**시킨다:

| 컴포넌트 | 필요한 이유 |
|---|---|
| **MAX78000** 디바이스 지원 | 우리 타깃. 이게 없으면 예제도 링커스크립트도 없다 |
| **GNU Tools for ARM (arm-none-eabi)** | M4 컴파일러 |
| **GNU RISC-V Embedded GCC** | MAX78000 은 RISC-V 코어를 함께 갖는다. 도전 2순위(RISC-V 게이팅)에 필요 |
| **OpenOCD** | 플래싱·디버깅 |
| **MSYS2** | `make` 가 도는 셸 |
| **SDK Examples / Libraries** | `kws20_demo` 가 여기 들어 있다 |

설치 경로는 **기본값 `C:\MaximSDK` 를 그대로 쓴다.**
- 경로에 **공백이나 한글이 들어가면 안 된다.** MSYS2 의 make 가 깨진다.
  `C:\Users\1124j\...` 아래에 넣지 말 것.
- 설치 후 PowerShell 을 **새로 열어야** `MAXIM_PATH` 가 보인다.

설치 직후 확인:

```powershell
$env:MAXIM_PATH
& "C:\MaximSDK\Tools\OpenOCD\openocd.exe" --version
Get-ChildItem C:\MaximSDK\Tools\GNUTools      # 버전 폴더 이름 확인
```

> `GNUTools` 아래 버전 폴더 이름(`10.3` 등)은 인스톨러 버전마다 다르다.
> 아래 명령의 `10.3` 은 위에서 본 실제 이름으로 바꾼다.

```powershell
& "C:\MaximSDK\Tools\GNUTools\10.3\bin\arm-none-eabi-gcc.exe" --version
```

### 1.3 MSYS2 셸 열기

빌드는 여기서 한다.

```powershell
& "C:\MaximSDK\Tools\MSYS2\msys2_shell.cmd"
```

(시작 메뉴 바로가기 `Analog Devices > MSYS2` 로 열어도 된다. 버전에 따라
`msys2.bat` 인 경우도 있다.)

열린 셸에서:

```bash
$ echo $MAXIM_PATH
$ arm-none-eabi-gcc --version
$ make --version
```

**성공이면 이렇게 보인다**

```
$ echo $MAXIM_PATH
C:/MaximSDK

$ arm-none-eabi-gcc --version
arm-none-eabi-gcc (GNU Arm Embedded Toolchain 10.3-2021.10) 10.3.1 20210824
...

$ make --version
GNU Make 4.x
```

세 줄이 다 나오면 툴체인이 산다. `MAXIM_PATH` 가 비어 있으면 MSYS2 를 닫고
PowerShell 을 새로 연 뒤 다시 실행한다.

---

## 2. Hello_World — 빌드·플래싱·시리얼

보드가 살아 있는지, 툴체인이 맞는지, 시리얼이 나오는지를 한 번에 본다.
**우리 펌웨어보다 먼저 이걸 성공시킨다.** 여기서 막히면 뒤가 전부 막힌다.

### 2.1 빌드

MSYS2 셸에서:

```bash
$ cd /c/MaximSDK/Examples/MAX78000/Hello_World
$ make clean
$ make BOARD=FTHR_RevA -j4
```

⚠️ **`BOARD=FTHR_RevA` 를 반드시 준다.** 기본값은 EVKIT 이고, EVKIT 로 빌드한
바이너리는 FTHR 에서 콘솔 UART 핀이 달라 **아무것도 출력되지 않는다.**
빌드 자체는 성공하므로 이 실수는 조용히 지나간다 — 시리얼이 안 나오면
여기부터 의심한다.

매번 치기 싫으면 셸에서 `export BOARD=FTHR_RevA` 해 둔다.

**성공이면 이렇게 보인다**

```
arm-none-eabi-size --format=berkeley build/max78000.elf
   text    data     bss     dec     hex filename
  1xxxx     1xx    2xxx   1xxxx   3xxx build/max78000.elf
```

그리고 `build/` 에 파일이 생긴다:

```bash
$ ls -la build/max78000.*
build/max78000.elf
build/max78000.bin
build/max78000.map
```

`.elf` 와 `.bin` 이 둘 다 있어야 한다. 아래 두 플래싱 방법이 각각 하나씩 쓴다.

### 2.2 플래싱 방법 A — DAPLink 드래그앤드롭 (쉬움, 먼저 이걸로)

보드를 꽂으면 `DAPLINK` 이동식 디스크가 뜬다. 거기에 **`.bin` 을 복사**하면 된다.

PowerShell 로:

```powershell
# DAPLINK 드라이브 문자 찾기
Get-Volume |
    Where-Object { $_.FileSystemLabel -match 'DAPLINK|MAINTENANCE' } |
    Select-Object DriveLetter, FileSystemLabel, SizeRemaining

# 예: D: 였다면
Copy-Item C:\MaximSDK\Examples\MAX78000\Hello_World\build\max78000.bin D:\
```

탐색기에서 끌어다 놓아도 똑같다.

⚠️ **`.elf` 가 아니라 `.bin` 이다.** `.elf` 를 떨구면 `FAIL.TXT` 가 생긴다.

**성공이면 이렇게 보인다** — 복사 직후 DAPLINK 드라이브가 **한 번 사라졌다가
다시 마운트**되고, 보드가 리셋되면서 프로그램이 돈다. 드라이브 안에
`FAIL.TXT` 가 **없어야** 한다.

```powershell
Get-Content D:\FAIL.TXT -ErrorAction SilentlyContinue   # 없으면 아무것도 안 나온다 = 성공
Get-Content D:\DETAILS.TXT                              # DAPLink 버전·타깃 정보
```

`FAIL.TXT` 가 있으면 열어서 이유를 읽는다
(`The transfer timed out`, `The interface firmware FAILED to reset/halt the
target MCU` 등). 후자면 5.4 로 간다.

### 2.3 플래싱 방법 B — OpenOCD (우리가 실제로 쓸 방법)

디버깅과 반복 플래싱에는 이쪽이 낫다. PowerShell 에서:

```powershell
$OCD = "C:\MaximSDK\Tools\OpenOCD"
$ELF = "C:\MaximSDK\Examples\MAX78000\Hello_World\build\max78000.elf"

& "$OCD\openocd.exe" -s "$OCD\scripts" `
    -f interface/cmsis-dap.cfg `
    -f target/max78000.cfg `
    -c "program $ELF verify reset exit"
```

MSYS2 셸에서 하고 싶으면 예제 디렉터리에서 `make flash.openocd` 가 같은 일을
한다. 타깃 이름이 버전마다 다르므로 먼저 확인:

```bash
$ make -n flash.openocd | head -5
```

**성공이면 이렇게 보인다**

```
Open On-Chip Debugger 0.12.0+dev-...
Info : CMSIS-DAP: SWD supported
Info : SWCLK/TCK = 1 SWDIO/TMS = 1 ...
Info : [max78000.cpu] Cortex-M4 r0p1 processor detected
Info : [max78000.cpu] target has 6 breakpoints, 4 watchpoints
** Programming Started **
** Programming Finished **
** Verify Started **
** Verified OK **
** Resetting Target **
shutdown command invoked
```

`** Verified OK **` 가 핵심이다. 이게 없으면 플래시가 안 들어간 것이다.

### 2.4 시리얼 포트 찾기와 열기

PowerShell 에서 COM 포트 목록:

```powershell
Get-CimInstance Win32_PnPEntity |
    Where-Object { $_.Name -match '\(COM\d+\)' } |
    Select-Object Name | Format-Table -AutoSize
```

MAX78000FTHR 은 DAPLink 의 CDC 로 하나 잡힌다. 이름은 DAPLink 버전과 Windows
설정에 따라 `USB 직렬 장치 (COMn)` / `mbed Serial Port (COMn)` 중 하나다.

다른 USB 장치와 헷갈리면 **보드를 뽑았다 꽂으며 목록 차이**를 본다:

```powershell
# 보드를 뽑은 상태에서
$before = (Get-CimInstance Win32_SerialPort).DeviceID
# 꽂은 뒤
(Get-CimInstance Win32_SerialPort).DeviceID | Where-Object { $_ -notin $before }
```

**설정: 115200 baud, 8 데이터비트, 패리티 없음, 1 스톱비트, 흐름제어 없음
(115200 8N1).**

터미널은 아무거나 좋다. 파이썬이 이미 있으므로:

```powershell
py -m pip install pyserial                       # 한 번만
py -m serial.tools.miniterm COM7 115200          # COM7 은 위에서 찾은 값
```

(빠져나오기: `Ctrl+]`)

**성공이면 이렇게 보인다** — 터미널을 연 **뒤에** 보드 리셋 버튼을 누르면

```
Hello World!
count : 0
count : 1
count : 2
...
```

그리고 보드의 LED 가 1초 주기로 깜빡인다.

> 문구는 MSDK 버전마다 조금 다르다. `Hello_World/main.c` 의 `printf` 를 먼저
> 읽어 두면 무엇이 나와야 하는지 안다 — **기대 출력을 모르는 채로 터미널을
> 보면 "안 나온다" 와 "다르게 나온다" 를 구분할 수 없다.**

여기까지 되면 **툴체인·플래싱·시리얼 3종이 전부 검증된 것**이다.

---

## 3. kws20_demo — 온보드 마이크와 NPU 확인

우리 시스템(④ raw 파형 1D CNN)과 **입력 파이프라인이 같은** 공식 예제다.
16kHz 마이크 → int8 변환 → `(128,128)` reshape → NPU 추론이 전부 여기 들어
있다. 우리 펌웨어의 출발점이므로 반드시 돌려 본다.

### 3.1 빌드·플래싱

```bash
$ cd /c/MaximSDK/Examples/MAX78000/CNN/kws20_demo
$ make clean
$ make BOARD=FTHR_RevA -j4
```

플래싱은 2.2 / 2.3 과 동일 (`build/max78000.bin` 또는 `.elf`).

⚠️ 이 예제는 TFT 디스플레이 옵션이 딸려 있다. **MAX78000FTHR 에는 온보드
디스플레이가 없다** (CLAUDE.md 3장). 빌드가 TFT 헤더에서 깨지면 끈다:

```bash
$ grep -rn "TFT" Makefile project.mk 2>/dev/null
```

`FTHR_RevA` 보드 설정에서는 보통 자동으로 꺼지지만, 켜져 있으면
`project.mk` 의 해당 정의를 주석 처리한다.

### 3.2 기대 동작과 말할 단어

리셋 후 시리얼에 초기화 로그가 뜨고, 마이크가 상시 열린다.
**단어를 하나 또박또박 말하면** 그 단어와 신뢰도가 찍힌다.

**말할 단어 (KWS20 v3, 20개 + `_unknown_`)** — 출처는
`ai8x-training/datasets/kws20.py:81` 의 `dataset_dict['KWS_20']` 이다.
아래 순서가 곧 클래스 인덱스다.

| # | 단어 | # | 단어 |
|---:|---|---:|---|
| 0 | `up` | 10 | `one` |
| 1 | `down` | 11 | `two` |
| 2 | `left` | 12 | `three` |
| 3 | `right` | 13 | `four` |
| 4 | `stop` | 14 | `five` |
| 5 | `go` | 15 | `six` |
| 6 | `yes` | 16 | `seven` |
| 7 | `no` | 17 | `eight` |
| 8 | `on` | 18 | `nine` |
| 9 | `off` | 19 | `zero` |
| | | 20 | `_unknown_` |

영어 발음으로, 보드에서 **10~20cm** 거리에서, 단어 사이를 1초 이상 띄운다.
`stop`, `go`, `yes`, `no`, `zero` 가 가장 잘 잡힌다.

**성공이면 이렇게 보인다**

```
*** CNN Inference Test kws20_v3 ***

*** READY ***
Word starts from index 512, avg:xx > 350
Word ends at index 1xxxx, avg:xx < 350, ...
Time for CNN: xxx us

Classification results:
[ -xxxxx] -> Class 0 up: 0.0%
...
[  xxxxx] -> Class 4 stop: 9x.x%
...
-----------------------------------------
Detected word: stop (9x.x%)
```

`Time for CNN:` 이 **수백 µs** 수준이면 NPU 가 실제로 돌고 있는 것이다
(밀리초 단위면 CPU 경로를 의심한다).

아무 말도 안 했는데 계속 뭔가 검출되면 마이크 입력이 포화됐거나 검출 임계값이
환경 소음보다 낮은 것이다 — 조용한 방에서 다시 본다.

---

## 4. 랩어라운드 확인 (우리 설계에 직결)

### 4.1 왜 보는가

CLAUDE.md 7장 "실기기 스케일" 에 적어 둔 것의 **실기기 확인**이다.
MSDK 데모의 int16 → int8 변환에는 클램프가 없다:

```c
micBuff[micBufIndex] = (sample) * SAMPLE_SCALE_FACTOR / 256;   // int32 → int8_t
```

`HPF()` 는 int16 범위를 명시적으로 클리핑하지만 이 대입에는 포화가 없어서,
int32 결과가 127 을 넘으면 **하위 8bit 절단**으로 부호가 뒤집힌다.
예: int16 10,000 → `10000*4/256 = 156` → int8 대입 시 **−100**.

`SAMPLE_SCALE_FACTOR = 4` 에서 이 경계는 **마이크 풀스케일 대비 −12.1dB** 다.
드문 일이 아니다.

우리 전처리의 `to_int8` 은 `np.clip` 으로 **포화**시키므로, 펌웨어가
랩어라운드면 큰 소리에서 학습 분포와 추론 분포가 **정반대**가 된다.

→ 먼저 **실제로 일어나는지 눈으로 확인**하고, 그다음 우리 펌웨어는 포화로
구현한다 (TASKS.md Phase 5).

### 4.2 최소 수정안

원본을 되돌릴 수 있게 먼저 백업:

```powershell
Copy-Item C:\MaximSDK\Examples\MAX78000\CNN\kws20_demo\main.c `
          C:\MaximSDK\Examples\MAX78000\CNN\kws20_demo\main.c.orig
```

수정할 줄을 찾는다 (줄 번호는 버전마다 다르다):

```bash
$ grep -n "SAMPLE_SCALE_FACTOR" main.c
```

**(a) 파일 위쪽, 전역 변수 선언부에 추가**

```c
/* ── 랩어라운드 관찰용 (임시 계측 코드. 확인 끝나면 제거) ───────────── */
static volatile int32_t  g_dbg_max32 = -100000, g_dbg_min32 = 100000;
static volatile int8_t   g_dbg_max8  = -128,    g_dbg_min8  = 127;
static volatile uint32_t g_dbg_wrap  = 0,       g_dbg_n     = 0;
```

**(b) `MicReadChunk()` 안, `micBuff[...] = ... SAMPLE_SCALE_FACTOR / 256;`
줄을 아래로 교체**

```c
    /* 원래 동작을 **그대로 두고** 관찰만 한다 — 여기서 클램프를 넣으면
       무엇을 확인하려던 것인지 사라진다. */
    int32_t dbg_scaled = (sample) * SAMPLE_SCALE_FACTOR / 256;
    micBuff[micBufIndex] = dbg_scaled;              /* ← 원래 줄과 동일 */

    if (dbg_scaled > g_dbg_max32) g_dbg_max32 = dbg_scaled;
    if (dbg_scaled < g_dbg_min32) g_dbg_min32 = dbg_scaled;
    if (micBuff[micBufIndex] > g_dbg_max8) g_dbg_max8 = micBuff[micBufIndex];
    if (micBuff[micBufIndex] < g_dbg_min8) g_dbg_min8 = micBuff[micBufIndex];
    if (dbg_scaled > 127 || dbg_scaled < -128) g_dbg_wrap++;
    g_dbg_n++;
```

> 변수명 `micBufIndex` 와 `sample` 은 그 버전의 실제 이름에 맞춘다.
> `micBuff[micBufIndex++] = ...` 처럼 **증가가 붙어 있으면**, 증가를
> 마지막 줄로 따로 빼고 위 코드를 그 앞에 둔다. 안 그러면 관찰 코드가
> 엉뚱한 칸을 읽는다.

**(c) 상시 루프 안에 주기 출력 추가** (예: `main()` 의 `while (1)` 안,
`MicReadChunk` 호출 근처)

```c
        /* 약 1초마다 한 번 (16kHz 기준) */
        if (g_dbg_n >= 16000) {
            printf("[mic] n=%lu  int32[%ld..%ld]  int8[%d..%d]  wrap=%lu (%lu/1000)\n",
                   (unsigned long)g_dbg_n,
                   (long)g_dbg_min32, (long)g_dbg_max32,
                   (int)g_dbg_min8,  (int)g_dbg_max8,
                   (unsigned long)g_dbg_wrap,
                   (unsigned long)(1000UL * g_dbg_wrap / g_dbg_n));
            g_dbg_n = 0; g_dbg_wrap = 0;
            g_dbg_max32 = -100000; g_dbg_min32 = 100000;
            g_dbg_max8  = -128;    g_dbg_min8  = 127;
        }
```

> `%f` 대신 정수 천분율을 쓴 이유: newlib-nano 는 float `printf` 가 기본으로
> 꺼져 있어 `%f` 가 그냥 안 찍히거나 링크가 깨진다.

빌드·플래싱은 3.1 과 같다.

### 4.3 관찰 절차

1. **조용한 방에서 가만히 둔다** (약 10초). → 노이즈 플로어.
2. **보통 목소리로 `stop`** 을 30cm 거리에서 말한다.
3. **마이크에 5cm 까지 붙여 크게 박수**를 치거나 큰 소리를 낸다.

**성공이면 이렇게 보인다** — 세 구간이 확연히 다르다.

```
# 1) 무음
[mic] n=16000  int32[-3..2]      int8[-3..2]      wrap=0 (0/1000)

# 2) 보통 목소리
[mic] n=16000  int32[-58..61]    int8[-58..61]    wrap=0 (0/1000)

# 3) 박수 / 큰 소리   ← 여기가 관건
[mic] n=16000  int32[-412..503]  int8[-128..127]  wrap=37 (2/1000)
```

**판정**

- 3번에서 `wrap > 0` 이면 **랩어라운드가 실제로 일어난다.** 우리 펌웨어를
  포화로 구현해야 한다는 근거가 실측으로 확보된 것이다.
- int32 범위가 ±127 을 크게 벗어나는데 int8 이 `[-128..127]` 를 **꽉 채우고
  있다**는 것 자체가 증거다. **포화였다면** int8 최댓값이 +127 에 붙고
  최솟값은 −128 까지 안 내려간다(양의 피크가 음수로 넘어가지 않으므로).
- 1·2 번에서 `wrap > 0` 이 나오면 뭔가 잘못됐다 — `SAMPLE_SCALE_FACTOR` 값과
  `>> 14` 시프트를 확인한다.

**함께 기록할 것** (논문 7장 "실기기 스케일" 의 실측 근거가 된다)

| 항목 | 값 |
|---|---|
| `SAMPLE_SCALE_FACTOR` 실제 값 | (grep 결과) |
| 무음 시 int8 절댓값 최대 | ← 노이즈 플로어. `--floor-zero` 재검토 입력 |
| 보통 목소리 int8 범위 | |
| 큰 소리 시 wrap 비율 | |
| 측정 환경(방·거리·시각) | |

이 표를 채우면 CLAUDE.md 7장의 "⚠️ 마이크의 절대 감도와 무음 환경 노이즈
플로어는 보드 확보 후 실측" 항목이 닫힌다.

### 4.4 되돌리기

```powershell
Move-Item -Force C:\MaximSDK\Examples\MAX78000\CNN\kws20_demo\main.c.orig `
                 C:\MaximSDK\Examples\MAX78000\CNN\kws20_demo\main.c
```

계측 코드는 예제에 남기지 말고, 우리 펌웨어(`firmware/`)에
`MEASURE_BUILD` 플래그 아래로 옮긴다 (CLAUDE.md 6장).

---

## 5. 흔한 실패와 대처

### 5.1 보드가 아예 안 잡힌다 (DAPLINK 드라이브도, COM 포트도 없다)

순서대로 본다.

1. **케이블.** 충전 전용 Micro-B 케이블이면 전원만 들어오고 열거가 안 된다.
   다른 케이블로 바꾼다. **이게 제일 흔하다.**
2. **포트.** USB 허브를 거치지 말고 PC 본체에 직접 꽂는다.
3. **열거 자체를 확인**:
   ```powershell
   Get-PnpDevice -PresentOnly |
       Where-Object { $_.FriendlyName -match 'DAPLink|CMSIS-DAP|mbed|COM' } |
       Select-Object Status, Class, FriendlyName | Format-Table -AutoSize
   ```
   `Status` 가 `Error` 면 드라이버 문제, 아예 목록에 없으면 물리 연결 문제다.
4. 보드 위 **리셋 버튼**을 한 번 누른다.

**성공이면 이렇게 보인다**

```
Status Class     FriendlyName
------ -----     ------------
OK     HIDClass  CMSIS-DAP
OK     Ports     USB 직렬 장치 (COM7)
OK     DiskDrive MBED VFS USB Device
```

(정확한 이름은 DAPLink 버전마다 다르다. **세 종류 — HID/디버거, COM 포트,
디스크 — 가 다 보이는지**가 판단 기준이다.)

### 5.2 DAPLINK 대신 `MAINTENANCE` 드라이브가 뜬다

DAPLink 펌웨어 업데이트 모드다. 보드의 작은 버튼을 누른 채 꽂으면 이 모드로
들어간다. 뽑았다가 **버튼을 누르지 않고** 다시 꽂으면 정상 모드로 돌아온다.
돌아오지 않으면 DAPLink 펌웨어(`.hex`)를 이 드라이브에 떨궈 재기록한다.

### 5.3 OpenOCD 연결 실패

증상별로:

```
Error: unable to find a matching CMSIS-DAP device
```
→ 보드가 안 잡힌 것이다. 5.1 로.
→ **다른 프로그램이 디버거를 점유**하고 있을 수도 있다. VS Code 디버그
  세션이나 이전 openocd 프로세스를 죽인다:
```powershell
Get-Process openocd -ErrorAction SilentlyContinue | Stop-Process -Force
```

```
Error: [max78000.cpu] Debug regions are unpowered, an unexpected reset might have happened
Error: Target not examined, will not halt
```
→ 5.4 (저전력 락아웃) 이다.

```
Error: Can't find interface/cmsis-dap.cfg
```
→ `-s` 로 준 scripts 경로가 틀렸다. 실제 위치 확인:
```powershell
Get-ChildItem C:\MaximSDK\Tools\OpenOCD\scripts\interface\cmsis-dap.cfg
Get-ChildItem C:\MaximSDK\Tools\OpenOCD\scripts\target\max78000.cfg
```

### 5.4 저전력 예제를 올린 뒤 디버거가 안 붙는다 ★

**가장 겁나는 실패다.** 코어가 부팅 직후 deep sleep 으로 들어가면 디버거가
붙을 틈이 없다. 보드가 벽돌이 된 것처럼 보이지만 **거의 항상 복구된다.**
아래를 순서대로.

**(1) 리셋 홀드 후 연결** — 대부분 이걸로 된다.

보드의 리셋 버튼을 **누른 채로** 아래를 실행하고, `Info : CMSIS-DAP` 줄이
뜨는 순간 **손을 뗀다**. 코어가 sleep 코드에 도달하기 전에 halt 로 잡는 것이다.

```powershell
$OCD = "C:\MaximSDK\Tools\OpenOCD"
& "$OCD\openocd.exe" -s "$OCD\scripts" `
    -f interface/cmsis-dap.cfg -f target/max78000.cfg `
    -c "init" -c "reset halt" -c "halt"
```

붙으면 그 세션을 유지한 채 **다른** PowerShell 창에서 이어 간다 (아래 2번).

**(2) 플래시 전체 지우기**

```powershell
& "$OCD\openocd.exe" -s "$OCD\scripts" `
    -f interface/cmsis-dap.cfg -f target/max78000.cfg `
    -c "init" -c "reset halt" -c "max32xxx mass_erase 0" -c "exit"
```

지우고 나면 플래시가 비어 코어가 sleep 에 들어가지 못하므로 다음 연결은
쉽다. 바로 Hello_World 를 올린다 (2.3).

**(3) DAPLink 드래그앤드롭으로 덮어쓰기**

OpenOCD 가 계속 안 되면, 리셋을 누른 상태에서 꽂아 DAPLINK 드라이브를 띄우고
`Hello_World` 의 `.bin` 을 떨군다. DAPLink 쪽이 자체적으로 halt 를 시도하므로
성공할 때가 있다.

**(4) SWD 클럭을 낮춰 재시도**

```powershell
& "$OCD\openocd.exe" -s "$OCD\scripts" `
    -f interface/cmsis-dap.cfg -f target/max78000.cfg `
    -c "adapter speed 100" -c "init" -c "reset halt"
```

**성공이면 이렇게 보인다**

```
Info : [max78000.cpu] Cortex-M4 r0p1 processor detected
Info : [max78000.cpu] target has 6 breakpoints, 4 watchpoints
[max78000.cpu] halted due to debug-request, current mode: Thread
xPSR: 0x01000000 pc: 0x1000xxxx msp: 0x2001xxxx
```

`halted` 가 뜨면 살아난 것이다. 즉시 Hello_World 를 올려 상태를 되돌린다.

> **예방**: 저전력·deep sleep 코드를 올리기 전에 **깨어 있는 구간을 앞에 둔다.**
> 부팅 직후 3~5초 동안 LED 만 깜빡이며 아무것도 하지 않는 구간을 넣으면 그
> 사이에 디버거가 붙는다. 우리 펌웨어의 `MEASURE_BUILD` 에도 이 구간을
> 조건부로 넣어 둘 것 — RISC-V 게이팅(도전 2순위)을 건드릴 때 반드시 필요하다.

### 5.5 빌드는 되는데 시리얼에 아무것도 안 나온다

1. **`BOARD=FTHR_RevA` 를 빼먹었다.** 제일 흔하다 (2.1 의 경고).
   `make clean` 후 다시 빌드한다.
2. baud 가 115200 이 아니다.
3. 포트를 다른 프로그램이 잡고 있다 (닫지 않은 이전 miniterm 창).
4. 터미널을 연 **뒤에** 리셋을 눌러야 초기 출력을 본다.

### 5.6 `make` 가 경로에서 깨진다

```
make: *** No rule to make target ...
```
MSDK 를 공백·한글이 든 경로에 설치했을 때 난다. `C:\MaximSDK` 로 재설치한다.

---

## 6. 여기까지 끝나면

| 확인된 것 | 다음 |
|---|---|
| 툴체인·플래싱·시리얼 | 우리 펌웨어 골격 (TASKS.md Phase 5) |
| 온보드 마이크 → int8 → NPU 경로 | ④ 구성 이식의 출발점 |
| 랩어라운드 실측 | 포화 구현의 근거, 논문 7장 |
| 노이즈 플로어 실측 | `--floor-zero` 재검토 (CLAUDE.md 5장 규칙 6) |

**아직 하지 않은 것**: 전력 계측. 온보드 계측 회로가 없으므로 외부 장비가
필요하다 (CLAUDE.md 7장). 장비 확보 전까지 G3 의 에너지 축은 착수하지 않는다.
