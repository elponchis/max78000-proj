# 외부 검증 세트 (AudioSet strong eval)

> **평가 전용. 학습에 쓰지 않는다. 오디오 재배포 금지.** 레포에는 영상 ID·구간·라벨 csv만 있고
> 오디오와 변환 결과(`windows_int8.npy`)는 레포 밖 `~/safesound-external/` 에만 있다.
>
> **학습 데이터 정의(v1)는 바꾸지 않았다.** 외부 세트는 두 정의로 따로 채점한다 — 평가는 주 세션이 한다.

작성 2026-10-04. 수집·가공은 보조 세션이 했고, 이 문서의 수치는 전부 `meta.csv` / `download_state.csv` 실측이다.

## 1. 무엇이고 왜 있는가

FSD50K(Freesound) 기반 테스트셋과 **출처가 다른** 검증 세트다. 같은 분포 안의 점수가 아니라
다른 수집 경로(유튜브 영상)에서 온 소리에 대한 점수를 얻는 용도다.

- 출처: AudioSet **평가(eval) 분할**의 strong label 판본 (`audioset_eval_strong.tsv`, 16,996구간, 시간 경계 포함).
- 누수: 키가 **유튜브 영상 ID**라 Freesound 원본 ID(FSD50K·US8K·ESC-50)와 무관하다. PANNs 학습은
  AudioSet balanced/unbalanced **train** 분할이고 여기서는 **eval 분할만** 쓴다.
  ⚠️ 한계: 유튜브 영상이 Freesound 업로드와 같은 소리를 우연히 공유하는 경우는 막을 방법이 없다.
- ⚠️ **분포 차이**: 유튜브 오디오는 AAC(m4a) 또는 Opus(webm)로 **손실 압축**되어 있다. FSD50K 는 무손실 wav 다.
  외부 세트 점수가 낮으면 그중 일부는 이 코덱 차이일 수 있다.

## 2. 정의 두 개 (둘 다 태그, 채점은 정의별)

mid 는 `mid_to_display_name.tsv` 기준. glass 의 `/m/07rn7sz` 는 PANNs 온톨로지에서 `Shatter` 다.

| 클래스 | (가) v1 정의 — `build_class_manifest.classify()` | (나) 실사용 정의 |
|---|---|---|
| siren | `Siren` | `Siren` + Civil defense siren · Ambulance · Police car · Fire engine |
| glass | `Shatter` | `Shatter` |
| dog_bark | `Bark` 또는 `Dog` | `Bark` 만 |
| scream | Screaming·Yell·Shout 중 하나, **v1 군중 목록**이 없을 것 | `Screaming` 만 (군중·음악·웃음 동반 구간 유지) |

- v1 군중 목록: Cheering, Applause, Clapping, Crowd, Chatter(strong 어휘에 없어 무효), Laughter, Giggle, Chuckle, Music, Singing.
- 어느 정의에서든 **대상 클래스가 2개 이상 겹치는 구간은 제외**했다.
- 배경: **두 정의 모두에서** 대상 mid 가 하나도 없는 구간.
- `def_gap`: 실사용 정의에서는 배경이고 v1 정의에서는 양성인 구간 (`Dog` 만 / `Yell`·`Shout` 만). 정의 차이의 영향을 재는 별도 그룹이다.
- 어느 정의에서도 양성이 아니면서 대상 라벨이 있는 구간(예: Shout + 군중)은 배경에도 def_gap 에도 넣지 않았다.

## 3. 선별 규칙 (결과를 보기 전에 고정, `tools/external_set/build_sampling.py`)

- 고정 시드 **78000**. 삭제된 영상은 **대체 없이** 유효 개수만 보고한다.
- 클래스당 **실사용 정의를 먼저 균일 무작위로 100개**(glass 는 84개 전부) 뽑고, v1 정의는 v1 에만 속하는 구간으로 채웠다.
  포함 관계(Bark ⊂ Bark|Dog, Siren ⊂ Siren+하위) 때문에 한 정의를 먼저 정해야 했다.
  - **siren 예외(사용자 승인)**: v1 siren 100개를 맞추려 v1 구간 41개를 추가해 **실사용 siren 이 141개**다.
    하위 종류 비율이 틀어지지 않게 **원래 뽑힌 100개에 `real_primary=1`** 을 표시했다.
- 배경 300, def_gap 100. 스피커 재생용 부분집합은 실사용 정의 기준 클래스당 30·배경 60 을 같은 시드로 미리 지정(`speaker=1`).
- 합집합 889구간을 한 번만 받았다. 이후 한 번도 규칙을 바꾸지 않았다.

## 4. 수집 결과

시도 889 → **유효 759**, 후처리 단계에서 읽기 실패 1 → **창 758개**.

| 그룹 | 목록 | 유효(다운로드) | 유효율 |
|---|---:|---:|---:|
| siren | 141 | 122 | 86.5% |
| glass | 84 | 74 | 88.1% |
| scream | 164 | 130 | 79.3% |
| dog_bark | 100 | 87 | 87.0% |
| background | 300 | 264 | 88.0% |
| def_gap | 100 | 82 | 82.0% |

**실패 사유 (130건)**

| 사유 | 건수 |
|---|---:|
| unavailable (삭제·사용 불가) | 57 |
| other — 전부 `ffmpeg exited with code 1` | 29 |
| private | 19 |
| login_required (`Please sign in`) | 13 |
| age_restricted | 8 |
| removed | 4 |

- `login_required` 13건은 연속 3회 미만이었고 중간에 성공이 섞여 있어 **영상 단위 게이트**로 판단했다(사용자 결정 2026-10-04).
  쿠키·로그인은 쓰지 않았다. 봇 차단·429 는 한 번도 나오지 않았다.
- `other` 29건의 원인(구간이 영상 길이를 넘음 등)은 확인하지 않았다 — 추측하지 않는다.
- 메모리 부족으로 다운로드가 한 번 종료됐고(상태 파일로 이어받음) 결과에 영향 없다.

## 5. 가공

- 16 kHz 모노 (`ffmpeg -ac 1 -ar 16000`), **창 길이 16384 샘플**(= `prepare_safesound.WIN`, 1.024초).
- 창 위치: 구간 안 **가장 긴 단일 이벤트**의 중심. 이벤트가 짧아도 중심에 맞춘다. 배경은 시드 고정 무작위 위치.
- 변환은 `scripts/prepare_safesound.to_int8` 를 **import** 한 고정 스케일 ×127 + 포화. **정규화 없음.**
  기존 코드는 수정하지 않았다.
- 오디오 포맷: m4a(140) 우선. 일부(**52개**)는 m4a 가 없어 webm/opus 로 내려갔고, 이 파일은 구간 다운로드가
  `[시작−10초, 시작+10초]` 20초로 나온다. 처음에는 m4a 와의 상호상관 **3건**으로 실제 구간이 9.944초 지점부터라고
  보고 앞 9.944초를 잘랐다. 이후 아래 "webm 오프셋 검증"으로 52개 전체를 점검했다 (`webm_origin=1`).
  나머지 706개는 처음부터 m4a 이고 `webm_origin=0`.

### webm 오프셋 검증 (2026-10-05, `tools/external_set/webm_audit.py`)

52개는 m4a(140)가 **없는 영상**이라 m4a 재시도(`fail:unavailable` 52/52)로는 교체할 수 없었다. 처음 3건의 근거는
m4a 가 있는 영상에서 얻은 것이라 이 집단에 그대로 쓸 수 없었으므로, **독립 기준**을 따로 만들었다:
전체 오디오를 받아 ffmpeg 로 정확히 `[시작, 시작+10초]` 를 로컬에서 자른 뒤(구간 다운로드 경로와 무관), 20초
webm 파일과 상호상관으로 오프셋을 쟀다. 쿠키 없음, 80MB 초과 건너뜀.

- 오프셋 **n=51**: 평균 **9.967초**, 표준편차 0.018초, 범위 9.899~9.972초. 보정값 9.944초와 **약 23ms 차이**였다.
  상관 피크 중앙값 0.74. 52번째(`1OFDyTzUj24_30000`, 배경, 피크 0.09)는 약한 상관이라 교체하지 않았고 오프셋은
  9.936초로 같은 값이었다.
- **50개를 정확한 로컬 절단 오디오로 교체**했다 (`webm_replaced_exact=1`, 원본 webm 은 `raw_webm_replaced/`).
  남은 `webm_trim=1` 은 2개(`1OFDyTzUj24_30000` 배경, 전체 다운로드가 실패한 1개). 이 2개는 9.944초 보정이 그대로다.
- **자동 점검**(이벤트 구간 RMS − 구간 밖 RMS, 52개 중 이벤트가 있고 구간 밖이 1초 이상인 31개): ok 23, SUSPECT 8,
  판정 불가 21(전부 배경 — 이벤트가 없다). SUSPECT 8개도 상호상관 오프셋은 9.90~9.97초로 **정렬이 틀린 것은 아니었다**.
  이 지표는 이벤트가 구간 밖보다 크지 않으면 낮게 나오므로, 저레벨 이벤트나 라벨 경계 부정확 때문으로 **보이지만 원인은
  확인하지 않았다**. 목록은 `~/safesound-external/webm_audit.csv` (`verdict` 열).
- 결론: 오프셋 보정 자체는 **독립 기준 51건으로 검증**됐다. 검수용 wav 에도 webm 출신 10개를 넣었다
  (`spotcheck/webm_*`, 이벤트 클래스 우선).
- 이벤트 시각은 AudioSet strong 구간 시작 기준 상대값이다.

⚠️ **채우기·필터를 하지 않았다.** v1 학습 데이터에 적용한 `--floor-zero` 하한, 디지털 0 채움은 **여기에 없다**.
외부 세트는 있는 그대로의 입력이다. 조용하거나 거의 빈 창이 있다 — 아래는 meta.csv 실측이다 (webm 교체 후 재집계).

| 그룹 | n | RMS 중앙 (dBFS) | RMS 최소 | int8 0 비율 중앙 | 0 비율 > 0.5 | 0 비율 > 0.95 | int8 전부 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| siren | 121 | −15.7 | −50.1 | 0.019 | 1 | 0 | 0 |
| glass | 74 | −19.0 | −45.7 | 0.053 | 3 | 0 | 0 |
| scream | 130 | −16.7 | −39.2 | 0.028 | 1 | 0 | 0 |
| dog_bark | 87 | −22.0 | −50.7 | 0.132 | 10 | 0 | 0 |
| background | 264 | −23.1 | −180.0 | 0.081 | 39 | 8 | 5 |
| def_gap | 82 | −19.3 | −57.2 | 0.057 | 7 | 1 | 0 |

- 전체 758개 중 0 비율 > 0.5 인 창은 **61개**(8.0%)다. 이벤트 4그룹만 보면 412개 중 **15개**(3.6%), 전부 RMS < −40 dBFS 다.
- 배경 8개가 0 비율 > 0.95 이고, 그중 **int8 에서 전부 0 인 창은 5개**다. 5개 중 **3개는 원본 자체가 디지털 무음**(−180 dBFS)이고
  2개는 −76·−60 dBFS 라 양자화로 사라졌다. (앞 버전 문서의 "완전 무음 1개" 는 RMS 최솟값만 보고 쓴 것이라 틀렸다.)
  배경에는 하한이 없어 **표시만 하고 유지**했다 (`bg_near_silent=1`).
- **`v1_floor_excluded`**: 이벤트 창(배경 제외)의 int8 0 비율 > 0.95, 즉 v1 `--floor-zero` 규칙에 걸리는 창. **1개**
  (`VZRQd3d6jjA_50000`, def_gap dog_bark, 0 비율 0.991, −57.2 dBFS). 지우지 않고 표시만 했다. 클래스 4그룹에는 해당이 0개다.
- 조용한 창은 `rms_dbfs`·`zero_frac_int8` 열로도 걸러 볼 수 있다.

## 6. 정의별 유효 개수 (채점 모집단)

양성 기준: v1 = 클래스 그룹 중 `v1_class==c`, 실사용 = `real_primary==1` 이고 `real_class==c`.
음성 기준: 두 정의 모두 `group==background`.

| 클래스 | v1 양성 | 실사용 양성 (real_primary) | 실사용 양성 (전체) | 합집합 | 스피커 |
|---|---:|---:|---:|---:|---:|
| siren | 84 | 87 | 121 | 121 | 25 |
| glass | 74 | 74 | 74 | 74 | 26 |
| scream | 81 | 74 | 74 | 130 | 24 |
| dog_bark | 87 | 87 | 87 | 87 | 26 |
| background | 264 | 264 | 264 | 264 | 53 |
| def_gap | v1 양성 82 (dog_bark 45, scream 37) / 실사용 배경 82 | | | 82 | — |

- 클래스 그룹 안에서 **정의가 갈리는 구간**: siren 37개(하위 종류만 있고 `Siren` 없음 → v1 에선 음성), scream 49개(v1 군중 목록 때문에 v1 음성)와 56개(Yell/Shout 만 → 실사용 음성). glass·dog_bark 는 갈리는 구간이 없다(dog 는 `Dog` 만 있는 구간이 def_gap 으로 빠졌다).
- 스피커 부분집합이 30/60 에 못 미치는 것은 삭제 영상 때문이다.
- siren 실사용 `real_primary` 87개의 하위 종류: `Siren` 34, Civil defense 12, Fire engine 12, Police 9, Civil defense+Siren 7, Ambulance 4, 그 외 복합 9. 이 비율이 외부 세트의 하위 종류 분포다.
- 이벤트 길이(가장 긴 단일 이벤트) 중앙값: siren 9.3초 / glass 1.4초 / scream 1.2초 / dog_bark 0.6초. 1초 이하 비율은 dog_bark 59/87, scream 57/130, glass 19/74, siren 4/121 — 창이 이벤트보다 길어 **배경 소리가 섞인다**(학습 데이터와 같은 조건).

- **def_gap 과 scream 그룹의 "Shout만" 구간은 같은 모집단의 무작위 추출이다(모집단 134, 뽑힘 92).** 선별 경로와 공변량 9개에서
  차이를 찾지 못했다. 두 집단의 발화율 차이는 우연일 가능성을 배제할 수 없으므로 합쳐서 한 집단으로 다룬다.
  (모집단 = strong eval 에서 Shout 만 있고 Screaming·Yell·v1 군중 목록·다른 대상 클래스 라벨이 없는 구간. scream 그룹 55 + def_gap 37.)
- 모집단 134개 중 뽑히지 않은 42개의 사유: **선별됐으나 다운로드 실패 20개**(other 5, private 5, login_required 4, age_restricted 4, unavailable 2),
  **추출 상한 22개**(scream v1 100·def_gap 100 도달로 목록 밖), 기타 0개. 선별은 112개(scream 63 + def_gap 49)였고 유효 92개(82%)다.

### 배경 구성 (유효 264)

라벨(구간 수, 상위 20; 한 구간이 여러 라벨을 가진다):

| 라벨 | 구간 수 | 라벨 | 구간 수 |
|---|---:|---|---:|
| Music | 93 | Conversation | 16 |
| Male speech, man speaking | 83 | Inside, small room | 16 |
| Mechanisms | 38 | Laughter | 15 |
| Breathing | 35 | Wind noise (microphone) | 15 |
| Female speech, woman speaking | 34 | Clicking | 14 |
| Male singing | 28 | Chirp, tweet | 13 |
| Tap | 26 | Water | 13 |
| Speech | 25 | Medium engine (mid frequency) | 12 |
| Tick | 22 | Silence | 12 |
| Generic impact sounds | 18 | Female singing | 17 |

하드 네거티브 유사 열(`hardneg_like`): 해당 없음 217, speech 25, laughter 15, crowd 9, crying 3, alarm 2, glass 비파손 1.
⚠️ 배경은 **무작위**라 FSD50K 테스트셋(49.9% 하드 네거티브)과 구성이 다르다. 오경보율을 비교할 때 이 차이를 함께 적는다.

## 7. 파일과 사용법

레포(`tools/external_set/`): `build_sampling.py`, `sampling_list.csv`(영상 ID·구간·라벨만), `download_clips.py`,
`process_clips.py`, `summarize_download.py`.

레포 밖 `~/safesound-external/` (WSL2):

```
external_v1/windows_int8.npy   (758, 16384) int8, meta.csv 행 순서와 같다
external_v1/meta.csv           sampling_list 열 + label_v1, label_real, win_start_*, rms_dbfs, peak, zero_frac_int8,
                               v1_floor_excluded, bg_near_silent, webm_trim, webm_origin, webm_replaced_exact
external_v1/fail.csv           후처리 실패 (0N0C0Wbe6AI_30000 read_error 1건)
raw/*.wav                      10초 원본 구간 (재배포 금지)
raw_webm_replaced/             교체 전 webm 원본 20초 (50개)
download_state.csv             구간별 ok/fail/사유
webm_audit.csv                 webm 52개 점검·오프셋 검증 결과
spotcheck/                     검수용 wav 70개: 그룹별 10개(siren/glass/scream/dog_bark/background) + webm_* 10 + defgap_* 10
```

라벨 정수: 0 siren, 1 glass, 2 scream, 3 dog_bark, 4 background. `label_v1` / `label_real` 은 그 정의가 양성이라 부르는 클래스, 아니면 4.

```python
import csv, numpy as np
d = "/home/max78000/safesound-external/external_v1"
X = np.load(f"{d}/windows_int8.npy")            # (758, 16384) → (128,128) reshape 해서 입력
M = list(csv.DictReader(open(f"{d}/meta.csv")))
idx = [i for i, r in enumerate(M) if r["group"] in ("siren","glass","scream","dog_bark","background")]

# (가) v1 정의
y_v1 = np.array([int(M[i]["label_v1"]) for i in idx])
# (나) 실사용 정의 — 양성은 real_primary=1 만, 배경은 그대로
y_re = np.array([int(M[i]["label_real"]) if M[i]["real_primary"] == "1" or M[i]["group"] == "background"
                 else -1 for i in idx])           # -1 은 채점 제외
# def_gap: label_v1=양성 클래스, label_real=4. 정의 차이의 영향을 따로 잰다.
```

채점 제안(강제 아님): 클래스별 재현율, 배경 오경보율, 정의별 macro-F1 을 **원본이 아닌 구간 수 기준으로** 보고하고,
구간 단위 신뢰구간을 쓴다. siren 은 `siren_sub` 별로 나눠 보면 하위 종류의 영향이 보인다.
스피커 재생 실험은 `speaker=1` 만 쓴다.

## 8. 재현

모두 WSL2, `~/safesound-external/venv` (yt-dlp + numpy + soundfile; ai8x venv 와 무관), 시스템 ffmpeg 4.4.2(apt).
`nice -n 19` 로 낮은 우선순위, 호출 사이 4~8초 대기. 정적 ffmpeg(pip) 는 DNS 조회에서 segfault 라 쓰지 않았다.

```bash
python3 tools/external_set/build_sampling.py --meta ~/safesound-external/meta --out tools/external_set/sampling_list.csv
python  tools/external_set/download_clips.py --list tools/external_set/sampling_list.csv --raw ~/safesound-external/raw
python  tools/external_set/process_clips.py  --list tools/external_set/sampling_list.csv --raw ~/safesound-external/raw \
        --out ~/safesound-external/external_v1 --spot ~/safesound-external/spotcheck
```

메타데이터 출처: `storage.googleapis.com/us_audioset/youtube_corpus/` (strong/audioset_eval_strong.tsv, strong/mid_to_display_name.tsv, v1/csv/class_labels_indices.csv).

## 9. 한계

1. **유튜브 손실 압축(AAC/Opus).** 오디오가 m4a(AAC) 또는 webm(Opus)로 손실 압축되어 있다. FSD50K·US8K·ESC-50 은 무손실 wav 다.
   외부 세트에서 점수가 낮아지면 그중 일부는 코덱 차이(고주파 손실·프리에코)일 수 있고, 이 효과는 분리해 재지 않았다.
   758개 전부 같은 ffmpeg 기본 리샘플러로 16kHz 로 바꿨고, librosa 등 다른 리샘플러와의 차이는 비교하지 않았다.
2. **webm 52개의 오프셋 보정.** m4a 가 없는 영상 52개는 구간 다운로드가 20초(`[시작−10초, 시작+10초]`)로 나왔다.
   처음에는 m4a 가 있는 영상 3건의 상호상관(9.944초)으로 앞을 잘랐고, 이후 **전체 오디오를 받아 로컬에서 정확히 자른
   독립 기준 51건**으로 검증했다(오프셋 평균 9.967초, 표준편차 0.018초, 보정과의 차이 약 23ms). 50개는 정확한 절단 오디오로
   교체했고 2개(`1OFDyTzUj24_30000` 배경, 전체 다운로드 실패 1개)는 9.944초 보정을 유지한다. 자동 점검의 SUSPECT 8개의
   원인(저레벨 이벤트·라벨 경계 부정확 추정)은 확인하지 않았다.
3. **ffmpeg 실패 29건의 원인 미확인.** 다운로드 단계에서 `ERROR: ffmpeg exited with code 1` 로 실패한 29건(전체 실패의 22%)은
   구간이 영상 길이를 넘었는지, 스트림 형식 문제인지 확인하지 않았다. 이 29건은 유효 개수에서 빠져 있고, 특정 클래스·길이에
   쏠렸는지는 보지 않았다.
4. **무음·저레벨 창.** 채우기·무음 필터를 적용하지 않았다. 전체 758개 중 int8 0 비율 > 0.5 인 창이 **61개(8.0%)**,
   이벤트 4그룹 412개 중 **15개(3.6%)**(전부 RMS < −40 dBFS), 배경 264개 중 0 비율 > 0.95 가 8개(int8 전부 0 이 5개, 원본
   디지털 무음 3개)다. v1 `--floor-zero`(0 비율 > 0.95) 해당은 이벤트 창 1개(def_gap)뿐이고 `v1_floor_excluded` 로 표시했다.
   표시만 했고 제외한 채 만든 별도 집계는 없다. 배경의 무음 8개는 유지했다.
5. 삭제 영상이 무작위가 아닐 수 있다 — 유효율은 85.4%(759/889)이고 사라진 영상이 특정 유형에 쏠렸는지 확인하지 않았다.
6. 라벨은 AudioSet strong 라벨 그대로다. **청취 검증을 하지 않았다** (v1 에서는 청취로 정답률을 쟀다). 검수용 wav 70개가 그 첫 단계다.
7. 창이 이벤트보다 길다 — 특히 dog_bark(중앙 0.6초).
8. 샘플 수가 작다(클래스당 74~121). 신뢰구간이 넓다.

## 10. train_v2 (AudioSet strong **train** 분할) — 학습에 아직 쓰지 않음

> **받아 두기만 했다. 학습에 아직 쓰지 않는다.** 평가 세트(`external_v1`)와 목적이 다르다. 오디오는 레포 밖
> `~/safesound-external/train_v2/raw/` (16kHz 모노 wav, 2,362개, 713MB), 레포에는 `tools/external_set/train_v2_list.csv`
> (ID·구간·라벨 2,800행)와 `build_train_v2.py` 만 있다 (커밋 `e95bcb0`). 재배포 금지.

### 10.1 선별 (결과 보기 전에 고정, 시드 78002, `build_train_v2.py`)

- 클래스(siren·glass·scream·dog_bark)마다 **합집합 400**, background **1,200**. 사용자 승인 2026-10-05.
- 정의 상수는 `build_sampling.py` 것을 import (v1 / 실사용). 두 정의로 모두 태그(`in_v1`, `in_real`, `pass`).
  클래스마다 **패스 1 = v1 후보에서 200개**, **패스 2 = 실사용 후보에서 합집합이 400이 될 때까지**. 한 정의를 먼저 몰아 뽑으면
  포함 관계(Siren ⊂ Siren+하위, Bark ⊂ Bark|Dog) 때문에 다른 정의가 쏠려서다.
- 어느 정의에서든 대상 클래스가 2개 이상 겹치는 구간 92개 제외. 배경은 두 정의 모두 대상 mid 가 없는 구간(후보 94,705)에서 무작위.
- **영상 ID 겹침: eval 분할 전체와 0개, `external_v1`(889구간)과 0개.** train 은 구간 ID 도 eval 과 0개 겹친다. 겹침이 생기면 제외하는
  코드가 들어 있고 이번에는 제외된 구간이 없다. 삭제 영상은 대체하지 않았다.

### 10.2 수집 결과 (2026-10-05 21:45 ~ 10-06 20:20, 2,800구간 전부 시도)

| 클래스 | 목록 | 유효 | 실패 | 유효율 |
|---|---:|---:|---:|---:|
| siren | 400 | 347 | 53 | 87% |
| glass | 400 | 350 | 50 | 88% |
| scream | 400 | 324 | 76 | 81% |
| dog_bark | 400 | 332 | 68 | 83% |
| background | 1,200 | 1,009 | 191 | 84% |
| 합계 | 2,800 | **2,362** | 438 | 84% |

실패 사유(438건): other 173, unavailable 144, private 55, login_required 40, removed 15, age_restricted 11. 봇 차단·429 는 0건.
`other` 173건은 `ffmpeg exited with code 1` 171건(원인 미확인, `external_v1` 과 같은 유형)과 **일시적 DNS 실패 2건**
(`Temporary failure in name resolution`)이다. 이 2건은 영상 문제가 아니라 네트워크 일시 오류인데 실패로 기록됐고 다시 받지 않았다.

### 10.3 받은 뒤 기준 정의별 유효 개수 (클래스 그룹)

| 클래스 | 합집합(유효) | v1 양성 | 실사용 양성 | 둘 다 | v1만 | 실사용만 | 패스1(v1) | 패스2(real) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| siren | 347 | 226 | 347 | 226 | 0 | 121 | 169 | 178 |
| glass | 350 | 350 | 350 | 350 | 0 | 0 | 173 | 177 |
| scream | 324 | 213 | 204 | 93 | 120 | 111 | 160 | 164 |
| dog_bark | 332 | 332 | 271 | 271 | 61 | 0 | 159 | 173 |

(v1 양성 = `v1_class==클래스`, 실사용 양성 = `real_class==클래스`. siren 은 `Siren` ⊂ 실사용, dog_bark 는 `Bark` ⊂ v1 이라 한쪽이 0 이다.)

### 10.4 배경의 하드 네거티브 — v1 학습 배경과 나란히

train_v2 배경은 **대상 라벨이 없는 구간을 무작위로** 뽑았다(하드 네거티브를 일부러 모으지 않았다). `hardneg_like` 열은 AudioSet 라벨로 판정한다.
v1 학습 배경은 `data/processed/safesound/train/background/index.csv` 의 `note` 열 집계다 (**창 단위**, 규칙 3-1 대로 FSD50K 에서 일부러 모은 것).

| 구분 | train_v2 배경 (유효 1,009구간) | v1 학습 배경 (9,577창) | v1 테스트 배경 (3,838창) |
|---|---:|---:|---:|
| 알람 (`Alarm`) | 7 (0.7%) | 2,527 (26.4%) | 1,211 (31.6%) |
| 아기 울음 (`Crying, sobbing` ↔ `baby_cry`) | 11 (1.1%) | 244 (2.5%) | 96 (2.5%) |
| 유리 비파손 (`Glass`·`Glass chink` ↔ `glass_dishes`·`glass_ring`) | 2 (0.2%) | 924 (9.6%) | 371 (9.7%) |
| 군중 환호 + 비명류 (`scream_crowd`) | 해당 없음(배경은 비명류 라벨이 없다) | 77 (0.8%) | 236 (6.1%) |
| **v1형 하드 네거티브 합계** | **20 (2.0%)** | **3,772 (39.4%)** | **1,914 (49.9%)** |
| 넓은 정의(+speech 23, laughter 68, crowd 57) | 156 (15.5%) | — (v1 은 이 라벨을 하드 네거티브로 보지 않는다) | — |

- **v1형 하드 네거티브 비율이 train_v2 는 2.0%, v1 학습 배경은 39.4%** 로 약 20배 차이가 난다. train_v2 를 v1 과 섞어 쓰면 배경의 하드 네거티브 농도가 크게 떨어진다.
- 단위(구간 대 창)와 판정 근거(AudioSet 라벨 대 FSD50K 라벨·규칙)가 달라 숫자는 **대략적인 비교**다. 같은 정의로 다시 재지 않았다.
- v1 테스트 배경은 CLAUDE.md 8장의 49.9% 와 같다.

### 10.5 webm 폴백 54개

- m4a(140)가 없는 영상은 구간 다운로드가 20초(`[시작−10초, 시작+10초]`)로 나온다. **앞 9.967초를 잘라 뒤 10초를 쓴다**
  (`ffmpeg -ss 9.967 -t 10`). 9.967초는 `external_v1` 독립 검증 51건의 평균(표준편차 0.018초)이다. train_v2 안에서 따로 검증하지 않았다.
- **표시는 있다**: `train_v2/download_state.csv` 의 `reason` 열이 `webm_trim` (status=ok 인 행 54개). 나중에 빼고 학습하려면 이 열로 거르면 된다.
  편의로 ID 목록을 `~/safesound-external/train_v2/webm_fallback_ids.txt` 에 만들어 두었다(레포 밖).
- 그룹: background 35, glass 9, siren 4, dog_bark 4, scream 2. 파일 길이는 8.0~10.0초(영상 자체가 짧은 경우 포함).

### 10.6 다운로드 중단 시각 (주 세션 학습과 대조용)

시스템 메모리 부족으로 Claude Code 가 백그라운드 다운로드를 종료했다. 아래는 **상태 파일의 마지막 기록 시각**이다.
실제 종료는 그 직후(구간 하나를 처리하는 수십 초 이내)이고, 종료 알림의 수신 시각은 따로 기록하지 않았다. 시각은 WSL 로컬(KST).

| 다운로드 | 종료 원인 | 마지막 기록 시각 | 다음 재개 |
|---|---|---|---|
| train_v2 실행 1 | 메모리 | 10-05 22:15:21 | 10-06 12:01:10 (사용자 재개 지시 대기) |
| train_v2 실행 2 | 메모리 | 10-06 12:31:54 | 12:32:15 |
| train_v2 실행 3 | 메모리 | 10-06 13:11:06 | 13:11:24 |
| train_v2 실행 4 | 메모리 | 10-06 15:15:39 | 15:28:51 (사용자 지시 대기) |
| train_v2 실행 5 | 메모리 | 10-06 15:58:29 | 15:59:15 |
| train_v2 실행 6 | 정상 종료 | 10-06 20:20:04 | — |
| external_v1 실행 2 | 메모리 | 10-04 00:31:23 | 10-04 21:23:50 (사용자 지시 대기) |
| external_v1 (참고) 실행 1 | 메모리 아님 — `Please sign in` 감지로 자체 중단 | 10-03 23:42:26 | 10-04 00:00:27 |

- 메모리 종료는 train_v2 5회, external_v1 1회다. 상태 파일 시각은 그 직전 구간의 처리가 끝난 시각이라, 종료 시점이 그보다 앞서지는 않는다.
- ⚠️ 이 시각은 **다운로드 쪽 기록**이다. 주 세션 학습이 실제로 영향을 받았는지는 학습 로그와 대조해야 한다 (확인하지 않았다).

### 10.7 추가 배치 `extra1_scream` (2026-10-07, scream 400구간)

- 선별 규칙은 10.1 과 같다(패스 1 v1 200 + 패스 2 실사용으로 합집합 400, 시드 78003, `build_train_v2.py --extra scream`).
  **영상 ID 는 기존 train_v2 2,800구간·`external_v1`·eval 전체와 겹치지 않는다 (0개).**
- 표시: `train_v2_list.csv` 와 `download_state.csv` 에 **`batch` 열**을 추가했다. 기존 2,800행은 `base`, 추가 400행은 `extra1_scream`.
  이 열이 없던 때의 상태 파일은 `download_state.before_extra1.csv` 로 백업했다(레포 밖).
- 결과: 400구간 중 **유효 327 (82%)**, 실패 73, 차단·429 0. 실패 사유: other 23(전부 `ffmpeg exited with code 1`), unavailable 20,
  login_required 14, private 13, removed 2, age_restricted 1. webm 폴백 4개(10.5 와 같은 방식으로 처리).
- 유효 327개의 정의별 개수: v1 양성 196, 실사용 양성 194 (둘 다 63, v1만 133, 실사용만 131).
  추가분에는 v1 에서는 scream 인데 실사용에서는 siren(하위 종류 라벨 때문)인 구간이 4개 들어 있다.
- 누적 scream: 목록 800, 유효 **651** (base 324 + extra1 327). train_v2 전체 유효는 2,689. 메모리 종료 없이 한 번에 끝났다.
- 학습에 아직 쓰지 않는다.
