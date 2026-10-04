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
- 오디오 포맷: m4a(140) 우선. 일부(52개)는 webm/opus 로 내려갔고 이 파일은 구간이 `[시작−10초, 시작+10초]` 20초다.
  m4a 와의 상호상관으로 실제 구간이 **9.944초 지점부터**임을 3/3 확인해 앞 9.944초를 잘랐다 (`webm_trim=1`).
  나머지 706개는 m4a 이고 `webm_trim=0`.
- 이벤트 시각은 AudioSet strong 구간 시작 기준 상대값이다.

⚠️ **채우기·필터를 하지 않았다.** v1 학습 데이터에 적용한 `--floor-zero` 하한, 디지털 0 채움은 **여기에 없다**.
외부 세트는 있는 그대로의 입력이다. 조용하거나 거의 빈 창이 있다 — 아래는 meta.csv 실측이다.

| 그룹 | n | RMS 중앙 (dBFS) | RMS 최소 | int8 0 비율 중앙 | 0 비율 > 0.5 인 창 |
|---|---:|---:|---:|---:|---:|
| siren | 121 | −15.7 | −50.1 | 0.019 | 1 |
| glass | 74 | −19.0 | −45.7 | 0.052 | 3 |
| scream | 130 | −16.7 | −39.2 | 0.028 | 1 |
| dog_bark | 87 | −22.0 | −50.7 | 0.133 | 10 |
| background | 264 | −23.1 | −180.0 | 0.081 | 39 |
| def_gap | 82 | −19.3 | −57.2 | 0.057 | 7 |

배경 중 1개는 **완전한 디지털 무음**(−180 dBFS, 0 비율 1.0)이다. 조용한 창은 `rms_dbfs`·`zero_frac_int8` 열로 걸러 볼 수 있다.

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
external_v1/meta.csv           sampling_list 열 + label_v1, label_real, win_start_*, rms_dbfs, peak, zero_frac_int8, webm_trim
external_v1/fail.csv           후처리 실패 (0N0C0Wbe6AI_30000 read_error 1건)
raw/*.wav                      10초 원본 구간 (재배포 금지)
download_state.csv             구간별 ok/fail/사유
spotcheck/                     검수용 wav (클래스당 10개)
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

1. 유튜브 손실 압축 오디오 — FSD50K 와 코덱 분포가 다르다(§1).
2. 유효율 85% 안팎이고 삭제 영상이 무작위가 아닐 수 있다. 공개 시점 이후 사라진 영상일수록 특정 유형으로 쏠렸는지 확인하지 않았다.
3. 라벨은 AudioSet strong 라벨 그대로다. **청취 검증을 하지 않았다** (v1 에서는 청취로 정답률을 쟀다). 검수용 wav 10개씩이 그 첫 단계다.
4. 채우기·무음 필터가 없어 조용한 창이 있다(§5 표). v1 `--floor-zero` 를 적용해 걸러 낸 수치는 만들지 않았다.
5. 창이 이벤트보다 길다 — 특히 dog_bark(중앙 0.6초).
6. 샘플 수가 작다(클래스당 74~121). 신뢰구간이 넓다.
