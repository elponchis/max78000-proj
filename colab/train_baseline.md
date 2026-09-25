# Colab 기준선 학습 런북 — SafeSound v1

KWS20 v3 백본 + 5클래스 출력 + QAT 로 **기준선(v1)** 을 학습한다.
결과물의 우선순위는 **혼동행렬**이다 — 전체 정확도는 배경음이 3배라 쉽게 높아
보이므로 클래스별로 봐야 한다.

- 실행 환경: **Google Colab** (로컬 GPU 없음, CPU 는 1 epoch 77분 — CLAUDE.md 0장)
- 전제: WSL2 에서 `prepare_safesound.py` 로 샤드를 만들고 tar 로 묶어 Drive 에 올린 상태
- 데이터 태그: `dataset-v1` (규칙 변경은 보드 실측 뒤 v2 에서)

---

## 0. WSL2 에서 먼저 (Colab 아님)

```bash
# 샤드 생성 → tar → Drive 업로드용 파일 만들기
cd ~/max78000-proj
python3 scripts/prepare_safesound.py                       # data/processed/safesound/
python3 tools/pack_dataset.py                              # safesound-v1.tar.gz + sha256
```

`tools/pack_dataset.py` 가 tar 와 `MANIFEST.json`(클래스별 창 수·설정·커밋 해시)을
함께 넣는다. **Drive 에는 이 tar 하나만 올린다** — 원본 오디오 30GB 는 올리지 않는다
(CLAUDE.md 6장).

---

## 1. Colab 셀 순서

### 셀 1 — conda 설치 (런타임 재시작 유발)

```python
!pip install -q condacolab
import condacolab; condacolab.install()
```

> 재시작 후 셀 2부터 다시 실행한다. CLAUDE.md 12장의 복구 절차와 같다.

### 셀 2 — 환경 복구 (Drive 백업에서)

```python
from google.colab import drive
drive.mount('/content/drive')
!tar xzf /content/drive/MyDrive/max78000/ai8x-conda-env.tar.gz -C /usr/local/envs
!rm -rf /content/ai8x-training
!tar xzf /content/drive/MyDrive/max78000/ai8x-training-patched.tar.gz -C /content
!apt-get -qq install -y libsox3 libsox-dev libsox-fmt-all sox
import os; os.environ['MPLBACKEND'] = 'Agg'
%cd /content/ai8x-training
!rm -rf logs && ln -s /content/drive/MyDrive/max78000/logs logs
```

### 셀 3 — 우리 레포 가져오기 + 심볼릭 링크

```python
%cd /content
!rm -rf max78000-proj
!git clone -q https://github.com/elponchis/max78000-proj.git
!ln -sf /content/max78000-proj/datasets/safesound.py /content/ai8x-training/datasets/safesound.py
!ln -sf /content/max78000-proj/models/ai85net-safesound.py /content/ai8x-training/models/ai85net-safesound.py
!ls -l /content/ai8x-training/datasets/safesound.py /content/ai8x-training/models/ai85net-safesound.py
```

> 데이터로더·모델은 **레포에 두고 링크만 건다** (CLAUDE.md 11장). ai8x 레포를
> 직접 고치면 어디까지가 본인 작업인지 흐려진다.

### 셀 4 — 데이터셋 풀기 + 무결성 확인

```python
import hashlib, json, pathlib
src = '/content/drive/MyDrive/max78000/safesound-v1.tar.gz'
h = hashlib.sha256(open(src,'rb').read()).hexdigest()
print('sha256', h)
print('기대값 ', open(src + '.sha256').read().split()[0])
assert h == open(src + '.sha256').read().split()[0], 'tar 가 손상됐다'
!mkdir -p /content/ai8x-training/data/SafeSound
!tar xzf {src} -C /content/ai8x-training/data/SafeSound
print(json.load(open('/content/ai8x-training/data/SafeSound/MANIFEST.json'))['counts'])
```

> 경로는 `data/SafeSound/{train,test}/{클래스}/` 가 되어야 한다
> (`safesound_get_datasets` 가 `data_dir/SafeSound` 를 본다).

### 셀 5 — 데이터로더 KAT (학습 전 필수)

```python
%cd /content/max78000-proj
!python tools/kat_safesound.py
```

> **여기서 실패하면 학습하지 말 것.** 축이 뒤바뀌어도 손실은 내려가므로,
> 합성해서 보드에 올린 뒤에야 드러난다 (CLAUDE.md 7장, G4).

### 셀 6 — 한 배치만 확인 (샘플 수·클래스 분포)

```python
%cd /content/ai8x-training
!conda run -n ai8x --no-capture-output python - <<'PY'
import sys, collections, torch
sys.path.insert(0, '.')
import ai8x, datasets.safesound as S
ai8x.set_device(85, False, False)
class A: act_mode_8bit=False; truncate_testset=False
tr, te = S.safesound_get_datasets(('data', A()), True, True)
print('train', len(tr), 'test', len(te))
print('train 분포', collections.Counter(t for _, t in ((None, y) for _, y in
      ((tr.index[i][0], tr.index[i][0]) for i in range(len(tr))))))
x, y = tr[0]
print('입력', tuple(x.shape), x.dtype, '값범위', float(x.min()), float(x.max()))
PY
```

### 셀 7 — 기준선 학습 (QAT 포함)

```python
%cd /content/ai8x-training
!conda run -n ai8x --no-capture-output python train.py \
  --epochs 150 --optimizer Adam --lr 0.001 --wd 0 --deterministic \
  --compress /content/max78000-proj/colab/schedule_safesound.yaml \
  --qat-policy /content/max78000-proj/colab/qat_policy_safesound.yaml \
  --model ai85safesoundnet --dataset SafeSound \
  --device MAX78000 --confusion --batch-size 128 \
  --name safesound-v1 --out-dir logs
```

| 인자 | 이유 |
|---|---|
| `--qat-policy` | MAX78000 은 QAT 필수 (CLAUDE.md 3장). 60 epoch 부터 8bit |
| `--confusion` | **혼동행렬이 1순위 산출물**. 배경음 3배라 전체 정확도는 과대평가된다 |

**클래스 가중치는 데이터로더가 들고 있다.** `datasets/safesound.py` 의
`class_weights()` 가 역빈도(`N/(K·n_c)`)를 계산해 `datasets` 딕셔너리의 `weight`
로 넘기고, ai8x 가 그대로 `nn.CrossEntropyLoss(weight=...)` 에 넣는다 (train.py:452).
v1 값은 siren 4.93 / glass 5.45 / scream 5.30 / dog_bark 1.35 / background 0.27 이다.

> 배경음만 낮추는 것으로는 부족하다 — 이벤트끼리도 dog_bark 1,934 vs glass 477 로
> **4배** 차이라, 가중치 없이 학습하면 dog_bark 쪽으로 기울고 정작 취약한
> glass·siren 재현율이 낮게 수렴한다. 정규화가 `N/(K·n_c)` 라 표본당 평균
> 가중치가 1이므로 학습률을 다시 잡을 필요는 없다.
> 재집계로 수량이 바뀌면 `class_weights('<샤드 경로>')` 로 다시 뽑아 상수를 갱신할 것.
| `--deterministic` | 논문 재현 절차에 시드 고정이 필요하다 |
| `--device MAX78000` | 지원 연산·반올림 규칙을 학습에 반영 |

> Colab 무료 티어는 세션이 끊긴다. `logs` 가 Drive 심볼릭 링크라 체크포인트는
> 남는다. 재개는 `--resume-from logs/safesound-v1/checkpoint.pth.tar`.

### 셀 8 — 혼동행렬·클래스별 지표 뽑기 (두 단위 + 신뢰구간)

먼저 ai8x 자체 평가(창 단위, 로그 대조용):

```python
%cd /content/ai8x-training
!conda run -n ai8x --no-capture-output python train.py \
  --evaluate --model ai85safesoundnet --dataset SafeSound --device MAX78000 \
  --exp-load-weights-from logs/safesound-v1/best.pth.tar \
  --confusion --batch-size 128 --out-dir logs --name safesound-v1-eval
```

그다음 **원본 단위까지** 내는 우리 스크립트:

```python
!conda run -n ai8x --no-capture-output python /content/max78000-proj/tools/eval_confusion.py \
  --checkpoint logs/safesound-v1/best.pth.tar \
  --data /content/ai8x-training/data --ai8x /content/ai8x-training
```

내는 것이 넷이다.

1. **창 단위 혼동행렬** — ai8x 로그와 대조된다
2. **원본 단위 혼동행렬** — 같은 `fsid` 의 창들을 다수결로 묶는다. 동점은 정답을
   **피하는 쪽**으로 깨어 성능을 부풀리지 않는다
3. **클래스별 재현율 95% 신뢰구간** — **원본을 복원추출**하는 부트스트랩
4. **배경음 오탐률 → 시간당 오경보** (hop 250ms → 시간당 14,400회 추론).
   연속 N프레임 다수결(N=1,2,3)을 적용했을 때의 시간당 오경보도 함께 낸다

> **4번이 G7 의 실질 기준이다.** 창 단위 오탐률 0.1% 는 작아 보이지만 시간당
> 14회가 울린다. 무인 8시간이면 115회다 — 쓸 수 없는 물건이라는 뜻이다.
> N프레임 수치는 펌웨어의 이벤트 병합 상태머신(CLAUDE.md 6장)을 **근사**한 것이고,
> 스크립트가 근사의 한계(선택된 창들의 실제 간격, 인접 쌍 비율)를 함께 출력한다.
> 확정치는 보드 8시간 무인 구동으로 잰다.

> ⚠️ 창 단위 숫자만 논문에 쓰지 말 것. 한 원본에서 여러 창이 나오므로 창을
> 독립 표본으로 세면 신뢰구간이 실제보다 좁아진다 (CLAUDE.md 5장 규칙 1).
> 실기기의 체감 단위도 창이 아니라 "이 소리를 맞혔나"다 — 이벤트 병합
> 상태머신이 연속 추론을 하나로 묶기 때문이다 (6장).
>
> 결과는 `docs/results/` 에 옮겨 적는다. **두 단위의 macro-F1 과 클래스별
> 재현율·CI를 함께** 기록할 것 — siren(test 원본 54개)과 glass(147개)가
> 취약 후보이고, 원본이 적은 만큼 CI 가 넓게 나올 것이다.

### 셀 9 — 체크포인트 Drive 백업

```python
!cp -v logs/safesound-v1/best.pth.tar /content/drive/MyDrive/max78000/
```

---

## 2. 학습 후 (WSL2 로 돌아와서)

```bash
# 양자화 → 합성 → 메모리 제약 확인 (442KB / 512KB)
cd ~/ai8x-synthesis
python quantize.py best.pth.tar safesound-q8.pth.tar --device MAX78000
python ai8xize.py --test-dir out --prefix safesound --checkpoint-file safesound-q8.pth.tar \
  --config-file networks/safesound.yaml --device MAX78000 --compact-data --softmax
```

합성 로그의 가중치 메모리 사용량을 `TASKS.md` 에 기록한다. 초과하면 `width_mult`
를 낮춘다 (`models/ai85net-safesound.py` 의 표 참조).

---

## 3. 확인할 것 / 함정

- **전체 정확도를 성과로 쓰지 말 것.** 배경음이 이벤트 합의 약 2.8배라 전부
  배경음으로 찍어도 74% 가 나온다. 혼동행렬 → 클래스별 재현율 → macro-F1 순으로 본다
- **목표 미달 클래스(siren 528 / glass 477 / scream 491, 목표 train 800)는
  기준선 결과를 보고 판단한다.** 지금 추가 수집하지 않는다 — 재현율이 이미
  쓸 만하면 수집은 낭비이고, 낮더라도 원인이 수량인지 라벨 품질인지
  혼동행렬을 봐야 갈린다
- **scream ↔ background 혼동**을 먼저 볼 것. 태거 임계값 0.025 로 내리면서 약 45%
  오염을 감수했다 (`docs/results/listening-verification.md` 4.2). 이 혼동이 크면
  임계값을 0.10 으로 되돌리는 판단이 선다
- **가중치 조정은 오탐률을 보고 한다.** 기준선은 순수 역빈도(`power=1.0`)로
  고정한다. 시간당 오경보가 과하면 두 갈래다 — background 가중치만 올리거나,
  `class_weights(power=0.5)`(제곱근 역빈도, background 0.27 → 0.64)로 전체를
  완화한다. 둘 다 이벤트 재현율을 일부 내주는 맞교환이라 수치를 보고 정한다
- Colab RAM 12GB — 데이터로더가 lazy 라 문제없어야 하지만, `num_workers` 를
  과하게 올리면 워커마다 mmap 페이지를 잡는다. 4 이하로 둘 것
