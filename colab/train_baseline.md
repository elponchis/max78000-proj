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
| `--deterministic` | 논문 재현 절차에 시드 고정이 필요하다 |
| `--device MAX78000` | 지원 연산·반올림 규칙을 학습에 반영 |

> Colab 무료 티어는 세션이 끊긴다. `logs` 가 Drive 심볼릭 링크라 체크포인트는
> 남는다. 재개는 `--resume-from logs/safesound-v1/checkpoint.pth.tar`.

### 셀 8 — 혼동행렬·클래스별 지표 뽑기

```python
%cd /content/ai8x-training
!conda run -n ai8x --no-capture-output python train.py \
  --evaluate --model ai85safesoundnet --dataset SafeSound --device MAX78000 \
  --exp-load-weights-from logs/safesound-v1/best.pth.tar \
  --confusion --batch-size 128 --out-dir logs --name safesound-v1-eval
```

> 로그의 혼동행렬을 `docs/results/` 로 옮겨 적을 것. **macro-F1 과 클래스별
> 재현율을 함께** 기록한다 — 특히 siren(원본 129개)과 glass 가 취약 후보다.

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

- **전체 정확도를 성과로 쓰지 말 것.** 배경음이 이벤트 합의 약 3배라 전부
  배경음으로 찍어도 75% 가 나온다. 혼동행렬 → 클래스별 재현율 → macro-F1 순으로 본다
- **scream ↔ background 혼동**을 먼저 볼 것. 태거 임계값 0.025 로 내리면서 약 45%
  오염을 감수했다 (`docs/results/listening-verification.md` 4.2). 이 혼동이 크면
  임계값을 0.10 으로 되돌리는 판단이 선다
- **배경음 가중치 0.34** 는 오탐률 보고 조정할 파라미터다 (CLAUDE.md 7장)
- Colab RAM 12GB — 데이터로더가 lazy 라 문제없어야 하지만, `num_workers` 를
  과하게 올리면 워커마다 mmap 페이지를 잡는다. 4 이하로 둘 것
