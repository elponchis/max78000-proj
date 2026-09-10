# max78000-proj

MAX78000 기반 상시 동작 음향 이벤트 감지 시스템 — 졸업 프로젝트.

> **MAX78000 기반 상시 동작 음향 감지 시스템에서 종단간 추론 파이프라인 설계**
> — 전처리 위치가 실시간 제약 충족과 에너지 효율에 미치는 영향

배터리로 상시 구동되는 소형 음향 이벤트 감지 장치를 MAX78000FTHR 단독으로 구현하고,
전처리를 CPU에서 수행하는 구성과 NPU 친화적으로 재설계한 구성을 대조하여
NPU 탑재 MCU에서의 파이프라인 설계 지침을 정량적으로 도출한다.

## 문서

| 파일 | 내용 |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | 프로젝트 불변 컨텍스트 — 제약·규칙·목표. **작업 전 필독** |
| [`TASKS.md`](TASKS.md) | 실행 체크리스트 |
| [`docs/results/`](docs/results/) | 실험·조사 결과 |
| [`docs/data-layout.md`](docs/data-layout.md) | `data/` 디렉터리 구조 (git 제외) |

## 현재 상태

- Phase 0 (환경 구축) 완료 — 로컬 WSL2 + Colab, 합성 169KB / 442KB
- Phase 1 (레포 구성) 완료
- Phase 2.1 (클래스 확정) 진행 중 — 라벨 조사 완료, 클래스 매핑 미확정
- **MAX78000FTHR 보드 미확보** — Phase 4 이후 착수 불가

## 실행 환경

| 작업 | 환경 |
|---|---|
| 데이터 전처리, 양자화·합성 | WSL2 (Ubuntu 22.04) |
| 모델 학습 | Google Colab (T4) |
| MSDK 펌웨어 빌드·플래싱 | Windows 네이티브 |
| 호스트 GUI | Windows 네이티브 |

벤더 툴체인(`ai8x-training`, `ai8x-synthesis`)은 이 레포에 포함하지 않는다.
수정 사항은 [`patches/`](patches/)에 `git diff`로 기록한다.

## 라이선스 / 데이터 출처

- FSD50K — CC 라이선스
- UrbanSound8K, ESC-50 — 비상업 연구용
