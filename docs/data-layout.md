# data/ 디렉터리 구조 (git 제외)

```
data/
├── raw/          # 원본 다운로드
│   └── FSD50K_meta/FSD50K.ground_truth/{vocabulary,dev,eval}.csv
├── interim/      # 라벨 단위 중간 산출물 (윈도우 추출 결과)
└── processed/    # 학습용 최종 캐시 (memmap / 청크 파일)
```

`data/` 전체가 `.gitignore` 대상이다. 재생성 절차는 `scripts/` 참조.
