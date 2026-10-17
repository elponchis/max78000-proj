#!/usr/bin/env python3
"""②′ (safesound-v4d-fe, 시드 1~3) 를 eval_v4.score_ckpt 로 채점해 외부 mAP 등을 `data/external_eval/fe_eval.json` 에 남긴다.

v4_eval.json 은 다섯 구성 + (a) 의 판별 결과만 담고 있어 (eval_v4.EDITIONS), ②′ 의 외부 mAP (표 2 열) 를 같은 코드로 따로 뽑는다.
같은 체크포인트·같은 문턱값 규칙(시험셋 배경 @300/h)·같은 ap_of 라서 v4_eval.json 과 나란히 쓸 수 있다.
scream_listened_test.json 의 "②′ fe" 와 thr·testL F1·외부 F1 이 1e-9 안에서 같아야 한다 (make_paper_assets.py 가 대조).

사용 (WSL2): OPENBLAS_NUM_THREADS=1 ~/ai8x-training/venv/bin/python tools/fe_ext_eval.py
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import eval_v4 as Q                                   # noqa: E402

OUT = os.path.join(REPO, "data", "external_eval", "fe_eval.json")
CONFIG, PREFIX, NAME = "wave_fe", "safesound-v4d-fe", "②′ fe"


def main():
    S = Q.setup()
    rows = []
    for ck in Q.seeds_of(PREFIX):
        r = Q.score_ckpt(CONFIG, ck, S)
        r["ckpt"] = os.path.basename(ck)
        rows.append(r)
        print(f"{r['ckpt']}: thr {r['thr']:.4f}  testL F1 {r['test_f1_L']:.4f}  외부 F1 {r['ext_f1_real31L']:.4f}  "
              f"외부 mAP {r['ext_map_real31L']:.4f}  발화 {r['ext_fire_n']}")
    out = {"config": CONFIG, "prefix": PREFIX, "name": NAME, "note": "eval_v4.score_ckpt 와 같은 경로 (tools/fe_ext_eval.py)", "seeds": rows}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"저장: {OUT} ({len(rows)} 시드)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
