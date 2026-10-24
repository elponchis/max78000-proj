#!/usr/bin/env python3
"""②′ 계열 (safesound-v4d-fe[-b64|-f31|-b64f31], 시드 1~3) 를 eval_v4.score_ckpt 로 채점해 외부 mAP 등을
`data/external_eval/fe_eval[_<변형>].json` 에 남긴다.

v4_eval.json 은 다섯 구성 + (a) 의 판별 결과만 담고 있어 (eval_v4.EDITIONS), ②′ 의 외부 mAP (표 2 열) 를 같은 코드로 따로 뽑는다.
같은 체크포인트·같은 문턱값 규칙(시험셋 배경 @300/h)·같은 ap_of 라서 v4_eval.json 과 나란히 쓸 수 있다.
scream_listened_test.json 의 같은 구성과 thr·testL F1·외부 F1 이 1e-9 안에서 같아야 한다 (make_paper_assets.py 가 ②′ 를 대조).

--variant fe (기본, ②′ → fe_eval.json) | b64 | f31 | b64f31 (②′ 곡선 변형, docs/results/fe-curve-design.md → fe_eval_<변형>.json)

사용 (WSL2): OPENBLAS_NUM_THREADS=1 ~/ai8x-training/venv/bin/python tools/fe_ext_eval.py [--variant b64]
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import eval_v4 as Q                                   # noqa: E402

# 변형 → (평가 config, 실행 이름 접두사, 표시 이름, 출력 파일). scream_stage1.CONFIGS 와 같은 이름을 쓴다.
VARIANTS = {
    "fe":     ("wave_fe",        "safesound-v4d-fe",        "②′ fe",     "fe_eval.json"),
    "b64":    ("wave_fe_b64",    "safesound-v4d-fe-b64",    "②′ b64",    "fe_eval_b64.json"),
    "f31":    ("wave_fe_f31",    "safesound-v4d-fe-f31",    "②′ f31",    "fe_eval_f31.json"),
    "b64f31": ("wave_fe_b64f31", "safesound-v4d-fe-b64f31", "②′ b64f31", "fe_eval_b64f31.json"),
    # 후속 연구 — p1 8비트 + Abs + NPU 합산 전체 QAT 재학습 (docs/results/npu-readback-design.md 7절)
    "fe_v2":      ("wave_fe_v2",      "safesound-v4d-fe-v2",      "②′ v2",         "fe_eval_fe_v2.json"),
    "fe_v2_b64f31": ("wave_fe_v2_b64f31", "safesound-v4d-fe-v2-b64f31", "②′ v2 b64f31", "fe_eval_fe_v2_b64f31.json"),
    "fe_v2_b64":  ("wave_fe_v2_b64",  "safesound-v4d-fe-v2-b64",  "②′ v2 b64",  "fe_eval_fe_v2_b64.json"),
    "fe_v2_f31":  ("wave_fe_v2_f31",  "safesound-v4d-fe-v2-f31",  "②′ v2 f31",  "fe_eval_fe_v2_f31.json"),
    "fe8":        ("wave_fe8",        "safesound-v4d-fe8",        "②′ fe8",        "fe_eval_fe8.json"),
    "fe8_b64f31": ("wave_fe8_b64f31", "safesound-v4d-fe8-b64f31", "②′ fe8 b64f31", "fe_eval_fe8_b64f31.json"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="fe")
    a = ap.parse_args()
    config, prefix, name, fn = VARIANTS[a.variant]
    out_path = os.path.join(REPO, "data", "external_eval", fn)
    S = Q.setup()
    rows = []
    for ck in Q.seeds_of(prefix):
        r = Q.score_ckpt(config, ck, S)
        r["ckpt"] = os.path.basename(ck)
        rows.append(r)
        print(f"{r['ckpt']}: thr {r['thr']:.4f}  testL F1 {r['test_f1_L']:.4f}  외부 F1 {r['ext_f1_real31L']:.4f}  "
              f"외부 mAP {r['ext_map_real31L']:.4f}  발화 {r['ext_fire_n']}")
    out = {"config": config, "prefix": prefix, "name": name, "variant": a.variant,
           "note": "eval_v4.score_ckpt 와 같은 경로 (tools/fe_ext_eval.py)", "seeds": rows}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"저장: {out_path} ({len(rows)} 시드)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
