#!/usr/bin/env python3
"""받아 둔 10초 구간 → 이벤트 중심 1창(16384샘플, 16kHz) → 고정 스케일 int8.

변환은 기존 `scripts/prepare_safesound.py` 의 `to_int8` / `WIN` / `SR` 을 **import** 해서 쓴다
(복사 아님, 기존 코드 수정 없음). 정규화 없음 — CLAUDE.md 7장.

산출 (레포 밖, --out):
  windows_int8.npy  (N, 16384) int8      — meta.csv 행 순서와 같다
  meta.csv                               — sampling_list 열 + label_v1/label_real + 창 위치·레벨
  fail.csv                               — 처리 실패 (읽기 실패·너무 짧음)
검수용 wav (--spot): 그룹별 10개, 파일명에 클래스 포함, 16kHz int16 (양자화 전 창).

라벨 정수: 0 siren, 1 glass, 2 scream, 3 dog_bark, 4 background. 정의별로 따로 있다.
  label_v1 / label_real = 그 정의가 양성이라 부르는 클래스, 아니면 4.
  def_gap 은 label_v1=양성 클래스, label_real=4 (정의 차이를 재는 그룹).

⚠️ 평가 전용·재배포 금지. 학습에 쓰지 않는다.

사용 (낮은 우선순위): nice -n 19 python process_clips.py --list sampling_list.csv \
        --raw ~/safesound-external/raw --out ~/safesound-external/external_v1 \
        --spot ~/safesound-external/spotcheck
"""
import argparse
import csv
import os
import random
import subprocess
import sys

import numpy as np
import soundfile as sf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from prepare_safesound import SR, WIN, to_int8  # noqa: E402  (기존 변환 재사용)

CLS = ["siren", "glass", "scream", "dog_bark", "background"]
SEED = 78000
WEBM_PREROLL_S = 9.944


def read16k(path):
    p = subprocess.run(["/usr/bin/ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR),
                        "-f", "s16le", "-"], capture_output=True, timeout=60)
    if p.returncode != 0 or not p.stdout:
        return None
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spot", default=None)
    ap.add_argument("--n-spot", type=int, default=10)
    a = ap.parse_args()
    raw, out = os.path.expanduser(a.raw), os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)

    rows = list(csv.DictReader(open(a.list)))
    # webm 로 내려갔다가 webm_audit.py --retry --full 로 정확한 로컬 절단 오디오로 교체된 구간
    webm_dir = os.path.join(os.path.dirname(raw.rstrip("/")), "raw_webm_replaced")
    webm_replaced = {f[:-4] for f in os.listdir(webm_dir)} if os.path.isdir(webm_dir) else set()
    wins, meta, fails = [], [], []
    floats = {}                       # 검수용: sid → float 창 (그룹별 일부만 쓰므로 전부 보관해도 작다)
    for r in rows:
        sid = r["segment_id"]
        path = os.path.join(raw, f"{sid}.wav")
        if not os.path.exists(path):
            continue                  # 다운로드 실패분은 download_state.csv 에 사유가 있다
        x = read16k(path)
        if x is None:
            fails.append((sid, "read_error")); continue
        # webm 폴백 파일은 [시작-10s, 시작+10s] 20초다. 실제 구간은 9.944초 지점부터 10초
        # (m4a 와의 상호상관으로 3/3 확인, 2026-10-03). 이벤트 시각은 구간 시작 기준이라 잘라야 맞는다.
        webm_trim = len(x) > 15 * SR
        if webm_trim:
            x = x[int(round(WEBM_PREROLL_S * SR)):]
        if len(x) < WIN:
            fails.append((sid, f"short_{len(x) / SR:.2f}s")); continue
        if r["group"] == "background":
            start = random.Random(f"{SEED}-{sid}").randint(0, len(x) - WIN)
        else:
            center = (float(r["event_start"]) + float(r["event_end"])) / 2 * SR
            start = int(round(min(max(center - WIN / 2, 0), len(x) - WIN)))
        w = x[start:start + WIN]
        q = to_int8(w)
        g = r["group"]
        lv1 = CLS.index(r["v1_class"]) if r["v1_class"] else 4
        lre = CLS.index(r["real_class"]) if r["real_class"] else 4
        rms = float(np.sqrt(np.mean(w.astype(np.float64) ** 2)))
        zf = float(np.mean(q == 0))
        meta.append({**r, "label_v1": lv1, "label_real": lre, "win_start_sample": start,
                     "win_start_s": f"{start / SR:.3f}", "audio_len_s": f"{len(x) / SR:.2f}",
                     "webm_trim": int(webm_trim),
                     "webm_origin": int(webm_trim or sid in webm_replaced),   # 처음에 webm 로 내려간 구간 52개
                     "webm_replaced_exact": int(sid in webm_replaced),        # 정확한 로컬 절단으로 교체됨
                     "rms_dbfs": f"{20 * np.log10(max(rms, 1e-9)):.1f}",
                     "peak": f"{float(np.abs(w).max()):.3f}",
                     "zero_frac_int8": f"{zf:.3f}",
                     # v1 `--floor-zero` 규칙(int8 0 비율 > 0.95)에 걸리는 이벤트 창. 이벤트 클래스에만
                     # 하한이 걸린다 (CLAUDE.md 7장). **표시만 하고 지우지 않는다.**
                     "v1_floor_excluded": int(g != "background" and zf > 0.95),
                     # 배경은 하한이 없어 유지한다. 거의 빈/완전 무음 배경 표시용.
                     "bg_near_silent": int(g == "background" and zf > 0.95)})
        wins.append(q)
        floats[sid] = w

    np.save(os.path.join(out, "windows_int8.npy"), np.stack(wins) if wins else np.zeros((0, WIN), np.int8))
    if meta:
        with open(os.path.join(out, "meta.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, list(meta[0].keys()))
            wr.writeheader(); wr.writerows(meta)
    with open(os.path.join(out, "fail.csv"), "w", newline="") as f:
        csv.writer(f).writerows([("segment_id", "reason"), *fails])
    print(f"창 {len(wins)}개 저장, 처리 실패 {len(fails)}개 → {out}")

    if a.spot:
        spot = os.path.expanduser(a.spot)
        os.makedirs(spot, exist_ok=True)
        for g in CLS:
            ids = sorted(m["segment_id"] for m in meta if m["group"] == g)
            random.Random(SEED + 3).shuffle(ids)
            for k, sid in enumerate(ids[:a.n_spot], 1):
                w = floats[sid]
                sf.write(os.path.join(spot, f"{g}_{k:02d}_{sid}.wav"),
                         np.clip(np.round(w * 32767), -32768, 32767).astype(np.int16), SR, subtype="PCM_16")
        def dump(name, sid):
            w = floats[sid]
            sf.write(os.path.join(spot, name),
                     np.clip(np.round(w * 32767), -32768, 32767).astype(np.int16), SR, subtype="PCM_16")

        # 추가 검수 (2026-10-05): webm 오프셋 보정 구간 10개(이벤트 클래스 우선), def_gap 10개
        webm = sorted((m["segment_id"] for m in meta if m["webm_origin"] == 1 and m["group"] in CLS[:4]))
        rest = sorted(m["segment_id"] for m in meta if m["webm_origin"] == 1 and m["group"] not in CLS[:4])
        random.Random(SEED + 4).shuffle(webm)
        random.Random(SEED + 4).shuffle(rest)
        gmap = {m["segment_id"]: m["group"] for m in meta}
        for k, sid in enumerate((webm + rest)[:a.n_spot], 1):
            dump(f"webm_{k:02d}_{gmap[sid]}_{sid}.wav", sid)
        gap = sorted(m["segment_id"] for m in meta if m["group"] == "def_gap")
        random.Random(SEED + 4).shuffle(gap)
        for k, sid in enumerate(gap[:a.n_spot], 1):
            lab = next(m["v1_class"] for m in meta if m["segment_id"] == sid)
            dump(f"defgap_{k:02d}_{lab}_{sid}.wav", sid)
        print("검수용 wav →", spot)


if __name__ == "__main__":
    main()
