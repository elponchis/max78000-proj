#!/usr/bin/env python3
"""webm 폴백 구간(20초, [시작-10s, 시작+10s])의 오프셋 보정(9.944초)을 검증한다.

1) 자동 점검 (네트워크 없음): 이벤트 구간 RMS − 구간 밖 RMS(대비, dB)를 **보정 오프셋**과
   **다른 모든 오프셋**(0~10초, 50ms 간격)에서 계산한다. 보정이 맞다면 보정 오프셋에서 대비가 크다.
   의심: 대비 ≤ 0.5dB 이거나, 다른 오프셋이 6dB 넘게 더 높고 그 위치가 0.5초 이상 떨어져 있을 때.
   구간 밖이 1초 미만이면(siren 처럼 거의 전 구간이 이벤트) 대비를 못 재고 'n/a' 로 둔다.
   배경 구간은 이벤트가 없어 점검 불가(n/a).
2) --retry: m4a(140)만 강제로 다시 받아, 받아지면 webm 과 **상호상관 오프셋**을 재서 기록하고
   raw 를 m4a 로 교체한다 (webm 원본은 raw_webm_replaced/ 로 옮긴다). 쿠키 없음, 봇 차단 시 즉시 중단.

산출: ~/safesound-external/webm_audit.csv

사용: python webm_audit.py --meta external_v1/meta.csv --raw raw [--retry]
"""
import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import time

import numpy as np

SR = 16000
PREROLL = 9.944
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from download_clips import classify  # noqa: E402  (실패 사유 분류 재사용)


def read16k(p):
    o = subprocess.run(["/usr/bin/ffmpeg", "-v", "error", "-i", p, "-ac", "1", "-ar", str(SR),
                        "-f", "s16le", "-"], capture_output=True, timeout=60).stdout
    return np.frombuffer(o, np.int16).astype(np.float64) / 32768.0


def db(x):
    return 10 * np.log10(max(float(np.mean(x ** 2)), 1e-12)) if len(x) else float("nan")


def contrast(seg, es, ee):
    """seg: 10초 후보 구간. 이벤트 [es,ee] 안 RMS − 밖 RMS (dB). 밖이 1초 미만이면 None."""
    a, b = int(max(es, 0) * SR), int(min(ee, 10) * SR)
    inside = seg[a:b]
    outside = np.concatenate([seg[:a], seg[b:]])
    if len(inside) < SR * 0.05 or len(outside) < SR:
        return None
    return db(inside) - db(outside)


def audit(x, es, ee):
    base = int(round(PREROLL * SR))
    exp = contrast(x[base:base + 10 * SR], es, ee)
    best, bo = None, None
    for o in np.arange(0, 10.001, 0.05):
        s = int(round(o * SR))
        c = contrast(x[s:s + 10 * SR], es, ee) if s + 10 * SR <= len(x) else None
        if c is not None and (best is None or c > best):
            best, bo = c, o
    return exp, best, bo


def xcorr_offset(a, b):
    n = len(a) + len(b)
    cc = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    lag = int(np.argmax(cc))
    lag = lag if lag < n // 2 else lag - n
    return lag / SR, float(cc.max() / np.sqrt((a ** 2).sum() * (b ** 2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--retry", action="store_true")
    ap.add_argument("--full", action="store_true", help="--retry 와 함께: 전체 오디오를 받아 로컬에서 10초를 자른다")
    a = ap.parse_args()
    raw = os.path.expanduser(a.raw)
    out = a.out or os.path.join(os.path.dirname(raw.rstrip("/")), "webm_audit.csv")
    rows = [r for r in csv.DictReader(open(os.path.expanduser(a.meta))) if r["webm_trim"] == "1"]
    print("webm_trim 구간", len(rows))
    res = []
    for r in rows:
        sid = r["segment_id"]
        x = read16k(os.path.join(raw, f"{sid}.wav"))
        rec = {"segment_id": sid, "group": r["group"], "v1_class": r["v1_class"], "real_class": r["real_class"],
               "event_start": r["event_start"], "event_end": r["event_end"], "len_s": f"{len(x) / SR:.2f}",
               "contrast_expected_db": "n/a", "contrast_best_db": "n/a", "best_offset_s": "n/a",
               "verdict": "n/a", "m4a_retry": "", "xcorr_offset_s": "", "xcorr_peak": ""}
        if r["group"] != "background" and r["event_start"]:
            ce, cb, bo = audit(x, float(r["event_start"]), float(r["event_end"]))
            if ce is None:
                rec["verdict"] = "n/a(구간 밖<1s)"
            else:
                rec.update(contrast_expected_db=f"{ce:.1f}", contrast_best_db=f"{cb:.1f}", best_offset_s=f"{bo:.2f}")
                sus = ce <= 0.5 or (cb - ce > 6 and abs(bo - PREROLL) > 0.5)
                rec["verdict"] = "SUSPECT" if sus else "ok"
        res.append(rec)

    if a.retry:
        os.makedirs(os.path.join(raw, "..", "raw_webm_replaced"), exist_ok=True)
        tmp = os.path.join(raw, "..", "raw_m4a_retry")
        os.makedirs(tmp, exist_ok=True)
        random.seed(78000)
        for rec in res:
            sid = rec["segment_id"]
            yt, st = sid.rsplit("_", 1)
            s = int(st) // 1000
            if a.full:
                # --full: 구간 지정 없이 전체 오디오를 받고 ffmpeg 로 정확히 [s, s+10] 을 로컬에서 자른다.
                # webm 구간 다운로드의 20초 동작(오프셋 보정)과 무관한 독립 기준이다. 80MB 초과는 건너뛴다.
                fullp = os.path.join(tmp, f"{sid}.full")
                for old in os.listdir(tmp):
                    if old.startswith(f"{sid}.full"):
                        os.remove(os.path.join(tmp, old))
                p = subprocess.run(["yt-dlp", "-q", "--no-warnings", "--no-playlist", "--max-filesize", "80M",
                                    "-f", "251/250/249/bestaudio", "-o", fullp + ".%(ext)s",
                                    f"https://www.youtube.com/watch?v={yt}"],
                                   capture_output=True, text=True, timeout=300)
                got = [f for f in os.listdir(tmp) if f.startswith(f"{sid}.full")]
                if p.returncode == 0 and got:
                    subprocess.run(["/usr/bin/ffmpeg", "-v", "error", "-y", "-ss", str(s), "-t", "10",
                                    "-i", os.path.join(tmp, got[0]), os.path.join(tmp, f"{sid}.wav")],
                                   capture_output=True, timeout=120)
                    os.remove(os.path.join(tmp, got[0]))
            else:
                p = subprocess.run(["yt-dlp", "--ffmpeg-location", "/usr/bin", "-q", "--no-warnings", "--no-playlist",
                                    "-f", "140", "--download-sections", f"*{s}-{s + 10}", "-x", "--audio-format", "wav",
                                    "-o", os.path.join(tmp, f"{sid}.%(ext)s"), f"https://www.youtube.com/watch?v={yt}"],
                                   capture_output=True, text=True, timeout=180)
            reason, detail = classify((p.stderr or "") + (p.stdout or ""))
            newp = os.path.join(tmp, f"{sid}.wav")
            if p.returncode == 0 and os.path.exists(newp):
                w = read16k(os.path.join(raw, f"{sid}.wav"))
                m = read16k(newp)
                off, pk = xcorr_offset(w, m)
                rec.update(m4a_retry="ok", xcorr_offset_s=f"{off:.3f}", xcorr_peak=f"{pk:.2f}")
                if abs(len(m) / SR - 10) < 0.6 and pk > 0.2:     # 교체 조건: 길이 정상 + 상관 뚜렷
                    shutil.move(os.path.join(raw, f"{sid}.wav"), os.path.join(raw, "..", "raw_webm_replaced", f"{sid}.wav"))
                    shutil.move(newp, os.path.join(raw, f"{sid}.wav"))
                    rec["m4a_retry"] = "replaced"
            elif reason == "BLOCK":
                print("중단: 차단 감지", sid, detail)
                rec["m4a_retry"] = "blocked"
                break
            else:
                rec["m4a_retry"] = f"fail:{reason}"
            print(sid, rec["m4a_retry"], rec["xcorr_offset_s"], flush=True)
            time.sleep(random.uniform(4, 8))

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, list(res[0].keys()))
        w.writeheader()
        w.writerows(res)
    import collections
    print("verdict", collections.Counter(r["verdict"] for r in res))
    print("m4a_retry", collections.Counter(r["m4a_retry"] for r in res))
    offs = [float(r["xcorr_offset_s"]) for r in res if r["xcorr_offset_s"]]
    if offs:
        print(f"xcorr 오프셋 n={len(offs)} 평균 {np.mean(offs):.3f} 최소 {min(offs):.3f} 최대 {max(offs):.3f}")


if __name__ == "__main__":
    main()
