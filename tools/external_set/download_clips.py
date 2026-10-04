#!/usr/bin/env python3
"""sampling_list.csv 의 구간을 YouTube 에서 10초씩 받는다 (WSL2, 낮은 우선순위로 실행).

- 쿠키·로그인을 쓰지 않는다. 봇 차단·쿠키 요구·429 가 나오면 **즉시 중단**하고 상태를 남긴다.
- 영상 삭제·비공개·지역/저작권 차단은 구간별 실패 사유로 기록하고 계속한다 (대체 없음).
- 연령 제한은 영상 하나의 콘텐츠 게이트라 실패 사유(age_restricted)로만 기록한다. 쿠키로 우회하지 않는다.
- 호출 사이 4~8초 대기. 이어받기 가능 (state csv). `STOP` 파일을 만들면 현재 구간 후 멈춘다.
- 오디오는 레포 밖(--raw)에만 둔다. 재배포 금지, 평가 전용.

사용: nice -n 19 python download_clips.py --list sampling_list.csv --raw ~/safesound-external/raw
"""
import argparse
import csv
import os
import random
import subprocess
import sys
import time

BLOCK = ("not a bot", "http error 429", "too many requests", "captcha")
FAIL = [  # (사유, 부분 문자열) — 위에서부터 첫 매치
    ("age_restricted", "confirm your age"),
    ("private", "private video"),
    ("removed", "removed"),
    ("terminated", "terminated"),
    ("copyright", "copyright"),
    ("region_blocked", "your country"),
    ("members_only", "members-only"),
    ("unavailable", "video unavailable"),
    ("unavailable", "not available"),
    ("unavailable", "this video is"),
]


def classify(text):
    t = text.lower()
    last = text.strip().splitlines()[-1][:200] if text.strip() else "no output"
    if any(b in t for b in BLOCK):                      # 명백한 봇 차단·속도 제한
        return "BLOCK", last
    for reason, key in FAIL:                            # 영상별 사유 (메시지에 cookies 가 섞여도 여기서 걸린다)
        if key in t:
            return reason, ""
    # `Please sign in. Use --cookies...` 는 영상 하나가 로그인을 요구하는 경우에도 나온다 (2026-10-04
    # 사용자 결정: 영상 단위 실패로 기록). 연속으로 쌓이면 main 이 전체 차단으로 보고 중단한다.
    if "please sign in" in t:
        return "login_required", ""
    if "sign in" in t or "cookies" in t or "login" in t:   # 그 밖의 로그인·쿠키 요구는 안전하게 중단
        return "BLOCK", last
    return "other", last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--state", default=None)
    ap.add_argument("--limit", type=int, default=0, help="시험용: 이번 실행에서 시도할 최대 개수")
    ap.add_argument("--sleep", type=float, nargs=2, default=(4.0, 8.0))
    ap.add_argument("--seed", type=int, default=78000)
    a = ap.parse_args()
    raw = os.path.expanduser(a.raw)
    os.makedirs(raw, exist_ok=True)
    state_path = a.state or os.path.join(os.path.dirname(raw.rstrip("/")), "download_state.csv")
    stop_file = os.path.join(os.path.dirname(raw.rstrip("/")), "STOP")

    rows = list(csv.DictReader(open(a.list)))
    random.Random(a.seed).shuffle(rows)      # 중간에 멈춰도 클래스가 한쪽으로 쏠리지 않게

    done = {}
    if os.path.exists(state_path):
        for r in csv.DictReader(open(state_path)):
            done[r["segment_id"]] = r
    new_state = not os.path.exists(state_path)
    sf = open(state_path, "a", newline="")
    w = csv.writer(sf)
    if new_state:
        w.writerow(["segment_id", "status", "reason", "time"])

    def log(sid, status, reason):
        w.writerow([sid, status, reason, time.strftime("%F %T")])
        sf.flush()

    tried = 0
    consec_login = 0
    for r in rows:
        sid = r["segment_id"]
        if sid in done and done[sid]["status"] in ("ok", "fail"):
            continue
        if os.path.exists(stop_file):
            print("STOP 파일 감지 — 중단"); break
        if a.limit and tried >= a.limit:
            break
        out = os.path.join(raw, f"{sid}.wav")
        if os.path.exists(out):
            log(sid, "ok", "exists"); continue
        s = int(float(r["start_s"]))
        cmd = ["yt-dlp", "--ffmpeg-location", "/usr/bin", "-q", "--no-warnings", "--no-playlist",
               # 140(m4a)은 구간이 정확히 10초다. webm/opus(251)는 [시작-10s, 시작+10s] 20초가
               # 나온다 (2026-10-03 상호상관으로 확인) → 140 우선, 없을 때만 폴백하고 후처리에서 자른다.
               "-f", "140/bestaudio/best", "--download-sections", f"*{s}-{s + 10}",
               "-x", "--audio-format", "wav", "-o", os.path.join(raw, f"{sid}.%(ext)s"),
               f"https://www.youtube.com/watch?v={r['ytid']}"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            text = (p.stderr or "") + (p.stdout or "")
            rc = p.returncode
        except subprocess.TimeoutExpired:
            text, rc = "timeout", 1
        tried += 1
        if rc == 0 and os.path.exists(out) and os.path.getsize(out) > 44:
            consec_login = 0
            log(sid, "ok", "")
            print(f"[{tried}] ok   {sid}", flush=True)
        else:
            reason, detail = classify(text)
            if reason == "BLOCK":
                log(sid, "blocked", detail)
                print(f"\n중단: 봇 차단/로그인·쿠키 요구/429 감지 — {sid}\n  {detail}", flush=True)
                sys.exit(3)
            consec_login = consec_login + 1 if reason == "login_required" else 0
            if consec_login >= 3:
                log(sid, "blocked", "login_required x3 연속 — 전체 차단으로 판단")
                print(f"\n중단: login_required 3회 연속 — {sid}", flush=True)
                sys.exit(3)
            log(sid, "fail", reason + (f": {detail}" if reason == "other" else ""))
            print(f"[{tried}] fail {sid} {reason} {detail}", flush=True)
        time.sleep(random.uniform(*a.sleep))
    print("종료. 시도", tried)


if __name__ == "__main__":
    main()
