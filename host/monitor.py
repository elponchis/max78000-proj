#!/usr/bin/env python3
"""보드 UART 스트림을 읽어 터미널에 실시간으로 보여준다.

`docs/uart-protocol.md` 의 파서를 **GUI 와 공유**하는 기준 구현이다. Qt 없이
표준 라이브러리 + pyserial 만 쓴다.

용도 셋
  1. **보드 브링업** — `docs/board-bringup.md` 2.4 의 miniterm 대신. 깨진 줄을
     세어 주고, 프레임 드롭·지연·이벤트를 바로 보여준다
  2. **무인 구동(G7)** — `--log events.csv` 로 SD 스키마와 같은 CSV 를 남긴다.
     보드 SD 와 호스트 CSV 가 일치하는지 대조할 수 있다
  3. GUI 의 파서·상태 모델 (`Monitor` 클래스를 그대로 가져다 쓴다)

사용 (Windows 네이티브 — 시리얼은 WSL 에서 안 된다):
    py host/monitor.py --port COM7
    py host/monitor.py --port COM7 --log events.csv --raw raw.txt

보드 없이 (WSL 에서도 됨):
    python3 host/fake_serial.py --rate 30 | python3 host/monitor.py
    python3 host/monitor.py --self-test
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fake_serial import parse  # noqa: E402  (같은 파서를 쓴다)

CLASSES = ["siren", "glass", "scream", "dog_bark", "background"]
URGENT = {"siren", "glass", "scream"}       # 알림 등급 (CLAUDE.md 4장)


def cls_name(i):
    try:
        return CLASSES[int(i)]
    except (ValueError, TypeError, IndexError):
        return f"?{i}"


class Monitor:
    """프로토콜 줄을 먹여 상태를 갱신한다. 출력을 하지 않는다 — GUI 재사용."""

    def __init__(self):
        self.banner = {}
        self.frames = 0
        self.drops = 0
        self.bad_lines = 0      # 체크섬 불일치·형식 오류 (재동기화의 지표)
        self.unknown = {}       # 모르는 TYPE 별 카운트
        self.last_frame = None
        self.last_status = {}
        self.events = []
        self.lat_us = []
        self.t0_wall = time.time()
        self.seq_gaps = 0       # seq 가 건너뛴 횟수 = 유실
        self._last_seq = None

    def feed(self, line):
        """한 줄. (종류, 내용) 또는 None."""
        r = parse(line)
        if r is None:
            self.bad_lines += 1
            return None
        typ, f = r
        if typ == "B":
            self.banner = f
            # 보드가 재부팅하면 seq 가 1부터 다시 시작한다 — 유실로 세지 않는다
            self._last_seq = None
        elif typ == "F":
            self.frames += 1
            self.last_frame = f
            try:
                s = int(f.get("seq", 0))
                if self._last_seq is not None and s > self._last_seq + 1:
                    self.seq_gaps += s - self._last_seq - 1
                self._last_seq = s
            except ValueError:
                pass
            try:
                self.lat_us.append(int(f["lat_us"]))
            except (KeyError, ValueError):
                pass
        elif typ == "E":
            self.events.append(f)
        elif typ == "S":
            self.last_status = f
            try:
                self.drops = int(f.get("drops", 0))
            except ValueError:
                pass
        else:
            self.unknown[typ] = self.unknown.get(typ, 0) + 1
        return typ, f

    # ── 표시 ─────────────────────────────────────────────────────────
    def bar(self, width=28):
        """마지막 프레임의 로짓을 막대로. 마진 기준선을 함께 그린다."""
        f = self.last_frame
        if not f or "l" not in f:
            return ["(프레임 없음)"]
        try:
            lg = [int(v) for v in f["l"].split(",")]
        except ValueError:
            return ["(로짓 파싱 실패)"]
        bg = lg[-1]
        lo, hi = min(lg), max(lg)
        span = max(hi - lo, 1)
        out = []
        for i, v in enumerate(lg):
            n = int(round((v - lo) / span * width))
            mark = "*" if cls_name(i) != "background" and v > bg else " "
            out.append(f"  {mark}{cls_name(i):<11}{v:>7}  "
                       + "#" * n + "." * (width - n))
        out.append(f"   {'margin':<11}{f.get('margin','?'):>7}   "
                   f"(> thr 이면 검출)")
        return out

    def summary(self):
        b, s = self.banner, self.last_status
        up = int(s.get("up", 0)) if s else 0
        lat = sorted(self.lat_us)
        lat_s = (f"{lat[len(lat)//2]}/{lat[int(len(lat)*0.99)]}/{lat[-1]} us"
                 if lat else "—")
        return [
            f"  펌웨어   {b.get('fw','?')}  빌드 {b.get('build','?')}  "
            f"모델 {b.get('model','?')}",
            f"  경로     {s.get('path', b.get('path','?'))}   "
            f"thr {b.get('thr','?')}   hop {b.get('hop_ms','?')}ms",
            f"  가동     {up/1000:,.1f}s   프레임 {self.frames:,}   "
            f"드롭 {self.drops}   seq 유실 {self.seq_gaps}",
            f"  지연     중앙/p99/최악  {lat_s}",
            f"  배터리   {s.get('vbat','?')}mV  {s.get('soc','?')}%",
            f"  줄 오류  {self.bad_lines}"
            + (f"   모르는 TYPE {self.unknown}" if self.unknown else ""),
        ]

    def recent_events(self, n=8):
        out = []
        for e in self.events[-n:]:
            c = cls_name(e.get("cls"))
            flag = "!!" if c in URGENT else "  "
            out.append(f"  {flag} {int(e.get('t',0))/1000:>9.2f}s  {c:<10}"
                       f"conf {e.get('conf','?'):>3}%  "
                       f"{int(e.get('dur_ms',0)):>5}ms  "
                       f"n={e.get('n','?'):<3} peak={e.get('peak','?')}")
        return out or ["  (아직 없음)"]


def render(m):
    lines = ["=" * 62, " MAX78000FTHR 모니터", "=" * 62]
    lines += m.summary()
    lines += ["", " 마지막 프레임 로짓"] + m.bar()
    lines += ["", f" 최근 이벤트 (총 {len(m.events)}개)"] + m.recent_events()
    return "\n".join(lines)


def self_test():
    """fake_serial 출력을 그대로 먹여 파서·상태 모델을 검증한다."""
    from fake_serial import FakeBoard, pkt

    m = Monitor()
    b = FakeBoard(seed=1, event_rate_per_min=30, drop_rate=0.02)
    for ln in b.lines(duration_s=600):
        m.feed(ln)
    # 깨진 줄과 모르는 TYPE 을 섞어도 죽지 않아야 한다 (부트로더 출력 등)
    for junk in ("", "garbage", "F|seq=1|ZZ", "X|foo=1|00",
                 "\x00\xff binary noise", "B|fw=1|"):
        m.feed(junk)

    # 모르는 TYPE 은 **체크섬이 맞아야** 그렇게 분류된다 — 체크섬을 먼저
    # 보기 때문이다. 틀린 체크섬으로 만들면 그냥 깨진 줄로 센다
    # (위 junk 목록의 "X|foo=1|00" 이 그 경우다).
    m.feed(pkt("X|foo=1|"))

    print(render(m))
    print()
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'OK  ' if cond else 'FAIL'}  {name}"
              + (f"   {detail}" if detail and not cond else ""))

    chk("프레임을 받았다", m.frames > 2000, f"frames={m.frames}")
    chk("이벤트를 받았다", len(m.events) > 10, f"n={len(m.events)}")
    chk("배너를 받았다", m.banner.get("fw") is not None)
    chk("드롭을 집계했다", m.drops > 0, f"drops={m.drops}")
    chk("드롭이 seq 유실로도 잡힌다", m.seq_gaps >= m.drops,
        f"gaps={m.seq_gaps} drops={m.drops}")
    chk("깨진 줄을 세고 죽지 않는다", m.bad_lines >= 4,
        f"bad={m.bad_lines}")
    chk("모르는 TYPE 을 버리고 센다", m.unknown.get("X") == 1,
        f"unknown={m.unknown}")
    chk("지연 통계가 있다", len(m.lat_us) == m.frames)
    print()
    print("전부 통과" if ok else "실패")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None,
                    help="시리얼 포트 (COM7 / /dev/ttyACM0). 없으면 표준입력")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--log", default=None, help="이벤트를 SD 스키마 CSV 로 저장")
    ap.add_argument("--raw", default=None, help="원본 줄을 그대로 저장")
    ap.add_argument("--interval", type=float, default=0.5, help="화면 갱신 주기")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        sys.exit(self_test())

    if a.port:
        try:
            import serial
        except ImportError:
            sys.exit("pyserial 이 필요하다:  py -m pip install pyserial")
        src = serial.Serial(a.port, a.baud, timeout=1)
        reader = (lambda: src.readline().decode("ascii", "replace"))
    else:
        reader = sys.stdin.readline

    m = Monitor()
    raw = open(a.raw, "w", encoding="utf-8") if a.raw else None
    logf = None
    if a.log:
        new = not os.path.exists(a.log)
        logf = open(a.log, "a", newline="", encoding="utf-8")
        if new:
            logf.write("timestamp,class,confidence,duration\n")

    n_logged = 0
    last_draw = 0.0
    try:
        while True:
            line = reader()
            if not line:
                if a.port:
                    continue
                break                       # 표준입력 끝
            if raw:
                raw.write(line if line.endswith("\n") else line + "\n")
            r = m.feed(line)
            if r and r[0] == "E" and logf:
                e = r[1]
                # timestamp 는 시각 동기화 전이면 상대값이다 (프로토콜 2절)
                logf.write(f"+{e.get('t',0)},{cls_name(e.get('cls'))},"
                           f"{e.get('conf',0)},{e.get('dur_ms',0)}\n")
                logf.flush()
                n_logged += 1
            now = time.time()
            if now - last_draw >= a.interval:
                last_draw = now
                sys.stdout.write("\x1b[2J\x1b[H" + render(m)
                                 + (f"\n\n 기록 {n_logged}개 -> {a.log}"
                                    if logf else "") + "\n")
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        for f in (raw, logf):
            if f:
                f.close()
        print("\n" + render(m))


if __name__ == "__main__":
    main()
