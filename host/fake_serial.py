#!/usr/bin/env python3
"""보드 없이 GUI 를 만들기 위한 더미 이벤트 스트림 생성기.

`docs/uart-protocol.md` 의 패킷을 **그대로** 뱉는다. GUI 는 이 출력만 보고
완성할 수 있고, 보드가 붙으면 입력 소스만 pyserial 로 바꾸면 된다.

⚠️ **실기기 대신 쓰는 것이 아니다.** G4(온디바이스 실측 vs PC 시뮬레이션 괴리)
는 실측이고, 이 파일은 그 실험에 등장하지 않는다. 용도는 하나 —
보드가 없거나 점유 중일 때 호스트 코드를 진행시키는 것.

사용 (Windows 네이티브 / WSL2 어디서든, 의존성 없음):

    # 표준출력으로 실시간 스트림
    python host/fake_serial.py

    # 60초치를 최대 속도로 (GUI 회귀 테스트용)
    python host/fake_serial.py --duration 60 --no-realtime

    # 가상 시리얼 포트에 물리기 (리눅스)
    socat -d -d pty,raw,echo=0 pty,raw,echo=0
    python host/fake_serial.py > /dev/pts/N

    # 파이썬에서 직접
    from fake_serial import FakeBoard
    for line in FakeBoard().lines():
        ...
"""

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eventmerge import EventMerger  # noqa: E402

CLASSES = ["siren", "glass", "scream", "dog_bark", "background"]
BG = 4
FW = "0.1.0"
MODEL = "safesound-fb"

# 클래스별 전형적 지속시간(ms). `docs/results/class-mapping.md` 의 클립 성격에서
# 왔다 — glass 는 0.3초 과도음, siren 은 수 초 이어진다. GUI 가 현실적인 길이의
# 이벤트를 받아야 타임라인 레이아웃을 제대로 잡는다.
DUR_MS = {0: (2000, 8000), 1: (250, 700), 2: (500, 2500), 3: (300, 1500)}

# 알림 등급 (CLAUDE.md 4장). GUI 색·부저 매핑 확인용으로 함께 내보낸다.
URGENT = {0, 1, 2}


def checksum(body):
    """줄 시작부터 마지막 '|' 직전까지의 XOR, 2자리 대문자 16진수."""
    c = 0
    for b in body.encode("ascii"):
        c ^= b
    return f"{c:02X}"


def pkt(body):
    """'TYPE|...|' 에 체크섬을 붙여 완성한다."""
    return f"{body}{checksum(body)}"


class FakeBoard:
    """그럴듯한 프레임·이벤트 스트림을 만든다.

    실제 보드와 **같은 상태머신**(`eventmerge.EventMerger`)을 통과시킨다.
    더미가 자기 마음대로 이벤트를 만들면 GUI 가 실기기에서 다르게 동작한다.
    """

    def __init__(self, seed=0, hop_ms=250, thr=10.0, path="NPU",
                 event_rate_per_min=6.0, drop_rate=0.0):
        self.rng = random.Random(seed)
        self.hop = hop_ms
        self.thr = thr
        self.path = path
        self.drop_rate = drop_rate
        # 프레임당 이벤트 시작 확률
        self.p_start = event_rate_per_min / 60.0 * (hop_ms / 1000.0)
        self.merger = EventMerger(n_classes=5, bg=BG, thr=thr, hop_ms=hop_ms)
        self.seq = 0
        self.t = 0
        self.frames = 0
        self.drops = 0
        self.active = None          # (cls, 남은 프레임 수)
        self.vbat = 4100

    # ── 로짓 생성 ─────────────────────────────────────────────────────
    def _logits(self):
        """배경음 기본, 이벤트 중이면 해당 클래스를 띄운다.

        마진 분포를 실측(`docs/results/g8-first-comparison.md`)과 비슷한
        모양으로 둔다 — 배경음 마진은 대부분 음수이고 꼬리가 길다.
        """
        lg = [int(self.rng.gauss(0, 18)) for _ in range(5)]
        lg[BG] = 100 + int(self.rng.gauss(0, 12))
        if self.active is not None:
            c, left = self.active
            # 이벤트 중에는 마진이 양수. 시작·끝에서 약해지게 살짝 기울인다
            peak = self.rng.uniform(1.2, 2.5) * self.thr
            lg[c] = lg[BG] + int(peak * (0.6 + 0.4 * self.rng.random()))
            self.active = (c, left - 1)
            if left - 1 <= 0:
                self.active = None
        elif self.rng.random() < self.p_start:
            c = self.rng.choice([0, 1, 2, 3])
            lo, hi = DUR_MS[c]
            n = max(1, int(self.rng.uniform(lo, hi) / self.hop))
            self.active = (c, n)
        return lg

    @staticmethod
    def _softmax_pct(lg, c):
        """표시용 확신도. **판정에 쓰이지 않는다** (판정은 마진이다)."""
        m = max(lg)
        e = [pow(2.718281828, (v - m) / 24.0) for v in lg]   # T=24, 로짓 스케일
        return int(round(100.0 * e[c] / sum(e)))

    # ── 스트림 ────────────────────────────────────────────────────────
    def banner(self):
        return pkt(f"B|fw={FW}|build=DEMO|model={MODEL}|path={self.path}"
                   f"|thr={self.thr:.2f}|hop_ms={self.hop}|")

    def lines(self, duration_s=None):
        """패킷 줄을 순서대로 낳는다 (개행 없음)."""
        yield self.banner()
        last_status = -1000
        end = None if duration_s is None else duration_s * 1000
        while end is None or self.t < end:
            self.t += self.hop
            self.seq += 1

            # 프레임 드롭 모의 — GUI 의 드롭 카운터와 상태머신의 시각 기준
            # gap 처리를 함께 시험한다
            if self.rng.random() < self.drop_rate:
                self.drops += 1
                continue
            self.frames += 1

            lg = self._logits()
            cls, margin = self.merger.detect(lg)
            conf = self._softmax_pct(lg, cls if cls >= 0 else BG)
            yield pkt(f"F|seq={self.seq}|t_ms={self.t}"
                      f"|l={','.join(str(v) for v in lg)}"
                      f"|margin={margin}|cls={cls if cls >= 0 else BG}"
                      f"|lat_us={self.rng.randint(300, 340)}"
                      f"|path={self.path}|")

            for ev in self.merger.push(self.t, lg, conf):
                yield pkt(f"E|t={ev.t_start_ms}|cls={ev.cls}|conf={ev.conf}"
                          f"|dur_ms={ev.dur_ms}|n={ev.n_frames}"
                          f"|peak={ev.peak_margin}|")

            if self.t - last_status >= 1000:
                last_status = self.t
                # 4시간에 4.20V -> 3.50V 로 선형 하강 (모양만 그럴듯하게)
                self.vbat = max(3500, 4100 - self.t // 20000)
                soc = max(0, min(100, (self.vbat - 3500) * 100 // 600))
                yield pkt(f"S|up={self.t}|frames={self.frames}"
                          f"|drops={self.drops}|path={self.path}"
                          f"|vbat={self.vbat}|soc={soc}|")

        for ev in self.merger.flush(self.t):
            yield pkt(f"E|t={ev.t_start_ms}|cls={ev.cls}|conf={ev.conf}"
                      f"|dur_ms={ev.dur_ms}|n={ev.n_frames}"
                      f"|peak={ev.peak_margin}|")


def parse(line):
    """패킷 한 줄 -> (type, dict) 또는 체크섬 불일치·형식 오류면 None.

    **GUI 와 보드가 같은 파서를 쓴다.** 모르는 TYPE 과 깨진 줄을 조용히 버리는
    것이 규약이다 (부팅 중 부트로더 출력이 섞여 들어온다).
    """
    line = line.strip()
    if len(line) < 4 or "|" not in line:
        return None
    body, ck = line[:-2], line[-2:]
    if checksum(body) != ck.upper():
        return None
    parts = body.split("|")
    typ, fields = parts[0], {}
    for p in parts[1:]:
        if not p:
            continue
        k, _, v = p.partition("=")
        fields[k] = v
    return typ, fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hop-ms", type=int, default=250)
    ap.add_argument("--thr", type=float, default=10.0)
    ap.add_argument("--path", default="NPU", choices=("NPU", "CPU"))
    ap.add_argument("--rate", type=float, default=6.0,
                    help="분당 이벤트 수")
    ap.add_argument("--drop-rate", type=float, default=0.0,
                    help="프레임 드롭 확률 (GUI 카운터 시험용)")
    ap.add_argument("--duration", type=float, default=None,
                    help="모의 시간(초). 없으면 무한")
    ap.add_argument("--no-realtime", action="store_true",
                    help="실시간 대기 없이 최대 속도로 뱉는다")
    ap.add_argument("--self-test", action="store_true",
                    help="생성 -> 파싱 왕복을 검사하고 끝낸다")
    a = ap.parse_args()

    b = FakeBoard(a.seed, a.hop_ms, a.thr, a.path, a.rate, a.drop_rate)

    if a.self_test:
        n, bad, kinds = 0, 0, {}
        for ln in b.lines(duration_s=600):
            n += 1
            r = parse(ln)
            if r is None:
                bad += 1
                print("파싱 실패:", ln)
            else:
                kinds[r[0]] = kinds.get(r[0], 0) + 1
        # 한 글자 깨뜨려 체크섬이 실제로 잡는지 확인한다
        broken = b.banner()
        broken = broken[:5] + ("X" if broken[5] != "X" else "Y") + broken[6:]
        caught = parse(broken) is None
        print(f"600초 모의: {n:,}줄, 파싱 실패 {bad}, 종류 {kinds}")
        print(f"체크섬이 손상된 줄을 잡는가: {'예' if caught else '아니오'}")
        sys.exit(0 if bad == 0 and caught else 1)

    try:
        for ln in b.lines(duration_s=a.duration):
            sys.stdout.write(ln + "\n")
            sys.stdout.flush()
            if not a.no_realtime and ln.startswith("F|"):
                time.sleep(a.hop_ms / 1000.0)
    except (BrokenPipeError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
