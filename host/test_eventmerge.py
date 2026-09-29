#!/usr/bin/env python3
"""이벤트 병합 상태머신 단위 테스트.

`docs/uart-protocol.md` 3.4 가 고정한 7가지를 그대로 검증한다.
**펌웨어 C 이식본도 이 표를 통과해야 한다** — 같은 입력에 같은 출력이
나오지 않으면 보드와 논문이 다른 시스템을 말하게 된다.

실행 (WSL2, 의존성 없음):
    python3 host/test_eventmerge.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eventmerge import EventMerger  # noqa: E402

BG = 4
SIREN, GLASS, SCREAM, DOG = 0, 1, 2, 3
HOP = 250
THR = 10.0


def lg(cls=None, margin=20):
    """로짓 5개를 만든다. cls=None 이면 배경음(마진 음수)."""
    v = [0, 0, 0, 0, 0]
    v[BG] = 100
    if cls is None:
        return v                       # 마진 = 0 - 100 = -100 < THR
    v[cls] = 100 + margin
    return v


def run(frames, **kw):
    """frames 는 (t_ms, cls|None, margin) 리스트. 이벤트 리스트를 돌려준다."""
    m = EventMerger(n_classes=5, bg=BG, thr=THR, hop_ms=HOP, **kw)
    out = []
    t_last = 0
    for t, c, mg in frames:
        out += m.push(t, lg(c, mg))
        t_last = t
    out += m.flush(t_last)
    return out


def seq(start, n, cls, margin=20, hop=HOP):
    return [(start + i * hop, cls, margin) for i in range(n)]


FAILED = []


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {name}"
          + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


print("이벤트 병합 상태머신 테스트")

# 1. 연속 12프레임 사이렌 -> 이벤트 1개, dur = 11*250 + 250 = 3000
ev = run(seq(1000, 12, SIREN))
check("1. 연속 12프레임 -> 이벤트 1개", len(ev) == 1, f"len={len(ev)}")
if ev:
    check("1. dur_ms = 3000", ev[0].dur_ms == 3000, f"dur={ev[0].dur_ms}")
    check("1. t_start = 1000", ev[0].t_start_ms == 1000)
    check("1. n_frames = 12", ev[0].n_frames == 12, f"n={ev[0].n_frames}")
    check("1. cls = siren", ev[0].cls == SIREN)

# 2. 사이 1프레임 끊김 -> 여전히 1개 (gap 250ms < 2000ms)
f = seq(1000, 5, SIREN) + [(2250, None, 0)] + seq(2500, 5, SIREN)
ev = run(f)
check("2. 1프레임 끊김 -> 이벤트 1개", len(ev) == 1, f"len={len(ev)}")
if ev:
    check("2. n_frames = 10 (gap 프레임 제외)", ev[0].n_frames == 10,
          f"n={ev[0].n_frames}")

# 3. GAP_MS 이상 끊김 -> 2개.  마지막 검출 1000+4*250=2000,
#    2000+2000=4000 이상에서 닫힌다 -> 다음 검출을 4000 에 둔다
f = seq(1000, 5, SIREN) + seq(4000, 5, SIREN)
ev = run(f)
check("3. 2초 이상 끊김 -> 이벤트 2개", len(ev) == 2, f"len={len(ev)}")
if len(ev) == 2:
    check("3. 첫 이벤트 dur = 1250", ev[0].dur_ms == 1250, f"{ev[0].dur_ms}")
    check("3. 둘째 t_start = 4000", ev[1].t_start_ms == 4000)

# 4. 사이렌 중 1프레임 dog_bark -> 사이렌 1개 (쪼개지지 않는다)
f = seq(1000, 5, SIREN) + [(2250, DOG, 20)] + seq(2500, 5, SIREN)
ev = run(f)
check("4. 중간에 다른 클래스 1프레임 -> 이벤트 1개", len(ev) == 1,
      f"len={len(ev)}")
if ev:
    check("4. cls = siren (다수결)", ev[0].cls == SIREN, f"cls={ev[0].cls}")

# 5. 사이렌 -> dog_bark 로 완전 전환 -> 2개, 시각이 겹치지 않는다
f = seq(1000, 6, SIREN) + seq(2500, 12, DOG)
ev = run(f)
check("5. 클래스 전환 -> 이벤트 2개", len(ev) == 2, f"len={len(ev)}")
if len(ev) == 2:
    check("5. 클래스가 siren, dog_bark", ev[0].cls == SIREN and ev[1].cls == DOG,
          f"{ev[0].cls},{ev[1].cls}")
    check("5. 시각이 겹치지 않는다",
          ev[1].t_start_ms >= ev[0].t_start_ms + ev[0].dur_ms,
          f"{ev[0].t_start_ms}+{ev[0].dur_ms} vs {ev[1].t_start_ms}")

# 6. 프레임 드롭(시각 점프) -> **시각 기준**으로 끊긴다.
#    프레임 수로 셌다면 gap 프레임이 0개라 안 끊겼을 상황이다.
f = seq(1000, 4, SIREN) + [(1750 + 3000, SIREN, 20)]
ev = run(f)
check("6. 시각 점프 3초 -> 이벤트 2개 (프레임 수가 아니라 시각으로 센다)",
      len(ev) == 2, f"len={len(ev)}")

# 7. THR 미만만 -> 이벤트 0개
ev = run([(1000 + i * HOP, SIREN, 5) for i in range(10)])   # margin 5 < THR 10
check("7. 문턱값 미만 -> 이벤트 0개", len(ev) == 0, f"len={len(ev)}")

# 8. peak_margin 이 최대값을 잡는가 (표시용 conf 와 독립)
ev = run([(1000, SIREN, 12), (1250, SIREN, 45), (1500, SIREN, 20)])
check("8. peak_margin = 45", ev and ev[0].peak_margin == 45,
      f"{ev[0].peak_margin if ev else '없음'}")

# 9. START_FRAMES=2 면 단발 검출은 이벤트가 되지 않는다
ev = run([(1000, SIREN, 20), (1250, None, 0)] + seq(4000, 4, SIREN),
         start_frames=2)
check("9. START_FRAMES=2 에서 단발 검출 무시 -> 이벤트 1개", len(ev) == 1,
      f"len={len(ev)}")

print()
if FAILED:
    print(f"실패 {len(FAILED)}건: {', '.join(FAILED)}")
    sys.exit(1)
print("전부 통과")
