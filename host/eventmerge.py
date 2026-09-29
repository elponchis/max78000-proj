#!/usr/bin/env python3
"""이벤트 병합 상태머신 — **펌웨어 C 이식의 기준 구현**.

초당 4회 추론이므로 3초짜리 사이렌이 12개 레코드로 쪼개진다. 연속 동일 클래스를
하나로 묶고, 일정 시간 이상 끊기면 종료 처리한다 (CLAUDE.md 6장).

규칙과 근거는 `docs/uart-protocol.md` 3절에 있다. 요약:

  det(frame) = argmax_{c != bg} logit[c]   if  margin > THR
             = None                        otherwise
  margin     = max_{c != bg} logit[c] - logit[bg]

  ⚠️ 판정에 **softmax 를 쓰지 않는다.** 오프라인 평가(`tools/eval_threshold.py`)
     와 같은 기준을 써야 보드의 동작점과 논문 표의 동작점이 같아진다. FC 가
     wide=True 라 NPU 가 로짓을 그대로 주므로 뺄셈 한 번이면 된다.

── C 이식을 위한 제약 (일부러 지킨다) ──────────────────────────────────
  · **정수만 쓴다.** 부동소수는 `conf`(표시용 softmax %) 하나뿐이고 그것도
    바깥에서 계산해 넣는다 — `MEASURE_BUILD` 에서 softmax 를 빼도 판정이
    바뀌지 않아야 한다
  · 동적 할당이 없다. 상태는 고정 크기 구조체 하나다
  · 이벤트는 닫히는 순간 콜백으로 **하나씩** 나간다 (리스트에 쌓지 않는다)
  · 표준 라이브러리 호출이 없다
"""

from dataclasses import dataclass, field

# ── 기본 상수. 전부 CLAUDE.md / uart-protocol.md 에서 왔다 ────────────────
GAP_MS_DEFAULT = 2000      # 이 시간 이상 끊기면 종료 (CLAUDE.md 6장 "약 2초")
START_FRAMES_DEFAULT = 1   # glass 가 0.3초 과도음이라 1 이 기본. G7 후 조정
HOP_MS_DEFAULT = 250


@dataclass
class Event:
    """방출되는 병합 이벤트. SD 레코드와 `E` 패킷의 원본이다."""

    t_start_ms: int
    cls: int
    dur_ms: int
    n_frames: int
    peak_margin: int
    conf: int = 0                       # 표시용 softmax %, 없으면 0
    counts: tuple = ()                  # 클래스별 검출 프레임 수 (진단용)


@dataclass
class _Active:
    """진행 중인 이벤트의 누적 상태. C 구조체 하나에 대응한다."""

    t_start_ms: int
    t_last_det_ms: int
    peak_margin: int
    conf: int
    counts: list = field(default_factory=list)


class EventMerger:
    """프레임을 먹이면 병합된 이벤트를 돌려준다.

    사용:
        m = EventMerger(n_classes=5, bg=4, thr=16.85)
        for t_ms, logits in stream:
            for ev in m.push(t_ms, logits):
                ...
        for ev in m.flush(t_ms_last):      # 스트림 끝에서 열린 이벤트 닫기
            ...

    `thr` 은 오프라인에서 고정 오경보 지점으로 정한 값이다. 정수 로짓과
    비교하므로 내부에서는 **부동소수 비교 한 번**만 한다 — C 로 옮길 때는
    문턱값을 정수로 반올림해 두면 부동소수가 완전히 사라진다.
    """

    def __init__(self, n_classes=5, bg=4, thr=0.0, hop_ms=HOP_MS_DEFAULT,
                 gap_ms=GAP_MS_DEFAULT, start_frames=START_FRAMES_DEFAULT):
        self.n = n_classes
        self.bg = bg
        self.thr = thr
        self.hop_ms = hop_ms
        self.gap_ms = gap_ms
        self.start_frames = start_frames
        self.cur = None            # _Active 또는 None
        self.pend_cls = -1         # 시작 대기 중인 클래스
        self.pend_n = 0            # 그 클래스가 연속으로 나온 프레임 수
        self.pend_t0 = 0
        self.pend_peak = 0
        self.pend_conf = 0

    # ── 검출 ──────────────────────────────────────────────────────────
    def detect(self, logits):
        """(cls, margin). 미검출이면 (-1, margin)."""
        best_c, best_v = -1, None
        for c in range(self.n):
            if c == self.bg:
                continue
            if best_v is None or logits[c] > best_v:
                best_c, best_v = c, logits[c]
        margin = best_v - logits[self.bg]
        return (best_c if margin > self.thr else -1), margin

    # ── 프레임 입력 ───────────────────────────────────────────────────
    def push(self, t_ms, logits, conf=0):
        """프레임 하나. 이번 프레임에서 **닫힌** 이벤트들을 돌려준다.

        `conf` 는 표시용 softmax 확률(%)이며 판정에 쓰이지 않는다.
        """
        out = []
        cls, margin = self.detect(logits)

        # 1) 진행 중인 이벤트의 종료 판정 — **시각 기준**이다.
        #    프레임 수로 세면 드롭이 있을 때 이벤트가 더 길게 이어져,
        #    G1(프레임 드롭 0)과 G7 이 서로를 오염시킨다.
        if self.cur is not None and t_ms - self.cur.t_last_det_ms >= self.gap_ms:
            out.append(self._close())

        # 2) 미검출이면 여기서 끝. 시작 대기는 **연속**이어야 하므로 리셋한다
        if cls < 0:
            self.pend_cls, self.pend_n = -1, 0
            return out

        # 3) 진행 중인 이벤트와 같은 클래스면 연장
        if self.cur is not None and self._cur_cls() == cls:
            self._extend(t_ms, cls, margin, conf)
            self.pend_cls, self.pend_n = -1, 0
            return out

        # 4) 진행 중인 이벤트가 있는데 **다른** 이벤트 클래스가 나왔다.
        #    그 프레임은 현재 이벤트의 gap 으로 두고(연장하지 않는다),
        #    새 클래스를 후보로 쌓는다. 사이렌 중간에 개 짖는 소리가 한 프레임
        #    섞였다고 사이렌이 둘로 쪼개지지 않게 하면서, 실제로 소리가 바뀌면
        #    놓치지 않기 위해서다.
        if self.cur is not None:
            # 진행 이벤트에는 **세지 않는다** — 그래야 gap 이 시각으로 자라고
            # 다수결 클래스도 흔들리지 않는다.
            self._bump_pending(t_ms, cls, margin, conf)
            return out

        # 5) 진행 중인 이벤트가 없다 — 시작 대기
        self._bump_pending(t_ms, cls, margin, conf)
        if self.pend_n >= self.start_frames:
            self.cur = _Active(self.pend_t0, t_ms, self.pend_peak,
                               self.pend_conf, [0] * self.n)
            self.cur.counts[cls] = self.pend_n
            self.pend_cls, self.pend_n = -1, 0
        return out

    def flush(self, t_ms=None):
        """스트림 끝. 열려 있는 이벤트를 닫아 돌려준다."""
        if self.cur is None:
            return []
        return [self._close()]

    # ── 내부 ──────────────────────────────────────────────────────────
    def _bump_pending(self, t_ms, cls, margin, conf):
        if cls != self.pend_cls:
            self.pend_cls, self.pend_n = cls, 0
            self.pend_t0, self.pend_peak, self.pend_conf = t_ms, margin, conf
        self.pend_n += 1
        if margin > self.pend_peak:
            self.pend_peak = margin
        if conf > self.pend_conf:
            self.pend_conf = conf

    def _extend(self, t_ms, cls, margin, conf):
        self.cur.t_last_det_ms = t_ms
        self.cur.counts[cls] += 1
        if margin > self.cur.peak_margin:
            self.cur.peak_margin = margin
        if conf > self.cur.conf:
            self.cur.conf = conf

    def _cur_cls(self):
        """검출 프레임이 가장 많았던 클래스. 동수면 먼저 나온 쪽이다."""
        best_c, best_n = -1, -1
        for c in range(self.n):
            if self.cur.counts[c] > best_n:
                best_c, best_n = c, self.cur.counts[c]
        return best_c

    def _close(self):
        a = self.cur
        self.cur = None
        # 마지막 프레임도 1초 창을 본 것이므로 hop 을 더한다. 안 더하면
        # 이벤트가 체계적으로 짧게 기록되고, hop 스윕(G10)에서 지속시간이
        # hop 에 따라 달라 보인다.
        dur = a.t_last_det_ms - a.t_start_ms + self.hop_ms
        return Event(t_start_ms=a.t_start_ms,
                     cls=max(range(self.n), key=lambda c: a.counts[c]),
                     dur_ms=dur,
                     n_frames=sum(a.counts),
                     peak_margin=a.peak_margin,
                     conf=a.conf,
                     counts=tuple(a.counts))
