#!/usr/bin/env python3
"""이벤트 단위 평가 — 창 단위 @300/h 가 아니라 **실사용 지표**.

창 단위 macro-F1 은 후처리 이전의 값이라 실사용 성능이 아니다. 설계한 후처리
(연속 프레임 다수결 + 이벤트 병합 + 쿨다운, `docs/uart-protocol.md` 3절)를
거친 뒤의 이벤트 recall 과 **회/일** 오경보를 낸다. 재학습은 없다.

── ★ 왜 테스트셋 창을 쓰지 않고 오디오를 다시 훑는가 ─────────────────────
테스트셋의 창은 클립당 점수 순으로 골라 뽑은 것이라 **시간적으로 떨어져
있다** (같은 클립 창 간격 중앙값 2.0초, hop 0.25초와 붙은 쌍은 0%).
그 위에서 k/m 다수결을 돌리면 떨어진 창들의 오류가 서로 독립에 가까워
**다수결이 실제보다 잘 듣는다 — 항상 낙관적으로 틀린다**
(`tools/eval_confusion.py` 의 경고, `stream_eval.py` docstring).

그래서 여기서는 **원본 오디오를 hop 간격으로 빈틈없이** 훑는다. 실기기가
상시 구동에서 보는 것과 같은 프레임 열이고, 인접 프레임이 0.75초를 겹쳐
오류가 함께 간다.

── 두 단계 ──────────────────────────────────────────────────────────────
  --scan  : 오디오를 훑어 **로짓**을 클립별로 캐시한다 (모델당 한 번)
  기본     : 캐시 위에서 문턱값·(m/k)·쿨다운 격자를 돌린다 (빠르다)

⚠️ **격자 최적값을 테스트셋에서 고르면 과적합**이다. `--select` 로 train
   일부(검증용)에서 고르고, 테스트는 그 설정으로 **한 번만** 잰다.

사용 (WSL2):
    # 1) 캐시 (ai8x venv)
    ~/ai8x-training/venv/bin/python tools/event_eval.py --scan \\
        --checkpoint <ckpt> --config mel --split test --tag mel-s1
    # 2) 평가 (numpy 만)
    python3 tools/event_eval.py --tag mel-s1 --split test --mk 3/5 --cooldown 2
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI8X = os.environ.get("AI8X_DIR") or os.path.expanduser("~/ai8x-training")
sys.path[:0] = [AI8X, os.path.join(REPO, "scripts"),
                os.path.join(REPO, "datasets"), os.path.join(REPO, "tools")]

CLASSES = ["siren", "glass", "scream", "dog_bark", "background"]
EVENTS = CLASSES[:4]
BG = 4
CACHE = os.path.join(REPO, "data", "eventcache")


# ─────────────────────────────────────────────────────────── 1) 스캔
def scan(a):
    import torch
    import ai8x
    import prepare_safesound as P
    from eval_confusion import load_model

    torch.set_num_threads(a.threads)
    hop = max(1, int(P.SR * a.hop_ms / 1000))
    roots = {"fsd": a.fsd_dir, "us8k": a.us8k_dir, "esc50": a.esc50_dir}
    rows = [r for r in csv.DictReader(open(a.manifest, encoding="utf-8"))
            if r["split"] == a.split]
    rows = [(r, P.clip_path(r, roots)) for r in rows]
    rows = [(r, p) for r, p in rows if p]
    if a.every > 1:
        # 격자 선택용 검증 부분집합. 매니페스트가 클래스 순으로 정렬돼 있어
        # N개마다 뽑으면 클래스 비율이 유지된다 (층화 추출과 같은 효과).
        rows = rows[::a.every]
    if a.limit:
        rows = rows[:a.limit]

    model = load_model(a.checkpoint, len(CLASSES), a.ai8x, False, a.bias,
                       config=a.config)
    if a.config == "mel":
        import melfeat as MFEAT
    norm = ai8x.normalize(args=argparse.Namespace(act_mode_8bit=False))

    print(f"{a.split} 원본 {len(rows)}개를 hop {a.hop_ms}ms 로 빈틈없이 훑는다")
    out, t0, nsec = {}, time.time(), 0.0
    for i, (r, path) in enumerate(rows, 1):
        x = P.load_audio(path)
        if x is None:
            continue
        if len(x) < P.WIN:
            continue          # 1초 미만 — 실기기도 프레임이 만들어지지 않는다
        starts = list(range(0, len(x) - P.WIN + 1, hop))
        nsec += len(x) / P.SR
        lg = []
        for b in range(0, len(starts), a.batch_size):
            ch = starts[b:b + a.batch_size]
            q = np.stack([P.to_int8(x[s:s + P.WIN]) for s in ch])
            if a.config == "mel":
                q = np.stack([MFEAT.log_mel_int8(w) for w in q])
                t = (torch.from_numpy(q.astype(np.int16)) + 128).float() / 256.0
                t = t.unsqueeze(1)
            else:
                t = (torch.from_numpy(q.astype(np.int16)) + 128).float() / 256.0
                t = torch.transpose(t.reshape(len(ch), -1, 128), 2, 1)
            with torch.no_grad():
                lg.append(model(norm(t)).numpy().astype(np.float16))
        out[r["clip_id"]] = (r["fsid"], r["cls"], np.concatenate(lg),
                             len(x) / P.SR)
        if i % 200 == 0:
            print(f"  {i}/{len(rows)}  오디오 {nsec/3600:.2f}h  "
                  f"{time.time()-t0:.0f}s", flush=True)

    os.makedirs(CACHE, exist_ok=True)
    sfx = a.split + (f"every{a.every}" if a.every > 1 else "")
    p = os.path.join(CACHE, f"{a.tag}-{sfx}-hop{a.hop_ms}.npz")
    np.savez_compressed(
        p,
        clip=np.array(list(out)),
        fsid=np.array([v[0] for v in out.values()]),
        cls=np.array([v[1] for v in out.values()]),
        dur=np.array([v[3] for v in out.values()], dtype=np.float64),
        lens=np.array([len(v[2]) for v in out.values()]),
        logits=np.concatenate([v[2] for v in out.values()]),
        hop_ms=a.hop_ms)
    print(f"저장 {p}  ({nsec/3600:.2f}시간, "
          f"{sum(len(v[2]) for v in out.values()):,}프레임, "
          f"{time.time()-t0:.0f}s)")


# ─────────────────────────────────────────────── 2) 후처리와 지표
def load_cache(tag, split, hop_ms):
    p = os.path.join(CACHE, f"{tag}-{split}-hop{hop_ms}.npz")
    if not os.path.isfile(p):
        sys.exit(f"[에러] 캐시가 없다: {p}  (먼저 --scan)")
    z = np.load(p, allow_pickle=True)
    off, per = 0, []
    for c, f, cl, d, n in zip(z["clip"], z["fsid"], z["cls"], z["dur"],
                              z["lens"]):
        per.append((str(c), str(f), str(cl), float(d),
                    z["logits"][off:off + int(n)].astype(np.float32)))
        off += int(n)
    return per



def load_exclude(path):
    """제외할 fsid 목록 (한 줄에 하나, # 주석 허용). 없으면 빈 집합."""
    if not path:
        return set()
    out = set()
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.split("#")[0].strip()
            if ln:
                out.add(ln)
    print(f"  제외 fsid {len(out)}개 ({path})")
    return out

def detect(lg, thr):
    """프레임별 검출 클래스 (미검출 -1). 창 단위와 **같은 마진 규칙**이다."""
    ev = lg[:, :BG]
    best = ev.argmax(1)
    marg = ev.max(1) - lg[:, BG]
    return np.where(marg > thr, best, -1)


def fire_events(det, m, k, cooldown_s, hop_s, per_class=None):
    """(m/k) 다수결 + 병합 + 쿨다운 → [(프레임 인덱스, 클래스), ...].

    규칙
      · 최근 k 프레임 중 같은 클래스가 m 개 이상이면 그 클래스가 **활성**
      · 활성이 이어지는 동안은 **하나의 이벤트** (병합)
      · 이벤트가 끝난 뒤 `cooldown_s` 동안 **같은 클래스**는 다시 못 켠다
    """
    n = len(det)
    # per_class: {클래스 인덱스: (m, k)}. 없으면 전 클래스 공통.
    # glass 처럼 0.3초 과도음인 클래스는 3/4 를 원리적으로 못 채운다
    # (원본의 28%가 프레임 2개 이하) — 그래서 클래스별 규칙이 필요하다.
    mk = per_class or {}
    out, active, last_end = [], {}, {}
    for i in range(n):
        on = -1
        for c in range(BG):
            mc, kc = mk.get(c, (m, k))
            win = det[max(0, i - kc + 1):i + 1]
            if int((win == c).sum()) >= mc:
                on = c                      # 동시 충족은 낮은 인덱스 우선
                break
        for c in list(active):
            if c != on:
                last_end[c] = i
                del active[c]
        if on >= 0 and on not in active:
            t = i * hop_s
            if t >= last_end.get(on, -1e9) * hop_s + cooldown_s:
                out.append((i, on))
                active[on] = i
            else:
                active[on] = i              # 쿨다운 중 — 발화 없이 활성만
    return out


def evaluate(per, thr, m, k, cooldown_s, hop_s, per_class=None):
    """이벤트 recall / 오경보 / 오분류 발화."""
    hit = {c: set() for c in range(BG)}      # 클래스별 정답 발화된 fsid
    orig = {c: set() for c in range(BG)}     # 클래스별 전체 이벤트 fsid
    wrong = 0                                # 이벤트 원본에서 다른 이벤트 발화
    fa = 0                                   # 배경 원본 발화
    bg_sec = 0.0
    for _clip, fsid, cls, dur, lg in per:
        ci = CLASSES.index(cls)
        evs = fire_events(detect(lg, thr), m, k, cooldown_s, hop_s,
                          per_class)
        if ci == BG:
            bg_sec += dur
            fa += len(evs)
            continue
        orig[ci].add(fsid)
        for _i, c in evs:
            if c == ci:
                hit[ci].add(fsid)
            else:
                wrong += 1
    rec = {CLASSES[c]: (len(hit[c]) / len(orig[c]) if orig[c] else float("nan"))
           for c in range(BG)}
    macro = float(np.nanmean(list(rec.values())))
    hours = bg_sec / 3600.0
    return {"recall": rec, "macro_recall": macro, "fa": fa,
            "fa_per_hour": fa / hours if hours else float("nan"),
            "fa_per_day": 24.0 * fa / hours if hours else float("nan"),
            "bg_hours": hours, "wrong_fire": wrong,
            "n_origin": {CLASSES[c]: len(orig[c]) for c in range(BG)}}


def thr_for_fa(per, target_per_hour, m, k, cd, hop_s, lo=-40.0,
               hi=60.0, per_class=None):
    """목표 오경보(회/h) 이하가 되는 **가장 낮은** 문턱값 (이분 탐색)."""
    for _ in range(40):
        mid = (lo + hi) / 2
        r = evaluate(per, mid, m, k, cd, hop_s, per_class)
        if r["fa_per_hour"] > target_per_hour:
            lo = mid
        else:
            hi = mid
    return hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--hop-ms", type=int, default=250)
    # 스캔용
    ap.add_argument("--checkpoint")
    ap.add_argument("--config", default="wave")
    ap.add_argument("--manifest", default="data/interim/manifest.csv")
    ap.add_argument("--fsd-dir", default="data/raw/FSD50K_clips")
    ap.add_argument("--us8k-dir", default="data/raw/US8K_audio")
    ap.add_argument("--esc50-dir", default="data/raw/ESC50_audio")
    ap.add_argument("--ai8x", default=AI8X)
    ap.add_argument("--bias", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--every", type=int, default=1,
                    help="N개마다 하나만 (train 검증 부분집합용)")
    # 평가용
    ap.add_argument("--mk", default=None, help="'m/k' 하나만 평가")
    ap.add_argument("--cooldown", type=float, default=None)
    ap.add_argument("--grid", action="store_true", help="격자 전체")
    ap.add_argument("--mk-per-class", default=None,
                    metavar="siren=3/4,glass=1/1,...",
                    help="클래스별 (m/k). 지정하지 않은 클래스는 --mk 를 쓴다")
    ap.add_argument("--targets", default="1,0.0417",
                    help="목표 오경보 회/h (0.0417 = 1회/일)")
    ap.add_argument("--exclude-fsid", default=None,
                    help="제외할 fsid 목록 파일. 청취에서 (b)"
                         "=라벨 누락 실제 이벤트로 판정된 배경 원본")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.scan:
        scan(a)
        return

    per = load_cache(a.tag, a.split, a.hop_ms)
    ex = load_exclude(a.exclude_fsid)
    if ex:
        n0 = len(per)
        # fsid 단위로 뺀다 — '이 원본은 배경음이 아니다' 라는
        # 판정과 일관되려면 같은 원본의 다른 구간도 빠져야 한다.
        per = [r for r in per if r[1] not in ex]
        print(f"  원본 {n0} -> {len(per)}")
    hop_s = a.hop_ms / 1000.0
    bg_h = sum(d for _c, _f, cl, d, _l in per if cl == "background") / 3600.0
    print(f"[{a.tag} / {a.split}] 원본 {len(per)}개, 배경음 {bg_h:.2f}시간")
    print(f"  ⚠️ 0회 관측 시 95% 상한 = 3/{bg_h:.2f}h = "
          f"{3/bg_h:.3f}회/h = **{24*3/bg_h:.1f}회/일** — 이보다 낮은 지점은")
    print("     이 테스트셋으로 **분해할 수 없다**.")
    print()

    pc = None
    if a.mk_per_class:
        pc = {}
        for part in a.mk_per_class.split(","):
            cname, _, v = part.partition("=")
            mm, kk2 = (int(x) for x in v.split("/"))
            pc[CLASSES.index(cname.strip())] = (mm, kk2)
        print("  클래스별 규칙: " + "  ".join(
            f"{CLASSES[c]} {v[0]}/{v[1]}" for c, v in sorted(pc.items())))
    grids = ([(1, 1), (2, 3), (3, 4), (3, 5)] if a.grid
             else [tuple(int(x) for x in a.mk.split("/"))])
    cds = [0.0, 2.0, 5.0] if a.grid else [a.cooldown or 0.0]
    targets = [float(x) for x in a.targets.split(",")]

    res = []
    for (m, k) in grids:
        for cd in cds:
            for tg in targets:
                thr = thr_for_fa(per, tg, m, k, cd, hop_s, per_class=pc)
                r = evaluate(per, thr, m, k, cd, hop_s, pc)
                r.update({"m": m, "k": k, "cooldown": cd, "target": tg,
                          "thr": thr})
                res.append(r)
                print(f"  m/k {m}/{k}  cd {cd:.0f}s  목표 {tg:g}/h  "
                      f"thr {thr:+.2f}  실제 {r['fa_per_hour']:.3f}/h "
                      f"({r['fa_per_day']:.1f}/일)  "
                      f"macro recall {r['macro_recall']:.4f}  "
                      + " ".join(f"{c[:4]} {100*r['recall'][c]:.0f}%"
                                 for c in EVENTS)
                      + f"  오분류발화 {r['wrong_fire']}")
    if a.json:
        json.dump(res, open(a.json, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"저장: {a.json}")


if __name__ == "__main__":
    main()
