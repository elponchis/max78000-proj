#!/usr/bin/env python3
"""매니페스트 → 1초 윈도우 추출 → int8 캐시.

TASKS.md Phase 2.3. 데이터로더(`datasets/safesound.py`)와 **분리**되어 있다
(CLAUDE.md 6장). 이 스크립트는 디스크에 캐시를 만들고, 데이터로더는 읽기만 한다.

── 메모리 설계 (CLAUDE.md 6장, 3회 OOM의 교훈) ────────────────────────────
전체를 메모리에 올리지 않는다. 처음부터 lazy 구조로 간다.
  - 클립 하나씩 읽어 윈도우를 뽑고 **즉시** 샤드에 append 한다
  - 출력은 클래스·split 별 `.npy` 샤드(기본 2048윈도우/샤드) + 인덱스 CSV.
    단일 거대 `.pt` 를 만들지 않는다
  - 진행 상태를 `progress.json` 에 남겨 중단 후 이어하기를 지원한다
샤드 하나는 2048 × 16384 int8 = 32MB 다. 데이터로더는 `np.memmap` 으로 필요한
샤드만 매핑한다.

── 에너지 필터 (CLAUDE.md 5장 규칙 2) ─────────────────────────────────────
⚠️ **절대 RMS 임계값 하나를 전 클래스에 적용하면 안 된다.**
glass 는 0.3초 과도음, siren 은 4초 지속음, dog_bark 는 짧은 반복음이다.
절대값 단일 기준은 glass 만 대량 폐기시키고 클래스별 편향을 만든다.

→ 두 기준을 **병용**한다 (둘 다 통과해야 채택).
  1. **절대 하한** (`--floor-db`, 기본 -60dBFS): 순수 무음 구간만 제거한다.
     낮게 잡는다. 이건 품질 판정이 아니라 쓰레기 제거다.
  2. **클립 내 상대 기준** (`--rel-ratio`, 기본 0.30): 그 클립의 최대 윈도우
     RMS 대비 비율(진폭 기준, 0.30 ≈ −10.5dB). 조용히 녹음된 클립도 살아남고
     클래스 간 편향이 줄어든다.

두 값 모두 인자다. `--sweep` 으로 민감도표를 뽑아 논문 데이터셋 절의
"필터 전후 데이터 수" 표에 그대로 쓴다. 판정 로직은 `pick_windows` 하나이며
민감도표·청취 표본·캐시 생성이 모두 같은 함수를 쓴다.

── 리샘플 ──────────────────────────────────────────────────────────────────
`scipy.signal.resample_poly` (안티앨리어싱 FIR 포함). `resample` 참조.

사용법 (WSL2):
    python3 scripts/prepare_safesound.py --sweep          # 민감도표만, 캐시 생성 안 함
    python3 scripts/prepare_safesound.py --export-samples --per-class 30 --max-win 1 \\
        --n-samples 30                                    # 청취용 표본 (클립당 1윈도우)
    python3 scripts/prepare_safesound.py                  # 캐시 생성
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from scipy.signal import resample_poly
except ImportError:
    resample_poly = None

SR = 16000              # 목표 샘플레이트 (CLAUDE.md 4장)
WIN = 16384             # 1초 윈도우 = 16384 샘플 → (128,128) reshape
MAX_WIN_PER_CLIP = 3    # 클립당 윈도우 상한
SHARD = 2048            # 샤드당 윈도우 수 (2048 × 16384 int8 = 32MB)
CLASSES = ["siren", "glass", "scream", "dog_bark", "background"]
SPLITS = ["train", "test"]


# ────────────────────────────────────────────────────────────── 오디오 로드
def load_audio(path):
    """16kHz mono float32 로 읽는다. 실패 시 None."""
    if sf is None:
        sys.exit("[에러] soundfile 미설치. WSL2에서: pip install soundfile")
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as e:                                  # noqa: BLE001
        print(f"  읽기 실패 {os.path.basename(path)}: {e}", file=sys.stderr)
        return None
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = resample(x, sr, SR)
    return x


def resample(x, sr_in, sr_out):
    """다상(polyphase) 리샘플 — `scipy.signal.resample_poly`.

    up/down 을 최대공약수로 약분해 넘긴다 (44.1k→16k = 160/441). scipy 가 Kaiser 창
    FIR 저역통과를 함께 걸어 새 나이퀴스트(8kHz) 위 성분을 제거한다.

    이전 구현(3샘플 이동평균 후 선형보간)은 이동평균의 주파수 응답상 7.35kHz에서
    −3.5dB, 10kHz에서 −7.3dB 밖에 감쇠하지 못해 고주파가 접혀 들어왔다.
    고주파 성분이 많은 glass 에 가장 불리했다.
    US8K 는 파일마다 샘플레이트가 달라 비율을 고정하지 않는다.
    """
    if sr_in == sr_out:
        return x
    if resample_poly is None:
        sys.exit("[에러] scipy 미설치. WSL2에서: pip install scipy")
    g = int(np.gcd(int(sr_in), int(sr_out)))
    return resample_poly(x, sr_out // g, int(sr_in) // g).astype(np.float32)


# ────────────────────────────────────────────────────────── 윈도우 후보 추출
def window_rms(x, hop):
    """비중첩(hop=WIN) 또는 지정 hop 으로 자른 윈도우별 RMS."""
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    starts = list(range(0, len(x) - WIN + 1, hop))
    if not starts:
        starts = [0]
    rms = np.array([float(np.sqrt(np.mean(x[s:s + WIN] ** 2))) for s in starts])
    return starts, rms


def pick_windows(starts, rms, floor_lin, rel_ratio, max_win):
    """에너지 기준을 통과한 윈도우 시작점을 RMS 내림차순으로 최대 max_win 개.

    절대 하한과 상대 기준을 **함께** 건다. 클립 최대 윈도우는 상대 기준을 항상
    통과하므로(rel_ratio ≤ 1), 클립이 통째로 사라지는 경우는 최대 윈도우조차
    절대 하한에 못 미칠 때(= 무음 클립)뿐이다.

    반환: (선택된 시작점, 후보 수, 절대 하한 탈락 수, 상대 기준 탈락 수,
           클립당 상한 탈락 수)
    """
    peak = rms.max() if len(rms) else 0.0
    keep_floor = rms >= floor_lin
    keep_rel = rms >= peak * rel_ratio
    keep = keep_floor & keep_rel

    n_floor_drop = int((~keep_floor).sum())
    n_rel_drop = int((keep_floor & ~keep_rel).sum())

    order = np.argsort(-rms, kind="stable")
    passed = [starts[i] for i in order if keep[i]]
    picked = passed[:max_win]
    return (sorted(picked), len(starts), n_floor_drop, n_rel_drop,
            len(passed) - len(picked))


def select_windows(x, floor_lin, rel_ratio, hop, max_win):
    """오디오 → `pick_windows` 결과."""
    starts, rms = window_rms(x, hop)
    return pick_windows(starts, rms, floor_lin, rel_ratio, max_win)


def to_int8(w):
    """[-1,1] float → int8 [-128,127]. 클리핑 포함."""
    return np.clip(np.round(w * 127.0), -128, 127).astype(np.int8)


# ──────────────────────────────────────────────────────────── 경로 해석
def clip_path(row, roots):
    """매니페스트 행 → 실제 오디오 경로. 없으면 None."""
    src = row["source"]
    if src == "FSD50K":
        p = os.path.join(roots["fsd"], f"{row['clip_id']}.wav")
    elif src == "US8K":
        p = os.path.join(roots["us8k"], row["clip_id"])
        if not os.path.isfile(p):
            for fold in range(1, 11):
                q = os.path.join(roots["us8k"], f"fold{fold}", row["clip_id"])
                if os.path.isfile(q):
                    return q
    elif src == "ESC-50":
        p = os.path.join(roots["esc50"], row["clip_id"])
    else:
        return None
    return p if os.path.isfile(p) else None


# ──────────────────────────────────────────────────────────────── 샤드 기록
class ShardWriter:
    """클래스·split 별 int8 샤드를 순차 기록한다. 메모리에 쌓지 않는다."""

    def __init__(self, out_dir, cls, split, shard_size=SHARD):
        self.dir = os.path.join(out_dir, split, cls)
        os.makedirs(self.dir, exist_ok=True)
        self.cls, self.split = cls, split
        self.shard_size = shard_size
        self.buf = []
        self.index = []          # (shard, row, clip_id, fsid, start_sample)
        self.shard_id = 0

    def add(self, w_int8, clip_id, fsid, start):
        self.index.append((self.shard_id, len(self.buf), clip_id, fsid, start))
        self.buf.append(w_int8)
        if len(self.buf) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        arr = np.stack(self.buf)
        np.save(os.path.join(self.dir, f"shard_{self.shard_id:04d}.npy"), arr)
        self.shard_id += 1
        self.buf = []

    def close(self):
        self.flush()
        with open(os.path.join(self.dir, "index.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["shard", "row", "clip_id", "fsid", "start_sample"])
            w.writerows(self.index)
        return len(self.index)


# ──────────────────────────────────────────────────────────────────── 메인
def iter_rows(manifest, roots, limit=0, per_class=0, seed=0):
    """매니페스트 → [(행, 오디오경로)]. 오디오가 없는 행은 제외한다.

    `per_class` 를 주면 클래스별로 그만큼만 무작위 추출한다(민감도표·청취 표본용).
    매니페스트가 클래스순으로 정렬돼 있어 단순 `limit` 은 한 클래스에 쏠린다.
    """
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    out = []
    for r in rows:
        p = clip_path(r, roots)
        if p:
            out.append((r, p))

    if per_class:
        rng = np.random.default_rng(seed)
        by = defaultdict(list)
        for item in out:
            by[item[0]["cls"]].append(item)
        picked = []
        for c in CLASSES:
            v = by.get(c, [])
            if len(v) > per_class:
                sel = rng.choice(len(v), per_class, replace=False)
                v = [v[i] for i in sorted(sel)]
            picked.extend(v)
        out = picked

    if limit:
        out = out[:limit]
    return out


def sweep(rows, args):
    """임계값 민감도표 — 클래스·split 별 잔존 윈도우 수. 캐시는 만들지 않는다.

    윈도우 RMS 는 임계값과 무관하므로 **클립당 한 번만** 읽어 계산하고, 임계값
    조합은 그 RMS 배열로 평가한다. 보관하는 것은 클립당 RMS 몇 개뿐이라
    오디오를 메모리에 쌓지 않는다.
    """
    combos = [(f, r) for f in args.sweep_floor for r in args.sweep_rel]
    cols = [(c, sp) for c in CLASSES for sp in SPLITS]

    clips = []                          # (cls, split, rms)
    n_clip = Counter()
    n_fail = 0
    for i, (r, path) in enumerate(rows, 1):
        x = load_audio(path)
        if x is None:
            n_fail += 1
            continue
        _starts, rms = window_rms(x, args.hop)
        clips.append((r["cls"], r["split"], rms))
        n_clip[(r["cls"], r["split"])] += 1
        if i % 500 == 0:
            print(f"  RMS 계산 {i}/{len(rows)}", flush=True)

    def tally(floor_lin, rel, max_win):
        win, dead = Counter(), 0
        for c, sp, rms in clips:
            picked = pick_windows(range(len(rms)), rms, floor_lin, rel, max_win)[0]
            win[(c, sp)] += len(picked)
            dead += not picked
        return win, dead

    def row_md(label_a, label_b, win, dead):
        cells = [label_a, label_b] + [str(win[k]) for k in cols]
        cells += [str(sum(win.values())), str(dead)]
        return "| " + " | ".join(cells) + " |"

    head = ["floor(dBFS)", "rel"] + [f"{c}/{sp}" for c, sp in cols] + ["합계", "하한 탈락 클립"]
    sep = "|" + "|".join(["---:"] * len(head)) + "|"

    src = Counter(r["source"] for r, _ in rows)
    lines = [
        "# 에너지 필터 민감도표",
        "",
        "`scripts/prepare_safesound.py --sweep` 자동 생성. 손으로 고치지 말 것.",
        "",
        f"- 입력: 오디오가 있는 클립 {len(rows)}개 ("
        + ", ".join(f"{k} {v}" for k, v in src.most_common())
        + f"), 읽기 실패 {n_fail}",
        f"- 윈도우: {WIN}샘플 @ {SR}Hz, hop {args.hop_ms}ms, 클립당 최대 {args.max_win}개",
        "- floor: 윈도우 RMS 절대 하한 (dBFS, 최대 진폭 1.0 기준)",
        "- rel: 클립 최대 윈도우 RMS 대비 비율 (진폭 기준). 두 기준 모두 통과해야 채택",
        "- 값은 **윈도우 수**다. 원본 수가 아니므로 신뢰구간 계산에 쓰지 말 것",
        "",
        "## 클립 수",
        "",
        "| " + " | ".join(f"{c}/{sp}" for c, sp in cols) + " |",
        "|" + "|".join(["---:"] * len(cols)) + "|",
        "| " + " | ".join(str(n_clip[k]) for k in cols) + " |",
        "",
        "## 잔존 윈도우 수",
        "",
        "| " + " | ".join(head) + " |",
        sep,
    ]
    no_cap, _ = tally(0.0, 0.0, 10 ** 9)
    lines.append(row_md("필터 없음", "상한 없음", no_cap, 0))
    capped, _ = tally(0.0, 0.0, args.max_win)
    lines.append(row_md("필터 없음", f"상한 {args.max_win}", capped, 0))
    for floor_db, rel in combos:
        win, dead = tally(10 ** (floor_db / 20.0), rel, args.max_win)
        lines.append(row_md(f"{floor_db:.0f}", f"{rel:.2f}", win, dead))
        print(lines[-1], flush=True)

    # 클립이 통째로 빠지는 것은 절대 하한뿐이다(최대 윈도우는 상대 기준을 항상 통과).
    # 하한이 클래스 편향을 만드는지 보려면 클래스별로 봐야 한다.
    n_cls = Counter(c for c, _sp, _rms in clips)
    lines += [
        "",
        "## 절대 하한으로 통째로 탈락한 클립 (클래스별)",
        "",
        "최대 윈도우 RMS 조차 floor 미만인 클립 수 / 전체. rel 과 무관하다.",
        "",
        "| floor(dBFS) | " + " | ".join(CLASSES) + " |",
        "|" + "|".join(["---:"] * (len(CLASSES) + 1)) + "|",
    ]
    for floor_db in args.sweep_floor:
        fl = 10 ** (floor_db / 20.0)
        dead_cls = Counter(c for c, _sp, rms in clips if rms.max() < fl)
        lines.append(f"| {floor_db:.0f} | " + " | ".join(
            f"{dead_cls[c]}/{n_cls[c]} ({100 * dead_cls[c] / max(n_cls[c], 1):.1f}%)"
            for c in CLASSES) + " |")

    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.sweep_out) or ".", exist_ok=True)
    with open(args.sweep_out, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"민감도표: {args.sweep_out}")


def export_samples(rows, args):
    """청취용 표본 추출 — 자동화가 안 되는 검증이므로 사람이 직접 듣는다."""
    out = args.sample_dir
    floor_lin = 10 ** (args.floor_db / 20.0)
    per_cls = defaultdict(int)
    written = Counter()
    for r, path in rows:
        c = r["cls"]
        if per_cls[c] >= args.n_samples:
            continue
        x = load_audio(path)
        if x is None:
            continue
        picked, *_ = select_windows(x, floor_lin, args.rel_ratio,
                                    args.hop, args.max_win)
        if not picked:
            continue
        d = os.path.join(out, c)
        os.makedirs(d, exist_ok=True)
        for k, s in enumerate(picked):
            if per_cls[c] >= args.n_samples:
                break
            w = x[s:s + WIN] if s + WIN <= len(x) else np.pad(
                x[s:], (0, WIN - len(x[s:])))
            name = f"{c}_{r['clip_id']}_{s}.wav"
            sf.write(os.path.join(d, name), w, SR)
            per_cls[c] += 1
            written[c] += 1

    print(f"청취용 표본을 {out} 에 저장했다.")
    for c in CLASSES:
        print(f"  {c:<12}{written[c]:>4}개")
    print("\n⚠️ 직접 들어볼 것. 숫자로는 통과인데 무음이거나 엉뚱한 소리일 수 있다.")
    print("   특히 glass — 최대 에너지 지점이 파손음이 아니라 뒤따르는 사람")
    print("   목소리일 수 있다. 여기서 문제를 잡으면 나중에 몇 주를 아낀다.")


def build(rows, args):
    """실제 캐시 생성. 클립 하나씩 처리해 즉시 샤드에 append."""
    floor_lin = 10 ** (args.floor_db / 20.0)
    os.makedirs(args.out, exist_ok=True)
    prog_path = os.path.join(args.out, "progress.json")
    done = set()
    if os.path.isfile(prog_path) and not args.restart:
        done = set(json.load(open(prog_path))["done"])
        print(f"이어하기: 이미 처리한 클립 {len(done)}개 건너뜀")

    writers = {}
    stats = Counter()
    drops = Counter()
    for i, (r, path) in enumerate(rows, 1):
        key = f"{r['source']}:{r['clip_id']}"
        if key in done:
            continue
        x = load_audio(path)
        if x is None:
            drops["읽기 실패"] += 1
            continue
        picked, n_cand, n_floor, n_rel, n_cap = select_windows(
            x, floor_lin, args.rel_ratio, args.hop, args.max_win)
        stats[f"후보:{r['cls']}"] += n_cand
        drops["절대 하한 미달"] += n_floor
        drops["클립 내 상대기준 미달"] += n_rel
        drops["클립당 상한 초과"] += n_cap
        if not picked:
            drops["전 윈도우 탈락(무음)"] += 1
            done.add(key)
            continue

        wk = (r["cls"], r["split"])
        if wk not in writers:
            writers[wk] = ShardWriter(args.out, r["cls"], r["split"])
        for s in picked:
            seg = x[s:s + WIN]
            if len(seg) < WIN:
                seg = np.pad(seg, (0, WIN - len(seg)))
            writers[wk].add(to_int8(seg), r["clip_id"], r["fsid"], s)
            stats[f"채택:{r['cls']}"] += 1

        done.add(key)
        if i % 200 == 0:
            json.dump({"done": sorted(done)}, open(prog_path, "w"))
            print(f"  {i}/{len(rows)} 처리", flush=True)

    totals = {}
    for (c, sp), w in writers.items():
        totals[(c, sp)] = w.close()
    json.dump({"done": sorted(done)}, open(prog_path, "w"))

    print("\n=== 필터 전후 데이터 수 (논문 데이터셋 표) ===")
    print(f"{'클래스':<12}{'후보 윈도우':>12}{'채택 윈도우':>12}{'통과율':>9}")
    print("-" * 45)
    for c in CLASSES:
        cand = stats[f"후보:{c}"]
        keep = stats[f"채택:{c}"]
        if cand:
            print(f"{c:<12}{cand:>12}{keep:>12}{100 * keep / cand:>8.1f}%")
    print("\n=== 폐기 사유 ===")
    for k, v in drops.most_common():
        print(f"  {v:>8}  {k}")
    print("\n=== 샤드 (split/class) ===")
    for (c, sp), n in sorted(totals.items()):
        print(f"  {sp:<6}{c:<12}{n:>8} 윈도우")
    print(f"\n캐시: {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/interim/manifest.csv")
    ap.add_argument("--fsd-dir", default="data/raw/FSD50K_clips")
    ap.add_argument("--us8k-dir", default="data/raw/US8K_audio")
    ap.add_argument("--esc50-dir", default="data/raw/ESC50_audio")
    ap.add_argument("--out", default="data/processed/safesound")
    ap.add_argument("--sample-dir", default="data/interim/listen")

    ap.add_argument("--floor-db", type=float, default=-60.0,
                    help="절대 하한 (dBFS). 무음 제거 전용, 낮게 잡는다. "
                         "끄려면 --floor-db=-inf")
    ap.add_argument("--rel-ratio", type=float, default=0.30,
                    help="클립 최대 윈도우 RMS 대비 비율 (0~1, 진폭 기준)")
    ap.add_argument("--hop-ms", type=int, default=1000,
                    help="윈도우 hop (ms). 1000=비중첩")
    ap.add_argument("--max-win", type=int, default=MAX_WIN_PER_CLIP)

    ap.add_argument("--sweep", action="store_true", help="민감도표만 출력")
    ap.add_argument("--sweep-floor", type=float, nargs="+",
                    default=[-70, -60, -50, -40])
    ap.add_argument("--sweep-rel", type=float, nargs="+",
                    default=[0.0, 0.2, 0.3, 0.5])
    ap.add_argument("--sweep-out", default="docs/results/energy-filter-sweep.md",
                    help="민감도표 마크다운 저장 경로")
    ap.add_argument("--export-samples", action="store_true",
                    help="청취용 표본 wav 추출")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="처음 N클립만 (디버깅)")
    ap.add_argument("--per-class", type=int, default=0,
                    help="클래스별 N클립만 무작위 추출 (민감도표·청취 표본용)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--restart", action="store_true", help="진행 상태 무시하고 처음부터")
    args = ap.parse_args()
    args.hop = max(1, int(SR * args.hop_ms / 1000))

    rels = [args.rel_ratio] + (args.sweep_rel if args.sweep else [])
    if any(not 0.0 <= v <= 1.0 for v in rels):
        ap.error("rel 비율은 0~1 이어야 한다")

    roots = {"fsd": args.fsd_dir, "us8k": args.us8k_dir, "esc50": args.esc50_dir}
    rows = iter_rows(args.manifest, roots, args.limit, args.per_class, args.seed)
    if not rows:
        sys.exit("[에러] 처리할 오디오가 없다. 먼저 download_clips.py 를 실행할 것.")

    have = Counter(r["source"] for r, _ in rows)
    print(f"오디오가 있는 클립 {len(rows)}개  " +
          " ".join(f"{k}:{v}" for k, v in have.most_common()) + "\n")

    if args.sweep:
        sweep(rows, args)
    elif args.export_samples:
        export_samples(rows, args)
    else:
        build(rows, args)


if __name__ == "__main__":
    main()
