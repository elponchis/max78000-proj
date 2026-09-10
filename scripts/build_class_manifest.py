#!/usr/bin/env python3
"""클래스 매니페스트 생성 — 확정 매핑을 클립 목록으로 구체화한다.

TASKS.md Phase 2.1의 산출물. 오디오 없이 메타데이터만으로 실행한다.
`prepare_safesound.py`는 이 매니페스트를 입력으로 받아 윈도우를 추출한다.

출력: data/interim/manifest.csv
    clip_id, fsid, cls, split, source, duration_sec

`clip_id` 는 개별 오디오 파일, `fsid` 는 그 파일이 유래한 Freesound 원본이다.
UrbanSound8K 는 원본 하나에서 여러 슬라이스를 뜬다 — siren 929슬라이스가
**고유 원본 74개**에서 나온다(원본당 중앙값 9, 최대 100). 분할도 신뢰구간도
반드시 `fsid` 단위로 계산해야 하며, `clip_id` 수는 데이터 양일 뿐이다.

── 설계 근거 (CLAUDE.md 4·5장) ──────────────────────────────────────────────
1) AudioSet 온톨로지 조상 라벨 문제
   FSD50K는 조상 라벨을 함께 부여한다. `Siren` 클립은 132/132(100%)가 `Alarm`을
   동시 보유하므로, 다중라벨 필터를 문자열에 그대로 적용하면 siren이 전멸한다.
   → 구체 클래스가 먼저 판정되도록 규칙 순서를 고정했다.

2) 데이터셋 간 원본 중복
   FSD50K / UrbanSound8K / ESC-50 은 모두 Freesound 파생이며 ID 체계가 같다
   (fname / fsID / src_file). 1,002개 원본이 2개 이상 데이터셋에 등장하고,
   각 데이터셋의 공식 분할을 그대로 쓰면 337개가 train/test로 갈라진다.
   → **분할은 Freesound 원본 ID 단위로 전역 1회만** 결정한다.

3) 분할 정책
   FSD50K 공식 eval은 전수 검증된 고품질 라벨이므로 **전량 test에 유지**한다.
   그 위에 dev 원본을 결정적 해시로 추가 이관해 test 비율을 TEST_RATIO 까지 올린다.
   FSD50K 벤치마크에 참가하는 것이 아니므로 공식 분할 고수의 이득보다
   평가 신뢰구간을 좁히는 이득이 크다. 논문 데이터셋 절에 사유를 명시할 것.

사용법 (WSL2):
    python3 scripts/build_class_manifest.py
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict

# ── 확정 클래스 (5클래스: 이벤트 4 + 배경음) ────────────────────────────────
CLASSES = ["siren", "glass", "scream", "dog_bark", "background"]

# 클래스 판정에 쓰는 FSD50K 라벨
L_SIREN = {"Siren"}
L_GLASS = {"Glass", "Shatter"}
L_SCREAM = {"Screaming", "Yell", "Shout"}
L_DOG = {"Bark", "Dog"}

# 배제 라벨 — 클래스 순도를 위해 제외하며, 제외분은 배경음 하드 네거티브로 간다
EX_DISH = {"Chink_and_clink", "Dishes_and_pots_and_pans", "Cutlery_and_silverware",
           "Coin_(dropping)", "Tap", "Liquid", "Pour"}
EX_CROWD = {"Cheering", "Applause", "Clapping", "Crowd", "Chatter", "Laughter",
            "Giggle", "Chuckle_and_chortle", "Music", "Singing"}

# alarm 은 클래스에서 제외됐다 (docs/results/alarm-class-analysis.md).
# `Alarm` 보유 클립 전량이 배경음 하드 네거티브가 된다.
HARD_NEG_MARKERS = {
    "alarm": {"Alarm"},
    "baby_cry": {"Crying_and_sobbing"},
}

TEST_RATIO = 0.30          # 목표 test 비율 (원본 클립 기준)
BYTES_PER_SEC = 44100 * 2  # FSD50K: PCM 16bit / 44.1kHz / mono

US8K_MAP = {"siren": "siren", "dog_bark": "dog_bark"}
ESC50_MAP = {"siren": "siren", "glass_breaking": "glass", "dog": "dog_bark"}


def classify(labels):
    """FSD50K 라벨 리스트 → 클래스명 또는 None(배경음 후보).

    순서가 규칙이다. siren 이 glass/scream 보다, 그리고 무엇보다 `Alarm`
    조상 라벨보다 먼저 판정되어야 한다.
    """
    s = set(labels)
    if s & L_SIREN:
        return "siren"
    if "Shatter" in s or ("Glass" in s and not (s & EX_DISH)):
        return "glass"
    if (s & L_SCREAM) and not (s & EX_CROWD):
        return "scream"
    if s & L_DOG:
        return "dog_bark"
    return None


def hard_negative_reason(labels):
    """배경음으로 보내되 하드 네거티브로 표시할 사유. 없으면 None."""
    s = set(labels)
    if "Alarm" in s:
        return "alarm"
    if "Crying_and_sobbing" in s:
        return "baby_cry"
    if "Glass" in s and (s & EX_DISH):
        return "glass_dishes"
    if (s & L_SCREAM) and (s & EX_CROWD):
        return "scream_crowd"
    return None


def wilson_halfwidth(p, n, z=1.96):
    if n <= 0:
        return 0.0
    den = 1 + z * z / n
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den


def windows_for(duration, cap=3):
    """비중첩 1초 윈도우 수의 **길이 기반 상한**.

    ⚠️ 실제 수량이 아니다. CLAUDE.md 5장 규칙 2의 RMS 에너지 임계값 필터를
    적용하면 이보다 줄어든다. 특히 glass 는 0.3초 내외 과도음이라 긴 클립에서도
    유효 윈도우가 1개만 남을 수 있다. 확정 수치는 prepare_safesound.py 실행 후.
    """
    return min(max(1, int(duration // 1.0)), cap)


def load_fsd(raw):
    """FSD50K → {fsid: (labels, official_split, duration)}"""
    gt = os.path.join(raw, "FSD50K_meta/FSD50K.ground_truth")
    sizes_path = os.path.join(raw, "FSD50K_meta/clip_sizes.json")
    sizes = json.load(open(sizes_path)) if os.path.isfile(sizes_path) else {}

    out = {}
    for fn, official in (("dev.csv", "dev"), ("eval.csv", "eval")):
        for r in csv.DictReader(open(os.path.join(gt, fn), encoding="utf-8")):
            labels = [x.strip() for x in r["labels"].split(",") if x.strip()]
            size = sizes.get(official, {}).get(r["fname"])
            d = max(0.0, (size - 44) / BYTES_PER_SEC) if size else 0.0
            out[r["fname"]] = (labels, official, d)
    return out


def rank(fsid):
    """원본 ID의 결정적 순위값. 난수 시드 없이 재현 가능한 분할을 만든다."""
    return int(hashlib.md5(fsid.encode()).hexdigest(), 16)


def split_class(originals, forced_test):
    """한 클래스의 원본들을 train/test 로 나눈다.

    FSD50K 공식 eval 원본(`forced_test`)은 전수 검증된 라벨이므로 전량 test 에
    유지한다. 그것만으로 TEST_RATIO 에 못 미치면 나머지에서 결정적 순위로
    추가 이관한다. 이미 넘으면 그대로 둔다(검증된 클립을 train 으로 내리지 않는다).
    """
    test = set(forced_test) & originals
    target = math.ceil(TEST_RATIO * len(originals))
    if len(test) < target:
        pool = sorted(originals - test, key=rank)
        test |= set(pool[:target - len(test)])
    return {o: ("test" if o in test else "train") for o in originals}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", help="원본 메타데이터 디렉터리")
    ap.add_argument("--out", default="data/interim/manifest.csv")
    args = ap.parse_args()

    fsd = load_fsd(args.raw)
    if not fsd:
        sys.exit("[에러] FSD50K ground_truth 를 찾을 수 없다.")

    us_path = os.path.join(args.raw, "US8K_meta/UrbanSound8K.csv")
    esc_path = os.path.join(args.raw, "ESC50_meta/esc50.csv")
    us_rows = list(csv.DictReader(open(us_path, encoding="utf-8"))) \
        if os.path.isfile(us_path) else []
    esc_rows = list(csv.DictReader(open(esc_path, encoding="utf-8"))) \
        if os.path.isfile(esc_path) else []

    # ── 1. 클래스별 원본 집합을 모은 뒤 원본 단위로 분할 ───────────────────
    cls_of = {}          # fsid -> class
    forced_test = set()  # FSD50K 공식 eval 원본
    hard_neg = Counter()
    fsd_originals = set()

    for fsid, (labels, official, _d) in fsd.items():
        c = classify(labels)
        if c:
            cls_of[fsid] = c
            fsd_originals.add(fsid)
            if official == "eval":
                forced_test.add(fsid)
        else:
            reason = hard_negative_reason(labels)
            if reason:
                hard_neg[reason] += 1

    for r in us_rows:
        c = US8K_MAP.get(r["class"])
        if c and r["fsID"] not in fsd_originals:
            cls_of.setdefault(r["fsID"], c)
            if r["fold"] == "10":
                forced_test.add(r["fsID"])
    for r in esc_rows:
        c = ESC50_MAP.get(r["category"])
        if c and r["src_file"] not in fsd_originals:
            cls_of.setdefault(r["src_file"], c)
            if r["fold"] == "5":
                forced_test.add(r["src_file"])

    by_class = defaultdict(set)
    for fsid, c in cls_of.items():
        by_class[c].add(fsid)

    split_of = {}
    for c, originals in by_class.items():
        split_of.update(split_class(originals, forced_test))

    # ── 2. 클립 수집 ───────────────────────────────────────────────────────
    # 개별 슬라이스는 모두 유지하되(서로 다른 구간 = 실제 추가 데이터),
    # 원본이 FSD50K에 이미 있으면 타 데이터셋 사본은 버린다(오디오 중복 회피).
    rows = []
    dropped = Counter()

    for fsid, (labels, _official, d) in fsd.items():
        c = cls_of.get(fsid)
        if c:
            rows.append((fsid, fsid, c, split_of[fsid], "FSD50K", round(d, 3)))

    for r in us_rows:
        c = US8K_MAP.get(r["class"])
        if not c:
            continue
        fsid = r["fsID"]
        if fsid in fsd_originals:
            dropped[f"{c}: US8K 슬라이스(원본이 FSD50K에 있음)"] += 1
            continue
        d = float(r["end"]) - float(r["start"])
        rows.append((r["slice_file_name"], fsid, c, split_of[fsid],
                     "US8K", round(d, 3)))

    for r in esc_rows:
        c = ESC50_MAP.get(r["category"])
        if not c:
            continue
        fsid = r["src_file"]
        if fsid in fsd_originals:
            dropped[f"{c}: ESC-50 클립(원본이 FSD50K에 있음)"] += 1
            continue
        rows.append((r["filename"], fsid, c, split_of[fsid], "ESC-50", 5.0))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "fsid", "cls", "split", "source", "duration_sec"])
        w.writerows(sorted(rows, key=lambda x: (x[2], x[3], x[1])))

    # ── 3. 보고 ────────────────────────────────────────────────────────────
    by = defaultdict(lambda: defaultdict(list))          # 클래스 -> split -> 길이
    orig = defaultdict(lambda: defaultdict(set))         # 클래스 -> split -> fsid
    for _cid, fsid, c, sp, _src, d in rows:
        by[c][sp].append(d)
        orig[c][sp].add(fsid)

    EVENTS = ("siren", "glass", "scream", "dog_bark")

    print("=" * 78)
    print("클래스 매니페스트 —", args.out)
    print("=" * 78)

    print("\n[확정] 고유 Freesound 원본 수 — 분할·신뢰구간의 기준 단위")
    print(f"{'클래스':<10}{'train':>8}{'test':>7}{'test%':>7}{'95%CI':>9}"
          f"   {'출처별 원본':<34}")
    print("-" * 78)
    for c in EVENTS:
        tr, te = orig[c]["train"], orig[c]["test"]
        n = len(tr) + len(te)
        srcmap = defaultdict(set)
        for _cid, fsid, cc, _sp, s, _d in rows:
            if cc == c:
                srcmap[s].add(fsid)
        srctxt = " ".join(f"{k}:{len(v)}" for k, v in
                          sorted(srcmap.items(), key=lambda x: -len(x[1])))
        print(f"{c:<10}{len(tr):>8}{len(te):>7}{100*len(te)/max(n,1):>6.0f}%"
              f"{'±%.1f%%p' % (100*wilson_halfwidth(0.90, len(te))):>9}   {srctxt:<34}")
    print("-" * 78)
    print(f"{'합계':<10}{sum(len(orig[c]['train']) for c in EVENTS):>8}"
          f"{sum(len(orig[c]['test']) for c in EVENTS):>7}")
    print("  CI = 재현율 90% 가정 시 95% Wilson 반폭. 표본 크기의 성질이지 실측값이 아니다.")
    print("  한 원본에서 나온 슬라이스·윈도우는 독립이 아니므로 CI는 원본 수로 계산한다.")

    print("\n[확정] 오디오 파일 수 — 데이터 양 (US8K는 원본당 여러 슬라이스)")
    print(f"{'클래스':<10}{'train':>8}{'test':>7}{'원본당':>8}")
    print("-" * 36)
    for c in EVENTS:
        tr, te = by[c]["train"], by[c]["test"]
        n = len(orig[c]["train"]) + len(orig[c]["test"])
        print(f"{c:<10}{len(tr):>8}{len(te):>7}{(len(tr)+len(te))/max(n,1):>8.2f}")

    print("\n[잠정] 윈도우 수 — 길이 기반 상한, 에너지 필터 적용 전")
    print(f"{'클래스':<10}{'train상한':>11}{'test상한':>10}")
    print("-" * 34)
    for c in EVENTS:
        wtr = sum(windows_for(d) for d in by[c]["train"])
        wte = sum(windows_for(d) for d in by[c]["test"])
        print(f"{c:<10}{wtr:>11}{wte:>10}")
    print("  ⚠️ 목표 달성 여부를 이 표로 판정하지 말 것. RMS 에너지 임계값 필터")
    print("     적용 후 실측이 필요하다 (TASKS.md Phase 2.3 '실측 후 재판정').")

    print("\n[확정] 배경음 하드 네거티브 (배제분 재활용)")
    for k, v in hard_neg.most_common():
        print(f"  {v:5d}  {k}")
    print(f"  {sum(hard_neg.values()):5d}  합계")

    if dropped:
        print("\n[확정] 원본 중복으로 폐기된 사본")
        for k, v in sorted(dropped.items()):
            print(f"  {v:5d}  {k}")


if __name__ == "__main__":
    main()
