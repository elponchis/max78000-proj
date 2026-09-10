#!/usr/bin/env python3
"""FSD50K 라벨 실재 확인 및 클래스별 클립 수 집계.

TASKS.md Phase 2.1 대응. 오디오 없이 메타데이터(ground_truth)만으로 실행한다.

하는 일
  1) vocabulary.csv 에 CLAUDE.md 4장 클래스 매핑의 라벨이 실제로 있는지 확인
     - 부재 라벨은 키워드 기반 대체 후보를 함께 제시
  2) dev.csv / eval.csv 로 클래스별 클립 수 집계 (CLAUDE.md 5장 규칙 1: dev/eval 분할 준수)
  3) 목표 수량 달성 가능성 및 scream 300개 기준 판정

── FSD50K 온톨로지 주의 (이 스크립트 설계의 핵심) ──────────────────────────
FSD50K는 AudioSet 온톨로지를 따르며 **상위(조상) 라벨을 함께 부여**한다.
예: `Siren` 클립은 예외 없이 `Alarm`도 함께 달고 있다 (실측 132/132 = 100%).
따라서 CLAUDE.md 5장 규칙 3("타깃 클래스 중 정확히 1개만 포함")을 라벨 문자열에
그대로 적용하면 siren이 전부 alarm과 충돌해 0개가 된다.

→ 해결: CLASS_PRIORITY 로 **더 구체적인 클래스가 상위 클래스를 이긴다**.
   siren < alarm 관계처럼 온톨로지상 포함관계인 쌍은 우선순위로 해소하고,
   포함관계가 아닌 진짜 충돌(예: scream + baby_cry)만 클립을 폐기한다.

사용법 (WSL2):
    python3 scripts/check_vocabulary.py --gt-dir data/raw/FSD50K_meta/FSD50K.ground_truth
"""

import argparse
import csv
import os
import sys
from collections import Counter

# ── CLAUDE.md 4장 클래스 정의를 그대로 옮긴 것 ──────────────────────────────
# 값은 FSD50K vocabulary.csv 의 display_name 후보.
# FSD50K 표기는 공백 대신 '_', '&'/','는 '_and_' 로 쓴다.
CLASS_MAP = {
    "siren": [
        "Siren",
        "Ambulance_(siren)",
        "Police_car_(siren)",
        "Fire_engine_and_fire_truck_(siren)",
    ],
    "alarm": [
        "Alarm",
        "Alarm_clock",
        "Smoke_detector_and_smoke_alarm",
    ],
    "glass": [
        "Glass",
        "Shatter",
    ],
    "scream": [
        "Screaming",
        "Yell",
        "Shout",
    ],
    "baby_cry": [
        "Crying_and_sobbing",
    ],
    "dog_bark": [
        "Bark",
        "Dog",
    ],
}

# 온톨로지 포함관계 해소용 우선순위. 앞쪽이 이긴다.
# alarm 은 AudioSet 에서 siren 의 조상이므로 반드시 마지막에 둔다.
CLASS_PRIORITY = ["siren", "glass", "scream", "baby_cry", "dog_bark", "alarm"]

# 우선순위로 해소해도 되는 쌍(= 온톨로지 포함관계). 그 외 조합은 진짜 충돌로 보고 폐기.
SUBSUMED_PAIRS = {("siren", "alarm")}

# 라벨이 없을 때 대체 후보를 찾기 위한 키워드 (부분 일치, 대소문자 무시)
FALLBACK_KEYWORDS = {
    "siren": ["siren", "ambulance", "police", "fire_engine", "emergency"],
    "alarm": ["alarm", "smoke", "buzzer", "beep", "bell", "ringtone", "horn"],
    "glass": ["glass", "shatter", "break"],
    "scream": ["scream", "yell", "shout", "shriek"],
    "baby_cry": ["cry", "sob", "baby", "infant", "child"],
    "dog_bark": ["bark", "dog", "growl", "howl", "whimper"],
}

MIN_SCREAM_CLIPS = 300     # 이 미만이면 scream 제외 → 6클래스 (CLAUDE.md 4장)
TARGET_PER_CLASS = 800     # 클래스당 목표 학습 윈도우 수 (CLAUDE.md 5장)
WINDOWS_PER_CLIP = 1.5     # 클립당 추출 가능 윈도우 수 가정 (보수적)


def load_vocabulary(path):
    """vocabulary.csv → (display_name -> mid). 헤더 없음: index,display_name,mid"""
    vocab = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 3 or row[0].strip().lower() in ("index", "id"):
                continue
            vocab[row[1].strip()] = row[2].strip()
    return vocab


def load_ground_truth(path):
    """dev.csv / eval.csv → [(fname, [labels...], split)]. 헤더 있음."""
    clips = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels = [x.strip() for x in row["labels"].split(",") if x.strip()]
            clips.append((row["fname"], labels, row.get("split", "")))
    return clips


def search(vocab, keywords):
    """키워드 부분 일치로 vocabulary 검색."""
    return sorted(n for n in vocab
                  if any(k.lower() in n.lower() for k in keywords))


def resolve(hit_classes):
    """클립이 걸린 클래스 집합 → 최종 단일 클래스 또는 None(폐기).

    온톨로지 포함관계(SUBSUMED_PAIRS)면 우선순위로 해소하고,
    그 외 다중 히트는 CLAUDE.md 5장 규칙 3에 따라 폐기한다.
    """
    if len(hit_classes) == 0:
        return None
    if len(hit_classes) == 1:
        return next(iter(hit_classes))
    ordered = sorted(hit_classes, key=CLASS_PRIORITY.index)
    winner, losers = ordered[0], ordered[1:]
    if all((winner, l) in SUBSUMED_PAIRS for l in losers):
        return winner
    return None


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True,
                    help="FSD50K.ground_truth 디렉터리 (vocabulary.csv, dev.csv, eval.csv)")
    ap.add_argument("--show-contamination", action="store_true",
                    help="클래스별 동시 등장 상위 라벨 출력 (클래스 정의 점검용)")
    args = ap.parse_args()

    gt = args.gt_dir
    vocab_path = os.path.join(gt, "vocabulary.csv")
    if not os.path.isfile(vocab_path):
        sys.exit(f"[에러] vocabulary.csv 없음: {vocab_path}")

    vocab = load_vocabulary(vocab_path)

    # ────────────────────────────────────────────────── 1) 라벨 실재 확인
    section(f"1. vocabulary.csv 라벨 실재 확인  (총 {len(vocab)}개 라벨)")

    resolved_labels = {}   # 클래스 -> 실재하는 라벨
    missing_report = {}    # 클래스 -> 부재 라벨

    for cls, candidates in CLASS_MAP.items():
        found = [l for l in candidates if l in vocab]
        missing = [l for l in candidates if l not in vocab]
        resolved_labels[cls] = found
        missing_report[cls] = missing

        print(f"\n[{cls}]")
        for label in found:
            print(f"  O  {label:<40s} {vocab[label]}")
        for label in missing:
            print(f"  X  {label:<40s} (vocabulary에 없음)")
        if missing:
            alt = [n for n in search(vocab, FALLBACK_KEYWORDS[cls]) if n not in found]
            if alt:
                print(f"     └ 대체 후보: {', '.join(alt)}")

    # ────────────────────────────────────────────────── 2) 클립 수 집계
    dev_path = os.path.join(gt, "dev.csv")
    eval_path = os.path.join(gt, "eval.csv")
    if not (os.path.isfile(dev_path) and os.path.isfile(eval_path)):
        print("\n[경고] dev.csv / eval.csv 없음 → 클립 수 집계 생략.")
        return

    dev = load_ground_truth(dev_path)
    ev = load_ground_truth(eval_path)

    label2class = {l: c for c, ls in resolved_labels.items() for l in ls}

    def tally(clips):
        raw, final = Counter(), Counter()
        dropped = Counter()          # 폐기 사유(클래스 조합) 집계
        for _fname, labels, _split in clips:
            hits = {label2class[l] for l in labels if l in label2class}
            for c in hits:
                raw[c] += 1
            if not hits:
                continue
            winner = resolve(hits)
            if winner:
                final[winner] += 1
            else:
                dropped["+".join(sorted(hits))] += 1
        return raw, final, dropped

    dev_raw, dev_fin, dev_drop = tally(dev)
    ev_raw, ev_fin, ev_drop = tally(ev)

    section("2. 클래스별 클립 수 (FSD50K dev/eval 분할 준수)")
    print("raw = 라벨 포함 클립 / final = 온톨로지 우선순위 + 다중라벨 필터 통과")
    print()
    print(f"{'클래스':<12}{'dev raw':>9}{'dev final':>11}"
          f"{'eval raw':>10}{'eval final':>12}{'final 합':>10}")
    print("-" * 78)
    totals = {}
    for cls in CLASS_MAP:
        tot = dev_fin[cls] + ev_fin[cls]
        totals[cls] = tot
        print(f"{cls:<12}{dev_raw[cls]:>9}{dev_fin[cls]:>11}"
              f"{ev_raw[cls]:>10}{ev_fin[cls]:>12}{tot:>10}")
    print("-" * 78)
    print(f"{'합계':<12}{'':>9}{sum(dev_fin.values()):>11}"
          f"{'':>10}{sum(ev_fin.values()):>12}{sum(totals.values()):>10}")
    print(f"\n전체 클립 수: dev {len(dev)} / eval {len(ev)}")
    print(f"배경음 후보(타깃 라벨 전혀 없는 클립): "
          f"dev {len(dev) - sum(dev_raw.values()) + 0} 내외 / "
          f"eval {len(ev) - sum(ev_raw.values()) + 0} 내외")

    drops = dev_drop + ev_drop
    if drops:
        print("\n다중 클래스 충돌로 폐기된 클립:")
        for combo, n in drops.most_common():
            print(f"   {n:5d}  {combo}")

    if args.show_contamination:
        section("2b. 클래스별 동시 등장 상위 라벨 (클래스 정의 점검용)")
        for cls in CLASS_MAP:
            c = Counter()
            n = 0
            for _f, labels, _s in dev + ev:
                if any(l in resolved_labels[cls] for l in labels):
                    n += 1
                    for l in labels:
                        if l not in resolved_labels[cls]:
                            c[l] += 1
            print(f"\n[{cls}]  raw {n} clips")
            for l, k in c.most_common(8):
                print(f"   {k:5d} ({100 * k / max(n, 1):5.1f}%)  {l}")

    # ────────────────────────────────────────────────── 3) 판정
    section("3. 판정")

    if "Smoke_detector_and_smoke_alarm" in missing_report["alarm"]:
        print("- Smoke_detector_and_smoke_alarm / Alarm_clock 모두 부재.")
        print("  → alarm 은 상위 라벨 `Alarm` 하나로만 정의된다. 이 라벨은 전화벨·")
        print("    초인종·차량 경적까지 포함하는 광범위한 카테고리이므로,")
        print("    --show-contamination 으로 오염도를 확인하고 클래스 정의를 재검토할 것.")
    else:
        print("- alarm 세부 라벨 존재 → CLAUDE.md 정의 유지 가능.")

    if len(resolved_labels["siren"]) == 1:
        print("- siren 은 세부 라벨 없이 `Siren` 하나뿐이며, 전량이 `Alarm` 하위다.")
        print("  → alarm 과의 분리는 우선순위로 처리했으나 두 클래스의 음향적 구분이")
        print("    어려울 수 있다. 혼동행렬에서 siren↔alarm 혼동을 주시할 것.")

    scream_total = totals["scream"]
    if scream_total < MIN_SCREAM_CLIPS:
        print(f"- scream {scream_total} clips < {MIN_SCREAM_CLIPS} "
              f"→ 6클래스 축소 검토.")
    else:
        print(f"- scream {scream_total} clips >= {MIN_SCREAM_CLIPS} "
              f"→ 7클래스 유지 가능. 샘플 청취로 품질 확인 필요.")

    need = int(TARGET_PER_CLASS / WINDOWS_PER_CLIP)
    print(f"\n- 클래스당 목표 {TARGET_PER_CLASS} 윈도우, 클립당 {WINDOWS_PER_CLIP} 윈도우 "
          f"가정 → 최소 {need} 클립 필요")
    weak = [c for c in CLASS_MAP if totals[c] < need]
    if weak:
        for c in weak:
            print(f"   부족: {c:<10s} {totals[c]:>5d} clips  "
                  f"(부족분 {need - totals[c]} clips)")
        print("   → US8K / ESC-50 보강 또는 클래스 정의 확장 필요.")
    else:
        print("   모든 클래스가 목표 수량 달성 가능 범위.")

    print("\n[주의] 위 수치는 메타데이터 기반 상한이다. RMS 에너지 필터 통과 후")
    print("       실제 윈도우 수는 줄어든다. 확정 수치는 prepare_safesound.py 실행 후 기록할 것.")


if __name__ == "__main__":
    main()
