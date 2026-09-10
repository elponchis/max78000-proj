"""alarm 클래스에 실제로 무엇이 들어있는지 Freesound 태그/제목으로 확인."""
import csv, json, os, collections, re

R = os.path.expanduser("~/max78000-proj/data/raw/")
GT = R + "FSD50K_meta/FSD50K.ground_truth/"
MD = R + "FSD50K_meta/FSD50K.metadata/"

info = {}
info.update(json.load(open(MD + "dev_clips_info_FSD50K.json")))
info.update(json.load(open(MD + "eval_clips_info_FSD50K.json")))

TEL = {"Telephone", "Ringtone"}
BELLS = {"Doorbell", "Bell", "Church_bell", "Bicycle_bell", "Cowbell",
         "Chime", "Wind_chime"}
HORN = {"Vehicle_horn_and_car_horn_and_honking"}

def load(fn):
    return [(r["fname"], [x.strip() for x in r["labels"].split(",")])
            for r in csv.DictReader(open(GT + fn))]

alarm = [f for f, L in load("dev.csv") + load("eval.csv")
         if "Alarm" in L and "Siren" not in L and not (set(L) & (TEL | BELLS | HORN))]
print(f"정제 후 alarm 클립: {len(alarm)}\n")

# 의미 카테고리 키워드 (제목+태그에서 검색)
CATS = {
    "화재/연기 경보":   ["smoke", "fire alarm", "firealarm", "fire-alarm", "co detector",
                        "carbon monoxide", "detector"],
    "알람시계/타이머":  ["alarm clock", "alarmclock", "clock alarm", "wake", "timer",
                        "kitchen timer", "egg timer"],
    "차량 후진/경고음": ["reverse", "reversing", "backup", "back up", "truck",
                        "forklift", "excavator", "beeping truck"],
    "철도/전철":        ["train", "subway", "metro", "tram", "platform", "station",
                        "level crossing", "railway", "railroad", "u-bahn"],
    "가전 비프음":      ["microwave", "oven", "fridge", "refrigerator", "washing machine",
                        "dishwasher", "appliance"],
    "보안/도난 경보":   ["security", "burglar", "intruder", "siren alarm", "car alarm"],
    "의료기기":         ["hospital", "monitor", "medical", "ecg", "ventilator"],
    "일반 비프/전자음": ["beep", "bleep", "buzzer", "buzz", "electronic", "tone"],
}

hits = collections.Counter()
uncat = []
examples = collections.defaultdict(list)

for f in alarm:
    d = info.get(f, {})
    text = (d.get("title", "") + " " + " ".join(d.get("tags", [])) + " "
            + d.get("description", "")[:300]).lower()
    text = re.sub(r"[_\-]", " ", text)
    matched = False
    for cat, kws in CATS.items():
        if any(k in text for k in kws):
            hits[cat] += 1
            if len(examples[cat]) < 3:
                examples[cat].append(d.get("title", "?")[:60])
            matched = True
    if not matched:
        uncat.append((f, d.get("title", "?")[:60]))

print("=== 의미 카테고리별 클립 수 (중복 집계) ===")
for c, n in hits.most_common():
    print(f"  {n:5d} ({100*n/len(alarm):5.1f}%)  {c}")
    for t in examples[c]:
        print(f"           예: {t}")
print(f"\n  {len(uncat):5d} ({100*len(uncat)/len(alarm):5.1f}%)  미분류")
print("  미분류 제목 예시 15개:")
for f, t in uncat[:15]:
    print(f"           {t}")

# 안전 관련 vs 비안전
SAFETY = {"화재/연기 경보", "보안/도난 경보", "알람시계/타이머", "의료기기"}
NONSAFE = {"차량 후진/경고음", "철도/전철", "가전 비프음"}
s = sum(1 for f in alarm if any(
    any(k in re.sub(r"[_\-]", " ", (info.get(f, {}).get("title", "") + " "
        + " ".join(info.get(f, {}).get("tags", []))).lower()) for k in CATS[c])
    for c in SAFETY))
n = sum(1 for f in alarm if any(
    any(k in re.sub(r"[_\-]", " ", (info.get(f, {}).get("title", "") + " "
        + " ".join(info.get(f, {}).get("tags", []))).lower()) for k in CATS[c])
    for c in NONSAFE))
print(f"\n=== 제목+태그 기준 ===")
print(f"  안전 관련(화재/보안/알람시계/의료) 신호 있음: {s} ({100*s/len(alarm):.1f}%)")
print(f"  비안전(차량/철도/가전) 신호 있음:            {n} ({100*n/len(alarm):.1f}%)")
