#!/usr/bin/env python3
"""① 체크포인트를 증류 교사용으로 감싼다 (`net.` 접두사 부여).

C(지식 증류)에서 교사는 `AI85SafeSoundMelTeacher` 이고, 그 안에 멜 CNN 이
`self.net` 으로 들어 있다. distiller 는 `--kd-resume` 를
`apputils.load_lean_checkpoint(teacher, path)` 로 **교사 모듈 전체에** 얹으므로,
키가 `net.voice_...` 형태여야 한다. ①의 체크포인트는 접두사가 없으니 붙여 준다.

붙이지 않으면 `strict=False` 로 조용히 넘어가 **교사가 무작위 가중치로 돈다.**
그러면 증류가 아무 정보도 전달하지 않는데 학습은 정상처럼 돌고, "증류가 효과
없다" 로 잘못 결론 내린다. 그래서 이 도구가 **얹힌 키 수를 세어 출력**한다.

사용법 (WSL2 또는 Colab):
    python3 tools/wrap_teacher.py --src <① qat_best.pth.tar> \\
        --out data/teacher-mel.pth.tar --ai8x ~/ai8x-training
"""

import argparse
import importlib.util
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="① mel 의 qat_best 체크포인트")
    ap.add_argument("--out", default=os.path.join(REPO, "data",
                                                  "teacher-mel.pth.tar"))
    ap.add_argument("--ai8x", default=os.path.expanduser("~/ai8x-training"))
    ap.add_argument("--classes", type=int, default=5)
    a = ap.parse_args()

    if not os.path.isfile(a.src):
        sys.exit(f"[에러] 없다: {a.src}")
    sys.path.insert(0, a.ai8x)
    import torch

    import ai8x

    ai8x.set_device(85, False, False)
    spec = importlib.util.spec_from_file_location(
        "melmod", os.path.join(REPO, "models", "ai85net-safesound-mel.py"))
    mm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mm)
    teacher = mm.ai85safesoundmelteacher(num_classes=a.classes)

    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    src = {k.replace("module.", ""): v for k, v in ck.get("state_dict", ck).items()}
    wrapped = {"net." + k: v for k, v in src.items()}

    want = teacher.state_dict()
    hit = [k for k in wrapped if k in want and
           tuple(wrapped[k].shape) == tuple(want[k].shape)]
    missing = [k for k in want if k not in wrapped]
    extra = [k for k in wrapped if k not in want]

    print(f"원본: {a.src}  (키 {len(src)})")
    print(f"교사 모듈 키 {len(want)} / 접두사 부여 후 일치 **{len(hit)}**")
    if missing:
        print(f"  교사에만 있음 {len(missing)}: {missing[:6]}")
    if extra:
        print(f"  ⚠️ 교사에 없는 키 {len(extra)}: {extra[:6]}")
    # 멜 CNN 의 conv/fc 가중치가 전부 얹혔는지가 핵심이다
    w_hit = [k for k in hit if k.endswith("op.weight")]
    print(f"  그중 op.weight {len(w_hit)}개 (멜 CNN 은 conv 6 + fc 1 = **7** 이어야 한다)")
    if len(w_hit) != 7:
        sys.exit(f"[에러] op.weight 가 7개가 아니다 ({len(w_hit)}) — "
                 "교사가 무작위 가중치로 돌 수 있다. 중단한다.")

    merged = dict(want)
    merged.update({k: wrapped[k] for k in hit})
    teacher.load_state_dict(merged, strict=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"state_dict": teacher.state_dict(), "epoch": 0,
                "arch": "ai85safesoundmelteacher",
                "extras": {"wrapped_from": os.path.basename(a.src),
                           "matched_keys": len(hit)}}, a.out)
    print(f"\n저장: {a.out}")
    print("학습: --kd-teacher ai85safesoundmelteacher "
          f"--kd-resume {a.out}")
    print("⚠️ 먼저 `tools/kat_teacher.py --checkpoint <① qat_best>` 로 교사 로짓이")
    print("   ① 로짓과 같은지 확인할 것 — 다르면 증류가 ①의 경계를 전달하지 않는다.")


if __name__ == "__main__":
    main()
