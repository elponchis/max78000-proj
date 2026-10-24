#!/usr/bin/env python3
"""후속 연구 fe8 (p1 8비트 + Abs + NPU 합산, 전체 QAT 재학습) — **착수 전 검산** (PC, 보드 미접촉).

설계·사전 등록: docs/results/npu-readback-design.md 7절. 모델: models/ai85net-safesound-fe.py `p1_bits8=True`.

1. 그래프 ≡ 정수 경로. ②′ 시드 1 QAT 체크포인트를 fe8 모델에 얹고 p1 가중치에 1/2 을 곱한다
   (max|w| 1.35 → 0.67, 출력 시프트 +1 → 0; 정수 가중치 W_int 는 그대로). 학습 창 표본 2,048개에서
   fe8 그래프의 q 를 시뮬레이션 도구의 정수 경로 `Front.A_sum(s=0, 대역 배율 1)` 과 비교한다.
   기대: 8비트 단계 a 와 q 가 **비트 일치** (0 불일치). 모델은 디바이스 순서(반올림 → 포화 → |·|)를 따른다 — ai8x 의
   activation="Abs" (|·| 뒤 반올림) 를 썼을 때는 "음수 v 가 정확히 k+0.5" 인 자리 0.34 % 가 어긋나고 조용한 대역에서 q 가
   최대 44 단계 움직였다 (2026-10-24 첫 검산). 그 자리의 수(`a_mismatch_at_neg_half` 후보)도 참고로 센다.
2. 기울기. QAT 모드에서 손실이 fe.op.weight 까지 흐르는가 (8비트 STE → |·| → 합 → 정수 로그 STE → p2).
3. 그대로 F1. 재학습 없이 ②′ 가중치(×1/2)를 fe8 그래프에 얹은 시험(testL-S3)·외부(real31L) F1 —
   시뮬레이션 '그대로 A-sum' (0.538 / 0.363, 대역 배율 포함) 과 같은 자리의 값이다. 재학습이 이 값을 얼마나 되찾는지가 7절의 질문.
4. 8비트 창 통계. Gabor 초기값과 학습된 ②′ 가중치의 포화(a = 127)·0 비율 — 사전 게인이 필요한가의 근거.
5. ai8x `apply_scales` 가 로그단을 건너 fe → cls.conv1 을 '이전 층' 으로 잇는가 (잇는다: conv1 문턱 = 자기 값 − t_fe) 와
   `qat_link_cut=True` 가 그 연결을 끊는가. 설계 문서 7.1 (다).

실행 (WSL2): export OPENBLAS_NUM_THREADS=1; ~/ai8x-training/venv/bin/python tools/fe8_check.py
출력: docs/results/npu-readback-sim/fe8_check.json + 표준출력
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = os.path.expanduser("~/ai8x-training")
OUT = os.path.join(REPO, "docs", "results", "npu-readback-sim")
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, T)
sys.path.insert(0, os.path.join(T, "datasets"))

N_SAMPLE = 2048


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def batches(ds, idxs, bs=256):
    import torch
    for i in range(0, len(idxs), bs):
        xs, ts = [], []
        for j in idxs[i:i + bs]:
            x, t = ds[j]
            xs.append(x); ts.append(int(t) if not hasattr(t, "__len__") else int(t[0]))
        yield torch.stack(xs), np.array(ts)


def logits_of(model, ds, bs=256):
    import torch
    outs, ys = [], []
    with torch.no_grad():
        for x, t in batches(ds, list(range(len(ds))), bs):
            outs.append(model(x).numpy()); ys.append(t)
    return np.concatenate(outs), np.concatenate(ys)


def score_model(model, ds_t, ds_e, S, E):
    """npu_readback_sim.score 와 같은 규칙 (시험 배경 @300/h 문턱 → testL F1, 외부 real31L F1), 모델을 직접 돌린다."""
    lt, yt = logits_of(model, ds_t)
    assert np.array_equal(yt, S["test_y"])
    thr = E.thr_at(lt, yt == E.BG)
    p = E.predict(lt, thr)
    keep = S["keep_t"]
    f1_L = E.metrics(yt[keep], p[keep])[0]
    le, _ = logits_of(model, ds_e)
    pe = E.predict(le, thr)
    y = S["Y"]["real31L"]; sel = y >= 0
    f1_ext = E.metrics(y[sel], pe[sel])[0]
    return {"thr": float(thr), "test_f1_L": float(f1_L), "ext_f1_real31L": float(f1_ext),
            "ext_fire_n": int((pe[S["isbg"]] != E.BG).sum())}


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="fe", choices=["fe", "b64f31"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-f1", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)

    import ai8x
    import eval_external as E
    import eval_v4 as Q
    EC = load_mod(os.path.join(REPO, "tools", "eval_confusion.py"), "eval_confusion")
    RS = load_mod(os.path.join(REPO, "tools", "npu_readback_sim.py"), "npu_readback_sim")
    fe_mod = load_mod(os.path.join(REPO, "models", "ai85net-safesound-fe.py"), "ai85net_safesound_fe_chk")
    cfg8 = "wave_fe8" + ("" if a.variant == "fe" else f"_{a.variant}")
    cfg0 = "wave_fe" + ("" if a.variant == "fe" else f"_{a.variant}")
    prefix = "safesound-v4d-fe" + ("" if a.variant == "fe" else f"-{a.variant}")
    res = {"variant": a.variant, "config": cfg8, "when": time.strftime("%F %T")}
    ck = E.checkpoint(prefix, 1)
    res["ckpt"] = os.path.relpath(ck, REPO)
    print(f"=== fe8 검산 — {a.variant}, ②′ 시드 1 체크포인트 {res['ckpt']}")

    # ── 모델 둘: 기준 ②′(wide) 과 fe8 그래프에 같은 체크포인트 ──
    base = EC.load_model(ck, 5, T, simulate=False, config=cfg0)
    m8 = EC.load_model(ck, 5, T, simulate=False, config=cfg8)
    assert m8.p1_bits8 and m8.lo_q8 == base.lo_q8 + fe_mod.LO8_SHIFT_Q8, (m8.lo_q8, base.lo_q8)
    thr_act = float(m8.fe.activation_threshold.detach())
    res["activation_threshold_loaded"] = thr_act
    if thr_act != 0.0:
        print(f"  [주의] fe.activation_threshold {thr_act} ≠ 0 — 검산에서는 0 으로 둔다 (wide 층의 값이라 8비트 층에는 뜻이 없다)")
        with torch.no_grad():
            m8.fe.activation_threshold.zero_()
    W = RS.p1_int_weights(base)                                   # (2n, 64, 8) int64 — 보드가 쓰는 정수 가중치
    w_max = float(base.fe.op.weight.detach().abs().max())
    shift_base = int(-(np.floor(np.log2(1.0 / w_max))))
    with torch.no_grad():
        m8.fe.op.weight.mul_(0.5 ** shift_base)                   # 출력 시프트 → 0, W_int 불변
    W8 = RS.p1_int_weights(m8)
    res["w_int_equal_after_rescale"] = bool(torch.equal(W, W8))
    print(f"  ②′ max|w| {w_max:.4f} (출력 시프트 {shift_base:+d}) → fe8 가중치 ×2^{-shift_base}; W_int 동일: {res['w_int_equal_after_rescale']}")
    assert res["w_int_equal_after_rescale"]

    # ── 데이터 ──
    tr = ai8x.normalize(args=argparse.Namespace(act_mode_8bit=False))
    import safesound as SS
    ds_tr = SS.SafeSound(os.path.join(T, "data", "SafeSoundV4"), "train", transform=tr, augment=False)
    idxs = sorted(set(np.linspace(0, len(ds_tr) - 1, N_SAMPLE).astype(int).tolist()))

    # ── 1. 그래프 ≡ 정수 경로 ──
    front = RS.Front(fe_mod, a.variant, W)
    front.g = np.ones(front.n, dtype=np.int64); front.g2 = np.ones(2 * front.n, dtype=np.int64); front.lg_g = np.zeros(front.n, dtype=np.int64)
    n_a = n_a_bad = n_a_half = n_half_cand = n_E0 = n_E = 0
    dq_all, q_base_all, q8_all = [], [], []
    sat = zero = n_used = 0
    for x, _ in batches(ds_tr, idxs):
        y = RS.y_int_of(x, W)                                     # (B, 2n, 249) int64
        q_sim, a_sim, E_sim = front.A_sum(y, 0)                   # 정수 경로 (반올림 → 포화 → |·|)
        q_b, _ = front.base(y)
        n_E0 += int((E_sim == 0).sum()); n_E += E_sim.size
        with torch.no_grad():
            B = x.size(0)
            s_ = x.transpose(1, 2).reshape(B, -1); x2 = s_.reshape(B, fe_mod.NPOS, fe_mod.FOLD).transpose(1, 2)
            a8 = (m8.fe(x2).abs().clamp(max=127.0 / 128.0) * 128).round().to(torch.int64).numpy()    # 8비트 단계 a (모델과 같은 순서)
            q8 = m8.intlog(m8.frontend(x)).to(torch.int64).numpy()
        used = a_sim[:, :, front.idx]
        sat += int((used == 127).sum()); zero += int((used == 0).sum()); n_used += used.size
        diff = a8 != a_sim
        n_a += a_sim.size; n_a_bad += int(diff.sum())
        # "음수이고 정확히 k+0.5" 자리 (y_int·2^(0−7) 의 소수부가 .5 ⇔ y_int mod 128 == 64) — ai8x Abs 활성화를 썼다면 여기서 어긋난다
        half_neg = (y < 0) & ((y % 128) == 64)
        n_half_cand += int(half_neg.sum()); n_a_half += int((diff & half_neg).sum())
        dq_all.append((q8 - q_sim).ravel()); q_base_all.append(q_b.ravel()); q8_all.append(q8.ravel())
    dq = np.concatenate(dq_all); qb = np.concatenate(q_base_all).astype(np.int64); q8v = np.concatenate(q8_all)
    res["graph_vs_int"] = {
        "n_windows": len(idxs), "a_total": n_a, "a_mismatch": n_a_bad, "a_mismatch_at_neg_half": n_a_half, "neg_half_positions": n_half_cand,
        "q_equal_frac": float((dq == 0).mean()), "q_abs_diff_max": int(np.abs(dq).max()), "q_abs_diff_gt1_frac": float((np.abs(dq) > 1).mean()),
        "sat_frac_a": sat / n_used, "zero_frac_a": zero / n_used, "E_zero_frame_frac": n_E0 / n_E,
        "rmse_db_vs_base": float(np.sqrt(((q8v - qb) ** 2).mean()) * RS.STEP_DB)}
    g = res["graph_vs_int"]
    print(f"     합이 0 인 (대역, 프레임) 비율 {100 * g['E_zero_frame_frac']:.2f} % (옛 시뮬레이션은 여기서 q 를 −84 로 냈다 — 2026-10-24 수정)")
    print(f"  1) 8비트 단계 a: 불일치 {g['a_mismatch']:,} / {g['a_total']:,} ({100 * g['a_mismatch'] / g['a_total']:.3f} %); "
          f"'음수·정확히 .5' 자리 {g['neg_half_positions']:,}개 중 불일치 {g['a_mismatch_at_neg_half']:,}")
    print(f"     q: 일치 {100 * g['q_equal_frac']:.3f} %, |Δq| 최대 {g['q_abs_diff_max']}, |Δq|>1 {100 * g['q_abs_diff_gt1_frac']:.4f} %; "
          f"포화 {100 * g['sat_frac_a']:.2f} % / 0 {100 * g['zero_frac_a']:.2f} %; 기준 q 대비 RMSE {g['rmse_db_vs_base']:.2f} dB")
    assert g["a_mismatch"] == 0, "8비트 단계가 정수 경로와 다르다 — 반올림·포화·|·| 순서를 확인할 것"
    assert g["q_abs_diff_max"] == 0, "q 가 정수 경로와 다르다 — 로그단 경계·LO 를 확인할 것"

    # ── 2. 기울기 ──
    m8.train()
    x, t = next(batches(ds_tr, idxs[:64], 64))
    m8.zero_grad()
    out = m8(x)
    loss = torch.nn.functional.cross_entropy(out, torch.from_numpy(t).long())
    loss.backward()
    gfe = m8.fe.op.weight.grad
    gcls = sum(float(p.grad.abs().sum()) for p in m8.cls.parameters() if p.grad is not None)
    res["grad"] = {"fe_weight_grad_abs_mean": float(gfe.abs().mean()), "fe_weight_grad_nonzero_frac": float((gfe != 0).float().mean()),
                   "cls_grad_abs_sum": gcls, "loss": float(loss)}
    print(f"  2) 기울기: fe.op.weight |grad| 평균 {res['grad']['fe_weight_grad_abs_mean']:.3e}, 0 아닌 비율 {100 * res['grad']['fe_weight_grad_nonzero_frac']:.1f} %; "
          f"p2 |grad| 합 {gcls:.3e}")
    assert res["grad"]["fe_weight_grad_nonzero_frac"] > 0.5
    m8.eval()

    # ── 4. 8비트 창 통계 (Gabor 초기값 vs 학습된 ②′) — float |y| 기준 ──
    def win_stats(model, tag):
        w = model.fe.op.weight.detach()
        vals = []
        for x, _ in batches(ds_tr, idxs[::8]):                    # 2,048 표본 중 256개, 학습셋 전체에 고르게
            B = x.size(0)
            s_ = x.transpose(1, 2).reshape(B, -1); x2 = s_.reshape(B, fe_mod.NPOS, fe_mod.FOLD).transpose(1, 2)
            with torch.no_grad():
                vals.append(torch.nn.functional.conv1d(x2, w).abs().flatten().numpy())
        v = np.concatenate(vals)
        d = {"w_max": float(w.abs().max()), "sat8": float((v >= 127.5 / 128).mean()), "zero8": float((v < 0.5 / 128).mean()),
             "y_p50": float(np.percentile(v, 50)), "y_p99": float(np.percentile(v, 99))}
        print(f"  4) {tag}: max|w| {d['w_max']:.3f}, |y| p50 {d['y_p50']:.4f} p99 {d['y_p99']:.3f}, 8비트 포화 {100 * d['sat8']:.2f} % / 0 {100 * d['zero8']:.2f} %")
        return d
    init = fe_mod.AI85SafeSoundFE(variant=a.variant, p1_bits8=True)
    res["window_stats"] = {"gabor_init": win_stats(init, "Gabor 초기값"), "trained_fe": win_stats(base, "학습된 ②′ (시드 1)")}

    # ── 5. ai8x apply_scales 의 '이전 층' 연결 (7.1 (다)) — fe 문턱 3 을 수동으로 넣고 cls.conv1 문턱이 어떻게 되는가 ──
    import torch.nn as nn
    link = {}
    for cut in (False, True):
        mm = fe_mod.AI85SafeSoundFE(variant=a.variant, p1_bits8=True, qat_link_cut=cut)
        for nm, mod in mm.named_modules():
            if isinstance(mod, ai8x.QuantizationAwareModule):
                mod.activation_threshold = nn.Parameter(torch.tensor(3.0 if nm == "fe" else 0.0), requires_grad=False)
        ai8x.apply_scales(mm)
        link["cut" if cut else "linked"] = {nm: float(mod.activation_threshold) for nm, mod in mm.named_modules()
                                            if isinstance(mod, ai8x.QuantizationAwareModule) and nm in ("fe", "cls.conv1", "cls.conv2")}
    res["qat_link"] = link
    print(f"  5) apply_scales: 연결 그대로 → cls.conv1 문턱 {link['linked']['cls.conv1']:+.0f} (fe 3 을 뺀다; ②′ 원 학습이 이랬다), "
          f"qat_link_cut → {link['cut']['cls.conv1']:+.0f}")
    assert link["linked"]["cls.conv1"] == -3.0 and link["cut"]["cls.conv1"] == 0.0

    # ── 3. 그대로 F1 ──
    if not a.no_f1:
        S = Q.setup(); V = Q.V
        ds_t = EC.make_dataset(cfg8, V.TEST, "test", tr)
        ds_e = EC.make_dataset(cfg8, S["ext_root"], "test", tr)
        res["asis_f1"] = score_model(m8, ds_t, ds_e, S, E)
        r = res["asis_f1"]
        print(f"  3) 그대로 F1 (②′ 가중치 ×1/2, fe8 그래프, 재학습 없음): 시험 {r['test_f1_L']:.4f}  외부 {r['ext_f1_real31L']:.4f}  (문턱 {r['thr']:.2f})")
        ref = os.path.join(REPO, "data", "external_eval", "fe_eval" + ("" if a.variant == "fe" else f"_{a.variant}") + ".json")
        if os.path.isfile(ref):
            rr = json.load(open(ref, encoding="utf-8"))["seeds"][0]
            res["base_f1_ref"] = {"test_f1_L": rr["test_f1_L"], "ext_f1_real31L": rr["ext_f1_real31L"]}
            print(f"     ②′ 기준(기록): 시험 {rr['test_f1_L']:.4f}  외부 {rr['ext_f1_real31L']:.4f}")
    out_path = os.path.join(OUT, f"fe8_check{'' if a.variant == 'fe' else '_' + a.variant}.json")
    json.dump(res, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"=== 저장 {os.path.relpath(out_path, REPO)}")


if __name__ == "__main__":
    main()
