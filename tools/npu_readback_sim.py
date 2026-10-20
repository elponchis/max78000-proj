#!/usr/bin/env python3
"""NPU 내부 합산 + 8비트 출력 — **PC 시뮬레이션** (보드 미접촉, 2026-10-21).

②′(와 변형 b64/f31/b64f31)의 CPU 몫 가운데 "가속기 메모리 읽기 + |·| + 8위치 합" (실측 67 %)
을 NPU 층(Abs → 8위치 합/평균 → cos+sin 합성곱)으로 옮기는 안 A/B/C 의
  · 특징 충실도 (포화율 · 0 비율 · 대역별 상관 · 로그 영역 오차) — 학습 창 표본, 무증강
  · p2 그대로의 시험·외부 F1, p2 만 짧게 미세조정한 F1
을 기존 체크포인트(시드 1~3)로 잰다. 설계·사전 등록·비용 모델: docs/results/npu-readback-design.md.

정수 경로는 펌웨어와 같은 규약이다 (fe_board_prep.py 3단계와 같은 식):
  y[o][p] = Σ W_int[o][r][j] X[r][p+j]  (int32, p1 wide)   — W_int 는 QAT 가중치의 int8 값 (q8 체크포인트와 대조)
  기준(int32 경로): E = Σ_{8위치} (|y_cos| + |y_sin|) → q = to_int8(log2_q8(E), LO_Q8, SPAN_Q8)
  안 A/B/C: y → 출력 시프트 s 로 int8 (floor(v·2^(s−7)+0.5), 포화) → |·| (127 상한) → 합/평균 → 로그 (상수 보정은 CPU)

단계:
  --stage sim     변형·시드별 통계 + F1(그대로) + 미세조정 F1 → docs/results/npu-readback-sim/<변형>_s<시드>.json
  --stage report  JSON 을 모아 표·비용 추정·그림 → summary.md, fig_readback.png (논문 그림은 건드리지 않는다)

실행 (WSL2): export OPENBLAS_NUM_THREADS=1; ~/ai8x-training/venv/bin/python tools/npu_readback_sim.py --stage sim --variant fe --seed 1
"""
import argparse
import copy
import csv
import glob
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
from log2_q8 import log2_q8, to_int8                # noqa: E402

VARIANTS = {   # fe_ext_eval.VARIANTS 와 같다
    "fe":     ("wave_fe",        "safesound-v4d-fe",        "②′"),
    "b64":    ("wave_fe_b64",    "safesound-v4d-fe-b64",    "②′ b64"),
    "f31":    ("wave_fe_f31",    "safesound-v4d-fe-f31",    "②′ f31"),
    "b64f31": ("wave_fe_b64f31", "safesound-v4d-fe-b64f31", "②′ b64f31"),
}
SCHEMES = ["base", "A-sum", "A-avg", "B", "C"]
SHIFTS = list(range(-6, 9))          # 출력 시프트 후보 (디바이스 −15..15 안)
B_STEP = 4                           # 안 B: 병렬 복제의 시프트 간격 (24 dB)
B_SAT = 1016                         # 안 B: 이 값 미만이면 그 복제를 쓴다 (최대 16·127 = 2032 의 절반)
N_TRAIN_SAMPLE = 2048                # 충실도 통계용 학습 창 표본 (무증강, 클래스 교차 균등 간격)
FT_EPOCHS, FT_LR, FT_BATCH = 3, 1e-4, 128
STEP_DB = 70.0 / 255.0               # int8 한 단계 = dB


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ───────────────────────── 정수 프론트엔드 ─────────────────────────
def p1_int_weights(model):
    """QAT 그래프가 쓰는 int8 가중치 (2n, 64, 8). fe 는 wide 층이라 디바이스가 출력 시프트를 걸지 않는다 —
    y_int = Σ W_int·x_int 이고 학습 그래프의 E·2^14 와 같다 (fe_board_prep 3단계가 시드 1 로 확인)."""
    import torch
    fe = model.fe
    w = fe.op.weight.detach()
    out_shift = fe.calc_out_shift(torch.flatten(w), fe.output_shift.detach())
    wq = fe.clamp_weight(fe.quantize_weight(w.mul(fe.calc_weight_scale(out_shift))))
    W = (wq * 128).round().to(torch.int64)
    assert W.abs().max() <= 128
    return W


def y_int_of(x, W):
    """x (B,128,128) float [-1,1) → p1 wide 출력 y (B, 2n, 249) int64. float32 conv 는 |부분합| < 2^24 라 정확하다."""
    import torch
    import torch.nn.functional as F
    B = x.shape[0]
    xi = (x * 128).round()
    s = xi.transpose(1, 2).reshape(B, -1)                      # x[r][c] = s[128c + r]
    x2 = s.reshape(B, 256, 64).transpose(1, 2).contiguous()    # X[r][p] = s[64p + r]
    y = F.conv1d(x2, W.float(), stride=1)
    return y.round().to(torch.int64).numpy()


def band_gain(W, n_band):
    """안 A 의 대역별 고정 배율 (2 의 거듭제곱, 가중치 여유 안). cos/sin 두 채널의 여유 중 작은 쪽."""
    m = np.abs(W.numpy()).reshape(n_band, 2, -1).max(-1)       # (n, 2)
    h = np.floor(np.log2(127.0 / np.maximum(m, 1))).min(1).clip(min=0)
    return (2.0 ** h).astype(np.int64)                        # (n,)


def abs8(y, s, g2):
    """y (B,2n,P) int64 → 디바이스 8비트 출력 → |·|. g2 (2n,) 배율. 시프트 s: floor(v·2^(s−7)+0.5) → 포화 → |·| (127 상한)."""
    v = y * g2[None, :, None]
    if s < 7:
        r = np.floor_divide(v + (1 << (6 - s)), 1 << (7 - s))    # floor(v·2^(s−7) + 0.5), 음수도 바닥
    elif s == 7:
        r = v
    else:
        r = v << (s - 7)
    r = np.clip(r, -128, 127)
    return np.minimum(np.abs(r), 127)


class Front:
    """변형 하나의 정수 프론트엔드. q(y) 를 안별로 만든다."""

    def __init__(self, fe_mod, variant, W):
        V = fe_mod.VARIANTS[variant]
        self.n, self.F = V["n_band"], V["n_frame"]
        self.ppf, self.pool, self.lo = V["pos_per_frame"], V["pool_pos"], V["lo_q8"]
        self.span = fe_mod.SPAN_Q8
        self.idx = (np.arange(self.F) * self.ppf)[:, None] + np.arange(self.pool)[None, :]
        self.W = W
        self.g = band_gain(W, self.n)                         # (n,)
        self.g2 = np.repeat(self.g, 2)                        # (2n,)
        self.lg_g = (256 * np.log2(self.g)).astype(np.int64)  # 배율 보정 (Q8, 2 의 거듭제곱이라 정확)

    def pooled(self, a):
        """a (B,2n,P) → cos/sin 각각의 8위치 합 (B,n,F)"""
        c = a[:, 0::2][:, :, self.idx].sum(-1)
        s = a[:, 1::2][:, :, self.idx].sum(-1)
        return c, s

    def q_of(self, lg):
        return to_int8(lg - self.lg_g[None, :, None], self.lo, self.span)

    def base(self, y):
        mag = np.abs(y[:, 0::2]) + np.abs(y[:, 1::2])
        E = mag[:, :, self.idx].sum(-1)
        return to_int8(log2_q8(E), self.lo, self.span), E

    def A_sum(self, y, s):
        a = abs8(y, s, self.g2)
        c, sn = self.pooled(a)
        E = c + sn                                            # ≤ 16·127 = 2032 (k=8 합성곱 합, wide)
        return self.q_of(log2_q8(E) + 256 * (7 - s)), a, E

    def A_avg(self, y, s):
        a = abs8(y, s, self.g2)
        c, sn = self.pooled(a)
        E = c // 8 + sn // 8                                  # 평균 풀링 floor, 1×1 wide
        return self.q_of(log2_q8(E) + 256 * (10 - s)), a, E

    def C(self, y, s):
        a = abs8(y, s, self.g2)
        c, sn = self.pooled(a)
        v = np.minimum((c // 8 + sn // 8 + 1) // 2, 127)      # 1×1 가중치 0.5·(c+s) → int8 출력 (256칸 LUT 가 로그를 맡는다)
        return self.q_of(log2_q8(v) + 256 * (11 - s)), a, v

    def B(self, y, s):
        """복제 3개 (s+4, s, s−4). (b,f) 마다 포화하지 않은 가장 높은 배율의 복제를 쓴다."""
        shifts = [s + B_STEP, s, s - B_STEP]
        Es, As = [], []
        for sk in shifts:
            a = abs8(y, sk, self.g2)
            c, sn = self.pooled(a)
            Es.append(c + sn); As.append(a)
        pick = np.full(Es[0].shape, len(shifts) - 1, dtype=np.int64)
        for k in range(len(shifts) - 2, -1, -1):
            pick = np.where(Es[k] < B_SAT, k, pick)
        E = np.choose(pick, Es)
        sk = np.choose(pick, [np.full(pick.shape, v, dtype=np.int64) for v in shifts])
        q = self.q_of(log2_q8(E) + 256 * (7 - sk))
        pk = np.broadcast_to(pick[..., None], pick.shape + (self.pool,))
        a_sel = np.concatenate([np.choose(pk, [A[:, 0::2][:, :, self.idx] for A in As]),
                                np.choose(pk, [A[:, 1::2][:, :, self.idx] for A in As])], axis=1)   # 선택된 복제의 cos·sin 값
        return q, a_sel, E, pick

    def run(self, scheme, y, s):
        if scheme == "base":
            q, E = self.base(y); return q, None
        if scheme == "A-sum":
            return self.A_sum(y, s)[:2]
        if scheme == "A-avg":
            return self.A_avg(y, s)[:2]
        if scheme == "C":
            return self.C(y, s)[:2]
        if scheme == "B":
            q, a, _, _ = self.B(y, s); return q, a
        raise ValueError(scheme)

    def stats(self, scheme, y, s, q_base):
        q, a = self.run(scheme, y, s)
        if scheme == "B":
            used = a                                          # 선택된 복제의 cos 값 (B,n,F,8)
        else:
            used = a[:, :, self.idx]                          # 프레임에 들어간 값만 (B,2n,F,8)
        d = q.astype(np.int64) - q_base.astype(np.int64)
        qb = q_base.reshape(-1, self.n, self.F).transpose(1, 0, 2).reshape(self.n, -1).astype(np.float64)
        qs = q.reshape(-1, self.n, self.F).transpose(1, 0, 2).reshape(self.n, -1).astype(np.float64)
        corr = []
        for b in range(self.n):
            sb, ss = qb[b].std(), qs[b].std()
            corr.append(float(np.corrcoef(qb[b], qs[b])[0, 1]) if sb > 0 and ss > 0 else 0.0)
        return {"shift": s, "sat": float((used == 127).mean()), "zero": float((used == 0).mean()),
                "rmse_db": float(np.sqrt((d ** 2).mean()) * STEP_DB), "mae_steps": float(np.abs(d).mean()),
                "frac_gt2": float((np.abs(d) > 2).mean()), "corr_mean": float(np.mean(corr)), "corr_min": float(np.min(corr)),
                "q_sat_hi": float((q == 127).mean()), "q_sat_lo": float((q == -128).mean()),
                "qb_sat_hi": float((q_base == 127).mean()), "qb_sat_lo": float((q_base == -128).mean())}


# ───────────────────────── 데이터·평가 ─────────────────────────
def windows_of(ds, idxs, bs=256):
    import torch
    for i in range(0, len(idxs), bs):
        xs, ts = [], []
        for j in idxs[i:i + bs]:
            x, t = ds[j]
            xs.append(x); ts.append(int(t) if not hasattr(t, "__len__") else int(t[0]))
        yield torch.stack(xs), np.array(ts)


def feats(front, ds, W, schemes, shifts, idxs=None):
    """ds 전체(또는 idxs) 의 안별 q (N, n, F) int8 과 정답."""
    idxs = list(range(len(ds))) if idxs is None else idxs
    out = {k: [] for k in schemes}
    ys = []
    for x, t in windows_of(ds, idxs):
        y = y_int_of(x, W)
        for k in schemes:
            out[k].append(front.run(k, y, shifts.get(k, 0))[0])
        ys.append(t)
    return {k: np.concatenate(v) for k, v in out.items()}, np.concatenate(ys)


def logits_of(cls, q, bs=512):
    import torch
    outs = []
    with torch.no_grad():
        for i in range(0, len(q), bs):
            x = torch.from_numpy(q[i:i + bs].astype(np.float32) / 128.0).unsqueeze(1)
            outs.append(cls(x).numpy())
    return np.concatenate(outs)


def score(cls, qt, yt, qe, S, E):
    lt = logits_of(cls, qt)
    thr = E.thr_at(lt, yt == E.BG)
    p = E.predict(lt, thr)
    keep = S["keep_t"]
    f1_L = E.metrics(yt[keep], p[keep])[0]
    le = logits_of(cls, qe)
    pe = E.predict(le, thr)
    y = S["Y"]["real31L"]; sel = y >= 0
    f1_ext = E.metrics(y[sel], pe[sel])[0]
    return {"thr": float(thr), "test_f1_L": float(f1_L), "ext_f1_real31L": float(f1_ext),
            "ext_fire_n": int((pe[S["isbg"]] != E.BG).sum())}


def finetune(model, front, W, scheme, s, train_ds, seed, log):
    """p2(cls) 만 FT_EPOCHS 에폭. 프론트엔드는 정수 경로(고정) — 기울기가 흐를 것이 없다. 손실 CE + 클래스 가중치(역빈도)."""
    import torch
    import torch.nn as nn
    import safesound as SS
    torch.manual_seed(seed); np.random.seed(seed)
    cls = copy.deepcopy(model.cls)
    cls.train()
    params = [p for p in cls.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=FT_LR)
    wts = torch.tensor(SS.class_weights(train_ds.root, "train"), dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=wts)
    g = torch.Generator(); g.manual_seed(seed)
    dl = torch.utils.data.DataLoader(train_ds, batch_size=FT_BATCH, shuffle=True, num_workers=4, generator=g, drop_last=True)
    for ep in range(FT_EPOCHS):
        t0, tot, n = time.time(), 0.0, 0
        for x, t in dl:
            t = t if t.dim() == 1 else t[:, 0].long()
            y = y_int_of(x, W)
            q = front.run(scheme, y, s)[0]
            xq = torch.from_numpy(q.astype(np.float32) / 128.0).unsqueeze(1)
            opt.zero_grad()
            loss = lossf(cls(xq), t.long())
            loss.backward(); opt.step()
            tot += float(loss) * len(t); n += len(t)
        log(f"    [{scheme}] ft 에폭 {ep + 1}/{FT_EPOCHS} 손실 {tot / n:.4f} ({time.time() - t0:.0f}s)")
    cls.eval()
    return cls


def stage_sim(a):
    import torch
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    config, prefix, name = VARIANTS[a.variant]
    import eval_external as E
    import eval_v4 as Q
    V = Q.V
    EC = load_mod(os.path.join(REPO, "tools", "eval_confusion.py"), "eval_confusion")
    fe_mod = load_mod(os.path.join(REPO, "models", "ai85net-safesound-fe.py"), "ai85net_safesound_fe_sim")
    import ai8x
    out_path = os.path.join(OUT, f"{a.variant}_s{a.seed}.json")
    logf = open(os.path.join(OUT, f"{a.variant}_s{a.seed}.log"), "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    ck = E.checkpoint(prefix, a.seed)
    log(f"=== {name} 시드 {a.seed}  {os.path.relpath(ck, REPO)}  {time.strftime('%F %T')}")
    model = EC.load_model(ck, 5, T, simulate=False, config=config)
    W = p1_int_weights(model)
    q8 = os.path.join(REPO, "data", "synth", f"{prefix}-p1-q8.pth.tar")
    if a.seed == 1 and os.path.isfile(q8):
        Wq = torch.load(q8, map_location="cpu", weights_only=False)["state_dict"]["fe.op.weight"].to(torch.int64)
        assert torch.equal(Wq, W), "QAT int8 가중치가 q8 체크포인트와 다르다 — 스케일 규약 확인"
        log("  W_int == q8 체크포인트 (보드가 쓴 가중치와 같다)")
    front = Front(fe_mod, a.variant, W)
    log(f"  대역 {front.n} × 프레임 {front.F}, LO_Q8 {front.lo}; 대역 배율 g: " +
        ", ".join(f"{int(v)}" for v in front.g))
    tr = ai8x.normalize(args=argparse.Namespace(act_mode_8bit=False))
    S = Q.setup()
    ds_t = EC.make_dataset(config, V.TEST, "test", tr)
    ds_e = EC.make_dataset(config, S["ext_root"], "test", tr)
    assert np.array_equal(np.array([m[0] for m in ds_t.meta]).astype(str), S["test_clips"].astype(str))
    import safesound as SS
    root_tr = os.path.join(T, "data", "SafeSoundV4")
    ds_tr_stat = SS.SafeSound(root_tr, "train", transform=tr, augment=False)
    ds_tr = SS.SafeSound(root_tr, "train", transform=tr, mix_prob=SS.MIX_PROB, mix_bg=True)   # v4 학습 증강 (KD 제외)
    res = {"variant": a.variant, "name": name, "seed": a.seed, "ckpt": os.path.relpath(ck, REPO), "n_band": front.n, "n_frame": front.F,
           "lo_q8": front.lo, "gain": [int(v) for v in front.g], "shifts": {}, "sweep": {}, "stats": {}, "asis": {}, "ft": {},
           "ft_recipe": {"epochs": FT_EPOCHS, "lr": FT_LR, "batch": FT_BATCH, "loss": "CE + 역빈도 클래스 가중치", "data": "SafeSoundV4 train 16,669 창, v4 증강(게인·시프트·배경 혼합), KD 없음"}}

    # ── 1. 학습 표본으로 시프트 선택 (사전 등록 규칙: 기준 q 와의 RMSE(dB) 최소) ──
    n_tr = len(ds_tr_stat)
    idxs = sorted(set(np.linspace(0, n_tr - 1, N_TRAIN_SAMPLE).astype(int).tolist()))
    ys = []
    for x, _ in windows_of(ds_tr_stat, idxs):
        ys.append(y_int_of(x, W))
    y_tr = np.concatenate(ys)
    q_b, E_b = front.base(y_tr)
    res["train_sample"] = {"n": int(len(y_tr)), "E_p50": float(np.median(E_b)), "E_p99": float(np.percentile(E_b, 99)),
                           "E_max": int(E_b.max()), "E_p01": float(np.percentile(E_b, 1))}
    shifts = {"base": 0}
    for k in ("A-sum", "A-avg", "C"):
        sw = {s: front.stats(k, y_tr, s, q_b) for s in SHIFTS}
        best = min(sw, key=lambda s: sw[s]["rmse_db"])
        shifts[k] = best
        res["sweep"][k] = {str(s): {kk: round(vv, 5) for kk, vv in sw[s].items()} for s in SHIFTS}
        res["stats"][k] = sw[best]
        log(f"  {k:6s} 시프트 {best:+d}: RMSE {sw[best]['rmse_db']:.3f} dB, 포화 {100 * sw[best]['sat']:.2f}%, 0 {100 * sw[best]['zero']:.2f}%, "
            f"상관 평균 {sw[best]['corr_mean']:.4f} (최소 {sw[best]['corr_min']:.4f}), |Δq|>2 {100 * sw[best]['frac_gt2']:.2f}%")
    shifts["B"] = shifts["A-sum"]
    res["stats"]["B"] = front.stats("B", y_tr, shifts["B"], q_b)
    st = res["stats"]["B"]
    log(f"  B      시프트 {shifts['B']:+d}±{B_STEP}: RMSE {st['rmse_db']:.3f} dB, 포화 {100 * st['sat']:.2f}%, 0 {100 * st['zero']:.2f}%, 상관 {st['corr_mean']:.4f}")
    res["shifts"] = shifts
    res["stats"]["base"] = {"qb_sat_hi": float((q_b == 127).mean()), "qb_sat_lo": float((q_b == -128).mean())}

    # ── 2. p2 그대로: 시험·외부 F1 ──
    qt, yt = feats(front, ds_t, W, SCHEMES, shifts)
    qe, _ = feats(front, ds_e, W, SCHEMES, shifts)
    assert np.array_equal(yt, S["test_y"])
    for k in SCHEMES:
        res["asis"][k] = score(model.cls, qt[k], yt, qe[k], S, E)
        log(f"  그대로 {k:6s}: 시험 {res['asis'][k]['test_f1_L']:.4f}  외부 {res['asis'][k]['ext_f1_real31L']:.4f}  (문턱 {res['asis'][k]['thr']:.2f})")
    ref = os.path.join(REPO, "data", "external_eval", "fe_eval" + ("" if a.variant == "fe" else f"_{a.variant}") + ".json")
    if os.path.isfile(ref):
        r = json.load(open(ref, encoding="utf-8"))["seeds"][a.seed - 1]
        d1, d2 = res["asis"]["base"]["test_f1_L"] - r["test_f1_L"], res["asis"]["base"]["ext_f1_real31L"] - r["ext_f1_real31L"]
        res["asis"]["base"]["ref_diff"] = [d1, d2]
        log(f"  기준 대조: 기록 시험 {r['test_f1_L']:.4f} / 외부 {r['ext_f1_real31L']:.4f} → 차 {d1:+.4f} / {d2:+.4f}")
        assert abs(d1) <= 0.003 and abs(d2) <= 0.003, "정수 경로가 기록된 ②′ 평가와 다르다"
    json.dump(res, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ── 3. p2 미세조정 (기준 포함 — 같은 조리법이 기준 F1 을 얼마나 움직이는지가 대조군) ──
    if not a.no_ft:
        for k in SCHEMES:
            if a.ft_schemes and k not in a.ft_schemes:
                continue
            cls = finetune(model, front, W, k, shifts[k], ds_tr, a.seed, log)
            res["ft"][k] = score(cls, qt[k], yt, qe[k], S, E)
            log(f"  미세조정 {k:6s}: 시험 {res['ft'][k]['test_f1_L']:.4f}  외부 {res['ft'][k]['ext_f1_real31L']:.4f}")
            json.dump(res, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"=== 끝 {time.strftime('%F %T')} → {os.path.relpath(out_path, REPO)}")


# ───────────────────────── 보고: 표·비용·그림 ─────────────────────────
# 비용 모델 상수 (docs/results/npu-readback-design.md 4절). 전부 **추정**이다.
C_RD_WORD = 32          # 가속기 메모리 1워드 읽기만 (사이클). 실측 40.35 = 읽기 + |·| + 합 + 루프; APB 쓰기 실측 27.5
C_LOG = 190             # 정수 로그 1회 (실측, 64bit 나눗셈 포함)
C_LOG_FIX = 60          # 64bit 나눗셈을 32bit 로 바꾼 정수 로그 (추정 — 보드 실측 전)
C_LUT = 15              # 256칸 LUT + 대역 오프셋 + 포화 + 저장 (추정)
C_FIXED = 1400          # 버퍼 밀기 + p2 입력 포장 (실측 직선의 절편)
F_CORE = 100e6
CFG_PER_LAYER_MS = 0.011   # 층 하나의 레지스터 설정 (cfg2 0.086 ms / 8층)


def cost(L, variant_V, scheme, log_fix):
    """변형의 보드 실측 지연 L(detail) 에서 안별 예상 지연 (ms) — 전처리(CPU)·추론(NPU) 만 바뀐다."""
    n, F, ppf = variant_V["n_band"], variant_V["n_frame"], variant_V["pos_per_frame"]
    fph = 4 if ppf == 16 else 8                     # 판단당 새 프레임
    P = 57 if ppf == 16 else 64                     # p1 출력 위치 (판단당)
    n_val = n * fph
    d = L["detail"]
    c_log = C_LOG_FIX if log_fix else C_LOG
    if scheme == "base":
        words = 2 * n * fph * 8                         # fe 2,048 / b64 4,096 / f31 4,096 / b64f31 8,192 (int32, cos·sin·8위치)
        cpu = d["cpu"]
        if log_fix:
            cpu = cpu + n_val * (c_log - C_LOG) / F_CORE * 1e3   # 로그만 싸진다 (실측 CPU 몫에서 차감)
        return {"pre": cpu, "infer1": d["infer1"], "cfg1": d["cfg1"], "words": words, "n_log": n_val}
    mac_p1 = 64 * 2 * n * 8 * P
    if scheme in ("A-sum", "B"):
        mac_b = 2 * n * n * 8 * (P - 7)
    else:
        mac_b = 2 * n * n * (P // 8)
    reps = 3 if scheme == "B" else 1
    infer1 = reps * (d["infer1"] * (1 + mac_b / mac_p1))
    cfg1 = d["cfg1"] + CFG_PER_LAYER_MS * (reps * 2 - 1 + (1 if scheme in ("A-avg", "C") else 0))
    if scheme == "C":
        words = -(-n_val // 4); cyc = C_FIXED + words * C_RD_WORD + n_val * C_LUT
    else:
        words = reps * n_val; cyc = C_FIXED + words * C_RD_WORD + n_val * c_log + (n_val * 6 if scheme == "B" else 0)
    return {"pre": cyc / F_CORE * 1e3, "infer1": infer1, "cfg1": cfg1, "words": words, "n_log": n_val}


def stage_report(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    fe_mod = load_mod(os.path.join(REPO, "models", "ai85net-safesound-fe.py"), "ai85net_safesound_fe_rep")
    N = json.load(open(os.path.join(REPO, "docs", "paper", "numbers.json"), encoding="utf-8"))
    C = json.load(open(os.path.join(REPO, "docs", "paper", "paper_config.json"), encoding="utf-8"))
    for p in C.get("fonts", []):                      # 논문 그림과 같은 한글 글꼴
        if os.path.isfile(p):
            fm.fontManager.addfont(p)
            plt.rcParams["font.family"] = fm.FontProperties(fname=p).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    PS = {k: v for k, v in C["power_sources"].items() if v.get("use_in_curve")}
    files = sorted(glob.glob(os.path.join(OUT, "*_s*.json")))
    R = [json.load(open(f, encoding="utf-8")) for f in files]
    lines = ["# NPU 내부 합산 + 8비트 출력 — PC 시뮬레이션 결과 (자동 생성: tools/npu_readback_sim.py --stage report)", "",
             f"생성 {time.strftime('%F %T')}. 보드 미접촉. 에너지·지연은 **추정**(기존 보드 실측 × 사이클 모델). 설계: npu-readback-design.md.", ""]
    rows_tab = []
    pts = []                                      # 그림용
    for variant, (config, prefix, name) in VARIANTS.items():
        rs = [r for r in R if r["variant"] == variant]
        if not rs:
            continue
        V = fe_mod.VARIANTS[variant]
        L = N[f"latency.{name}"]["value"]
        lines.append(f"## {name} (대역 {V['n_band']} × 프레임 {V['n_frame']}, 시드 {sorted(r['seed'] for r in rs)})"); lines.append("")
        lines.append("| 안 | 시프트 | 읽기 워드 (지금 → 안) | 포화율 | 0 비율 | 상관(평균/최소) | RMSE dB | F1 시험 그대로 | F1 시험 미세조정 | F1 외부 그대로 | F1 외부 미세조정 | 전처리 ms | NPU p1 ms | 판단 ms | μJ (데이터시트/선행연구/Moosmann) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for k in SCHEMES:
            cs = cost(L, V, k, log_fix=False)
            pre, infer = cs["pre"], L["infer"] - L["detail"]["infer1"] + cs["infer1"]
            load = L["load"] - L["detail"]["cfg1"] + cs["cfg1"]
            tot = pre + load + infer + L["out"]
            Es = {s: pre * PS[s]["P_load_mW"] + (load + L["out"]) * PS[s]["P_load_mW"] + infer * PS[s]["P_npu_mW"] for s in PS}
            f = lambda key, kk: (np.mean([r[key][k][kk] for r in rs if k in r[key]]) if any(k in r[key] for r in rs) else np.nan)
            st = rs[0]["stats"].get(k, {})
            sh = rs[0]["shifts"].get(k, 0)
            w0 = cost(L, V, "base", False)["words"]
            rows_tab.append((name, k, sh, w0, cs["words"], st.get("sat"), st.get("zero"), f("asis", "test_f1_L"), f("ft", "test_f1_L"), f("asis", "ext_f1_real31L"), f("ft", "ext_f1_real31L"), tot, Es))
            lines.append(f"| {k} | {sh:+d}{'±4' if k == 'B' else ''} | {w0:,} → {cs['words']:,} | "
                         + (f"{100 * st['sat']:.2f}% | {100 * st['zero']:.2f}% | {st['corr_mean']:.3f}/{st['corr_min']:.3f} | {st['rmse_db']:.2f}" if "sat" in st else "— | — | — | —")
                         + f" | {f('asis', 'test_f1_L'):.3f} | {f('ft', 'test_f1_L'):.3f} | {f('asis', 'ext_f1_real31L'):.3f} | {f('ft', 'ext_f1_real31L'):.3f} | "
                         f"{pre:.3f} | {cs['infer1']:.3f} | {tot:.2f} | {Es['datasheet']:.1f} / {Es['literature']:.1f} / {Es['moosmann']:.1f} |")
            pts.append((name, k, Es, f("ft", "test_f1_L"), f("asis", "test_f1_L"), f("ft", "ext_f1_real31L"), len(rs)))
        # 로그 나눗셈 고침만 (특징 불변)
        cs = cost(L, V, "base", log_fix=True)
        lines.append(f"| base + 로그 32bit 나눗셈 (특징 동일) | 0 | {cs['words']:,} | — | — | — | — | = base | = base | = base | = base | {cs['pre']:.3f} | {cs['infer1']:.3f} | "
                     f"{cs['pre'] + L['load'] + L['infer'] + L['out']:.2f} | " + " / ".join(f"{(cs['pre'] * PS[s]['P_load_mW'] + (L['load'] + L['out']) * PS[s]['P_load_mW'] + L['infer'] * PS[s]['P_npu_mW']):.1f}" for s in PS) + " |")
        lines.append("")
        lines.append(f"지금(보드 실측): 전처리 {L['pre']:.3f} ms, 판단 {L['total']:.2f} ms, μJ " + " / ".join(f"{L['E'][s]:.1f}" for s in PS) if "E" in L else "")
        lines.append("")
    # ── 그림: 기존 CPU 곡선·NPU 점(실측) + 안별 점(추정) ──
    tab = list(csv.DictReader(l for l in open(os.path.join(REPO, "docs", "paper", "table2_main.csv"), encoding="utf-8-sig") if not l.startswith("#")))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ecol, fcol = next(c for c in tab[0] if "데이터시트" in c or "e_datasheet" in c.lower()), next(c for c in tab[0] if "시험" in c and "F1" in c)
    for r in tab:
        try:
            e, f1 = float(r[ecol]), float(str(r[fcol]).split("±")[0])
        except (ValueError, KeyError):
            continue
        nm = r[next(iter(tab[0]))]
        cpu = "CPU" in nm and "추론" not in nm
        ax.scatter(e, f1, marker="o" if cpu else "s", s=36, color="#1f77b4" if cpu else ("#d62728" if "NPU" in nm else "k"), zorder=3)
        ax.annotate(nm.replace("CPU log-mel, ", ""), (e, f1), textcoords="offset points", xytext=(5, 4), fontsize=7)
    mk = {"A-sum": "^", "A-avg": "v", "B": "D", "C": "P"}
    for name, k, Es, f1ft, f1as, _, nseed in pts:
        if k == "base" or np.isnan(f1as):
            continue
        yv = f1ft if not np.isnan(f1ft) else f1as
        ax.errorbar(Es["datasheet"], yv, xerr=[[Es["datasheet"] - min(Es.values())], [max(Es.values()) - Es["datasheet"]]],
                    fmt=mk[k], mfc="none", color="#ff7f0e", ms=7, capsize=2, lw=0.8, zorder=4)
        ax.annotate(f"{name.replace('②′', '').strip() or 'fe'} {k}", (Es["datasheet"], yv), textcoords="offset points", xytext=(4, -9), fontsize=6, color="#ff7f0e")
    ax.set_xscale("log"); ax.set_xlabel("판단 1회 추정 에너지 (μJ, 로그축; 가로 막대 = 세 전력 출처 범위)"); ax.set_ylabel("macro-F1 시험 데이터")
    ax.set_title("NPU 내부 합산 안 (주황 속 빈 마커 = 추정, p2 미세조정 F1) vs 기존 보드 실측 점", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    plt.rcParams["font.family"] = plt.rcParams["font.family"]
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_readback.png"), dpi=160)
    lines.append("그림: fig_readback.png (논문 그림 아님)")
    open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sim", "report"], default="sim")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="fe")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-ft", action="store_true")
    ap.add_argument("--ft-schemes", nargs="*", default=None)
    a = ap.parse_args()
    if a.stage == "sim":
        stage_sim(a)
    else:
        stage_report(a)


if __name__ == "__main__":
    main()
