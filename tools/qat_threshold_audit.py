"""QAT 활성화 문턱 감사 — 논문의 전 구성에서 분류기 층별 저장 문턱(activation_threshold)·출력 포화(잘림)·0 비율을 표로 낸다.

배경 (docs/results/npu-readback-design.md 7.1 (다)): ai8x `apply_scales` 는 torch.fx 로 그래프를 거슬러 "이전 conv 층" 의 문턱을
다음 층 문턱에서 뺀다. ②′ 는 fe(p1) 와 cls.conv1 사이에 CPU 정수 로그가 있어 그 보상이 틀리고, conv1 문턱이 −3 (출력 ×8) 이 됐다.
같은 오류가 다른 구성에도 있는지 — 구조적으로는 **두 ai8x conv 사이에 스케일을 지우는 비-ai8x 연산이 있는 구성**만 해당한다:
  · CPU log-mel (가·다·간격 400·①′): p1 이 CPU 라 모델 안에 conv1 이전 층이 없다 → 연결 자체가 없다.
  · ④ 원본 파형, (a) NPU 근사 log: 층이 전부 ai8x conv 로 이어지고 사이 연산이 선형(ReLU·풀링·구간선형)이라 보상이 **맞다**.
  · ②′ 계열: 사이에 |·|·합·정수 로그·/128 → 보상이 틀리다.
그래도 말로만 두지 않고 저장된 문턱과 실제 포화율을 전 구성에서 **같은 자로** 잰다. 보드·학습 없음 — 기존 체크포인트, 시험 창 256개.

실행 (WSL2):  OPENBLAS_NUM_THREADS=1 ~/ai8x-training/venv/bin/python tools/qat_threshold_audit.py --threads 4
출력: docs/results/qat-threshold-audit.md, docs/results/npu-readback-sim/qat_threshold_audit.json
"""
import os, sys, json, argparse, importlib.util, datetime
import numpy as np, torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = os.path.expanduser("~/ai8x-training")
sys.path.insert(0, os.path.join(REPO, "tools")); sys.path.insert(0, T); sys.path.insert(0, os.path.join(T, "datasets"))


def load_mod(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


EC = load_mod(os.path.join(REPO, "tools", "eval_confusion.py"), "eval_confusion")
import eval_external as E, eval_v4 as Q, ai8x  # noqa: E402

# (표 이름, 평가 config, 체크포인트 접두, 계열, 모델 안의 'p1→p2' 사이 비-ai8x 연산)
CONFIGS = [
    ("④ 원본 파형",            "wave",          "safesound-v4d-wave",          "NPU 전량",      "없음 (전부 ai8x conv)"),
    ("(a) NPU 근사 log (주)",   "wave_logstage", "safesound-v4d-logstage",      "NPU 근사 log",  "없음 (구간선형 층도 ai8x)"),
    ("(a) 보조 lr 1e-3",        "wave_logstage", "safesound-v4d-logstage-lr1e3", "NPU 근사 log",  "없음"),
    ("가 u1000 (62.5 ms)",      "mel_u1000",     "safesound-v4d-melu1000",      "CPU log-mel",   "p1 이 CPU — 모델 밖"),
    ("다 u800 (50 ms)",         "mel_u800",      "safesound-v4d-melu800",       "CPU log-mel",   "p1 이 CPU — 모델 밖"),
    ("간격 400",                "mel_h400",      "safesound-v4d-melh400",       "CPU log-mel",   "p1 이 CPU — 모델 밖"),
    ("①′ (16 ms)",              "mel_inc",       "safesound-v4d-melinc",        "CPU log-mel",   "p1 이 CPU — 모델 밖"),
    ("②′ Gabor",                "wave_fe",       "safesound-v4d-fe",            "NPU 전처리 ②′", "|·|·합·정수 로그·/128 (CPU)"),
    ("②′ b64",                  "wave_fe_b64",   "safesound-v4d-fe-b64",        "NPU 전처리 ②′", "〃"),
    ("②′ f31",                  "wave_fe_f31",   "safesound-v4d-fe-f31",        "NPU 전처리 ②′", "〃"),
    ("②′ b64f31",               "wave_fe_b64f31", "safesound-v4d-fe-b64f31",    "NPU 전처리 ②′", "〃"),
    # chain34 (2026-10-24~25): 연결을 끊은 대조군과 fe8 — 결함이 사라졌는지 같은 자로 확인
    ("②′ v2 (연결 끊음)",       "wave_fe_v2",    "safesound-v4d-fe-v2",         "chain34",       "〃 (qat_link_cut)"),
    ("②′ fe8 (p1 8비트)",       "wave_fe8",      "safesound-v4d-fe8",           "chain34",       "p1 8비트 + Abs, 합은 NPU; 로그만 CPU (qat_link_cut)"),
]
SAT_FLAG = 0.05     # 출력 포화 5 % 이상이면 표시 — 마지막 fc(로짓) 는 제외


def probe(config, prefix, seed, n_win, tr):
    ck = E.checkpoint(prefix, seed)
    m = EC.load_model(ck, 5, T, simulate=False, bias=(config in Q.BIAS_CONFIGS), config=config)
    m.eval()
    ds = EC.make_dataset(config, Q.V.TEST, "test", tr)
    idxs = sorted(set(np.linspace(0, len(ds) - 1, n_win).astype(int).tolist()))
    X = torch.stack([ds[i][0] for i in idxs])
    caps, hooks = {}, []
    for name, mod in m.named_modules():
        if isinstance(mod, ai8x.QuantizationAwareModule):
            hooks.append(mod.register_forward_hook(lambda mod_, inp, out, n=name: caps.__setitem__(n, out.detach())))
    with torch.no_grad():
        m(X)
    for h in hooks:
        h.remove()
    layers = []
    mods = dict(m.named_modules())
    names = [n for n, _ in m.named_modules() if n in caps]
    for i, name in enumerate(names):
        mod = mods[name]
        o = caps[name].abs().flatten().numpy()
        w = mod.op.weight.detach(); wmax = float(w.abs().max())
        wide = bool(getattr(mod, "wide", False))
        layers.append({"layer": name, "thr": float(mod.activation_threshold), "wmax": wmax,
                       "shift": int(-(np.floor(np.log2(1.0 / wmax)))) if wmax > 0 else 0, "wide": wide,
                       "sat": float((o >= 127 / 128).mean()), "zero": float((o == 0).mean()),
                       "p99": float(np.percentile(o, 99)), "last": i == len(names) - 1})
    return os.path.basename(ck), layers


def first_cls(layers):
    """분류기 첫 층 = 첫 '비-wide' 층 (②′ 는 fe 가 wide 라 건너뛴다)."""
    for l in layers:
        if not l["wide"]:
            return l
    return layers[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-win", type=int, default=256)
    ap.add_argument("--seeds", default="1,2,3")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tr = ai8x.normalize(args=argparse.Namespace(act_mode_8bit=False))
    seeds = [int(s) for s in a.seeds.split(",")]
    out = {"date": datetime.date.today().isoformat(), "n_win": a.n_win, "configs": []}
    for label, config, prefix, family, between in CONFIGS:
        rows = []
        for seed in seeds:
            try:
                ck, layers = probe(config, prefix, seed, a.n_win, tr)
            except SystemExit as e:
                print(f"  [건너뜀] {label} 시드 {seed}: {e}"); continue
            rows.append({"seed": seed, "ckpt": ck, "layers": layers})
            flag = [l for l in layers if not l["last"] and not l["wide"] and l["sat"] >= SAT_FLAG]
            print(f"  {label:22s} 시드 {seed}: " + ", ".join(f"{l['layer']} thr{l['thr']:+.0f} 포화{100*l['sat']:.1f}%" for l in layers)
                  + (f"   <- 포화 {int(SAT_FLAG*100)} % 이상: {[l['layer'] for l in flag]}" if flag else ""))
        out["configs"].append({"label": label, "config": config, "prefix": prefix, "family": family, "between": between, "runs": rows})

    jp = os.path.join(REPO, "docs", "results", "npu-readback-sim", "qat_threshold_audit.json")
    json.dump(out, open(jp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ── 마크다운 표 ──
    L = [f"# QAT 활성화 문턱 감사 — 전 구성 분류기 층별 포화율 ({out['date']})", "",
         f"`tools/qat_threshold_audit.py`. 배경·질문은 `npu-readback-design.md` 7.1 (다). 시험 창 {a.n_win}개(testL-S3 균등 간격), "
         "체크포인트 qat_best, 보드·학습 없음 (사용자 지시 2026-10-24: 기존 체크포인트로만 계산).",
         "", "**포화** = 층 출력 |·| ≥ 127/128 (int8 잘림) 비율, **0** = 출력이 정확히 0 인 비율. **thr** = 저장된 `activation_threshold` "
         "(출력에 2^(−thr); 음수면 출력을 2^|thr| 배 키운다).",
         "wide 층(int32 출력)은 포화 개념이 없고(값은 참고), 마지막 fc 는 로짓이라 포화가 정상이다 — 둘은 판정에서 뺀다.", "",
         "**구조적 판단.** `apply_scales` 의 '이전 층 보상' 이 틀리려면 두 ai8x conv 사이에 **스케일을 지우는 비-ai8x 연산**이 있어야 한다. "
         "CPU log-mel 계열은 p1 이 모델 밖(CPU)이라 연결이 없고, ④·(a) 는 층 사이가 ReLU·풀링·구간선형(모두 ai8x)이라 보상이 맞다. "
         "따라서 결함 (다) 는 **②′ 계열에만** 생길 수 있다 — 아래 표가 그것을 실측으로 확인한다.", ""]
    L += ["## 1. 요약 — 분류기 첫 층과 중간층 (시드 1~3)", "",
          "| 구성 | 계열 | p1→p2 사이의 비-ai8x 연산 | 첫 층 | 첫 층 thr | 첫 층 포화 평균 (범위) | 중간층 최대 포화 (층) | fc 포화 | 판정 |",
          "|---|---|---|---|---:|---:|---:|---:|---|"]
    for c in out["configs"]:
        if not c["runs"]:
            L.append(f"| {c['label']} | {c['family']} | {c['between']} | — | — | — | — | — | 체크포인트 없음 |"); continue
        firsts = [first_cls(r["layers"]) for r in c["runs"]]
        thr = sorted(set(f"{f['thr']:+.0f}" for f in firsts))
        sat1 = [f["sat"] for f in firsts]
        mids = []
        for r in c["runs"]:
            f0 = first_cls(r["layers"])
            m_ = [l for l in r["layers"] if not l["wide"] and not l["last"] and l is not f0]
            if m_:
                mx = max(m_, key=lambda l: l["sat"]); mids.append((mx["sat"], mx["layer"]))
        fcs = [r["layers"][-1]["sat"] for r in c["runs"]]
        big_mid = max((s for s, _ in mids), default=0.0)
        if np.mean(sat1) >= SAT_FLAG and min(f["thr"] for f in firsts) < 0:
            verdict = "**결함 (다) 와 같은 양상** — 첫 층 thr 음수·포화 큼"
        elif np.mean(sat1) >= SAT_FLAG or big_mid >= SAT_FLAG:
            verdict = "포화 큼 (thr 0 이상 — 다른 원인)"
        else:
            verdict = "정상"
        L.append(f"| {c['label']} | {c['family']} | {c['between']} | {firsts[0]['layer']} | {'/'.join(thr)} | "
                 f"{100*np.mean(sat1):.2f} % ({100*min(sat1):.2f}~{100*max(sat1):.2f}) | "
                 + (f"{100*big_mid:.2f} % ({max(mids)[1]})" if mids else "—")
                 + f" | {100*np.mean(fcs):.0f} % | {verdict} |")
    L += ["", "## 2. 전 층 (시드별)", ""]
    for c in out["configs"]:
        L += [f"### {c['label']}  (`{c['config']}`, `{c['prefix']}`)", ""]
        if not c["runs"]:
            L += ["(체크포인트 없음)", ""]; continue
        L += ["| 시드 | 층 | wide | thr | max\\|w\\| | 출력 p99 | 포화 | 0 |", "|---:|---|---|---:|---:|---:|---:|---:|"]
        for r in c["runs"]:
            for l in r["layers"]:
                mark = " ⚠" if (not l["last"] and not l["wide"] and l["sat"] >= SAT_FLAG) else ""
                L.append(f"| {r['seed']} | {l['layer']}{' (fc)' if l['last'] else ''} | {'wide' if l['wide'] else ''} | {l['thr']:+.0f} | {l['wmax']:.3f} | "
                         f"{l['p99']:.3f} | {100*l['sat']:.2f} %{mark} | {100*l['zero']:.1f} % |")
        L.append("")
    mp = os.path.join(REPO, "docs", "results", "qat-threshold-audit.md")
    open(mp, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("->", mp, "/", jp)


if __name__ == "__main__":
    main()
