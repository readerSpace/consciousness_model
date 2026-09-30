"""履歴コヒーレント貼り合わせ指標の検証実験 E1-E7。

    python history_coherence_experiment/experiment.py --output history_coherence_experiment/results.json

事前登録: PREREGISTRATION.md (実行前に記述)。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import (analytic, bandpass, compute, delay_embed, holonomy_from_edges,  # noqa: E402
                     row_spearman_all_pairs, triangles, wrap)
from simulate import CONDITIONS, Gen, _ar1, holdout_conditions, mixing_matrix, simulate  # noqa: E402

KEYS = ["c_hist", "mean_plv", "mean_r", "f_instant", "f_windowed", "f_coherency",
        "wpli", "eff_rank", "i_coherent", "i_literal", "i_repair"]


def band_of(g: Gen):
    return (g.f0 - 3.0, g.f0 + 3.0)


def circular_shift_surrogate(x, fs, rng, min_shift_sec=1.0):
    """各チャネルを独立に巡回シフト: 自己の履歴は保存、チャネル間関係 (混合含む) は破壊。"""
    t = x.shape[1]
    lo = int(min_shift_sec * fs)
    return np.stack([np.roll(row, rng.integers(lo, t - lo)) for row in x])


def summarize(x, g, seed, win_sec, n_surr):
    m = compute(x, g.fs, band=band_of(g), win_sec=win_sec)
    row = {k: float(getattr(m, k).mean()) for k in KEYS}
    row["max_abs_instant_holonomy"] = m.max_abs_instant_holonomy
    if n_surr:
        rng = np.random.default_rng(10_000 + seed)
        surr = {k: [] for k in KEYS}
        for _ in range(n_surr):
            ms = compute(circular_shift_surrogate(x, g.fs, rng), g.fs, band=band_of(g), win_sec=win_sec)
            for k in KEYS:
                surr[k].append(float(getattr(ms, k).mean()))
        for k in KEYS:
            mu, sd = np.mean(surr[k]), np.std(surr[k]) + 1e-12
            row[k + "_surr"] = float(mu)
            row[k + "_z"] = float((row[k] - mu) / sd)
        # 修正候補: 代理データで補正した wPLI × 有効ランク
        row["i_repair_corr"] = max(row["wpli"] - row["wpli_surr"], 0.0) * row["eff_rank"]
        row["i_coherent_corr"] = max(row["c_hist"] - row["c_hist_surr"], 0.0) * row["f_windowed"]
    return row, m


def agg(rows):
    keys = rows[0].keys()
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "sd": float(np.std([r[k] for r in rows]))}
            for k in keys}


# ---------------------------------------------------------------- E1
def e1_identity(seeds):
    """文字通りの F_patch は任意データで 1。白色雑音・ランダムウォークでも。"""
    worst = 0.0
    rng = np.random.default_rng(0)
    cases = {}
    for name, x in {
        "white_noise": rng.standard_normal((10, 5000)),
        "random_walk": np.cumsum(rng.standard_normal((10, 5000)), axis=1),
        "heavy_tailed": rng.standard_cauchy((10, 5000)),
    }.items():
        m = compute(x, 250.0, band=(7, 13))
        cases[name] = {"f_instant_min": float(m.f_instant.min()), "max_abs_holonomy": m.max_abs_instant_holonomy}
        worst = max(worst, m.max_abs_instant_holonomy)
    for name, g in CONDITIONS.items():
        for s in seeds:
            x, _ = simulate(g, s)
            m = compute(x, g.fs, band=band_of(g))
            worst = max(worst, m.max_abs_instant_holonomy)
    return {"cases": cases, "worst_abs_holonomy_all_runs": worst,
            "reason": "Δθ_ab = θ_a - θ_b は頂点関数の差 (コバウンダリ) なので三角形和は恒等的に 0 (δδ=0)"}


# ---------------------------------------------------------------- E2
def e2_conditions(seeds, n_surr):
    res = {}
    for win in (0.5, 2.0):
        table = {}
        for name, g in CONDITIONS.items():
            rows = [summarize(simulate(g, s)[0], g, s, win, n_surr)[0] for s in seeds]
            table[name] = agg(rows)
        res[f"win_{win}s"] = table
    return res


# ---------------------------------------------------------------- E3
def e3_lag_sweep(seeds):
    """単一源の進行波: 結合は常に完全 (同じ1つの振動子)。変えるのは伝播遅延だけ。"""
    out = []
    for delay_ms in (0, 2, 5, 10, 15, 20, 25):
        g = replace(CONDITIONS["single_source_wave"], wave_delay=delay_ms / 1000.0)
        rows = [summarize(simulate(g, s)[0], g, s, 0.5, 0)[0] for s in seeds]
        a = agg(rows)
        out.append({"neighbor_delay_ms": delay_ms,
                    **{k: a[k]["mean"] for k in ("c_hist", "mean_r", "mean_plv", "wpli", "f_windowed", "i_coherent")}})
    return out


# ---------------------------------------------------------------- E4
def e4_mixing_sweep(seeds, n_surr):
    """結合ゼロの独立振動子。空間混合だけを強める。真の相互作用は常にゼロ。"""
    out = []
    for mix in (0.0, 0.2, 0.4, 0.6, 0.8, 0.95):
        g = replace(CONDITIONS["independent"], mix=mix, mix_width=0.3)
        rows = [summarize(simulate(g, s)[0], g, s, 0.5, n_surr)[0] for s in seeds]
        a = agg(rows)
        out.append({"mix": mix, **{k: a[k]["mean"] for k in
                    ("c_hist", "c_hist_z", "i_coherent", "i_coherent_corr", "wpli", "wpli_z", "i_repair_corr")}})
    return out


# ---------------------------------------------------------------- E5
def multisource(n, q, fs, seconds, seed, max_delay=0.03, background=0.3):
    """q 個の独立振動子を、チャネルごとに異なる遅延で重ね合わせる。

    q=1 なら複素位相行列は rank-1 → 窓ホロノミー 0 のはず。
    """
    rng = np.random.default_rng(seed)
    t = int(seconds * fs)
    pad = int(max_delay * fs) + 1
    srcs = []
    for _ in range(q):
        f = 10 + 0.8 * rng.standard_normal()
        th = np.cumsum(2 * np.pi * f / fs + np.sqrt(2 * 1.0 / fs) * rng.standard_normal(t + pad))
        srcs.append(np.cos(th))
    x = np.zeros((n, t))
    for k in range(n):
        for s in srcs:
            dl = rng.integers(0, pad)
            x[k] += rng.uniform(0.5, 1.0) * s[pad - dl: pad - dl + t]
    return x + background * _ar1(rng, n, t)


def e5_rank_holonomy(seeds):
    out = []
    for q in (1, 2, 3, 4, 6, 8):
        rows = []
        for s in seeds:
            x = multisource(16, q, 250.0, 30.0, s)
            m = compute(x, 250.0, band=(7, 13))
            rows.append({"f_windowed": float(m.f_windowed.mean()), "eff_rank": float(m.eff_rank.mean()),
                         "c_hist": float(m.c_hist.mean()), "i_coherent": float(m.i_coherent.mean()),
                         "mean_plv": float(m.mean_plv.mean())})
        a = agg(rows)
        out.append({"n_sources": q, **{k: a[k]["mean"] for k in rows[0]}})
    return out


# ---------------------------------------------------------------- E6
def e6_redundancy(seeds):
    """C_hist の「履歴」部分が瞬時同期以上の情報を持つか。"""
    cw, pw, fw, pl2 = [], [], [], []
    samp_r, samp_cos = [], []
    for name, g in CONDITIONS.items():
        for s in seeds:
            x, _ = simulate(g, s)
            m = compute(x, g.fs, band=band_of(g))
            cw += list(m.c_hist)
            fw += list(m.f_windowed)
            pw += list(m.mean_plv)
            # 比較用: 履歴なしの瞬時量 Re(PLV)*|PLV| の平均
            xf = bandpass(x, g.fs, *band_of(g))
            emb, t0 = delay_embed(xf, 5, max(1, int(round(g.fs / (4 * g.f0)))))
            z = analytic(xf)[:, t0:]
            th = np.angle(z)
            w = int(0.5 * g.fs)
            off = ~np.eye(g.n, dtype=bool)
            for k in range(z.shape[1] // w):
                u = np.exp(1j * th[:, k * w:(k + 1) * w])
                p = u @ u.conj().T / w
                pl2.append(float((p.real * np.abs(p))[off].mean()))
            if s == seeds[0]:
                R = row_spearman_all_pairs(emb[:, :1500])
                c = np.cos(th[:, None, :1500] - th[None, :, :1500])
                samp_r.append(R[off].ravel())
                samp_cos.append(c[off].ravel())
    r = np.concatenate(samp_r)
    c = np.concatenate(samp_cos)
    return {
        "n_windows": len(cw),
        "spearman_chist_vs_meanPLV": float(spearmanr(cw, pw)[0]),
        "spearman_chist_vs_RePLVxAbsPLV_no_history": float(spearmanr(cw, pl2)[0]),
        "pearson_chist_vs_RePLVxAbsPLV_no_history": float(np.corrcoef(cw, pl2)[0, 1]),
        "spearman_fwindowed_vs_meanPLV": float(spearmanr(fw, pw)[0]),
        "pearson_sample_R_vs_cos_dtheta": float(np.corrcoef(r, c)[0, 1]),
    }


# ---------------------------------------------------------------- E7
def e7_holdout(seeds, n_surr):
    """修正候補を、設計に使っていない生成パラメータで評価。"""
    worlds = holdout_conditions()
    table = {}
    for (v, c), g in worlds.items():
        rows = [summarize(simulate(g, s)[0], g, s, 0.5, n_surr)[0] for s in seeds]
        a = agg(rows)
        table.setdefault(v, {})[c] = {k: a[k]["mean"] for k in ("i_coherent", "i_coherent_corr", "i_repair_corr", "c_hist")}
    verdict = {}
    for v, t in table.items():
        vd = {}
        for metric in ("i_coherent", "i_coherent_corr", "i_repair_corr"):
            ranked = sorted(t, key=lambda k: -t[k][metric])
            vd[metric] = {
                "top": ranked[0],
                "hypersync_rank": ranked.index("hypersync") + 1,
                "volume_conduction_minus_independent": t["volume_conduction"][metric] - t["independent"][metric],
                "wake_like_rank": ranked.index("wake_like") + 1,
            }
        verdict[v] = vd
    return {"table": table, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(Path(__file__).resolve().parent / "results.json"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--surrogates", type=int, default=10)
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    t0 = time.time()
    res = {}
    for name, fn in [
        ("E1_literal_F_identity", lambda: e1_identity(seeds)),
        ("E2_conditions", lambda: e2_conditions(seeds, args.surrogates)),
        ("E3_lag_sweep", lambda: e3_lag_sweep(seeds)),
        ("E4_mixing_sweep", lambda: e4_mixing_sweep(seeds, args.surrogates)),
        ("E5_rank_holonomy", lambda: e5_rank_holonomy(seeds)),
        ("E6_redundancy", lambda: e6_redundancy(seeds)),
        ("E7_repair_holdout", lambda: e7_holdout(seeds, args.surrogates)),
    ]:
        t = time.time()
        res[name] = fn()
        print(f"{name}: {time.time() - t:.1f}s", flush=True)
    res["runtime_sec"] = time.time() - t0
    Path(args.output).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"total {res['runtime_sec']:.1f}s -> {args.output}")


if __name__ == "__main__":
    main()
