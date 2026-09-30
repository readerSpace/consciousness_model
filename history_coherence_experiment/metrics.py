"""提案文書の指標 (C_hist, F_patch) と比較・修正用の指標。

入力は x: shape (N, T) の実数時系列。すべて numpy/scipy のみ。
提案の式をそのまま実装したもの (literal) と、辺を対ごとに推定し直したもの
(windowed) を明確に分けている。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt


def wrap(x):
    """角度を [-pi, pi) に収める。"""
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def bandpass(x, fs, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def analytic(x):
    return hilbert(x, axis=-1)


def delay_embed(x, d, tau):
    """X_k(t) = (x(t), x(t-tau), ..., x(t-(d-1)tau))。shape (N, T', d)。

    先頭 (d-1)*tau サンプルは履歴が足りないので落とす。戻り値の t=0 は元の
    t0=(d-1)*tau に対応する。
    """
    n, t = x.shape
    t0 = (d - 1) * tau
    cols = [x[:, t0 - j * tau: t - j * tau] for j in range(d)]
    return np.stack(cols, axis=-1), t0


def row_spearman_all_pairs(emb):
    """各時刻で d 次元履歴ベクトル同士のスピアマン相関 R_ab(t)。shape (N, N, T')。

    連続値なので同順位は無い → 中心化順位の分散は (d^2-1)/12 で共通。
    """
    n, t, d = emb.shape
    ranks = emb.argsort(axis=-1).argsort(axis=-1).astype(np.float64)
    ranks -= (d - 1) / 2.0
    denom = d * (d * d - 1) / 12.0
    return np.einsum("atd,btd->abt", ranks, ranks) / denom


def triangles(n):
    return np.array(list(combinations(range(n), 3)), dtype=int)


def holonomy_from_edges(phi, tri):
    """phi: (N, N[, ...]) 反対称な辺位相。三角形ごとの Wrap(phi_ab+phi_bc+phi_ca)。"""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    return wrap(phi[a, b] + phi[b, c] + phi[c, a])


@dataclass
class WindowMetrics:
    c_hist: np.ndarray          # 提案① (窓ごと)
    mean_plv: np.ndarray        # R を掛けない mean |PLV|
    mean_r: np.ndarray          # 窓平均の履歴スピアマン
    f_instant: np.ndarray       # 提案② を文字通り (瞬時位相差) → 恒等的に 1 のはず
    max_abs_instant_holonomy: float
    f_windowed: np.ndarray      # 辺 = arg(窓内PLV) にした修正版②
    f_coherency: np.ndarray     # 辺 = arg(窓内複素コヒーレンシ) (振幅重み)
    wpli: np.ndarray            # mean wPLI (ゼロ遅れ混合に頑健)
    eff_rank: np.ndarray        # PLV 行列の固有値エントロピー / log N
    i_coherent: np.ndarray      # 提案の C_hist * F_patch(windowed)
    i_literal: np.ndarray       # 提案の C_hist * F_patch(literal)
    i_repair: np.ndarray        # 修正候補 wPLI * eff_rank


def compute(x, fs, band=(8.0, 13.0), d=5, tau=None, win_sec=0.5, max_windows=None):
    """提案パイプライン (1)-(5) を実行して窓ごとの指標を返す。"""
    n, _ = x.shape
    xf = bandpass(x, fs, *band)
    if tau is None:
        # 中心周波数の 1/4 周期 (遅延埋め込みの標準的な選び方)
        tau = max(1, int(round(fs / (4 * np.mean(band)))))
    emb, t0 = delay_embed(xf, d, tau)
    z = analytic(xf)[:, t0:]
    theta = np.angle(z)
    w = int(round(win_sec * fs))
    nwin = z.shape[1] // w
    if max_windows is not None:
        nwin = min(nwin, max_windows)
    tri = triangles(n)
    offdiag = ~np.eye(n, dtype=bool)

    out = {k: [] for k in WindowMetrics.__dataclass_fields__ if k != "max_abs_instant_holonomy"}
    max_inst = 0.0
    for k in range(nwin):
        sl = slice(k * w, (k + 1) * w)
        th = theta[:, sl]
        zz = z[:, sl]
        u = np.exp(1j * th)
        plv = (u @ u.conj().T) / w                       # P_ab = <e^{i(θa-θb)}>
        r = row_spearman_all_pairs(emb[:, sl, :]).mean(axis=-1)

        c_hist = (r * np.abs(plv))[offdiag].mean()

        # 文字通りの②: 各時刻の瞬時位相差で三角形和
        dth = th[:, None, :] - th[None, :, :]            # (N,N,W)
        hol_inst = holonomy_from_edges(dth, tri)         # (Ntri, W)
        max_inst = max(max_inst, float(np.abs(hol_inst).max()))
        f_inst = np.exp(-np.abs(hol_inst).mean(axis=0)).mean()

        f_win = np.exp(-np.abs(holonomy_from_edges(np.angle(plv), tri)).mean())
        coh = (zz @ zz.conj().T) / w
        f_coh = np.exp(-np.abs(holonomy_from_edges(np.angle(coh), tri)).mean())

        im = np.imag(zz[:, None, :] * zz[None, :, :].conj())
        num = np.abs(im.mean(axis=-1))
        den = np.abs(im).mean(axis=-1)
        wpli = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)[offdiag].mean()

        ev = np.clip(np.linalg.eigvalsh(plv).real, 0, None)
        p = ev / ev.sum()
        p = p[p > 0]
        eff = float(-(p * np.log(p)).sum() / np.log(n))

        out["c_hist"].append(c_hist)
        out["mean_plv"].append(np.abs(plv)[offdiag].mean())
        out["mean_r"].append(r[offdiag].mean())
        out["f_instant"].append(f_inst)
        out["f_windowed"].append(f_win)
        out["f_coherency"].append(f_coh)
        out["wpli"].append(wpli)
        out["eff_rank"].append(eff)
        out["i_coherent"].append(c_hist * f_win)
        out["i_literal"].append(c_hist * f_inst)
        out["i_repair"].append(wpli * eff)
    arr = {k: np.asarray(v) for k, v in out.items()}
    return WindowMetrics(max_abs_instant_holonomy=max_inst, **arr)
