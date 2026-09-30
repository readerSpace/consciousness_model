import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment import circular_shift_surrogate, multisource  # noqa: E402
from metrics import (compute, delay_embed, holonomy_from_edges, row_spearman_all_pairs,  # noqa: E402
                     triangles, wrap)
from simulate import CONDITIONS, simulate  # noqa: E402


def test_wrap_range():
    x = np.linspace(-20, 20, 1001)
    w = wrap(x)
    assert w.min() >= -np.pi and w.max() < np.pi
    assert np.allclose(np.exp(1j * w), np.exp(1j * x))


def test_literal_patch_flatness_is_identically_one_on_any_data():
    rng = np.random.default_rng(1)
    for x in (rng.standard_normal((8, 3000)), rng.standard_cauchy((8, 3000))):
        m = compute(x, 250.0, band=(7, 13))
        assert m.max_abs_instant_holonomy < 1e-9
        assert np.allclose(m.f_instant, 1.0)


def test_edge_phase_from_vertex_potential_has_zero_holonomy():
    rng = np.random.default_rng(2)
    psi = rng.uniform(-np.pi, np.pi, 7)
    phi = psi[:, None] - psi[None, :]
    assert np.abs(holonomy_from_edges(phi, triangles(7))).max() < 1e-12


def test_rank_one_plv_gives_flat_windowed_patch():
    # 全チャネル = 共通位相 + 定数オフセット (+位相ゆらぎも共通) → P は rank-1
    t = np.arange(5000) / 250.0
    rng = np.random.default_rng(3)
    common = 2 * np.pi * 10 * t + np.cumsum(0.05 * rng.standard_normal(t.size))
    off = rng.uniform(-np.pi, np.pi, 9)
    x = np.cos(common[None, :] + off[:, None])
    m = compute(x, 250.0, band=(7, 13))
    assert m.f_windowed.min() > 0.999
    assert m.eff_rank.max() < 0.02


def test_two_sources_with_delays_break_flatness():
    x1 = multisource(12, 1, 250.0, 10.0, seed=0)
    x2 = multisource(12, 3, 250.0, 10.0, seed=0)
    f1 = compute(x1, 250.0, band=(7, 13)).f_windowed.mean()
    f2 = compute(x2, 250.0, band=(7, 13)).f_windowed.mean()
    assert f1 > 0.99 and f2 < f1 - 0.05


def test_delay_embedding_layout():
    x = np.arange(20, dtype=float)[None, :]
    emb, t0 = delay_embed(x, d=3, tau=2)
    assert t0 == 4
    assert emb[0, 0].tolist() == [4.0, 2.0, 0.0]
    assert emb[0, -1].tolist() == [19.0, 17.0, 15.0]


def test_row_spearman_matches_scipy():
    rng = np.random.default_rng(4)
    emb = rng.standard_normal((3, 6, 5))
    r = row_spearman_all_pairs(emb)
    for t in range(6):
        assert np.isclose(r[0, 1, t], spearmanr(emb[0, t], emb[1, t])[0])
        assert np.isclose(r[2, 2, t], 1.0)


def test_surrogate_preserves_each_channel_but_breaks_zero_lag_mixing():
    g = replace(CONDITIONS["volume_conduction"], seconds=10.0)
    x, _ = simulate(g, 0)
    s = circular_shift_surrogate(x, g.fs, np.random.default_rng(0))
    assert np.allclose(np.sort(x, axis=1), np.sort(s, axis=1))
    c = np.corrcoef(x)[~np.eye(g.n, dtype=bool)].mean()
    cs = np.corrcoef(s)[~np.eye(g.n, dtype=bool)].mean()
    assert c > 0.2 and abs(cs) < 0.1


def test_c_hist_rises_with_pure_mixing_and_no_coupling():
    lo = replace(CONDITIONS["independent"], seconds=8.0, mix=0.0)
    hi = replace(lo, mix=0.9, mix_width=0.3)
    c_lo = compute(simulate(lo, 0)[0], 250.0, band=(7, 13)).c_hist.mean()
    c_hi = compute(simulate(hi, 0)[0], 250.0, band=(7, 13)).c_hist.mean()
    assert c_hi > c_lo + 0.2


def test_c_hist_collapses_under_pure_delay_with_perfect_coupling():
    g0 = replace(CONDITIONS["single_source_wave"], seconds=8.0, wave_delay=0.0)
    g1 = replace(g0, wave_delay=0.01)
    m0 = compute(simulate(g0, 0)[0], 250.0, band=(7, 13))
    m1 = compute(simulate(g1, 0)[0], 250.0, band=(7, 13))
    assert m1.mean_plv.mean() > 0.9           # 結合は依然ほぼ完全
    assert m0.c_hist.mean() > 0.8 and m1.c_hist.mean() < 0.05
