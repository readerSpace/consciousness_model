"""生成過程が既知の合成「脳波」。

各チャネルの源は雑音つき Kuramoto-Sakaguchi 振動子 (α帯) + 背景 AR(1) 雑音。
その後、リング上のガウスカーネルで空間混合 (体積伝導の模型) を掛ける。
条件名は意識状態を主張しない。生成の違いだけを表す。
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Gen:
    n: int = 16
    fs: float = 250.0
    seconds: float = 30.0
    f0: float = 10.0            # 中心周波数 Hz
    f_sd: float = 1.0           # 固有周波数のばらつき Hz
    k_in: float = 0.0           # モジュール内結合 (rad/s)
    k_out: float = 0.0          # モジュール間結合
    n_modules: int = 4
    alpha: float = 0.0          # Sakaguchi 位相遅れ (rad)
    drive: float = 0.0          # 共通駆動の強さ
    phase_noise: float = 1.0    # 位相拡散係数 D (rad^2/s)
    amp: float = 1.0
    background: float = 0.7
    mix: float = 0.3            # 空間混合の割合
    mix_width: float = 0.08     # リング上のカーネル幅 (周長=1)
    wave: bool = False          # 単一源の進行波
    wave_delay: float = 0.005   # 隣接チャネル間遅延 s
    substeps: int = 4


def mixing_matrix(n, mix, width):
    pos = np.arange(n) / n
    dist = np.abs(pos[:, None] - pos[None, :])
    dist = np.minimum(dist, 1 - dist)
    g = np.exp(-dist**2 / (2 * width**2))
    g /= g.sum(axis=1, keepdims=True)
    return (1 - mix) * np.eye(n) + mix * g


def _coupling(g: Gen):
    mod = np.arange(g.n) * g.n_modules // g.n
    same = mod[:, None] == mod[None, :]
    k = np.where(same, g.k_in, g.k_out).astype(float)
    np.fill_diagonal(k, 0.0)
    deg = np.maximum((k > 0).sum(axis=1, keepdims=True), 1)
    return k / deg


def _ar1(rng, n, t, rho=0.95):
    e = rng.standard_normal((n, t))
    out = np.empty_like(e)
    out[:, 0] = e[:, 0]
    for i in range(1, t):
        out[:, i] = rho * out[:, i - 1] + e[:, i]
    return out / out.std(axis=1, keepdims=True)


def simulate(g: Gen, seed: int):
    rng = np.random.default_rng(seed)
    t = int(g.seconds * g.fs)
    dt = 1.0 / (g.fs * g.substeps)
    n = g.n
    if g.wave:
        # 1つの振動子を各チャネルが遅延付きで見る
        steps = t * g.substeps
        omega = 2 * np.pi * g.f0
        th = np.cumsum(omega * dt + np.sqrt(2 * g.phase_noise * dt) * rng.standard_normal(steps))
        th = th[:: g.substeps]
        lag = np.round(np.arange(n) * g.wave_delay * g.fs).astype(int)
        pad = lag.max()
        base = np.concatenate([np.full(pad, th[0]), th])
        theta = np.stack([base[pad - l: pad - l + t] for l in lag])
    else:
        omega = 2 * np.pi * (g.f0 + g.f_sd * rng.standard_normal(n))
        k = _coupling(g)
        th = rng.uniform(-np.pi, np.pi, n)
        thd = 0.0
        theta = np.empty((n, t))
        sq = np.sqrt(2 * g.phase_noise * dt)
        for i in range(t):
            for _ in range(g.substeps):
                diff = th[None, :] - th[:, None] - g.alpha
                dth = omega + (k * np.sin(diff)).sum(axis=1)
                if g.drive:
                    dth = dth + g.drive * np.sin(thd - th)
                    thd += 2 * np.pi * g.f0 * dt
                th = th + dth * dt + sq * rng.standard_normal(n)
            theta[:, i] = th
    # 緩やかな振幅変調
    amp = g.amp * (1 + 0.3 * _ar1(rng, n, t, rho=0.999))
    src = amp * np.cos(theta) + g.background * _ar1(rng, n, t)
    x = mixing_matrix(n, g.mix, g.mix_width) @ src
    return x, theta


CONDITIONS: dict[str, Gen] = {
    "independent": Gen(),
    "modular": Gen(k_in=20.0),
    "wake_like": Gen(k_in=14.0, k_out=4.0, alpha=0.4),
    "hypersync": Gen(k_in=60.0, k_out=60.0, f_sd=0.3, amp=2.0, alpha=0.0),
    "single_source_wave": Gen(wave=True),
    "common_drive": Gen(drive=10.0),
    "volume_conduction": Gen(mix=0.85, mix_width=0.3),
}


def holdout_conditions():
    """修正案 R3 用: 生成パラメータを変えた未使用の世界。"""
    variants = {
        "N12_f6Hz": dict(n=12, f0=6.0),
        "N20_f11Hz": dict(n=20, f0=11.0),
        "wide_mix": dict(mix=0.5, mix_width=0.15),
        "alpha0.8": dict(),  # wake_like の alpha のみ変更 (下で処理)
    }
    out = {}
    for vname, kw in variants.items():
        for cname, g in CONDITIONS.items():
            if cname == "volume_conduction":
                g2 = replace(g, **{k: v for k, v in kw.items() if k in ("n", "f0")})
            else:
                g2 = replace(g, **kw)
            if vname == "alpha0.8" and cname == "wake_like":
                g2 = replace(g2, alpha=0.8)
            out[(vname, cname)] = g2
    return out
