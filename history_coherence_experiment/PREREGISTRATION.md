# 事前登録: 履歴コヒーレント貼り合わせ指標の検証 (2026-09-24, 実行前に記述)

対象: 提案文書の  C_hist, F_patch, I_coherent = C_hist × F_patch。
正解の意識状態は合成データでは存在しないので、**指標自身が主張している性質**を
生成過程が既知のデータで検査する。意識の有無は一切検証しない。

## 数学的予測（実行前に導出）
M1. 瞬時位相差 Δθ_ab(t)=θ_a(t)-θ_b(t) を辺に置くと、三角形の巡回和は
    (θa-θb)+(θb-θc)+(θc-θa) ≡ 0 (mod 2π)。辺の値が頂点ポテンシャルの差
    (コバウンダリ δθ) なので δδ=0。**F_patch(文字通り) ≡ 1 がどんなデータでも成り立つ。**
M2. 非自明なホロノミーを得るには辺を「対ごとに推定」する必要がある
    (例: 窓内 PLV の偏角 φ_ab = arg⟨e^{iΔθ_ab}⟩)。複素 PLV 行列
    P = (1/W)Σ u u^H (u_k=e^{iθ_k}) は Hermite 半正定値。
    P が rank-1 (全チャネルが単一の位相パターンに固定) ⇒ φ_ab = ψ_a-ψ_b ⇒ ホロノミー 0。
    **したがって修正版 F_patch は「単一位相パターンへの支配」を最大に評価する。**

## シミュレーション予測（提案文の主張 → 予測する実測）
条件: independent / modular / wake_like / hypersync / single_source_wave /
      volume_conduction (結合ゼロ＋空間混合) / common_drive
P1. F_instant は全条件・全窓で 1 (|Ω|<1e-9)。
P2. F_windowed は hypersync・single_source_wave で最大、independent で最小。
    窓ごとの F_windowed と平均|PLV| の順位相関 > 0.8 (独立情報ではない)。
P3. I_coherent の最大は hypersync か single_source_wave (提案文の「てんかんは別」という
    記述は数式に入っていない)。
P4. volume_conduction の C_hist は independent より明確に高い (結合ゼロなのに)。
P5. C_hist と「窓平均 R を掛けない単なる mean|PLV|」の窓ごとの順位相関 > 0.9
    (履歴埋め込みが追加情報をほぼ持たない)。
P6. 提案文の期待順序 wake_like > modular > independent が C_hist では成立する
    (結合強度の単調関数なので)。ただし volume_conduction が modular 以上に来る。

## 修正案の検査 (パラメータなし, 事後 hold-out で評価)
R1. 虚部ベース (wPLI) は volume_conduction を independent 水準に落とす。
R2. 位相行列 P の有効ランク (固有値エントロピー) は hypersync/single_source で低い。
R3. 候補 I' = mean wPLI × H_norm(P) は hypersync を最大にしない。
    これは設計した条件で合格しても循環的なので、生成パラメータを変えた
    hold-out (N, 周波数, 混合カーネル, 位相遅れ) で順序が保たれるかを見る。

## 反証条件
- P1 が破れたら実装バグ (数学的恒等式)。
- P3 が破れて wake_like が最大なら、提案指標は少なくとも合成上は主張どおり。
