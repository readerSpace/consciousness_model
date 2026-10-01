# 圧縮ゲノム進化 (Compressed Genome Evolution) — exp588

設計メモの問い

> **圧縮圧は「短いゲノム」を作るだけなのか、それとも再利用可能な遺伝的規則を進化させるのか？**

を、実際に復号可能なビット長を目的関数に据えて実装検証する。
`phylogenetic_compression_experiment` と同じ規律
（`L` は復号可能なビット列の長さ・中核はドメイン中立・対照条件つき・シードに `hash()` を使わない）
を踏襲している。

対応する定式化（設計メモの表そのまま）:

| 系 | 遺伝表現 | 目的 `J` |
|---|---|---|
| A | 固定長ゲノム | `F` |
| B | プログラムゲノム | `F − λ·L(G)` |
| C | プログラムゲノム (MDL) | `F − λ·(L(G)+L(D))` |
| D | 関連環境上の MDL | `mean_e[F_e − λ·(L(G)+L(D))]`（将来環境への適応性） |

を **構造あり / 構造なし** 環境で比較する。


## 1. 主張と非主張

**主張する**

* `L(G)`・`L(D)` は実際に復号可能なビット列の長さである。`encode(g)` が出力し、
  `decode` がそれだけからゲノムを完全復元し、`description_length` が同じ値を算術的に返す。
  三者一致はテスト（`test_encode_length_equals_description_length` ほか、500 個の乱ゲノム）で検証済み。
  `L(G)`（主プログラム部）+ `L(D)`（マクロ表部）= 総記述長、も検査済み。
* 中核 `genome.py` の実行コードにはアプリ語（fitness / evolution / selection / environment /
  mutation / population / phenotype）が 1 語も無い。`test_core_is_application_neutral` が全文検査する。
  中核が知るのは symbol / instruction / program / macro / expand / bits のみ。
* **中心的発見**（下記 3 節）: 総記述長 `L(G)+L(D)` を罰する MDL 圧（系 C）だけが
  再利用可能な規則（小さなマクロを何度も呼ぶ）を進化させる。**プログラム長だけを罰する系 B は
  「見かけのプログラム」を縮めるが、真のゲノム（総記述長）はむしろ A より大きくなる**。
* 構造の無いヌル環境では、C は高 `F` と短ゲノムを両立できない（`F` を落とさなければ縮まない）。
  シャッフル対照も同様に振る舞う。＝圧縮は「見つける構造がある時だけ」効く。

**主張しない**

* GA は貪欲な進化探索であり大域最適の保証はない。報告する記述長は常に厳密。
  「これ以上短い記述が存在しない」ことは示していない（`L*` の全列挙監査は未実施＝次項）。
* `F` が 1.0 に満たない系 B/C の値は、圧縮圧がヌイサンス位置や境界の数ビットを
  切り捨てた結果であり、バグではなく MDL のトレードオフである。
* この設定の「ゲノム」は 1 本の表現型列を符号化する。「未知環境へ同一ゲノムで一般化」は
  この設定では ill-posed なので測っていない。代わりに **再適応世代数**（系 3）で
  「規則の再利用」を直接測る。


## 2. 符号（`genome.py`）

ゲノム = マクロ表 + 主プログラム。命令は 4 種のフラットなバイトコード:

| 命令 | 意味 | 符号 |
|---|---|---|
| `LIT s` | 記号 s を出力 | 2bit opcode + ⌈log2 A⌉ bit |
| `CALL k` | マクロ k を展開 | 2bit + γ(k+1) |
| `REP n k` | マクロ k を n 回 | 2bit + γ(n) + γ(k+1) |
| `MIR k` | マクロ k + その反転 | 2bit + γ(k+1) |

マクロ i は index < i のマクロのみ参照可（DAG・再帰なし）。
展開は各段で長さ L に打ち切る（REP の爆発をコスト面で封じる）。

* `L(G)` = 主プログラム部のビット長（**B と C の両方が罰する**）。
* `L(D)` = マクロ表部のビット長（**C だけが罰する**）。
* インタプリタは全個体で共有の固定物＝定数なので数えない。
  だから「答え全体を巨大マクロ 1 個に隠す」手は `L(D)` を食うが `L(G)` は食わない
  → B は隠せてしまい、C は隠せない。この非対称性が実験の要。


## 3. 結果（`experiment.py`, A=4, L=32, pop=160, gens=220, seeds=1..5, 約233秒）

### 3.1 主比較（系 × 環境）— seed 5 本平均

| 環境 | 系 | F | 総記述長 | L(G) | L(D) | 再利用率 | マクロ平均長 | 適応効率(×1e3) |
|---|---|---|---|---|---|---|---|---|
| periodic | A | 1.000 | 140.0 | 139 | 1 | 0.0 | 0.0 | 5.36 |
| periodic | B | 0.956 | **149.0** | 9.2 | 139.8 | 5.3 | **16.7** | 9.38 |
| periodic | C | 0.956 | **36.0** | 14.2 | 21.8 | 6.4 | **3.0** | **12.91** |
| structured | A | 1.000 | 140.0 | 139 | 1 | 0.0 | 0.0 | 5.36 |
| structured | B | 0.881 | **174.6** | 23.0 | 151.6 | 5.1 | 8.1 | 4.71 |
| structured | C | 0.850 | **65.4** | 39.0 | 26.4 | 2.9 | 3.5 | **9.39** |
| unstructured | A | 1.000 | 140.0 | 139 | 1 | 0.0 | 0.0 | 5.36 |
| unstructured | B | 0.700 | 193.8 | 24.8 | 169.0 | 17.7 | 13.0 | 3.46 |
| unstructured | C | 0.625 | 57.0 | 37.2 | 19.8 | 6.8 | 3.0 | 5.87 |

読み方:
* **A** は丸暗記（LD≈0, 総長 140 固定）。ノイズすら丸暗記して F=1.0。
* **B** は主プログラムを極小化（LG=9〜25）するが、内容を無罰のマクロに押し込むため
  **総記述長は縮まない（149・174・193 で A の 140 以上）／マクロ平均長が大きい（8〜17）**。
  ＝「短いプログラム」であって「短いゲノム」ではない。見かけの圧縮。
* **C** だけが総記述長を縮め（36・65）、それは **小さなマクロ（平均長 3）を何度も呼ぶ**
  形で達成される＝再利用可能な規則。適応効率（1bit あたりの適応）も全条件で最大。
* **ヌル（unstructured）**: C は F を落とさずには縮められない（F=0.625）。
  ＝見つける規則が無ければ圧縮は買えない。

**「自明に短くなった」と「再利用可能な規則が進化した」の分離**は、
`L(G)`（B も C も小）ではなく **総記述長 `L(G)+L(D)` と マクロ平均長** が担う。

### 3.2 λ スイープ（系 C）— 表現の相転移

periodic（seed 5 本平均）:

| λ | F | 総記述長 | マクロ平均長 |
|---|---|---|---|
| 0 | 1.000 | 152.8 | 8.1 |
| 1e-4 | 0.956 | 39.2 | 3.2 |
| 2e-3 | 0.956 | 36.0 | 3.0 |
| 4e-3 | 0.906 | 31.0 | 3.2 |
| 8e-3 | 0.619 | 20.8 | 1.0 |
| 1.6e-2 | 0.331 | 8.0 | 0.0 |

* λ=0→1e-4 で **総長 153→39・マクロ平均長 8→3** の跳び＝「大きな塊の記憶」→「規則の符号化」。
* λ≳8e-3 で F が崩壊（圧縮圧が適応を上回る）。**規則が生き残る λ の窓**が存在する。
* structured でも同型（λ=0 で 223bit/平均長13 → 小 λ で 65〜90bit/平均長3）。

### 3.3 再適応（環境変化後の回復世代数, A vs C）

同じ周期規則で motif だけ違う関連環境へ切替え、F=0.95 到達までの世代数:

| 系 | 回復 | 世代数（平均） | 切替直後 F |
|---|---|---|---|
| A | 5/5 | **36.8** | 0.331 |
| C | 5/5 | **12.0** | 0.463 |

C は **約 3 倍速く再適応**する。丸暗記の A は L 記号を学び直すのに対し、
C は 4 記号の motif だけ差し替えれば済む＝規則の再利用が直接効いている。

### 3.4 変異頑健性（構造ありチャンピオン, 単変異 300 回）

| 系 | 平均 |ΔF| | 中立率 | 有害率 | 最悪 |
|---|---|---|---|---|---|
| A | 0.025 | 0.19 | 0.81 | −0.031 |
| B | 0.190 | 0.36 | 0.64 | −0.719 |
| C | 0.136 | 0.25 | 0.75 | −0.406 |

**圧縮の代償**: 規則ベース（C）は 1 変異が多数の位置を同時に動かすため、
1 変異あたりの振れ幅は A より大きい（プレイオトロピーの集中）。
これは正直なトレードオフとして報告する（隠さない）。

### 3.5 対照（シャッフル）

periodic を位置シャッフル（ヒストグラム保存）すると、C は
構造 37bit(F=0.91) → シャッフル 50bit(F=0.74) と圧縮できなくなる。A は両方 140bit。
＝縮むのは構造由来であって記号頻度由来ではない。


## 4. 落とし穴（再発しやすい）

1. **λ のスケール**。ビットコストは O(100)、F は O(1)。素朴に λ=0.02 とすると
   罰則 0.02×140≈2.8 が F を圧殺し、C は空ゲノムに崩壊する（初回実測で観測）。
   規則が生き残る窓は λ≈1e-4〜4e-3。目的は必ず「F の尺度」に合わせてスケールする。
2. **罰する長さを間違えると圧縮は幻**。L(G) だけ罰する（系 B）と、
   内容がマクロへ移るだけで総ゲノムは縮まない。**必ず総記述長で評価する**。
3. **「短さ」だけでは規則の証拠にならない**。マクロ平均長も併記しないと、
   「大きな塊を 1 回呼ぶ（隠蔽）」と「小さな motif を何度も呼ぶ（規則）」を区別できない。
4. **展開は各段で L に打ち切る**。さもないと REP n×大マクロでコストが爆発する。
5. **マクロは DAG（前方参照禁止）**。再帰を許すと停止性と符号の一意性が壊れる。`validate` が強制。
6. **ヌル対照は必須**。C がノイズを縮めない（F を落とす）ことを示して初めて、
   構造環境での圧縮が「規則発見」だと言える。
7. シードは numpy Generator。Python `hash(` はプロセスごとに salt されるので使わない（テストあり）。


## 5. exp589 — 最適性 / Pareto 監査

exp588 の主張を「C は短い表現を発見した」から
**「C はその環境に存在する最小記述にどこまで接近したか」** へ強化する。
小さい問題（A=2, L=8/12/16）で、fitness を固定した**厳密**な最小記述長

$$L^*(F\ge f)=\min_{G,D}\{\,L(G)+L(D)\mid F(D(G))\ge f\,\}$$

と厳密 Pareto frontier を、`genome.py` と**同一のビット単位**で計算し、GA 解を

$$\Delta L=L_{\mathrm{GA}}-L^*(F\ge F_{\mathrm{GA}}),\qquad R_L=\frac{L_{\mathrm{GA}}}{L^*}$$

で評価する（`optimal.py` / `optimality_experiment.py`）。L=8,12 は全 $A^L$ 列挙で
大域 frontier を厳密算出、L=16 は高 F の Hamming 球（$F\ge0.75$ で厳密）で配置に必要な領域を算出。

**監査記述クラス（厳密性の適用範囲を明示）**: マクロ $\le2$ 個・各マクロは長さ $\le$ mlen の
LIT 列で**その展開文字列は x の部分列**（下記により十分）、プログラムは LIT/CALL/REP/MIR を
2次元 DP（位置 × プログラム長）で最適合成。符号の**パディング（記号0で長さ L まで補完）と
切り詰め**まで DP に組み込み、`genome.py` の記述長と厳密一致させている
（`test_min_program_matches_bruteforce_programs` が有界全 DSL 総当たりと突き合わせ）。
部分列で十分な理由: CALL/REP は文字列を verbatim に、MIR は前半を verbatim に出力するので
マクロの前向き文字列は必ず x 内に連続出現する。1回しか CALL されないマクロは総長を下げない
ので、有用な候補は「2回以上出力されうる部分列」に限られる。

### 5.1 中心的結果 — fitness 固定時の最適性ギャップ（平均 $R_L=L_{\mathrm{GA}}/L^*$）

| 環境 | A（記憶） | C\*（MDL 代表点） |
|---|---|---|
| periodic | **1.497** | **1.000** |
| structured | 1.252 | **1.054** |
| unstructured（ヌル） | 1.227 | **1.243** |

* **periodic**: C は $\Delta L=0$、すなわち**存在する最小記述に厳密到達**（L=12 で 29 bit、
  L=16 で 31 bit）。一方 A は 44/58 bit（$\Delta L=15/27$）で 50% 前後も冗長。
* **structured**（tile+mirror+nuisance）: C は最適の 5% 以内（L=12 で $\Delta L=0$、L=16 で $\Delta L=6$）。
* **unstructured（ヌル）**: C は A より最適に近づけ**ない**（$R_L$ が A と同等 1.24 前後）。
  ＝規則が無い環境では MDL は接近の優位を生まない。**最適性監査自体が対照を含む**。

### 5.2 交絡の除去（fitness 固定）

exp588 の 140 vs 36 bit 比較には「A は F=1.00、C は F=0.96」という交絡があった。
exp589 は **F=1.00 に固定**して比較できる。例（structured L=16）: A=(58 bit, F=1.0)、
C\*=(43 bit, F=1.0)、最適 $L^*(1.0)=37$。同一 fitness で C は A より 15 bit 最適に近い。

### 5.3 λ スイープが厳密 frontier をなぞる

λ を上げると C の champion は厳密 frontier 上を降りていく（F と L がともに frontier に沿って低下）。
periodic L=12/16 では λ∈[0.003, 0.015] のすべてで C が **frontier 上（$\Delta L=0$）** に乗る。
`pareto_frontier.png`（3 サイズ × 3 環境）: 灰=厳密 frontier、赤■=A（frontier の右方＝冗長）、
青●=C の λ スイープ（frontier の膝に張り付く）。ヌル環境では frontier 自体が急峻（圧縮余地なし）で
A も C も非圧縮コーナーへ押し込まれる。

### 5.4 exp589 の落とし穴

1. **符号のパディング/切り詰めを監査に写す**。記号0で L まで補完されるので、x の末尾ゼロは
   無料・末尾のブロックは L を超えて出力しても切り詰められる。これを DP に入れないと
   $L^*$ を過大評価する（実測で顕在化。0110 が literal 18 でなく空プログラム+補完で 15 bit）。
2. **λ のスケールは L 依存**。A=2 では literal が 20–50 bit なので有効 λ は 3e-3〜1.5e-2。
   exp588（A=4, L=32）の 2e-3 とは桁が違う。目的は必ず F の尺度に合わせる。
3. **$L^*$ は監査クラスに対して厳密**という限定を明示する。GA が $L^*$ を下回ったら
   （クラス外＝入れ子マクロ等を使った証拠）ランナーが検出して報告する設計。
4. 全列挙は A=2 でのみ現実的（$2^{16}=65536$）。A=4, L=32 の直接列挙は不可なので
   exp589 は A=2 の小問題に限定し、傾向の外挿は別途要検証。

## 6. exp590 — 変異近傍 / 探索効率監査

exp588 の再適応差（A=36.8 世代, C=12 世代）が **(a) 探索空間が小さいだけ** なのか
**(b) 圧縮で相関した表現型単位をまとめて操作できるようになった** のかを切り分ける
（`neighborhood.py` / `neighborhood_experiment.py`）。

各表現を「位置グルーピング復号」$\mathrm{decode}(\theta)_i=\theta_{\,\mathrm{group}(i)}$ で表し、
グルーピングだけを変えて 4 条件を作る（変異＝$\theta$ の 1 記号反転）:

| 条件 | グルーピング | 自由度 K | 意味 |
|---|---|---|---|
| A | 全個別 `[0..L-1]` | L | 非圧縮（記憶） |
| C | 周期整合 `i mod p` | p | **MDL-GA が実際に発見する構造** |
| R | ランダム | p | 同サイズだが**環境構造と無関係**な縮約表現 |
| C_shuffle | サイズ保存ランダム | p | ビット長・macro 長・CALL 数を保存し**再利用の配置だけ破壊** |

E0→E1（同周期・別 motif）で近傍指標と再適応 $T_{0.95}$（1+1 ES の手数）を測る。
$\eta_\mu=(\text{E1 で一致が増えた位置数})/(\text{変わった位置数})$ が「意味のある変異」の検出器。

### 6.1 結果（24 family 平均, A=2, L=16, p=4）

| 条件 | F(E0) | P_useful | E[ΔF\|>0] | d_P | η_ben | 再適応成功 | $T_{0.95}$ 手数 |
|---|---|---|---|---|---|---|---|
| A | 1.00 | 0.57 | 0.062 | **1.0** | 1.00 | 1.00 | **89.1** |
| C | 1.00 | 0.57 | **0.250** | **4.0** | 1.00 | 1.00 | **12.4** |
| R | 0.75 | 0.43 | 0.133 | 4.0 | **0.55** | **0.12** | ∞（頭打ち 0.70） |
| C_shuffle | 0.74 | 0.44 | 0.125 | 4.0 | **0.50** | **0.12** | ∞（頭打ち 0.70） |

**機構の分解（事前登録判定つき）**:

* **C vs A（機構＝相関した大きな移動）**: η_ben も P_useful も**同じ**（Hamming 目的は加法的で
  エピスタシスが無いため）。差は**移動の大きさ**: C は 1 変異で d_P=4 位置を同時に正しく動かし
  （E[ΔF|>0]=0.25 は A の 4 倍）、$T_{0.95}$ が 12.4 手 vs A の 89.1 手（約 7 倍速）。
  → 事前登録の「P_useful(C)>P_useful(A)」は**成立せず（引き分け）**。これは陰性だが有意味な結果で、
  「A に対する C の優位は 1 手あたり効率ではなく手の大きさ」と正確に言い切れる。
* **C vs R（機構＝サイズでなく構造整合）**: R は C と同次元（K=p）・同 d_P（=4）だが、
  グルーピングが環境と無関係なので大きな移動が**無駄**（η_ben 0.55<1、E0 すら完全再現できず 0.75）。
  **R は E1 に再適応できない（成功率 0.12、頭打ち 0.70）**。→ 「P_useful(C)>P_useful(R)」成立、
  「$T_{0.95}(C)<T_{0.95}(R)$」成立、「η(C)>η(R)」成立。
* **C vs C_shuffle（機構＝再利用の配置そのもの）**: 総ビット長・macro 長・CALL 数を保存し
  配置だけシャッフルすると、やはり再適応できない（成功率 0.12）。
  → **記述長ではなく再利用構造の配置が因果的**。

**事前登録の主判定**: $T_{0.95}(C)<T_{0.95}(A),T_{0.95}(R)$ **成立**、
$P_{\rm useful}(C)>P_{\rm useful}(R)$ **成立**（$>P_{\rm useful}(A)$ は加法目的ゆえ引き分け）。
**中核結論「探索空間縮小だけでは説明できず、圧縮で獲得した構造そのものが再適応を助ける」＝支持**。

補助（block 2, 実ゲノム近傍）: C の champion は L=31（A は 58）・reuse=4・span=4 で、
構造変異は d_P=2.5・|ΔL|=5.3 bit（長さが動く）、A は d_P=0.5・ΔL=0。
検証（block 3）: MDL-GA は 8 seed 中 5 で厳密な周期 p=4 規則（F≥0.95, reuse≥2, span≤5, L<45）に収束
＝block 1 の C が実際に発見される構造であることを確認（残り 3 も圧縮解だが別配置）。

### 6.2 限界（結果を弱く保つための明示）

* この機構は **高 fitness・構造あり条件** で現れる。exp589 の図の通り、低 fitness 側や
  無構造条件では C は最適から外れる。「C は常に最適/常に速い」とは主張しない。
* Hamming 目的は加法的でエピスタシスが無いため、A に対する C の優位は「手の大きさ」に限られ
  「1 手あたり効率 η」では出ない。$\eta_\mu(C)>\eta_\mu(A)$ を出すには、共適応ブロックを
  まとめて動かさないと得点しない**符号エピスタシス**目的が必要（→ 次の exp591 候補）。

## 7. exp591 — Epistatic Module Audit（最も強い識別）

exp590 の限界（加法的目的では $\eta_\mu(C)=\eta_\mu(A)$）を超えるため、**符号エピスタシス**
（Royal Road 型）目的を導入する。ブロック内が全ビットそろって初めて得点する:

$$F(x)=\frac1B\sum_{b=1}^{B}\mathbf 1[\,x_{\text{block }b}=t_{\text{block }b}\,]$$

（`epistatic.py` / `epistatic_experiment.py`）。1-bit オペレータ（A_bit）は谷を越えられない
（部分一致は得点ゼロ＝勾配なし）。モジュール境界がブロック境界と一致すれば、1 変異で
$0000\to1111$ を跳べて谷を越えられる。

**核心の対照**: これだけだと「C に有利な問題を作った」だけになりうるので、圧縮モジュールと
エピスタシス・ブロックの**一致**を 3 条件で振る:

| 条件 | エピスタシス分割 Π | C のモジュールとの関係 |
|---|---|---|
| E_aligned | 連続 p ブロック | **一致**（Π = C のモジュール） |
| E_shifted | 連続 p ブロックを p/2 ずらす | 不一致（C は 2 ブロックにまたがる） |
| E_random | ランダム size-p 分割 | 不一致 |

表現（**遺伝子＝ブロックの全ビット値**なので全表現が E0 を完全再現。差は「どの位置が一緒に
変異するか」だけ）: A_bit（1-bit GA）/ A_block（正解構造を手で与えた専門家オペレータ＝oracle）
/ C（連続 p ブロック＝MDL が発見する構造）/ R（同数ランダム）/ C_shuffle（サイズ保存・配置破壊）。

### 7.1 決定的結果（12 family 平均, L=16, p=4, B=4。E1 は E0 のビット反転＝最悪の谷）

| 条件 | 指標 | A_bit | A_block(oracle) | **C** | R | C_shuffle |
|---|---|---|---|---|---|---|
| **aligned** | P_useful | 0.000 | 0.068 | **0.068** | 0.000 | 0.000 |
| | η | 0.000 | 0.062 | **0.062** | 0.000 | 0.000 |
| | 谷越え手数 | 412 | 131 | **131** | 355 | 371 |
| **shifted** | P_useful | 0.000 | 0.067 | **0.000** | 0.000 | 0.000 |
| | 谷越え手数 | 427 | 129 | **314** | 371 | 356 |
| **random** | P_useful | 0.000 | 0.066 | **0.000** | 0.000 | 0.000 |
| | 谷越え手数 | 445 | 140 | **363** | 390 | 394 |

* **aligned のみ** C が A_bit を上回る: $P_{\rm useful}(C)=0.068>0=P_{\rm useful}(A_{\rm bit})$、
  $\eta_\mu(C)=0.062>0$（exp590 では出せなかった $\eta_\mu(C)>\eta_\mu(A)$ が成立）、
  $E[\Delta F|>0]_C=0.25>0$。そして **C ≈ A_block（131=131）≫ A_bit(412), R(355), C_shuffle(371)**。
* **shifted / random では優位が消失**: 同じ圧縮 C が、モジュールがブロックと一致しなくなると
  勾配を失い（P_useful→0）、谷越えが 314/363 手に悪化。一方 A_block（常に Π と一致する oracle）は
  条件によらず 129–140 手のまま。
* A_bit は全条件で勾配ゼロ（P_useful=0）＝谷では中立ドリフトでしか越えられない。

### 7.2 主張の格上げ

事前登録の中核判定「**aligned で優位・shifted/random で消失**」＝**成立**。これは
「圧縮量」ではなく「**圧縮が発見した構造と環境の依存構造の一致**」が原因だという最も強い識別。
主張は

> 圧縮表現は進化を高速化する

ではなく

> **環境の依存構造を圧縮表現が正しく捉えたとき、その表現は進化に有利な変異オペレータを暗黙に形成する**

になる。とくに **C ≈ A_block**（aligned）は、「MDL 自体に魔法の探索能力があるのではなく、
MDL が環境構造を発見した結果、専門家が手設計した適切な変異表現に近いものが自動形成される」
ことを示す。

### 7.3 限界

* 谷越えは全条件で最終的には成功する（中立ドリフトがあるため）。差は**手数と勾配の有無**で、
  「A は原理的に不可能」ではない（p を大きくすると差は指数的に開く）。
* Π・E0 は合成。実データの相互作用構造での再現は今後。

## 8. exp592 — Endogenous Interaction Discovery（相互作用構造の自力発見）

exp591 は真の相互作用分割 Π\* を実験者が与えて alignment を操作した。exp592 は **Π\* を
学習フェーズに一切渡さず**、raw な表現型標本と black-box fitness だけから圧縮表現を学習させ、
その分割 $\hat\Pi$ が Π\* に一致するか、そして未経験ゴールで良い変異オペレータになるかを検証する
（`endogenous.py` / `endogenous_experiment.py`）。

盲目の学習器は 2 つ:
* **naive_compressor**: 高 fitness 表現型標本の**共変動**で位置をクラスタ。共変動する位置は
  何でもまとめる＝fitness と無関係な nuisance も拾う（**foil**）。
* **fitness_driven**: 同じ共変動で候補モジュールを**提案**し、各候補を black-box fitness で
  **選別**（そのモジュールを動かすと fitness が動くか）。＝共変動が提案し fitness が選ぶ。

**leakage control**: 学習器は `(sample)` / `(sample, draw_goal, rng)` だけを受け取り、Π\* を一切
参照しない（テスト 2 種で担保: ①学習器のソースに `causal_blocks` 等が現れない、②真の Π\* を
**非連続に入れ替えた世界**でも学習器が fitness の定めるブロックを追随して復元 ARI>0.95）。
学習は目標 ∈ {0000, 1111}（整合モチーフ）のみで行い、**評価は未経験の任意 p-bit 目標**（hold-out）で
行う＝丸暗記でなく構造発見を要求。

### 8.1 決定的結果（6 world seed 平均, L=20＝causal 3×4 + nuisance 2×4）

**構造（Π\* はここで初めて開示）**

| 学習器 | ARI(·,Π\*) | pairwise P / R | nuisance をまとめた割合 |
|---|---|---|---|
| **fitness_driven** | **1.000** | 1.00 / 1.00 | **0.00** |
| naive | 0.716 | 0.60 / 1.00 | 1.00 |
| random | −0.02 | – | – |

**機能（hold-out ゴールでの谷越え $T_{0.95}$、$\hat\Pi$ を変異オペレータ化）**

| オペレータ | A_bit | R(非整合) | naive | **C_learned** | A_block(oracle) |
|---|---|---|---|---|---|
| $T_{0.95}$ 手数 | 393 | 789 | 309 | **309** | 309 |

* **盲目発見が成立**: fitness_driven の $\hat\Pi$ = Π\*（ARI=1.0）。fitness 観測だけから、
  相互作用ブロックと nuisance-singleton を完全復元。
* **nuisance 対照（minimal sufficient）**: naive は nuisance を全部まとめる（ARI 0.716、
  nuisance 割合 1.0）が、fitness_driven は causal だけ選ぶ（ARI 1.0、nuisance 割合 0.0）。
  ＝「圧縮可能な規則を見つけた」ではなく「**適応に関係する圧縮可能な規則を選択した**」。
* **機能が oracle に一致**: 未経験ゴールで **C_learned(309) = A_block(309, oracle) < A_bit(393)**、
  非整合 R は 789。学習した構造が、専門家が Π\* を手で与えたオペレータと同等に谷を越える。

事前登録の中核判定（$\operatorname{ARI}(\hat\Pi,\Pi^*)\gg\operatorname{ARI}(\text{random})$
かつ $T_{0.95}(C_{\rm learned})<T_{0.95}(A_{\rm bit})$ かつ
$T_{0.95}(C_{\rm learned})\approx T_{0.95}(A_{\rm block}^{oracle})$）＝**全て成立**。

（注: 機能テストは各オペレータの **causal 位置のグルーピングだけ**を比較するよう nuisance を一律
singleton に正規化してある。素の分割のままだと「nuisance を大モジュールにまとめた方が無駄引きが
減って速い」という手数の交絡が入るため。naive と C_learned が機能上同点なのは、両者とも causal
ブロックは発見しており、差は nuisance を拾うか否かという**構造/簡潔性**にあることを反映している。）


## 9. exp596 — Endogenous Complexity Escalation（内生的複雑性エスカレーション）

exp588–595 は「こちらが構造（目標・相互作用分割・分割列）を用意し、それを発見／再利用できるか」
を問うてきた。exp596 は**設計者をループから外す**: 個体ゲノム `G` と環境／課題 `E` の**両方**を
進化させ、`(G_t,E_t)→(G_{t+1},E_{t+1})`、そして

$$
\boxed{\text{新しい構造を必要とする理由そのものが系内部から生まれるか}}
$$

を測る。`escalation.py` / `escalation_experiment.py` / `test_escalation.py`。

**定式化（設計メモそのまま）**

$$
J_G=F(G,E)-\lambda L(G),\qquad
J_E=D(E,G)-\alpha L(E)-\beta\,\mathrm{Impossible}(E).
$$

`E` も exp588 コーデックの**ゲノム**として表す。ゆえに `L(E)` は復号可能な実ビット長、環境変異は
個体と同じ近傍の 1 ステップ、`Impossible(E)=1` は `C*(E)>K_max`（個体容量を超える＝到達不能）。

**中心指標は「ゲノム長」ではなく、環境を解くのに本当に必要な最小記述量**

$$
C_t=L^*(F\ge f\mid E_t)\quad(\text{exp589 の厳密 } L^*,\ A=2,\ L=16)
$$

を時系列で測る。加えて `L(G), K(G)=L(G)+L(D), N_{modules}, N_{interactions},
N_{novel}, N_{reused}, N_{recombined}` を追跡（機能的新規性は「履歴の全モジュールとの機能距離
`>ε` **かつ** knockout で fitness 低下」の 2 条件を満たすもののみ数える）。

### 9.1 環境目的の設計上の要点（落とし穴の宝庫）

素朴な `D=hardness=1-F0` は**サイクリング**に陥る: 全 0 個体に対し全 1 目標（どちらも `C=2`）が
最も「難しい」ので、環境は複雑性を上げずに 0↔1 を往復するだけになる（実測で確認）。本実装は
**Minimal-Criterion Coevolution** を採る:

$$
D(E,G)=\underbrace{[\,\text{到達可能}]}_{\text{最小規準}}\times
\big(\underbrace{\mathrm{novelty}(E;\text{archive})}_{\text{既習目標から離れる}}
+\underbrace{\text{compositional reachability}}_{\text{1 変異で多数位置が直る}}\big).
$$

到達可能（champion が少数変異で `mc_reach` 以上）でない課題＝ノイズを排除し、既習目標の
アーカイブから離れる新規性が、単純構造を枯渇させて**複雑性のラチェット**を生む。`C_t` は
全 0（`C=2`）から自発的に上昇する。

### 9.2 対照と容量スケーリング

| 対照 | 仕掛け | 期待 | 実測 |
|---|---|---|---|
| coevolution | 全系 | `C_t` が上昇 | init 2 → mean 7.8 / p75 11.7 / **peak 22.3** |
| **frozen** | `E` 固定（最重要） | plateau | **C=2 のまま**（mean 2.0, peak 2.0） |
| no-MDL | `λ=0` | 冗長膨張 | K=18.3 で `K−C` gap が最大（+7.8） |
| no-recomb | 再結合モジュール禁止 | 新規性停滞 | 再結合 0（機構として停止） |
| random-env | `E` を毎回ランダム | 適応不能 | F=0.94（coevo 0.96 より低い＝追随できない） |

容量 `K_max∈{8,16,24,32}` を振ると、境界膨張（`C≈K_max` が常に成立）では**なく**、
**フロンティア `C_peak` が容量とともに上昇**（同じ生成機構がより大きな容量で更に複雑な構造を作る）。
exp597 監査（90 ステップ）でも `C` は frozen ベースライン（2）を大きく上回って**持続**（崩壊しない）。

### 9.3 判定（`escalation_results.json` / `escalation.png`, 約140秒）

$$
\boxed{\textbf{ENDOGENOUS COMPLEXITY ESCALATION（頑健な核）= 成立}}
$$

* **c1 必要最小複雑度の上昇**（mean ≫ 2×初期, peak ≫ frozen）: ✔
* **c4 固定環境で停止**（frozen は種の複雑性 C=2 で plateau）: ✔
* **c5 容量を上げると同じ機構が継続**（`C_peak` が `K_max` とともに上昇, 境界張り付きでない）: ✔
* 支持対照: no-MDL 冗長膨張 ✔ / random-env 適応失敗 ✔ / 長時間で持続 ✔

一方、**より強い「モジュール水準の構成的 open-endedness」（birth→reuse→recombination→
functional novelty）は L=16 の厳密-`C` 世界では未解決（False）**。理由（正直に明記）:

1. **マクロ発見には勾配が無い**。正しいマクロを組むには複数の協調変異が必要で、その中間状態は
   fitness を上げない（干し草の中の針）。exp588 はこれを**固定目標 220 世代**で見つけた。
   coevolution は目標が毎マクロステップ動くため、短い内側 GA（8 世代）ではマクロを発見できず、
   champion はほぼリテラルで解く（実測: 周期3目標でも F=0.5 のリテラル接頭辞で停滞）。
2. **L=16 は階層を作る余地が小さい**。上がる `C_t` は主に run-length／リテラル由来で、
   多記号モチーフの反復（reuse）や入れ子（recombination）を要求しない。「`C` の上昇」と
   「モジュール再利用」は**階層構造**でのみ両立するが、厳密 `L*` の計算可能性（A=2, L≤16）と
   衝突する（L=24 では `C_t` 1 回 ~1 秒で全実験が非現実的）。

したがって exp596 は設計メモの結論どおり、**「endogenous complexity escalation の証拠」
であって「open-ended evolution そのもの」ではない**。モジュール水準の構成性は、より大きな世界・
長時間・独立シードで `C_t` を上界（発見済み最小記述）で追う exp597 の課題として残す。

### 9.4 落とし穴（exp596 固有・再発しやすい）

1. **スケール整合**。`D` は O(0.1–1)、`α·L(E)` は `α·数十` になりうる。素朴な `α=0.01` は
   `D` を圧殺し、環境は「最短の `E`（全 0）」へ崩壊する（実測）。`α≲0.001` に。
2. **`λ` は fitness 尺度に合わせる**。`λ=0.01` は F を潰し、frozen でも自明目標を F=0.58 でしか
   解けなくなる。A=2,L=16 では `λ≈0.006`。
3. **hardness は 1 個体 `G` に対して測る**。集団最大 F で測ると多様な集団が全課題を解いて
   hardness→0、環境が停滞する（実測）。
4. **純 hardness はサイクリングする**。アーカイブ novelty＋最小規準（到達可能）が必須（9.1）。
5. **`C_t` は振動する**。環境が動き続けるので傾きは不安定。頑健判定は mean / peak / 分位点と
   **frozen 対照**で行う（傾き単独に頼らない）。
6. **`C_t` は厳密だが小世界限定**。Hamming 半径 `⌊(1-f)L⌋` の球のみ監査するので `f_solve` と
   `L` に依存。L を上げると厳密計算が急激に重くなる（L=24 で実用外）。


## 10. exp597 — Hierarchical Compositional Escalation（階層的構成エスカレーション）

exp596 が示したのは弱い矢印 **外部 curriculum → 不要**（必要最小複雑度は内生的に上昇する）だけで、
強い矢印は失敗していた:

$$
\text{内生的複雑化}\ \not\Rightarrow\ \text{構成的新規性の継続}.
$$

exp597 は**この矢印だけ**を検証する。`hierarchical.py` / `hierarchical_experiment.py` /
`test_hierarchical.py`。

**exp596 からの変更点**

* 主実験から exact `L*`（L≤16）制約を外し **L=32/64/128**。複雑度は**上界付き MDL**
  `U_t`=「F≥f に達した復号可能ゲノムの最短記述長」（`L*` の厳密上界）で測り、exact `L*` は
  小規模 audit だけに残す（§Block4 で `U_t ≥ L*` を検証: 3/3 成立）。
* 環境を明示的に階層化: **atom → motif → module（2 motif の合成）→ composition（module の列）**。
  環境は **τ_E 世代ごとにだけ**、持続する motif を**再利用**し新 module に**再結合**する小変化で動く。
* 個体 decoder は **macro が macro を参照する階層**を許し、Re-Pair 風の `factorize`
  リファクタ演算（自分のゲノムだけを圧縮；目標は覗かない）でモジュール形成を可能にする。

測るのは `n_modules` 単独ではなく **D_hierarchy（階層深さ）, R_reuse（再利用呼び出し率）,
N_recombination, N_functional_novelty(t)**。新規性は exp596 基準（機能距離 `>ε` ∧ knockout で
fitness 低下）を維持し、**composition 由来か de-novo 由来かを系譜で区別**する。

### 10.1 中心的問い（事前登録・厳格判定）

$$
\boxed{\text{open-ended な構成的進化には、環境変化と表現学習の時間尺度の整合 }\tau_E\sim\tau_{\rm module}\text{ が必要か？}}
$$

`C_t` escalation **だけでは不合格**とし、以下を全要求:
functional novelty rate>0・reuse>0・**recombination>0**・hierarchy depth>1・
**後半でも novelty 継続**・容量を上げても継続。

### 10.2 結果（A=2, L=48, seeds=1..3, env_changes=6, τ_E∈{10,30,100,300}, 約16秒）

| 判定基準 | 結果 |
|---|---|
| C_t escalation（上界 MDL 上昇） | ✔ |
| functional novelty rate>0 | ✔ |
| reuse rate>0 | ✔（reuse_late≈0.4–0.5） |
| **recombination rate>0** | **✗（唯一の不合格）** |
| hierarchy depth>1 | ✔（深さ 2–3 が出現） |
| novelty 後半継続 | ✔ |
| 容量増でも継続 | ✔（K_max 120→200 で持続） |
| **COMPOSITIONAL_OPEN_ENDEDNESS（全基準）** | **✗** |

$$
\boxed{\textbf{強い矢印は「ほぼ」成立：6 基準中 5 達成、唯一 RECOMBINATION が未達}}
$$

* **flat reuse は即座に形成**（`τ_module`≈0）＝再利用はそもそも時間尺度律速では**ない**。
* **hierarchy depth>1・機能的新規性・後半継続・容量持続**は成立＝「内生的複雑化 → 構成的
  **新規性の継続**」は**達成**（新しい機能モジュールが後半でも生まれ続ける）。
* **唯一 recombination（≥2 個の異なる module を合成する macro）が未達**。原因を正直に特定:
  移動する目標上では個体が **F≈1 の clean match に到達できず**（best-τ の平均 F≈0.74）、
  貪欲圧縮器は **run/chain 階層**を作り、**multi-child の合成木**を作らない。
  すなわちボトルネックは τ_E の時間尺度**ではなく match 品質**であり、ユーザ仮説を**精緻化**する。
* **時間尺度の相転移は本データでは不成立**（novelty は 3.3/2.7/1.7/3.0 でノイズ的・内点ピーク無し）。
  2–3 seed では τ_E 依存の明瞭な相転移は確認できず、正直に False と報告。

### 10.3 対照と監査

* **no-factorize 対照**: factorize を切ると深い再利用機構が弱まる（factorize が機構）。
* **frozen-env 対照**: 環境を止めると後半 novelty が 0（新構造の要求が無い）。
* **exact-L\* audit**: 小 L=16 で `U_t ≥ L*`（39→51, 21→29, 21→21）＝上界 MDL は厳密 `L*` の
  妥当な上界。大規模系で `U_t` を複雑度プロキシに使う正当化。

### 10.4 到達点と落とし穴

$$
\text{内生的複雑化}\ \Rightarrow\ \boxed{\text{構成的「新規性」の継続（達成）}}\ ;\qquad
\text{構成的「再結合」}\ \Rightarrow\ \boxed{\text{未達（次の frontier）}}
$$

1. **`λ` スケール**（再掲・最頻再発）。`λ=0.01` は F を潰し、個体が自明ゲノムに崩壊（K=10, F=0.47）。
   literal が収まるよう `K_max` は寛容に、`λ≈0.001` で「潰さず圧縮」。
2. **recombination には clean parse が要る**。F<1 の近似一致を貪欲圧縮すると chain になる。
   貪欲 Re-Pair は周期列で run 階層（同一 child の入れ子）を作り、multi-child 合成を作りにくい。
   真の構成的再結合には、より良い文法推論（大域 Re-Pair＋index 管理）か F≈1 到達が要る。
3. **`τ_module` は意味のある形成で測る**。ランダム初期ゲノムは gen0 で自明な reuse を持つので
   「champion が reuse かつ F≥0.6」を条件にしないと `τ_module`=0 と誤測する。
4. **相転移主張は seed 数とノイズに注意**。2–3 seed では novelty のτ_E 依存はノイズに埋もれる。
   明瞭な内点ピーク（端点比 >1.3 倍）を要求し、無ければ正直に「未確立」と報告。


## 11. exp598 — Recombination Bottleneck Audit（再結合ボトルネックの切り分け）

exp597 の唯一の未達 `recombination rate>0` について、**別機能を足さず**、τ_E sweep も外して
（相転移は不成立だったため）、**なぜ `M_A+M_B → M_AB` が選択されないのか** だけを切り分ける。
`recomb_audit.py` / `recomb_audit_experiment.py` / `test_recomb_audit.py`。

**決定的な量**は、既存の 2 モジュール呼び出しを合成したときの**直接の MDL 差**:

$$
\Delta L_{\rm compose}=L(G_{\rm before})-L(G_{\rm after}),\qquad
\Delta L_{\rm compose}>0 \iff \text{recombination が得}.
$$

### 11.1 中心的結果（A=2, seeds=1..3, 約13秒）

**(1) 純 `ΔL_compose` グリッド**（`[CALL A, CALL B]` が k 回出現 → `M_AB` に合成）:

| k（合成対の再利用数） | 2 | 3 | 4 | 6 | 10 | 14 | 18 | 20 | 24 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| `ΔL_compose`（bit） | −9 | −6 | −7 | −3 | −1 | **+5** | +7 | +9 | +15 | +21 |

* **現実的な再利用（k≲10）では一貫して `ΔL_compose<0`**。損益分岐は **k≈14**。
* **motif 長に依存しない**（literal ではなく CALL を合成するため）。

**(2) 4 条件（match quality × factorizer, exp597 regime k=2）**:

| 条件 | recomb |
|---|---|
| `ΔL_compose`（flat vs recomb genome） | **−23** |
| evolved-imperfect（F≈0.87, greedy） | 0 |
| oracle-phenotype（F=1, greedy） | ~0.7（準最適・非再現） |
| oracle-factorization（evolved, optimal） | 0 |
| **oracle-both（F=1, optimal）** | **0** |

**(3) propose / survive / fix**: recombination は**変異で生成される**（k=2 で平均 9.3 回提案）が、
**MDL で淘汰され固定しない**（fixed≈0.3）。k=20 でも proposed 17.3 → fixed 0。

**(4) 大域 MDL 最適**: 最適 factorizer は**どの環境構造でも recombination を含まない**
（periodic/aperiodic/shared-motif いずれも `recomb_in_optimum=0`）。**REP が反復構造をより安く捕捉**する。

### 11.2 判定（決定表）

$$
\boxed{\text{ボトルネック = CODEC/MDL：現在の符号では recombination 自体に選択圧が無い }(\Delta L_{\rm compose}\le 0)}
$$

ユーザの決定表に正確に対応:

| F=1 | optimal factorizer | 本実験の観測 | 解釈 |
|---|---|---|---|
| 成功 | 成功 | — | match quality |
| 失敗 | 成功 | — | greedy factorizer |
| **失敗** | **失敗** | **oracle-both=0 かつ `ΔL≤0`** | **符号/MDL が recombination を報酬しない** |

* **match quality ではない**：F=1 を与えても最適表現に recombination は現れない。
* **greedy factorizer でもない**：最適（合成対応・MDL 単調）factorizer でも最適解は recomb=0。
* **根本原因**：`CALL`（2bit opcode + Elias-γ index）は抽象する 2–3 literal とほぼ同コストで、
  さらに module macro は**以降の全 macro index の γ コストを押し上げる**。ゆえに合成が得になるのは
  **k≈14 以上の再利用**に限られ、しかも反復構造は **REP がより安く**吸収する。
* **処方**：recombination を自発進化させるには **符号そのものの変更**（2-macro の `COMPOSE` opcode、
  index コスト削減など）が必要で、より大きな世界や探索強化では解けない。**変異で無理に出せば
  「自発的進化」ではなくなる**（＝ユーザの警告どおり、`ΔL≤0` で強制しない）。

### 11.3 到達点

$$
\text{birth}\rightarrow\text{reuse}\rightarrow\text{hierarchy}\rightarrow
\boxed{\text{recombination は符号コスト構造で頭打ち}}\quad(\text{次は codec 改良で再挑戦})
$$

exp596–597 で `endogenous challenge → required-complexity escalation → continuing
functional novelty → reuse → hierarchy` までは同一進化系で閉じた。exp598 は残る
recombination の非出現を **探索問題でも match 問題でもなく符号コスト問題**と**定量的に特定**した
（`ΔL_compose` の符号・損益分岐 k・REP 競合・propose/survive/fix）。次の自然な一手は、
`genome.py` の符号に安価な合成演算を導入し、`ΔL_compose>0` を現実的な k で成立させた上で、
**target を知らずにその合成を発見する探索**へ戻すこと。

### 11.4 落とし穴（exp598 固有）

1. **`ΔL_compose` は「合成対応・MDL 単調」な factorizer で測る**。素の Re-Pair は K を**増やす**
   factorization も返すため、`optimal_factorize` は「各段で記述長が減る時だけ採用」にする。
2. **clean env は 1 タイル正確長**にし、`cfg.L = env.L` を同期（さもないと phen 長≠target 長で
   `IndexError`）。
3. **REP 競合を忘れない**。flat vs recomb の 2 項比較だけだと recomb 有利に見える k でも、
   大域最適は REP を使い recomb=0。必ず**大域最適**（REP 込み）で判定する。
4. **oracle-phenotype(greedy) は稀に recomb を出す**が準最適（L が最小でない）。判定は
   **oracle-both（F=1＋最適）**で行う。


## 12. exp599a — Codec Evolution / Representation-Language Selection（遺伝言語の進化）

exp598 は「固定遺伝符号では recombination 自体に選択圧が無い」と特定した。exp599 は**符号を固定
する前提をやめ、遺伝言語 `D` 自体を選択対象**にする。`codec_evo.py` /
`codec_evo_experiment.py` / `test_codec_evo.py`（`genome.py` には触れない自己完結コーデック）。

$$
(G,D)\to x,\qquad J=F-\lambda\,[\,L(G\mid D)+L(D)\,],\qquad
D_0=\{\mathrm{LIT,CALL,REP}\},\ \ D_1=D_0+\{\mathrm{COMPOSE}\}.
$$

`COMPOSE a b` は 2 モジュールの合成を**1 opcode**で表す（`[CALL a, CALL b]` の 2 opcode に対し
`OPCODE_BITS` を節約）。新命令を持つこと自体に `L(D_1)=C_{\rm opcode}` を課金するので、`D_1` は
**合成が十分豊富な世界でのみ**得になる。

### 12.1 中心的問い

$$
\boxed{\text{recombination が有用な世界では、その操作を安く書ける遺伝言語そのものが選択されるか？}}
$$

### 12.2 結果（A=2, C_opcode=48, k=3, motif_len=4, seeds=1..3, 約8秒）

**(1) 三世界の言語選択**（予測表どおり）:

| world | 選択言語 |
|---|---|
| flat / REP（`ABAB…`） | **D0**（REP で十分） |
| compositional（多数の組合せ再利用） | **D1**（COMPOSE が得） |
| random | **D0**（COMPOSE コストが無駄） |

**(2) 厳密 MDL 相転移**（distinct compositions を掃引）: `n_comp≤6` で D0、`n_comp≥8` で D1。
`n_comp*_MDL = 8`。

**(3) 言語を遺伝子にした進化**: champion が D1 になる割合は `n_comp=8` で 0.67、`≥12` で 1.0。
**`n_comp_evo = 8 = n_comp*_MDL`**（進化の採用点が厳密 MDL 点と一致）。

**(4) opcode 価格が採用点を決める**: `C_opcode = 24/48/72/96 → n_comp*_MDL = 4/8/12/16`（単調）。
高価な命令ほど、採用に必要な合成圧が大きい。

$$
\boxed{\textbf{REPRESENTATION\_LANGUAGE\_IS\_SELECTED = True}\quad(n_{\rm evo}\approx n^*_{\rm MDL})}
$$

### 12.3 意味

これは単に「COMPOSE を足したら recombination できた」ではない:

> **既存の遺伝言語では高価だった進化操作に対し、環境の統計構造が十分な圧力を与えると、その操作を
> primitive として持つ新しい表現体系そのものが選択される。**

exp598 で見つけた「表現言語がボトルネック」という問題を、**世界を複雑化するのではなく進化側に
解かせた**。系列の位置づけ: 593 表現容量が進化 → 594 遺伝子機能の分化 → 595 構築規則の再利用 →
596 課題複雑度の内生的上昇 → 597 階層・novelty の継続 → 598 固定遺伝言語が進化を制限と判明 →
**599 遺伝言語そのものを進化させる**。

### 12.4 落とし穴（exp599 固有）

1. **flat 表現を候補から外さない**。`best_cost` に flat pair-macro（対の literal を 1 macro に格納）を
   入れ忘れると、COMPOSE が偽陽性で勝つ。短い motif では flat が安いので**必ず候補に含める**。
2. **motif 分解と合成を分離**する。literal から貪欲 MDL 降下させると、言語ごとに別の局所解
   （D0=flat、D1=stuck）へ発散する。**LIT 部分列の motif 分解（言語非依存）を先に済ませ**、その共有
   基底から各言語の**合成**を最適化する。
3. **C_opcode の谷を越えさせる**。D1 個体は最初に `C_opcode` を丸ごと払い、多数の合成を済ませて
   初めて回収する。1 変異 1 合成だと回収前に淘汰される。**合成は full pass**（その言語の最適へ一気に）
   にすると、選択が各言語の真の最適同士を比較でき、`n_evo≈n*_MDL` になる。
4. **COMPOSE の利得は per-composition（1 opcode）**。掃引軸は reuse `k` より**distinct compositions
   の数**（＝「多数の組合せ」）が素直。利得は `≈ 2·(#compositions)` bit、採用点 `≈ C_opcode/2`。


### 8.2 限界

* 発見機構は「ブロック内が {0000,1111} のように**位置が共変動**する」ことに依存（共変動が候補を提案）。
  共変動しない高次エピスタシス（各ブロック目標が一様ランダム）は pairwise では提案されず、
  部分集合探索が要る。実データ相互作用構造での再現は今後。
* fitness relevance $U$ は「モジュール単独で fitness を上げられるか」。ブロックを跨ぐ候補は
  提案段階で出ないので、この選別で十分だった。

## 9. exp593 — Adaptive Genome Capacity（表現能力そのものの進化）

exp588–592 は「**何を**表現するか」の進化だった。exp593 は初めて「**どれだけの表現能力を持つか自体**」を
検証する（`capacity.py` / `capacity_experiment.py`）。ゲノムを可変長にし、値変更だけでなく
**構造変更変異**を許す: ADD（新規マクロ）/ DUP（遺伝子重複 M→(M,M')）/ SPLIT / MERGE /
DELETE（未使用マクロ・命令の削除）/ REUSE（既存マクロの参照）。
**「増やせ／減らせ」という規則は一切実装せず**、増減変異と MDL 選択 $J=F-\lambda L$ だけを置く。

環境を段階的に複雑化・単純化する（rule 数 = 独立情報単位）:
$E_0\to E_1\to E_2\to E_3\to E_4 = 1\to2\to4\to2\to1$ rule。各 rule は別モチーフを**再利用**（tile）するので
マクロが得をする＝MDL 最適のマクロ数が rule 数を追う。

### 9.1 主軌跡（MDL on, 6 seed 平均, A=4 L=32）— 容量が需要を追う ↗↗↘↘

| stage | rule | F | L_total | used_macros | n_params |
|---|---|---|---|---|---|
| E0 | 1 | 1.000 | 39.7 | 1.0 | 4.3 |
| E1 | 2 | 0.990 | 59.3 | 2.3 | 5.3 |
| E2 | **4** | 0.943 | **110.8** | **4.0** | 11.0 |
| E3 | 2 | 0.990 | 77.8 | 2.3 | 9.5 |
| E4 | 1 | 1.000 | 39.7 | 1.0 | 4.7 |

L_total と used_macros がともに $1\to2\to4\to2\to1$ を描く。**「不足なら増え、冗長なら減る」が
選択圧から自発的に現れる**（H1・H2 とも成立）。「ゲノムが長くなった」のではなく、新しい独立情報が
要るとき**新しい表現単位（マクロ）そのものが誕生**し、要らなくなると**死ぬ**。

### 9.2 表現不足は測れる事実（exact $F^*(K)$, A=2 L=12 で全列挙）

$F^*(K)=\max_{L(G)\le K}F(G)$ を厳密計算。$K_{\min}(F{=}1)$: 1 rule=**23**, 2 rule=**32**, 3 rule=**32** bit。
rule 数が増えると最小記述長が伸びる＝**その容量では原理的に表現不足**という事実。よって
$F^*(K)<F_{\rm target}\Rightarrow K_{t+1}>K_t$（容量拡大）は「なんとなく」ではない。

### 9.3 対照 — growth と pruning を別々に因果分離

| 条件 | peak(4-rule) L | peak F | 1-rule に戻した後 L | 縮小量 |
|---|---|---|---|---|
| **MDL on** | 110.8 | 0.943 | 39.7 | **+71.2**（刈られる） |
| MDL off ($J=F$) | 495.0 | 0.854 | 585.7 | −90.7（**膨張・刈られない**） |
| ADD なし（固定容量） | 18.3 | **0.302** | 18.3 | 0（**fitness 頭打ち**） |
| DELETE なし | 191.5 | 0.885 | 199.0 | −7.5（増えるが**減らせない**） |

* **growth には ADD が必要**（ADD なしは 4-rule で F=0.30 に頭打ち＝表現不足のまま）。
* **pruning には MDL＋DELETE が必要**（MDL なしは膨張、DELETE なしは減らせない）。
両者が独立に因果検証された。

### 9.4 遺伝子重複と刈り込み

冗長な同一複製 $(M,M')$ から進化させると、**MDL あり → 総マクロ数 1.0（冗長複製が除去）**、
**MDL なし → 2.0（冗長複製が残存）**。duplication → redundancy →（MDL）pruning が確認できる。
分化（$M'\to$ 別モチーフ）は 2-rule 期に DUP＋値変異の経路で新モジュールを供給する
（DUP 演算は複製後の occurrence を張り替えるので、分化がそのまま特化になる）。

### 9.5 ヒステリシス（進化履歴の記憶）

同じ複雑度に戻しても一度複雑化を経たゲノムは完全には戻らない。complexity=2: $L_{\rm up}$(E1)=59.3 に対し
$L_{\rm down}$(E3)=**77.8**（+18.5）。complexity=1 は完全復帰（39.7=39.7）。中間複雑度で
$L_{\rm down}>L_{\rm up}$ ＝**進化履歴の記憶**が残る（探索的観察）。

事前登録判定（H1・H2、MDL-on は刈る、ADD-off は頭打ち、DELETE-off は減らせない、冗長複製は
MDL で刈られる）＝**全て成立**。

### 9.6 限界

* 容量ダイナミクスは A=4 L=32（マクロが確実に得をする系）、$F^*(K)$ は A=2 L=12（全列挙可能な小系）で、
  別パラメータ。$F^*(K)$ は原理（rule↑→最小記述↑）を厳密に示し、大系 GA は実際の増減を示す。
* λ=0.001 では peak でやや過剰成長（4 rule に 4〜6 マクロ）してから刈られる。厳密な rule 数一致は要求していない。
* ヒステリシスは中間複雑度でのみ観測。機構（どの構造が残るか）の同定は今後。

## 10. exp594 — Duplication / Divergence / Specialization

exp593 は「マクロ**数**が増減した」を示した。exp594 は「**遺伝子が新機能を獲得した**」への橋渡し:
新モジュールは (a) 既存の有用モジュールを**複製して分化**したのか、(b) 無関係な de-novo ADD が
たまたま獲得したのか、を**系譜**で区別する（`lineage.py` / `lineage_experiment.py`）。
各マクロに immutable id と parent id を付け、$M_a\xrightarrow{\rm DUP}(M_a,M_b)$ で parent を保存する。

環境: 機能 A（領域 R_A）と B（領域 R_B）を別領域に置く。B は 2 種:
$B_{\rm related}$（A から 1 記号だけ違う＝A の複製が 1 変異で届く）、$B_{\rm unrelated}$（各記号違う）。
$E_0$ は R_A のみ、$E_1$ は R_A+R_B を採点。選択は $J=F-\lambda L$ のみ（「複製せよ」とは教えない）。

### 10.1 系譜と機能分化（knockout, B_related, 8 seed）

2 モジュールに特化した run（4/8）で:
* **系譜**: B モジュールは **3/4 が複製由来（parent あり）**、de-novo ADD は 1/4。
* **特化（knockout $S(M_i,\cdot)$）**: S(A-mod,A)=**0.62** / S(A-mod,B)=0.02、
  S(B-mod,B)=**0.65** / S(B-mod,A)=0.00 ＝各モジュールが別機能を担う対角行列。
  → **duplication → divergence → specialization** が系譜つきで確認。

### 10.2 DUP-off 対照と related/unrelated（適応世代数 $T_{\rm adapt}$）

| 環境 | DUP-on | DUP-off | DUP 優位 |
|---|---|---|---|
| $B_{\rm related}$ | 154 | 298 | **+144** |
| $B_{\rm unrelated}$ | 92 | 212 | +120 |

* **DUP は適応を速める**（DUP-on ≪ DUP-off）。単に容量を増やせることではなく、
  **既存の有用構造をコピーして改変できること**に価値がある。
* **DUP 優位は $B_{\rm related}$（+144）> $B_{\rm unrelated}$（+120）**＝複製が「既存構造の再利用」である
  ことの強い対照（A に近い機能ほど複製が効く）。

### 10.3 潜在記憶（$A\to A+B\to A(\text{休眠})\to A+B$）— ヒステリシスは有用な記憶か

| 条件 | 再取得 $T$ | B の痕跡が休眠を生き延びた |
|---|---|---|
| first acquire（初回） | 97 | — |
| 休眠=MDL で刈る | **22** | 0/8 |
| 休眠=MDL なしで保持 | 69 | 8/8 |
| reset（記憶なしの素の集団） | 137 | — |

* **ヒステリシスは有用な進化的記憶**: A+B を経験した集団は、素の集団（reset 137）より
  **はるかに速く再取得**（22〜69）。exp593 の $L_{\rm down}>L_{\rm up}$ は単なるゴミではない。
* **しかも記憶は「圧縮された生成能力」**: 文字通りの $M_B$ 痕跡は MDL に刈られる（0/8）のに、
  刈った条件が**最速**（22）。つまり記憶は保存された B 疑似遺伝子ではなく、
  **洗練された A-module ＋ 複製経路**という圧縮された生成能力に宿る。MDL は記憶を壊すのではなく
  **研ぎ澄ます**。これは系列全体の「最小表現」テーマと一致する強い（かつ予想外の）結果。

事前登録判定（系譜=複製優勢、特化、DUP が効く、related>unrelated、経験は再取得を速める）＝**全て成立**、
副結果として「記憶は圧縮された生成能力」も成立。

### 10.4 限界

* 2 モジュール特化は 4/8 で発生（残りは R_B を literal で解く非モジュール解）。特化した run では
  複製由来が優勢という条件付きの主張。特化率を上げる環境設計は今後。
* 「痕跡の保持」は休眠期の λ（MDL の有無）に依存。MDL ありでは疑似遺伝子は残らない。

## 11. 限界と次

* 命令言語は LIT/CALL/REP/MIR の 4 種、マクロは DAG のみ、表現型は長さ L の 1 列。
* $L^*$ は「マクロ $\le2$・部分列・flat」クラスに対して厳密（入れ子マクロを含む全 DSL は未算出）。
* **次**: exp595 の続き。

## 11. exp595 — Dynamic Interaction Recomposition

exp594 までで「部品の生成・特化・削除・再獲得」まで来た。exp595 は**部品を保持したまま
相互作用構造だけが時間変化する世界**で、**既存部品の組み替えによる適応**を検証する
（`recompose.py` / `recompose_experiment.py`）。真の構造を
$\Pi_0^*\to\Pi_1^*\to\cdots$ と変化させ、変化を
REUSE / MERGE / SPLIT / RECOMBINE / NOVEL に分類する。

すべて **atom**（常に共変動する位置ペア A=[0,1]…H=[14,15]）上で定義。$\Pi_t^*$ は atom を
block にまとめる。学習器に $\Pi_t^*$ は渡さず、高 fitness 表現型を観測して共変動でクラスタするだけ
（exp592 と同じ blind）。**主指標は fitness でなく「新構造発見コスト」**:

$$N_{0.9}^{(t)}=\min\{n:\operatorname{ARI}(\hat\Pi_t,\Pi_t^*)\ge0.9\}$$

条件（記憶は clustering の単位と候補ライブラリを変える）: reset（位置・記憶なし）/
modules-only（過去 block の block 単位編集: merge/split）/ operators-only（atom の移動・交換＝
recombine を可能にする構築演算）/ full / frozen（block 固定・merge のみ）/ oracle。

### 11.1 発見コスト $N_{0.9}$（変化種別 × 条件）

| 変化 | reset | modules | operators | full | frozen | oracle |
|---|---|---|---|---|---|---|
| REUSE | 14 | 8 | **5** | 8 | 8 | 2 |
| MERGE | 10 | **3** | 10 | **3** | **3** | 2 |
| SPLIT | 10 | 10 | 10 | 10 | 10 | 2 |
| RECOMBINE | 8 | 8 | **6** | 8 | 8 | 2 |
| NOVEL | 8 | 8 | 8 | 8 | 8 | 2 |

* **REUSE/MERGE/RECOMBINE で記憶が発見を速める**（memory < reset）。組み替え可能な変化では、
  過去に獲得した部品が**新構造の材料**になる。
* **NOVEL では転移が消える**（全条件 = reset = 8）。atom を跨ぐ構造は既知部品と無関係なので、
  記憶は役に立たない（負転移も生じない設計＝候補が合わなければ clustering に fallback）。
* **frozen（block 固定）は RECOMBINE できない**（8、= reset）。組み替えには block を割って atom を
  跨がせる必要がある。
* **SPLIT は 1-step 記憶では transfer せず**（全 10）。ABCD→AB,CD の復元には 2 手前の部分 block
  記憶が要る。正直な限界。

### 11.2 何が記憶されているのか（operators vs modules）— RECOMBINE

| 条件 | reset | modules | operators | full |
|---|---|---|---|---|
| RECOMBINE $N_{0.9}$ | 8 | 8 | **6** | 8 |

RECOMBINE を安くできるのは **operators-only（atom の再グループ化演算）だけ**。過去 block を
そのまま持つ modules-only は AB,CD を AC,BD にできず reset と同じ。＝exp594 の「記憶の正体は
昔の遺伝子でなく**有用な表現を作り出す方法そのもの**」を、時間方向で再確認。
（full は modules 候補を足す分わずかに spurious 選択が増え operators より劣る＝古い部品の抱え込みは
組み替えにはむしろノイズ。）

### 11.3 二段階 $N_{0.9}\to T_{0.95}$

full 記憶は発見コストを下げ（REUSE 8<20, MERGE 2<8, RECOMBINE 8<14）、**発見した構造を変異
オペレータに使うと適応も速い**（$T_{0.95}$ 記憶 vs reset が各変化で約 2 倍速）。＝「構造を早く
理解したから適応も早い」という二段階を時間順序で確認。

### 11.4 系譜による組み替えの直接確認

block id の親をたどると、新 block が過去部品から作られたことを分類できる:
MERGE は `[0..7]<-MERGE[1,2]`、RECOMBINE は `[0,1,4,5]<-parents[1002,1003]`（atom を割って再結合）、
NOVEL は既知部品と無関係。＝「速く適応した」でなく「**過去に獲得した遺伝的部品が新構造の材料に
使われた**」を系譜から確認。

事前登録判定（REUSE/MERGE/RECOMBINE で transfer、NOVEL で消失、frozen は RECOMBINE 不可、
記憶＝構築演算）＝**全て成立**。

### 11.5 限界

* atom は常に完全共変動なので clustering で自明に見つかる。転移の効きは「候補集合による model
  selection の安さ」に由来。ノイズ入り観測での再現は今後。
* 1-step 記憶（直前の学習分割のみ）。深い系譜記憶（数手前の部分 block）を持てば SPLIT も transfer
  し得る。
* NOVEL は fallback により負転移を回避。強制保持（frozen 的）にすれば構造的慣性（負転移）も出せる。

## 12. 限界と次

* 命令言語は LIT/CALL/REP/MIR の 4 種、マクロは DAG のみ、表現型は長さ L の 1 列。
* **本命の次**: 環境複雑度を事前段階設定するのをやめ、**環境と個体を共進化**させる。新課題が継続的に
  生まれたときゲノム複雑度が上限に張り付かず **birth → reuse → recombination → novelty** を継続するか
  ＝ここで初めて「open-ended evolution」を実験的に検討できる。
  実データ接続、および `Consciousness_model` の `CompressedWorkspace` との対応づけ。

**因果鎖（exp588–595 で通した）**:
environmental structure → **compression**(588) → **minimal representation**(589) →
**modular mutation**(590) → **epistatic valley crossing**(591) → **epistasis 構造の自力発見**(592) →
**表現能力の適応的増減**(593) → **複製→分化→特化と圧縮された進化的記憶**(594) →
**部品の組み替えによる新構造適応・転移する記憶＝構築演算**(595) → evolvability / open-endedness。
各段で「構造あり・整合・因果・需要・関連・組替可能条件でのみ効果が出て、無構造・非整合・nuisance・
規則欠如・NOVEL では消える」対照を必ず置いている。

**この系列の到達点**: 圧縮圧の下で、**表現内容だけでなく、表現容量・モジュール・系譜・再構成規則までが
適応対象になり得る**ことを、各段で厳密／対照つきに実装検証した。

## 13. ファイル

| ファイル | 役割 |
|---|---|
| `genome.py` | 復号可能なゲノム符号（DSL・encode/decode・L(G)/L(D)・展開・再利用計量）。中核はドメイン中立 |
| `environments.py` | 構造あり/なし・シャッフル対照・関連環境族の目標列 |
| `evolution.py` | GA エンジンと 4 目的（A/B/C/D）・変異作用素 |
| `metrics.py` | 適応効率・変異頑健性・再適応 |
| `experiment.py` | exp588 全 5 ブロックのランナー |
| `optimal.py` / `optimality_experiment.py` | exp589 厳密最適性・Pareto 監査 |
| `neighborhood.py` / `neighborhood_experiment.py` | exp590 変異近傍・探索効率 |
| `epistatic.py` / `epistatic_experiment.py` | exp591 エピスタシス・モジュール監査 |
| `endogenous.py` / `endogenous_experiment.py` | exp592 相互作用構造の自力発見 |
| `capacity.py` / `capacity_experiment.py` | exp593 可変容量・遺伝子誕生死 |
| `lineage.py` / `lineage_experiment.py` | exp594 複製・分化・特化・潜在記憶 |
| `recompose.py` / `recompose_experiment.py` | exp595 動的組み替え・転移・発見コスト |
| `escalation.py` / `escalation_experiment.py` | exp596 内生的複雑性エスカレーション（G と E の共進化） |
| `hierarchical.py` / `hierarchical_experiment.py` | exp597 階層的構成エスカレーション（factorize・階層・τ_E 相転移） |
| `recomb_audit.py` / `recomb_audit_experiment.py` | exp598 再結合ボトルネック監査（ΔL_compose・決定表） |
| `codec_evo.py` / `codec_evo_experiment.py` | exp599a 遗伝言語の進化（COMPOSE・L(D)・言語相転移） |
| `test_compressed_genome.py` | 全 exp の符号・対照・監査・系譜・組み替え（56 件） |
| `test_escalation.py` | exp596 の決定性・C*・モジュール/新規性/再利用/再結合会計・対照（13 件） |
| `test_hierarchical.py` | exp597 の factorize 不変性・階層深さ・再利用率・再結合・対照（10 件） |
| `test_recomb_audit.py` | exp598 の手組みゲノム・ΔL_compose 符号・factorizer 不変性（8 件） |
| `test_codec_evo.py` | exp599 の COMPOSE 符号・三世界選択・MDL 相転移・n_evo≈n*_MDL（9 件） |

実行:
```
python -m pytest test_compressed_genome.py -q   # 56 passed（総当たり・段階GAを含むため約2分）
python experiment.py                            # exp588  （約4分）
python optimality_experiment.py                 # exp589  （約7分）
python neighborhood_experiment.py               # exp590  （約25秒）
python epistatic_experiment.py                  # exp591  （約50秒）
python endogenous_experiment.py                 # exp592  （約30秒）
python capacity_experiment.py                   # exp593  （約7分）
python lineage_experiment.py                    # exp594  （約5分）
python recompose_experiment.py                  # exp595  （約70秒）
python -m pytest test_escalation.py -q          # exp596  （13 passed, 約7秒）
python escalation_experiment.py                 # exp596  （約2〜2.5分）
python -m pytest test_hierarchical.py -q         # exp597  （10 passed, 約1秒）
python hierarchical_experiment.py               # exp597  （約15秒）
python -m pytest test_recomb_audit.py -q         # exp598  （8 passed, 約1秒）
python recomb_audit_experiment.py               # exp598  （約13秒）
python -m pytest test_codec_evo.py -q            # exp599  （9 passed, 約10秒）
python codec_evo_experiment.py                  # exp599  （約8秒）
```

## 14. exp599b -- Emergent Genetic Instruction Set（primitive grammar からの命令発明）

exp599a は候補 `COMPOSE` を持つ言語の採用を示した。exp599b はその候補名も意味分岐も decoder に
与えない。新 opcode は postfix の小プログラムとして、固定 primitive
`{LITERAL, REF, CONCAT, REPEAT}` から機械的に列挙する。

$$
O_i(\theta)=P_i,\qquad L_{\rm total}=L(D)+L(G\mid D),\qquad
L(D)=\sum_i L(P_i).
$$

定義の名前ではなく、複数の reference-input case で
`O_i(a,b) == CONCAT(REF(a), REF(b))` を満たすかを**後から**検査して
compose-equivalent と分類する。`instruction_evo.py` はこの DSL・評価器・定義長・出生選択を実装し、
`instruction_evo_experiment.py` と `test_instruction_evo.py` が対照・監査を行う。

### 14.1 結果（A=2, k=3, motif_len=4, definition max=3 tokens）

| world | birth OFF | birth ON | oracle | born opcode |
|---|---:|---:|---:|---|
| flat / REP | 29 | 29 | 29 | なし |
| compositional (`n_comp=8`) | 306 | **264** | **264** | あり、compose-equivalent |
| random | 302 | 302 | 302 | なし |

出生した定義は primitive token 列 `REF(0), REF(1), CONCAT` であり、`L(D)=8` bit、
`L(G|D)=256` bit、opcode 使用数 24、definition depth 2 である。これは `COMPOSE` という
別実装・enum 名を選んだ結果ではなく、構文列挙後の意味テストで同値と判定されたもの。

`n_comp=1` では fixed / birth ON / oracle がいずれも 48 bit で出生しない。`n_comp=2` で
`92 -> 86` bit となり、厳密 MDL 出生点は `n_opcode_birth*_MDL = 2`。primitive definition の
価格は 8 bit であり、exp599a の 48-bit 固定 opcode より安いので、相転移点が早いことはモデルの
予測どおりである。grammar mutation を順不同に提案し、全記述長を下げる定義だけを保持する対照的な
proposal-selection でも `n_opcode_birth_evo = 2` となった。

$$
\boxed{L(D)\uparrow,\quad L(G\mid D)\downarrow,\quad L_{\rm total}\downarrow}
$$

### 14.2 解釈と限界

この結果は「与えられた primitive grammar の有限探索範囲で、composition の統計が十分あるとき、
compose-equivalent な短い decoder program が MDL により保持される」ことを示す。汎用的な新命令発明や、
opcode が別 opcode を呼ぶ二次言語階層まではまだ検証していない。現実的な次段階は、定義 DSL に既存
opcode 呼出を導入し、環境の変更後にも発明済み操作が保持・転用されるかを独立した holdout で測ること。

## 15. exp600 -- Instruction Necessity / Alternative-Grammar Audit（抽象操作の収束監査）

exp599b の `REF(0), REF(1), CONCAT` は postfix DSL 固有の偶然かを検査する。異なる token 名・stack
規則・補助 primitive を持つ3 grammarで、短い定義を独立に全列挙し、**構文名ではなく入力出力意味論**で
比較する。

| grammar | 発明された token 列 | total bits | concatenate-equivalent |
|---|---|---:|---|
| `postfix_ref_concat` | `REF0, REF1, CONCAT` | 264 | True |
| `copy_append` | `COPY0, COPY1, APPEND` | 264 | True |
| `stack_merge` | `PUSH0, PUSH1, MERGE` | 264 | True |

3 grammar とも fixed language は 306 bit で、`L(D)=8` bit を支払って 264 bit へ下がった。各 winner は
空列・異なる長さ・異なる順序を含む reference-input cases で $a\Vert b$ と一致する。birth threshold も全て
`n_opcode_birth*_MDL=2` だった。

対照として、flat/REP と random world では全 capable grammar が opcode を出生させない。また binary
merge を持たない `copy_repeat_only` grammar は compositional world でも出生せず、306 bit の fixed 表現に
留まる。

$$
\boxed{\text{within the audited grammars: composition semantics converge, but primitive availability constrains invention}}
$$

この結果は「構文の表面形ではなく composition の意味論が選ばれた」ことを強める。ただし3 capable
grammar はすべて順序保存する二項 merge primitive（`CONCAT` / `APPEND` / `MERGE`）を持つ。従って
environment 単独から任意の basis を越えて composition が必然的に発見される証明ではない。次の exp601 は
invented opcode 呼出を definition grammar に追加し、二次 opcode・保持・環境切替後の再利用を holdout で
測る。

## 16. exp601 -- Primitive Basis Ablation / Computational Necessity（計算能力の必要性）

exp600 の capable grammar はすべて順序保存 binary merge を持っていた。exp601 は primitive basis $P$ ごとに
concatenation の意味論的記述長

$$
K_P(a\Vert b)=\min_{q\in P:\ q(a,b)=a\Vert b}L(q)
$$

を全列挙で求め、$K_P$ と命令出生閾値を比較する。`primitive_basis_ablation.py` は function expressibility と
opcode reuse を独立に操作する。

| basis | $a\Vert b$ | $K_P$ | reuse | $n^*_{birth,MDL}$ | outcome |
|---|---|---:|---|---:|---|
| `full` | expressible | 8 | 可 | 2 | `REF0, REF1, MERGE` が出生 |
| `distant_box_pair_flatten` | expressible | 14 | 可 | 3 | BOX/PAIR/FLATTEN 経路で出生 |
| `minus_ref` | 非表現 | -- | 可 | -- | 出生なし |
| `minus_order` | 非表現 | -- | 可 | -- | 出生なし |
| `minus_merge` | 非表現 | -- | 可 | -- | 出生なし |
| `minus_reuse` | expressible | 8 | 不可 | -- | 定義を償却できず出生なし |

`n_comp=8` では full は `306 -> 264` bit、distant basis は `306 -> 270` bit となる。後者も有利だが、
より長い decoder definition を回収するため birth は `n_comp=3` まで遅れる。full の `n_comp=2` より遅い
この差は、同じ意味が表現可能かだけでなく、その basis 上でどれだけ短く書けるかが選択を変えることを示す。

flat/REP と random では full / distant のどちらも出生しない。これは `K_P` だけでなく環境側の反復による
償却が必要であることを確認する対照である。

$$
\boxed{\text{within audited bases: }K_P(a\Vert b)\uparrow\ \Rightarrow\ n^*_{birth}\uparrow}
$$

この関係は有限の token 集合・最大探索深度・この compositional world に対する実証であり、一般の全計算体系への
定理ではない。それでも、exp602 で invented opcode を新 primitive として追加し、
$K_{P_0\cup O_1}(f_2)<K_{P_0}(f_2)$ が二次命令の早い出生として現れるかを測るための、必要な基準線になる。

## 17. exp602 -- Cumulative Instruction Bootstrapping（累積的表現進化）

exp602 は exp601 の distant typed basis で E1 から得た O1 を、後続 definition grammar の callable primitive
として保持する。O1 は名前で与えず、exp601 の最短定義探索から
`REF0, BOX, REF1, PAIR, FLATTEN`（14 bit）として回収する。

E2 では四項意味論 $f_2(a,b,c,d)=a\Vert b\Vert c\Vert d$ の O2 を調べる。base、O1 を保持した useful
history、同じ 14 bit だが $a\Vert a$ を実行する irrelevant history を比較する。過去の history は既に E1
で保持された sunk cost として条件づけ、O2 の**追加**記述長だけを比較する。

| history | $K(f_2)$ | $n^*_{birth,MDL}$ |
|---|---:|---:|
| base | 36 | 3 |
| useful O1 | **18** | **2** |
| irrelevant（同じ履歴コスト） | 36 | 3 |

useful history だけが `O1(r0, O1(r1, O1(r2,r3)))` を許し、意味論的距離と出生閾値を下げる。従って単に
language が大きくなったのではなく、**後続課題に適合する過去の抽象**が O2 を近くしたと識別できる。

小規模 cumulative trace も実装した。各段で同じ課題を original basis $P_0$ と累積 basis で比較する。

| stage | target arity | $K_{P_0}$ | $K_{P_{t-1}}$ | $n^*_{P_0}$ | $n^*_{history}$ |
|---:|---:|---:|---:|---:|---:|
| O1 | 2 | 14 | 14 | 3 | 3 |
| O2 | 4 | 36 | **18** | 3 | **2** |
| O3 | 8 | 80 | **30** | 6 | **2** |

$$
\boxed{\text{environment}\rightarrow O_1\rightarrow P_1\rightarrow O_2\rightarrow P_2\rightarrow O_3}
$$

これは open-endedness の証明ではない。有限の typed normal-form grammar、arity 2/4/8、既知の反復世界における
**cumulative representational evolution** の実証である。次の検証では、過去の opcode が環境切替後も保持され、
独立 holdout 課題へ転用されるかを測る必要がある。

## 18. exp603 -- Abstraction Retention Audit（抽象命令の保持・削除）

exp602 は O1/O2 を自動保持していた。exp603 ではそれを外し、最終 task を記述する言語が O1/O2 を retain / delete
するかを、維持費を含む MDL だけで選ばせる。

$$
L_{\rm total}=L(D_{\rm retained})+\sum_jL(T_j\mid D_{\rm retained}).
$$

O1 の維持費は 14 bit。O2 は O1 を残すと 18 bit だが、O1 を削除して O2 だけ残すと、O1 への参照を
self-contained にコンパイルするため 36 bit を払う。これにより delete-O1 条件が隠れた依存を無料で使えない。

### 18.1 単一の O3 task（arity 8）

| condition | retained | $L(D)$ | $L(O3\mid D)$ | total |
|---|---|---:|---:|---:|
| reset | -- | 0 | 80 | 80 |
| delete-O2 | O1 | 14 | 38 | **52** |
| delete-O1 | compiled O2 | 36 | 36 | 72 |
| full-history | O1, O2 | 32 | 30 | 62 |

MDL-free-choice は **O1 のみ保持して O2 を削除**する。O2 の短縮効果 8 bit は、その task 一件では 18-bit の
維持費を回収できない。これは命令が永久に増えるのでなく、過去の抽象を必要に応じて prune する言語版の
birth/death である。

### 18.2 将来 reuse による保持

独立した O3 型の将来 task が3件ある workload では、free-choice は full-history を選ぶ。

| condition | total |
|---|---:|
| reset | 240 |
| O1 のみ | 128 |
| compiled O2 のみ | 144 |
| O1, O2 | **122** |

$$
\boxed{\text{MDL retains an intermediate abstraction iff its future reuse amortizes maintenance}}
$$

これは有限の候補集合 `{O1,O2}` と既知の O3 型 workload に対する**厳密 MDL subset selection**であって、
mutation による長期進化の実証ではない。それでも exp604 で endogenous challenge generation と可変言語を
接続する前に必要な、強制的な永久保持という足場を外した検証になっている。

## 19. exp604 -- Endogenous Abstraction Ecology（内生的抽象生態系）

`endogenous_abstraction_ecology.py` は環境 $E_t$、保持言語 $D_t$、到達可能な命令集合 $G_t$ を結合し、
各 step で **birth -> use -> reuse -> retain -> compile/prune** を追跡する。候補は arity 2/4/8 の
`O1/O2/O3` に限定するが、高 arity task は時刻ではなく現在 retained されている前段命令によってのみ
到達可能になる。したがって O1 を削除した系は O2/O3 の task を環境から受け取れない。これは手書きの
stage schedule ではなく basis-dependent reachability である一方、候補集合と score rule は明示的に有限である。

各 task について、保持 subset は維持費と task 符号長を含む正確な MDL で選ぶ。

$$
A_t=K_{P_0}(E_t)-K_{P_t}(E_t),
\qquad
L_t=L(D_t)+L(E_t\mid D_t).
$$

対照は `adaptive_language`、`fixed_language`、`no_retention`、`keep_all`。`no_retention` はその step で
命令を生んで使えても直後に捨てる。`keep_all` は生まれた命令を永久保持する。runner は各命令について
birth/use/reuse/retain/prune を出力し、$N_{\rm birth}$、$N_{\rm retained}$、$N_{\rm pruned}$、
$N_{\rm reused}$、$D_{\rm language}$、$A_t$ を記録する。

### 19.1 結果（30 steps, seeds 1--3）

| condition | late birth | late old-instruction reuse | prune | late $A_t$ | max $L(D)$ | mean $L_t$ |
|---|---:|---:|---:|---:|---:|---:|
| adaptive-language | 15.0 | 25.0 | 19.0 | 31.3 | 52 | **142.3** |
| fixed-language | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 57.7 |
| no-retention | 15.0 | 15.0 | 0.0 | 6.0 | 0 | 24.0 |
| keep-all | 0.0 | 45.0 | 0.0 | 54.0 | 62 | 176.1 |

adaptive は late birth、過去命令の late reuse、retain と prune、正の late advantage をすべて示す。
固定言語と no-retention は O1 水準を越えず、累積 advantage も adaptive より小さい。keep-all はより大きい
task advantage を保持しうるが、同じ到達性を持つ adaptive より平均記述長が大きい。従って「adaptive が
keep-all より強い」を raw advantage でなく、**役割を失った定義を捨てたときの記述効率**として判定する。
adaptive の最大 $L(D)=52$ は capacity 64 未満であり、上限飽和だけが結果を作ってはいない。

事前登録した全判定は `endogenous_abstraction_ecology_experiment.py` で真となった。この系が示すのは、
有限 candidate ecology 内の **self-sustaining cumulative abstraction** である。未知の操作を無制限に作る
open-ended evolution、任意環境での一般化、自然進化の機構を示すものではない。

## 20. exp605 -- Scaling / Plateau Audit（時間・容量・意味 catalog の天井監査）

exp604 の birth 数だけを伸び続ける novelty と解釈しないため、`scaling_plateau_audit.py` は time $T$、
language capacity $K_{\max}$、有限 arity catalog $A_{\max}$ を独立に sweep する。catalog は
`{2,4,...,Amax}` の order-preserving concatenation operation であり、上位 task は下位 operation が
retained の場合だけ到達可能である。

命令 $O$ は、過去の semantic history に含まれる operation と typed arity が異なる
$\nu(O)=1$（同一なら 0）で、かつその task の MDL を実際に下げる初回 birth のときだけ functional
semantic novelty と数える。従って prune 後の同じ arity の birth は novelty でなく reinvention である。
この typed distance は本 experiment の concat-only DSL では正確だが、異種の意味 operator 間の豊かな
距離ではない。後者は将来の grammar 拡張で別途必要になる。

### 20.1 時間 sweep（$K_{\max}=512$, $A_{\max}=64$, seeds 1--3）

| $T$ | total novelty | late novelty | late birth | remaining catalog | late $A_t$ |
|---:|---:|---:|---:|---:|---:|
| 30 | 6 | 0 | 15 | 0 | 288.7 |
| 100 | 6 | 0 | 50 | 0 | 289.5 |
| 300 | 6 | 0 | 150 | 0 | 288.7 |
| 1000 | 6 | 0 | 500 | 0 | 288.8 |

長時間でも birth は止まらないが、これは **reinnovation cycle** である。6 個の有限 catalog を早期に
使い切った後、late semantic novelty は常に 0 であり、late advantage の傾きも実質 0 になる。

### 20.2 capacity sweep（$T=300$, $A_{\max}=64$）

| $K_{\max}$ | reachable max arity | total novelty | remaining | max $L(D)/K_{\max}$ |
|---:|---:|---:|---:|---:|
| 64 | 8 | 3 | 3 | 0.81 |
| 128 | 16 | 4 | 2 | 0.70 |
| 256 | 32 | 5 | 1 | 0.59 |
| 512 | 64 | 6 | 0 | 0.51 |

capacity を増やすと reachable frontier は伸びるが、どの条件も $L(D)\rightarrow K_{\max}$ の memory bloat
ではない。むしろ frontier は有限候補集合の最後まで到達し、その後に plateau する。catalog sweep
($A_{\max}=8,16,32,64$; $K_{\max}=512$) でも各条件は全候補を消費し、late novelty は 0 だった。

したがって exp605 の判定は

$$
\boxed{\text{bounded catalog plateau, not an open-endedness candidate}}
$$

である。この結果は失敗ではなく、exp604 の循環を「無限の意味新規性」と取り違えないための境界条件を
確定した。open-endedness 候補へ進むには、arity catalog の拡張だけでなく、有限リスト外の新しい semantic
operator family と、その operator が将来 task を短縮することを監査する必要がある。

## 21. exp606 -- Generative Semantic Space（生成的意味空間）

exp605 の finite-catalog plateau を受け、exp606 は候補 `O1...O64` を task source として使わない。
環境は可変長 semantic program

$$
f_d(x)=\operatorname{repeat}^{d}(\operatorname{map/reverse}(x),2)
$$

を $d=1,2,\ldots$ と生成する。$d$ に上限はなく、task space は無限である。semantic equality は構文木 ID
でなく、登録 probe 上の出力を正規形 `(repeat factor, reverse, flip)` に畳み込んで比較する。異なる factor は
nonempty probe の出力長を変え、同 factor の map/reverse 変更も少なくとも一つの probe で異なるため、
$d_{\rm sem}=1$ はこの DSL 内で exact である。指数長の出力を materialize せずに $d=3000$ まで比較できる。

adaptive は `ITERATE_REPEAT2(d)` を一度だけ発明できる。定義費 20 bit と future reuse を含む MDL が
raw repeat tree より短いときだけ保持する。fixed-language は同じ generator から proposal を受けるが、raw tree
cost が manageable threshold を越えると新しい意味を受け取れない。よって

$$
P_t\rightarrow\text{manageable semantic neighborhood}\rightarrow E_{t+1}
$$

を同一 generator と fixed-language control で検査できる。

### 21.1 時間 sweep（$K_{\max}=64$, seeds 1--3）

| $T$ | adaptive late novelty rate | adaptive frontier | fixed frontier | fixed late novelty |
|---:|---:|---:|---:|---:|
| 100 | 1.000 | 100 | 6 | 0 |
| 300 | 1.000 | 300 | 6 | 0 |
| 1000 | 1.000 | 1000 | 6 | 0 |
| 3000 | 1.000 | 3000 | 6 | 0 |

adaptive は新 instruction を birth=1、retain=1、reuse=2999 回（$T=3000$）とし、late semantic novelty を
持続する。capacity sweep では 20-bit 定義を保持できない $K=16$ は fixed と同じ depth 6 で止まり、
$K=32,64,128$ は同じ mechanism で depth 1000 に到達する。最大言語サイズは 20 bit で、$K=64$ では
$L(D)/K=0.31$ であり capacity 飽和ではない。

事前登録 4 条件（late novelty、retain/reuse、adaptive frontier expansion、fixed の早期 plateau）と、
$r_{\rm nov}(T)>0.25$、capacity 非飽和はすべて真となった。

$$
\boxed{\text{generative semantic-space candidate (one-dimensional operator family)}}
$$

これは open-ended evolution の証明ではない。無限性は repeat depth という**一次元 parametric family**に由来し、
発明される generic instruction も一つである。次段階では `concat`、`permute`、`map` 等の異種 operator family を
複数生成し、probe distance と将来 MDL 利得を保ったまま新しい operator 自体が継続的に発明されるかを監査する必要がある。

## 22. exp607 -- Multidimensional Semantic Innovation（多次元意味新規性）

exp607 は exp606 の `Repeat^d` 一軸を、environment-only grammar

$$
S ::= x\mid\operatorname{Repeat}(S,2)\mid\operatorname{Reverse}(S)
\mid\operatorname{MapFlip}(S)\mid\operatorname{Interleave}(S,S)
$$

へ拡張する。個体はこれらを instruction として初期保有しない。raw tree spelling から始め、future reuse を含む
MDL が definition cost を回収するときだけ `ITERATE`、`REVERSE`、`INTERLEAVE`、`MAP_FLIP` を発明・保持できる。

environment generator は二条件で比較する。

| generator | 生成する新規性 |
|---|---|
| `parametric_only` | `ITERATE` の depth だけを増やす |
| `compositional_generative` | repeat depth とともに reverse/interleave/map を再帰 DAG へ入れ子に追加する |

semantic novelty は probe-equivalent canonical transducer summary が未観測かで数える。さらに DAG operator sequence が
未観測で、`ITERATE` 単独ではないものだけを structural novelty とした。したがって depth だけの増加は
$N_{\rm parametric}$、新しい composition DAG は $N_{\rm structural}$ に分離される。operator class は opcode 名でなく
probe 上の behavior `(length-scale, permutation, merge, value-map)` を cluster key として数える。

### 22.1 長時間対照（$T=3000$, seeds 1--3）

| generator / condition | late structural novelty | late parametric novelty | operator classes | frontier |
|---|---:|---:|---:|---:|
| parametric-only / adaptive | 0 | 1500 | 1 | 3000 |
| compositional-generative / adaptive | **1500** | 0 | **4** | 3000 |
| compositional-generative / fixed | 0 | 0 | 0 | 7 |
| compositional-generative / no-retention | 0 | 0 | 4 (unretained) | 7 |
| compositional-generative / keep-all | 1500 | 0 | 4 | 3000 |

adaptive compositional condition では新しい operator classes が4件 birth・retain され、総 reuse は 11990 回、
$L(D)=70 < 0.95K_{\max}$ である。parametric-only が structural novelty 0 のままなのに対し、
compositional generator で $r_{\rm structural}^{late}=1.0$ が $T=100,300,1000,3000$ を通じて維持された。
fixed と no-retention は early raw-tree frontier で停止した。

$$
\boxed{\text{open-ended semantic innovation candidate within a finite generator grammar}}
$$

これは強い open-ended evolution の証明ではない。environment grammar は有限で、operator classes も4で早期に飽和する。
late structural novelty は、その有限 class を使った無限深さの composition DAG に由来する。次の天井は generator grammar
自体を可変にし、新しい operator **class** が late phase にも増え続けるかを、同じ probe/MDL/retention controls で監査することである。

## 23. exp608 -- Operator-Class Birth（operator class の出生）

exp608 は有限の opcode catalog を数える代わりに、primitive affine actions から parent-pair synthesis
`q_i,q_j -> q_k` を生成する。class key は operator 名ではなく probes `(-2,-1,0,1,2)` に対する action
signature である。新 class は、(i) 未観測 signature、(ii) future MDL で definition cost を回収可能、
(iii) retained 後に少なくとも2回 descendant の親として再利用、(iv) opcode knockout が task description
を raw spelling へ悪化、の全条件を満たした時点でのみ confirmed functional birth とする。

`generative + adaptive` は $T=100$ で 97 confirmed classes（後半47、parent reuse 197）を得る一方、
`fixed-environment + adaptive` は0、`generative + fixed-individual-language` も0で raw frontier 7 に停止する。
lineage は例えば `q_1,q_2 -> q_3`、つづいて `q_3,q_1 -> q_4` と記録される。

$$
\boxed{\text{operator-class birth candidate under a fixed synthesis primitive}}
$$

これは無制限の semantic evolution の証明ではない。生成子は依然として固定の二項 affine-synthesis primitive
であり、検出も有限 probe suite 上の行動等価性である。したがって次段階では、異種の synthesis primitive と probe
拡張に対して class birth が維持されるかを監査する必要がある。

## 24. exp609 -- Meta-Operator Evolution（meta-rule の進化）

exp609 は operator $q_i:X\to X$ と、その二つの親から child operator を作る meta-rule $R_j$ を分離する。
meta-rule は完成した `ADD` や `COMPOSE` の名前ではなく、`LEFT`、`RIGHT`、`DUP`、`ADD` の stack bytecode
で記述する。環境は前半に $R_1=[L,R,+]$、後半の一部 demand に $R_2=[L,R,\mathrm{DUP},+,+]$ を要求する。

`fixed_meta_rule` は $R_1$ だけを人為的に保有する。`adaptive_meta_rule` は raw bytecode spelling の反復が
definition cost を回収する場合だけ rule を発明・保持する。meta-rule class は pair probes 上の挙動で数え、functional
birth には semantic novelty、MDL learnability、2回以上の生成再利用、rule knockout 時の追加記述長を全て要求する。

この設計は「operator が増えた」だけでは合格にしない。adaptive が fixed より operator class / frontier を拡張し、
かつ後半に functional $R_2$ が birth することを事前登録した。これは小規模な二層検証であり、stack bytecode
meta-language 自体は依然として固定されている。exp610 では operator と generator を同一の universal program
representation に統合して、この特別な階層を除去する必要がある。

### 24.1 結果（seeds 1--3）

| $T$ | fixed operator classes | adaptive operator classes | adaptive late functional meta-rules |
|---:|---:|---:|---:|
| 100 | 75 | **100** | **1** |
| 300 | 225 | **300** | **1** |
| 1000 | 750 | **1000** | **1** |

fixed は supplied $R_1$ を使い続けるため operator は増えるが、後半 demand の $R_2$ を表せず、functional
meta-rule birth は0である。adaptive では $R_2$ が後半で birth し、25--250回再利用され、knockout の追加記述長は
正である。capacity fraction は $0.001$ 未満で、差は meta-language capacity の飽和では説明できない。

$$
\boxed{\text{meta-operator evolution candidate under a fixed universal-looking bytecode}} 
$$

## 25. exp610 -- Unified Program Ecology（単一 Program 表現）

exp610 は exp609 の `Operator` と `MetaRule` を一つの first-class `Program` AST に統合する。唯一の typed
interpreter は整数に適用されると data operator、Program に適用されると program transformer になる。`ADD` は data
では加算、Program では AST composition を実行する。したがって role はクラス名でなく application の入力型から生じる。

primitive `ARG0` と `ARG1` に generic crossover をかけると $R_1=\operatorname{ADD}(ARG0,ARG1)$、generic subtree
duplication mutation をかけると $R_2=\operatorname{ADD}(ARG0,\operatorname{ADD}(ARG1,ARG1))$ になる。同じ evaluator
で $R_i(q_a,q_b)=q_c$ と $q_c(x)$ を実行できるので、$P_i(P_j)=P_k$ は専用 meta-program 型を必要としない。

`specialized_two_layer` 対照は外部 rule で同じ task を生成するが、first-class transformer invention は数えない。
`unified_program_ecology` は transformer の behavior novelty、MDL benefit、2回以上の reuse、knockout 追加記述長を
同時に満たす時だけ functional transformer class とする。semantic class、transformer class、そして Program DAG の
最大 ancestry depth を併記し、単なる random rewrite を invention と数えない。

これは universal computation や無制限の self-modification の証明ではない。instruction set と typed dispatch は固定で、
環境も $R_1/R_2$ を必要とするよう設計されている。結論は、exp608/609 の二層現象を**専用の二層データ型なし**に再現できる、
という限定されたものに留まる。

### 25.1 結果（seeds 1--3）

| $T$ | specialized semantic classes | unified semantic classes | unified late transformer classes | max ancestry depth |
|---:|---:|---:|---:|---:|
| 100 | 100 | **100** | **1** | 100 |
| 300 | 300 | **300** | **1** | 300 |
| 1000 | 1000 | **1000** | **1** | 1000 |

unified は specialized の operator-level phenomenon を落とさずに再現した。後半 transformer $R_2$ は
25/75/250 回再利用され、knockout 時の追加記述長は正である。specialized は同じ environment を外部 rule で
解けるが、first-class transformer birth は0である。capacity fraction は $0.001$ 未満である。

$$
\boxed{\text{unified-program ecology candidate: roles arise from application, not data types}}
$$

## 26. exp611 -- Transformer Diversity / Self-Modification Audit

exp611 は exp610 の統一 VM を変更せず、Program-to-Program transformer の**意味的 class 数**を直接監査する。
environment は正係数対 $(a,b)$ を対角列挙し、$T_{a,b}(P,Q)=aP+bQ$ の transformation demand を3回ずつ提示する。
候補は完成済み transformer catalog から取らず、既存の generic crossover と subtree-duplication mutation で構成する。
最初の二つは $T_{1,1}$ と $T_{1,2}=\operatorname{mutate}(T_{1,1})$ である。

class key は Program probes $(Q_0,Q_1),(Q_1,Q_0),(Q_0,Q_0)$ に対する child Program の data semantics である。
functional transformer は behavior novelty、MDL benefit、少なくとも2回の再利用、knockout 追加記述長、生成 child の
$\Delta L>0$ と task $\Delta F>0$ を全て満たす必要がある。raw child program 数は主指標にしない。

5対照は `unified_adaptive`、`self_application_off`、`transformer_retention_off`、`fixed_transformer`、
`random_program_rewrite` である。random rewrite は raw program diversity を作れるが、ancestry depth は0とし、
functional transformer として数えない。

### 26.1 結果（seeds 1--3）

| $T$ | adaptive functional transformer classes | late functional transformer classes |
|---:|---:|---:|
| 100 | **33** | **16** |
| 300 | **100** | **50** |
| 1000 | **333** | **166** |

4つの mechanism control は functional transformer classes 0、random rewrite は $T=1000$ で raw programs 1000でも
functional transformer 0である。これは self-modification depth や raw rewrite 数ではなく、再利用され MDL を改善する
transformer の behavior diversity が増えたことを示す。

$$
\boxed{\text{transformer semantic-diversity candidate within a fixed additive VM}}
$$

ただし多様性は固定された `ADD` language 内の係数対 family に限られる。したがって異種 transformation semantics
（swap、wrap、factor 等）が同じ VM から自発的に維持されることは、まだ示していない。
