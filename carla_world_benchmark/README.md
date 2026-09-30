# carla_world_benchmark — CARLA 上での意識モデル検証環境

`Consciousness_model` の有限ワークスペースを、運転という連続制御タスクの
**危険候補選択層**として接続し、容量 K を独立変数にして運転成績を比較する環境。

`coding_world_benchmark` / Minecraft 系と同じ方針で、
「意識モデルが goal・注意・制動を出し、実行器（`BasicAgent`）は経路追従だけを担う」
という境界にしてある。

---

## 1. 実行経路

CARLA の配布物は 2026-09-05 時点で **nightly (Backblaze B2) しか生きていない**。
Leaderboard 2.0 パッケージは `AllAccessDisabled`、0.9.15 の CDN は 403（§2）。
その nightly を Windows 版として**すでに展開済み**なので、経路は 3 つある。

| | **A. Windows + 公式 leaderboard** | **B. Windows + 自作 runner** | **C. WSL2 + 公式 leaderboard** |
| --- | --- | --- | --- |
| CARLA | `CARLA_Latest/`（展開済み） | 同左 | Linux nightly（要 DL、数十 GB） |
| 追加 DL | AdditionalMaps のみ | 不要 | CARLA 本体 + AdditionalMaps |
| Python | 3.10（PyPI の cp310 wheel） | 3.12（同梱 wheel） | 3.10 |
| マップ | Town12 / Town13 | Town01–10HD（+追加可） | Town12 / Town13 |
| 評価 | **公式 leaderboard スコア** | 同形式の近似スコア | 公式 leaderboard スコア |
| 位置づけ | **推奨** | 依存リスクが最小 | A が動かない場合 |

経路 A が推奨。**WSL も CARLA の再ダウンロードも要らない** — 足りないのは Town12/Town13 が
入っている AdditionalMaps だけで、それは既に持っている zip と同じ nightly ホストから取れる。
`carla==0.9.16` は PyPI に **cp310 の win_amd64 wheel** があるので、
同梱の cp312 wheel とは別に Python 3.10 環境を作れば公式評価器が動く。

3 経路とも同じ agent コード・同じ設定・同じ集計スクリプトを使う。

---

## 2. 配布物の現状（2026-09-05 実測）

| 配布元 | 状態 |
| --- | --- |
| Leaderboard 2.0 パッケージ (S3 us-west-2) | **403 `AllAccessDisabled`** — オブジェクト単位で停止。リトライ不可 |
| CARLA 0.9.15 / AdditionalMaps (tiny.carla.org → b-cdn.net) | **403** |
| **nightly (Backblaze B2)** | **生きている**。公式 `Docs/download.md` が指す唯一のホスト |

SourceForge のミラーはソースのみでビルド済みパッケージは無い。
**Town12 / Town13 は本体ではなく AdditionalMaps に入っている。**

到達性はいつでも確認できる（ダウンロードしない）:

```bash
bash scripts/wsl/03_check_download.sh
```

### Python は CARLA のバージョンに従う

| CARLA | client wheel | Python | 依存セット |
| --- | --- | --- | --- |
| 0.9.14（leaderboard パッケージ） | cp37 / cp38 | 3.8 | upstream の pin をそのまま |
| 0.9.16（nightly） | cp310 / cp311 / cp312 | 3.10 | `requirements-leaderboard-py310.txt` |

どちらも実測済み（§5）。

### leaderboard に必要なパッチ

`ElementTree.Element.getchildren()` は **Python 3.9 で削除**されていて、
leaderboard-2.0 / scenario_runner-2.0 の 3 ファイルがまだ呼んでいる。
放置するとどのルートファイルも解析できない。`list(elem)` は 3.7 でも同じ挙動なので
どの経路でも安全。`apply_patches.sh` / `apply_patches.ps1` が適用し、
セットアップスクリプトが自動で呼ぶ。

---

## 3. セットアップ

### 経路 A: Windows + 公式 leaderboard（推奨）

Python は自分で入れなくてよい。CARLA のクライアントは特定の CPython 版にしか
wheel が無く（同梱 wheel は 3.12、leaderboard 用は 3.10）、システムの Python が
それと一致することはまず無いので、セットアップは **`uv` に該当バージョンを
取ってこさせる**。`uv` が無ければ自動で入れる。python.org から手で入れる必要は無い。

```powershell
# 1) Town12 / Town13 を追加（唯一必要なダウンロード）
powershell -ExecutionPolicy Bypass -File .\scripts\get_additional_maps.ps1

# 2) leaderboard 用の Python 3.10 環境 + パッチ（CARLA は展開済みなので -SkipExtract）
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -SkipExtract -Leaderboard

# 3) サーバ起動（このシェルは開いたまま）
. .\scripts\env.ps1
.\scripts\start_carla.ps1
```

```powershell
# 4) 別シェルで確認 → 実行
. .\scripts\env.ps1 -Leaderboard
python .erifyerify_setup.py                     # maps に Town12/Town13 が出ること
.\scripts
un_leaderboard.ps1 -AgentConfig agent\configs\k4_workspace.json
.\scripts
un_ablation.ps1
```

### 経路 B: Windows + 自作 runner

依存が numpy / networkx / shapely だけなので最も壊れにくい。Town12/13 が無くても走る。
こちらも Python 3.12 は `uv` が用意する。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -SkipExtract
. .\scripts\env.ps1
.\scripts\start_carla.ps1                           # 別シェル
.\scripts
un_experiment.ps1 -Town Town10HD -Seeds 3 -NoRender
```

### 経路 C: WSL2

A が動かないときだけ。CARLA 本体から落とし直すので数十 GB かかる。

```powershell
powershell -ExecutionPolicy Bypass -File .\scriptsootstrap_wsl.ps1
```

```bash
bash scripts/wsl/00_check.sh          # WSL2 / GPU / vulkan / 空き容量
bash scripts/wsl/03_check_download.sh # 落とせるホストの確認
bash scripts/wsl/01_setup.sh --carla=nightly
bash scripts/wsl/02_check_binary.sh   # 解決できない .so が無いか（01 が自動で呼ぶ）
source scripts/wsl/env.sh
bash scripts/wsl/start_carla.sh       # 別シェル
```

CARLA と venv は WSL の ext4 側（`$HOME`）に置く。`/mnt/c` は実用にならないほど遅い。
`nvidia-smi` は通るのに `vulkaninfo` が落ちるのが WSL の定番で、
`bash scripts/wsl/01_setup.sh --write-vulkan-icd` で ICD が入る。

---

## 4. 実験設計

独立変数は「危険候補をいくつ、どう選ぶか」だけ。制御器・ルート・seed は固定する。

| config | selection | 意味 |
| --- | --- | --- |
| `k4_workspace` | workspace K=4 | 本命。salience 上位 4 件だけが制動に影響できる |
| `k2_workspace` | workspace K=2 | 中間点 |
| `k1_workspace` | workspace K=1 | 容量を絞った場合 |
| `k4_random` | random K=4 | **容量は同じで順位付けだけ壊す統制**。ここと差が出なければ「価値順に選ぶこと」自体には意味がない |
| `unbounded` | all | 容量制限なし（上限） |
| `blind` | none | 危険を一切見ない（下限） |

構造上の担保として、**採択されなかった危険は制動に一切影響できない**
（`_select` の外に出た候補は decision に入らない）。仮説を規約ではなく構造で強制している。

`BasicAgent` は `ignore_vehicles / ignore_traffic_lights / ignore_stop_signs` を立てて
経路追従専用にしてある。切らないと標準制御が危険回避を肩代わりして K の効果が消える。

もう 1 本、outcome→制御のフィードバックが `risk`:
前サイクルで**棄却した**危険が今サイクル critical になったら「容量起因の見落とし」として計上し、
その率で risk を上げる。risk は salience を広げ、同時に目標速度を下げる。

結果は `results/` に設定ごと 2 ファイル:
`<tag>.json`（1 ステップごとの採択・棄却・memory/attention・risk・goal）と
`<tag>.leaderboard.json`（スコア）。集計は `python verify/analyze_ablation.py`。

---

## 5. すでに検証できていること

**CARLA サーバなしで再現可能**（42 件 OK）:

```bash
python verify/test_bridge.py -v      # 選択・制動・較正 24 件
python verify/test_scoring.py        # スコア計算 12 件
python verify/test_analysis.py       # 集計 6 件
python verify/simulate_ablation.py   # 1 次元トイ回廊での配線確認
```

`simulate_ablation.py`（40 seed × 14 actor, 40 秒）:

| config | collisions/ep | distance (m) | final risk |
| --- | --- | --- | --- |
| k4_workspace | 0.07 | 75.9 | 0.075 |
| k2_workspace | 0.10 | 81.4 | 0.125 |
| k1_workspace | 0.20 | 84.0 | 0.192 |
| k4_random | 0.38 | 68.1 | 0.216 |
| unbounded | 0.00 | 65.0 | 0.000 |
| blind | 2.23 | 166.8 | 0.684 |

読み取れるのは「K が behaviour まで届いていること」と
「同じ容量でも順位付けを壊すと悪化すること（0.07 → 0.38）」の 2 点だけ。
**配線の確認であって運転の結果ではない。**

**Linux / Python 3.10 + CARLA 0.9.16 + 緩めた依存で実測したこと**（現在の既定）:

- `carla==0.9.16` + shapely 2.1.2 / networkx 3.4.2 / numpy 2.2.6 / py-trees 0.8.3 が入る。
- `srunner`（carla_data_provider, scenario_manager, route_parser, openscenario_parser,
  atomic_criteria, atomic_behaviors）と `leaderboard`（evaluator, agent_wrapper,
  statistics_manager, route_scenario）が全部 import できる。
- `agent/consciousness_agent.py` が setup → sensors → run_step → destroy まで動き、
  **leaderboard 自身の `validate_sensor_configuration` が valid と判定**。
- パッチ後、**3 つのルートファイルが全部解析できる**:
  devtest 2 ルート / 119 シナリオ（Town12）、training 90 / 4629（Town12）、
  validation 20 / 1786（Town13）。
- 未パッチだと `Element.getchildren()` で全滅する（Python 3.9 で削除された API）。
- `setuptools>=81` だと `leaderboard_evaluator` が `pkg_resources` を見つけられない
  → `setuptools<81` に固定。

**Linux / Python 3.8 + CARLA 0.9.14 でも実測済み**（leaderboard パッケージが復活した場合）:

- `carla==0.9.14` + `scenario_runner/requirements.txt` + `leaderboard/requirements.txt` が
  3.8.20 に**そのまま全部入る**（numpy 1.18.4 / opencv 4.2.0.32 / Shapely 1.7.1 も cp38 wheel あり）。
- `import carla` / `srunner` / `leaderboard.leaderboard_evaluator` / `agents.navigation.basic_agent` が通る。
- `agent/consciousness_agent.py` が import でき、`get_entry_point()` → `setup()` → `sensors()` →
  `run_step()` → `destroy()` まで動く。
- **leaderboard 自身の `validate_sensor_configuration` がこのセンサ構成を valid と判定する。**
- leaderboard / scenario_runner の 122 ファイルに Python 3.8 で動かない構文は 0 件。
- `verify/verify_setup.py --offline` が required / optional とも全 PASS。

**Windows 経路のコード検証**: 同梱 wheel の型スタブ（`libcarla.pyi` / `command.pyi`）に対して
`runner/carla_runner.py` が呼ぶ CARLA API を全件照合済み
（Client / World / TrafficManager / WalkerAIController / command 系）。

**未確認**: GPU が要る部分すべて。CARLA サーバの起動、実際の走行、
WSL2 の Vulkan パススルー、Town12/13 のロード。

---

## 6. うまくいかないとき

### WSL2（経路 A）

- **`nvidia-smi` が無い**: Windows 側の NVIDIA ドライバを更新する。
  WSL 内に `nvidia-driver-*` を apt で入れてはいけない（パススルーが壊れる）。
- **`vulkaninfo` が落ちる**: `bash scripts/wsl/01_setup.sh --write-vulkan-icd`。
- **CARLA が起動直後に落ちる**: ほぼ Vulkan。`00_check.sh` からやり直す。
- **極端に遅い**: CARLA か venv が `/mnt/c` にある。`config.sh` の
  `CARLA_INSTALL_ROOT` / `VENV_ROOT` を `$HOME` 配下にする。
- **ダウンロードが途中で切れる**: `01_setup.sh` を再実行すれば `curl -C -` で再開する。
- **`xz: File format not recognized` / `tar: Child returned status 1`**:
  ダウンロードされたのがアーカイブではなくエラー応答。
  `bash scripts/wsl/03_check_download.sh` で各ホストのステータスと本文を確認し、
  `--carla=release0915` に切り替える。壊れた小さいファイルは次回実行時に自動で捨てられる。
- **サーバが無言で落ちる / 起動しない**: `bash scripts/wsl/02_check_binary.sh`。
  未解決の `.so` を並べて対処法まで出す。24.04 で多いのは `libtiff.so.5`
  （24.04 は `libtiff6` しか無い）で、`/usr/lib/x86_64-linux-gnu` に
  `sudo ln -s libtiff.so.6 libtiff.so.5` で通る。
- **apt のパッケージ名**: 24.04 は 64bit time_t 移行で `libpng16-16` →
  `libpng16-16t64`、`libtiff5` → `libtiff6` に改名されている。
  `01_setup.sh` は候補名を順に試すので、リリースが変わっても止まらない。
- **`python` が venv のものにならない**: `env.sh` が警告を出す。anaconda を
  auto-activate していると PATH で勝つことがある。`conda deactivate` してから
  `source scripts/wsl/env.sh` するか、`$VENV_ROOT/bin/python` を直接呼ぶ。
  `01_setup.sh` はダウンロードと展開に `/usr/bin/curl` と `/usr/bin/tar` を
  優先的に使うので、anaconda 版が使われることはない。

### Windows（経路 B）

- **`python` がシステムの 3.13 などを指す**: CARLA のクライアントは 3.13 用の wheel が無い。
  `. .\scripts\env.ps1` は venv の python に解決されているかを検査して警告する。
  警告が出たら venv がまだ作られていない（`setup_windows.ps1` を実行する）か、
  PATH で別の Python が勝っている。
- **venv が作られない**: `setup_windows.ps1` は `uv` で CPython を取ってくる。
  ネットワークが塞がれていると `uv` の導入に失敗するので、その場合だけ
  python.org から該当バージョン（3.12 と 3.10）を手で入れれば `py` ランチャー経由で動く。
- **`import carla` が失敗**: `. .\scripts\env.ps1` を実行したか、venv が有効か。

### 共通

- **車が動かない**: 制動はすべて意識モデル側（`BasicAgent` の判断は切ってある）。
  `results/<tag>.json` の `admitted` / `min_ttc_admitted` / `goal` を見る。
  `blind` 以外で止まりっぱなしなら salience か閾値の問題。
- **重い**: `-NoRender` / `start_carla.sh`（既定で `-RenderOffScreen`）/
  quality を `Low` / 交通量を減らす。

---

## 7. ファイル構成

```
carla_world_benchmark/
  agent/
    carla_consciousness_bridge.py   CARLA非依存。危険→salience→有限workspace→運転判断
    carla_perception.py             CARLA world → 記号的危険（特権知覚）。両経路で共用
    consciousness_agent.py          leaderboard 用 AutonomousAgent 実装
    configs/*.json                  アブレーション設定
  runner/
    carla_runner.py                 スタンドアロン実験系（leaderboard 不要）
    scoring.py                      スコア計算（CARLA非依存・テスト済み）
  scripts/
    wsl/                            経路 A: config.sh lib_download.sh
                                            00_check.sh 01_setup.sh
                                            02_check_binary.sh 03_check_download.sh env.sh
                                            start_carla.sh run_leaderboard.sh run_ablation.sh
    apply_patches.sh                external/ を Python 3.9+ で動くようにする
    get_additional_maps.ps1         経路 B に Town12/13 を追加（Windows）
    bootstrap_wsl.ps1               Windows から WSL 側の事前確認を起動（-SelfTest あり）
    config.ps1 env.ps1 setup_windows.ps1 start_carla.ps1   経路 B
    run_experiment.ps1              経路 B のアブレーション一括実行
    run_leaderboard.ps1 run_ablation.ps1                   Windows から leaderboard を叩く場合
    requirements-standalone.txt requirements-leaderboard-py312.txt
  verify/
    verify_setup.py                 事前チェック（required / optional を分離、Win/Linux 両対応）
    test_bridge.py test_scoring.py test_analysis.py   オフライン単体テスト
    simulate_ablation.py            CARLA無しの配線確認
    analyze_ablation.py             結果集計
  CARLA_Latest/                     展開済み Windows パッケージ（git管理外）
  external/                         leaderboard / scenario_runner（git管理外）
  results/                          実行結果（git管理外）
```
