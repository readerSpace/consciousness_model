# consciousness_io — 入出力を設定できる意識モデル

`consciousness_model` を土台に、**意識状態を1本のベクトルではなく集合として定式化し、
センサ入力（観測）と行動出力（世界への作用）を利用者が差し替え可能に設定できる**ように
したパッケージです。既存の基盤モジュール（有限/圧縮ワークスペース、現象的状態、
アプリ接続ブリッジ、物理法則発見）はそのまま同梱・再利用しています。

> 本パッケージは機能的メカニズムのみを提供します。主観的経験を主張するものではありません。

## 集合論的な意識状態

時刻 `t` の意識状態を、単一ベクトルではなく次の集合の組として定義します。

```
C_t = (O_t, B_t, P_t, G_t, Q_t, A_t, W_t)
C_{t+1} = F(C_t, O_{t+1}, a_t)
```

| 記号 | 意味 | このコードでの実体 |
|------|------|---------------------|
| `O_t` | 観測集合（カメラ・音声・距離・関節角…） | `SensorChannel.encode` が生成する `Fact` 群 |
| `Z_t` | 圧縮状態 `argmin_Z [L(Z)+λL(O｜Z)]` | `CompressedWorkspace`（MDL的圧縮）の概念集合 |
| `B_t` | 世界について成立していると考える命題集合 | `BeliefStore`（`B_{t+1}=B_t∪f(O_t)`、キーで上書き） |
| `P_t` | 未来予測 `P(a,B_t)` | 各行動の `PredictedEffect` と目標距離 |
| `G_t` | 目的集合 | 目標 `Fact` 群（実行時に差し替え可） |
| `Q_t` | 分からないことの集合 `{x｜H(O_{t+1}｜Z_t,x)>θ}` | 予測不確実性が θ を超える `Question` |
| `A_t` | 現在可能な行動集合 | `ActionSchema` を束縛した `ActionBinding` 群 |
| `W_t` | 有限ワークスペース（いま意識に上っている情報） | `G_t∪Q_t` で焦点化した上位 `capacity` 件 |

循環 `F`（`ConsciousAgent.step` 1回）は設計ノートの流れをそのまま実装しています。

```
O_t --compression--> Z_t --prediction--> P_t
Z_t, P_t --mismatch--> Q_t
Z_t, Q_t, G_t --attention--> W_t
W_t --action selection--> a_t
a_t --world--> O_{t+1}
```

**操作的定義**: 意識 = 予測・目的・疑問・行動に必要な、いま参照可能な有限情報集合（＝ `W_t`）。

## 入出力の設定方法

このパッケージの主眼は、**何を感じ・何ができるかを利用者が宣言的に与える**ことです。

### 入力ポート `SensorChannel`

生センサ値 → 観測 `Fact` への変換器を登録します。エージェントは生値の中身を一切見ません。

```python
from consciousness_io import SensorChannel, fact

def vision_to_facts(frame):
    yield fact("ON", "cup", "table")
    yield fact("AT", "cup", tuple(frame["cup_xyz"]))

camera = SensorChannel("camera", encode=vision_to_facts)
range_sensor = SensorChannel("lidar", encode=lidar_to_facts, optional=True)
```

各ステップの生入力は「チャンネル名 → 生値」の辞書で渡します: `agent.step({"camera": frame, "lidar": scan})`。

### 出力ポート `ActionSchema`

行動テンプレートに、コスト・**予測される信念変化**（`effect`）・任意の**実行器**（`execute`）を与えます。

```python
from consciousness_io import ActionSchema, PredictedEffect, fact

def grasp_effect(beliefs, binding):
    reachable = any(f.predicate == "REACHABLE" for f in beliefs)
    if reachable:
        return PredictedEffect(added=(fact("HAVE", "robot", "cup"),), confidence=0.95)
    return PredictedEffect(added=(), confidence=0.5)   # 低信頼 → Q_t の材料

grasp = ActionSchema("GRASP", effect=grasp_effect, cost=1.5,
                     execute=lambda binding: robot.grasp())   # 省略可
```

`effect` があることで行動が「計画可能」になります。目標駆動選択は `effect` を目標と比較し、
好奇心駆動選択は `effect.confidence`（＝不確実性 `1-conf`）を情報信号として読みます。

### ループの設定 `AgentConfig`

```python
from consciousness_io import AgentConfig

cfg = AgentConfig(
    workspace_capacity=7,     # |W_t| の上限
    compressed_capacity=16,   # Z_t を作る圧縮WSの容量
    question_threshold=0.5,   # θ: これを超える不確実性は疑問になる
    curiosity_weight=0.3,     # λ: I(Q;O｜a) - λ·Cost(a)
    policy="auto",            # "auto" | "goal" | "curiosity"
)
```

## 行動選択

- **目標駆動**: `a* = argmin_a D(P(a,B_t), G_t)`（未達成の目標命題数で距離を測る、1手先読み）
- **好奇心駆動**: `a* = argmax_a [ I(Q_t; O_{t+1}｜a) − λ·Cost(a) ]`
- **auto**: 目標距離を縮める行動があればそれを実行。無ければ疑問を最も減らす行動を選ぶ。目標充足時は NOOP。

## 使い方

```python
from consciousness_io import ConsciousAgent, AgentConfig, fact

agent = ConsciousAgent(
    sensors=[camera],
    actions=[reach, grasp],
    goals=[fact("HAVE", "robot", "cup")],
    config=AgentConfig(policy="auto"),
)

state = agent.step({"camera": frame})
print(state.summary())          # O_t / Z_t / B_t / Q_t / W_t と選ばれた a_t を表示
print(state.decision.label)     # 例: "GRASP"
```

## サンプル

```
python examples/cup_on_table.py         # 目標駆動: 机の上のコップを取る（REACH→GRASP）
python examples/red_object_curiosity.py # 好奇心駆動: 赤い物体は動かせるか（PUSH を選ぶ）
```

## テスト

```
python -m pytest tests/
```

## パッケージ構成

```
consciousness_io/
  facts.py               Fact / BeliefStore / goal_distance（集合の要素と距離）
  io_config.py           SensorChannel / ActionSchema / PredictedEffect / AgentConfig（★I/O設定）
  agent.py               ConsciousAgent / ConsciousState（C_t と循環 F）
  compressed_workspace.py  CompressedWorkspace（MDL圧縮 → Z_t）※既存を同梱
  consciousness.py         FiniteWorkspace / PhenomenalState 等          ※既存を同梱
  consciousness_model.py   ConsciousnessController                        ※既存を同梱
  integration.py           ConsciousnessConnection                       ※既存を同梱
  core.py                  法則発見・数値積分                             ※既存を同梱
examples/                两サンプル
tests/                   pytest
```

## 設計ノートとの対応（発展）

設計ノートにある「集合の要素数を固定しない」＝ **予測誤差→状態の誕生 / 冗長性→状態の統合・消滅**
は、同梱の `CompressedWorkspace` が merge / abstract / chunk（圧縮）と効用ベースの選択（冗長性による淘汰）
として既に持っています。新しい述語が観測に現れれば `Z_t` に新概念が自発的に生まれ、
冗長な概念は選択段階で落ちます。まずは「最初は `位置` しか概念を持たないロボットが、
物体操作の予測誤差から `物体`・`可動性`・`障害物`・`容器` を自発的に作れるか」を、この I/O 設定の上で
検証していくのが自然な次の一歩です。
