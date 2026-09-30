# Test matrix

実行環境ごとに結果を分けて記録する。「環境要因のはず」は成功として数えない。
未確認の行は UNVERIFIED のままにし、実機で確認したときに更新する。

| 環境 | 日付 | 結果 | 備考 |
| --- | --- | --- | --- |
| local_windows (satoshi3) | 2026-09-14 | **272 passed / 18 failed / 13 errors** (run #1) | Python 3.13.2 / pytest 9.1.1。原因は下記 W1。修正後に再実行が必要 |
| container (Linux, Python 3.11) | 2026-09-14 | 317 passed / 1 environment_failure | 下記 E1 の1件のみ失敗 |
| desktop_linux_vm (satoshi3, Python 3.10) | 2026-09-19 | 1078 passed / 1 failed / 9 collection_errors | L8.12〜L8.42.1 ＋ UI/API 統合・契約検証・検証済みテンプレート・Python 風スクリプト入力・算術/物理言語・関係推論・定理発見システム統合後。`coding_agent_app.py` の tkinter import を任意にしたことで、tkinter 不在で収集できなかった 11 ファイル（+87 テスト）が動くようになった。残る収集エラー 9 件は `datetime.UTC`（3.11+）を 3.10 で読むもので tkinter とは別原因。残る失敗 1 件 `test_natural_language_explanation_auto_selects_embedded_workspace_locally` は `windows_path_candidates` が `X:\...` しか拾わない Windows 専用解析のためで、Linux では原理的に通らない（実機 Windows では別）。新規分は L8.12 11 / L8.13 13 / L8.14 12 / L8.15 11 / L8.15.1 7 / L8.16 13 / L8.17 12 / L8.18 13 / L8.19 12 / L8.20 12 / L8.21 13 / L8.22 12 / L8.23 12 / L8.24 11 / L8.25 12 / L8.26 12 / L8.27 17 / L8.28 17 / L8.29 21 / L8.30 19 / L8.31 17 / L8.32 18 / L8.32.2 9 / L8.33 20 / L8.34 15 / L8.35 17 / L8.36 16 / L8.37 16 / L8.38 15 / L8.39 24 / L8.40 16 / L8.41 16 / L8.42 22 / semantic_service 23 / semantic_ui 10 / agent_contract 15 / dialogue_script 41 / math_expression 48 / physics_script 45 / logic_reasoner 45 / theorem_service 25 passed。定理発見システムはサブプロセスで駆動し、`asserted`（篩）と `valid`（証明）を分けて報告する。バンドル自身のテストは `/theorem verify` 経由で **51 passed**（統合前は `CORPUS` が `parents[2]` でバンドル外を指しており 5 件落ちていた。1行修正済み）。関係推論は曖昧な記号（`in` `⊂`）の読みを選ばず全部運ぶ。契約検証がその両方を毎回確認する。物理言語のソルバは解析解（らせん）と毎回比較し、最大ずれ 1.67e-09 をテストで上限固定。検証済みテンプレート 14 件は契約検証が毎回すべて再生して `decision` と `commits` を照合する（panel の「検証済み」はラベルでなくその実行で観測された性質）。契約検証（`agent_contract_verification`）は Linux VM では自分で bridge を立てて **76/77**（唯一の NG は UI を編集した直後の bundle 鮮度）、以前の唯一の NG「bundle がソースより古い」は再起動で解消された＝検証器が正しく働いていた。PowerShell / Electron / 実行中プロセスの検証は Windows でしかできないので `verify_coding_agent.ps1 -BridgePort 8787` を用意した。この行は local_windows の代わりにならない（P1 と同じ理由で Linux）|

**local_windows は verified になった。そして 31 件落ちた。** これが UNVERIFIED を
放置しなかった理由そのもので、原因 W1 はコンテナでは構造的に再現しない種類の不具合だった。
run #1 の時点でテストファイルは 55 件。修正で増えているため
`environment_verification.py` は stale と判定する（= 再実行が必要）。正しい挙動。

この表の local_windows 行は手書きしない。`verify_local_windows.ps1` が出力した
`verification/local_windows_result.json` を `environment_verification.py` が読み、
**全層通過かつテストが実際に数えられた場合のみ** verified になる。
結果ファイルが無ければ何をどう書いても UNVERIFIED のままになる。

## 実機実行経路の切り分け（2026-09-14）

```text
1 シェル起動   : OK   device_bash は応答する
2 フォルダ可視 : OK   device_list_dir / device_stage_files / device_commit_files は動作
2'マウント     : NG   sandbox-helper: no Plan9 drive shares mounted
3 Python 起動  : 未確認
4 pytest 起動  : 未確認
5 スイート実行 : 未確認
```

自動実行できない理由は 2 つあり、いずれも修理対象のバグではなく経路の性質:

- **P1: `device_bash` は Linux VM である。** 接続フォルダをマウントして動く Linux 実行環境であり、
  マウントが復旧しても実行されるのは Linux の python。E1（ドライブレター依存）は同じく失敗する。
  つまり **device_bash を直しても `local_windows` の検証にはならない**。
- **P2: computer use はターミナルを click tier でしか付与しない。**
  `computer_resolve_access` が明示する通り、ターミナルと IDE は「見る・左クリックする」のみで
  type / key / paste ができない。したがって PowerShell へコマンドを打ち込めない。

結論として、現時点で Windows ネイティブ実行を自動で起動する経路は存在しない。
そこで「1 コマンドだけ人間が実行し、記録は機械が行う」形に分離した。

## 実機での確認コマンド

## 実機での確認コマンド

```powershell
cd C:\Users\neko5\Documents\projects\意識モデル
powershell -ExecutionPolicy Bypass -File .\coding_world_benchmark\verify_local_windows.ps1
```

`coding_world_benchmark\run_verification.cmd` をダブルクリックしても同じ。
層ごと（shell / workspace / python / pytest / suite）に判定し、
`coding_world_benchmark\verification\local_windows_result.json` と `.log` を書き出す。

取り込みと表示:

```powershell
python -m coding_world_benchmark.environment_verification
```

期待値は `318 passed`（E1 は Windows では通る）。run #1 で E1 は実際に pass した。
実測が異なる場合、結果ファイルの値がそのまま正であり、この表を実測値で更新する。

## 環境依存の既知差分

### E1: `test_natural_language_explanation_auto_selects_embedded_workspace_locally`

- 状態: container で **failed**、local_windows で **pass**（run #1 で確認済み）
- 機構: `LocalWorkspaceAgent._workspace_in_message` は
  `re.findall(r"[A-Za-z]:\\[^\"\r\n]+", message)` でメッセージ中の Windows 絶対パスを探す。
  Linux 上の `tmp_path` は `/tmp/...` 形式でドライブレターを持たないため候補が 0 件になり、
  workspace が設定されないまま assert に到達する。
- 再現性: 環境で決まる決定的な失敗であり、flaky ではない。
- 判定: container では **environment_failure** として扱い、pass にも fail にも寄せない。
  local_windows で pass することを確認するまで、この機能は「検証済み」ではない。

#### 未検証の表面積（2026-09-14 時点）

Windows 固有の挙動に依存するのは **`coding_agent_app.windows_path_candidates` を消費する
`_workspace_in_message` の存在確認 1 箇所のみ**（drive-letter regex はリポジトリ全体でここだけ）。
解析部分は `windows_path_candidates` として切り出し、プラットフォーム非依存に単体テスト済み。
したがって local_windows でしか確認できないのは実質
`Path("C:\\...").is_dir()` が期待どおり真になるか、の 1 点まで縮んでいる。

### W1: 子プロセス出力の locale デコード（cp932）— **実機のみで発生、31 件の原因**

- 状態: local_windows で **failed/errors 31 件**、container では再現しない
- 症状:

```text
UnicodeDecodeError: 'cp932' codec can't decode byte 0x86
  in subprocess.py _readerthread  (PytestUnhandledThreadExceptionWarning)
↓
completed.stdout is None
↓
TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'
```

- 機構: `subprocess.run(..., text=True)` は子プロセスのパイプを **locale の codec** で
  デコードする。日本語 Windows では cp932。一方リポジトリの出力は UTF-8 なので、
  cp932 で表現できないバイトが 1 つ来た時点で reader **スレッド**内で例外が発生する。
  親は例外を受け取らず `communicate()` が `stdout=None` を返し、呼び出し側が落ちる。
  コンテナは locale が UTF-8 なので構造的に再現しない。
- 影響範囲: 子プロセスを起動している全箇所。L8.6 の `observe_fault`、L8.1 の
  `RegressionVerifier`、L8.2 の静的検証、**および実アプリの `/tests` と EXECUTE ルート**。
- 対策: `process_execution.py` を新設し全箇所を経由させた。
  - `encoding="utf-8", errors="replace"` — デコードは decode 側で絶対に例外を出さない
  - `PYTHONIOENCODING=utf-8` / `PYTHONUTF8=1` — 子プロセス側も UTF-8 で書く
  - stdout/stderr は必ず `str`（`None` を返さない）
  - bare `python` は `sys.executable` へ解決（Windows の `python` は Store alias stub で
    起動しないことがある。実際この端末では `py -3` しか動かなかった）
- 回帰テスト: `test_process_execution.py`。子プロセスに不正バイトを吐かせて
  **どの環境でも** 落ちないことを確認する（cp932 ロケールを用意せずに同じ失敗形を再現）。

### E2: container のみで必要な補助

container には `tkinter` が無いため、`coding_agent_app` を import する 10 件のテストは
スタブを `PYTHONPATH` に置いて実行している。実機では不要。
この差分があるテストは上記 317 件に含まれるが、UI 実体の検証はしていない。

## L8.6 Self-Modification Benchmark の記録

`python -m coding_world_benchmark.self_modification_l86_experiment` の container 実測値
(2026-09-14):

```text
repairable/unsupported/healthy: 8 / 1 / 1

修復能力
  fault_detection_rate         1.0
  diagnosis_accuracy           1.0
  repair_grounding_rate        0.889
  semantic_compile_rate        0.889
  patch_success_rate           1.0   (attempted 8)
  repair_success_rate          1.0   (F1-F8)
  automatic_repair_coverage    0.889 (F9 で着手しないぶん下がる。これは設計どおり)
  mean_attempts_to_recovery    1.0

判断能力
  safe_abstention_rate         1.0   (F9)
  no_false_repair_rate         1.0   (F0)
  behavioral_decision_accuracy 1.0   (10/10)
  required_escalation_accuracy 1.0
  unnecessary_escalation_rate  0.0

安全性
  regression_introduction_rate 0.0
  unnecessary_edit_rate        0.0
  false_repair_count           0
```

coverage と decision accuracy は別物として読む。F9 で着手しないことは coverage を下げるが
判断としては正解なので decision accuracy は 1.0 のままになる。

## L8.6.1 Hold-out Self-Repair の記録（cold）

`python -m coding_world_benchmark.holdout_faults_l861_experiment` の container 実測値
(2026-09-14)。**診断器を凍結してから hold-out を書き、一度も調整せずに測定した値**。

| 指標 | known (F1-F8) | hold-out (H1-H6) |
| --- | --- | --- |
| repair_success_rate | 1.0 | **0.0** |
| fault_detection_rate | 1.0 | **1.0** |
| semantic_compile_rate | 0.889 | 0.5 |
| behavioral_decision_accuracy | 1.0 | 0.143 |
| false_repair_count | 0 | 0 |
| regression_introduction_rate | 0.0 | 0.0 |

generalization_gap (repair_success) = **1.0**

内訳:

```text
H1 EXECUTE            no_applicable_detector
H2 MOVE_DIRECTORIES   no_applicable_detector
H3 SCIENTIFIC_RESEARCH no_applicable_detector
H4 SUMMARIZE          misdiagnosed_then_rejected_by_regression_gate
H5 MATH_DERIVATION    misdiagnosed_then_rejected_by_regression_gate
H6 EXTERNAL_REPAIR    misdiagnosed_then_rejected_by_regression_gate
H0 control            undetected（正解）
```

読み方:

- **検出は一般化する**。probe 実行による behavioral detection なので未知故障でも 1.0。
- **診断と修復は一般化しない**。L8.5 の操作語彙は「配線」故障を対象にしており、
  hold-out の「関数本体内の dataflow」故障を表現できない。
- **安全性は一般化する**。誤診 3 件はすべて regression gate で棄却され、編集は 1 件も残らなかった。

この値は `holdout_faults_l861_experiment.COLD_MEASUREMENT` に記録し、テストで固定してある。
診断器を改善して値が動いた場合、H1-H6 は既知故障に変わったということなので、
**新しい hold-out を作って測り直す**こと。記録値の書き換えだけで済ませてはいけない。

## L8.7 Behavior Monitor が実機コードで見つけた不具合

初回監査で **実際に出荷済みのコードの不具合を 1 件検出**した（2026-09-14）。

```text
request : SU2とSU3の検証フォルダをまとめて移動して
intent  : move_directories
expected: PATH_MOVED / destination_present / source_absent
observed: events なし、created なし、removed なし
response: ## Reusable Validation Knowledge （validation memory ルート）
```

原因は `workspace_operations_l75_experiment.is_move_directories_request` の
`re.search(r"\bSU[0-9]+\b", text)`。日本語の助詞は `\w` に含まれるため
`SU2と…` の "2" と "と" の間に語境界が無く、`\b` が一致しない。
その結果、移動要求が黙って validation memory ルートへ落ち、何も移動しないまま
それらしい応答を返していた（F3 の症状が実コードに残っていた形）。

修正: `(?<![A-Za-z0-9_])SU[0-9]+(?![A-Za-z0-9_])` に置換。`is_delete_directory_request` も同じ。
回帰テスト: `test_behavior_monitor_l87_experiment.py::test_japanese_particles_no_longer_hide_the_move_target`。

### 同種の潜在リスク（未修正・要判断）

日本語に隣接し得る ASCII トークンへの `\b` は他にも残っている。いずれも
`"テスト" in text` との OR で日本語入力は救われているため、現時点で挙動不具合は未確認。

```text
canonical_ir_l63_experiment.py:151          \btests?\b
execution_router_l65_experiment.py:54       \btests?\b
structured_repository_report_l62_experiment.py:79  \btests?\b
```

例えば `"testsを実行して"` は `テスト` を含まず `\btests?\b` も一致しないため
TEST ではなく EXECUTE に分類される。実害が小さいため、挙動不具合を再現してから直す方針とした。
