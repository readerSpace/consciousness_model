# Idea Research Workspace - Phase 1

研究メモを `RawNote -> AtomicClaim -> Concept -> Research Graph -> Gap -> Experiment` に変換する、設計書の最小実装です。自然言語の構造化は透明な規則ベースのベースラインで実装しており、将来は `claim_extractor.py` と `experiment_generator.py` をJSON Schema対応LLMへ置き換えられます。

## Run

ワークスペースのルートから実行します。

```powershell
python -m idea_generate.run_pipeline idea_generate/data/sample_notes.jsonl
```

出力先は既定で `idea_generate/data/output/` です。`claims.json`、`concepts.json`、`gaps.json`、`research_graph.json`、`experiments.json`、`digest.txt` を生成します。

```powershell
python -m unittest discover -s idea_generate/tests -v
```

入力はUTF-8のJSONLで、各行に `id` と `text`、任意で `created_at` と `tags` を指定します。

## Experiment Design

実験提案は固定Schemaで保存されます。`target_hypothesis` は質問から検証可能な仮説へ変換され、`experiment_type` に応じて `procedure`、`metrics`、資源、反証条件を設計します。型は `controlled_experiment`、`simulation`、`benchmark`、`ablation`、`theoretical_analysis`、`literature_check`、`observational_analysis`、`search_experiment` です。

介入・対照が適さない理論解析、観察分析、探索実験では、`intervention` と `control_condition` はJSONの `null` になります。生成後には `ExperimentReview` が、手順、測定可能性、反証可能性、および比較形式の対照条件を検査します。

## Google Keep Import

Google Keepエクスポートの `Keep` ディレクトリを入力にできます。研究キーワードに一致する非ゴミ箱メモを編集日時順に最大100件選びます。

```powershell
python -m idea_generate.import_keep idea_generate/data/keepmemo/Keep
python -m idea_generate.run_pipeline idea_generate/data/keepmemo_research_notes.jsonl --output idea_generate/data/keepmemo_output
```