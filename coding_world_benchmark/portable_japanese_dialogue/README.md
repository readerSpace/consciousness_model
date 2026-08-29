# Portable Japanese Dialogue

依存ライブラリなしで使える、日本語文の解析、有限文脈に基づく応答行為の推論、返答文生成をまとめた移植用フォルダである。別プロジェクトには、このフォルダ全体をコピーして使う。

## 含まれる機能

- 日本語文を発話行為、話題、簡易イベント、時間・照応参照、感情手掛かりへ解析する。
- 過去発話を salience で順位付けし、容量制限された workspace に選択する。
- 選択文脈から memory/attention 信号を計算する。
- 挨拶、質問、同意、軽い煽り、記憶参照、話題の掘り下げを概念的な応答行為として選択する。
- 応答行為を短い日本語に表層化し、解析結果と選択文脈を返す。

これは透明なルールベースの基礎実装であり、自由な日本語理解、事実検証、外部知識検索、LLM による自然な長文生成は含まない。

## 使用例

コピー先のプロジェクトで、親ディレクトリが import path にある状態なら次のように使える。

```python
from portable_japanese_dialogue import JapaneseDialogueEngine

engine = JapaneseDialogueEngine(workspace_capacity=4)
result = engine.reply(
    "昨日Minecraftを買ったけど難しい！",
    display_name="たろう",
    history=("こんにちは", "ゲームは好き？"),
    memories=("前に洞窟を探している",),
)

print(result.text)
print(result.response_act)
print(result.frame.topics)
```

`history` には文字列または `SemanticFrame` を渡せる。`memories` は呼出側の DB、ベクトル検索、またはセッション状態から取得した短い根拠文を渡す。保存、検索、ユーザー識別、HTTP API、LLM 接続はこのパッケージの責務ではない。

## 動作確認

```powershell
python portable_japanese_dialogue/example.py
```

## 移植時の拡張点

- より高精度な解析器を導入する場合は `JapaneseDialogueEngine.analyze` を置き換える。
- DB を使う場合は、過去発話を `history`、検索済み記憶を `memories` として渡す。
- LLM を使う場合は、`DialogueReply.response_act` と `frame` をプロンプトへ渡し、最終表層化だけを外部 renderer に任せる。
- HTTP サービス化する場合は、`reply` の引数と `DialogueReply` をそのまま request/response schema にできる。