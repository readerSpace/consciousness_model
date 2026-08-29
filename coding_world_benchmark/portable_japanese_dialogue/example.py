"""Run with: python portable_japanese_dialogue/example.py"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portable_japanese_dialogue import JapaneseDialogueEngine


def main() -> None:
    engine = JapaneseDialogueEngine(workspace_capacity=4)
    reply = engine.reply(
        "昨日Minecraftを買ったけど難しい！",
        display_name="たろう",
        history=("こんにちは", "ゲームは好き？"),
        memories=("前に洞窟を探している",),
    )
    print(reply.text)
    print(f"response_act={reply.response_act}")
    print(f"topics={reply.frame.topics}")
    print(f"events={reply.frame.events}")
    print(f"memory_signal={reply.memory_signal}")
    print(f"attention_signal={reply.attention_signal}")


if __name__ == "__main__":
    main()