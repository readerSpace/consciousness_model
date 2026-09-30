"""A Python-shaped surface for the dialogue -- parsed, never executed.

Typing 「はい」 is fine for one reply and hopeless for a sequence.  A session that
opens an episode, asks, answers, names a model and re-runs is five utterances
whose *order* is the whole point, and prose gives no way to write that down, keep
it, or hand it to someone else.  So the same actions get a code-shaped form::

    open_episode("lab")
    episode("lab")
    ask("64x64のケースで処理時間ではなくフガ率をならして")
    no()
    rerun()

Three decisions make this safe, and each is the project's own rule reappearing at
the input layer.

**Nothing is evaluated.**  The source is parsed with ``ast`` and matched against a
fixed table of calls with literal arguments.  ``eval`` and ``exec`` are never
used, so there is no expression, attribute, import or comprehension that could
reach anything; a form outside the table is refused by name rather than tried.
That is the same fail-closed default L8.27 had to be repaired into having.

**The whole script is parsed before any of it runs.**  A typo on line 4 means
lines 1 to 3 do not happen either.  This is L8.18's execution gate -- evoked and
not resolved means do not execute -- applied to the input language rather than to
an operation's arguments: a half-understood script is not half-run.

**The calls are the session's own actions and nothing more.**  There is no
arithmetic, no variables and no control flow, because the point is not to program
the agent but to write down a conversation exactly.  Anything the layers do not
already expose has no call here.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

#: The literal types an argument may be. Everything else -- names, attributes,
#: f-strings, lists, calls -- is refused, which keeps the surface a transcript
#: rather than a language.
LITERALS = (str, int, float, bool)


class ScriptError(ValueError):
    """A script that will not be run, with the line that stopped it."""

    def __init__(self, message: str, line: int = 0):
        super().__init__(message)
        self.line = line

    def __str__(self) -> str:
        return (f"{self.args[0]}（{self.line} 行目）" if self.line
                else str(self.args[0]))


@dataclass(frozen=True)
class Signature:
    name: str
    params: tuple[str, ...]
    required: int
    doc: str
    example: str


SIGNATURES: tuple[Signature, ...] = (
    Signature("ask", ("request",), 1, "要求を解釈にかける。未知語があれば質問する",
              'ask("64x64のケースで処理時間ではなくフガ率をならして")'),
    Signature("answer", ("reply",), 1, "保留中の質問に答える", 'answer("はい")'),
    Signature("yes", (), 0, "はい", "yes()"),
    Signature("no", (), 0, "いいえ", "no()"),
    Signature("dont_know", (), 0, "分からない。否定として扱われない", "dont_know()"),
    Signature("hedge", (), 0, "たぶん違う。確定的でないので記録されない", "hedge()"),
    Signature("confused", (), 0, "質問の意味が分からない。訊き方を変えさせる",
              "confused()"),
    Signature("interrupt", (), 0, "作業をやめる。語の意味に痕跡を残さない",
              "interrupt()"),
    Signature("correct", ("field",), 1, "指標を名指して訂正する",
              'correct("成功率")'),
    Signature("episode", ("name",), 1, "現在の episode を切り替える",
              'episode("lab")'),
    Signature("open_episode", ("name",), 1,
              "episode を開き、BLOCKED の質問を訊けるようにする",
              'open_episode("lab")'),
    Signature("model", ("label",), 1,
              "時期か場面か、どちらが意味を決めているかを名指す", 'model("場面")'),
    Signature("rerun", ("request",), 0,
              "直近の要求を episode ごとに実行し直す", "rerun()"),
    Signature("state", (), 0, "解釈状態を書き出す", "state()"),
    Signature("reset", (), 0, "解釈状態を捨てる", "reset()"),
    Signature("help", (), 0, "使える呼び出しを並べる", "help()"),
    Signature("theorem_search", ("query",), 1,
              "定理コーパスを検索する", 'theorem_search("assoc")'),
    Signature("theorem_run", ("cycles",), 0,
              "定理発見核を回す（篩を通った候補であって証明ではない）",
              "theorem_run(12)"),
)

BY_NAME = {signature.name: signature for signature in SIGNATURES}

#: Replies whose behaviour the frozen dialogue hold-out fixed; the zero-argument
#: calls are named after what they do rather than after the words, so a script
#: reads as a sequence of moves.
_SHORTHAND = {
    "yes": "はい", "no": "いいえ", "dont_know": "分からない",
    "hedge": "たぶん違う", "confused": "質問の意味が分からない",
    "interrupt": "やっぱりこの作業やめて",
}


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple
    kwargs: dict
    line: int
    source: str


def _literal(node: ast.AST, line: int):
    if not isinstance(node, ast.Constant) or not isinstance(node.value, LITERALS):
        raise ScriptError("引数はリテラル（文字列・数値・真偽値）だけです", line)
    return node.value


def parse(source: str) -> tuple[Call, ...]:
    """Read the script into calls, or refuse the whole thing.

    Refusing the whole thing is the point: the caller gets either every call or
    none, so a script is never partly run because its last line was wrong.
    """
    text = source.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```")
                         else lines[1:])
    try:
        tree = ast.parse(text, mode="exec")
    except SyntaxError as error:
        raise ScriptError(f"構文が読めません: {error.msg}", error.lineno or 0) from error

    numbered = text.splitlines()
    found: list[Call] = []
    for node in tree.body:
        line = getattr(node, "lineno", 0)
        raw = numbered[line - 1].strip() if 0 < line <= len(numbered) else ""
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            raise ScriptError("1行につき1つの呼び出しだけを書いてください", line)
        call = node.value
        if not isinstance(call.func, ast.Name):
            raise ScriptError("呼び出せるのは決められた関数名だけです", line)
        signature = BY_NAME.get(call.func.id)
        if signature is None:
            known = "、".join(sorted(BY_NAME))
            raise ScriptError(f"`{call.func.id}` は使えません（使えるのは {known}）", line)
        args = tuple(_literal(item, line) for item in call.args)
        kwargs = {}
        for keyword in call.keywords:
            if keyword.arg is None:
                raise ScriptError("`**` は使えません", line)
            if keyword.arg not in signature.params:
                raise ScriptError(
                    f"`{signature.name}` に `{keyword.arg}` という引数はありません", line)
            kwargs[keyword.arg] = _literal(keyword.value, line)
        total = len(args) + len(kwargs)
        if total > len(signature.params):
            raise ScriptError(f"`{signature.name}` の引数が多すぎます", line)
        if total < signature.required:
            raise ScriptError(f"`{signature.name}` には引数が必要です", line)
        if len(args) > len(signature.params):
            raise ScriptError(f"`{signature.name}` の引数が多すぎます", line)
        for index, _value in enumerate(args):
            if signature.params[index] in kwargs:
                raise ScriptError(
                    f"`{signature.params[index]}` が二重に指定されています", line)
        found.append(Call(signature.name, args, kwargs, line, raw))
    if not found:
        raise ScriptError("実行できる呼び出しがありません")
    return tuple(found)


def _bound(call: Call) -> dict:
    signature = BY_NAME[call.name]
    values = dict(zip(signature.params, call.args))
    values.update(call.kwargs)
    return values


def help_text() -> str:
    lines = ["**使える呼び出し**", "", "| 呼び出し | 意味 |", "| --- | --- |"]
    for signature in SIGNATURES:
        lines.append(f"| `{signature.example}` | {signature.doc} |")
    lines += ["", "1行につき1呼び出し。引数はリテラルのみ。"
              "`#` のコメントと空行は無視されます。"
              "**1行でも読めなければ1行も実行しません。**"]
    return "\n".join(lines)


def run(session, source: str) -> dict:
    """Parse the whole script, then perform it, and report each line.

    ``session`` is a :class:`~coding_world_benchmark.semantic_service.SemanticSession`;
    every call below is one of its own methods, so this module adds no behaviour
    of its own -- only a way to write the behaviour down.
    """
    from .semantic_service import render_state

    try:
        calls = parse(source)
    except ScriptError as error:
        return {"ok": False, "error": str(error), "results": [],
                "state": session.state(), "output": f"**実行しませんでした**: {error}"}

    results: list[dict] = []
    state = session.state()
    for call in calls:
        values = _bound(call)
        detail = ""
        if call.name == "ask":
            state = session.ask(str(values["request"]))
            detail = state.get("reply", "")
        elif call.name == "answer":
            state = session.answer(str(values["reply"]))
            detail = state.get("reply", "")
        elif call.name in _SHORTHAND:
            state = session.answer(_SHORTHAND[call.name])
            detail = state.get("reply", "")
        elif call.name == "correct":
            state = session.answer(f"それじゃなくて{values['field']}")
            detail = state.get("reply", "")
        elif call.name == "episode":
            session.use_episode(str(values["name"]))
            state = session.state()
            detail = f"現在の episode: {session.episode}"
        elif call.name == "open_episode":
            session.open_episode(str(values["name"]))
            state = session.state()
            detail = f"{values['name']} を開きました"
        elif call.name == "model":
            state = session.answer_structural(str(values["label"]))
            detail = state.get("reply", "")
        elif call.name == "rerun":
            request = values.get("request") or session.last_request
            if not request:
                detail = "実行し直す要求がありません"
                state = session.state()
            else:
                programs = session.rerun(str(request))
                state = session.state()
                state["rerun"] = programs
                detail = " / ".join(f"{name}: {program or '実行せず'}"
                                    for name, program in programs.items())
        elif call.name == "reset":
            session.reset()
            state = session.state()
            detail = "解釈状態を初期化しました"
        elif call.name == "state":
            state = session.state()
            detail = f"{state['status']} / {state['diagnosis']}"
        elif call.name == "help":
            state = session.state()
            detail = help_text()
        elif call.name in ("theorem_search", "theorem_run"):
            from .theorem_service import (
                DEFAULT_CYCLES, format_run, format_search, run_discovery, search,
            )
            state = session.state()
            if call.name == "theorem_search":
                query = str(values["query"])
                detail = format_search(query, search(query))
            else:
                outcome = run_discovery(int(values.get("cycles") or DEFAULT_CYCLES))
                detail = (format_run(outcome.payload) if outcome.ok
                          else f"回せませんでした: {outcome.detail}")
        results.append({"line": call.line, "source": call.source,
                        "call": call.name, "status": state.get("status", ""),
                        "detail": detail})

    lines = ["**スクリプトを実行しました**", "",
             "| 行 | 呼び出し | 状態 | 結果 |", "| --- | --- | --- | --- |"]
    for item in results:
        shown = item["detail"].replace("\n", " ")
        lines.append(f"| {item['line']} | `{item['source']}` | {item['status']} "
                     f"| {shown[:120]} |")
    if any(item["call"] == "help" for item in results):
        lines += ["", help_text()]
    lines += ["", render_state(state)]
    return {"ok": True, "error": None, "results": results, "state": state,
            "output": "\n".join(lines)}


def looks_like_script(message: str) -> bool:
    """Is this message a script rather than something someone said?

    Only a source that parses *completely* into known calls counts, so ordinary
    prose -- and any code-looking thing outside the table -- falls through to the
    normal path untouched.
    """
    text = message.strip()
    if text.startswith("```"):
        return True
    if "(" not in text or ")" not in text:
        return False
    try:
        parse(text)
    except ScriptError:
        return False
    return True
