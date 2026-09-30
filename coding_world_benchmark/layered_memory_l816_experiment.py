"""L8.16: measure what belongs in the compressed state and what must stay verbatim.

The question is not how large a context window can be made, but which parts of
a long conversation survive compression and which have to be kept as original
text and fetched on demand.  That is measurable here without a model, because
the compression is real: L7.3's ``_compact_lines`` keeps only lines carrying a
marker such as 結論 or 判定 and discards everything else, so

    sim_007.py を作成しました。
    格子サイズは 64x64、乱数種は 20260917 を使用しました。
    結論: 相転移点は T=2.27 付近に現れる。

reaches the long-term state as its last line alone.  The conclusion survives.
The parameters do not exist anywhere after compression, so "そのままの値を出して"
has no grounding left, however well the state is searched.

Three layers are compared on the same conversation:

* **short term** -- the last K episodes as raw text;
* **compressed** -- L7.3 facts plus the L8.13 episode index (what exists today);
* **verbatim** -- original request/response per episode, fetched only when the
  query asks for exact wording, addressed through the L8.15 resolver.

Measurement is retrieval-side on purpose.  Each probe carries a span the answer
requires, held by the generator, and a configuration scores by whether that span
reaches the context at all -- what a model would then say is a separate
question, and mixing the two would hide which layer was at fault.

One probe strains the retrieval-side proxy and is called out rather than
hidden: counting episodes is a computation, so the full transcript holds the
material without holding the answer and scores as misleading even though a
model could count it.  For that probe the meaningful column is cost, not
correctness; for the other four the span is literal text that is either
present or not.

The metric that matters is not recall but `misleading_context_rate`: how often a
configuration returns evidence it presents as sufficient while the required span
is absent.  That is the state in which a model confabulates fluently, and it is
worse than an empty answer.  A configuration that abstains scores zero on both
recall and misleading context; only one of those is a failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re

from .behavior_monitor_l87_experiment import BehaviorEvent
from .conversation_memory_l73_experiment import ConversationKnowledgeMemory
from .episodic_memory_l813_experiment import EpisodicMemory
from .goal_execution_loop_l812_experiment import run_goal_loop, trace_with
from .structural_resolver_l815_experiment import query_tokens, resolve_reference_structural

_VERBATIM_MARKERS = ("一字一句", "そのまま", "原文", "逐語", "正確に")
_AGGREGATE_MARKERS = ("何件", "いくつ", "何個", "件数")
#: "What to return", as opposed to "which episode".  A request for exact
#: wording names a thing *and* an aspect of it, and feeding the aspect words
#: into episode resolution destroys coverage: 「さっきのパラメータを原文のまま」
#: resolves to nothing because パラメータ and 原文 match no episode name.
#: The two halves are separated before resolution.
_ASPECT_WORDS = ("パラメータ", "設定値", "設定", "コマンド", "出力", "値")

_TARGETS = ("イジング", "ハイゼンベルク", "ポッツ", "XY", "球面")
_LATTICES = ("32x32", "64x64", "128x128", "256x256", "512x512")
_SEEDS = ("20260101", "20260215", "20260330", "20260512", "20260624")
_CRITICAL = ("2.27", "1.44", "0.89", "3.05", "4.51")


@dataclass(frozen=True)
class ConversationTurn:
    episode_index: int
    request: str
    response: str
    parameter_line: str
    conclusion_line: str
    artifact: str


@dataclass
class LayeredMemory:
    """The three layers over one conversation.  Layers 2 and 3 are independent."""

    episodic: EpisodicMemory
    turns: tuple[ConversationTurn, ...]
    verbatim: dict[str, ConversationTurn] = field(default_factory=dict)

    def transcript_chars(self) -> int:
        return sum(len(turn.request) + len(turn.response) for turn in self.turns)

    def compressed_chars(self) -> int:
        facts = sum(len(fact.value) for fact in self.episodic.memory.facts)
        index = sum(len(episode.request) for episode in self.episodic.episodes)
        return facts + index


@dataclass(frozen=True)
class Retrieval:
    layer: str
    evidence: str
    can_answer: bool
    reason: str


@dataclass(frozen=True)
class LayerConfig:
    name: str
    short_term_episodes: int
    compressed: bool
    verbatim: bool


CONFIGURATIONS: tuple[LayerConfig, ...] = (
    LayerConfig("short_term_only", 2, False, False),
    LayerConfig("compressed_only", 0, True, False),
    LayerConfig("layered", 2, True, True),
    LayerConfig("full_transcript", 10_000, False, False),
)


def build_conversation(count: int = 24) -> LayeredMemory:
    store = ConversationKnowledgeMemory(_NullPath(), max_facts=max(count * 2, 240))
    store.facts = []
    episodic = EpisodicMemory(store)
    turns: list[ConversationTurn] = []

    for index in range(count):
        target = _TARGETS[index % 5]
        lattice = _LATTICES[(index // 5) % 5]
        seed = _SEEDS[index % 5]
        critical = _CRITICAL[(index // 5) % 5]
        artifact = f"sim_{index:03d}.py"
        request = f"run{index:03d}: {target}模型の相転移を解くコードを作成して"
        parameter_line = f"格子サイズは {lattice}、乱数種は {seed} を使用しました。"
        conclusion_line = f"結論: {target}模型の相転移点は T={critical} 付近に現れる。"
        response = f"{artifact} を作成しました。\n{parameter_line}\n{conclusion_line}"

        def executor(_request, _directives, response=response):
            return response, trace_with(BehaviorEvent.FILE_WRITTEN)

        episode = episodic.record(run_goal_loop(request, executor))
        turn = ConversationTurn(index, request, response, parameter_line, conclusion_line, artifact)
        turns.append(turn)

    layered = LayeredMemory(episodic, tuple(turns))
    for position, turn in enumerate(turns):
        layered.verbatim[f"ep-{position + 1:03d}"] = turn
    return layered


class _NullPath:
    def is_file(self) -> bool:
        return False


def reference_part(query: str) -> str:
    """Drop the projection so only the episode reference is resolved.

    Also applied before looking for a continuation marker, because L8.13
    matches markers as substrings and 「そのまま」 contains 「その」 -- without this
    an ungrounded request reads as a reference to the most recent episode.
    """
    text = query
    for word in _VERBATIM_MARKERS + _ASPECT_WORDS:
        text = text.replace(word, "")
    return text


def _wants_verbatim(query: str) -> bool:
    return any(marker in query for marker in _VERBATIM_MARKERS)


def _wants_aggregate(query: str) -> bool:
    return any(marker in query for marker in _AGGREGATE_MARKERS)


def _window_text(memory: LayeredMemory, size: int) -> str:
    if size <= 0:
        return ""
    recent = memory.turns[-size:]
    return "\n".join(f"{turn.request}\n{turn.response}" for turn in recent)


def _ordinary(memory: LayeredMemory, config: LayerConfig, query: str) -> Retrieval:
    """Short-term window plus compressed lookup.

    The window is unconditional: it is the context a turn carries, not a
    retrieval result, so it is supplied whether or not it is relevant.  That is
    exactly why a window-only system answers questions it has no evidence for.
    """
    parts = []
    window = _window_text(memory, config.short_term_episodes)
    if window:
        parts.append(window)
    if config.compressed:
        facts = memory.episodic.memory.recall(query, limit=4)
        if facts:
            parts.append("\n".join(fact.value for fact in facts))
    evidence = "\n".join(parts)
    return Retrieval(
        "short-term+compressed" if parts else "none", evidence, bool(evidence), "ordinary lookup"
    )


def retrieve(memory: LayeredMemory, config: LayerConfig, query: str) -> Retrieval:
    if _wants_aggregate(query):
        if not config.compressed:
            # Falls through rather than abstaining: a configuration without an
            # index does not know the question needs one, it just hands over
            # whatever text it holds.
            return _ordinary(memory, config, query)
        tokens = [
            token
            for token in query_tokens(reference_part(query))
            if not any(marker in token for marker in _AGGREGATE_MARKERS)
        ]
        # Conjunctive: an episode counts only when it carries every named term.
        # Matching on any one term counts 模型 across the whole corpus.
        matched = [
            episode
            for episode in memory.episodic.episodes
            if tokens and all(token in episode.request for token in tokens)
        ]
        evidence = f"該当 {len(matched)} 件\n" + "\n".join(item.request for item in matched)
        return Retrieval("compressed-index", evidence, bool(matched), "counted over the episode index")

    if _wants_verbatim(query) and config.verbatim:
        reference = reference_part(query)
        resolution = resolve_reference_structural(memory.episodic, reference)
        if resolution.decision == "CONTINUE_EPISODE" and resolution.episode_id in memory.verbatim:
            turn = memory.verbatim[resolution.episode_id]
            return Retrieval(
                "verbatim", f"{turn.request}\n{turn.response}", True, "original text supplied"
            )
        return Retrieval(
            "verbatim", "", False,
            f"exact wording requested but no episode could be identified ({resolution.decision})",
        )

    return _ordinary(memory, config, query)


@dataclass(frozen=True)
class Probe:
    text: str
    kind: str
    required_span: str | None
    note: str


def build_probes(memory: LayeredMemory) -> tuple[Probe, ...]:
    turns = memory.turns
    recent = turns[-1]
    old = turns[2]
    probes = [
        Probe(
            f"さっきの実験のパラメータを原文のまま出して",
            "RECENT_VERBATIM",
            recent.parameter_line,
            "recent exact wording: the short-term window alone should carry it",
        ),
        Probe(
            f"{old.artifact} で使ったパラメータを一字一句そのまま出して",
            "VERBATIM_OLD",
            old.parameter_line,
            "old exact wording: compression dropped this line, only the verbatim layer has it",
        ),
        Probe(
            f"{old.artifact} の実験で分かったことは？",
            "COMPRESSED_FACT",
            old.conclusion_line.replace("結論: ", "")[:20],
            "semantic recall: exactly what the compressed state is for",
        ),
        Probe(
            "イジング模型の実験は何件やった？",
            "AGGREGATE",
            "該当 5 件",
            "counting over episodes needs the index, not the text",
        ),
        Probe(
            "超伝導ギャップの測定値をそのまま出して",
            "UNANSWERABLE",
            None,
            "control: never discussed, so every configuration must abstain",
        ),
    ]
    return tuple(probes)


def _ratio(numerator: int, denominator: int) -> float | str:
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def run_experiment(count: int = 24) -> dict[str, object]:
    memory = build_conversation(count)
    probes = build_probes(memory)
    results = []

    for config in CONFIGURATIONS:
        records = []
        for probe in probes:
            retrieval = retrieve(memory, config, probe.text)
            if probe.required_span is None:
                recalled = False
                misleading = retrieval.can_answer
            else:
                recalled = probe.required_span in retrieval.evidence
                misleading = retrieval.can_answer and not recalled
            records.append(
                {
                    "kind": probe.kind,
                    "note": probe.note,
                    "layer": retrieval.layer,
                    "recalled": recalled,
                    "misleading": misleading,
                    "abstained": not retrieval.can_answer,
                    "chars": len(retrieval.evidence),
                    "reason": retrieval.reason,
                }
            )

        answerable = [item for item in records if item["kind"] != "UNANSWERABLE"]
        unanswerable = [item for item in records if item["kind"] == "UNANSWERABLE"]
        results.append(
            {
                "config": config.name,
                "records": records,
                "evidence_recall": _ratio(sum(1 for i in answerable if i["recalled"]), len(answerable)),
                "misleading_context_rate": _ratio(
                    sum(1 for i in records if i["misleading"]), len(records)
                ),
                "abstention_accuracy": _ratio(
                    sum(1 for i in unanswerable if i["abstained"]), len(unanswerable)
                ),
                "mean_context_chars": round(
                    sum(i["chars"] for i in records) / len(records), 1
                ),
                "per_kind": {
                    item["kind"]: ("hit" if item["recalled"] else ("misleading" if item["misleading"] else "abstain"))
                    for item in records
                },
            }
        )

    return {
        "memory": memory,
        "probes": probes,
        "results": results,
        "transcript_chars": memory.transcript_chars(),
        "compressed_chars": memory.compressed_chars(),
        "compression_ratio": round(memory.compressed_chars() / memory.transcript_chars(), 3),
    }


def format_experiment(report: dict[str, object]) -> str:
    kinds = [probe.kind for probe in report["probes"]]
    lines = [
        "## Layered Memory (L8.16)",
        "",
        f"- transcript: `{report['transcript_chars']}` chars",
        f"- compressed state: `{report['compressed_chars']}` chars "
        f"(compression_ratio `{report['compression_ratio']}`)",
        "",
        "| config | evidence_recall | misleading_context | abstention_acc | mean_chars |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in report["results"]:
        lines.append(
            f"| {result['config']} | {result['evidence_recall']} | {result['misleading_context_rate']} "
            f"| {result['abstention_accuracy']} | {result['mean_context_chars']} |"
        )

    lines += ["", "| config | " + " | ".join(kinds) + " |", "| --- |" + " --- |" * len(kinds)]
    for result in report["results"]:
        lines.append(
            f"| {result['config']} | " + " | ".join(result["per_kind"][kind] for kind in kinds) + " |"
        )
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - manual entry point
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
