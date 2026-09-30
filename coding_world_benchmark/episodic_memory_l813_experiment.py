"""L8.13: give the conversation memory episode boundaries.

L7.3 (``ConversationKnowledgeMemory``) already compresses a conversation into
reusable facts, but it stores one flat pool keyed by *topic*.  Nothing records
where one piece of work ended and the next began, so recall can only separate
tasks by guessing from words.  That is why ``recall`` carries hand-written
policy blacklists -- ``repair_related_only`` literally drops any fact
mentioning ``gauge``/``su2``/``quantum`` so that repair work does not pull in
physics facts.  Those blacklists are a workaround for the missing boundary,
and they have to be extended by hand for every new pair of topics.

This module adds the boundary structurally instead.  An ``Episode`` is one
completed unit of work, distilled from the L8.12 ``LoopOutcome``, and it does
**not** copy the knowledge: it stores the *keys* of the L7.3 facts that the
work produced.  L7.3 stays the single store of facts, unmodified -- it is
still what ``coding_agent_app`` and ``coding_agent_bridge`` read -- and
episodes are an index over it.

Recall then follows the boundary rather than the vocabulary:

* a request that continues earlier work restores exactly that episode's facts;
* a request that starts new work gets **nothing** from earlier episodes, which
  is the contamination fix, and needs no per-topic blacklist to achieve it.

Resolution refuses to guess in the two cases where guessing is what corrupts
long conversations: a name matching several episodes, and a name matching
none.  Substituting "the most recent episode" for a name the user actually
typed is the failure this layer exists to prevent, so both answer ``CLARIFY``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import json
from pathlib import Path
import re

from .conversation_memory_l73_experiment import ConversationKnowledgeMemory, MemoryFact
from .goal_execution_loop_l812_experiment import LoopOutcome

#: Words that mark a request as continuing earlier work rather than starting new work.
_CONTINUATION_MARKERS: tuple[str, ...] = (
    "それ", "その", "これ", "さっき", "先ほど", "さきほど", "前の", "続き", "続けて",
    "もう一度", "同じ", "引き続き",
    "that one", "the previous", "continue", "again", "same as before",
)

#: Vocabulary that describes *work* rather than naming a *thing*.  A token in
#: this set can never identify an episode, because a word every task uses would
#: otherwise capture whichever episode happened to contain it -- "さっきの計算"
#: resolving to the SU(2) episode simply because its request said 計算.
#:
#: This list is the brittle part of the layer and is maintained by hand, in the
#: same way the repository keeps an explicit list of the `\b`-with-Japanese
#: sites.  Adding a domain here is cheap; failing to add one shows up as a
#: wrong_episode_reuse in the L8.13 probes rather than as silent corruption.
_REFERENCE_STOPWORDS: frozenset[str] = frozenset(
    {
        "修正", "変更", "作成", "実行", "検証", "実験", "確認", "追加", "削除", "移動",
        "続き", "先ほど", "同じ", "結果", "内容", "場合", "処理", "対応", "方法",
        "ファイル", "フォルダ", "ディレクトリ", "コード", "テスト",
        "計算", "理論", "運動", "作業", "解析", "実装", "設計", "関数", "出力", "入力",
        "データ", "ソルバー", "プログラム", "スクリプト", "モジュール",
        "fix", "change", "create", "run", "verify", "test", "again", "continue",
        "file", "folder", "directory", "code", "the", "that", "this", "previous",
        "script", "module", "function", "program", "solver", "data", "output",
    }
)

_NAME_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_-]+|[一-龥]{2,}|[ァ-ヶー]{2,}")
_FILENAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-./\\]*\.[A-Za-z0-9]{1,5}")


@dataclass(frozen=True)
class Episode:
    """One completed unit of work.

    Deliberately a distillation of ``LoopOutcome`` rather than a copy of it:
    a trace is not serialisable and not useful later, while the goal state,
    the artifacts and the produced fact keys are.  ``fact_keys`` is the whole
    point -- the knowledge itself stays in L7.3.
    """

    episode_id: str
    request: str
    topic: str
    goal_state: str
    operators: tuple[str, ...]
    status: str
    achieved: bool
    iteration_count: int
    artifacts: tuple[str, ...]
    fact_keys: tuple[str, ...]
    names: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceResolution:
    decision: str  # NEW_EPISODE | CONTINUE_EPISODE | CLARIFY
    episode_id: str | None
    candidates: tuple[str, ...]
    reason: str


def _named_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for match in _NAME_TOKEN.findall(text):
        token = match.lower() if match[0].isascii() else match
        if token in _REFERENCE_STOPWORDS or len(token) < 2:
            continue
        tokens.append(token)
    for match in _FILENAME.findall(text):
        tokens.append(match.lower())
    return tuple(dict.fromkeys(tokens))


def _is_specific_name(token: str) -> bool:
    """A name that identifies a thing, not a description of one.

    Filenames, and latin tokens carrying a digit (``su2``, ``rk4``).  Plain
    words are excluded on purpose: they are how a generic noun silently
    captures an unrelated episode.
    """
    if _FILENAME.fullmatch(token):
        return True
    return token.isascii() and any(character.isdigit() for character in token)


def has_continuation_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in text or marker in lowered for marker in _CONTINUATION_MARKERS)


def _artifacts_of(outcome: LoopOutcome) -> tuple[str, ...]:
    names: list[str] = []
    for iteration in outcome.iterations:
        if iteration.observation.file_written_count > 0:
            names.extend(_FILENAME.findall(iteration.response_text))
    return tuple(dict.fromkeys(names))


class EpisodicMemory:
    """Episode boundaries layered over an existing L7.3 memory.

    The L7.3 instance is used, not replaced or subclassed: this class never
    writes to its file and never changes its format, so the shipped agent keeps
    reading the same ``.knowledge_memory.json.gz``.
    """

    def __init__(self, memory: ConversationKnowledgeMemory, path: Path | None = None):
        self.memory = memory
        self.path = path
        self.episodes: list[Episode] = []
        if path is not None:
            self.load()

    # -- recording ---------------------------------------------------------

    def record(self, outcome: LoopOutcome) -> Episode:
        """Close one unit of work and index the facts it produced."""
        final_response = outcome.iterations[-1].response_text if outcome.iterations else ""
        before = {fact.key: fact.uses for fact in self.memory.facts}
        self.memory.learn(outcome.request, final_response)
        produced = tuple(
            fact.key
            for fact in self.memory.facts
            if fact.key not in before or fact.uses > before[fact.key]
        )

        artifacts = _artifacts_of(outcome)
        names = tuple(
            dict.fromkeys(_named_tokens(outcome.request) + tuple(item.lower() for item in artifacts))
        )
        episode = Episode(
            episode_id=f"ep-{len(self.episodes) + 1:03d}",
            request=outcome.request,
            topic=ConversationKnowledgeMemory._topic_key(outcome.request),
            goal_state=outcome.plan.goal.desired_state,
            operators=tuple(op.value for op in outcome.plan.goal.operators),
            status=outcome.status,
            achieved=outcome.achieved,
            iteration_count=outcome.iteration_count,
            artifacts=artifacts,
            fact_keys=produced,
            names=names,
        )
        self.episodes.append(episode)
        return episode

    def episode(self, episode_id: str) -> Episode | None:
        return next((item for item in self.episodes if item.episode_id == episode_id), None)

    # -- reference resolution ---------------------------------------------

    def resolve_reference(self, request: str) -> ReferenceResolution:
        """Decide whether a request continues earlier work, and which.

        Specific names are checked before continuation markers, because naming
        an artifact an earlier episode produced ("lorentz_particle.py を修正して")
        is a reference whether or not the sentence also says "さっきの".  A
        *specific* name is a filename or a latin token containing a digit
        (``su2``, ``l813``) -- deliberately narrow, so that ordinary nouns like
        ``計算`` can never identify an episode on their own.  Matching on such
        a generic word is how "さっきのSU4の計算" ends up resolved to the SU2
        episode, which is the exact substitution this layer must refuse.
        """
        marker = has_continuation_marker(request)
        if not self.episodes:
            return ReferenceResolution("NEW_EPISODE", None, (), "no earlier episode to continue")

        tokens = set(_named_tokens(request))
        specific = {token for token in tokens if _is_specific_name(token)}

        if specific:
            matches = [
                episode for episode in self.episodes if specific & set(episode.names)
            ]
            if len(matches) == 1:
                return ReferenceResolution(
                    "CONTINUE_EPISODE",
                    matches[0].episode_id,
                    (matches[0].episode_id,),
                    "named one episode by a specific name",
                )
            if len(matches) > 1:
                return ReferenceResolution(
                    "CLARIFY",
                    None,
                    tuple(episode.episode_id for episode in matches),
                    "the name matches more than one episode",
                )
            if marker:
                return ReferenceResolution(
                    "CLARIFY",
                    None,
                    tuple(episode.episode_id for episode in self.episodes),
                    "named a target that no episode matches",
                )
            return ReferenceResolution("NEW_EPISODE", None, (), "a new name, not a reference")

        if not marker:
            return ReferenceResolution("NEW_EPISODE", None, (), "no continuation marker")

        matches = [episode for episode in self.episodes if tokens & set(episode.names)]
        if len(matches) == 1:
            return ReferenceResolution(
                "CONTINUE_EPISODE", matches[0].episode_id, (matches[0].episode_id,), "named one episode"
            )
        if len(matches) > 1:
            return ReferenceResolution(
                "CLARIFY",
                None,
                tuple(episode.episode_id for episode in matches),
                "the name matches more than one episode",
            )

        latest = self.episodes[-1]
        return ReferenceResolution(
            "CONTINUE_EPISODE", latest.episode_id, (latest.episode_id,), "bare reference means the latest episode"
        )

    # -- scoped recall -----------------------------------------------------

    def scoped_recall(self, request: str, limit: int = 6) -> tuple[MemoryFact, ...]:
        """Facts a new turn may see.  Empty unless the turn continues an episode."""
        return self.scoped_recall_for(self.resolve_reference(request), request, limit)

    def scoped_recall_for(
        self, resolution: ReferenceResolution, request: str, limit: int = 6
    ) -> tuple[MemoryFact, ...]:
        """Same filtering, for a resolution produced elsewhere (see L8.15).

        Split out so an alternative resolver can be measured without copying the
        isolation rule: whatever decides *which* episode, the rule that a turn
        only ever sees that episode's facts stays in one place.
        """
        if resolution.decision != "CONTINUE_EPISODE" or resolution.episode_id is None:
            return ()
        episode = self.episode(resolution.episode_id)
        if episode is None:
            return ()
        allowed = set(episode.fact_keys)
        ranked = self.memory.recall(f"{request} {episode.request}", limit=max(limit * 5, 30))
        picked = [fact for fact in ranked if fact.key in allowed]
        if not picked:
            picked = [fact for fact in self.memory.facts if fact.key in allowed]
        return tuple(picked[:limit])

    def scoped_context(self, request: str, limit: int = 6) -> str:
        facts = self.scoped_recall(request, limit)
        return "\n".join(
            f"- {fact.value} (memory:{fact.source}, uses:{fact.uses})" for fact in facts
        )

    # -- persistence -------------------------------------------------------

    def load(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        try:
            with gzip.open(self.path, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
            self.episodes = [
                Episode(
                    episode_id=item["episode_id"],
                    request=item["request"],
                    topic=item["topic"],
                    goal_state=item["goal_state"],
                    operators=tuple(item["operators"]),
                    status=item["status"],
                    achieved=item["achieved"],
                    iteration_count=item["iteration_count"],
                    artifacts=tuple(item["artifacts"]),
                    fact_keys=tuple(item["fact_keys"]),
                    names=tuple(item["names"]),
                )
                for item in payload.get("episodes", [])
                if isinstance(item, dict)
            ]
        except (OSError, EOFError, json.JSONDecodeError, TypeError, KeyError):
            self.episodes = []

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "episodes": [asdict(episode) for episode in self.episodes]}
        with gzip.open(self.path, "wt", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False)


def format_episodic_state(memory: EpisodicMemory, title: str = "Episodic State (L8.13)") -> str:
    lines = [f"## {title}", "", f"- episodes: `{len(memory.episodes)}`", f"- facts in L7.3: `{len(memory.memory.facts)}`", "", "【Episodes】"]
    for episode in memory.episodes:
        lines.append(
            f"- {episode.episode_id} [{episode.status}] `{episode.goal_state}`"
        )
        lines.append(
            f"    - names: {', '.join(episode.names) or 'none'}"
        )
        lines.append(
            f"    - artifacts: {', '.join(episode.artifacts) or 'none'} / facts: {len(episode.fact_keys)}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# experiment
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ProbeCase:
    request: str
    expected_decision: str
    expected_episode: str | None
    note: str


def _session_requests() -> tuple[tuple[str, str], ...]:
    """Five interleaved units of work: (request, final response)."""
    return (
        (
            "波動方程式を解くコードを作成して",
            "wave_solver.py を作成しました。結論: 中心差分で二階微分を離散化し CFL 条件 dt <= dx/c を満たす。",
        ),
        (
            "SU2ゲージ理論の計算コードを作成して",
            "su2_gauge.py を作成しました。結論: SU(2) の生成子はパウリ行列の半分で、Wilson ループはトレースで評価する。",
        ),
        (
            "一様磁場中で荷電粒子が運動するシミュレーションを作成して",
            "lorentz_particle.py を作成しました。結論: ローレンツ力 F = qv × B により旋回半径は mv/(qB) になる。",
        ),
        (
            "波動方程式のシミュレーションを作成して",
            "wave_simulation.py を作成しました。結論: 波動シミュレーションは CFL 条件で安定性が決まる。",
        ),
        (
            "テストを実行して",
            "テストを実行しました。終了コード: 0。判定: 226 passed。",
        ),
    )


def _probe_cases() -> tuple[ProbeCase, ...]:
    return (
        ProbeCase(
            "ナビエストークス方程式のソルバーを作成して",
            "NEW_EPISODE",
            None,
            "control: unrelated new work must not reach into any earlier episode",
        ),
        ProbeCase(
            "それを修正して",
            "CONTINUE_EPISODE",
            "ep-005",
            "bare reference is defined as the latest episode, not guessed",
        ),
        ProbeCase(
            "さっきのSU2の計算を続けて",
            "CONTINUE_EPISODE",
            "ep-002",
            "a named older episode wins over recency",
        ),
        ProbeCase(
            "さっきのSU4の計算を続けて",
            "CLARIFY",
            None,
            "control: an unknown name must not silently fall back to the latest episode",
        ),
        ProbeCase(
            "さっきの波動方程式の作業を続けて",
            "CLARIFY",
            None,
            "control: a name matching two episodes must ask instead of picking one",
        ),
        ProbeCase(
            "lorentz_particle.py を修正して",
            "CONTINUE_EPISODE",
            "ep-003",
            "an artifact filename identifies its episode",
        ),
    )


def build_session(tmp_path: Path | None = None) -> EpisodicMemory:
    from .behavior_monitor_l87_experiment import BehaviorEvent
    from .goal_execution_loop_l812_experiment import run_goal_loop, trace_with

    store = ConversationKnowledgeMemory(
        (tmp_path or Path(".")) / "_l813_facts.json.gz"
    )
    store.facts = []
    episodic = EpisodicMemory(store)

    for request, response in _session_requests():
        event = (
            BehaviorEvent.SUBPROCESS_STARTED
            if "実行" in request
            else BehaviorEvent.FILE_WRITTEN
        )

        def executor(_request: str, _directives, response=response, event=event):
            return response, trace_with(event)

        episodic.record(run_goal_loop(request, executor))
    return episodic


def run_experiment() -> dict[str, object]:
    episodic = build_session()
    results = []
    for case in _probe_cases():
        resolution = episodic.resolve_reference(case.request)
        facts = episodic.scoped_recall(case.request)
        decision_ok = resolution.decision == case.expected_decision
        episode_ok = resolution.episode_id == case.expected_episode
        results.append(
            {
                "request": case.request,
                "note": case.note,
                "decision": resolution.decision,
                "expected_decision": case.expected_decision,
                "episode": resolution.episode_id,
                "expected_episode": case.expected_episode,
                "reason": resolution.reason,
                "fact_count": len(facts),
                "decision_ok": decision_ok,
                "episode_ok": episode_ok,
            }
        )

    referencing = [item for item in results if item["expected_decision"] != "NEW_EPISODE"]
    unambiguous = [item for item in referencing if item["expected_decision"] == "CONTINUE_EPISODE"]
    ambiguous = [item for item in referencing if item["expected_decision"] == "CLARIFY"]
    fresh = [item for item in results if item["expected_decision"] == "NEW_EPISODE"]

    metrics = {
        "reference_resolution_accuracy": _ratio(
            sum(1 for item in unambiguous if item["decision_ok"] and item["episode_ok"]),
            len(unambiguous),
        ),
        "wrong_episode_reuse_rate": _ratio(
            sum(
                1
                for item in referencing
                if item["episode"] is not None and item["episode"] != item["expected_episode"]
            ),
            len(referencing),
        ),
        "stale_intent_leak_rate": _ratio(
            sum(1 for item in fresh if item["fact_count"] > 0), len(fresh)
        ),
        "clarification_accuracy": _ratio(
            sum(1 for item in ambiguous if item["decision_ok"]), len(ambiguous)
        ),
        "unnecessary_clarification_rate": _ratio(
            sum(1 for item in unambiguous + fresh if item["decision"] == "CLARIFY"),
            len(unambiguous) + len(fresh),
        ),
    }
    return {"episodes": episodic, "results": results, "metrics": metrics}


def _ratio(numerator: int, denominator: int) -> float | str:
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def format_experiment(report: dict[str, object]) -> str:
    lines = ["## Episodic State Memory Experiment (L8.13)", "", "【Probes】"]
    for item in report["results"]:
        mark = "OK" if item["decision_ok"] and item["episode_ok"] else "NG"
        target = item["episode"] or "-"
        lines.append(
            f"- [{mark}] `{item['request']}` -> {item['decision']} ({target}), facts={item['fact_count']}"
        )
        lines.append(f"    - {item['note']}")
    lines.extend(["", "【Metrics】"])
    for key, value in report["metrics"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - manual entry point
    report = run_experiment()
    print(format_episodic_state(report["episodes"]))
    print()
    print(format_experiment(report))


if __name__ == "__main__":  # pragma: no cover
    main()
