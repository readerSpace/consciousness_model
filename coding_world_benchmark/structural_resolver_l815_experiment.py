"""L8.15: resolve episode references by scoring instead of set intersection.

L8.14 measured L8.13 and separated two things that had looked like one
problem.  Isolation was intact at 100 episodes -- `stale_intent_leak_rate`,
`new_task_contamination_rate` and `wrong_episode_reuse_rate` were all 0.00 --
while phrase references collapsed from 1.00 to 0.00 once episodes began
sharing name components.  Every one of those failures was a clarification
request, never a wrong episode, so what is missing is not vocabulary but the
ability to *rank* candidates.  Adding stopwords cannot fix that, and L8.14
shows why: the resolver sees three candidates and has no reason to prefer the
one that matches every component over the two that match one.

So this layer replaces only the discrimination step:

    S(e,q) = w_a·A + w_t·T + w_g·G + w_r·R - w_c·C

with A an artifact hit, T the share of the query's content tokens the episode
covers, G goal agreement, R recency, C a specific name that belongs to some
*other* episode.  Coverage is what recovers the phrase case: an episode
matching 3/3 components outranks two episodes matching 1/3.

Two properties of L8.13 are kept as *rules*, not as scores, because L8.14
showed they were the parts that were working:

* whether a request is a reference at all is still decided structurally, so a
  new task can never be scored into continuing an old episode;
* a reference carrying no content tokens still means the latest episode,
  since that is the one case where recency *is* the meaning.

And the top candidate is not simply taken.  ``CONTINUE`` requires both
``S1 >= THRESHOLD_ABSOLUTE`` and ``S1 - S2 >= THRESHOLD_MARGIN``; otherwise the
answer stays ``CLARIFY``.  Without the margin a resolver would answer
"2次元SU2 or 3次元SU2?" by picking whichever is more recent, which is the
silent substitution the whole episode layer exists to prevent.

Weights and thresholds are tuned against L8.14, which makes L8.14 development
data.  The honest measurement of this resolver is the cold hold-out in L8.15.1.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from .episodic_memory_l813_experiment import (
    Episode,
    EpisodicMemory,
    ReferenceResolution,
    _REFERENCE_STOPWORDS,
    _is_specific_name,
    has_continuation_marker,
)
from .goal_semantics_l89_experiment import extract_goal

WEIGHT_ARTIFACT = 3.0
WEIGHT_TOKEN_COVERAGE = 2.0
WEIGHT_GOAL = 0.2
WEIGHT_RECENCY = 0.15
WEIGHT_CONFLICT = 4.0

THRESHOLD_ABSOLUTE = 0.8
THRESHOLD_MARGIN = 0.4

#: The weights are not free.  Goal agreement and recency are *circumstantial*:
#: they say an episode looks plausible, not that the user named it.  Their
#: combined spread is held below the margin so that neither, nor both together,
#: can ever lift a candidate over the threshold on its own -- only an artifact
#: hit or token coverage can license CONTINUE, and circumstance merely orders
#: candidates that evidence already separated.  This is what stops "さっき" from
#: quietly deciding between 二次元SU2 and 三次元SU2, and it is asserted in the
#: tests rather than left as a property of the current numbers.
assert WEIGHT_GOAL + WEIGHT_RECENCY < THRESHOLD_MARGIN

#: Adds a digit-prefixed form to L8.13's pattern so "3次元" survives as one
#: token.  L8.13 dropped the digit and kept "次元", which is precisely the
#: component that distinguishes two otherwise identical experiments.
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_-]+|[0-9０-９]+[一-龥]{2,}|[一-龥]{2,}|[ァ-ヶー]{2,}")
_FILENAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-./\\]*\.[A-Za-z0-9]{1,5}")

_KANJI_DIGITS = {"〇": "0", "一": "1", "二": "2", "三": "3", "四": "4",
                 "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
_WIDE_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def canonical(token: str) -> str:
    """Fold spellings that mean the same thing so they can match.

    Leading kanji numerals become ASCII digits, so 三次元 and 3次元 are one
    token.  Applied identically to episodes and queries, so this is a matching
    convention rather than a claim about Japanese: 一般 folds to 1般 on both
    sides and still matches itself.
    """
    index = 0
    digits = []
    while index < len(token) and token[index] in _KANJI_DIGITS:
        digits.append(_KANJI_DIGITS[token[index]])
        index += 1
    return ("".join(digits) + token[index:]).translate(_WIDE_DIGITS)


def query_tokens(text: str) -> tuple[str, ...]:
    tokens = []
    for match in _TOKEN.findall(text):
        token = match.lower() if match[0].isascii() else match
        if token in _REFERENCE_STOPWORDS or len(token) < 2:
            continue
        tokens.append(canonical(token))
    for match in _FILENAME.findall(text):
        tokens.append(canonical(match.lower()))
    return tuple(dict.fromkeys(tokens))


def episode_tokens(episode: Episode) -> frozenset[str]:
    return frozenset(canonical(name) for name in episode.names)


@dataclass(frozen=True)
class EpisodeScore:
    episode_id: str
    artifact: float
    coverage: float
    goal: float
    recency: float
    conflict: float
    score: float


def score_episodes(episodic: EpisodicMemory, request: str) -> tuple[EpisodeScore, ...]:
    tokens = query_tokens(request)
    token_set = set(tokens)
    specific = {token for token in tokens if _is_specific_name(token)}
    owners = {
        token: {
            episode.episode_id
            for episode in episodic.episodes
            if token in episode_tokens(episode)
        }
        for token in specific
    }
    try:
        query_ops = set(extract_goal(request).operators)
    except Exception:  # pragma: no cover - extract_goal is total, guard is cheap
        query_ops = set()

    total = len(episodic.episodes)
    scored = []
    for position, episode in enumerate(episodic.episodes):
        names = episode_tokens(episode)

        artifact = 1.0 if any(canonical(item.lower()) in token_set for item in episode.artifacts) else 0.0
        coverage = len(token_set & names) / len(token_set) if token_set else 0.0

        episode_ops = set(episode.operators)
        union = query_ops | {op for op in episode_ops}
        goal = (
            len({op.value for op in query_ops} & episode_ops) / len(union)
            if union
            else 0.0
        )

        recency = 1.0 / (1.0 + (total - 1 - position))

        conflict = 0.0
        for token, holders in owners.items():
            if holders and episode.episode_id not in holders:
                conflict = 1.0
                break

        score = (
            WEIGHT_ARTIFACT * artifact
            + WEIGHT_TOKEN_COVERAGE * coverage
            + WEIGHT_GOAL * goal
            + WEIGHT_RECENCY * recency
            - WEIGHT_CONFLICT * conflict
        )
        scored.append(
            EpisodeScore(episode.episode_id, artifact, coverage, goal, recency, conflict, round(score, 4))
        )
    return tuple(sorted(scored, key=lambda item: (-item.score, item.episode_id)))


def _is_reference(episodic: EpisodicMemory, request: str) -> ReferenceResolution | None:
    """Structural gate, unchanged in spirit from L8.13.

    Returns a final answer when the request is not a reference at all, and
    ``None`` when scoring should decide which episode it points at.
    """
    if not episodic.episodes:
        return ReferenceResolution("NEW_EPISODE", None, (), "no earlier episode to continue")

    marker = has_continuation_marker(request)
    tokens = query_tokens(request)
    specific = {token for token in tokens if _is_specific_name(token)}

    if specific:
        known = set()
        for episode in episodic.episodes:
            known |= episode_tokens(episode)
        if not (specific & known):
            if marker:
                return ReferenceResolution(
                    "CLARIFY",
                    None,
                    tuple(episode.episode_id for episode in episodic.episodes),
                    "named a target that no episode matches",
                )
            return ReferenceResolution("NEW_EPISODE", None, (), "a new name, not a reference")
        return None

    if not marker:
        return ReferenceResolution("NEW_EPISODE", None, (), "no continuation marker")
    return None


def resolve_reference_structural(episodic: EpisodicMemory, request: str) -> ReferenceResolution:
    gated = _is_reference(episodic, request)
    if gated is not None:
        return gated

    if not query_tokens(request):
        latest = episodic.episodes[-1]
        return ReferenceResolution(
            "CONTINUE_EPISODE", latest.episode_id, (latest.episode_id,),
            "bare reference means the latest episode",
        )

    scored = score_episodes(episodic, request)
    best = scored[0]
    runner_up = scored[1].score if len(scored) > 1 else float("-inf")
    margin = best.score - runner_up

    if best.score < THRESHOLD_ABSOLUTE:
        return ReferenceResolution(
            "CLARIFY",
            None,
            tuple(item.episode_id for item in scored[:3]),
            f"best score {best.score} below absolute threshold {THRESHOLD_ABSOLUTE}",
        )
    if margin < THRESHOLD_MARGIN:
        tied = tuple(item.episode_id for item in scored if best.score - item.score < THRESHOLD_MARGIN)
        return ReferenceResolution(
            "CLARIFY", None, tied, f"top candidates within {THRESHOLD_MARGIN} of each other"
        )
    return ReferenceResolution(
        "CONTINUE_EPISODE",
        best.episode_id,
        (best.episode_id,),
        f"score {best.score} clears runner-up by {round(margin, 4)}",
    )


def format_scores(episodic: EpisodicMemory, request: str, limit: int = 4) -> str:
    lines = [f"## Scores for `{request}`", "", "| episode | A | T | G | R | C | S |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for item in score_episodes(episodic, request)[:limit]:
        lines.append(
            f"| {item.episode_id} | {item.artifact} | {round(item.coverage, 3)} | "
            f"{round(item.goal, 3)} | {round(item.recency, 3)} | {item.conflict} | {item.score} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass  # noqa: E402

from .behavior_monitor_l87_experiment import BehaviorEvent  # noqa: E402
from .conversation_memory_l73_experiment import ConversationKnowledgeMemory  # noqa: E402
from .context_isolation_l814_experiment import (  # noqa: E402
    SCALING_SIZES,
    default_resolver,
    run_experiment as run_l814,
)
from .goal_execution_loop_l812_experiment import run_goal_loop, trace_with  # noqa: E402


@_dataclass(frozen=True)
class AmbiguityCase:
    text: str
    expected_decision: str
    expected_episode: str | None
    note: str


_AMBIGUITY_CORPUS: tuple[tuple[str, str], ...] = (
    ("su2ge12: 二次元SU2格子ゲージのコードを作成して", "su2_2d.py を作成しました。結論: 弱結合展開と一致する。"),
    ("hzb07: 三次元ハイゼンベルクのコードを作成して", "hzb_3d.py を作成しました。結論: 低温で秩序化する。"),
    ("su2ge28: 三次元SU2格子ゲージのコードを作成して", "su2_3d.py を作成しました。結論: 閉じ込め相が現れる。"),
)


def ambiguity_cases() -> tuple[AmbiguityCase, ...]:
    return (
        AmbiguityCase(
            "SU2の検証を続けて",
            "CLARIFY",
            None,
            "control: two SU2 episodes differ only by dimension, so a high score must not decide",
        ),
        AmbiguityCase(
            "3次元SU2の方を続けて",
            "CONTINUE_EPISODE",
            "ep-003",
            "the dimension disambiguates, and the ASCII spelling must match the kanji one",
        ),
        AmbiguityCase(
            "二次元SU2の方を続けて",
            "CONTINUE_EPISODE",
            "ep-001",
            "the same discrimination in the other direction, so recency cannot be what decides",
        ),
        AmbiguityCase(
            "su2_3d.py を修正して",
            "CONTINUE_EPISODE",
            "ep-003",
            "an artifact name still wins outright",
        ),
        AmbiguityCase(
            "今のとは別に新しく光格子時計のコードを作って",
            "NEW_EPISODE",
            None,
            "control: scoring must never pull a new task into an old episode",
        ),
    )


class _NullPath:
    def is_file(self) -> bool:
        return False


def build_ambiguity_memory() -> EpisodicMemory:
    store = ConversationKnowledgeMemory(_NullPath())
    store.facts = []
    episodic = EpisodicMemory(store)
    for request, response in _AMBIGUITY_CORPUS:
        def executor(_request, _directives, response=response):
            return response, trace_with(BehaviorEvent.FILE_WRITTEN)

        episodic.record(run_goal_loop(request, executor))
    return episodic


def run_ambiguity_controls() -> dict[str, object]:
    episodic = build_ambiguity_memory()
    rows = []
    for case in ambiguity_cases():
        for label, resolver in (("L8.13", default_resolver), ("L8.15", resolve_reference_structural)):
            resolution = resolver(episodic, case.text)
            rows.append(
                {
                    "resolver": label,
                    "text": case.text,
                    "note": case.note,
                    "decision": resolution.decision,
                    "episode": resolution.episode_id,
                    "expected_decision": case.expected_decision,
                    "expected_episode": case.expected_episode,
                    "correct": resolution.decision == case.expected_decision
                    and resolution.episode_id == case.expected_episode,
                }
            )
    return {"memory": episodic, "rows": rows}


def run_experiment(sizes: tuple[int, ...] = SCALING_SIZES) -> dict[str, object]:
    return {
        "baseline": run_l814(sizes=sizes, resolver=default_resolver),
        "structural": run_l814(sizes=sizes, resolver=resolve_reference_structural),
        "ambiguity": run_ambiguity_controls(),
    }


def _row(run: dict[str, object]) -> str:
    metric = run["metrics"]
    return "| {n} | {a} | {p} | {w} | {s} | {c} | {u} |".format(
        n=run["episode_count"],
        a=metric["reference_resolution_accuracy"],
        p=run["per_kind_accuracy"]["ENTITY_PHRASE"],
        w=metric["wrong_episode_reuse_rate"],
        s=metric["stale_intent_leak_rate"],
        c=metric["new_task_contamination_rate"],
        u=metric["unnecessary_clarification_rate"],
    )


def format_experiment(report: dict[str, object]) -> str:
    header = "| N | ref_acc | phrase_acc | wrong_reuse | stale_leak | new_task_contam | unnecessary_clarify |"
    divider = "| --- | --- | --- | --- | --- | --- | --- |"
    lines = ["## Structural Episode Resolver (L8.15)", "", "### L8.13 (baseline)", "", header, divider]
    lines += [_row(run) for run in report["baseline"]["runs"]]
    lines += ["", "### L8.15 (scored, margin-gated)", "", header, divider]
    lines += [_row(run) for run in report["structural"]["runs"]]

    lines += ["", "### Ambiguity controls", ""]
    for row in report["ambiguity"]["rows"]:
        mark = "OK" if row["correct"] else "NG"
        target = row["episode"] or "-"
        lines.append(f"- [{mark}] {row['resolver']} `{row['text']}` -> {row['decision']} ({target})")
    lines += ["", "### D(N) = A(2) - A(N)", ""]
    for label in ("baseline", "structural"):
        values = report[label]["episode_scaling_degradation"]
        rendered = ", ".join(f"N={size}: {value}" for size, value in values.items())
        lines.append(f"- {label}: {rendered}")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - manual entry point
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
