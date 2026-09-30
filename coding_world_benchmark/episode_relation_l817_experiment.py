"""L8.17: resolve references that name a *relation* between episodes.

L8.15.1 found the one place where the episode layer answers confidently and
wrongly.  Both failures are off-by-one and neither is a clarification:

* 「その前の実験を続けて」 -- 「前の」 is consumed as a plain continuation marker,
  nothing is left to match on, the bare-reference rule fires and the **latest**
  episode comes back instead of the one before it;
* 「Xの後にやった方を続けて」 -- X is matched as the *target* rather than as the
  anchor of a relation, so X itself comes back instead of its successor.

Similarity search cannot fix either, because neither query describes the
episode it wants.  It describes a step from one episode to another, so the
answer has to come from an edge, not from a score.

The design follows the split that L8.16 arrived at.  There, a request for exact
wording was found to name a thing *and* an aspect of it, and resolution only
worked once the aspect was taken out of the query.  The same split applies one
level up: a relational request names an operator and an operand, so the
relation phrase is removed before the anchor is resolved, and the two are
answered by different machinery.

The safety property is structural rather than scored.  Once a relation phrase
is recognised the bare-reference rule is **disabled for that query**, so the
path that produced both wrong answers no longer exists; a traversal that lands
on nothing, or on more than one episode, returns CLARIFY.  This is asserted
directly in the tests rather than inferred from a measured rate, because a rate
of zero is a property of the probes and the whole point of L8.14 was that such
a rate can be an accident of the query distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .episodic_memory_l813_experiment import Episode, EpisodicMemory, ReferenceResolution
from .structural_resolver_l815_experiment import (
    episode_tokens,
    query_tokens,
    resolve_reference_structural,
)


class RelationKind(Enum):
    PRECEDING = "PRECEDING"
    FOLLOWING = "FOLLOWING"
    FIRST = "FIRST"
    LAST = "LAST"
    REVISION = "REVISION"
    REUSE = "REUSE"
    FAILED = "FAILED"


#: Longest phrase first, so 「その前の」 is not shadowed by 「前の」 and
#: 「の後にやった」 is not shadowed by 「の後の」.  A closed list, and the part of
#: this layer that needs maintaining -- but unlike a stopword list a miss here
#: is safe: an unrecognised relation falls through to L8.15, which answers with
#: a clarification rather than a guess.
_RELATION_PHRASES: tuple[tuple[str, RelationKind], ...] = (
    ("うまくいかなかった方", RelationKind.FAILED),
    ("の結果を使った次の", RelationKind.REUSE),
    ("の結果を使った", RelationKind.REUSE),
    ("の後にやった方", RelationKind.FOLLOWING),
    ("の後にやった", RelationKind.FOLLOWING),
    ("一番最初", RelationKind.FIRST),
    ("一番最後", RelationKind.LAST),
    ("最初にやった", RelationKind.FIRST),
    ("最後にやった", RelationKind.LAST),
    ("失敗した方", RelationKind.FAILED),
    ("を直した方", RelationKind.REVISION),
    ("修正した方", RelationKind.REVISION),
    ("の修正版", RelationKind.REVISION),
    ("一つ前の", RelationKind.PRECEDING),
    ("ひとつ前の", RelationKind.PRECEDING),
    ("その前の", RelationKind.PRECEDING),
    ("その次の", RelationKind.FOLLOWING),
    ("次にやった", RelationKind.FOLLOWING),
    ("直前の", RelationKind.PRECEDING),
    ("前回の", RelationKind.PRECEDING),
    ("最初の", RelationKind.FIRST),
    ("の後の", RelationKind.FOLLOWING),
)

#: Broader than the phrase table on purpose.  An unrecognised relation is only
#: safe while it cannot reach the bare-reference rule: 「二つ前の実験を続けて」 parses
#: to no relation, leaves no content token, and 「前の」 is a continuation marker,
#: so without this guard it would resolve to the *latest* episode -- the same
#: confidently-wrong shape L8.15.1 found, one step further out.  When a cue is
#: present, no relation could be parsed, and nothing else is left to match on,
#: the answer is a clarification.  A query that still carries content tokens is
#: untouched, because L8.15 then either grounds it or clarifies, and both are safe.
_RELATION_CUES: tuple[str, ...] = (
    "前", "次", "後", "最初", "最後", "古い", "新しい", "手前", "以前", "直後", "直前", "先に", "昔",
)

#: Relations that mean nothing without an anchor.  PRECEDING is not among them:
#: 「その前の」 with no anchor is the episode before the latest, which is what a
#: reader understands and is exactly the case L8.15 got wrong.
_ANCHOR_REQUIRED = frozenset(
    {RelationKind.FOLLOWING, RelationKind.REVISION, RelationKind.REUSE}
)


@dataclass(frozen=True)
class RelationEdge:
    source: str
    relation: str
    target: str


@dataclass(frozen=True)
class EpisodeRelationGraph:
    """Edges derived from the episodes themselves, never hand-authored."""

    episodes: tuple[Episode, ...]
    edges: tuple[RelationEdge, ...]

    def index_of(self, episode_id: str) -> int | None:
        for position, episode in enumerate(self.episodes):
            if episode.episode_id == episode_id:
                return position
        return None

    def targets(self, source: str, relation: str) -> tuple[str, ...]:
        return tuple(
            edge.target for edge in self.edges if edge.source == source and edge.relation == relation
        )


def build_relation_graph(episodic: EpisodicMemory) -> EpisodeRelationGraph:
    episodes = tuple(episodic.episodes)
    edges: list[RelationEdge] = []

    for position in range(len(episodes) - 1):
        edges.append(
            RelationEdge(episodes[position].episode_id, "followed_by", episodes[position + 1].episode_id)
        )

    # A later episode that names an earlier one's artifact is working on it.
    # CHANGE in the later goal makes it a revision; otherwise it merely reuses.
    for earlier_position, earlier in enumerate(episodes):
        artifacts = {item.lower() for item in earlier.artifacts}
        if not artifacts:
            continue
        for later in episodes[earlier_position + 1:]:
            mentioned = artifacts & set(episode_tokens(later))
            if not mentioned:
                continue
            relation = "revises" if "CHANGE" in later.operators else "reuses"
            edges.append(RelationEdge(earlier.episode_id, relation, later.episode_id))

    return EpisodeRelationGraph(episodes, tuple(edges))


def parse_relation(request: str) -> tuple[RelationKind | None, str]:
    """Split the relation off the anchor, longest phrase first."""
    for phrase, kind in _RELATION_PHRASES:
        if phrase in request:
            return kind, request.replace(phrase, "", 1)
    return None, request


def _clarify(reason: str, candidates: tuple[str, ...] = ()) -> ReferenceResolution:
    return ReferenceResolution("CLARIFY", None, candidates, reason)


def _continue(episode_id: str, reason: str) -> ReferenceResolution:
    return ReferenceResolution("CONTINUE_EPISODE", episode_id, (episode_id,), reason)


def traverse(
    graph: EpisodeRelationGraph, kind: RelationKind, anchor: str | None
) -> ReferenceResolution:
    episodes = graph.episodes
    if not episodes:
        return _clarify("no episodes to traverse")

    if kind is RelationKind.FIRST:
        return _continue(episodes[0].episode_id, "first episode of the session")
    if kind is RelationKind.LAST:
        return _continue(episodes[-1].episode_id, "most recent episode")

    if kind is RelationKind.FAILED:
        failed = [episode.episode_id for episode in episodes if not episode.achieved]
        if len(failed) == 1:
            return _continue(failed[0], "the one episode that did not reach its goal")
        return _clarify(
            "no unfinished episode" if not failed else "several episodes did not reach their goal",
            tuple(failed),
        )

    if anchor is None:
        if kind in _ANCHOR_REQUIRED:
            return _clarify(f"{kind.value} needs an anchor episode and none was named")
        anchor = episodes[-1].episode_id

    position = graph.index_of(anchor)
    if position is None:
        return _clarify("anchor episode is not in the graph")

    if kind is RelationKind.PRECEDING:
        if position == 0:
            return _clarify("nothing precedes the anchor")
        return _continue(episodes[position - 1].episode_id, f"episode before {anchor}")

    if kind is RelationKind.FOLLOWING:
        if position >= len(episodes) - 1:
            return _clarify("nothing follows the anchor")
        return _continue(episodes[position + 1].episode_id, f"episode after {anchor}")

    relation = "revises" if kind is RelationKind.REVISION else "reuses"
    targets = graph.targets(anchor, relation)
    if len(targets) == 1:
        return _continue(targets[0], f"{relation} edge from {anchor}")
    return _clarify(
        f"no {relation} edge from {anchor}" if not targets else f"several {relation} edges from {anchor}",
        targets,
    )


def resolve_reference_relational(episodic: EpisodicMemory, request: str) -> ReferenceResolution:
    """L8.15 plus relation traversal.  Drop-in for the same signature."""
    kind, remainder = parse_relation(request)
    if kind is None:
        if any(cue in request for cue in _RELATION_CUES) and not query_tokens(request):
            return _clarify(
                "looks like a relation between episodes but none could be parsed, "
                "and nothing else identifies an episode"
            )
        return resolve_reference_structural(episodic, request)

    anchor: str | None = None
    if query_tokens(remainder):
        anchored = resolve_reference_structural(episodic, remainder)
        if anchored.decision == "CONTINUE_EPISODE":
            anchor = anchored.episode_id
        elif anchored.decision == "CLARIFY":
            # The anchor itself is ambiguous; stepping from an unknown place
            # would be a guess.  Note that the bare-reference rule is *not*
            # reachable from here -- that is the fix.
            return _clarify(
                f"{kind.value} requested but the anchor is ambiguous", anchored.candidates
            )

    return traverse(build_relation_graph(episodic), kind, anchor)
