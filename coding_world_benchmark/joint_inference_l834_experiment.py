"""L8.34: two unknown words at once, and why solving them one at a time is wrong.

Every layer from L8.27 has assumed one unknown surface.  Drop that and a request
like 「フガ率がホゲ尺より大きいケースの平均を出して」 breaks the assumption twice over.
Solving each word separately throws away the thing that carries most of the
information: **the relation between them is itself evidence.**  「より大きい」 is not
decoration around two independent blanks; with attested ranges of
runtime 8.0--51.75, success_rate 0.40--0.75 and energy_error 0.001--0.024 it is a
strict order, and it rules out three of the six ordered pairs before a single
word is looked at.

So the belief is over assignments rather than over words::

    B = { (m_1, m_2) in C_1 x C_2 : types, relations and observations all hold }

and the safety rule that follows is the one this project keeps rediscovering in
new clothes.  **A unique marginal is not a unique assignment.**  When the belief
is::

    { (runtime, success), (success, runtime) }

both surfaces have the same candidate set, every word is "narrowed to two", and
nothing is known -- the correspondence is exactly what is undetermined.  Reading
the marginals would report progress and execute the wrong program.  So execution
waits for ``|B| == 1``, which is L8.21's uniqueness rule lifted from one hole to
a tuple of them, and ``MARGINAL_ONLY`` exists as a state so the difference can be
reported rather than silently rounded off.

The gain is real and not merely tidy.  Two surfaces whose evidence leaves each of
them ambiguous between the same two fields are jointly determined the moment an
order relation holds between them, because only one of the two orderings is
possible in the data.  Neither word could be resolved alone; the pair resolves.
``independent_vs_joint_gain`` measures exactly that, and the hold-out is built so
that a solver which merely ran L8.29 twice would score zero on it.

Questions follow the same move.  L8.32 planned over one candidate set; here the
plan is over the joint, and a **relational** question -- 「フガ率とホゲ尺は別の指標
ですか？」 -- is a candidate alongside the per-word ones.  One question about the
pair can be worth more than one question about each, and whether it is depends on
the world rather than on a preference, so the planner is left to find out.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .active_observation_l832_experiment import answer_as_observation
from .cross_context_identification_l829_experiment import (
    FieldFacts, Observation, available_probes, constrain,
)
from .typed_operation_l819_experiment import ValueType

JOINT_RESOLVED = "JOINT_RESOLVED"
#: Every surface narrowed to one candidate set, and the correspondence between
#: them still open.  The state exists so this cannot be mistaken for progress.
MARGINAL_ONLY = "MARGINAL_ONLY"
UNRESOLVED = "UNRESOLVED"
CONTRADICTION = "CONTRADICTION"

DISTINCT, EXCEEDS, SAME_TYPE = "DISTINCT", "EXCEEDS", "SAME_TYPE"


@dataclass(frozen=True)
class Relation:
    """A constraint between two surfaces, read off the request's syntax."""

    kind: str
    left: int
    right: int
    note: str = ""


def _facts(universe: tuple[FieldFacts, ...], name: str) -> FieldFacts:
    return next(facts for facts in universe if facts.name == name)


def holds(relation: Relation, assignment: tuple[str, ...],
          universe: tuple[FieldFacts, ...]) -> bool:
    left, right = assignment[relation.left], assignment[relation.right]
    if relation.kind == DISTINCT:
        return left != right
    if relation.kind == SAME_TYPE:
        return _facts(universe, left).value_type is _facts(universe, right).value_type
    if relation.kind == EXCEEDS:
        # Can the left one ever be larger than the right one in the recorded
        # data?  With disjoint attested ranges this is a strict order and rules
        # out half the pairs; the ranges come from the store, not a vocabulary.
        one, other = _facts(universe, left), _facts(universe, right)
        if one.high is None or other.low is None:
            return False
        return one.high > other.low and left != right
    raise ValueError(f"unknown relation: {relation.kind}")


@dataclass(frozen=True)
class JointBelief:
    surfaces: tuple[str, ...]
    assignments: frozenset[tuple[str, ...]]
    universe: tuple[FieldFacts, ...]

    @classmethod
    def prior(cls, surfaces: tuple[str, ...],
              universe: tuple[FieldFacts, ...]) -> "JointBelief":
        names = [facts.name for facts in universe]
        return cls(surfaces,
                   frozenset(product(names, repeat=len(surfaces))), universe)

    def marginal(self, index: int) -> frozenset[str]:
        return frozenset(assignment[index] for assignment in self.assignments)

    @property
    def marginals(self) -> tuple[frozenset[str], ...]:
        return tuple(self.marginal(index) for index in range(len(self.surfaces)))

    @property
    def state(self) -> str:
        if not self.assignments:
            return CONTRADICTION
        if len(self.assignments) == 1:
            return JOINT_RESOLVED
        if all(len(m) == 1 for m in self.marginals):
            # Cannot actually happen with a product space, but the check is kept
            # because a future constraint kind could produce it and the wrong
            # answer here is the dangerous one.
            return JOINT_RESOLVED
        if all(len(m) < len(self.universe) for m in self.marginals):
            return MARGINAL_ONLY
        return UNRESOLVED

    @property
    def meanings(self) -> dict[str, str] | None:
        if self.state != JOINT_RESOLVED:
            return None
        assignment = next(iter(self.assignments))
        return dict(zip(self.surfaces, assignment))

    def observe(self, index: int, observation: Observation) -> "JointBelief":
        admitted = constrain(observation, self.universe)
        return JointBelief(self.surfaces, frozenset(
            a for a in self.assignments if a[index] in admitted), self.universe)

    def relate(self, relation: Relation) -> "JointBelief":
        return JointBelief(self.surfaces, frozenset(
            a for a in self.assignments if holds(relation, a, self.universe)),
            self.universe)


# --------------------------------------------------------------------------
# reading the relations out of the request
# --------------------------------------------------------------------------

_ORDER_CUES = ("より大きい", "より高い", "より長い", "を上回る", "より多い")
_DISTINCT_CUES = ("と", "および", "それぞれ")


def relations_from_request(request: str, surfaces: tuple[str, ...]) -> tuple[Relation, ...]:
    """Which constraints the sentence asserts between the unknown surfaces.

    Syntax only: the cues say that an order or a contrast is being drawn, and
    which way round, while what the words mean stays entirely unknown.  Two
    surfaces compared at all must be comparable, so SAME_TYPE is implied by the
    comparison rather than asserted separately.
    """
    if len(surfaces) < 2:
        return ()
    found: list[Relation] = []
    first, second = request.find(surfaces[0]), request.find(surfaces[1])
    if first < 0 or second < 0:
        return ()
    left, right = (0, 1) if first < second else (1, 0)
    between = request[min(first, second):max(first, second)]
    tail = request[max(first, second):]
    if any(cue in tail or cue in between for cue in _ORDER_CUES):
        found.append(Relation(SAME_TYPE, left, right, "compared at all"))
        found.append(Relation(EXCEEDS, left, right, "「より大きい」"))
    elif any(cue in between for cue in _DISTINCT_CUES):
        found.append(Relation(SAME_TYPE, left, right, "listed together"))
        found.append(Relation(DISTINCT, left, right, "two things named at once"))
    return tuple(found)


# --------------------------------------------------------------------------
# independent inference, for comparison
# --------------------------------------------------------------------------

def independent(belief: JointBelief,
                observations: dict[int, tuple[Observation, ...]]) -> tuple[frozenset[str], ...]:
    """What solving each surface on its own would leave standing.

    Deliberately ignores the relations: this is the baseline the joint solver has
    to beat, and it is what running L8.29 once per word amounts to.
    """
    names = frozenset(facts.name for facts in belief.universe)
    found = []
    for index in range(len(belief.surfaces)):
        live = names
        for observation in observations.get(index, ()):  # noqa: B910
            live &= constrain(observation, belief.universe)
        found.append(live)
    return tuple(found)


# --------------------------------------------------------------------------
# planning over the joint
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class JointQuery:
    """A question about one surface, or about the pair."""

    kind: str  # SURFACE | RELATION
    index: int | None
    observation: Observation | None
    relation: Relation | None
    text: str

    def split(self, belief: JointBelief) -> tuple[frozenset, frozenset]:
        if self.kind == "SURFACE":
            admitted = constrain(self.observation, belief.universe)
            yes = frozenset(a for a in belief.assignments if a[self.index] in admitted)
        else:
            yes = frozenset(a for a in belief.assignments
                            if holds(self.relation, a, belief.universe))
        return yes, belief.assignments - yes


def joint_queries(belief: JointBelief,
                  pool: tuple[Observation, ...]) -> tuple[JointQuery, ...]:
    found = [
        JointQuery("SURFACE", index, observation, None,
                   f"「{belief.surfaces[index]}」について {observation.kind}"
                   f"({observation.payload}) は成り立ちますか？")
        for index in range(len(belief.surfaces))
        for observation in pool
    ]
    if len(belief.surfaces) >= 2:
        found.append(JointQuery(
            "RELATION", None, None, Relation(DISTINCT, 0, 1),
            f"「{belief.surfaces[0]}」と「{belief.surfaces[1]}」は別の指標ですか？"))
        found.append(JointQuery(
            "RELATION", None, None, Relation(EXCEEDS, 0, 1),
            f"「{belief.surfaces[0]}」は「{belief.surfaces[1]}」より大きくなりえますか？"))
    return tuple(found)


def expected_size(query: JointQuery, belief: JointBelief) -> float:
    yes, no = query.split(belief)
    total = len(belief.assignments)
    if not total:
        return 0.0
    return (len(yes) ** 2 + len(no) ** 2) / total


def plan_joint(belief: JointBelief,
               pool: tuple[Observation, ...]) -> JointQuery | None:
    """The question that shrinks the *assignment* set most, whatever it asks about."""
    scored = [
        (expected_size(query, belief), index, query)
        for index, query in enumerate(joint_queries(belief, pool))
        if all(query.split(belief))
    ]
    if not scored:
        return None
    return min(scored, key=lambda item: (item[0], item[1]))[2]


def apply_answer(belief: JointBelief, query: JointQuery, answer: bool) -> JointBelief:
    if query.kind == "SURFACE":
        return belief.observe(query.index, answer_as_observation(query.observation, answer))
    yes, no = query.split(belief)
    return JointBelief(belief.surfaces, yes if answer else no, belief.universe)


def resolve(belief: JointBelief, truth: dict[str, str],
            pool: tuple[Observation, ...], budget: int = 16) -> tuple[JointBelief, int]:
    """Ask until the assignment is unique, or until nothing left would help."""
    asked = 0
    while belief.state not in (JOINT_RESOLVED, CONTRADICTION) and asked < budget:
        query = plan_joint(belief, pool)
        if query is None:
            break
        assignment = tuple(truth[surface] for surface in belief.surfaces)
        if query.kind == "SURFACE":
            answer = assignment[query.index] in constrain(query.observation, belief.universe)
        else:
            answer = holds(query.relation, assignment, belief.universe)
        belief = apply_answer(belief, query, answer)
        asked += 1
    return belief, asked
