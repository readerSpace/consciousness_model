"""Toy simulation: how concepts form from observations in consciousness_io.

A robot encounters objects one at a time.  Each observation is a relation
``(color:category, afford, action)`` such as ``("red:ball","afford","roll")``.
Colours vary; the category + affordance are stable.

We feed the growing observation history into the compression engine that
consciousness_io uses to build Z_t (``CompressedWorkspace``) and, at every step,
record every concept it holds — its kind, how many observations it compresses
(support), its utility score, and whether it was selected into the finite
workspace.  The point to *see*: as soon as two differently-coloured instances of
the same category appear, an abstract concept ``*:ball afford roll`` is born and,
with more evidence, overtakes the specific episodes.

Run:  python examples/concept_formation_sim.py            # prints a text trace
      python examples/concept_formation_sim.py --json     # emits the step log
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from consciousness_io import CompressedWorkspace


# One observation per step (the robot's experience stream).
STREAM = [
    ("red:ball", "afford", "roll"),
    ("blue:ball", "afford", "roll"),
    ("green:ball", "afford", "roll"),
    ("red:box", "afford", "stack"),
    ("blue:box", "afford", "stack"),
    ("red:ball", "afford", "roll"),      # a repeat -> consolidation (merge)
    ("yellow:box", "afford", "stack"),
    ("green:box", "afford", "stack"),
    ("purple:ball", "afford", "roll"),
]

KIND_LABEL = {
    "episode": "具体観測 (episode)",
    "merge": "重複統合 (merge)",
    "abstract": "抽象概念 (abstract)",
    "chunk": "系列 (chunk)",
    "compose": "推移合成 (compose)",
}


def snapshot(capacity: int = 6):
    """Re-compress the whole history each step (fresh workspace, exact support)."""
    log = []
    buffer = []
    for step, relation in enumerate(STREAM, start=1):
        buffer.append(relation)
        ws = CompressedWorkspace(capacity=capacity)
        ws.ingest(list(buffer))
        selected = set(ws.items)
        concepts = []
        for ident, c in ws.long_term.items():
            concepts.append({
                "id": ident,
                "kind": c.kind,
                "repr": c.representation,
                "support": c.support,
                "utility": round(c.utility(), 3),
                "selected": ident in selected,
            })
        concepts.sort(key=lambda d: (-d["utility"], d["id"]))
        log.append({
            "step": step,
            "observation": f"{relation[0]} {relation[1]} {relation[2]}",
            "n_observations": len(buffer),
            "concepts": concepts,
        })
    return log


def main() -> None:
    log = snapshot()
    if "--json" in sys.argv:
        print(json.dumps(log, ensure_ascii=False, indent=2))
        return
    for entry in log:
        print(f"\n=== step {entry['step']}  観測: {entry['observation']} "
              f"(累積 {entry['n_observations']} 件) ===")
        for c in entry["concepts"]:
            mark = "★選抜" if c["selected"] else "      "
            print(f"  {mark}  [{KIND_LABEL.get(c['kind'], c['kind']):>18}] "
                  f"u={c['utility']:5.2f}  support={c['support']}  {c['repr']}")


if __name__ == "__main__":
    main()
