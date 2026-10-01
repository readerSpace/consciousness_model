"""exp603 -- Abstraction Retention Audit.

Previously invented O1 and O2 are optional language definitions.  This module
charges their maintained definitions and exactly selects retain/delete subsets
by L(D) + sum L(task | D).  O2 is dependency-aware: deleting O1 requires O2 to
be stored in compiled, self-contained form rather than hiding O1 for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import cumulative_instruction_bootstrapping as B


O1_BITS = 14
O2_WITH_O1_BITS = 18
O2_COMPILED_BITS = 36


@dataclass(frozen=True)
class RetentionState:
    name: str
    keep_o1: bool
    keep_o2: bool

    def history(self) -> B.History:
        if self.keep_o1 and self.keep_o2:
            return B.USEFUL_O1_O2
        if self.keep_o1:
            return B.USEFUL_O1
        if self.keep_o2:
            return B.History("compiled_O2_only", (B.O2,), O2_COMPILED_BITS)
        return B.BASE

    def language_bits(self) -> int:
        if self.keep_o1 and self.keep_o2:
            return O1_BITS + O2_WITH_O1_BITS
        if self.keep_o1:
            return O1_BITS
        if self.keep_o2:
            return O2_COMPILED_BITS
        return 0


RESET = RetentionState("reset", False, False)
KEEP_O1 = RetentionState("delete_O2", True, False)
KEEP_O2 = RetentionState("delete_O1_compiled_O2", False, True)
FULL_HISTORY = RetentionState("full_history", True, True)
STATES = (RESET, KEEP_O1, KEEP_O2, FULL_HISTORY)


def task_cost(arity: int, state: RetentionState) -> Tuple[int, B.Expr]:
    return B.semantic_complexity(arity, state.history())


def describe_workload(arities: Iterable[int], state: RetentionState) -> Dict[str, object]:
    costs = [task_cost(arity, state)[0] for arity in arities]
    return {
        "state": state.name,
        "retains": [name for name, keep in (("O1", state.keep_o1), ("O2", state.keep_o2)) if keep],
        "L_language": state.language_bits(),
        "task_costs": costs,
        "L_tasks_given_D": sum(costs),
        "L_total": state.language_bits() + sum(costs),
    }


def audit_workload(arities: Tuple[int, ...]) -> Dict[str, object]:
    """Evaluate all retain/delete choices; free choice is their exact MDL argmin."""
    conditions = {state.name: describe_workload(arities, state) for state in STATES}
    chosen = min(conditions.values(), key=lambda row: row["L_total"])
    return {"arities": list(arities), "conditions": conditions, "free_choice": chosen}


def retention_trajectory() -> Dict[str, Dict[str, object]]:
    """One O3 favors pruning O2; a three-task O3 family amortizes O2 retention."""
    return {
        "single_O3": audit_workload((8,)),
        "O3_family_three": audit_workload((8, 8, 8)),
    }


def operation_accounting() -> Dict[str, int]:
    """Expose all maintenance costs for audit/reporting."""
    return {"O1": O1_BITS, "O2_given_O1": O2_WITH_O1_BITS,
            "O2_compiled_without_O1": O2_COMPILED_BITS}