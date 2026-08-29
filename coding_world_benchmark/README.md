# Coding World Benchmark

`coding_world_benchmark.py` creates controlled, isolated Python repositories for C2-style bug repair. Every trial edits a real source file and runs `unittest` for visible and hidden tests.

Run the benchmark from the repository root:

```powershell
python -m coding_world_benchmark.coding_world_benchmark
python -m pytest coding_world_benchmark
```

Conditions B--G compare a simple ReAct policy, the full closed-loop policy, and removal of failure memory, exploration, re-observation, or candidate suppression. The report records success, hidden/regression testing, repair iterations, file and test activity, failure recovery rate, and hypothesis efficiency.

The bundled policies are deterministic operationalizations of the experimental conditions, not LLMs. Use `save_external_result` to store Codex or same-model ON/OFF trial records in the same schema; do not interpret an absent external run as a Codex comparison.

`coding_dialogue.py` integrates `portable_japanese_dialogue` as a language interface. It turns Japanese instructions into inspectable coding goals, asks a clarification question for ambiguous deletion-history requirements, remembers change records for references such as "さっきの修正", and produces reports only from recorded execution results.

`autonomous_repair_experiment.py` adds the L1--L5 scaffolding experiment. Its L2 trials provide only Japanese task text, a repository path, and test execution: the transparent baseline scans source files, generates file-specific hypotheses and concrete edits, then validates them with `unittest`. The six hold-out domains are cache, inventory, parser, filesystem, scheduler, and session. This is evidence for observation-driven hypothesis generation within a deliberately narrow repair grammar, not evidence of general Codex-level repository understanding.

`heterogeneous_l3_experiment.py` is the L3 test. It holds the post-operation state contract constant while varying the implementation among dictionary, list, SQLite, JSON, dataclass, and cache-invalidation representations. It compares an observation-driven state-transition agent with an L2-only `get` to `pop` pattern baseline, and records hidden/regression pass rates, files read, irrelevant reads, generated/rejected hypotheses, attempted edits, and test runs.

`operation_discovery_l4_experiment.py` tests three contract types: removal, update visibility, and duplicate prevention. It derives candidate mutations from source and test observations, selects with real tests, and compares a rule baseline with closed-loop and recovery ablations. Its mutation grammar remains finite and explicit, so it measures operation discovery within that grammar rather than unrestricted repair invention; failure atomicity and transaction rollback remain hold-out contract types for a later expansion.

`l45_holdout_experiment.py` is an adversarial grammar-boundary test. Its independent seeded generator creates 100 repositories spanning transaction rollback, cache/persistent coherence, idempotency, and resource cleanup. The current L4 candidate generator receives only the repository and tests. A result of zero coverage and zero repair success is the expected falsification result: these contracts require operations outside the current finite grammar and must not be presented as L4 generalization.

`operator_invention_l46_experiment.py` tests L4.6 operator composition. A primitive composer derives `SAVE_STATE -> TRY -> RESTORE_STATE -> RAISE` from an observed mutation followed by a failure, records the successful composition as `ROLLBACK`, and reuses it in a hold-out repository with different state and parameter names. It compares acquisition and reuse costs against a no-operator baseline. This is one compositional operator family, not unrestricted operator invention.

`compositional_operator_l47_experiment.py` is the L4.7 compositional-search test. One composer receives a general primitive vocabulary -- `OPERATION`, `EFFECT`, `RESULT`, `TRY`, `EXCEPT`, `FINALLY`, `END`, `GUARD`, `REPEAT`, `RAISE`, `SAVE`, `RESTORE`, `READ`, `WRITE`, `REMOVE` plus the deliberately useless `LOG`, `SLEEP`, and `TOUCH` -- and no repair rule for any contract family. For each of four unseen contracts (transaction rollback, cache/persistent coherence, guaranteed resource cleanup, idempotency) it searches compositions of those primitives under a bounded grammar, selects only with real contract tests, compresses the successful trace into an operator whose arguments are stored as observation roles rather than concrete names, and reuses it. `search_is_family_agnostic` re-checks mechanically that no search, abstraction, or reuse function mentions a family name; operator labels such as `ROLLBACK` are attached post hoc from the discovered primitive signature and never steer the search.

The report records, per family, the discovered signature, search trials, the first rejected candidates, hold-out reuse cost, MDL gain, and re-verification of the accepted program by a real `unittest` process. It ablates compression with four conditions on the same eight hold-outs: `full`, `parameterized_operator`, `memorize_exact` (replay the first successful trace verbatim), and `no_compression` (search from primitives every time). A structural hold-out re-tests the rollback operator on a three-file repository whose state is an object attribute mutated through a helper. The composition grammar forbids repeating an identical primitive instance in one program, bounds program length, and bounds loop iterations, so this measures composition search inside a bounded general grammar, not unrestricted program synthesis.

```powershell
python -m coding_world_benchmark.compositional_operator_l47_experiment
```
