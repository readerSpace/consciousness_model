# Closed-Loop Abstraction Experiment

This experiment removes named cycle candidates. It starts with four direction
primitives and `compose` only, enumerates non-backtracking four-step paths,
and derives a `closed_loop` category only after finding repeated paths with
zero endpoint displacement.

The 2D compact U(1) reference is used to test whether a selected derived path
improves prediction and whether the derived category predicts invariance under
random local U(1) gauge transformations. It is a classical representation
experiment, not a quantum gauge-theory simulation.

```powershell
python closed_loop_abstraction_experiment/closed_loop_abstraction_experiment.py --output closed_loop_abstraction_experiment/results.json
pytest closed_loop_abstraction_experiment -q -p no:cacheprovider
```