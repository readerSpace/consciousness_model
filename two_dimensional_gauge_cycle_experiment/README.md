# 2D U(1) Closed-Cycle Discovery

This isolated, classical 2D compact U(1) lattice experiment tests whether a
raw oriented-link grammar can form and select a local four-link closed cycle.
The candidate is named only by its directional construction:
`cycle(+x,+y,-x,-y)`. The source and tests never use `plaquette` as an input
concept.

The cycle is generated beside open paths, then selected only by held-out
next-step prediction error. The report verifies its gauge invariance, its
leave-one-out predictive contribution, and frozen zero-shot transfer from
4x4/6x6 to 8x8–16x16.

```powershell
python two_dimensional_gauge_cycle_experiment/gauge_cycle_discovery_experiment.py --output two_dimensional_gauge_cycle_experiment/results.json
pytest two_dimensional_gauge_cycle_experiment -q -p no:cacheprovider
```