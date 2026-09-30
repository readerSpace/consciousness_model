# Causal Paranormal Intervention Experiment

This is a Phase-1 symbolic benchmark for hidden-cause validation.

Each case stores:

- `ground_truth`: the simulator-only true cause and latent factors.
- `initial_observations`: the only initial evidence shown to the agent.
- `available_actions`: interventions the agent may request.
- `action_outcomes`: simulator responses, hidden until the matching action is taken.
- `evaluation`: labels and minimum identifying interventions.

The benchmark checks whether a finite-workspace causal agent can avoid direct
surface matching, select informative interventions by expected information
gain, identify known causes, and say `UNEXPLAINED` when the generator itself
marks the cause as unknown.

Run:

```powershell
python .\causal_paranormal_intervention_experiment\paranormal_intervention_experiment.py --output .\causal_paranormal_intervention_experiment\results.json
python .\causal_paranormal_intervention_experiment\systematic_holdout_experiment.py --output .\causal_paranormal_intervention_experiment\systematic_holdout_results.json
python -m pytest .\causal_paranormal_intervention_experiment
```

`systematic_holdout_experiment.py` adds Experiment 2.  It measures random,
appearance, context, and factor-combination hold-outs across causal EIG with
shared concepts, random intervention, EIG without concept sharing, nearest
neighbor, and majority baselines.  It also repeats the factor-combination test
after replacing observation names with opaque `obs_###` tokens.
