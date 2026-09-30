#!/usr/bin/env python
"""Summarise an ablation sweep into one comparison table.

Reads, for every configuration tag found in ``results/``:

* ``<tag>.leaderboard.json`` -- the leaderboard checkpoint (driving score,
  route completion, infractions),
* ``<tag>.json`` -- the per-step record written by the agent (workspace
  occupancy, risk, capacity-induced misses, goal mode).

    python verify\\analyze_ablation.py --results-dir results
"""

from __future__ import print_function

import argparse
import json
import os
import sys


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarise_steps(payload):
    """Agent-side statistics. ``payload`` is the JSON written by the agent."""
    steps = payload.get('steps', [])
    if not steps:
        return {'steps': 0}
    config = payload.get('config', {})
    admitted = [len(step.get('admitted', [])) for step in steps]
    dropped = [len(step.get('dropped', [])) for step in steps]
    missed = sum(1 for step in steps if step.get('missed_critical'))
    unseen = sum(
        1 for step in steps
        if step.get('min_ttc_true') is not None
        and (step.get('min_ttc_admitted') is None
             or step['min_ttc_admitted'] > step['min_ttc_true'] + 1e-9)
    )
    return {
        'steps': len(steps),
        'selection': config.get('selection', '?'),
        'capacity': config.get('capacity', '?'),
        'mean_admitted': mean(admitted),
        'mean_dropped': mean(dropped),
        'unseen_step_rate': unseen / float(len(steps)),
        'miss_events': missed,
        'final_risk': steps[-1].get('risk', 0.0),
        'safety_rate': sum(1 for step in steps if step.get('goal') == 'SAFETY') / float(len(steps)),
        'mean_target_kmh': mean(step.get('target_speed_kmh', 0.0) for step in steps),
        'mean_speed_ms': mean(step.get('ego_speed', step.get('speed_ms', 0.0)) for step in steps),
        'brake_rate': sum(1 for step in steps if step.get('brake', 0.0) > 0.0) / float(len(steps)),
        'harsh_brake_rate': sum(1 for step in steps if step.get('brake', 0.0) >= 0.7) / float(len(steps)),
        'final_route_progress': steps[-1].get('route_progress'),
        'collision_steps': sum(1 for step in steps if step.get('collision')),
    }


def summarise_leaderboard(payload):
    """Driving metrics from a leaderboard checkpoint file."""
    checkpoint = payload.get('_checkpoint', {})
    records = checkpoint.get('records', [])
    globals_ = checkpoint.get('global_record') or {}
    scores = globals_.get('scores') or {}
    if not scores and records:
        scores = records[0].get('scores', {})
    infractions = globals_.get('infractions') or (records[0].get('infractions', {}) if records else {})

    def count(key):
        value = infractions.get(key, [])
        return len(value) if isinstance(value, list) else value

    return {
        'routes': len(records),
        'driving_score': scores.get('score_composed'),
        'route_completion': scores.get('score_route'),
        'infraction_penalty': scores.get('score_penalty'),
        'collisions_vehicle': count('collisions_vehicle'),
        'collisions_pedestrian': count('collisions_pedestrian'),
        'collisions_layout': count('collisions_layout'),
        'red_light': count('red_light'),
        'blocked': count('vehicle_blocked'),
    }


def collect(results_dir):
    rows = {}
    if not os.path.isdir(results_dir):
        raise SystemExit('no such directory: {}'.format(results_dir))
    for name in sorted(os.listdir(results_dir)):
        if not name.endswith('.json'):
            continue
        path = os.path.join(results_dir, name)
        with open(path) as handle:
            try:
                payload = json.load(handle)
            except ValueError:
                print('skipping unreadable {}'.format(name), file=sys.stderr)
                continue
        if name.endswith('.leaderboard.json'):
            tag = name[: -len('.leaderboard.json')]
            rows.setdefault(tag, {}).update(summarise_leaderboard(payload))
        else:
            tag = name[: -len('.json')]
            rows.setdefault(tag, {}).update(summarise_steps(payload))
    return rows


COLUMNS = (
    ('selection', '{}'), ('capacity', '{}'),
    ('driving_score', '{:.1f}'), ('route_completion', '{:.1f}'), ('infraction_penalty', '{:.2f}'),
    ('collisions_vehicle', '{}'), ('collisions_pedestrian', '{}'), ('red_light', '{}'),
    ('blocked', '{}'),
    ('mean_admitted', '{:.2f}'), ('unseen_step_rate', '{:.3f}'), ('miss_events', '{}'),
    ('final_risk', '{:.3f}'), ('safety_rate', '{:.3f}'), ('mean_target_kmh', '{:.1f}'),
    ('mean_speed_ms', '{:.2f}'), ('harsh_brake_rate', '{:.3f}'), ('final_route_progress', '{:.1f}'),
    ('steps', '{}'),
)


def render(rows):
    if not rows:
        return 'no results found'
    present = [name for name, _ in COLUMNS if any(name in row for row in rows.values())]
    header = ['config'] + present
    lines = ['| ' + ' | '.join(header) + ' |',
             '| ' + ' | '.join('---' for _ in header) + ' |']
    formats = dict(COLUMNS)
    for tag in sorted(rows):
        cells = [tag]
        for name in present:
            value = rows[tag].get(name)
            if value is None:
                cells.append('-')
            else:
                try:
                    cells.append(formats[name].format(value))
                except (ValueError, TypeError):
                    cells.append(str(value))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--json', action='store_true', help='emit raw JSON instead of a table')
    args = parser.parse_args()
    rows = collect(args.results_dir)
    print(json.dumps(rows, indent=1, sort_keys=True) if args.json else render(rows))


if __name__ == '__main__':
    main()
