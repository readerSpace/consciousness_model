"""Offline checks for the ablation summariser."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_ablation import collect, render, summarise_leaderboard, summarise_steps  # noqa: E402


def step(admitted, dropped, ttc_admitted, ttc_true, goal='CRUISE', risk=0.0, missed=()):
    return {'admitted': admitted, 'dropped': dropped, 'min_ttc_admitted': ttc_admitted,
            'min_ttc_true': ttc_true, 'goal': goal, 'risk': risk, 'missed_critical': list(missed),
            'target_speed_kmh': 25.0, 'brake': 0.0}


class SummariseStepsTest(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(summarise_steps({'steps': []}), {'steps': 0})

    def test_counts_steps_where_the_worst_hazard_was_not_admitted(self):
        payload = {'config': {'selection': 'workspace', 'capacity': 2}, 'steps': [
            step(['a', 'b'], [], 2.0, 2.0),          # workspace saw the worst hazard
            step(['a', 'b'], ['c'], 4.0, 1.2),       # it did not
            step(['a'], ['c'], None, 3.0),           # nothing admitted was closing
        ]}
        summary = summarise_steps(payload)
        self.assertEqual(summary['steps'], 3)
        self.assertEqual(summary['capacity'], 2)
        self.assertAlmostEqual(summary['unseen_step_rate'], 2 / 3.0)
        self.assertAlmostEqual(summary['mean_admitted'], 5 / 3.0)

    def test_counts_miss_events_and_safety_rate(self):
        payload = {'config': {}, 'steps': [
            step(['a'], ['b'], 2.0, 2.0, goal='SAFETY', risk=0.4, missed=('b',)),
            step(['a'], [], 9.0, 9.0),
        ]}
        summary = summarise_steps(payload)
        self.assertEqual(summary['miss_events'], 1)
        self.assertAlmostEqual(summary['safety_rate'], 0.5)
        self.assertEqual(summary['final_risk'], 0.0)

    def test_summarises_runner_trace_diagnostics(self):
        first = step(['a'], [], 2.0, 2.0)
        first.update({'ego_speed': 5.0, 'brake': 0.8, 'collision': True, 'route_progress': 20.0})
        second = step(['a'], [], 4.0, 4.0)
        second.update({'ego_speed': 7.0, 'brake': 0.1, 'collision': False, 'route_progress': 35.0})
        summary = summarise_steps({'config': {}, 'steps': [first, second]})
        self.assertAlmostEqual(summary['mean_speed_ms'], 6.0)
        self.assertAlmostEqual(summary['harsh_brake_rate'], 0.5)
        self.assertEqual(summary['collision_steps'], 1)
        self.assertEqual(summary['final_route_progress'], 35.0)


class SummariseLeaderboardTest(unittest.TestCase):

    def test_reads_global_record(self):
        payload = {'_checkpoint': {'records': [{}], 'global_record': {
            'scores': {'score_composed': 41.5, 'score_route': 63.0, 'score_penalty': 0.66},
            'infractions': {'collisions_vehicle': ['a', 'b'], 'red_light': [], 'vehicle_blocked': ['x']},
        }}}
        summary = summarise_leaderboard(payload)
        self.assertEqual(summary['driving_score'], 41.5)
        self.assertEqual(summary['collisions_vehicle'], 2)
        self.assertEqual(summary['red_light'], 0)
        self.assertEqual(summary['blocked'], 1)

    def test_falls_back_to_the_first_route_record(self):
        payload = {'_checkpoint': {'records': [
            {'scores': {'score_composed': 10.0, 'score_route': 20.0, 'score_penalty': 0.5},
             'infractions': {'collisions_pedestrian': ['p']}}]}}
        summary = summarise_leaderboard(payload)
        self.assertEqual(summary['driving_score'], 10.0)
        self.assertEqual(summary['collisions_pedestrian'], 1)


class CollectTest(unittest.TestCase):

    def test_merges_both_files_per_tag(self):
        import tempfile
        directory = tempfile.mkdtemp()
        with open(os.path.join(directory, 'k4_workspace.json'), 'w') as handle:
            json.dump({'config': {'selection': 'workspace', 'capacity': 4},
                       'steps': [step(['a'], [], 5.0, 5.0)]}, handle)
        with open(os.path.join(directory, 'k4_workspace.leaderboard.json'), 'w') as handle:
            json.dump({'_checkpoint': {'records': [], 'global_record': {
                'scores': {'score_composed': 55.0, 'score_route': 80.0, 'score_penalty': 0.7},
                'infractions': {}}}}, handle)
        rows = collect(directory)
        self.assertEqual(sorted(rows), ['k4_workspace'])
        self.assertEqual(rows['k4_workspace']['driving_score'], 55.0)
        self.assertEqual(rows['k4_workspace']['capacity'], 4)
        table = render(rows)
        self.assertIn('k4_workspace', table)
        self.assertIn('driving_score', table)


if __name__ == '__main__':
    unittest.main()
