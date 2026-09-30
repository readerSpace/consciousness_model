"""Offline checks for the standalone runner's score bookkeeping."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runner'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_ablation import summarise_leaderboard  # noqa: E402
from scoring import (  # noqa: E402
    attach_episode_metrics,
    aggregate,
    checkpoint_payload,
    collision_class,
    episode_summary,
    penalty_from,
)


class CollisionClassTest(unittest.TestCase):

    def test_maps_actor_types(self):
        self.assertEqual(collision_class('walker.pedestrian.0021'), 'collisions_pedestrian')
        self.assertEqual(collision_class('vehicle.tesla.model3'), 'collisions_vehicle')
        self.assertEqual(collision_class('static.prop.streetbarrier'), 'collisions_layout')
        self.assertEqual(collision_class(''), 'collisions_layout')
        self.assertEqual(collision_class(None), 'collisions_layout')


class PenaltyTest(unittest.TestCase):

    def test_clean_run_is_unpenalised(self):
        self.assertEqual(penalty_from({'collisions_vehicle': 0, 'red_light': 0}), 1.0)

    def test_penalties_compound(self):
        self.assertAlmostEqual(penalty_from({'collisions_vehicle': 2}), 0.36)
        self.assertAlmostEqual(
            penalty_from({'collisions_vehicle': 1, 'red_light': 1}), 0.42)

    def test_pedestrian_collision_is_the_harshest(self):
        self.assertLess(penalty_from({'collisions_pedestrian': 1}),
                        penalty_from({'collisions_vehicle': 1}))

    def test_unknown_infraction_is_free(self):
        self.assertEqual(penalty_from({'something_else': 3}), 1.0)

    def test_rejects_negative_counts(self):
        with self.assertRaises(ValueError):
            penalty_from({'red_light': -1})


class EpisodeSummaryTest(unittest.TestCase):

    def test_score_is_completion_times_penalty(self):
        summary = episode_summary(0, 'SUCCESS', 100.0, {'collisions_vehicle': 1},
                                  120.0, 800.0, 0.2)
        self.assertAlmostEqual(summary['infraction_penalty'], 0.60)
        self.assertAlmostEqual(summary['driving_score'], 60.0)

    def test_partial_route_scales_the_score(self):
        summary = episode_summary(1, 'BLOCKED', 40.0, {}, 90.0, 300.0, 0.0)
        self.assertAlmostEqual(summary['driving_score'], 40.0)

    def test_rejects_a_completion_outside_the_percentage_range(self):
        with self.assertRaises(ValueError):
            episode_summary(0, 'SUCCESS', 120.0, {}, 1.0, 1.0, 0.0)

    def test_attach_episode_metrics_preserves_score(self):
        summary = episode_summary(0, 'SUCCESS', 100.0, {}, 10.0, 80.0, 0.0)
        enriched = attach_episode_metrics(summary, {
            'mean_speed_ms': 8.0,
            'stopped_time_s': 1.5,
            'min_ttc_true': None,
            'harsh_brake_count': 3,
        })
        self.assertEqual(enriched['driving_score'], summary['driving_score'])
        self.assertEqual(enriched['mean_speed_ms'], 8.0)
        self.assertNotIn('min_ttc_true', enriched)


class AggregateTest(unittest.TestCase):

    def episodes(self):
        return [
            episode_summary(0, 'SUCCESS', 100.0, {'collisions_vehicle': 1}, 100.0, 900.0, 0.1),
            episode_summary(1, 'BLOCKED', 50.0, {'red_light': 2}, 90.0, 400.0, 0.3),
        ]

    def test_means_and_totals(self):
        summary = aggregate(self.episodes())
        self.assertEqual(summary['routes'], 2)
        self.assertAlmostEqual(summary['route_completion'], 75.0)
        self.assertAlmostEqual(summary['success_rate'], 0.5)
        self.assertEqual(summary['infractions']['collisions_vehicle'], 1)
        self.assertEqual(summary['infractions']['red_light'], 2)

    def test_averages_runner_diagnostics_when_present(self):
        episodes = [
            attach_episode_metrics(
                episode_summary(0, 'SUCCESS', 100.0, {}, 10.0, 80.0, 0.0),
                {'mean_speed_ms': 8.0, 'stopped_time_s': 1.0, 'harsh_brake_count': 2}),
            attach_episode_metrics(
                episode_summary(1, 'SUCCESS', 100.0, {}, 10.0, 60.0, 0.0),
                {'mean_speed_ms': 6.0, 'stopped_time_s': 3.0, 'harsh_brake_count': 4}),
        ]
        summary = aggregate(episodes)
        self.assertAlmostEqual(summary['mean_speed_ms'], 7.0)
        self.assertAlmostEqual(summary['stopped_time_s'], 2.0)
        self.assertAlmostEqual(summary['harsh_brake_count'], 3.0)

    def test_rejects_an_empty_sweep(self):
        with self.assertRaises(ValueError):
            aggregate([])


class CheckpointPayloadTest(unittest.TestCase):

    def test_the_analyser_reads_runner_output(self):
        episodes = [episode_summary(0, 'SUCCESS', 100.0, {'collisions_pedestrian': 1},
                                    100.0, 900.0, 0.0)]
        payload = checkpoint_payload(episodes, aggregate(episodes), {'runner': {'town': 'Town10HD'}})
        json.dumps(payload)                       # must survive the round trip to disk
        summary = summarise_leaderboard(payload)
        self.assertEqual(summary['routes'], 1)
        self.assertAlmostEqual(summary['driving_score'], 50.0)
        self.assertEqual(summary['collisions_pedestrian'], 1)
        self.assertEqual(payload['runner']['town'], 'Town10HD')


if __name__ == '__main__':
    unittest.main()
