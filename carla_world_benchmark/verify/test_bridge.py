"""Offline checks for the CARLA connection layer.

These run without CARLA, without a GPU and without a simulator connection, so
they are the part of the stack that can be verified before the Windows install
is finished.  Run with:  python -m unittest discover -s verify -v
"""

import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'agent'))

from carla_consciousness_bridge import (  # noqa: E402
    INF,
    ConsciousnessDrivingPolicy,
    HazardObservation,
    PolicyConfig,
    corridor_weight,
    salience,
    to_candidates,
)


def hazard(identifier, kind='vehicle', distance=30.0, lateral=0.0, relative_speed=0.0):
    return HazardObservation(identifier, kind, distance, lateral, relative_speed)


class HazardObservationTest(unittest.TestCase):

    def test_ttc_is_infinite_when_not_closing(self):
        self.assertEqual(hazard('a', relative_speed=0.0).time_to_collision, INF)
        self.assertEqual(hazard('a', relative_speed=-5.0).time_to_collision, INF)

    def test_ttc_matches_constant_velocity(self):
        self.assertAlmostEqual(hazard('a', distance=20.0, relative_speed=10.0).time_to_collision, 2.0)

    def test_rejects_invalid_geometry(self):
        with self.assertRaises(ValueError):
            HazardObservation('a', 'vehicle', -1.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            HazardObservation('a', 'spaceship', 1.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            HazardObservation('', 'vehicle', 1.0, 0.0, 0.0)


class SalienceTest(unittest.TestCase):

    def test_closing_hazard_outranks_static_one_at_same_distance(self):
        closing = salience(hazard('a', distance=20.0, relative_speed=10.0), ego_speed=10.0)
        static = salience(hazard('b', distance=20.0, relative_speed=0.0), ego_speed=10.0)
        self.assertGreater(closing, static)

    def test_lateral_offset_reduces_salience(self):
        on_path = salience(hazard('a', distance=20.0, relative_speed=10.0, lateral=0.0), 10.0)
        off_path = salience(hazard('b', distance=20.0, relative_speed=10.0, lateral=6.0), 10.0)
        self.assertGreater(on_path, off_path)
        self.assertLess(corridor_weight(6.0), 0.1)

    def test_walker_outranks_vehicle_all_else_equal(self):
        walker = salience(hazard('a', kind='walker', distance=15.0, relative_speed=8.0), 10.0)
        vehicle = salience(hazard('b', kind='vehicle', distance=15.0, relative_speed=8.0), 10.0)
        self.assertGreater(walker, vehicle)

    def test_risk_widens_salience(self):
        calm = salience(hazard('a', distance=15.0, relative_speed=8.0), 10.0, risk=0.0)
        alarmed = salience(hazard('a', distance=15.0, relative_speed=8.0), 10.0, risk=1.0)
        self.assertGreater(alarmed, calm)


class SelectionTest(unittest.TestCase):

    def scene(self, count):
        # Uniformly less urgent as the index grows.
        return [hazard('h{}'.format(i), distance=10.0 + 8.0 * i, relative_speed=9.0 - 0.5 * i)
                for i in range(count)]

    def test_workspace_admits_exactly_capacity_items(self):
        policy = ConsciousnessDrivingPolicy(PolicyConfig(capacity=3))
        decision = policy.step(ego_speed=10.0, hazards=self.scene(9))
        self.assertEqual(len(decision.admitted), 3)
        self.assertEqual(len(decision.dropped), 6)

    def test_workspace_admits_the_most_salient(self):
        scene = self.scene(9)
        policy = ConsciousnessDrivingPolicy(PolicyConfig(capacity=3))
        decision = policy.step(ego_speed=10.0, hazards=scene)
        expected = [item.identifier for item in
                    sorted(to_candidates(scene, 10.0), key=lambda i: -i.value)[:3]]
        self.assertEqual(sorted(decision.admitted), sorted(expected))

    def test_all_mode_admits_everything(self):
        policy = ConsciousnessDrivingPolicy(PolicyConfig(capacity=3, selection='all'))
        decision = policy.step(ego_speed=10.0, hazards=self.scene(9))
        self.assertEqual(len(decision.admitted), 9)
        self.assertEqual(decision.dropped, ())

    def test_blind_mode_admits_nothing_and_never_brakes(self):
        policy = ConsciousnessDrivingPolicy(PolicyConfig(capacity=3, selection='blind'))
        imminent = [hazard('close', distance=2.0, relative_speed=12.0)]
        decision = policy.step(ego_speed=12.0, hazards=imminent)
        self.assertEqual(decision.admitted, ())
        self.assertEqual(decision.brake, 0.0)
        self.assertFalse(decision.hazard_stop)
        self.assertEqual(decision.goal, 'CRUISE')

    def test_random_mode_respects_capacity_but_not_ranking(self):
        policy = ConsciousnessDrivingPolicy(PolicyConfig(capacity=3, selection='random', seed=7))
        decision = policy.step(ego_speed=10.0, hazards=self.scene(9))
        self.assertEqual(len(decision.admitted), 3)

    def test_dropped_hazard_cannot_trigger_braking(self):
        # The urgent hazard is off the path, three closer ones fill the workspace.
        scene = [hazard('urgent_offpath', distance=3.0, relative_speed=12.0, lateral=9.0)]
        scene += [hazard('near{}'.format(i), distance=8.0 + i, relative_speed=6.0) for i in range(3)]
        policy = ConsciousnessDrivingPolicy(PolicyConfig(capacity=3))
        decision = policy.step(ego_speed=12.0, hazards=scene)
        self.assertIn('urgent_offpath', decision.dropped)
        self.assertLess(decision.min_ttc_true, decision.min_ttc_admitted)


class ControlTest(unittest.TestCase):

    def test_imminent_admitted_hazard_forces_full_brake(self):
        policy = ConsciousnessDrivingPolicy(PolicyConfig(capacity=4))
        decision = policy.step(ego_speed=12.0, hazards=[hazard('a', distance=6.0, relative_speed=12.0)])
        self.assertTrue(decision.hazard_stop)
        self.assertEqual(decision.brake, 1.0)
        self.assertEqual(decision.goal, 'SAFETY')

    def test_empty_scene_cruises_at_the_ceiling(self):
        policy = ConsciousnessDrivingPolicy(PolicyConfig(capacity=4, base_target_speed_kmh=30.0))
        decision = policy.step(ego_speed=8.0, hazards=[])
        self.assertEqual(decision.goal, 'CRUISE')
        self.assertEqual(decision.brake, 0.0)
        self.assertGreater(decision.target_speed_kmh, 0.0)
        self.assertLessEqual(decision.target_speed_kmh, 30.0)

    def test_speed_limit_caps_the_target(self):
        policy = ConsciousnessDrivingPolicy(PolicyConfig(capacity=4, base_target_speed_kmh=90.0))
        decision = policy.step(ego_speed=8.0, hazards=[], speed_limit_kmh=20.0)
        self.assertLessEqual(decision.target_speed_kmh, 20.0)

    def test_safety_goal_has_hysteresis(self):
        policy = ConsciousnessDrivingPolicy(
            PolicyConfig(capacity=4, safety_enter_ttc=3.0, safety_release_ttc=5.0))
        entering = policy.step(ego_speed=10.0, hazards=[hazard('a', distance=25.0, relative_speed=10.0)])
        self.assertEqual(entering.goal, 'SAFETY')
        # TTC 4.0 s sits between enter and release: the safety goal must persist.
        holding = policy.step(ego_speed=10.0, hazards=[hazard('a', distance=40.0, relative_speed=10.0)])
        self.assertEqual(holding.goal, 'SAFETY')
        released = policy.step(ego_speed=10.0, hazards=[hazard('a', distance=80.0, relative_speed=10.0)])
        self.assertEqual(released.goal, 'CRUISE')

    def test_curvature_slows_the_target_speed(self):
        straight = ConsciousnessDrivingPolicy(PolicyConfig(capacity=4)).step(
            ego_speed=8.0, hazards=[], route_curvature=0.0)
        bend = ConsciousnessDrivingPolicy(PolicyConfig(capacity=4)).step(
            ego_speed=8.0, hazards=[], route_curvature=1.0)
        self.assertLess(bend.target_speed_kmh, straight.target_speed_kmh)


class CalibrationTest(unittest.TestCase):

    def test_capacity_induced_misses_raise_risk_and_lower_speed(self):
        policy = ConsciousnessDrivingPolicy(PolicyConfig(capacity=1))
        # One decoy keeps the single slot occupied; the ignored hazard closes in.
        for step in range(12):
            decoy = hazard('decoy', distance=1.0, lateral=0.2, relative_speed=15.0)
            sneaker = hazard('sneaker', distance=max(1.0, 20.0 - 1.6 * step),
                             lateral=3.4, relative_speed=11.0)
            decision = policy.step(ego_speed=12.0, hazards=[decoy, sneaker])
        self.assertIn('sneaker', decision.dropped)
        self.assertGreater(policy.risk, 0.0)

    def test_risk_stays_zero_without_misses(self):
        policy = ConsciousnessDrivingPolicy(PolicyConfig(capacity=8))
        for _ in range(10):
            policy.step(ego_speed=8.0, hazards=[hazard('far', distance=90.0, relative_speed=1.0)])
        self.assertEqual(policy.risk, 0.0)


class DeterminismTest(unittest.TestCase):

    def sequence(self):
        policy = ConsciousnessDrivingPolicy(PolicyConfig(capacity=3, seed=5))
        out = []
        for step in range(25):
            scene = [hazard('h{}'.format(i), distance=5.0 + 3.0 * i + step,
                            lateral=0.4 * i, relative_speed=10.0 - 0.7 * i)
                     for i in range(6)]
            out.append(policy.step(ego_speed=9.0, hazards=scene, route_curvature=0.1).as_record())
        return out

    def test_two_fresh_policies_agree_step_for_step(self):
        self.assertEqual(self.sequence(), self.sequence())

    def test_records_are_json_serialisable(self):
        json.dumps(self.sequence())


class ConfigTest(unittest.TestCase):

    def test_rejects_bad_thresholds(self):
        with self.assertRaises(ValueError):
            PolicyConfig(brake_ttc=4.0, safety_enter_ttc=3.0, safety_release_ttc=5.0)
        with self.assertRaises(ValueError):
            PolicyConfig(capacity=0)
        with self.assertRaises(ValueError):
            PolicyConfig(selection='telepathy')

    def test_shipped_configs_are_valid(self):
        directory = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'agent', 'configs')
        keys = ('capacity', 'selection', 'base_target_speed_kmh', 'min_target_speed_kmh',
                'safety_enter_ttc', 'safety_release_ttc', 'brake_ttc', 'stop_distance',
                'corridor_half_width', 'risk_window', 'seed')
        names = sorted(name for name in os.listdir(directory) if name.endswith('.json'))
        self.assertTrue(names)
        for name in names:
            with open(os.path.join(directory, name)) as handle:
                loaded = json.load(handle)
            PolicyConfig(**{key: value for key, value in loaded.items() if key in keys})


if __name__ == '__main__':
    unittest.main()
