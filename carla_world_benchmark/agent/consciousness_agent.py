#!/usr/bin/env python
"""Leaderboard 2.0 agent whose hazard attention is a finite conscious workspace.

``BasicAgent`` (CARLA PythonAPI) is used only as a path follower: route tracking,
steering and a speed controller.  Its built-in reactions to vehicles, traffic
lights and stop signs are switched off by default, so every hazard response comes
from :class:`ConsciousnessDrivingPolicy` and capacity K stays the single
independent variable.

Requires the leaderboard and scenario_runner stack.  If that stack is not
installed (see README section 2), ``runner/carla_runner.py`` runs the same policy
against a bare CARLA server instead.

Entry point: ``ConsciousnessAgent``.
"""

from __future__ import print_function

import json
import os
import sys

import carla
from agents.navigation.basic_agent import BasicAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from carla_consciousness_bridge import ConsciousnessDrivingPolicy, PolicyConfig  # noqa: E402
from carla_perception import (  # noqa: E402
    apply_decision,
    hazards_from_world,
    route_curvature,
    speed_of,
)


def get_entry_point():
    return 'ConsciousnessAgent'


DEFAULTS = {
    "capacity": 4,
    "selection": "workspace",
    "base_target_speed_kmh": 30.0,
    "min_target_speed_kmh": 6.0,
    "safety_enter_ttc": 3.0,
    "safety_release_ttc": 5.0,
    "brake_ttc": 1.5,
    "stop_distance": 4.0,
    "corridor_half_width": 1.75,
    "risk_window": 20,
    "seed": 0,
    # agent-level options
    "perception_radius": 60.0,
    "ignore_builtin_hazards": True,
    "use_camera": False,
    "record_path": "",
    "record_every": 1,
}

POLICY_KEYS = (
    "capacity", "selection", "base_target_speed_kmh", "min_target_speed_kmh",
    "safety_enter_ttc", "safety_release_ttc", "brake_ttc", "stop_distance",
    "corridor_half_width", "risk_window", "seed",
)


def load_options(path_to_conf_file):
    """Merge a JSON config over the defaults, rejecting unknown keys."""
    options = dict(DEFAULTS)
    if path_to_conf_file:
        with open(path_to_conf_file, 'r') as handle:
            loaded = json.load(handle)
        unknown = set(loaded) - set(DEFAULTS)
        if unknown:
            raise ValueError('unknown config keys: {}'.format(sorted(unknown)))
        options.update(loaded)
    return options


def make_policy(options):
    return ConsciousnessDrivingPolicy(
        PolicyConfig(**{key: options[key] for key in POLICY_KEYS}))


def build_follower(ego, global_plan_world_coord, options, carla_map):
    """A BasicAgent reduced to pure path following."""
    opt_dict = {}
    if options['ignore_builtin_hazards']:
        opt_dict = {'ignore_vehicles': True, 'ignore_traffic_lights': True,
                    'ignore_stop_signs': True}
    follower = BasicAgent(ego, options['base_target_speed_kmh'], opt_dict=opt_dict)
    plan = []
    previous = None
    for transform, _ in global_plan_world_coord:
        waypoint = carla_map.get_waypoint(transform.location)
        if previous is not None:
            plan.extend(follower.trace_route(previous, waypoint))
        previous = waypoint
    follower.set_global_plan(plan)
    return follower


class ConsciousnessAgent(AutonomousAgent):
    """Route following delegated to BasicAgent, hazard attention to the model."""

    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS
        self._options = load_options(path_to_conf_file)
        self._policy = make_policy(self._options)
        self._follower = None
        self._ego = None
        self._records = []
        self._frame = 0
        print('[ConsciousnessAgent] selection={} capacity={} builtin_hazards={}'.format(
            self._options['selection'], self._options['capacity'],
            not self._options['ignore_builtin_hazards']))

    def sensors(self):
        sensors = [
            {'type': 'sensor.speedometer', 'id': 'SPEED'},
            {'type': 'sensor.other.imu', 'x': 0.0, 'y': 0.0, 'z': 0.0,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'sensor_tick': 0.05, 'id': 'IMU'},
            {'type': 'sensor.other.gnss', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'id': 'GPS'},
        ]
        if self._options['use_camera']:
            sensors.append({
                'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.0, 'z': 1.6,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 400, 'height': 300, 'fov': 100, 'id': 'CAM_FRONT',
            })
        return sensors

    def _find_ego(self):
        # The provider has no world until the evaluator has loaded the route, and
        # the leaderboard calls run_step once before that on some paths.
        world = CarlaDataProvider.get_world()
        if world is None:
            return None
        for actor in world.get_actors():
            if actor.attributes.get('role_name') == 'hero':
                return actor
        return None

    def run_step(self, input_data, timestamp):
        self._frame += 1
        if self._ego is None:
            self._ego = self._find_ego()
            if self._ego is None:
                return carla.VehicleControl()
            self._follower = build_follower(
                self._ego, self._global_plan_world_coord, self._options,
                CarlaDataProvider.get_map())
            return carla.VehicleControl()

        ego = self._ego
        world = CarlaDataProvider.get_world()
        if world is None:
            return carla.VehicleControl()
        speed = input_data['SPEED'][1]['speed'] if 'SPEED' in input_data else speed_of(ego)

        decision = self._policy.step(
            ego_speed=max(0.0, float(speed)),
            hazards=hazards_from_world(world, ego, self._options['perception_radius']),
            route_curvature=route_curvature(self._follower.get_local_planner()),
            distance_to_goal=0.0,
            speed_limit_kmh=ego.get_speed_limit(),
        )

        self._follower.set_target_speed(decision.target_speed_kmh)
        control = apply_decision(self._follower.run_step(), decision)

        if self._options['record_path'] and self._frame % max(1, int(self._options['record_every'])) == 0:
            record = decision.as_record()
            record.update({'frame': self._frame, 'timestamp': timestamp,
                           'speed_ms': float(speed), 'throttle': control.throttle,
                           'brake': control.brake, 'steer': control.steer})
            self._records.append(record)
        return control

    def destroy(self):
        path = getattr(self, '_options', {}).get('record_path')
        if path and self._records:
            directory = os.path.dirname(os.path.abspath(path))
            if directory and not os.path.isdir(directory):
                os.makedirs(directory)
            with open(path, 'w') as handle:
                json.dump({'config': self._options, 'steps': self._records}, handle, indent=1)
            print('[ConsciousnessAgent] wrote {} steps to {}'.format(len(self._records), path))
        self._follower = None
        self._ego = None
        self._records = []
