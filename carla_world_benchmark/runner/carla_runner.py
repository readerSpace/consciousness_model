#!/usr/bin/env python
"""Standalone closed-loop experiment against a bare CARLA server.

The Leaderboard 2.0 routes live in Town12 and Town13, which the local
CARLA package does not ship (see README section 1).  This runner reproduces the
parts of the leaderboard that the ablation actually needs -- a routed ego, dense
traffic, collision and infraction accounting -- using nothing but the CARLA
PythonAPI, so the experiment can run on the maps that are installed.

    python runner\\carla_runner.py --config agent\\configs\\k4_workspace.json ^
        --town Town10HD --seeds 3 --vehicles 60 --walkers 30

Writes two files per configuration into ``results/``:
``<tag>.json`` (per-step policy diagnostics) and ``<tag>.leaderboard.json``
(scores in the leaderboard checkpoint schema), so ``verify/analyze_ablation.py``
reads runner output and leaderboard output the same way.
"""

from __future__ import print_function

import argparse
import json
import math
import os
import random
import sys
import time

BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BENCH_ROOT, 'agent'))
sys.path.insert(0, os.path.dirname(BENCH_ROOT))

import carla  # noqa: E402
from agents.navigation.basic_agent import BasicAgent  # noqa: E402

from carla_consciousness_bridge import (  # noqa: E402
    INF,
    ConsciousnessDrivingPolicy,
    HazardObservation,
    PolicyConfig,
)
from carla_perception import apply_decision, hazards_from_world, route_curvature, speed_of  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (  # noqa: E402
    PENALTY,
    attach_episode_metrics,
    aggregate,
    checkpoint_payload,
    collision_class,
    episode_summary,
)

DEFAULT_OPTIONS = {
    "capacity": 4, "selection": "workspace", "base_target_speed_kmh": 30.0,
    "min_target_speed_kmh": 6.0, "safety_enter_ttc": 3.0, "safety_release_ttc": 5.0,
    "brake_ttc": 1.5, "stop_distance": 4.0, "corridor_half_width": 1.75,
    "risk_window": 20, "seed": 0, "perception_radius": 60.0,
    "ignore_builtin_hazards": True, "use_camera": False,
    "record_path": "", "record_every": 1,
}
POLICY_KEYS = ("capacity", "selection", "base_target_speed_kmh", "min_target_speed_kmh",
               "safety_enter_ttc", "safety_release_ttc", "brake_ttc", "stop_distance",
               "corridor_half_width", "risk_window", "seed")


def load_options(path):
    options = dict(DEFAULT_OPTIONS)
    if path:
        with open(path) as handle:
            loaded = json.load(handle)
        unknown = set(loaded) - set(DEFAULT_OPTIONS)
        if unknown:
            raise ValueError('unknown config keys: {}'.format(sorted(unknown)))
        options.update(loaded)
    return options


class Scenario(object):
    """One episode: build the world, run it, tear it down. Always tears down."""

    def __init__(self, client, args, options, seed):
        self.client = client
        self.args = args
        self.options = options
        self.seed = seed
        self.random = random.Random(seed)
        self.world = None
        self.original_settings = None
        self.ego = None
        self.sensors = []
        self.destination = None
        self.vehicles = []
        self.walkers = []              # walker actor ids
        self.walker_controllers = []   # their controller actor ids, kept separate so
                                       # a partial controller spawn cannot desynchronise
                                       # teardown from the walkers it must stop first
        self.scenario_actors = []
        self.collisions = []
        self.traffic_manager = None
        self.stop_marker = None

    # -- world ---------------------------------------------------------------

    def _configure_world(self):
        current = self.client.get_world()
        name = current.get_map().name.split('/')[-1]
        self.world = current if name == self.args.town else self.client.load_world(self.args.town)
        self.original_settings = self.world.get_settings()

        self.traffic_manager = self.client.get_trafficmanager(self.args.tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(self.seed)

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.args.delta
        if self.args.no_render:
            settings.no_rendering_mode = True
        self.world.apply_settings(settings)
        self.world.set_pedestrians_seed(self.seed)

    def _spawn_ego(self):
        library = self.world.get_blueprint_library()
        candidates = [bp for bp in library.filter('vehicle.*')
                      if bp.get_attribute('base_type') and
                      bp.get_attribute('base_type').as_str() == 'car']
        blueprint = (candidates or list(library.filter('vehicle.*')))[0]
        blueprint.set_attribute('role_name', 'hero')

        spawn_points = self.world.get_map().get_spawn_points()
        if len(spawn_points) < 2:
            raise RuntimeError('map {} has too few spawn points'.format(self.args.town))
        self.random.shuffle(spawn_points)
        for transform in spawn_points:
            ego = self.world.try_spawn_actor(blueprint, transform)
            if ego is not None:
                self.ego = ego
                break
        if self.ego is None:
            raise RuntimeError('could not spawn the ego vehicle')

        # Destination: the farthest spawn point, so every seed gets a long route.
        origin = self.ego.get_location()
        self.destination = max(
            (point.location for point in spawn_points),
            key=lambda location: location.distance(origin))

    def _attach_collision_sensor(self):
        blueprint = self.world.get_blueprint_library().find('sensor.other.collision')
        sensor = self.world.spawn_actor(blueprint, carla.Transform(), attach_to=self.ego)
        sensor.listen(lambda event: self.collisions.append(
            getattr(event.other_actor, 'type_id', '') or ''))
        self.sensors.append(sensor)

    def _spawn_traffic(self):
        if self.args.scenario != 'random_traffic':
            return
        if self.args.vehicles > 0:
            library = self.world.get_blueprint_library()
            blueprints = sorted(library.filter('vehicle.*'), key=lambda bp: bp.id)
            points = self.world.get_map().get_spawn_points()
            self.random.shuffle(points)
            batch = []
            for transform in points[: self.args.vehicles]:
                blueprint = self.random.choice(blueprints)
                if blueprint.has_attribute('color'):
                    blueprint.set_attribute(
                        'color', self.random.choice(
                            blueprint.get_attribute('color').recommended_values))
                blueprint.set_attribute('role_name', 'autopilot')
                batch.append(carla.command.SpawnActor(blueprint, transform).then(
                    carla.command.SetAutopilot(carla.command.FutureActor, True,
                                               self.traffic_manager.get_port())))
            for response in self.client.apply_batch_sync(batch, True):
                if not response.error:
                    self.vehicles.append(response.actor_id)

        if self.args.walkers > 0:
            self._spawn_walkers()

    def _spawn_walkers(self):
        library = self.world.get_blueprint_library()
        walker_blueprints = list(library.filter('walker.pedestrian.*'))
        if not walker_blueprints:
            return
        transforms = []
        for _ in range(self.args.walkers):
            location = self.world.get_random_location_from_navigation()
            if location is not None:
                transform = carla.Transform()
                transform.location = location
                transforms.append(transform)

        batch, speeds = [], []
        for transform in transforms:
            blueprint = self.random.choice(walker_blueprints)
            if blueprint.has_attribute('is_invincible'):
                blueprint.set_attribute('is_invincible', 'false')
            if blueprint.has_attribute('speed'):
                speeds.append(blueprint.get_attribute('speed').recommended_values[1])
            else:
                speeds.append('0.0')
            batch.append(carla.command.SpawnActor(blueprint, transform))

        walker_ids, walker_speeds = [], []
        for index, response in enumerate(self.client.apply_batch_sync(batch, True)):
            if not response.error:
                walker_ids.append(response.actor_id)
                walker_speeds.append(speeds[index])

        controller_bp = library.find('controller.ai.walker')
        batch = [carla.command.SpawnActor(controller_bp, carla.Transform(), walker_id)
                 for walker_id in walker_ids]
        controller_ids = [response.actor_id
                          for response in self.client.apply_batch_sync(batch, True)
                          if not response.error]

        self.walkers = walker_ids
        self.walker_controllers = controller_ids
        self.world.tick()
        self.world.set_pedestrians_cross_factor(0.1)
        for index, controller in enumerate(self.world.get_actors(controller_ids)):
            controller.start()
            target = self.world.get_random_location_from_navigation()
            if target is not None:
                controller.go_to_location(target)
            if index < len(walker_speeds):
                controller.set_max_speed(float(walker_speeds[index]))

    def _car_blueprint(self):
        library = self.world.get_blueprint_library()
        candidates = [bp for bp in library.filter('vehicle.*')
                      if bp.get_attribute('base_type') and
                      bp.get_attribute('base_type').as_str() == 'car']
        return (candidates or list(library.filter('vehicle.*')))[0]

    def _waypoint_ahead(self, distance):
        waypoint = self.world.get_map().get_waypoint(self.ego.get_location())
        candidates = waypoint.next(distance)
        return candidates[0] if candidates else waypoint

    def _try_spawn_scenario_actor(self, blueprint, transform, role_name):
        blueprint.set_attribute('role_name', role_name)
        actor = self.world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            self.scenario_actors.append(actor.id)
        return actor

    def _spawn_controlled_scenario(self):
        """Create deterministic first-pass CARLA hazards for causal tracing."""
        if self.args.scenario == 'random_traffic':
            return
        if self.args.scenario == 'route_stop':
            marker = self._waypoint_ahead(35.0)
            self.stop_marker = marker.transform.location
            return

        blueprint = self._car_blueprint()
        if self.args.scenario == 'sudden_stop':
            lead = self._waypoint_ahead(28.0)
            actor = self._try_spawn_scenario_actor(blueprint, lead.transform, 'scenario_lead')
            if actor is not None:
                actor.set_target_velocity(lead.transform.get_forward_vector() * 7.0)
            return

        if self.args.scenario == 'cut_in':
            ego_lane = self.world.get_map().get_waypoint(self.ego.get_location())
            lane = ego_lane.get_left_lane() or ego_lane.get_right_lane()
            if lane is None or lane.lane_type != carla.LaneType.Driving:
                lane = ego_lane
            ahead = lane.next(18.0)[0] if lane.next(18.0) else lane
            actor = self._try_spawn_scenario_actor(blueprint, ahead.transform, 'scenario_cutin')
            if actor is not None:
                actor.set_target_velocity(ahead.transform.get_forward_vector() * 6.0)
            return

        raise RuntimeError('unknown scenario {}'.format(self.args.scenario))

    def _build_follower(self):
        opt_dict = {}
        if self.options['ignore_builtin_hazards']:
            opt_dict = {'ignore_vehicles': True, 'ignore_traffic_lights': True,
                        'ignore_stop_signs': True}
        follower = BasicAgent(self.ego, self.options['base_target_speed_kmh'], opt_dict=opt_dict)
        follower.set_destination(self.destination)
        return follower

    # -- run -----------------------------------------------------------------

    def run(self):
        try:
            self._configure_world()
            self._spawn_ego()
            self._attach_collision_sensor()
            self._spawn_controlled_scenario()
            self._spawn_traffic()
            self.world.tick()
            follower = self._build_follower()
            return self._loop(follower)
        finally:
            self.cleanup()

    def _loop(self, follower):
        policy = ConsciousnessDrivingPolicy(
            PolicyConfig(**{key: self.options[key] for key in POLICY_KEYS}))
        planner = follower.get_local_planner()
        total_waypoints = max(1, len(planner.get_plan()))

        steps = []
        elapsed = 0.0
        stopped_for = 0.0
        travelled = 0.0
        previous_location = self.ego.get_location()
        red_light_runs = 0
        pending_red = None
        outcome = 'TIMEOUT'
        frame = 0
        speed_sum = 0.0
        speed_count = 0
        stopped_time = 0.0
        harsh_brakes = 0
        min_ttc_true_seen = INF
        min_ttc_admitted_seen = INF

        while elapsed < self.args.timeout:
            self.world.tick()
            frame += 1
            elapsed += self.args.delta
            self._tick_controlled_scenario(elapsed)

            speed = speed_of(self.ego)
            location = self.ego.get_location()
            travelled += location.distance(previous_location)
            previous_location = location
            speed_sum += speed
            speed_count += 1

            hazards = hazards_from_world(self.world, self.ego,
                                         self.options['perception_radius'])
            hazards = tuple(list(hazards) + list(self._synthetic_hazards(speed)))
            decision = policy.step(
                ego_speed=speed,
                hazards=hazards,
                route_curvature=route_curvature(planner),
                distance_to_goal=location.distance(self.destination),
                speed_limit_kmh=self.ego.get_speed_limit(),
            )
            follower.set_target_speed(decision.target_speed_kmh)
            control = apply_decision(follower.run_step(), decision)
            self.ego.apply_control(control)
            if decision.min_ttc_true != INF:
                min_ttc_true_seen = min(min_ttc_true_seen, decision.min_ttc_true)
            if decision.min_ttc_admitted != INF:
                min_ttc_admitted_seen = min(min_ttc_admitted_seen, decision.min_ttc_admitted)
            if control.brake >= 0.7:
                harsh_brakes += 1

            # Red-light accounting: entering a red light's box and leaving it while
            # still moving counts as a run.
            light = self.ego.get_traffic_light()
            at_red = (light is not None
                      and self.ego.get_traffic_light_state() == carla.TrafficLightState.Red)
            if at_red:
                pending_red = light.id
            elif pending_red is not None:
                if speed > 2.0:
                    red_light_runs += 1
                pending_red = None

            stopped_for = stopped_for + self.args.delta if speed < 0.1 else 0.0
            stopped_time = stopped_time + self.args.delta if speed < 0.1 else stopped_time
            remaining = len(planner.get_plan())
            route_progress = 100.0 * max(0.0, min(1.0, 1.0 - remaining / float(total_waypoints)))

            if frame % max(1, int(self.options['record_every'])) == 0:
                record = decision.as_record()
                record.update({'episode': self.seed, 'frame': frame, 'timestamp': elapsed,
                               'sim_time': elapsed, 'speed_ms': speed,
                               'ego_speed': speed, 'hazard_candidates': [
                                   self._hazard_record(hazard) for hazard in hazards
                               ], 'selected_workspace': list(decision.admitted),
                               'target_speed': decision.target_speed_kmh,
                               'throttle': control.throttle, 'brake': control.brake,
                               'steer': control.steer, 'collision_count': len(self.collisions),
                               'collision': bool(self.collisions),
                               'route_progress': route_progress})
                steps.append(record)

            if follower.done() or location.distance(self.destination) < 5.0:
                outcome = 'SUCCESS'
                break
            if stopped_for > self.args.blocked_timeout:
                outcome = 'BLOCKED'
                break

        remaining = len(planner.get_plan())
        completion = 100.0 * max(0.0, min(1.0, 1.0 - remaining / float(total_waypoints)))
        if outcome == 'SUCCESS':
            completion = 100.0

        infractions = dict((key, 0) for key in PENALTY)
        for type_id in self.collisions:
            infractions[collision_class(type_id)] += 1
        infractions['red_light'] = red_light_runs

        summary = episode_summary(self.seed, outcome, completion, infractions,
                                  elapsed, travelled, policy.risk)
        summary = attach_episode_metrics(summary, {
            'mean_speed_ms': speed_sum / float(speed_count) if speed_count else 0.0,
            'stopped_time_s': stopped_time,
            'min_ttc_true': None if min_ttc_true_seen == INF else min_ttc_true_seen,
            'min_ttc_admitted': None if min_ttc_admitted_seen == INF else min_ttc_admitted_seen,
            'harsh_brake_count': harsh_brakes,
            'scenario': self.args.scenario,
        })
        summary['collision_details'] = list(self.collisions)
        return summary, steps

    def _tick_controlled_scenario(self, elapsed):
        actors = list(self.world.get_actors(self.scenario_actors)) if self.scenario_actors else []
        if self.args.scenario == 'sudden_stop':
            for actor in actors:
                if elapsed >= 3.0:
                    actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        elif self.args.scenario == 'cut_in' and elapsed >= 3.0:
            target = self._waypoint_ahead(20.0).transform.location
            for actor in actors:
                transform = actor.get_transform()
                transform.location.x += 0.08 * (target.x - transform.location.x)
                transform.location.y += 0.08 * (target.y - transform.location.y)
                actor.set_transform(transform)

    def _synthetic_hazards(self, speed):
        if self.args.scenario != 'route_stop' or self.stop_marker is None:
            return ()
        transform = self.ego.get_transform()
        origin = transform.location
        yaw = math.radians(transform.rotation.yaw)
        forward = (math.cos(yaw), math.sin(yaw))
        right = (-math.sin(yaw), math.cos(yaw))
        dx, dy = self.stop_marker.x - origin.x, self.stop_marker.y - origin.y
        longitudinal = dx * forward[0] + dy * forward[1]
        lateral = dx * right[0] + dy * right[1]
        if longitudinal < -2.0 or longitudinal > self.options['perception_radius']:
            return ()
        return (HazardObservation(
            identifier='route_stop:0',
            kind='traffic_light',
            distance=max(0.0, longitudinal),
            lateral=abs(lateral),
            relative_speed=max(0.0, speed),
            metadata={'scenario': 'route_stop'},
        ),)

    def _hazard_record(self, hazard):
        return {
            'id': hazard.identifier,
            'kind': hazard.kind,
            'distance': round(hazard.distance, 3),
            'lateral': round(hazard.lateral, 3),
            'relative_speed': round(hazard.relative_speed, 3),
            'ttc': None if hazard.time_to_collision == INF else round(hazard.time_to_collision, 3),
        }

    # -- teardown ------------------------------------------------------------

    def cleanup(self):
        for sensor in self.sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                pass
        self.sensors = []
        if self.world is not None and self.walker_controllers:
            for controller in self.world.get_actors(self.walker_controllers):
                try:
                    controller.stop()
                except RuntimeError:
                    pass
        # Controllers first: a controller outliving its walker throws on the server.
        ids = (list(self.walker_controllers) + list(self.walkers) +
               list(self.vehicles) + list(self.scenario_actors))
        if ids:
            self.client.apply_batch([carla.command.DestroyActor(actor_id) for actor_id in ids])
        self.vehicles, self.walkers, self.walker_controllers, self.scenario_actors = [], [], [], []
        if self.ego is not None:
            try:
                self.ego.destroy()
            except RuntimeError:
                pass
            self.ego = None
        if self.traffic_manager is not None:
            try:
                self.traffic_manager.set_synchronous_mode(False)
            except RuntimeError:
                pass
        if self.world is not None and self.original_settings is not None:
            try:
                self.world.apply_settings(self.original_settings)
            except RuntimeError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='agent config JSON')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--tm-port', type=int, default=8000)
    parser.add_argument('--town', default='Town10HD')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--first-seed', type=int, default=0)
    parser.add_argument('--vehicles', type=int, default=60)
    parser.add_argument('--walkers', type=int, default=30)
    parser.add_argument('--scenario', default='random_traffic',
                        choices=('random_traffic', 'sudden_stop', 'cut_in', 'route_stop'),
                        help='deterministic controlled scenario or random traffic sweep')
    parser.add_argument('--delta', type=float, default=0.05)
    parser.add_argument('--timeout', type=float, default=300.0, help='sim seconds per episode')
    parser.add_argument('--blocked-timeout', type=float, default=90.0)
    parser.add_argument('--no-render', action='store_true',
                        help='disable rendering on the server; much faster, no camera output')
    parser.add_argument('--results-dir', default=os.path.join(BENCH_ROOT, 'results'))
    args = parser.parse_args()

    options = load_options(args.config)
    tag = os.path.splitext(os.path.basename(args.config))[0]

    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    print('[runner] server {} / client {}'.format(
        client.get_server_version(), client.get_client_version()))

    episodes, all_steps = [], []
    for index in range(args.seeds):
        seed = args.first_seed + index
        started = time.time()
        summary, steps = Scenario(client, args, options, seed).run()
        episodes.append(summary)
        all_steps.extend(steps)
        print('[runner] {} seed={} {} completion={:.1f}% penalty={:.2f} score={:.1f} '
              '({:.0f}s wall)'.format(tag, seed, summary['outcome'],
                                      summary['route_completion'],
                                      summary['infraction_penalty'],
                                      summary['driving_score'], time.time() - started))

    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    steps_path = os.path.join(args.results_dir, '{}.json'.format(tag))
    with open(steps_path, 'w') as handle:
        json.dump({'config': options, 'steps': all_steps}, handle, indent=1)

    summary = aggregate(episodes)
    scores_path = os.path.join(args.results_dir, '{}.leaderboard.json'.format(tag))
    with open(scores_path, 'w') as handle:
        json.dump(checkpoint_payload(episodes, summary, {'runner': vars(args)}), handle, indent=1)

    print('[runner] {}: score={:.1f} completion={:.1f}% success={:.0%}'.format(
        tag, summary['driving_score'], summary['route_completion'], summary['success_rate']))
    print('[runner] wrote {} and {}'.format(steps_path, scores_path))


if __name__ == '__main__':
    main()
