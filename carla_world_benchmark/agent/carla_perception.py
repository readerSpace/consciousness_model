"""Privileged perception: CARLA world state -> symbolic hazards.

Shared by the leaderboard agent and the standalone runner so both experiments
see the world through exactly the same encoder.  Hazards are read from the
simulator's actor list rather than from raw sensors: the variable under test is
the selection mechanism, and perception noise would confound it.  That also
means anything built on this module is a research instrument, not a leaderboard
submission.
"""

from __future__ import print_function

import math

import carla

from carla_consciousness_bridge import HazardObservation


def speed_of(actor):
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def hazard_kind(actor):
    type_id = actor.type_id
    if type_id.startswith('walker'):
        return 'walker'
    if type_id.startswith('vehicle'):
        wheels = actor.attributes.get('number_of_wheels')
        if wheels is not None and int(wheels) == 2:
            return 'bicycle'
        return 'vehicle'
    return 'static'


def hazards_from_world(world, ego, radius=60.0, behind_margin=2.0, height_margin=6.0):
    """Project nearby actors and a red light into the ego frame."""
    transform = ego.get_transform()
    origin = transform.location
    yaw = math.radians(transform.rotation.yaw)
    forward = (math.cos(yaw), math.sin(yaw))
    right = (-math.sin(yaw), math.cos(yaw))
    ego_velocity = ego.get_velocity()

    hazards = []
    for actor in world.get_actors():
        if actor.id == ego.id:
            continue
        kind = hazard_kind(actor)
        if kind == 'static':
            continue
        location = actor.get_location()
        dx, dy = location.x - origin.x, location.y - origin.y
        longitudinal = dx * forward[0] + dy * forward[1]
        lateral = dx * right[0] + dy * right[1]
        planar = math.sqrt(dx * dx + dy * dy)
        if planar > radius or longitudinal < -behind_margin:
            continue
        if abs(location.z - origin.z) > height_margin or planar < 1e-3:
            continue
        unit = (dx / planar, dy / planar)
        other = actor.get_velocity()
        # Closing rate along the line of sight; positive means the gap shrinks.
        closing = (ego_velocity.x - other.x) * unit[0] + (ego_velocity.y - other.y) * unit[1]
        hazards.append(HazardObservation(
            identifier='{}:{}'.format(kind, actor.id),
            kind=kind,
            distance=max(0.0, longitudinal),
            lateral=abs(lateral),
            relative_speed=closing,
            metadata={'actor_id': actor.id, 'type_id': actor.type_id},
        ))

    light = ego.get_traffic_light()
    if light is not None and ego.get_traffic_light_state() == carla.TrafficLightState.Red:
        light_location = light.get_transform().location
        dx, dy = light_location.x - origin.x, light_location.y - origin.y
        longitudinal = dx * forward[0] + dy * forward[1]
        lateral = dx * right[0] + dy * right[1]
        if longitudinal > -behind_margin:
            hazards.append(HazardObservation(
                identifier='traffic_light:{}'.format(light.id),
                kind='traffic_light',
                distance=max(0.0, longitudinal),
                lateral=abs(lateral),
                relative_speed=max(0.0, speed_of(ego)),
                metadata={'actor_id': light.id},
            ))
    return hazards


def route_curvature(local_planner, lookahead=12, scale=90.0):
    """Normalised heading change over the next few planned waypoints."""
    plan = list(local_planner.get_plan())[:lookahead]
    if len(plan) < 3:
        return 0.0
    yaws = [waypoint.transform.rotation.yaw for waypoint, _ in plan]
    total = 0.0
    for previous, current in zip(yaws, yaws[1:]):
        total += abs((current - previous + 180.0) % 360.0 - 180.0)
    return min(1.0, total / scale)


def apply_decision(control, decision):
    """Overlay the consciousness decision on the path follower's control."""
    if decision.hazard_stop:
        control.throttle = 0.0
        control.brake = 1.0
    elif decision.brake > 0.0:
        control.brake = max(control.brake, decision.brake)
        control.throttle = min(control.throttle, max(0.0, 1.0 - decision.brake))
    control.hand_brake = False
    return control
