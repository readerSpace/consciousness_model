#!/usr/bin/env python
"""A CARLA-free closed-loop rehearsal of the ablation.

This is a one-dimensional toy: a kinematic ego on a straight corridor, scripted
hazards, and the real :class:`ConsciousnessDrivingPolicy` in the loop.  It is not
a driving result and says nothing about CARLA scores.  Its purpose is narrow and
useful: to show, before the simulator is installed, that the capacity parameter
actually reaches behaviour -- that ``blind``, ``k1`` and ``k4`` are separable
outcomes rather than three names for the same policy.  If this table is flat, the
CARLA runs will be flat too, and the bug is here rather than in the simulator.

    python verify\\simulate_ablation.py --seeds 40
"""

from __future__ import print_function

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'agent'))

from carla_consciousness_bridge import (  # noqa: E402
    ConsciousnessDrivingPolicy,
    HazardObservation,
    PolicyConfig,
)

DT = 0.05
HORIZON = 40.0
COLLISION_GAP = 2.0
COLLISION_LATERAL = 1.2


class Actor(object):
    """A hazard on the corridor, in the ego's longitudinal frame."""

    def __init__(self, identifier, kind, position, lateral, speed, cut_in_at=None):
        self.identifier = identifier
        self.kind = kind
        self.position = position      # metres ahead of the origin
        self.lateral = lateral
        self.speed = speed            # m/s along the corridor
        self.cut_in_at = cut_in_at    # simulation time at which it swerves into the lane

    def advance(self, now):
        self.position += self.speed * DT
        if self.cut_in_at is not None and now >= self.cut_in_at:
            self.lateral = max(0.0, self.lateral - 2.5 * DT)


def build_scene(rng, count):
    actors = []
    for index in range(count):
        kind = rng.choice(['vehicle', 'vehicle', 'vehicle', 'walker', 'bicycle'])
        actors.append(Actor(
            identifier='{}:{}'.format(kind, index),
            kind=kind,
            position=rng.uniform(8.0, 90.0),
            lateral=rng.choice([0.0, 0.3, 0.8, 2.5, 4.0, 6.0, 8.0]),
            speed=rng.uniform(0.0, 9.0),
            # A third of the off-lane actors eventually cut in: these are the
            # hazards a small workspace is most likely to have dropped.
            cut_in_at=rng.uniform(3.0, 18.0) if rng.random() < 0.33 else None,
        ))
    return actors


def run_episode(config, seed, actor_count):
    rng = random.Random(seed)
    actors = build_scene(rng, actor_count)
    policy = ConsciousnessDrivingPolicy(config)

    ego_position = 0.0
    ego_speed = 8.0
    collisions = 0
    hit = set()
    now = 0.0

    while now < HORIZON:
        hazards = []
        for actor in actors:
            gap = actor.position - ego_position
            if gap < -5.0 or gap > 120.0:
                continue
            hazards.append(HazardObservation(
                identifier=actor.identifier,
                kind=actor.kind,
                distance=max(0.0, gap),
                lateral=actor.lateral,
                relative_speed=ego_speed - actor.speed,
            ))

        decision = policy.step(ego_speed=ego_speed, hazards=hazards)

        target = decision.target_speed_kmh / 3.6
        if decision.hazard_stop:
            ego_speed = max(0.0, ego_speed - 7.0 * DT)
        elif decision.brake > 0.0:
            ego_speed = max(0.0, ego_speed - 7.0 * decision.brake * DT)
        else:
            ego_speed += max(-3.0, min(2.5, (target - ego_speed) * 1.5)) * DT
            ego_speed = max(0.0, ego_speed)

        ego_position += ego_speed * DT
        for actor in actors:
            actor.advance(now)
            gap = actor.position - ego_position
            if (actor.identifier not in hit and actor.lateral < COLLISION_LATERAL
                    and -1.0 < gap < COLLISION_GAP):
                hit.add(actor.identifier)
                collisions += 1
        now += DT

    return {
        'collisions': collisions,
        'distance': ego_position,
        'risk': policy.risk,
        'mean_speed': ego_position / HORIZON,
    }


VARIANTS = (
    ('k4_workspace', PolicyConfig(capacity=4, selection='workspace')),
    ('k2_workspace', PolicyConfig(capacity=2, selection='workspace')),
    ('k1_workspace', PolicyConfig(capacity=1, selection='workspace')),
    ('k4_random', PolicyConfig(capacity=4, selection='random', seed=11)),
    ('unbounded', PolicyConfig(capacity=4, selection='all')),
    ('blind', PolicyConfig(capacity=4, selection='blind')),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=40)
    parser.add_argument('--actors', type=int, default=14)
    args = parser.parse_args()

    print('toy corridor rehearsal: {} seeds x {} actors, {:.0f}s episodes'.format(
        args.seeds, args.actors, HORIZON))
    print('| config | collisions/ep | distance (m) | mean speed (m/s) | final risk |')
    print('| --- | --- | --- | --- | --- |')
    for name, config in VARIANTS:
        results = [run_episode(config, seed, args.actors) for seed in range(args.seeds)]
        n = float(len(results))
        print('| {} | {:.2f} | {:.1f} | {:.2f} | {:.3f} |'.format(
            name,
            sum(r['collisions'] for r in results) / n,
            sum(r['distance'] for r in results) / n,
            sum(r['mean_speed'] for r in results) / n,
            sum(r['risk'] for r in results) / n))
    print('')
    print('Read this as a wiring check only. A useful sweep separates blind from')
    print('the bounded variants, and shows capacity trading safety against progress.')


if __name__ == '__main__':
    main()
