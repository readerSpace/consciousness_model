"""Score bookkeeping for the standalone runner. No CARLA import, so it is testable.

The composite follows the Leaderboard 2.0 shape -- route completion multiplied by
a product of per-infraction coefficients -- so runner numbers sit on the same
scale as leaderboard numbers.  It reproduces that metric only for the infraction
types the runner can observe, and is an approximation, not the official score.
"""

from __future__ import print_function

PENALTY = {
    'collisions_pedestrian': 0.50,
    'collisions_vehicle': 0.60,
    'collisions_layout': 0.65,
    'red_light': 0.70,
}


def collision_class(type_id):
    """Map a CARLA actor type_id onto a leaderboard infraction class."""
    type_id = type_id or ''
    if type_id.startswith('walker'):
        return 'collisions_pedestrian'
    if type_id.startswith('vehicle'):
        return 'collisions_vehicle'
    return 'collisions_layout'


def penalty_from(infractions):
    """Multiplicative penalty in (0, 1]. Unknown infraction types are free."""
    result = 1.0
    for key, count in infractions.items():
        if count < 0:
            raise ValueError('infraction counts must be non-negative')
        result *= PENALTY.get(key, 1.0) ** count
    return result


def episode_summary(seed, outcome, completion, infractions, duration, travelled, risk):
    if not 0.0 <= completion <= 100.0:
        raise ValueError('completion must be a percentage')
    penalty = penalty_from(infractions)
    return {
        'seed': seed,
        'outcome': outcome,
        'duration_s': duration,
        'travelled_m': travelled,
        'route_completion': completion,
        'infraction_penalty': penalty,
        'driving_score': completion * penalty,
        'infractions': dict(infractions),
        'final_risk': risk,
    }


def attach_episode_metrics(summary, metrics):
    """Add runner-side diagnostic metrics without changing the score formula."""
    enriched = dict(summary)
    for key, value in metrics.items():
        if value is not None:
            enriched[key] = value
    return enriched


def aggregate(episodes):
    if not episodes:
        raise ValueError('aggregate needs at least one episode')
    count = float(len(episodes))
    totals = {}
    for episode in episodes:
        for key, value in episode['infractions'].items():
            totals[key] = totals.get(key, 0) + value
    result = {
        'routes': len(episodes),
        'driving_score': sum(e['driving_score'] for e in episodes) / count,
        'route_completion': sum(e['route_completion'] for e in episodes) / count,
        'infraction_penalty': sum(e['infraction_penalty'] for e in episodes) / count,
        'success_rate': sum(1 for e in episodes if e['outcome'] == 'SUCCESS') / count,
        'infractions': totals,
    }
    for key in ('mean_speed_ms', 'stopped_time_s', 'min_ttc_true',
                'min_ttc_admitted', 'harsh_brake_count'):
        values = [e[key] for e in episodes if key in e and e[key] is not None]
        if values:
            result[key] = sum(values) / float(len(values))
    return result


def checkpoint_payload(episodes, summary, extra=None):
    """Emit the leaderboard checkpoint schema so one analyser reads both sources."""
    def as_lists(infractions):
        return {key: ['x'] * value for key, value in infractions.items()}

    payload = {
        '_checkpoint': {
            'records': [{
                'index': episode['seed'],
                'status': episode['outcome'],
                'scores': {'score_composed': episode['driving_score'],
                           'score_route': episode['route_completion'],
                           'score_penalty': episode['infraction_penalty']},
                'infractions': as_lists(episode['infractions']),
            } for episode in episodes],
            'global_record': {
                'scores': {'score_composed': summary['driving_score'],
                           'score_route': summary['route_completion'],
                           'score_penalty': summary['infraction_penalty']},
                'infractions': as_lists(summary['infractions']),
            },
            'progress': [len(episodes), len(episodes)],
        },
        'episodes': episodes,
    }
    payload.update(extra or {})
    return payload
