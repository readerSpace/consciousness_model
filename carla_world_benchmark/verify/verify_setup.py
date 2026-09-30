#!/usr/bin/env python
"""Preflight for the CARLA environment.

    . .\\scripts\\env.ps1
    python verify\\verify_setup.py            # full check, needs a running server
    python verify\\verify_setup.py --offline  # skip everything needing the server

Checks are split into two groups.  REQUIRED checks gate the standalone runner
(``runner/carla_runner.py``), which is the path the shipped CARLA package
supports.  OPTIONAL checks cover the Leaderboard 2.0 stack and the Town12 /
Town13 maps it needs; they are expected to fail with the shipped package alone
and are reported as WARN, not FAIL.  The exit code reflects only the required
group.
"""

from __future__ import print_function

import argparse
import os
import sys
import traceback

LEADERBOARD_MAPS = ('Town12', 'Town13')

_VERIFY_DIR = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.dirname(_VERIFY_DIR)
_PROJECT_ROOT = os.path.dirname(_BENCH_ROOT)
for _path in (_PROJECT_ROOT, os.path.join(_BENCH_ROOT, 'agent'),
              os.path.join(_BENCH_ROOT, 'runner')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

_results = []


class SkipCheck(Exception):
    pass


def check(name, required=True):
    def decorate(function):
        def run(*args, **kwargs):
            try:
                detail = function(*args, **kwargs)
            except SkipCheck as skip:
                print('SKIP  {:<32} {}'.format(name, skip))
                _results.append((name, None, required))
                return None
            except Exception as error:  # a preflight reports, it never raises
                print('{}  {:<32} {}: {}'.format(
                    'FAIL' if required else 'WARN', name, type(error).__name__, error))
                if os.environ.get('VERIFY_TRACEBACK'):
                    traceback.print_exc()
                _results.append((name, False, required))
                return None
            print('PASS  {:<32} {}'.format(name, detail if detail is not None else ''))
            _results.append((name, True, required))
            return detail
        return run
    return decorate


# -- required: what the standalone runner needs ------------------------------

@check('python version')
def check_python():
    version = sys.version_info
    # 3.12  the wheel shipped inside the 0.9.16 package -> standalone runner
    # 3.10  PyPI's carla 0.9.16 client -> official leaderboard evaluator
    # 3.8/3.7  the 0.9.14 Leaderboard 2.0 package, if that ever returns
    supported = {(3, 12): 'standalone runner', (3, 10): 'leaderboard evaluator',
                 (3, 8): 'leaderboard 0.9.14', (3, 7): 'leaderboard 0.9.14'}
    role = supported.get(version[:2])
    if role is None:
        print('WARN  python {}.{} is not one of 3.12 / 3.10 / 3.8; no CARLA client '
              'wheel targets it'.format(version[0], version[1]))
    else:
        print('      interpreter role: {}'.format(role))
    return 'python {}.{}.{}'.format(*version[:3])


@check('environment variables')
def check_env():
    root = os.environ.get('CARLA_ROOT')
    if not root:
        raise RuntimeError('CARLA_ROOT unset (dot-source scripts/env.ps1)')
    if not os.path.isdir(root):
        raise RuntimeError('CARLA_ROOT points at a missing directory: {}'.format(root))
    # CarlaUE4.exe on Windows, CarlaUE4.sh in WSL / Linux.
    launchers = [name for name in ('CarlaUE4.exe', 'CarlaUE4.sh')
                 if os.path.exists(os.path.join(root, name))]
    if not launchers:
        raise RuntimeError('no CarlaUE4.exe or CarlaUE4.sh under {} '
                           '(run the setup script for your platform)'.format(root))
    return '{}  [{}]'.format(root, launchers[0])


@check('import carla')
def check_carla():
    import carla
    return 'carla {}'.format(getattr(carla, '__version__', '(no __version__)'))


@check('import BasicAgent')
def check_basic_agent():
    from agents.navigation.basic_agent import BasicAgent  # noqa: F401
    from agents.navigation.local_planner import LocalPlanner  # noqa: F401
    return 'PythonAPI/carla on PYTHONPATH'


@check('import Consciousness_model')
def check_model():
    from Consciousness_model.consciousness import FiniteWorkspace
    return 'capacity={}'.format(FiniteWorkspace(capacity=4).capacity)


@check('bridge smoke decision')
def check_bridge():
    from carla_consciousness_bridge import (ConsciousnessDrivingPolicy, HazardObservation,
                                            PolicyConfig)
    policy = ConsciousnessDrivingPolicy(PolicyConfig(capacity=2))
    hazards = [HazardObservation('v{}'.format(i), 'vehicle', 5.0 + 4.0 * i, 0.3 * i, 9.0 - i)
               for i in range(5)]
    decision = policy.step(ego_speed=10.0, hazards=hazards)
    if len(decision.admitted) != 2:
        raise RuntimeError('expected 2 admitted, got {}'.format(decision.admitted))
    return 'admitted={} brake={:.2f} goal={}'.format(
        decision.admitted, decision.brake, decision.goal)


@check('import runner')
def check_runner():
    import scoring  # noqa: F401
    import carla_perception  # noqa: F401
    import carla_runner  # noqa: F401
    return 'runner/carla_runner.py importable'


@check('carla server connection')
def check_server(host, port, offline):
    if offline:
        raise SkipCheck('--offline')
    import carla
    client = carla.Client(host, port)
    client.set_timeout(30.0)
    return 'server {} / client {}'.format(client.get_server_version(), client.get_client_version())


@check('maps installed')
def check_maps(host, port, offline):
    if offline:
        raise SkipCheck('--offline')
    import carla
    client = carla.Client(host, port)
    client.set_timeout(60.0)
    available = sorted({name.split('/')[-1] for name in client.get_available_maps()})
    if not available:
        raise RuntimeError('server reported no maps')
    return ', '.join(available)


# -- optional: the Leaderboard 2.0 path --------------------------------------

@check('import scenario_runner', required=False)
def check_srunner():
    from srunner.scenariomanager.carla_data_provider import CarlaDataProvider  # noqa: F401
    return 'srunner ok'


@check('import leaderboard', required=False)
def check_leaderboard():
    from leaderboard.autoagents.autonomous_agent import AutonomousAgent  # noqa: F401
    from leaderboard.leaderboard_evaluator import LeaderboardEvaluator  # noqa: F401
    return 'leaderboard ok'


@check('leaderboard maps present', required=False)
def check_leaderboard_maps(host, port, offline, available):
    if offline:
        raise SkipCheck('--offline')
    if available is None:
        raise SkipCheck('map list unavailable')
    missing = [town for town in LEADERBOARD_MAPS if town not in available]
    if missing:
        raise RuntimeError(
            '{} not installed, so leaderboard-2.0 routes cannot run. '
            'Install the AdditionalMaps package, or use runner/carla_runner.py '
            'on an installed map.'.format(', '.join(missing)))
    return 'Town12 and Town13 available'


@check('routes reference known towns', required=False)
def check_routes():
    import xml.etree.ElementTree as ElementTree
    root = os.environ.get('LEADERBOARD_ROOT') or os.path.join(
        _BENCH_ROOT, 'external', 'leaderboard')
    data = os.path.join(root, 'data')
    if not os.path.isdir(data):
        raise RuntimeError('no routes directory at {}'.format(data))
    towns = set()
    for name in sorted(os.listdir(data)):
        if name.endswith('.xml'):
            for route in ElementTree.parse(os.path.join(data, name)).getroot().iter('route'):
                towns.add(route.get('town'))
    return 'routes use {}'.format(', '.join(sorted(towns)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=int(os.environ.get('CARLA_PORT', 2000)))
    parser.add_argument('--offline', action='store_true',
                        help='skip checks that need a running CARLA server')
    args = parser.parse_args()

    print('--- required (standalone runner) ---')
    check_python()
    check_env()
    check_carla()
    check_basic_agent()
    check_model()
    check_bridge()
    check_runner()
    check_server(args.host, args.port, args.offline)
    maps = check_maps(args.host, args.port, args.offline)
    available = set(maps.split(', ')) if maps else None

    print('--- optional (leaderboard 2.0) ---')
    check_srunner()
    check_leaderboard()
    check_leaderboard_maps(args.host, args.port, args.offline, available)
    check_routes()

    passed = [name for name, ok, _ in _results if ok is True]
    failed = [name for name, ok, required in _results if ok is False and required]
    warned = [name for name, ok, required in _results if ok is False and not required]
    skipped = [name for name, ok, _ in _results if ok is None]
    print('')
    print('{} passed, {} failed, {} optional-missing, {} skipped'.format(
        len(passed), len(failed), len(warned), len(skipped)))
    if warned:
        print('optional missing: {}'.format(', '.join(warned)))
        print('  -> the standalone runner is unaffected; see README section 6.')
    if failed:
        print('failed: {}'.format(', '.join(failed)))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
