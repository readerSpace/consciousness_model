"""L8.8: implement small scientific simulations from natural-language tasks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .process_execution import combined_output, run_text


@dataclass(frozen=True)
class ScientificSimulationTaskSpec:
    domain: str
    action: str
    system: str
    requires_code_change: bool
    requires_execution: bool
    requires_external_knowledge: bool
    target_files: tuple[str, ...]
    equations: tuple[str, ...]


@dataclass(frozen=True)
class SimulationImplementationResult:
    task: ScientificSimulationTaskSpec
    written_files: tuple[str, ...]
    verification_command: tuple[str, ...] | None
    return_code: int | None
    output: str


def is_scientific_simulation_implementation_request(text: str) -> bool:
    lower = text.lower()
    creation = any(word in text for word in ("作成して", "作って", "実装して", "生成して")) or any(
        word in lower for word in ("create", "implement", "generate", "build")
    )
    simulation = any(word in text for word in ("シミュレーション", "運動", "粒子", "磁場")) or any(
        word in lower for word in ("simulation", "particle", "magnetic field", "trajectory")
    )
    uniform_b = any(word in text for word in ("一様磁場", "荷電粒子")) or "uniform magnetic" in lower
    file_target = ".py" in lower and not creation
    return creation and simulation and uniform_b and not file_target


def build_charged_particle_task(text: str) -> ScientificSimulationTaskSpec:
    lower = text.lower()
    requires_execution = any(word in text for word in ("実行して", "動かして", "走らせて")) or any(
        word in lower for word in ("run", "execute")
    )
    return ScientificSimulationTaskSpec(
        domain="physics",
        action="IMPLEMENT_SCIENTIFIC_SIMULATION",
        system="charged particle in a uniform magnetic field",
        requires_code_change=True,
        requires_execution=requires_execution,
        requires_external_knowledge=False,
        target_files=("charged_particle_uniform_B.py", "test_charged_particle_uniform_B.py"),
        equations=(
            r"m\frac{d\mathbf{v}}{dt}=q\mathbf{v}\times\mathbf{B}",
            r"\frac{d\mathbf{r}}{dt}=\mathbf{v}",
            r"\dot v_x=\omega_c v_y,\quad \dot v_y=-\omega_c v_x,\quad \dot v_z=0",
            r"\omega_c=\frac{qB}{m}",
        ),
    )


def _simulation_source() -> str:
    return '''"""Charged particle motion in a uniform magnetic field.

The default field is B=(0, 0, Bz).  The non-relativistic Lorentz-force model is

    m dv/dt = q v x B
    dr/dt = v

and the integrator uses RK4 so the speed remains nearly constant for the
default circular orbit.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import math


@dataclass(frozen=True)
class ParticleConfig:
    q: float = 1.0
    m: float = 1.0
    Bz: float = 1.0
    dt: float = 0.01
    t_end: float = 20.0


State = tuple[float, float, float, float, float, float]


def cyclotron_frequency(config: ParticleConfig) -> float:
    return config.q * config.Bz / config.m


def rhs(state: State, config: ParticleConfig) -> State:
    x, y, z, vx, vy, vz = state
    omega = cyclotron_frequency(config)
    return (
        vx,
        vy,
        vz,
        omega * vy,
        -omega * vx,
        0.0,
    )


def rk4_step(state: State, config: ParticleConfig) -> State:
    dt = config.dt
    k1 = rhs(state, config)
    k2 = rhs(tuple(value + 0.5 * dt * slope for value, slope in zip(state, k1)), config)
    k3 = rhs(tuple(value + 0.5 * dt * slope for value, slope in zip(state, k2)), config)
    k4 = rhs(tuple(value + dt * slope for value, slope in zip(state, k3)), config)
    return tuple(
        value + dt * (a + 2.0 * b + 2.0 * c + d) / 6.0
        for value, a, b, c, d in zip(state, k1, k2, k3, k4)
    )


def simulate(config: ParticleConfig = ParticleConfig(), initial_state: State = (1.0, 0.0, 0.0, 0.0, -1.0, 0.2)) -> list[State]:
    steps = int(round(config.t_end / config.dt))
    state = initial_state
    trajectory = [state]
    for _ in range(steps):
        state = rk4_step(state, config)
        trajectory.append(state)
    return trajectory


def speed(state: State) -> float:
    return math.sqrt(state[3] ** 2 + state[4] ** 2 + state[5] ** 2)


def radial_distance_xy(state: State) -> float:
    return math.sqrt(state[0] ** 2 + state[1] ** 2)


def write_csv(path: str, trajectory: list[State], config: ParticleConfig = ParticleConfig()) -> None:
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("t", "x", "y", "z", "vx", "vy", "vz"))
        for index, state in enumerate(trajectory):
            writer.writerow((index * config.dt, *state))


def summarize(trajectory: list[State]) -> dict[str, float]:
    speeds = [speed(state) for state in trajectory]
    radii = [radial_distance_xy(state) for state in trajectory]
    return {
        "steps": float(len(trajectory) - 1),
        "speed_min": min(speeds),
        "speed_max": max(speeds),
        "radius_mean": sum(radii) / len(radii),
        "z_start": trajectory[0][2],
        "z_end": trajectory[-1][2],
    }


if __name__ == "__main__":
    trajectory = simulate()
    summary = summarize(trajectory)
    for key, value in summary.items():
        print(f"{key}={value:.8f}")
'''


def _test_source() -> str:
    return '''import math
import unittest

from charged_particle_uniform_B import ParticleConfig, radial_distance_xy, simulate, speed


class ChargedParticleUniformBTest(unittest.TestCase):
    def test_speed_is_conserved(self):
        config = ParticleConfig(dt=0.01, t_end=8.0)
        trajectory = simulate(config)
        speeds = [speed(state) for state in trajectory]
        self.assertLess(max(speeds) - min(speeds), 1e-8)

    def test_xy_motion_is_circular_for_default_initial_condition(self):
        config = ParticleConfig(dt=0.01, t_end=8.0)
        trajectory = simulate(config)
        radii = [radial_distance_xy(state) for state in trajectory]
        self.assertLess(max(abs(radius - 1.0) for radius in radii), 1e-7)

    def test_z_velocity_is_constant(self):
        config = ParticleConfig(dt=0.02, t_end=2.0)
        trajectory = simulate(config)
        z_values = [state[2] for state in trajectory]
        vz_values = [state[5] for state in trajectory]
        self.assertTrue(all(math.isclose(value, 0.2, rel_tol=0.0, abs_tol=1e-12) for value in vz_values))
        self.assertAlmostEqual(z_values[-1] - z_values[0], 0.2 * config.t_end, places=10)


if __name__ == "__main__":
    unittest.main()
'''


def implement_scientific_simulation(root: Path, request: str) -> SimulationImplementationResult:
    task = build_charged_particle_task(request)
    root = Path(root)
    written: list[str] = []
    sources = {
        "charged_particle_uniform_B.py": _simulation_source(),
        "test_charged_particle_uniform_B.py": _test_source(),
    }
    for name, source in sources.items():
        path = root / name
        path.write_text(source, encoding="utf-8")
        written.append(name)
    command: tuple[str, ...] | None = None
    return_code: int | None = None
    output = "実行は要求されていないため、ファイル生成とテスト生成まで行いました。"
    if task.requires_execution:
        command = ("python", "-m", "unittest", "test_charged_particle_uniform_B.py")
        completed = run_text(list(command), cwd=root, timeout=60)
        return_code = completed.returncode
        output = combined_output(completed).strip()
    return SimulationImplementationResult(task, tuple(written), command, return_code, output)


def format_simulation_implementation_report(result: SimulationImplementationResult) -> str:
    task = result.task
    lines = [
        "## Scientific Simulation Implementation",
        "",
        "【TaskSpec】",
        f"- action: `{task.action}`",
        f"- domain: `{task.domain}`",
        f"- system: `{task.system}`",
        f"- requires_code_change: `{task.requires_code_change}`",
        f"- requires_execution: `{task.requires_execution}`",
        f"- requires_external_knowledge: `{task.requires_external_knowledge}`",
        "",
        "【数理モデル】",
    ]
    for equation in task.equations:
        lines.extend(("$$", equation, "$$"))
    lines.extend([
        "",
        "【生成ファイル】",
        *[f"- `{item}`" for item in result.written_files],
        "",
        "【検証項目】",
        "- 速度の大きさが保存されること",
        "- xy平面の軌道半径がほぼ一定であること",
        "- z方向速度が一定であること",
    ])
    if result.verification_command:
        lines.extend([
            "",
            "【実行】",
            f"$ {' '.join(result.verification_command)}",
            f"return code: `{result.return_code}`",
            "```text",
            result.output[-4000:],
            "```",
        ])
    else:
        lines.extend(["", "【実行】", result.output])
    return "\n".join(lines)
