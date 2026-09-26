from __future__ import annotations

from pathlib import Path

import numpy as np

from .qap import QAPInstance


def load_qaplib_instance(name: str, data_dir: str | Path) -> QAPInstance:
    data_dir = Path(data_dir)
    dat_path = data_dir / f"{name}.dat"
    sln_path = data_dir / f"{name}.sln"
    if not dat_path.exists():
        raise FileNotFoundError(f"QAPLIB data file not found: {dat_path}")

    with dat_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    n = int(lines[0])
    values: list[int] = []
    for line in lines[1:]:
        values.extend(int(x) for x in line.split())
    expected = 2 * n * n
    if len(values) < expected:
        raise ValueError(f"{dat_path} has {len(values)} values, expected at least {expected}.")

    F = np.asarray(values[: n * n], dtype=np.float64).reshape(n, n)
    D = np.asarray(values[n * n : 2 * n * n], dtype=np.float64).reshape(n, n)
    optimum = None
    if sln_path.exists():
        first = sln_path.read_text(encoding="utf-8").strip().split()
        if len(first) >= 2:
            optimum = float(first[1])
    return QAPInstance(name=name, F=F, D=D, optimum=optimum)


def load_qaplib_instances(names: list[str], data_dir: str | Path) -> list[QAPInstance]:
    return [load_qaplib_instance(name, data_dir) for name in names]

