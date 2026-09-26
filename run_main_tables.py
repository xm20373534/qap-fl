from __future__ import annotations

"""Run the three QAP-FL main-table protocols.

The default backend is the bundled official C++ BLS executable. ``--backend
python`` is a portable fallback for smoke tests; it is not bit-for-bit
equivalent to the C++ implementation.
"""

import argparse
import csv
import hashlib
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from qap_fl.official_bls import official_bls
from qap_fl.qap import QAPInstance, compute_cost, is_valid_perm
from src.features import build_enhanced_features, build_relaxation_pair_features
from src.model import EnhancedAdditivePhiModel, FeaturePoolPhiModel

SYNTH_BUDGETS = {20: 2.0, 50: 7.0, 100: 40.0}
TAIX_BUDGETS = {27: 2.0, 45: 5.0, 75: 20.0, 125: 30.0, 175: 40.0}
QAP_HARD_FAMILIES = {"tai", "lipa", "sko", "tho", "wil"}
QUANTIZATION_SCALE = 100_000
TAIX_BEST = {
    **dict(zip([f"tai27e{i:02d}" for i in range(1, 21)],
               [2558,2850,3258,2822,3074,2814,3428,2430,2902,2994,2906,3070,2966,3568,2628,3124,3840,2758,2514,2638])),
    **dict(zip([f"tai45e{i:02d}" for i in range(1, 21)],
               [6412,5734,7438,6698,7274,6612,7526,6554,6648,8286,6510,7510,6120,6854,7394,6520,8806,6906,7170,6510])),
    **dict(zip([f"tai75e{i:02d}" for i in range(1, 21)],
               [14488,14444,14154,13694,12884,12534,13782,13948,12650,14192,15250,12760,13024,12604,14294,14204,13210,13500,12060,15260])),
    **dict(zip([f"tai125e{i:02d}" for i in range(1, 21)],
               [35426,36776,30498,33934,38432,35546,32712,36354,35008,34898,33082,32402,35432,30548,34328,33998,35606,39600,33034,31996])),
    **dict(zip([f"tai175e{i:02d}" for i in range(1, 21)],
               [59732,51464,54234,64506,51526,55768,53180,57334,53604,52040,56416,59704,60276,55736,49920,57266,59022,52152,52526,57014])),
}


def stable_seed(name: str) -> int:
    digest = hashlib.sha256(f"d403|{name}".encode()).digest()
    return 403_000 + int.from_bytes(digest[:4], "little") % 1_000_000_000


def load_qaplib(path: Path) -> QAPInstance:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    n = int(lines[0])
    values = [int(value) for line in lines[1:] for value in line.split()]
    if len(values) < 2 * n * n:
        raise ValueError(f"{path} has too few matrix entries")
    optimum = None
    sln = path.with_suffix(".sln")
    if sln.exists():
        tokens = sln.read_text(encoding="utf-8").split()
        if len(tokens) >= 2:
            optimum = float(tokens[1])
    return QAPInstance(path.stem, np.asarray(values[:n*n], dtype=np.float64).reshape(n, n),
                       np.asarray(values[n*n:2*n*n], dtype=np.float64).reshape(n, n), optimum)


def load_taix(path: Path) -> QAPInstance:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    n = int(lines[0])
    values = [int(value) for line in lines[1:] for value in line.split()]
    name = path.stem
    return QAPInstance(name, np.asarray(values[:n*n], dtype=np.float64).reshape(n, n),
                       np.asarray(values[n*n:2*n*n], dtype=np.float64).reshape(n, n),
                       float(TAIX_BEST[name]))


def uniform_instance(n: int, index: int) -> QAPInstance:
    seed = int.from_bytes(hashlib.sha256(f"uniform|{n}|{index}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    def symmetric() -> np.ndarray:
        upper = np.triu(rng.uniform(0.0, 1.0, size=(n, n)), 1)
        return upper + upper.T
    return QAPInstance(f"plma_uniform_n{n}_{index:03d}", symmetric(), symmetric(), None)


def sawt_instances(n: int, count: int) -> list[QAPInstance]:
    data = ROOT / "data" / "synthetic" / "sawt"
    flow = np.load(data / f"erdos{n}_0.7_F_test.npy", mmap_mode="r")
    positions = np.load(data / f"erdos{n}_0.7_positions_test.npy", mmap_mode="r")
    output = []
    for index in range(min(count, len(flow))):
        coords = np.asarray(positions[index], dtype=np.float64)
        distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        np.fill_diagonal(distance, 0.0)
        output.append(QAPInstance(f"sawt_geo_n{n}_{index:03d}", np.asarray(flow[index], dtype=np.float64), distance, None))
    return output


def reference_costs() -> dict[str, float]:
    output: dict[str, float] = {}
    for path in sorted((ROOT / "data" / "synthetic" / "references").glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                # Compare original floating-point costs after undoing the
                # fixed-point scale used by the C++ synthetic protocol.
                output[row["instance"]] = float(row["reference_quantized_rescaled_cost"])
    if len(output) != 1536:
        raise RuntimeError(f"expected 1,536 synthetic references, found {len(output)}")
    return output


def quantized(instance: QAPInstance) -> QAPInstance:
    return QAPInstance(instance.name + "_q", np.rint(instance.F * QUANTIZATION_SCALE).astype(np.int64),
                        np.rint(instance.D * QUANTIZATION_SCALE).astype(np.int64), None)


def maximize_assignment(score: np.ndarray) -> np.ndarray:
    rows, columns = linear_sum_assignment(-np.asarray(score, dtype=np.float64))
    if not np.array_equal(rows, np.arange(len(rows))):
        raise RuntimeError("unexpected Hungarian row order")
    return columns.astype(np.int64)


def hamming(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.asarray(a) != np.asarray(b)))


def select_elites(candidates: list[tuple[np.ndarray, float, int]], width: int = 4) -> list[tuple[np.ndarray, float, int]]:
    unique: dict[tuple[int, ...], tuple[np.ndarray, float, int]] = {}
    for perm, cost, source in candidates:
        key = tuple(int(value) for value in perm)
        if key not in unique or cost < unique[key][1]:
            unique[key] = (np.asarray(perm, dtype=np.int64), float(cost), int(source))
    ordered = sorted(unique.values(), key=lambda item: (item[1], tuple(item[0].tolist())))
    selected = []
    for item in ordered:
        if all(hamming(item[0], previous[0]) >= 0.2 for previous in selected):
            selected.append(item)
            if len(selected) == width:
                return selected
    for item in ordered:
        if not any(np.array_equal(item[0], previous[0]) for previous in selected):
            selected.append(item)
            if len(selected) == width:
                break
    if len(selected) != width:
        raise RuntimeError("could not construct an elite set")
    return selected


def load_model() -> torch.nn.Module:
    torch.set_num_threads(1)
    model = FeaturePoolPhiModel(EnhancedAdditivePhiModel(39, 39, 192, 4, 0.0), 8, 96, 3)
    checkpoint = torch.load(ROOT / "checkpoints" / "d389_assign_swap.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 4, 39), torch.zeros(1, 4, 39), torch.zeros(1, 4, 4, 8))
    return model


def model_score(model: torch.nn.Module, F: np.ndarray, D: np.ndarray) -> np.ndarray:
    node_f, node_d = build_enhanced_features(F, D)
    pair = build_relaxation_pair_features(F, D, faq_iterations=8, spectral_dimension=4)
    with torch.no_grad():
        output = model(torch.from_numpy(node_f)[None], torch.from_numpy(node_d)[None], torch.from_numpy(pair)[None])[0]
    value = output.numpy().astype(np.float64)
    return (value - value.mean()) / max(float(value.std()), 1e-8)


def map_permutation(internal: np.ndarray, facility_order: np.ndarray, location_order: np.ndarray) -> np.ndarray:
    output = np.empty_like(internal)
    output[facility_order] = location_order[internal]
    return output


def qapfl_candidates(instance: QAPInstance, model: torch.nn.Module, seed: int) -> list[tuple[np.ndarray, float, str, int]]:
    order_rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 9_000_001)
    orders = [(order_rng.permutation(instance.n), order_rng.permutation(instance.n)) for _ in range(4)]
    noises = [noise_rng.gumbel(size=(instance.n, instance.n)).astype(np.float64) for _ in range(80)]
    learned, random = [], []
    for orbit, (fo, lo) in enumerate(orders):
        score = model_score(model, instance.F[np.ix_(fo, fo)], instance.D[np.ix_(lo, lo)])
        for local in range(10):
            noise = 0.0 if local == 0 else noises[orbit*10+local]
            perm = map_permutation(maximize_assignment(score + noise), fo, lo)
            learned.append((perm, compute_cost(perm, instance.F, instance.D), orbit))
    for index, noise in enumerate(noises[40:]):
        perm = maximize_assignment(noise)
        random.append((perm, compute_cost(perm, instance.F, instance.D), 100 + index))
    learned = select_elites(learned)
    random = select_elites(random)
    return [(*learned[i][:2], "qap-fl", learned[i][2]) for i in range(4)] + [(*random[i][:2], "random", random[i][2]) for i in range(4)]


def write_payload(path: Path, instance: QAPInstance) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"0 {instance.n}\n")
        for matrix in (instance.F, instance.D):
            for row in np.asarray(matrix, dtype=np.int64):
                handle.write(" ".join(str(int(value)) for value in row) + "\n")


def write_perm(path: Path, perm: np.ndarray) -> None:
    path.write_text(" ".join(str(int(value)) for value in perm) + "\n", encoding="ascii")


def run_backend(instance: QAPInstance, perm: np.ndarray, seed: int, seconds: float, backend: str, exe: Path, payload: Path, initial_path: Path) -> tuple[np.ndarray, float, float]:
    started = time.perf_counter()
    if backend == "cpp":
        write_perm(initial_path, perm)
        native = max(0.005, seconds - 0.05)
        with payload.open("r", encoding="ascii") as stream:
            completed = subprocess.run([str(exe), str(int(seed % 2_147_483_647)), "3600", "--initial_path", str(initial_path), "--wall_seconds", f"{native:.7f}", "--emit_perm"], stdin=stream, capture_output=True, encoding="ascii", errors="replace", check=True, timeout=seconds + 15)
        costs = list(re.finditer(r"Solution cost\s+(-?\d+)", completed.stdout))
        match = re.search(r"Final perm\s+([0-9 ]+)", completed.stdout)
        if not costs or not match:
            raise RuntimeError(f"BLS output could not be parsed: {completed.stdout[-500:]}")
        result_perm = np.fromiter((int(x) for x in match.group(1).split()), dtype=np.int64)
        cost = float(int(costs[-1].group(1)))
    else:
        result = official_bls(instance, seed=int(seed), initial_perm=np.asarray(perm, dtype=np.int64), max_time_sec=max(seconds, 1e-4), stop_at_optimum=False, method="full", delta_update_mode="incremental", wall_clock_includes_initialization=True)
        result_perm, cost = np.asarray(result.perm, dtype=np.int64), float(result.cost)
    if not is_valid_perm(result_perm, instance.n):
        raise RuntimeError("backend returned an invalid permutation")
    return result_perm, cost, time.perf_counter() - started


def run_one(instance: QAPInstance, budget: float, model: torch.nn.Module, backend: str, exe: Path, reference: float | None) -> list[dict[str, object]]:
    is_synthetic = instance.name.startswith(("sawt_", "plma_"))
    search_instance = quantized(instance) if is_synthetic else instance
    seed = stable_seed(instance.name)
    with tempfile.TemporaryDirectory(prefix="qapfl_") as temp_dir:
        temp = Path(temp_dir)
        payload, initial_path = temp / "bls.txt", temp / "initial.txt"
        write_payload(payload, search_instance)
        random_perm = np.random.default_rng(seed + 17_001).permutation(instance.n).astype(np.int64)
        result_perm, _, elapsed = run_backend(search_instance, random_perm, seed + 292_000, budget, backend, exe, payload, initial_path)
        outcomes = [("BLS", result_perm, elapsed, 1, 1, elapsed)]
        started = time.perf_counter()
        elites = qapfl_candidates(instance, model, seed)
        construction = time.perf_counter() - started
        selected = min(elites, key=lambda item: compute_cost(item[0], search_instance.F, search_instance.D))
        probe_results = []
        for lane, (perm, _, source, _) in enumerate(elites):
            remaining = budget - (time.perf_counter() - started)
            reserve = max(0.0, budget - 8 * min(1.0, budget * 0.05))
            call_budget = min(min(1.0, budget * 0.05), max(1e-4, (remaining - reserve) / max(1, 8-lane)))
            p, cost, elapsed = run_backend(search_instance, perm, seed * 1_000_003 + lane * 1009 + 252_000, call_budget, backend, exe, payload, initial_path)
            probe_results.append((source, lane, p, cost, elapsed))
        survivors = [min((item for item in probe_results if item[0] == source), key=lambda item: item[3]) for source in ("qap-fl", "random")]
        best_perm = selected[0]
        best_cost = compute_cost(best_perm, search_instance.F, search_instance.D)
        total_calls = len(probe_results)
        for index, (source, lane, p, _, _) in enumerate(survivors):
            remaining = max(1e-4, budget - (time.perf_counter() - started))
            p2, cost2, _ = run_backend(search_instance, p, seed * 1_000_003 + lane * 1009 + 2_520_000, remaining / max(1, 2-index), backend, exe, payload, initial_path)
            total_calls += 1
            if cost2 < best_cost:
                best_perm, best_cost = p2, cost2
        outcomes.append(("QAP-FL", best_perm, time.perf_counter() - started, total_calls, 80, construction))
    rows = []
    dataset = "synthetic" if is_synthetic else "taix" if instance.name.startswith("tai") else "qaplib"
    for method, perm, elapsed, starts, candidates, construction_time in outcomes:
        cost = compute_cost(perm, instance.F, instance.D)
        gap = "" if reference is None else 100.0 * (cost - reference) / abs(reference)
        rows.append({"dataset": dataset, "instance": instance.name, "n": instance.n, "method": method, "budget_sec": budget, "reference_cost": "" if reference is None else reference, "cost": cost, "gap_percent": gap, "runtime_sec": elapsed, "construction_runtime_sec": construction_time, "n_starts": starts, "candidate_count": candidates})
    return rows


def qaplib_budget(name: str, n: int) -> float:
    family_match = re.match(r"[A-Za-z]+", name)
    family = family_match.group(0).lower() if family_match else "unknown"
    if family == "tai":
        return 10.0 if n <= 100 else 20.0 if n <= 175 else 40.0
    if family in QAP_HARD_FAMILIES:
        return 10.0 if n <= 100 else 20.0
    return 4.0 if n <= 100 else 10.0


def load_tasks(dataset: str, limit: int, samples: int) -> list[tuple[QAPInstance, float, float | None]]:
    tasks = []
    if dataset == "synthetic":
        refs = reference_costs()
        for dist in ("sawt", "plma"):
            for n in (20, 50, 100):
                instances = sawt_instances(n, samples) if dist == "sawt" else [uniform_instance(n, i) for i in range(samples)]
                for item in instances:
                    suffix = item.name.rsplit("_", 1)[-1]
                    ref_name = f"{dist}_{'geo' if dist == 'sawt' else 'uniform'}_n{n}_{suffix}"
                    ref = refs.get(ref_name)
                    tasks.append((item, SYNTH_BUDGETS[n], ref))
    elif dataset == "taix":
        for n in (27, 45, 75, 125, 175):
            root = ROOT / "data" / "taixxeyy" / ("tai175e" if n == 175 else "")
            for path in sorted(root.glob(f"tai{n}e*.dat")):
                item = load_taix(path)
                tasks.append((item, TAIX_BUDGETS[n], item.optimum))
    else:
        for path in sorted((ROOT / "data" / "qaplib").glob("*.dat")):
            item = load_qaplib(path)
            tasks.append((item, qaplib_budget(item.name, item.n), item.optimum))
    return tasks[:limit] if limit > 0 else tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="QAP-FL reproducibility runner for the three main tables")
    parser.add_argument("--dataset", choices=("synthetic", "taix", "qaplib"), required=True)
    parser.add_argument("--limit", type=int, default=0, help="run only first N instances; useful for smoke tests")
    parser.add_argument("--samples", type=int, default=256, help="synthetic instances per distribution and size")
    parser.add_argument("--backend", choices=("auto", "cpp", "python"), default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    exe = ROOT / "bin" / "d301_official_bls_racing.exe"
    backend = "cpp" if args.backend == "auto" and exe.exists() else "python" if args.backend == "auto" else args.backend
    if backend == "cpp" and not exe.exists():
        raise FileNotFoundError(f"missing backend executable: {exe}")
    model = load_model()
    tasks = load_tasks(args.dataset, args.limit, args.samples)
    if not tasks:
        raise RuntimeError("no tasks selected")
    rows = []
    for index, (instance, budget, reference) in enumerate(tasks, 1):
        result = run_one(instance, budget, model, backend, exe, reference)
        rows.extend(result)
        print(f"{index}/{len(tasks)} {instance.name} " + " ".join(f"{r['method']}={float(r['gap_percent'] or 0):+.4f}%/{float(r['runtime_sec']):.2f}s" for r in result), flush=True)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {output} rows={len(rows)} backend={backend}")


if __name__ == "__main__":
    main()
