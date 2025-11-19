"""Run OpenFOAM solvers (blockMesh → setFields → interFoam) for all cases.

Iterates over case directories, invokes the three commands sequentially, and
logs success / failure for each case without aborting the whole sweep.

Usage
-----
python scripts/run_simulations.py \\
    --cases-dir simulations/ \\
    --np 1
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _openfoam_available() -> bool:
    return shutil.which("interFoam") is not None


def _run_command(
    cmd: list[str],
    cwd: Path,
    log_path: Path,
) -> bool:
    """Run a shell command; write stdout+stderr to log_path; return success."""
    with log_path.open("w") as fh:
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=fh,
                stderr=subprocess.STDOUT,
                timeout=3600,  # 1-hour hard limit per solver call
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            log.error("Timeout: %s in %s", " ".join(cmd), cwd)
            return False
        except FileNotFoundError:
            log.error("Command not found: %s", cmd[0])
            return False


def run_case(case_dir: Path, n_parallel: int = 1) -> bool:
    """Run blockMesh, setFields, interFoam for one case. Returns True on success."""
    log_dir = case_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    steps = [
        (["blockMesh"], "blockMesh.log"),
        (["setFields"], "setFields.log"),
    ]

    if n_parallel > 1:
        steps += [
            (["decomposePar"], "decomposePar.log"),
            (
                ["mpirun", "-np", str(n_parallel), "interFoam", "-parallel"],
                "interFoam.log",
            ),
            (["reconstructPar"], "reconstructPar.log"),
        ]
    else:
        steps += [(["interFoam"], "interFoam.log")]

    for cmd, log_name in steps:
        log.info("  %-20s  →  %s", " ".join(cmd), case_dir.name)
        ok = _run_command(cmd, case_dir, log_dir / log_name)
        if not ok:
            log.error("  FAILED: %s — see %s", " ".join(cmd), log_dir / log_name)
            return False

    return True


def load_config(cases_dir: Path) -> list[dict]:
    config_path = cases_dir / "cases_config.json"
    if not config_path.exists():
        log.error("Config not found: %s", config_path)
        sys.exit(1)
    with config_path.open() as f:
        return json.load(f)


def run_all(cases_dir: Path, n_parallel: int = 1) -> None:
    if not _openfoam_available():
        log.warning(
            "OpenFOAM not found on PATH. Skipping simulations.\n"
            "Use sample_data/ to run the ML pipeline without a solver."
        )
        return

    cases = load_config(cases_dir)
    results: list[dict] = []
    t0_global = time.perf_counter()

    for meta in cases:
        case_dir = Path(meta["case_dir"])
        if not case_dir.exists():
            log.warning("Case directory missing: %s — skipping", case_dir)
            results.append({**meta, "status": "missing"})
            continue

        log.info("Running %s ...", meta["case_id"])
        t0 = time.perf_counter()
        ok = run_case(case_dir, n_parallel)
        elapsed = time.perf_counter() - t0

        status = "ok" if ok else "failed"
        log.info(
            "%s  status=%-6s  elapsed=%.1f s", meta["case_id"], status, elapsed
        )
        results.append({**meta, "status": status, "elapsed_s": round(elapsed, 1)})

    # Summary
    n_ok = sum(1 for r in results if r["status"] == "ok")
    total_elapsed = time.perf_counter() - t0_global
    log.info(
        "Done: %d/%d cases succeeded in %.1f s total",
        n_ok,
        len(results),
        total_elapsed,
    )

    # Write run summary
    summary_path = cases_dir / "run_summary.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)
    log.info("Run summary → %s", summary_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cases-dir",
        type=Path,
        default=REPO_ROOT / "simulations",
        help="Directory containing case folders and cases_config.json",
    )
    p.add_argument(
        "--np",
        type=int,
        default=1,
        dest="n_parallel",
        help="MPI ranks for parallel interFoam (default: 1 = serial)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_all(args.cases_dir, args.n_parallel)


if __name__ == "__main__":
    main()
