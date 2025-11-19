"""Extract wave-front position and max velocity from OpenFOAM case output.

Reads native OpenFOAM ASCII field files (alpha.water, U) at each saved
time step, computes two scalar quantities per time step, and writes a CSV
per case.

Falls back gracefully if OpenFOAM output is absent (e.g., sample_data/).

Usage
-----
python scripts/extract_fields.py \\
    --cases-dir   simulations/ \\
    --output-dir  data/extracted/
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
ALPHA_THRESHOLD = 0.5   # VOF threshold for water/air interface detection


# ---------------------------------------------------------------------------
# OpenFOAM native ASCII reader
# ---------------------------------------------------------------------------

_INTERNAL_FIELD_RE = re.compile(
    r"internalField\s+nonuniform\s+List<(?:scalar|vector)>\s*\n(\d+)\n\((.+?)\)",
    re.DOTALL,
)


def _read_scalar_field(path: Path) -> np.ndarray | None:
    """Parse a nonuniform List<scalar> internalField from an OF ASCII file."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None

    m = _INTERNAL_FIELD_RE.search(text)
    if not m:
        # Might be uniform
        m_unif = re.search(r"internalField\s+uniform\s+([\d.eE+\-]+)", text)
        if m_unif:
            return np.array([float(m_unif.group(1))])
        return None

    n = int(m.group(1))
    values = np.fromstring(m.group(2), sep="\n", dtype=np.float64)
    if len(values) != n:
        log.warning("Field size mismatch in %s (%d vs %d)", path, len(values), n)
    return values


def _read_vector_field(path: Path) -> np.ndarray | None:
    """Parse a nonuniform List<vector> internalField → shape (N, 3)."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None

    m = _INTERNAL_FIELD_RE.search(text)
    if not m:
        m_unif = re.search(
            r"internalField\s+uniform\s+\(([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\)",
            text,
        )
        if m_unif:
            vals = [float(m_unif.group(i)) for i in range(1, 4)]
            return np.array([vals])
        return None

    n = int(m.group(1))
    # Each row is "(ux uy uz)"
    rows = re.findall(
        r"\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)",
        m.group(2),
    )
    if not rows:
        return None
    arr = np.array([[float(v) for v in row] for row in rows])
    if arr.shape[0] != n:
        log.warning("Vector field size mismatch in %s (%d vs %d)", path, arr.shape[0], n)
    return arr


def _cell_x_coords(case_dir: Path, n_cells: int) -> np.ndarray:
    """Return approximate x-coordinate for each cell using blockMesh geometry.

    Falls back to linearly spaced [0, 4] if mesh files are unavailable.
    """
    # Try reading cellCentres from mesh (written by writeMeshObj or checkMesh)
    centres_path = case_dir / "constant" / "polyMesh" / "cellCentres"
    if centres_path.exists():
        coords = _read_vector_field(centres_path)
        if coords is not None and coords.shape[0] == n_cells:
            return coords[:, 0]

    # Fallback: reconstruct from blockMeshDict
    bm_path = case_dir / "system" / "blockMeshDict"
    domain_x = 4.0
    if bm_path.exists():
        text = bm_path.read_text()
        m = re.search(r"xMax\s+([\d.]+)", text)
        if m:
            domain_x = float(m.group(1))

    # Assume 160 × 80 mesh; cells ordered row by row (x-major)
    nx, ny = 160, 80
    if n_cells == nx * ny:
        xs = np.linspace(domain_x / (2 * nx), domain_x - domain_x / (2 * nx), nx)
        return np.tile(xs, ny)

    # Last resort: uniform spacing
    return np.linspace(0.0, domain_x, n_cells)


# ---------------------------------------------------------------------------
# Per-case extraction
# ---------------------------------------------------------------------------

def extract_case(case_dir: Path) -> pd.DataFrame | None:
    """Extract scalar quantities for all time steps in one case directory."""
    time_dirs = sorted(
        [
            d
            for d in case_dir.iterdir()
            if d.is_dir() and _is_numeric(d.name) and float(d.name) > 0
        ],
        key=lambda d: float(d.name),
    )

    if not time_dirs:
        log.warning("No time directories found in %s", case_dir)
        return None

    rows: list[dict] = []
    x_coords: np.ndarray | None = None

    for t_dir in time_dirs:
        t = float(t_dir.name)
        alpha_path = t_dir / "alpha.water"
        u_path = t_dir / "U"

        alpha = _read_scalar_field(alpha_path)
        u_vec = _read_vector_field(u_path)

        if alpha is None or u_vec is None:
            log.debug("Skipping t=%.3f in %s (missing fields)", t, case_dir.name)
            continue

        n_cells = alpha.shape[0]
        if x_coords is None or x_coords.shape[0] != n_cells:
            x_coords = _cell_x_coords(case_dir, n_cells)

        # Wave front: max x where alpha > threshold
        water_mask = alpha > ALPHA_THRESHOLD
        wave_front_x = float(x_coords[water_mask].max()) if water_mask.any() else 0.0

        # Max velocity magnitude
        u_mag = np.linalg.norm(u_vec, axis=1)
        max_velocity = float(u_mag.max())

        rows.append(
            {
                "time": t,
                "wave_front_x": wave_front_x,
                "max_velocity": max_velocity,
            }
        )

    if not rows:
        log.warning("No valid time steps extracted from %s", case_dir)
        return None

    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def _is_numeric(name: str) -> bool:
    try:
        float(name)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract_all(cases_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = cases_dir / "cases_config.json"
    if not config_path.exists():
        log.error("cases_config.json not found in %s", cases_dir)
        return

    with config_path.open() as f:
        cases = json.load(f)

    n_ok = 0
    for meta in cases:
        case_dir = Path(meta["case_dir"])
        case_id = meta["case_id"]

        if not case_dir.exists():
            log.warning("Case dir missing: %s — skipping", case_dir)
            continue

        log.info("Extracting %s ...", case_id)
        df = extract_case(case_dir)

        if df is None:
            log.warning("No data extracted from %s", case_id)
            continue

        out_path = output_dir / case_id / "extracted_fields.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        log.info("  → %s  (%d rows)", out_path, len(df))
        n_ok += 1

    log.info("Extracted %d/%d cases → %s", n_ok, len(cases), output_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cases-dir",
        type=Path,
        default=REPO_ROOT / "simulations",
        help="Directory with case folders and cases_config.json",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "extracted",
        help="Where to write per-case extracted_fields.csv files",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    extract_all(args.cases_dir, args.output_dir)


if __name__ == "__main__":
    main()
