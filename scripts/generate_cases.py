"""Generate OpenFOAM dam-break case folders by sweeping (water_height, water_width).

Usage
-----
python scripts/generate_cases.py \\
    --n-cases 20 \\
    --height-range 0.4 0.8 \\
    --width-range  0.3 0.6 \\
    --output-dir   simulations/
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CASE = REPO_ROOT / "openfoam" / "base_case"


def _check_base_case() -> None:
    required = [
        BASE_CASE / "0" / "alpha.water",
        BASE_CASE / "system" / "blockMeshDict",
        BASE_CASE / "system" / "setFieldsDict",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        log.error("Base case files missing: %s", missing)
        sys.exit(1)


def _patch_blockMeshDict(path: Path, xmax: float, ymax: float) -> None:
    """Replace xMax and yMax values in blockMeshDict."""
    text = path.read_text()
    for key, val in [("xMax", xmax), ("yMax", ymax)]:
        # Replace the line "    xMax    <old>;" with the new value
        lines = text.splitlines()
        new_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(key):
                indent = line[: len(line) - len(line.lstrip())]
                new_lines.append(f"{indent}{key}    {val:.4f};")
            else:
                new_lines.append(line)
        text = "\n".join(new_lines)
    path.write_text(text)


def _patch_setFieldsDict(path: Path, water_width: float, water_height: float) -> None:
    """Replace the box coordinates in setFieldsDict."""
    text = path.read_text()
    # Find and replace the box line
    lines = text.splitlines()
    new_lines: list[str] = []
    for line in lines:
        if "box" in line and "(-1)" not in line and "1)" in line:
            indent = line[: len(line) - len(line.lstrip())]
            new_lines.append(
                f"{indent}box (0 0 -1) ({water_width:.4f} {water_height:.4f} 1);"
            )
        else:
            new_lines.append(line)
    path.write_text("\n".join(new_lines))


def generate_cases(
    n_cases: int,
    height_range: tuple[float, float],
    width_range: tuple[float, float],
    output_dir: Path,
    seed: int = 42,
) -> list[dict]:
    """Create N case directories and return a list of parameter dicts."""
    _check_base_case()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    heights = rng.uniform(height_range[0], height_range[1], n_cases)
    widths = rng.uniform(width_range[0], width_range[1], n_cases)

    # Domain dimensions fixed; only water column varies
    domain_x = 4.0
    domain_y = 2.0

    cases: list[dict] = []
    for i, (h, w) in enumerate(zip(heights, widths)):
        case_id = f"case_{i + 1:03d}"
        case_dir = output_dir / case_id

        log.info("Creating %s  height=%.3f m  width=%.3f m", case_id, h, w)

        # Copy base case
        if case_dir.exists():
            shutil.rmtree(case_dir)
        shutil.copytree(BASE_CASE, case_dir)

        # Patch blockMeshDict (domain size stays fixed; mesh adapts if needed)
        bm = case_dir / "system" / "blockMeshDict"
        _patch_blockMeshDict(bm, domain_x, domain_y)

        # Patch setFieldsDict (water column dimensions)
        sf = case_dir / "system" / "setFieldsDict"
        _patch_setFieldsDict(sf, w, h)

        cases.append(
            {
                "case_id": case_id,
                "case_dir": str(case_dir),
                "water_height": float(round(h, 6)),
                "water_width": float(round(w, 6)),
                "domain_x": domain_x,
                "domain_y": domain_y,
            }
        )

    config_path = output_dir / "cases_config.json"
    with config_path.open("w") as f:
        json.dump(cases, f, indent=2)
    log.info("Config written → %s", config_path)

    return cases


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-cases", type=int, default=10, help="Number of cases to generate")
    p.add_argument(
        "--height-range",
        nargs=2,
        type=float,
        default=[0.4, 0.8],
        metavar=("MIN", "MAX"),
        help="Water column height range [m]",
    )
    p.add_argument(
        "--width-range",
        nargs=2,
        type=float,
        default=[0.3, 0.6],
        metavar=("MIN", "MAX"),
        help="Water column width range [m]",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "simulations",
        help="Directory to write case folders",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cases = generate_cases(
        n_cases=args.n_cases,
        height_range=tuple(args.height_range),  # type: ignore[arg-type]
        width_range=tuple(args.width_range),     # type: ignore[arg-type]
        output_dir=args.output_dir,
        seed=args.seed,
    )
    log.info("Generated %d cases in %s", len(cases), args.output_dir)


if __name__ == "__main__":
    main()
