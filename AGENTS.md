# Repository Guidelines

Concise rules for contributing to the depthPlane workspace. Keep edits small, scripted, and repeatable so others can reproduce results quickly.

## Project Structure & Module Organization
- `src/` — Python modules for core logic (keep reusable functions/classes here; prefer type-annotated, testable code).
- `scripts/` — Short CLI entry points or data prep helpers that import from `src/`.
- `data/` — Example inputs: `CAE.las` (point cloud) and `only_structure.stl` (geometry). Treat as read-only; add new large files via download instructions, not commits.
- `extract_floorplan.py` — Main sketch for converting input data to floorplans. Extend this rather than adding new root-level scripts.

## Environment & Tooling (Pixi)
- Environment pinned in `pixi.lock` (Python 3.11). Install once with `pixi install` and enter with `pixi shell`.
- All repo commands should be executed via Pixi: prefix with `pixi run`, e.g., `pixi run python extract_floorplan.py --help`; this ensures the locked toolchain is used.
- Add dependencies with `pixi add <package>` so the lockfile stays in sync; avoid ad‑hoc `pip install` outside Pixi.

## Build, Run, and Development Commands
- `pixi run python extract_floorplan.py --input data/CAE.las --out out.svg` (example usage once CLI args are implemented).
- `pixi run python -m src.<module>` for module-level utilities during development.
- `pixi update` only when you intentionally refresh dependencies; commit the updated `pixi.lock`.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation; keep lines ≤ 100 chars.
- Modules/files: `lower_snake.py`; functions/variables: `snake_case`; classes: `CapWords`.
- Prefer pure functions in `src/`; keep I/O (file paths, argparse) in `scripts/` or the root CLI.
- Document non-obvious calculations with brief comments; prefer docstrings for public functions.

## Testing Guidelines
- Testing framework not yet wired; add `pytest` via `pixi add pytest` before first tests.
- Always run tests through Pixi: `pixi run pytest` (or `pixi run pytest tests/test_floorplan.py` for a subset).
- Place tests in `tests/` mirroring `src/` structure (`tests/test_floorplan.py`, etc.).
- Use deterministic fixtures from small slices of `data/`; avoid committing bulky derived outputs.
- Aim for coverage on geometry transforms and parsing helpers; include edge cases (empty point cloud, malformed STL).

## Commit & Pull Request Guidelines
- If/when Git is initialized, prefer conventional commit prefixes (`feat:`, `fix:`, `chore:`, `docs:`).
- PRs should describe the change, list runnable commands, and note data inputs/outputs touched. Attach screenshots or sample SVGs if visualization changes.
- Update this guide and inline help (`--help`) when CLI arguments change. Avoid force-pushing to shared branches once reviews start.

## Data & Security Tips
- Do not commit large or proprietary datasets; add them to a `.gitignore` entry or provide download steps instead.
- Validate external files before processing and guard file path handling (no implicit writes outside the repo).
