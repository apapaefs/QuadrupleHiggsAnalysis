#!/usr/bin/env python3
"""Build fixed one-bin direct and cascade resonant cross-section limits.

The input is a completed and validated AK4+AK8 resonance score-fit run.  The
total signal and Asimov yields in that run are independent of the subdivision
of the classifier score.  This postprocessor discards that subdivision and
uses exactly one inclusive event count at every generated mass point.  The
result is therefore a transparent counting analysis; the classifier score
does not enter the quoted limit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import numpy as np


METHOD_VERSION = "resonance-ak4ak8-one-bin-poisson-v1"
Q95 = 3.841458820694124
COLLIDER_ENERGY_TEV = 14.0
LUMINOSITY_FB = 3000.0
EXPECTED_POINTS = {"direct": 42, "cascade": 441}
CASCADE_SMOOTHING_STRENGTH = 5.0
CASCADE_GRID_SIZE = 420


class OneBinError(RuntimeError):
    """Raised when the fixed one-bin result cannot be constructed safely."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise OneBinError(f"refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in fields} for row in rows])
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def poisson_q_one_bin(asimov_count: float, signal_per_fb: float, sigma_fb: float) -> float:
    """Return the direct Poisson Asimov deviance for one inclusive count."""

    n = float(asimov_count)
    signal = float(signal_per_fb)
    sigma = float(sigma_fb)
    if not all(math.isfinite(value) for value in (n, signal, sigma)):
        raise ValueError("Poisson inputs must be finite")
    if n < 0.0 or signal < 0.0 or sigma < 0.0:
        raise ValueError("Poisson inputs must be non-negative")
    nu = n + sigma * signal
    if n == 0.0:
        return 2.0 * nu
    if nu <= 0.0:
        raise ValueError("the tested expectation must be positive")
    return 2.0 * (nu - n + n * math.log(n / nu))


def solve_sigma95_one_bin(asimov_count: float, signal_per_fb: float) -> float:
    """Solve q(sigma)=Q95 by deterministic bisection."""

    n = float(asimov_count)
    signal = float(signal_per_fb)
    if not math.isfinite(n) or not math.isfinite(signal) or n < 0.0 or signal <= 0.0:
        raise ValueError("a finite non-negative count and positive signal yield are required")
    low = 0.0
    high = 1.0
    while poisson_q_one_bin(n, signal, high) < Q95:
        high *= 2.0
        if high > 1.0e12:
            raise OneBinError("failed to bracket the one-bin 95% limit")
    for _ in range(160):
        middle = 0.5 * (low + high)
        if poisson_q_one_bin(n, signal, middle) < Q95:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise OneBinError(f"missing input table: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise OneBinError(f"empty input table: {path}")
    return rows


def build_one_bin_rows(topology: str, input_csv: Path) -> list[dict[str, Any]]:
    """Collapse the validated total yields to one Poisson count per point."""

    source_rows = _read_rows(input_csv)
    if len(source_rows) != EXPECTED_POINTS[topology]:
        raise OneBinError(
            f"{topology}: found {len(source_rows)} mass points, "
            f"expected {EXPECTED_POINTS[topology]}"
        )
    output: list[dict[str, Any]] = []
    for source in source_rows:
        if source.get("topology") != topology or source.get("status") != "ok":
            raise OneBinError(f"{source.get('point_id')}: input result is not valid")
        signal = float(source["signal_1fb_yield"])
        asimov = float(source["asimov_yield"])
        previous = float(source["sigma95_fb"])
        limit = solve_sigma95_one_bin(asimov, signal)
        if limit + 1.0e-10 < previous:
            raise OneBinError(
                f"{source['point_id']}: inclusive limit {limit:g} is unexpectedly "
                f"stronger than the score-fit limit {previous:g}"
            )
        q_at_limit = poisson_q_one_bin(asimov, signal, limit)
        if not math.isclose(q_at_limit, Q95, rel_tol=0.0, abs_tol=2.0e-10):
            raise OneBinError(f"{source['point_id']}: q crossing did not close")
        output.append(
            {
                "point_id": source["point_id"],
                "topology": topology,
                "MS_GeV": source.get("MS_GeV", ""),
                "M2_GeV": source.get("M2_GeV", ""),
                "M3_GeV": source.get("M3_GeV", ""),
                "fixed_bins": 1,
                "sigma95_fb": limit,
                "q_at_sigma95": q_at_limit,
                "signal_1fb_yield": signal,
                "asimov_yield": asimov,
                "source_scorefit_sigma95_fb": previous,
                "source_selected_bins": int(source["selected_bins"]),
            }
        )
    key = (
        (lambda row: float(row["MS_GeV"]))
        if topology == "direct"
        else (lambda row: (float(row["M3_GeV"]), float(row["M2_GeV"])))
    )
    return sorted(output, key=key)


def collapse_reference_yields(input_csv: Path) -> list[dict[str, Any]]:
    """Replace the former score-bin columns by their inclusive total."""

    rows = _read_rows(input_csv)
    output: list[dict[str, Any]] = []
    for row in rows:
        value = float(row["total"])
        if not math.isfinite(value) or value < -1.0e-12:
            raise OneBinError(f"non-physical inclusive yield for {row['process']}")
        output.append(
            {
                "process": row["process"],
                "kind": row["kind"],
                "inclusive_yield": max(0.0, value),
            }
        )
    return output


def conservative_direct_curve(
    rows: Sequence[Mapping[str, Any]], *, grid_size: int = 1200
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return the minimal non-increasing upper envelope of the exact limits."""

    from scipy.interpolate import PchipInterpolator  # type: ignore

    masses = np.asarray([float(row["MS_GeV"]) for row in rows], dtype=float) / 1000.0
    exact_log = np.log10(
        np.asarray([float(row["sigma95_fb"]) for row in rows], dtype=float)
    )
    if np.any(np.diff(masses) <= 0.0):
        raise OneBinError("direct mass points are not strictly increasing")
    upper_log = np.maximum.accumulate(exact_log[::-1])[::-1]
    dense_mass = np.linspace(float(np.min(masses)), float(np.max(masses)), grid_size)
    dense_log = PchipInterpolator(masses, upper_log)(dense_mass)
    lift = upper_log - exact_log
    audit = {
        "method": "minimal non-increasing upper envelope in log10(sigma95)",
        "interpolation": "shape-preserving PCHIP through upper-envelope nodes",
        "physical_points": len(rows),
        "conservative_at_every_generated_point": bool(
            np.all(upper_log + 1.0e-12 >= exact_log)
        ),
        "minimum_lift_log10": float(np.min(lift)),
        "median_lift_log10": float(np.median(lift)),
        "maximum_lift_log10": float(np.max(lift)),
        "paper_ready": True,
    }
    return dense_mass, np.power(10.0, dense_log), audit


def _plot_direct(
    output_dir: Path, rows: Sequence[Mapping[str, Any]]
) -> tuple[list[Path], dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    masses = np.asarray([float(row["MS_GeV"]) for row in rows]) / 1000.0
    limits = np.asarray([float(row["sigma95_fb"]) for row in rows])
    dense_mass, display_limit, audit = conservative_direct_curve(rows)

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.plot(
        dense_mass,
        display_limit,
        color="#C51B29",
        linewidth=2.4,
        label="Conservative smooth guide",
    )
    ax.scatter(
        masses,
        limits,
        s=25,
        facecolors="white",
        edgecolors="#8E1722",
        linewidths=0.9,
        zorder=3,
        label="Generated mass points",
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_S$ [TeV]", fontsize=15)
    ax.set_ylabel(r"Expected 95% upper limit on $\sigma_{\rm dir}$ [fb]", fontsize=15)
    ax.tick_params(axis="both", which="both", labelsize=13)
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False, fontsize=10.5, loc="upper right")
    ax.set_title(
        rf"Direct resonant four Higgs boson production, "
        rf"$\sqrt{{s}}={COLLIDER_ENERGY_TEV:g}$ TeV, "
        rf"$\mathcal{{L}}={LUMINOSITY_FB / 1000.0:g}\,\mathrm{{ab}}^{{-1}}$",
        fontsize=15.0,
        pad=10.0,
    )
    fig.tight_layout()
    outputs: list[Path] = []
    for suffix in ("pdf", "png"):
        path = output_dir / f"direct_expected_cross_section_limit.{suffix}"
        fig.savefig(path, dpi=250 if suffix == "png" else None)
        outputs.append(path)
    plt.close(fig)
    return outputs, audit


def conservative_cascade_surface(
    rows: Sequence[Mapping[str, Any]],
    *,
    smoothing_strength: float = CASCADE_SMOOTHING_STRENGTH,
    grid_size: int = CASCADE_GRID_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray, dict[str, Any]]:
    """Construct an audited smooth upper surface in log10(limit).

    The node values minimise a graph-Laplacian roughness penalty subject to a
    lower bound equal to every exact pointwise limit.  Thus the displayed
    surface is never stronger than the calculated limit at a generated point.
    """

    from scipy.interpolate import CloughTocher2DInterpolator  # type: ignore
    from scipy.optimize import minimize  # type: ignore
    from scipy.spatial import Delaunay  # type: ignore

    m2 = np.asarray([float(row["M2_GeV"]) for row in rows], dtype=float)
    m3 = np.asarray([float(row["M3_GeV"]) for row in rows], dtype=float)
    exact_log = np.log10(
        np.asarray([float(row["sigma95_fb"]) for row in rows], dtype=float)
    )
    low = np.asarray([np.min(m2), np.min(m3)], dtype=float)
    span = np.asarray([np.ptp(m2), np.ptp(m3)], dtype=float)
    coordinates = np.column_stack(
        [(m2 - low[0]) / span[0], (m3 - low[1]) / span[1]]
    )
    triangulation = Delaunay(coordinates)

    adjacency = np.zeros((len(rows), len(rows)), dtype=float)
    for simplex in triangulation.simplices:
        for first in simplex:
            for second in simplex:
                if first == second:
                    continue
                distance = float(np.linalg.norm(coordinates[first] - coordinates[second]))
                adjacency[first, second] = max(
                    adjacency[first, second], 1.0 / max(distance, 1.0e-12)
                )
    row_sums = np.sum(adjacency, axis=1)
    if np.any(row_sums <= 0.0):
        raise OneBinError("cascade smoothing graph contains an isolated point")
    adjacency /= row_sums[:, None]
    laplacian = np.eye(len(rows), dtype=float) - adjacency

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        residual = values - exact_log
        roughness = laplacian @ values
        loss = 0.5 * (
            float(residual @ residual)
            + float(smoothing_strength) * float(roughness @ roughness)
        )
        gradient = residual + float(smoothing_strength) * (laplacian.T @ roughness)
        return loss, gradient

    fit = minimize(
        objective,
        exact_log.copy(),
        jac=True,
        bounds=[(float(value), None) for value in exact_log],
        method="L-BFGS-B",
        options={"maxiter": 4000, "ftol": 1.0e-13, "gtol": 1.0e-10},
    )
    if not fit.success:
        raise OneBinError(f"cascade upper-surface fit failed: {fit.message}")
    upper_log = np.asarray(fit.x, dtype=float)
    constraint_violation = float(np.max(exact_log - upper_log))
    if constraint_violation > 2.0e-9:
        raise OneBinError("cascade upper surface violates a pointwise lower bound")

    interpolator = CloughTocher2DInterpolator(coordinates, upper_log)
    exact_residual = float(
        np.max(np.abs(np.asarray(interpolator(coordinates)) - upper_log))
    )
    if exact_residual > 1.0e-8:
        raise OneBinError("cascade display interpolation is not exact at its nodes")

    gx = np.linspace(np.min(m2), np.max(m2), int(grid_size))
    gy = np.linspace(np.min(m3), np.max(m3), int(grid_size))
    xx, yy = np.meshgrid(gx, gy)
    scaled_grid = np.column_stack(
        [
            (xx.ravel() - low[0]) / span[0],
            (yy.ravel() - low[1]) / span[1],
        ]
    )
    simplices = triangulation.find_simplex(scaled_grid)
    inside = (simplices >= 0) & (yy.ravel() > 2.0 * xx.ravel())
    display = np.full(len(scaled_grid), np.nan, dtype=float)
    display[inside] = np.asarray(interpolator(scaled_grid[inside]), dtype=float)

    inside_indices = np.flatnonzero(inside)
    vertices = triangulation.simplices[simplices[inside]]
    local_values = upper_log[vertices]
    local_minimum = np.min(local_values, axis=1)
    local_maximum = np.max(local_values, axis=1)
    before_clip = display[inside_indices].copy()
    display[inside_indices] = np.clip(
        before_clip, local_minimum, local_maximum
    )
    clipped_nodes = int(
        np.sum(np.abs(display[inside_indices] - before_clip) > 1.0e-12)
    )
    masked = np.ma.masked_invalid(display.reshape(xx.shape))
    masked.mask = np.ma.getmaskarray(masked) | (~inside.reshape(xx.shape))

    lift = upper_log - exact_log
    audit = {
        "method": "constrained graph-Laplacian upper surface in log10(sigma95)",
        "interpolation": "coordinate-rescaled Clough-Tocher with local vertex-range clipping",
        "smoothing_strength": float(smoothing_strength),
        "grid_size": int(grid_size),
        "physical_points": len(rows),
        "optimizer_success": bool(fit.success),
        "optimizer_iterations": int(fit.nit),
        "maximum_constraint_violation_log10": constraint_violation,
        "minimum_lift_log10": float(np.min(lift)),
        "median_lift_log10": float(np.median(lift)),
        "maximum_lift_log10": float(np.max(lift)),
        "interpolator_exact_node_residual_log10": exact_residual,
        "inside_grid_nodes": int(np.sum(inside)),
        "locally_clipped_grid_nodes": clipped_nodes,
        "conservative_at_every_generated_point": bool(
            np.all(upper_log + 2.0e-9 >= exact_log)
        ),
        "paper_ready": True,
    }
    return xx, yy, masked, audit


def _plot_cascade(
    output_dir: Path, rows: Sequence[Mapping[str, Any]]
) -> tuple[list[Path], dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    xx, yy, log_surface, audit = conservative_cascade_surface(rows)
    values = np.ma.power(10.0, log_surface)
    m2 = np.asarray([float(row["M2_GeV"]) for row in rows], dtype=float)
    m3 = np.asarray([float(row["M3_GeV"]) for row in rows], dtype=float)
    limits = np.asarray([float(row["sigma95_fb"]) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    image = ax.pcolormesh(
        xx,
        yy,
        values,
        shading="auto",
        cmap="viridis_r",
        norm=LogNorm(vmin=float(np.min(limits)), vmax=float(np.max(limits))),
        rasterized=True,
    )
    ax.scatter(
        m2,
        m3,
        s=8,
        facecolors="none",
        edgecolors="black",
        linewidths=0.32,
        alpha=0.72,
        label="Generated mass points",
    )
    boundary_x = np.linspace(float(np.min(m2)), float(np.max(m2)), 500)
    boundary_y = 2.0 * boundary_x
    boundary_mask = (boundary_y >= np.min(m3)) & (boundary_y <= np.max(m3))
    ax.plot(
        boundary_x[boundary_mask],
        boundary_y[boundary_mask],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=r"$M_3=2M_2$",
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(
        r"Expected 95% upper limit on $\sigma_{\rm cas}$ [fb]", fontsize=14
    )
    colorbar.ax.tick_params(labelsize=12)
    ax.set_xlabel(r"$M_2$ [GeV]", fontsize=15)
    ax.set_ylabel(r"$M_3$ [GeV]", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.set_title(
        rf"Cascade resonant scenario, $\sqrt{{s}}={COLLIDER_ENERGY_TEV:g}$ TeV, "
        rf"$\mathcal{{L}}={LUMINOSITY_FB / 1000.0:g}\,\mathrm{{ab}}^{{-1}}$",
        fontsize=14.8,
        pad=10.0,
    )
    fig.tight_layout()
    outputs: list[Path] = []
    for suffix in ("pdf", "png"):
        path = output_dir / f"cascade_expected_cross_section_limit.{suffix}"
        fig.savefig(path, dpi=250 if suffix == "png" else None)
        outputs.append(path)
    plt.close(fig)
    return outputs, audit


def _package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("scipy", "matplotlib"):
        module = __import__(name)
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def run(input_root: Path, output_root: Path) -> dict[str, Any]:
    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    input_records: list[dict[str, Any]] = []
    output_paths: list[Path] = []
    topology_summaries: dict[str, Any] = {}

    for topology in ("direct", "cascade"):
        topology_input = input_root / topology
        point_input = topology_input / "pointwise_limits.csv"
        reference_name = (
            "score_yields_MS_1500.csv"
            if topology == "direct"
            else "score_yields_M2_0625_M3_1500.csv"
        )
        yield_input = topology_input / reference_name
        for path in (point_input, yield_input):
            if not path.is_file():
                raise OneBinError(f"missing required source artifact: {path}")
            input_records.append(
                {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            )

        topology_output = output_root / topology
        topology_output.mkdir(parents=True, exist_ok=True)
        point_rows = build_one_bin_rows(topology, point_input)
        yield_rows = collapse_reference_yields(yield_input)
        point_csv = topology_output / "pointwise_limits.csv"
        point_json = topology_output / "pointwise_limits.json"
        yield_csv = topology_output / reference_name.replace("score_yields", "inclusive_yields")
        _write_csv(point_csv, point_rows)
        _write_json(
            point_json,
            {
                "method_version": METHOD_VERSION,
                "topology": topology,
                "fixed_bins": 1,
                "rows": point_rows,
            },
        )
        _write_csv(yield_csv, yield_rows)
        output_paths.extend((point_csv, point_json, yield_csv))

        limits = np.asarray([float(row["sigma95_fb"]) for row in point_rows])
        weakest = point_rows[int(np.argmax(limits))]
        strongest = point_rows[int(np.argmin(limits))]
        topology_summaries[topology] = {
            "points": len(point_rows),
            "minimum_sigma95_fb": float(np.min(limits)),
            "minimum_point_id": strongest["point_id"],
            "maximum_sigma95_fb": float(np.max(limits)),
            "maximum_point_id": weakest["point_id"],
        }

        if topology == "direct":
            plots, display_audit = _plot_direct(topology_output, point_rows)
            output_paths.extend(plots)
        else:
            plots, display_audit = _plot_cascade(topology_output, point_rows)
            output_paths.extend(plots)
        audit_path = topology_output / "display_audit.json"
        _write_json(audit_path, display_audit)
        output_paths.append(audit_path)

    readme = output_root / "README.md"
    readme.write_text(
        f"""# Fixed one-bin AK4+AK8 resonance limits

This directory contains the `{METHOD_VERSION}` direct and cascade results.
Every generated mass point uses one inclusive expected event count after the
AK4+AK8 reconstruction and analytic tagging weights.  The classifier-score
subdivision of the source run is discarded, so the XGBoost score does not
enter these quoted limits.

The Asimov count contains the conventional backgrounds and the physical SM
`hhhh`, `hhh+bb`, and `hh+4b` rates.  The 95% expected upper limit solves
`q(sigma95)={Q95:.12g}` with a resonant template normalised to 1 fb before the
four Higgs boson decays.  No pyhf model, nuisance parameter, expected band or
finite-simulation term enters the likelihood.

The direct display uses the minimal non-increasing upper envelope of the exact
generated limits, followed by a shape-preserving interpolation.  The cascade
colour map uses an audited smooth upper surface in `log10(sigma95)`.  Both
displays are constrained never to give a stronger limit at a generated point;
exact pointwise limits are retained in the CSV and JSON tables.
""",
        encoding="utf-8",
    )
    output_paths.append(readme)

    versions = _package_versions()
    versions_path = output_root / "package_versions.json"
    _write_json(versions_path, versions)
    output_paths.append(versions_path)
    manifest = {
        "method_version": METHOD_VERSION,
        "driver": str(Path(__file__).resolve()),
        "driver_sha256": _sha256(Path(__file__).resolve()),
        "status": "complete",
        "paper_ready": True,
        "fixed_bins": 1,
        "classifier_score_used_in_limit": False,
        "collider_energy_TeV": COLLIDER_ENERGY_TEV,
        "luminosity_fb_inverse": LUMINOSITY_FB,
        "q95": Q95,
        "input_root": str(input_root),
        "input_artifacts": input_records,
        "topologies": topology_summaries,
        "package_versions": versions,
        "outputs": [
            {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in output_paths
        ],
    }
    _write_json(output_root / "run_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        required=True,
        type=Path,
        help="Validated resonance score-fit output containing direct/ and cascade/.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run(args.input_root, args.output_dir)
    except Exception as error:
        print(f"one-bin resonance analysis failed: {type(error).__name__}: {error}")
        return 1
    print(
        f"fixed one-bin resonance analysis complete; "
        f"paper_ready={str(manifest['paper_ready']).lower()}; output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
