"""Quadratic cross-section fits for HEFT ``gg -> hh + 4b``.

The generator cross section is fitted before Higgs branching fractions,
K-factors, tagging efficiencies, or analysis efficiencies are applied.  The
fit variable is the repository convention ``c3 = kappa_lambda - 1``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FIT_SCHEMA = "sm-hh4b-c3-cross-section-fit-v1"
COEFFICIENT_ORDER = ("constant", "linear", "quadratic")


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def validate_hh4b_c3_fit(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one serialized quadratic fit."""

    if not isinstance(payload, Mapping):
        raise ValueError("hh+4b c3 fit must be a JSON object")
    if payload.get("schema") != FIT_SCHEMA:
        raise ValueError(
            f"hh+4b c3 fit schema must be {FIT_SCHEMA!r}, "
            f"got {payload.get('schema')!r}"
        )
    order = tuple(payload.get("coefficient_order", ()))
    if order != COEFFICIENT_ORDER:
        raise ValueError(
            "hh+4b c3 fit coefficient_order must be "
            f"{list(COEFFICIENT_ORDER)!r}"
        )

    coefficients = np.asarray(payload.get("coefficients_pb"), dtype=float)
    covariance = np.asarray(payload.get("covariance_pb2"), dtype=float)
    if coefficients.shape != (3,) or np.any(~np.isfinite(coefficients)):
        raise ValueError("hh+4b c3 fit coefficients_pb must contain 3 finite values")
    if covariance.shape != (3, 3) or np.any(~np.isfinite(covariance)):
        raise ValueError("hh+4b c3 fit covariance_pb2 must be a finite 3x3 matrix")
    if not np.allclose(covariance, covariance.T, rtol=1.0e-10, atol=1.0e-30):
        raise ValueError("hh+4b c3 fit covariance_pb2 must be symmetric")
    eigenvalues = np.linalg.eigvalsh(covariance)
    covariance_scale = max(1.0e-30, float(np.max(np.abs(covariance))))
    if float(np.min(eigenvalues)) < -1.0e-10 * covariance_scale:
        raise ValueError("hh+4b c3 fit covariance_pb2 must be positive semidefinite")

    points = payload.get("points")
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("hh+4b c3 fit must retain at least 3 source points")
    normalized_points = []
    unique_c3 = set()
    for index, point in enumerate(points):
        if not isinstance(point, Mapping):
            raise ValueError(f"hh+4b c3 fit point {index} must be a JSON object")
        c3 = _finite_float(point.get("c3"), f"fit point {index} c3")
        xsec_pb = _finite_float(
            point.get("cross_section_pb"),
            f"fit point {index} cross_section_pb",
        )
        error_pb = _finite_float(
            point.get("integration_error_pb"),
            f"fit point {index} integration_error_pb",
        )
        if xsec_pb <= 0.0 or error_pb <= 0.0:
            raise ValueError(
                f"hh+4b c3 fit point {index} must have positive cross section "
                "and uncertainty"
            )
        unique_c3.add(c3)
        normalized_points.append(
            {
                **dict(point),
                "c3": c3,
                "cross_section_pb": xsec_pb,
                "integration_error_pb": error_pb,
            }
        )
    if len(unique_c3) < 3:
        raise ValueError("hh+4b c3 fit requires at least 3 distinct c3 values")

    normalized = dict(payload)
    normalized["coefficient_order"] = list(COEFFICIENT_ORDER)
    normalized["coefficients_pb"] = coefficients.tolist()
    normalized["covariance_pb2"] = covariance.tolist()
    normalized["points"] = normalized_points
    return normalized


def load_hh4b_c3_fit(path: str | Path) -> dict[str, Any]:
    """Load and validate a serialized fit."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"hh+4b c3 fit file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"hh+4b c3 fit is not valid JSON: {path}: {exc}") from exc
    normalized = validate_hh4b_c3_fit(payload)
    normalized["fit_file"] = str(path)
    return normalized


def evaluate_hh4b_c3_fit(
    payload: Mapping[str, Any],
    c3: float,
) -> dict[str, float]:
    """Evaluate the raw generator cross section and propagated fit error."""

    fit = validate_hh4b_c3_fit(payload)
    c3 = _finite_float(c3, "c3")
    basis = np.asarray([1.0, c3, c3 * c3], dtype=float)
    coefficients = np.asarray(fit["coefficients_pb"], dtype=float)
    covariance = np.asarray(fit["covariance_pb2"], dtype=float)
    xsec_pb = float(basis @ coefficients)
    variance_pb2 = float(basis @ covariance @ basis)
    tolerance = 1.0e-12 * max(1.0, float(np.max(np.abs(covariance))))
    if variance_pb2 < -tolerance:
        raise ValueError(
            f"hh+4b c3 fit gives a negative variance at c3={c3:g}: "
            f"{variance_pb2:g} pb^2"
        )
    if xsec_pb <= 0.0:
        raise ValueError(
            f"hh+4b c3 fit gives a non-positive cross section at c3={c3:g}: "
            f"{xsec_pb:g} pb"
        )
    uncertainty_pb = math.sqrt(max(0.0, variance_pb2))
    return {
        "c3": c3,
        "cross_section_pb": xsec_pb,
        "cross_section_uncertainty_pb": uncertainty_pb,
        "cross_section_fb": 1.0e3 * xsec_pb,
        "cross_section_uncertainty_fb": 1.0e3 * uncertainty_pb,
    }


def fit_hh4b_c3_cross_section(
    points: Sequence[Mapping[str, Any]],
    *,
    source_campaign: str | None = None,
) -> dict[str, Any]:
    """Return a weighted least-squares quadratic fit to integration points."""

    normalized_points = []
    for index, point in enumerate(points):
        c3 = _finite_float(point.get("c3"), f"point {index} c3")
        xsec_pb = _finite_float(
            point.get("cross_section_pb"),
            f"point {index} cross_section_pb",
        )
        error_pb = _finite_float(
            point.get("integration_error_pb"),
            f"point {index} integration_error_pb",
        )
        if xsec_pb <= 0.0 or error_pb <= 0.0:
            raise ValueError(
                f"point {index} must have positive cross section and uncertainty"
            )
        normalized_points.append(
            {
                **dict(point),
                "c3": c3,
                "cross_section_pb": xsec_pb,
                "integration_error_pb": error_pb,
            }
        )

    if len(normalized_points) < 3:
        raise ValueError("A quadratic hh+4b fit requires at least 3 points")
    c3_values = np.asarray([point["c3"] for point in normalized_points])
    if len(np.unique(c3_values)) != len(c3_values):
        raise ValueError("The hh+4b fit points contain duplicate c3 values")
    values = np.asarray(
        [point["cross_section_pb"] for point in normalized_points],
        dtype=float,
    )
    errors = np.asarray(
        [point["integration_error_pb"] for point in normalized_points],
        dtype=float,
    )
    design = np.column_stack((np.ones_like(c3_values), c3_values, c3_values**2))
    weighted_design = design / errors[:, None]
    weighted_values = values / errors
    if np.linalg.matrix_rank(weighted_design) != 3:
        raise ValueError("The hh+4b c3 integration points are rank deficient")
    normal_matrix = weighted_design.T @ weighted_design
    covariance = np.linalg.inv(normal_matrix)
    coefficients = covariance @ weighted_design.T @ weighted_values
    fitted = design @ coefficients
    residuals = values - fitted
    pulls = residuals / errors
    chi2 = float(np.sum(pulls**2))
    ndof = int(len(values) - 3)

    stored_points = []
    for point, fit_value, residual, pull in zip(
        normalized_points, fitted, residuals, pulls
    ):
        stored_points.append(
            {
                **point,
                "fit_cross_section_pb": float(fit_value),
                "residual_pb": float(residual),
                "pull": float(pull),
            }
        )
    payload = {
        "schema": FIT_SCHEMA,
        "process": "HEFT gg -> hh + b bbar b bbar",
        "coupling_convention": "c3 = kappa_lambda - 1",
        "cross_section_convention": (
            "raw Sherpa generator cross section before K-factor, "
            "Higgs branching fractions, tagging, or analysis efficiency"
        ),
        "model": (
            "sigma_pb(c3) = constant + linear*c3 + quadratic*c3^2"
        ),
        "coefficient_order": list(COEFFICIENT_ORDER),
        "coefficients_pb": coefficients.tolist(),
        "covariance_pb2": covariance.tolist(),
        "chi2": chi2,
        "ndof": ndof,
        "source_campaign": source_campaign,
        "points": stored_points,
    }
    return validate_hh4b_c3_fit(payload)
