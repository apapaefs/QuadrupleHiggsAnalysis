"""One-dimensional c3/d4 confidence-interval extraction helpers.

The legacy and v2 analyses construct their two-dimensional expected 95% CL
regions differently.  This module keeps the fixed-c3 interval bookkeeping
common while preserving those distinct numerical inputs:

* v2 interpolates ``log10(sigma/sigma95)`` between scanned points at c3=0;
* legacy solves the fitted Chebyshev sigma*eff surface at c3=0.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _deduplicate_sorted(values: Sequence[float], tolerance: float) -> list[float]:
    output: list[float] = []
    for value in sorted(float(item) for item in values):
        if not output or abs(value - output[-1]) > tolerance:
            output.append(value)
        else:
            output[-1] = 0.5 * (output[-1] + value)
    return output


def _intervals_from_crossings(
    domain: tuple[float, float],
    crossings: Sequence[float],
    evaluate: Callable[[float], float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify finite-width intervals as allowed (value <= 0) or excluded."""

    lower, upper = (float(domain[0]), float(domain[1]))
    scale = max(abs(lower), abs(upper), upper - lower, 1.0)
    tolerance = 1.0e-10 * scale
    roots = [
        value
        for value in _deduplicate_sorted(crossings, tolerance)
        if lower - tolerance <= value <= upper + tolerance
    ]
    roots = [min(max(value, lower), upper) for value in roots]
    boundaries = _deduplicate_sorted([lower, *roots, upper], tolerance)

    allowed: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right - left <= tolerance:
            continue
        midpoint = 0.5 * (left + right)
        value = float(evaluate(midpoint))
        if not math.isfinite(value):
            continue
        interval = {
            "lower": float(left),
            "upper": float(right),
            "lower_is_scan_boundary": bool(abs(left - lower) <= tolerance),
            "upper_is_scan_boundary": bool(abs(right - upper) <= tolerance),
        }
        (allowed if value <= 0.0 else excluded).append(interval)
    return allowed, excluded


def _constraint_payload(
    *,
    c3: float,
    confidence_level: float,
    d4_domain: tuple[float, float],
    crossings: Sequence[float],
    allowed_intervals: Sequence[Mapping[str, Any]],
    excluded_intervals: Sequence[Mapping[str, Any]],
    method: str,
    strategy: str | None = None,
    limit_kind: str | None = None,
    limit_key: str | None = None,
    slice_point_count: int | None = None,
) -> dict[str, Any]:
    allowed = [dict(interval) for interval in allowed_intervals]
    excluded = [dict(interval) for interval in excluded_intervals]
    scan_limited = any(
        interval["lower_is_scan_boundary"] or interval["upper_is_scan_boundary"]
        for interval in allowed
    )
    if not allowed:
        constraint_type = "no_allowed_values_in_scan"
    elif len(allowed) == 1 and not scan_limited:
        constraint_type = "bounded_interval"
    elif len(allowed) == 1:
        constraint_type = "scan_limited_interval"
    else:
        constraint_type = "multiple_intervals"
    primary = allowed[0] if len(allowed) == 1 else None
    return {
        "status": "ok",
        "confidence_level": float(confidence_level),
        "c3": float(c3),
        "parameter_definition": "d4 = kappa4 - 1",
        "d4_domain": [float(d4_domain[0]), float(d4_domain[1])],
        "crossings_d4": [float(value) for value in crossings],
        "allowed_intervals": allowed,
        "excluded_intervals": excluded,
        "constraint_type": constraint_type,
        "scan_boundary_limited": bool(scan_limited),
        "lower_95cl": (
            None if primary is None else float(primary["lower"])
        ),
        "upper_95cl": (
            None if primary is None else float(primary["upper"])
        ),
        "method": method,
        "strategy": strategy,
        "limit_kind": limit_kind,
        "limit_key": limit_key,
        "slice_point_count": slice_point_count,
        "criterion": "allowed where sigma/sigma95 <= 1",
    }


def fixed_c3_scan_constraint(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit_key: str,
    c3: float = 0.0,
    confidence_level: float = 0.95,
    strategy: str | None = None,
    limit_kind: str | None = None,
    c3_tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    """Extract the allowed d4 interval from directly scanned fixed-c3 points."""

    confidence_level = float(confidence_level)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")

    points: list[tuple[float, float]] = []
    invalid_points: list[dict[str, Any]] = []
    for row in rows:
        row_c3 = _finite_float(row.get("c3"))
        if row_c3 is None or not math.isclose(
            row_c3, float(c3), rel_tol=0.0, abs_tol=float(c3_tolerance)
        ):
            continue
        d4 = _finite_float(row.get("d4"))
        xsec = _finite_float(row.get("xsec_fb"))
        limit = _finite_float(row.get(limit_key))
        if d4 is None or xsec is None or limit is None or xsec <= 0.0 or limit <= 0.0:
            invalid_points.append(
                {
                    "d4": d4,
                    "xsec_fb": xsec,
                    "limit_fb": limit,
                }
            )
            continue
        points.append((d4, math.log10(xsec / limit)))

    base = {
        "status": "unavailable",
        "confidence_level": confidence_level,
        "c3": float(c3),
        "parameter_definition": "d4 = kappa4 - 1",
        "strategy": strategy,
        "limit_kind": limit_kind,
        "limit_key": limit_key,
        "method": (
            "piecewise-linear interpolation of log10(sigma/sigma95) "
            "between scanned fixed-c3 points"
        ),
        "slice_point_count": len(points),
    }
    if invalid_points:
        return {
            **base,
            "reason": "one or more fixed-c3 scan points have a missing or non-positive limit",
            "invalid_points": invalid_points,
        }
    if len(points) < 2:
        return {
            **base,
            "reason": f"need at least two finite c3={float(c3):g} scan points; got {len(points)}",
        }

    points.sort()
    d4_values = np.asarray([point[0] for point in points], dtype=float)
    log_ratios = np.asarray([point[1] for point in points], dtype=float)
    duplicate_indices = np.flatnonzero(np.diff(d4_values) <= 0.0)
    if duplicate_indices.size:
        return {
            **base,
            "reason": "duplicate d4 coordinates occur on the fixed-c3 scan slice",
            "duplicate_d4": [
                float(d4_values[index]) for index in duplicate_indices
            ],
        }

    zero_tolerance = 1.0e-12
    crossings: list[float] = []
    for index in range(len(d4_values) - 1):
        left_d4 = float(d4_values[index])
        right_d4 = float(d4_values[index + 1])
        left = float(log_ratios[index])
        right = float(log_ratios[index + 1])
        if abs(left) <= zero_tolerance:
            crossings.append(left_d4)
        if left * right < 0.0:
            crossings.append(
                left_d4 - left * (right_d4 - left_d4) / (right - left)
            )
    if abs(float(log_ratios[-1])) <= zero_tolerance:
        crossings.append(float(d4_values[-1]))

    domain = (float(d4_values[0]), float(d4_values[-1]))
    crossing_tolerance = 1.0e-10 * max(
        abs(domain[0]), abs(domain[1]), domain[1] - domain[0], 1.0
    )
    crossings = _deduplicate_sorted(crossings, crossing_tolerance)

    def evaluate(d4_value: float) -> float:
        return float(np.interp(d4_value, d4_values, log_ratios))

    allowed, excluded = _intervals_from_crossings(domain, crossings, evaluate)
    payload = _constraint_payload(
        c3=c3,
        confidence_level=confidence_level,
        d4_domain=domain,
        crossings=crossings,
        allowed_intervals=allowed,
        excluded_intervals=excluded,
        method=base["method"],
        strategy=strategy,
        limit_kind=limit_kind,
        limit_key=limit_key,
        slice_point_count=len(points),
    )
    payload["slice_points"] = [
        {
            "d4": float(d4),
            "log10_sigma_over_sigma95": float(log_ratio),
        }
        for d4, log_ratio in points
    ]
    return payload


def chebyshev_fit_d4_constraint(
    fit: Mapping[str, Any],
    *,
    target_value: float,
    c3: float = 0.0,
    confidence_level: float = 0.95,
    strategy: str | None = "legacy-chebyshev-fit",
    limit_kind: str | None = "cut",
) -> dict[str, Any]:
    """Solve a fitted c3/d4 Chebyshev surface for its fixed-c3 CL crossings."""

    confidence_level = float(confidence_level)
    target_value = float(target_value)
    base = {
        "status": "unavailable",
        "confidence_level": confidence_level,
        "c3": float(c3),
        "parameter_definition": "d4 = kappa4 - 1",
        "strategy": strategy,
        "limit_kind": limit_kind,
        "limit_key": "effective_sigma_eff_fb",
        "method": "roots of the fitted Chebyshev sigma*eff surface at fixed c3",
    }
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if not math.isfinite(target_value) or target_value <= 0.0:
        return {**base, "reason": "the sigma*eff confidence-level target is not positive"}
    if fit.get("status") != "ok":
        return {
            **base,
            "reason": f"Chebyshev fit status is {fit.get('status', 'missing')}",
        }

    try:
        terms = [tuple(int(value) for value in term) for term in fit["terms"]]
        coefficients = np.asarray(fit["coefficients"], dtype=float)
        k3_range = tuple(float(value) for value in fit["k3_range"])
        k4_range = tuple(float(value) for value in fit["k4_range"])
    except (KeyError, TypeError, ValueError) as error:
        return {**base, "reason": f"invalid Chebyshev fit metadata: {error}"}
    if (
        len(terms) != len(coefficients)
        or not terms
        or np.any(~np.isfinite(coefficients))
        or k3_range[0] >= k3_range[1]
        or k4_range[0] >= k4_range[1]
    ):
        return {**base, "reason": "invalid Chebyshev terms, coefficients, or fit ranges"}

    k3 = 1.0 + float(c3)
    if not k3_range[0] <= k3 <= k3_range[1]:
        return {**base, "reason": "fixed c3 lies outside the Chebyshev fit domain"}
    x3 = (2.0 * k3 - k3_range[0] - k3_range[1]) / (
        k3_range[1] - k3_range[0]
    )
    max_i = max(i for i, _ in terms)
    t3 = np.polynomial.chebyshev.chebvander(x3, max_i).reshape(-1)
    max_j = max(j for _, j in terms)
    slice_coefficients = np.zeros(max_j + 1, dtype=float)
    for coefficient, (i, j) in zip(coefficients, terms):
        slice_coefficients[j] += coefficient * t3[i]
    slice_coefficients[0] -= target_value

    coefficient_scale = max(float(np.max(np.abs(slice_coefficients))), target_value, 1.0e-30)
    trimmed = np.polynomial.chebyshev.chebtrim(
        slice_coefficients,
        tol=1.0e-12 * coefficient_scale,
    )
    roots_x4 = np.polynomial.chebyshev.chebroots(trimmed)
    real_tolerance = 1.0e-9
    real_roots = [
        float(root.real)
        for root in roots_x4
        if abs(float(root.imag)) <= real_tolerance
        and -1.0 - real_tolerance <= float(root.real) <= 1.0 + real_tolerance
    ]
    real_roots = [min(max(root, -1.0), 1.0) for root in real_roots]

    def x4_to_d4(x4: float) -> float:
        k4 = 0.5 * (
            x4 * (k4_range[1] - k4_range[0])
            + k4_range[0]
            + k4_range[1]
        )
        return float(k4 - 1.0)

    domain = (float(k4_range[0] - 1.0), float(k4_range[1] - 1.0))
    crossings = _deduplicate_sorted(
        [x4_to_d4(root) for root in real_roots],
        1.0e-9 * max(domain[1] - domain[0], 1.0),
    )

    def evaluate(d4_value: float) -> float:
        k4 = 1.0 + float(d4_value)
        x4 = (2.0 * k4 - k4_range[0] - k4_range[1]) / (
            k4_range[1] - k4_range[0]
        )
        return float(
            np.polynomial.chebyshev.chebval(x4, slice_coefficients)
        )

    allowed, excluded = _intervals_from_crossings(domain, crossings, evaluate)
    payload = _constraint_payload(
        c3=c3,
        confidence_level=confidence_level,
        d4_domain=domain,
        crossings=crossings,
        allowed_intervals=allowed,
        excluded_intervals=excluded,
        method=base["method"],
        strategy=strategy,
        limit_kind=limit_kind,
        limit_key=base["limit_key"],
    )
    payload["target_effective_sigma_eff_fb"] = target_value
    return payload


def format_d4_constraint(constraint: Mapping[str, Any]) -> str:
    """Return a compact terminal summary for a fixed-c3 d4 constraint."""

    confidence_level = _finite_float(constraint.get("confidence_level"))
    confidence_label = (
        "95"
        if confidence_level is None
        else f"{100.0 * confidence_level:g}"
    )
    c3 = _finite_float(constraint.get("c3"))
    c3_label = "0" if c3 is None else f"{c3:g}"
    strategy = constraint.get("strategy")
    limit_kind = constraint.get("limit_kind")
    detail = ", ".join(
        str(value)
        for value in (strategy, limit_kind)
        if value not in (None, "")
    )
    heading = f"Expected {confidence_label}% C.L. constraint on d4 for c3 = {c3_label}"
    if detail:
        heading += f" ({detail})"
    if constraint.get("status") != "ok":
        return f"{heading}\n  unavailable: {constraint.get('reason', 'unknown reason')}"

    intervals = constraint.get("allowed_intervals") or []
    if not intervals:
        domain = constraint.get("d4_domain") or [None, None]
        return (
            f"{heading}\n"
            f"  no allowed d4 values in the scanned range "
            f"[{float(domain[0]):.6g}, {float(domain[1]):.6g}]"
        )

    interval_text = " U ".join(
        f"[{float(interval['lower']):.6g}, {float(interval['upper']):.6g}]"
        for interval in intervals
    )
    suffix = (
        " (limited by the scanned d4 range)"
        if constraint.get("scan_boundary_limited")
        else ""
    )
    return f"{heading}\n  allowed d4 = {interval_text}{suffix}"


__all__ = [
    "chebyshev_fit_d4_constraint",
    "fixed_c3_scan_constraint",
    "format_d4_constraint",
]
