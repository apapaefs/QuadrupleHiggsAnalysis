"""Shared, lightweight plotting conventions for the c3/d4 studies.

This module deliberately has no ROOT, XGBoost, scikit-learn or Matplotlib
dependency.  In addition to style constants it contains the small NumPy-only
cross-section fit used by the legacy contour family, so replotting existing
tables does not import the training stack.
"""

import gzip
import math
import re
from pathlib import Path

import numpy as np


DEFAULT_HHHH_PERTURBATIVITY_MH = 125.0
DEFAULT_HHHH_PERTURBATIVITY_V = 246.0
DEFAULT_HHHH_PERTURBATIVITY_LEVEL = 0.5
DEFAULT_HHHH_PERTURBATIVITY_SQRTS = np.arange(200.0, 5000.0, 10.0)
DEFAULT_C3D4_OVERLAY_AXIS_LABEL_FONTSIZE = 20
DEFAULT_C3D4_CHEBYSHEV_TERMS = (
    [(i, 0) for i in range(0, 7)]
    + [(i, 1) for i in range(0, 5)]
    + [(i, 2) for i in range(0, 3)]
)
DEFAULT_HHHH_XSEC_WIDE_RUNNUM = "3"
DEFAULT_HHHH_XSEC_EXPECTED_WIDE_RUNS = 17
ATL_PHYS_PUB_2025_003_LABEL = r"$gg\rightarrow hhh \rightarrow 6 b$, ATL-PHYS-PUB-2025-003 (no syst.)"
ATL_PHYS_PUB_2025_003_SOURCE_URL = "https://cds.cern.ch/record/2924772/files/ATL-PHYS-PUB-2025-003.pdf"
ATL_PHYS_PUB_2025_003_FIGURE = "Figure 7 black no-systematics curve"
ATL_PHYS_PUB_2025_003_NO_SYST_KAPPA34 = np.array(
    [
        (11.45514, -2.49205),
        (11.27509, 2.66879),
        (11.02789, 8.50124),
        (10.62500, 18.82291),
        (10.59327, 19.51219),
        (10.07379, 30.50548),
        (9.97491, 32.62637),
        (9.50561, 41.49876),
        (9.32482, 44.90986),
        (8.87766, 52.50972),
        (8.67547, 55.83245),
        (8.16632, 63.50301),
        (8.02538, 65.50018),
        (7.37530, 73.96607),
        (7.32881, 74.51396),
        (6.72521, 80.62920),
        (6.22786, 85.50725),
        (6.07512, 86.76211),
        (5.42503, 91.26900),
        (4.77494, 95.22800),
        (4.49749, 96.50053),
        (4.12559, 97.82609),
        (3.47550, 99.24001),
        (2.82541, 99.75256),
        (2.17532, 99.32838),
        (1.52524, 98.00283),
        (1.08102, 96.50053),
        (0.87515, 95.59915),
        (0.22506, 91.76388),
        (-0.42503, 87.38070),
        (-0.66116, 85.50725),
        (-1.07512, 81.61895),
        (-1.72447, 75.18558),
        (-1.78571, 74.51396),
        (-2.37456, 67.05550),
        (-2.64020, 63.50301),
        (-3.02465, 57.97102),
        (-3.37957, 52.50972),
        (-3.67473, 47.73772),
        (-4.02671, 41.49876),
        (-4.32482, 36.17886),
        (-4.60670, 30.50548),
        (-4.97491, 23.29445),
        (-5.14168, 19.51219),
        (-5.62500, 9.17285),
        (-5.65083, 8.50124),
        (-6.07586, -2.49205),
        (-6.27509, -7.22870),
        (-6.48908, -13.48533),
        (-6.91558, -24.49629),
        (-6.92444, -24.72605),
        (-7.05800, -28.95016),
        (-7.24174, -35.48957),
        (-7.57453, -44.94521),
        (-7.61659, -46.48286),
        (-7.90806, -57.49381),
        (-8.22462, -67.49735),
        (-8.24823, -68.48710),
        (-8.49395, -79.48038),
        (-8.78542, -90.49134),
        (-8.87470, -93.92011),
        (-9.01638, -101.48463),
        (-9.23111, -112.47791),
        (-9.46798, -123.48887),
        (-9.52479, -126.45811),
        (-9.63031, -134.48215),
        (-9.75945, -145.49311),
        (-9.86201, -156.48639),
        (-9.90186, -165.57087),
        (-9.89669, -167.47968),
        (-9.85021, -174.10746),
        (-9.78896, -177.39484),
        (-9.75649, -178.49063),
        (-9.63179, -181.58360),
        (-9.52479, -183.17427),
        (-9.40452, -184.00495),
        (-9.24292, -184.04030),
        (-8.87470, -182.25521),
        (-8.61423, -180.32874),
        (-8.40319, -178.49063),
        (-8.22462, -177.09438),
        (-7.95971, -174.46094),
        (-7.57453, -169.91870),
        (-7.37234, -167.47968),
        (-6.92444, -162.05373),
        (-6.45956, -156.48639),
        (-6.27509, -154.20643),
        (-5.62500, -146.35914),
        (-5.55121, -145.49311),
        (-4.97491, -138.26441),
        (-4.63474, -134.48215),
        (-4.32482, -130.77059),
        (-3.67473, -123.98374),
        (-3.62308, -123.48887),
        (-3.02465, -117.00248),
        (-2.50590, -112.47791),
        (-2.37456, -111.18770),
        (-1.83220, -106.29198),
        (-1.72447, -105.42595),
        (-1.14448, -101.48463),
        (-1.07512, -100.91905),
        (-0.52538, -96.96006),
        (-0.42503, -96.34146),
        (0.22506, -92.94804),
        (0.87515, -90.49134),
        (0.87884, -90.49134),
        (1.52524, -88.12301),
        (2.17532, -86.67374),
        (2.82541, -86.05515),
        (3.47550, -86.19654),
        (4.12559, -87.11559),
        (4.77494, -88.86533),
        (5.18079, -90.49134),
        (5.42503, -91.23365),
        (6.07512, -93.92011),
        (6.72521, -97.49028),
        (7.28527, -101.48463),
        (7.37530, -102.01485),
        (8.02538, -106.39802),
        (8.67547, -112.03606),
        (8.72417, -112.47791),
        (9.32482, -117.32061),
        (9.94835, -123.48887),
        (9.97491, -123.71863),
        (10.62500, -129.53341),
        (11.11349, -134.48215),
        (11.27509, -136.00212),
        (11.92518, -141.90527),
        (12.33766, -145.49311),
        (12.57527, -147.45493),
        (12.99734, -150.30046),
        (13.22535, -151.29021),
        (13.64448, -152.47437),
        (13.79649, -152.15624),
        (13.87544, -151.62602),
        (14.03040, -149.69954),
        (14.17208, -146.78332),
        (14.21930, -145.49311),
        (14.29900, -142.25875),
        (14.35655, -137.11559),
        (14.37057, -134.48215),
        (14.37131, -129.92224),
        (14.33146, -123.48887),
        (14.21488, -112.47791),
        (14.06656, -101.48463),
        (13.90791, -90.49134),
        (13.87544, -88.56487),
        (13.66293, -79.48038),
        (13.41795, -68.48710),
        (13.22535, -59.11983),
        (13.17960, -57.49381),
        (12.85714, -46.48286),
        (12.58264, -35.48957),
        (12.57527, -35.20679),
        (12.43506, -30.87664),
        (12.20632, -24.49629),
        (11.92518, -14.95228),
        (11.87057, -13.48533),
        (11.45514, -2.49205),
    ],
    dtype=float,
)


def _make_hhhh_xsec_log_levels(ratio):
    positive = ratio[np.isfinite(ratio) & (ratio > 0.0)]
    if len(positive) == 0:
        raise RuntimeError("No positive normalized hhhh cross-section values found")
    min_value = float(np.min(positive))
    max_value = float(np.max(positive))
    lo = math.floor(math.log10(min_value))
    hi = math.ceil(math.log10(max_value))
    # ``np.logspace(lo, hi, n)`` is constant when the surface is exactly a
    # power of ten (the historical all-1-fb failure is one such case), and
    # Matplotlib rejects duplicate contour levels.  Widen only this degenerate
    # display range; the underlying surface is unchanged.
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    nlevels = min(max((hi - lo) * 4 + 1, 12), 80)
    return np.logspace(lo, hi, int(nlevels))


def _make_hhhh_xsec_line_levels(filled_levels):
    lo = math.floor(math.log10(filled_levels[0]))
    hi = math.ceil(math.log10(filled_levels[-1]))
    levels = []
    for power in range(lo, hi + 1):
        value = 10.0 ** power
        if filled_levels[0] <= value <= filled_levels[-1]:
            levels.append(value)
    return levels


def _format_hhhh_xsec_level(value):
    if value >= 1000.0 or value < 0.01:
        return "%.0e" % value
    if value >= 10.0:
        return "%.0f" % value
    return "%.2g" % value


def _hhhh_perturbativity_partial_wave(s, c3_grid, d4_grid):
    k3_squared = (1.0 + c3_grid) ** 2
    k4 = 1.0 + d4_grid
    mh2 = DEFAULT_HHHH_PERTURBATIVITY_MH ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        prefactor = (
            3.0
            * mh2
            * np.sqrt(s ** 2 - 4.0 * mh2 * s)
            / (32.0 * np.pi * s * (s - mh2) * DEFAULT_HHHH_PERTURBATIVITY_V ** 2)
        )
        bracket = (
            -k4 * (s - mh2)
            - 3.0 * k3_squared * mh2
            + (
                6.0
                * k3_squared
                * mh2
                * (s - mh2)
                / (s - 4.0 * mh2)
                * np.log(s / mh2 - 3.0)
            )
        )
        value = np.abs(prefactor * bracket)
    return np.where(np.isfinite(value), value, 0.0)


def _hhhh_perturbativity_grid(c3_grid, d4_grid):
    max_partial_wave = np.zeros_like(c3_grid, dtype=float)
    for sqrt_s in DEFAULT_HHHH_PERTURBATIVITY_SQRTS:
        current = _hhhh_perturbativity_partial_wave(sqrt_s ** 2, c3_grid, d4_grid)
        max_partial_wave = np.maximum(max_partial_wave, current)
    return max_partial_wave


def _finite_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_hhhh_xsec_pb(lhe_file):
    with gzip.open(lhe_file, "rt", errors="ignore") as stream:
        for line in stream:
            match = re.search(
                r"Integrated weight \(pb\)\s*:\s*([0-9.eE+-]+)", line
            )
            if match is not None:
                return float(match.group(1))
    raise RuntimeError(f"No Integrated weight found in {lhe_file}")


def _read_hhhh_xsec_error_pb(process_dir, run_name, xsec_pb):
    html_file = Path(process_dir) / "HTML" / run_name / "results.html"
    if html_file.exists():
        html = html_file.read_text(errors="ignore")
        html = (
            html.replace("&plusmn;", "+/-")
            .replace("&#177;", "+/-")
            .replace("\u00b1", "+/-")
        )
        match = re.search(
            r"<b>s=\s*([0-9.eE+-]+)\s*\+/-\s*([0-9.eE+-]+)\s*\(pb\)</b>",
            html,
        )
        if match is not None:
            return float(match.group(2))
    return max(abs(float(xsec_pb)) * 0.01, 1.0e-30)


def _read_hhhh_xsec_points(source_dir):
    """Read the same completed MG5 ``run_gg_4h_*`` points as the legacy code."""

    process_dir = Path(source_dir)
    event_dir = process_dir / "Events"
    if not event_dir.exists():
        raise RuntimeError(
            f"hhhh cross-section Events directory not found: {event_dir}"
        )
    points = []
    counts = {}
    prefix = "run_gg_4h_"
    for run_dir in sorted(event_dir.glob(prefix + "*")):
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name
        parts = run_name[len(prefix) :].split("_")
        if len(parts) != 3:
            continue
        run_number, c3_text, d4_text = parts
        try:
            c3 = float(c3_text)
            d4 = float(d4_text)
        except ValueError:
            continue
        lhe_file = run_dir / "unweighted_events.lhe.gz"
        if not lhe_file.exists():
            continue
        xsec_pb = _read_hhhh_xsec_pb(lhe_file)
        points.append(
            {
                "c3": c3,
                "d4": d4,
                "xsec_pb": xsec_pb,
                "xsec_error_pb": _read_hhhh_xsec_error_pb(
                    process_dir, run_name, xsec_pb
                ),
                "runnum": run_number,
                "run_name": run_name,
            }
        )
        counts[run_number] = counts.get(run_number, 0) + 1
    if not points:
        raise RuntimeError(
            f"No completed hhhh cross-section runs found in {event_dir}"
        )
    points.sort(key=lambda row: (row["c3"], row["d4"], row["runnum"]))
    counts = dict(
        sorted(
            counts.items(),
            key=lambda item: (
                0,
                int(item[0]),
            )
            if str(item[0]).isdigit()
            else (1, str(item[0])),
        )
    )
    return points, counts


def _scale_to_chebyshev(value, value_range):
    minimum, maximum = value_range
    return (2.0 * value - minimum - maximum) / (maximum - minimum)


def _chebyshev_t(order, x):
    if order == 0:
        return np.ones_like(x, dtype=float)
    if order == 1:
        return np.asarray(x, dtype=float)
    previous = np.ones_like(x, dtype=float)
    current = np.asarray(x, dtype=float)
    for _ in range(2, order + 1):
        previous, current = current, 2.0 * x * current - previous
    return current


def _chebyshev_row(c3, d4, terms, k3_range, k4_range):
    x3 = _scale_to_chebyshev(1.0 + float(c3), k3_range)
    x4 = _scale_to_chebyshev(1.0 + float(d4), k4_range)
    return np.asarray(
        [_chebyshev_t(i, x3) * _chebyshev_t(j, x4) for i, j in terms],
        dtype=float,
    )


def _fit_c3d4_chebyshev(rows, value_key, error_key, terms, k3_range, k4_range):
    fit_points = []
    for row in rows:
        c3 = _finite_float(row.get("c3"))
        d4 = _finite_float(row.get("d4"))
        value = _finite_float(row.get(value_key))
        if c3 is None or d4 is None or value is None:
            continue
        error = _finite_float(row.get(error_key))
        if error is None or error <= 0.0:
            error = max(abs(value) * 0.05, 1.0e-30)
        fit_points.append((c3, d4, value, error))
    if len(fit_points) < len(terms):
        return {
            "status": "skipped",
            "reason": (
                f"need at least {len(terms)} finite c3/d4 points; "
                f"got {len(fit_points)}"
            ),
            "n_points": len(fit_points),
            "n_terms": len(terms),
        }
    design = np.asarray(
        [
            _chebyshev_row(c3, d4, terms, k3_range, k4_range)
            for c3, d4, _, _ in fit_points
        ],
        dtype=float,
    )
    values = np.asarray([point[2] for point in fit_points], dtype=float)
    errors = np.asarray([point[3] for point in fit_points], dtype=float)
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design / errors[:, None], values / errors, rcond=None
    )
    if int(rank) < len(terms):
        return {
            "status": "skipped",
            "reason": (
                f"rank-deficient c3/d4 fit: rank {int(rank)} for "
                f"{len(terms)} terms"
            ),
            "n_points": len(fit_points),
            "n_terms": len(terms),
            "rank": int(rank),
            "condition": float(np.linalg.cond(design)),
            "singular_values": [float(value) for value in singular_values],
        }
    predictions = np.dot(design, coefficients)
    residuals = values - predictions
    degrees_of_freedom = max(len(values) - len(coefficients), 1)
    return {
        "status": "ok",
        "n_points": len(fit_points),
        "n_terms": len(terms),
        "rank": int(rank),
        "condition": float(np.linalg.cond(design)),
        "chi2_dof": float(
            np.dot(residuals / errors, residuals / errors) / degrees_of_freedom
        ),
        "terms": [[int(i), int(j)] for i, j in terms],
        "k3_range": [float(k3_range[0]), float(k3_range[1])],
        "k4_range": [float(k4_range[0]), float(k4_range[1])],
        "coefficients": [float(value) for value in coefficients],
        "singular_values": [float(value) for value in singular_values],
        "residual_rms": float(np.sqrt(np.mean(np.square(residuals)))),
        "max_abs_residual": float(np.max(np.abs(residuals))),
    }


def _evaluate_c3d4_chebyshev(c3, d4, fit):
    terms = [tuple(term) for term in fit["terms"]]
    row = _chebyshev_row(c3, d4, terms, fit["k3_range"], fit["k4_range"])
    return float(np.dot(row, np.asarray(fit["coefficients"], dtype=float)))


def _evaluate_c3d4_chebyshev_grid(
    fit, c3_range, d4_range, number_c3, number_d4
):
    terms = [tuple(term) for term in fit["terms"]]
    coefficients = np.asarray(fit["coefficients"], dtype=float)
    c3_values = np.linspace(float(c3_range[0]), float(c3_range[1]), int(number_c3))
    d4_values = np.linspace(float(d4_range[0]), float(d4_range[1]), int(number_d4))
    c3_grid, d4_grid = np.meshgrid(c3_values, d4_values)
    x3 = _scale_to_chebyshev(1.0 + c3_grid, fit["k3_range"])
    x4 = _scale_to_chebyshev(1.0 + d4_grid, fit["k4_range"])
    t3 = [_chebyshev_t(i, x3) for i in range(max(i for i, _ in terms) + 1)]
    t4 = [_chebyshev_t(j, x4) for j in range(max(j for _, j in terms) + 1)]
    values = np.zeros_like(c3_grid, dtype=float)
    for coefficient, (i, j) in zip(coefficients, terms):
        values += coefficient * t3[i] * t4[j]
    return c3_grid, d4_grid, values


def _atlas_phys_pub_2025_003_c3d4_curve():
    curve = np.array(ATL_PHYS_PUB_2025_003_NO_SYST_KAPPA34, dtype=float, copy=True)
    curve[:, 0] -= 1.0
    curve[:, 1] -= 1.0
    return curve


def _plot_atlas_phys_pub_2025_003_curve(ax):
    curve = _atlas_phys_pub_2025_003_c3d4_curve()
    ax.plot(
        curve[:, 0],
        curve[:, 1],
        color="blue",
        linewidth=2.0,
        label=ATL_PHYS_PUB_2025_003_LABEL,
    )
    return {
        "label": ATL_PHYS_PUB_2025_003_LABEL,
        "source": ATL_PHYS_PUB_2025_003_SOURCE_URL,
        "figure": ATL_PHYS_PUB_2025_003_FIGURE,
        "coordinate_system": "digitized in kappa3,kappa4 and plotted as c3=kappa3-1, d4=kappa4-1",
        "n_points": int(len(curve)),
        "c3_min": float(np.min(curve[:, 0])),
        "c3_max": float(np.max(curve[:, 0])),
        "d4_min": float(np.min(curve[:, 1])),
        "d4_max": float(np.max(curve[:, 1])),
    }


def _plot_sm_marker(ax):
    ax.plot(
        [0.0],
        [0.0],
        marker="*",
        color="red",
        markeredgecolor="black",
        markeredgewidth=0.7,
        markersize=13,
        linestyle="None",
        zorder=12,
    )
    ax.annotate(
        "SM",
        xy=(0.0, 0.0),
        xytext=(8, 8),
        textcoords="offset points",
        color="red",
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
        zorder=12,
    )


__all__ = [
    "ATL_PHYS_PUB_2025_003_FIGURE",
    "ATL_PHYS_PUB_2025_003_LABEL",
    "ATL_PHYS_PUB_2025_003_NO_SYST_KAPPA34",
    "ATL_PHYS_PUB_2025_003_SOURCE_URL",
    "DEFAULT_C3D4_CHEBYSHEV_TERMS",
    "DEFAULT_C3D4_OVERLAY_AXIS_LABEL_FONTSIZE",
    "DEFAULT_HHHH_PERTURBATIVITY_LEVEL",
    "DEFAULT_HHHH_PERTURBATIVITY_MH",
    "DEFAULT_HHHH_PERTURBATIVITY_SQRTS",
    "DEFAULT_HHHH_PERTURBATIVITY_V",
    "DEFAULT_HHHH_XSEC_EXPECTED_WIDE_RUNS",
    "DEFAULT_HHHH_XSEC_WIDE_RUNNUM",
    "_atlas_phys_pub_2025_003_c3d4_curve",
    "_evaluate_c3d4_chebyshev",
    "_evaluate_c3d4_chebyshev_grid",
    "_fit_c3d4_chebyshev",
    "_format_hhhh_xsec_level",
    "_hhhh_perturbativity_grid",
    "_hhhh_perturbativity_partial_wave",
    "_make_hhhh_xsec_line_levels",
    "_make_hhhh_xsec_log_levels",
    "_plot_atlas_phys_pub_2025_003_curve",
    "_plot_sm_marker",
    "_read_hhhh_xsec_points",
]
