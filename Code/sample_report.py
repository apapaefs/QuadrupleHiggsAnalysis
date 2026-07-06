"""Utilities for 4H sample-rate reports.

The helpers in this module are intentionally independent of ROOT and XGBoost so
the normalization conventions can be tested without the full HEP runtime.
"""

from __future__ import annotations

import html
import math
import re
from pathlib import Path


def signal_generation_rate_factor(hbb_branching_ratio, hbb_power, k_factor):
    """Rate factor before tag efficiencies for a hhhh -> 8b signal sample."""

    return float(k_factor) * float(hbb_branching_ratio) ** int(hbb_power)


def signal_tag_rate_factor(btagging_rate, btag_power):
    """Per-event final tag factor for a signal sample."""

    return float(btagging_rate) ** int(btag_power)


def _metadata_text(metadata):
    metadata = metadata or {}
    keys = ("process_id", "description", "local_lhe", "process", "notes", "file")
    return " ".join(str(metadata.get(key, "")) for key in keys).lower().replace("_", " ")


def metadata_mentions_z(metadata):
    text = _metadata_text(metadata)
    return bool(re.search(r"(^|[^a-z0-9])z(?:0)?($|[^a-z0-9])", text))


def metadata_has_zbb_decay(metadata):
    text = _metadata_text(metadata)
    patterns = (
        r"z\s*(?:->|to)\s*b\s*(?:bbar|anti-?b|bar b|b)",
        r"z\s*(?:bb|bbar)",
        r"zbb",
        r"decayos",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def background_needs_zbb_branching(metadata):
    return metadata_mentions_z(metadata) and not metadata_has_zbb_decay(metadata)


def background_generation_rate_factor(metadata, k_factor, zbb_branching_ratio):
    """Rate factor before tag/mistag efficiencies for a background sample."""

    factor = float(k_factor)
    if background_needs_zbb_branching(metadata):
        factor *= float(zbb_branching_ratio)
    return factor


def background_tag_rate_factor(metadata, btagging_rate, c_mistag_rate, light_mistag_rate):
    """Final tag/mistag factor inferred from the background flavor composition."""

    metadata = metadata or {}
    return (
        float(btagging_rate) ** int(metadata.get("b_quarks", 0))
        * float(c_mistag_rate) ** int(metadata.get("c_quarks", 0))
        * float(light_mistag_rate) ** int(metadata.get("light_jets", 0))
    )


def cutflow_rates(
    raw_xsec_fb,
    generation_rate_factor,
    tag_rate_factor,
    normalisation_weight,
    input_weight_sum,
    selected_weight_sum,
):
    """Cross sections for the generation, input-selection, and XGBoost stages."""

    generation_xsec = float(raw_xsec_fb) * float(generation_rate_factor)
    normalisation = float(normalisation_weight or 0.0)
    if normalisation == 0.0:
        return {
            "generation_xsec_fb": generation_xsec,
            "input_xsec_fb": 0.0,
            "xgboost_xsec_fb": 0.0,
        }

    tagged_generation_xsec = generation_xsec * float(tag_rate_factor)
    return {
        "generation_xsec_fb": generation_xsec,
        "input_xsec_fb": tagged_generation_xsec * float(input_weight_sum) / normalisation,
        "xgboost_xsec_fb": tagged_generation_xsec * float(selected_weight_sum) / normalisation,
    }


def effective_entries(weights):
    """Effective MC entries for weighted events."""

    import numpy as np

    weights = np.asarray(weights, dtype=float)
    weights = weights[np.isfinite(weights)]
    if weights.size == 0:
        return 0.0
    sum_abs = float(np.sum(np.abs(weights)))
    sum_sq = float(np.sum(np.square(weights)))
    if sum_sq <= 0.0:
        return float(weights.size)
    return sum_abs * sum_abs / sum_sq


def _chi2_ppf(probability, degrees_of_freedom):
    try:
        from scipy.stats import chi2

        return float(chi2.ppf(float(probability), float(degrees_of_freedom)))
    except Exception:
        from statistics import NormalDist

        probability = min(max(float(probability), 1.0e-12), 1.0 - 1.0e-12)
        degrees_of_freedom = float(degrees_of_freedom)
        z = NormalDist().inv_cdf(probability)
        # Wilson-Hilferty approximation, used only when scipy is unavailable.
        return degrees_of_freedom * (1.0 - 2.0 / (9.0 * degrees_of_freedom) + z * math.sqrt(2.0 / (9.0 * degrees_of_freedom))) ** 3


def poisson_count_interval(count, confidence_level=0.95):
    """Garwood Poisson count interval.

    For zero observed counts this returns the usual one-sided upper limit,
    -log(1-CL), which is the convention needed to avoid reporting 0 +/- 0.
    """

    count = int(count)
    if count < 0:
        raise ValueError("Poisson count must be non-negative")
    confidence_level = float(confidence_level)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    alpha = 1.0 - confidence_level
    if count == 0:
        return 0.0, -math.log(alpha)
    lower = 0.5 * _chi2_ppf(alpha / 2.0, 2 * count)
    upper = 0.5 * _chi2_ppf(1.0 - alpha / 2.0, 2 * (count + 1))
    return float(lower), float(upper)


def poisson_event_interval(
    selected_entries,
    expected_events,
    input_entries=None,
    expected_input_events=None,
    confidence_level=0.95,
):
    """Scale a Poisson count interval to a luminosity-normalized yield."""

    selected_entries = int(selected_entries)
    expected_events = float(expected_events or 0.0)
    count_lower, count_upper = poisson_count_interval(selected_entries, confidence_level)

    if selected_entries > 0:
        event_scale = expected_events / float(selected_entries)
        event_lower = count_lower * event_scale
        event_upper = count_upper * event_scale
        return {
            "confidence_level": float(confidence_level),
            "is_upper_limit": False,
            "count_lower": count_lower,
            "count_upper": count_upper,
            "event_lower": float(event_lower),
            "event_upper": float(event_upper),
            "event_error_low": float(max(0.0, expected_events - event_lower)),
            "event_error_high": float(max(0.0, event_upper - expected_events)),
        }

    input_entries = int(input_entries or 0)
    expected_input_events = float(expected_input_events or 0.0)
    event_scale = expected_input_events / float(input_entries) if input_entries > 0 else 0.0
    event_upper = count_upper * event_scale
    return {
        "confidence_level": float(confidence_level),
        "is_upper_limit": True,
        "count_lower": count_lower,
        "count_upper": count_upper,
        "event_lower": 0.0,
        "event_upper": float(event_upper),
        "event_error_low": 0.0,
        "event_error_high": float(event_upper),
    }


def attach_poisson_event_interval(
    row,
    selected_entries_key,
    expected_events_key,
    input_entries_key,
    expected_input_events_key,
    output_prefix,
    confidence_level=0.95,
):
    """Add Poisson 95% CL yield interval fields to a result row."""

    interval = poisson_event_interval(
        selected_entries=row.get(selected_entries_key, 0),
        expected_events=row.get(expected_events_key, 0.0),
        input_entries=row.get(input_entries_key, 0),
        expected_input_events=row.get(expected_input_events_key, 0.0),
        confidence_level=confidence_level,
    )
    row[f"{output_prefix}_lower_95cl"] = interval["event_lower"]
    row[f"{output_prefix}_upper_95cl"] = interval["event_upper"]
    row[f"{output_prefix}_error_low_95cl"] = interval["event_error_low"]
    row[f"{output_prefix}_error_high_95cl"] = interval["event_error_high"]
    row[f"{output_prefix}_is_upper_limit"] = interval["is_upper_limit"]
    row[f"{output_prefix}_confidence_level"] = interval["confidence_level"]
    if interval["is_upper_limit"]:
        row[f"{output_prefix}_upper_limit_95cl"] = interval["event_upper"]
    return row


def threshold_mc_stat_summary(scores, labels, weights, threshold, sources=None):
    import numpy as np

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    weights = np.asarray(weights, dtype=float)
    selected_background = (labels == 0) & (scores >= float(threshold))
    selected_weights = weights[selected_background]
    summary = {
        "selected_background_entries": int(np.sum(selected_background)),
        "selected_background_effective_entries": float(effective_entries(selected_weights)),
        "selected_background_source_entries": {},
    }
    if sources is not None:
        sources = np.asarray(sources)
        per_source = {}
        for source in sorted(set(str(value) for value in sources[labels == 0])):
            per_source[source] = int(np.sum(selected_background & (sources.astype(str) == source)))
        summary["selected_background_source_entries"] = per_source
    return summary


def best_significance_threshold(
    scores,
    labels,
    physical_weights,
    systematics=0.0,
    background_sources=None,
    min_background_mc_entries=0,
    min_background_effective_entries=0.0,
    min_background_mc_entries_per_sample=0,
):
    import numpy as np

    thresholds = np.linspace(0.0, 1.0, 501)
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    physical_weights = np.asarray(physical_weights, dtype=float)
    background_sources = None if background_sources is None else np.asarray(background_sources)
    total_signal = np.sum(physical_weights[labels == 1])
    total_background = np.sum(physical_weights[labels == 0])

    def empty_best():
        return {
            "threshold": 0.5,
            "signal_events": 0.0,
            "background_events": 0.0,
            "significance": 0.0,
            "signal_efficiency": 0.0,
            "background_efficiency": 0.0,
            "selected_background_entries": 0,
            "selected_background_effective_entries": 0.0,
            "selected_background_source_entries": {},
            "mc_stat_requirement_satisfied": False,
        }

    best = empty_best()
    best_unconstrained = empty_best()

    min_background_mc_entries = int(min_background_mc_entries or 0)
    min_background_effective_entries = float(min_background_effective_entries or 0.0)
    min_background_mc_entries_per_sample = int(min_background_mc_entries_per_sample or 0)

    for threshold in thresholds:
        selected = scores >= threshold
        signal = float(np.sum(physical_weights[(labels == 1) & selected]))
        background = float(np.sum(physical_weights[(labels == 0) & selected]))
        if background <= 0.0:
            continue
        denominator = math.sqrt(background + (float(systematics) * background) ** 2)
        significance = signal / denominator if denominator > 0.0 else 0.0
        stats = threshold_mc_stat_summary(scores, labels, physical_weights, threshold, sources=background_sources)
        candidate = {
            "threshold": float(threshold),
            "signal_events": signal,
            "background_events": background,
            "significance": float(significance),
            "signal_efficiency": float(signal / total_signal) if total_signal > 0 else 0.0,
            "background_efficiency": float(background / total_background) if total_background > 0 else 0.0,
            **stats,
            "mc_stat_requirement_satisfied": True,
        }
        if significance > best_unconstrained["significance"]:
            best_unconstrained = dict(candidate)

        passes = (
            stats["selected_background_entries"] >= min_background_mc_entries
            and stats["selected_background_effective_entries"] >= min_background_effective_entries
        )
        if passes and min_background_mc_entries_per_sample > 0 and background_sources is not None:
            source_entries = stats["selected_background_source_entries"]
            if source_entries:
                passes = min(source_entries.values()) >= min_background_mc_entries_per_sample
        if not passes:
            continue
        if significance > best["significance"]:
            best = dict(candidate)

    if best["significance"] > 0.0:
        return best
    best_unconstrained["mc_stat_requirement_satisfied"] = False
    best_unconstrained["mc_stat_warning"] = (
        "No scanned threshold satisfied the requested background MC-statistics requirement; "
        "falling back to the unconstrained significance optimum."
    )
    return best_unconstrained


def safe_feature_filename(name):
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
    return stem or "feature"


def html_escape(value):
    return html.escape(str(value), quote=True)


def observable_axis_label(name):
    """Axis labels shared by XGBoost and LHE validation shape reports."""

    name = str(name)
    validation_labels = {
        "b_pt_all": r"$p_T(b)$ [GeV]",
        "b1_pt": r"$p_T(b_1)$ [GeV]",
        "b2_pt": r"$p_T(b_2)$ [GeV]",
        "b3_pt": r"$p_T(b_3)$ [GeV]",
        "b4_pt": r"$p_T(b_4)$ [GeV]",
        "dr_bb_all": r"$\Delta R(b,b)$",
        "m_bb_all": r"$m(b,b)$ [GeV]",
        "m_4b": r"$m(4b)$ [GeV]",
    }
    if name in validation_labels:
        return validation_labels[name]
    if name.startswith("bjet") and name.endswith("_pt"):
        index = name.removeprefix("bjet").removesuffix("_pt")
        return rf"$p_T(b_{index})$ [GeV]"
    if name == "m8b":
        return r"$m_{8b}$ [GeV]"
    if name == "chi8":
        return r"$\chi^2_{8b}$"
    if name.startswith("delta_m_"):
        label = name.removeprefix("delta_m_").replace("_", r"\,")
        return rf"$\Delta m_{{{label}}}$ [GeV]"
    if name.startswith("higgs") and name.endswith("_pt"):
        index = name.removeprefix("higgs").removesuffix("_pt")
        return rf"$p_T(h_{index})$ [GeV]"
    if name.startswith("dr_hh_"):
        indices = name.removeprefix("dr_hh_").split("_")
        if len(indices) == 2:
            return rf"$\Delta R(h_{indices[0]},h_{indices[1]})$"
    if name.startswith("dr_bb_h"):
        index = name.removeprefix("dr_bb_h")
        return rf"$\Delta R(b,b)_{{h_{index}}}$"
    return name.replace("_", r"\_")


def _effective_entries(weights):
    import numpy as np

    weights = np.asarray(weights, dtype=float)
    weights = weights[np.isfinite(weights)]
    if weights.size == 0:
        return 0.0
    sum_abs = float(np.sum(np.abs(weights)))
    sum_sq = float(np.sum(np.square(weights)))
    if sum_sq <= 0.0:
        return float(weights.size)
    return sum_abs * sum_abs / sum_sq


def feature_bin_edges(values, sample_weights, min_bins=5, max_bins=60, entries_per_bin=10.0):
    import numpy as np

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.linspace(0.0, 1.0, min_bins + 1)

    low = float(np.min(values))
    high = float(np.max(values))
    if not math.isfinite(low) or not math.isfinite(high) or low == high:
        center = low if math.isfinite(low) else 0.0
        width = abs(center) * 0.05 if center != 0.0 else 1.0
        return np.linspace(center - width, center + width, min_bins + 1)

    q25, q75 = np.percentile(values, [25.0, 75.0])
    iqr = float(q75 - q25)
    if iqr > 0.0:
        fd_width = 2.0 * iqr / float(values.size) ** (1.0 / 3.0)
        fd_bins = int(math.ceil((high - low) / fd_width)) if fd_width > 0.0 else max_bins
    else:
        fd_bins = int(math.ceil(math.sqrt(values.size)))

    effective_counts = [_effective_entries(weights) for weights in sample_weights if len(weights) > 0]
    if effective_counts:
        stats_cap = int(max(min_bins, math.floor(min(effective_counts) / float(entries_per_bin))))
    else:
        stats_cap = max_bins
    bins = max(min_bins, min(max_bins, fd_bins, max(min_bins, stats_cap)))
    return np.linspace(low, high, bins + 1)


def normalised_histogram(values, weights, edges):
    import numpy as np

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights)
    values = values[mask]
    weights = weights[mask]
    if values.size == 0:
        zeros = np.zeros(len(edges) - 1, dtype=float)
        return zeros, zeros

    counts, _ = np.histogram(values, bins=edges, weights=weights)
    sumw2, _ = np.histogram(values, bins=edges, weights=np.square(weights))
    norm = float(np.sum(counts))
    if norm == 0.0:
        norm = float(np.sum(np.abs(weights)))
    if norm == 0.0:
        zeros = np.zeros(len(edges) - 1, dtype=float)
        return zeros, zeros
    return counts / norm, np.sqrt(sumw2) / abs(norm)


def sample_style(index):
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442"]
    linestyles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 2, 1, 2))]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    return {
        "color": colors[index % len(colors)],
        "linestyle": linestyles[index % len(linestyles)],
        "marker": markers[index % len(markers)],
    }


def write_observable_shape_plot(path, observable_name, samples):
    """Write a normalized observable-shape plot in the sample-report style."""

    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pooled_values = []
    sample_weights = []
    prepared_samples = []
    for sample in samples:
        values = np.asarray(sample.get("values", []), dtype=float)
        weights = np.asarray(sample.get("weights", []), dtype=float)
        mask = np.isfinite(values) & np.isfinite(weights)
        values = values[mask]
        weights = weights[mask]
        if values.size:
            pooled_values.append(values)
        sample_weights.append(weights)
        prepared = dict(sample)
        prepared["values"] = values
        prepared["weights"] = weights
        prepared.setdefault("style", sample_style(len(prepared_samples)))
        prepared_samples.append(prepared)

    pooled = np.concatenate(pooled_values) if pooled_values else np.asarray([])
    edges = feature_bin_edges(pooled, sample_weights)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    plotted = False
    for sample in prepared_samples:
        values = sample["values"]
        weights = sample["weights"]
        if values.size == 0:
            continue
        y, yerr = normalised_histogram(values, weights, edges)
        if not np.any(np.isfinite(y)):
            continue
        style = sample["style"]
        ax.stairs(
            y,
            edges,
            label=sample["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.7,
        )
        ax.errorbar(
            centers,
            y,
            yerr=yerr,
            fmt=style["marker"],
            markersize=3.0,
            color=style["color"],
            linestyle="none",
            linewidth=0.9,
            capsize=1.8,
        )
        plotted = True

    ax.set_xlabel(observable_axis_label(observable_name))
    ax.set_ylabel("Normalized events / bin")
    ax.grid(True, which="major", linewidth=0.4, alpha=0.35)
    if plotted:
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No finite entries", transform=ax.transAxes, ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report_index(path, plot_rows, table_path, metadata, title="4H XGBoost Input Observables", table_label="Cutflow table"):
    """Write the card-gallery report index used before XGBoost training."""

    table_rel = Path(table_path).name
    cards = []
    for row in plot_rows:
        plot_rel = Path("plots") / Path(row["path"]).name
        if "importance" in row:
            detail = "importance = %.6g" % row["importance"]
        elif "entries" in row:
            detail = "entries = %s" % row["entries"]
        else:
            detail = row.get("detail", "")
        detail_html = f"<p>{html_escape(detail)}</p>" if detail else ""
        cards.append(
            "<article>"
            f"<a href=\"{html_escape(plot_rel)}\"><img src=\"{html_escape(plot_rel)}\" alt=\"{html_escape(row['feature'])}\"></a>"
            f"<h2>{html_escape(row['feature'])}</h2>"
            f"{detail_html}"
            "</article>"
        )

    sample_items = "".join(
        f"<li>{html_escape(row['label'])}: {html_escape(row['file'])}</li>"
        for row in metadata.get("samples", [])
    )
    report_line = metadata.get("report_line")
    if report_line is None and "luminosity_fb_inverse" in metadata:
        report_line = (
            "Luminosity: {luminosity:g} fb<sup>-1</sup>; XGBoost threshold: {threshold:g}".format(
                luminosity=metadata.get("luminosity_fb_inverse", 0),
                threshold=metadata.get("threshold", 0),
            )
        )
    report_line_html = f"    <p>{report_line}</p>\n" if report_line else ""
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html_escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem; color: #1d1d1f; }}
    header {{ max-width: 1100px; margin-bottom: 1.5rem; }}
    a {{ color: #005ea8; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.1rem; }}
    article {{ border: 1px solid #ddd; border-radius: 6px; padding: 0.75rem; background: #fff; }}
    img {{ width: 100%; height: auto; display: block; }}
    h1 {{ margin-bottom: 0.25rem; }}
    h2 {{ font-size: 1rem; margin: 0.65rem 0 0.2rem; }}
    p, li {{ line-height: 1.4; }}
    code {{ background: #f4f4f4; padding: 0.12rem 0.25rem; border-radius: 3px; }}
  </style>
</head>
<body>
  <header>
    <h1>{html_escape(title)}</h1>
    <p>{html_escape(table_label)}: <a href="{html_escape(table_rel)}"><code>{html_escape(table_rel)}</code></a></p>
{report_line_html}    <ul>{sample_items}</ul>
  </header>
  <main class="grid">
    {''.join(cards)}
  </main>
</body>
</html>
"""
    Path(path).write_text(html_text)


def latex_number(value, precision=3):
    value = float(value)
    if not math.isfinite(value):
        return "--"
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    if -2 <= exponent <= 3:
        text = f"{value:.{precision}g}"
        if "e" not in text and "E" not in text:
            return text
    mantissa = value / 10.0**exponent
    mantissa_text = f"{mantissa:.{precision}g}"
    return rf"{mantissa_text}\times 10^{{{exponent}}}"


def terminal_number(value, precision=4):
    value = float(value)
    if not math.isfinite(value):
        return "--"
    if value == 0.0:
        return "0"
    return f"{value:.{precision}g}"


def terminal_label(label):
    label = str(label)
    replacements = (
        (r"\bar{b}", "bbar"),
        (r"\bar{c}", "cbar"),
        (r"\to", "->"),
        (r"\,", " "),
        ("\\ ", " "),
        ("$", ""),
        ("{", ""),
        ("}", ""),
    )
    for old, new in replacements:
        label = label.replace(old, new)
    label = re.sub(r"\\mathrm\s*([^ ]+)", r"\1", label)
    label = re.sub(r"\\[A-Za-z]+", "", label)
    label = label.replace("bbbar", "b bbar").replace("ccbar", "c cbar")
    label = re.sub(r"\s+", " ", label).strip()
    label = label.replace(" ->", "->").replace("-> ", "->")
    label = label.replace(" ,", ",")
    return label


def _row_interval_value(row, prefix):
    value = row.get(prefix)
    if value is None and prefix == "expected_selected_events":
        value = row.get("xgboost_events")
    return value


def event_interval_text(row, prefix, latex=False):
    row = row or {}
    value = _row_interval_value(row, prefix)
    if value is None:
        return "--"

    upper_limit = row.get(f"{prefix}_upper_limit_95cl")
    is_upper_limit = bool(row.get(f"{prefix}_is_upper_limit", False))
    if is_upper_limit and upper_limit is not None:
        if latex:
            return rf"$< {latex_number(upper_limit)}$"
        return f"< {terminal_number(upper_limit)}"

    error_low = row.get(f"{prefix}_error_low_95cl")
    error_high = row.get(f"{prefix}_error_high_95cl")
    if error_low is not None and error_high is not None:
        if latex:
            return rf"${latex_number(value)}^{{+{latex_number(error_high)}}}_{{-{latex_number(error_low)}}}$"
        return f"{terminal_number(value)} -{terminal_number(error_low)} +{terminal_number(error_high)}"

    error = row.get(f"{prefix}_error")
    if error is None and prefix == "expected_selected_events":
        error = row.get("expected_selected_error")
    if error is None and prefix == "xgboost_events":
        error = row.get("xgboost_events_error")
    if error is not None:
        if latex:
            return rf"${latex_number(value)}\pm {latex_number(error)}$"
        return f"{terminal_number(value)} +/- {terminal_number(error)}"

    if latex:
        return f"${latex_number(value)}$"
    return terminal_number(value)


def _terminal_separator(widths):
    return "+" + "+".join("-" * (width + 2) for width in widths) + "+"


def _terminal_table(headers, rows, right_aligned=None):
    right_aligned = set(right_aligned or [])
    text_rows = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in text_rows))
        for index, header in enumerate(headers)
    ]
    separator = _terminal_separator(widths)

    def fmt_row(row):
        cells = []
        for index, value in enumerate(row):
            value = str(value)
            if index in right_aligned:
                cells.append(" " + value.rjust(widths[index]) + " ")
            else:
                cells.append(" " + value.ljust(widths[index]) + " ")
        return "|" + "|".join(cells) + "|"

    lines = [separator, fmt_row(headers), separator]
    lines.extend(fmt_row(row) for row in text_rows)
    lines.append(separator)
    return "\n".join(lines)


def terminal_cutflow_table(rows, luminosity, threshold):
    headers = [
        "Sample",
        "sigma_gen [fb]",
        "N_gen",
        "sigma_input [fb]",
        "N_input",
        "sigma_XGB [fb]",
        "N_XGB",
    ]
    table_rows = [
        [
            terminal_label(row["label"]),
            terminal_number(row["generation_xsec_fb"]),
            terminal_number(row["generation_events"]),
            terminal_number(row["input_xsec_fb"]),
            terminal_number(row["input_events"]),
            terminal_number(row["xgboost_xsec_fb"]),
            event_interval_text(row, "xgboost_events"),
        ]
        for row in rows
    ]
    lines = [
        f"4H sample cutflow / rates (L = {float(luminosity):g} fb^-1, XGBoost threshold = {float(threshold):g})",
        "Generation columns include K-factors and decay BRs; input/XGBoost columns also include b-tag/mistag factors.",
        "N_XGB uncertainties are Poisson 95% CL intervals from selected MC counts.",
        _terminal_table(headers, table_rows, right_aligned=set(range(1, len(headers)))),
    ]
    return "\n".join(lines)


def _score_row_label(row):
    row = row or {}
    label = row.get("label") or row.get("description") or row.get("process_id") or row.get("file") or "sample"
    return terminal_label(label)


def _expected_with_error(row):
    value = row.get("expected_selected_events")
    if value is None:
        value = row.get("xgboost_events")
    if value is None:
        return "--"
    prefix = "expected_selected_events" if row.get("expected_selected_events") is not None else "xgboost_events"
    interval_text = event_interval_text(row, prefix)
    if interval_text != terminal_number(value):
        return interval_text
    error = row.get("expected_selected_error")
    if error is None:
        error = row.get("xgboost_events_error")
    if error is None:
        return terminal_number(value)
    return f"{terminal_number(value)} +/- {terminal_number(error)}"


def terminal_xgboost_mc_table(rows, title="Per-sample XGBoost MC event counts", threshold=None):
    rows = list(rows or [])
    headers = ["Sample", "MC selected / input", "N_XGB (95% CL)"]
    table_rows = [
        [
            _score_row_label(row),
            f"{int(row.get('selected_entries', 0))} / {int(row.get('entries', 0))}",
            _expected_with_error(row),
        ]
        for row in rows
    ]
    if not table_rows:
        table_rows = [["(no samples)", "0 / 0", "--"]]
    lines = [str(title)]
    if threshold is not None:
        lines.append(f"XGBoost threshold = {float(threshold):g}")
    lines.append(
        "MC selected/input entries are raw classifier entries after the threshold; "
        "N_XGB is luminosity-normalized with Poisson 95% CL intervals."
    )
    lines.append(_terminal_table(headers, table_rows, right_aligned={1, 2}))
    return "\n".join(lines)


def sample_latex_label(metadata, is_signal=False):
    metadata = metadata or {}
    if is_signal:
        return r"SM $gg\to hhhh\to 8b$"

    process_id = str(metadata.get("process_id", "")).lower()
    description = str(metadata.get("description", ""))
    if process_id == "gg_to_8b":
        return r"$gg\to 8b$"
    if process_id == "pp_to_z_6b_z_to_bb":
        return r"$pp\to Z+6b,\ Z\to b\bar{b}$"
    if process_id == "gg_to_6b_2j":
        return r"$gg\to 6b+2j$"
    if process_id == "gg_to_6b_2c":
        return r"$gg\to 6b+c\bar{c}$"
    if process_id == "gg_to_4b_4c":
        return r"$gg\to 4b+4c$"
    if process_id == "gg_to_4b_2c_2j":
        return r"$gg\to 4b+c\bar{c}+2j$"
    if description:
        return description
    if process_id:
        return process_id.replace("_", r"\_")
    return "background"
