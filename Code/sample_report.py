"""Utilities for 4H sample-rate reports.

The helpers in this module are intentionally independent of ROOT and XGBoost so
the normalization conventions can be tested without the full HEP runtime.
"""

from __future__ import annotations

import html
import math
import re


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


def safe_feature_filename(name):
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
    return stem or "feature"


def html_escape(value):
    return html.escape(str(value), quote=True)


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
            terminal_number(row["xgboost_events"]),
        ]
        for row in rows
    ]
    lines = [
        f"4H sample cutflow / rates (L = {float(luminosity):g} fb^-1, XGBoost threshold = {float(threshold):g})",
        "Generation columns include K-factors and decay BRs; input/XGBoost columns also include b-tag/mistag factors.",
        _terminal_table(headers, table_rows, right_aligned=set(range(1, len(headers)))),
    ]
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
