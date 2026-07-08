"""Validate HEFT hhh+bb against hhh+g forced g -> b bbar at LHE 8b level."""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path
import subprocess

from .herwig_cards import DEFAULT_HERWIG_PDF_NAME, PROCESS_CONFIGS, higgs_decay_lhewriter_card, stage1_lhewriter_card
from .lhe_validation import normalize_lhe_file_process_ids, parse_lhe_events
from .lhe_weights import apply_weights, verify_weighted_lhe
from .validation_hbb import (
    HIGGS_PT_BINS,
    SOURCE_MATCH_MAX_SCORE,
    _combined_four_vector,
    _cos_theta_star,
    _delta_r,
    _event_weight,
    _four_momentum_match_score,
    _four_vector,
    _has_ancestor_pid,
    _init_xsec,
    _invariant_mass,
    _pt,
    _pt_from_four_vector,
    _read_lhe_text,
    _run_herwig,
    _validate_input_events,
    _write_text,
    weight_check_report_lines,
)


CODE_DIR = Path(__file__).resolve().parents[1] / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from sample_report import (  # noqa: E402
    safe_feature_filename,
    sample_style,
    terminal_number,
    write_observable_shape_plot,
    write_report_index,
)


HHHBB_OBSERVABLE_ORDER = [
    "b_pt_all",
    "b1_pt",
    "b2_pt",
    "b3_pt",
    "b4_pt",
    "b5_pt",
    "b6_pt",
    "b7_pt",
    "b8_pt",
    "dr_bb_all",
    "dr_associated_bb",
    "dr_higgs_bb",
    "dr_associated_higgs_cross_bb",
    "dr_inter_higgs_cross_bb",
    "dr_min_bb",
    "m_bb_all",
    "m_associated_bb",
    "m_higgs_bb",
    "m_associated_higgs_cross_bb",
    "m_inter_higgs_cross_bb",
    "h_pt",
    "h_eta",
    "dr_higgs_bb_hpt_lt150",
    "dr_higgs_bb_hpt_150_300",
    "dr_higgs_bb_hpt_ge300",
    "cos_theta_star_higgs_b",
    "m_8b",
    "ht_b",
]


HHHBB_OBSERVABLE_TITLES = {
    "b_pt_all": "inclusive_b_pt",
    "b1_pt": "b1_pt",
    "b2_pt": "b2_pt",
    "b3_pt": "b3_pt",
    "b4_pt": "b4_pt",
    "b5_pt": "b5_pt",
    "b6_pt": "b6_pt",
    "b7_pt": "b7_pt",
    "b8_pt": "b8_pt",
    "dr_bb_all": "all_pair_deltaR_bb",
    "dr_associated_bb": "associated_pair_deltaR_bb",
    "dr_higgs_bb": "higgs_decay_deltaR_bb",
    "dr_associated_higgs_cross_bb": "associated_higgs_cross_deltaR_bb",
    "dr_inter_higgs_cross_bb": "inter_higgs_cross_deltaR_bb",
    "dr_min_bb": "min_deltaR_bb",
    "m_bb_all": "all_pair_m_bb",
    "m_associated_bb": "associated_pair_m_bb",
    "m_higgs_bb": "higgs_decay_m_bb",
    "m_associated_higgs_cross_bb": "associated_higgs_cross_m_bb",
    "m_inter_higgs_cross_bb": "inter_higgs_cross_m_bb",
    "h_pt": "source_h_pt",
    "h_eta": "source_h_eta",
    "dr_higgs_bb_hpt_lt150": "higgs_decay_deltaR_bb_hpt_lt150",
    "dr_higgs_bb_hpt_150_300": "higgs_decay_deltaR_bb_hpt_150_300",
    "dr_higgs_bb_hpt_ge300": "higgs_decay_deltaR_bb_hpt_ge300",
    "cos_theta_star_higgs_b": "cos_theta_star_higgs_b",
    "m_8b": "m_8b",
    "ht_b": "ht_b",
}


HHHBB_SPLIT_LABEL = r"$gg \rightarrow hhhg,\ g\rightarrow b\bar{b}$ (forced split)"
HHHBB_DIRECT_LABEL = r"$gg \rightarrow h h h b\bar{b}$ (direct)"
HHHBB_REPORT_TITLE = r"4H HEFT LHE 8b Validation Observables ($m_t \rightarrow \infty$)"
HHHBB_PLOT_TITLE = r"HEFT validation, $m_t \rightarrow \infty$"
HHHBB_SHAPE_MIN_BINS = 5
HHHBB_SHAPE_MAX_BINS = 35
HHHBB_SHAPE_ENTRIES_PER_BIN = 25.0


@dataclass
class HHHBBValidationRunConfig(object):
    split_input_lhe: Path
    direct_input_lhe: Path
    workdir: Path
    events: int
    probe_trials: int = 0
    run_name: str = "hhhbb_validation"
    herwig: str = "Herwig"
    seed_stage1: int = 31122002
    seed_split_decay: int = 44071981
    seed_direct_decay: int = 44071982
    pdf_name: str = DEFAULT_HERWIG_PDF_NAME
    input_xsec_error: float = None
    allow_zero_probe_successes: bool = False
    allow_input_oversampling: bool = False
    overwrite: bool = False
    dry_run: bool = False


def _default_runner(command, cwd):
    subprocess.run(command, cwd=str(cwd), check=True)


def _eta(particle):
    momentum = math.sqrt(particle.px * particle.px + particle.py * particle.py + particle.pz * particle.pz)
    denominator = momentum - particle.pz
    numerator = momentum + particle.pz
    if denominator <= 0.0:
        return math.copysign(float("inf"), particle.pz if particle.pz != 0.0 else 1.0)
    if numerator <= 0.0:
        return math.copysign(float("inf"), particle.pz if particle.pz != 0.0 else -1.0)
    return 0.5 * math.log(numerator / denominator)


def _empty_observables():
    return {name: {"values": [], "weights": []} for name in HHHBB_OBSERVABLE_ORDER}


def _append(observables, name, value, weight):
    if value is None or not math.isfinite(float(value)):
        return
    observables[name]["values"].append(float(value))
    observables[name]["weights"].append(float(weight))


def _source_associated_b_quarks(source_event):
    return [
        particle
        for particle in source_event.particles
        if particle.status == 1 and abs(particle.pid) == 5 and not _has_ancestor_pid(source_event, particle, 25)
    ]


def _source_higgs_bosons(source_event):
    return [particle for particle in source_event.particles if particle.status == 1 and abs(particle.pid) == 25]


def _match_source_indices_with_score(source_particles, final_particles, candidate_indices=None):
    if candidate_indices is None:
        candidate_indices = range(len(final_particles))
    candidate_indices = list(candidate_indices)
    if len(source_particles) > len(candidate_indices):
        return None

    best_score = None
    best_indices = None
    for indices in permutations(candidate_indices, len(source_particles)):
        score = 0.0
        valid = True
        for source_particle, candidate_index in zip(source_particles, indices):
            candidate = final_particles[candidate_index]
            if source_particle.pid != candidate.pid:
                valid = False
                break
            score += _four_momentum_match_score(source_particle, candidate)
        if valid and (best_score is None or score < best_score):
            best_score = score
            best_indices = set(indices)
    if best_indices is None:
        return None
    return best_indices, best_score


def _find_source_context(source_events, final_b_quarks, preferred_index=None, max_score=SOURCE_MATCH_MAX_SCORE):
    if not source_events:
        return None, None
    source_indices = []
    if preferred_index is not None and preferred_index < len(source_events):
        source_indices.append(preferred_index)
    source_indices.extend(index for index in range(len(source_events)) if index not in source_indices)
    fallback_event = source_events[preferred_index] if preferred_index is not None and preferred_index < len(source_events) else source_events[0]
    for source_index in source_indices:
        associated = _source_associated_b_quarks(source_events[source_index])
        if len(associated) != 2:
            continue
        match = _match_source_indices_with_score(associated, final_b_quarks)
        if match is None:
            continue
        indices, score = match
        if score <= max_score:
            return source_events[source_index], indices
    return fallback_event, None


def _ancestor_higgs_index(event, particle):
    seen = set()
    stack = [index for index in (particle.mother1, particle.mother2) if index > 0]
    while stack:
        mother_index = stack.pop()
        if mother_index in seen or mother_index < 1 or mother_index > len(event.particles):
            continue
        seen.add(mother_index)
        mother = event.particles[mother_index - 1]
        if abs(mother.pid) == 25:
            return mother_index
        stack.extend(index for index in (mother.mother1, mother.mother2) if index > 0)
    return None


def _higgs_pairs_from_ancestry(event, b_quarks, candidate_indices):
    groups = {}
    for index in candidate_indices:
        mother_index = _ancestor_higgs_index(event, b_quarks[index])
        if mother_index is not None:
            groups.setdefault(mother_index, []).append(index)
    pairs = [frozenset(indices) for indices in groups.values() if len(indices) == 2]
    if len(pairs) != 3:
        return None
    return pairs


def _pair_partitions(indices, pair_count):
    indices = tuple(indices)
    if pair_count == 0:
        yield []
        return
    if len(indices) < 2 * pair_count:
        return
    first = indices[0]
    rest = indices[1:]
    for second in rest:
        remaining = tuple(index for index in rest if index != second)
        for tail in _pair_partitions(remaining, pair_count - 1):
            yield [frozenset((first, second))] + tail


def _combined_pair_score(pair, source_higgs, b_quarks):
    combined = _combined_four_vector([b_quarks[index] for index in pair])
    energy, px, py, pz = combined
    scale = max(1.0, abs(source_higgs.energy), abs(energy))
    return (
        ((source_higgs.energy - energy) / scale) ** 2
        + ((source_higgs.px - px) / scale) ** 2
        + ((source_higgs.py - py) / scale) ** 2
        + ((source_higgs.pz - pz) / scale) ** 2
    )


def _higgs_pairs_from_source(source_higgses, b_quarks, candidate_indices, max_score=SOURCE_MATCH_MAX_SCORE):
    if len(source_higgses) != 3 or len(candidate_indices) < 6:
        return None

    best_score = None
    best_pairs = None
    for pairs in _pair_partitions(candidate_indices, 3):
        for ordered_higgses in permutations(source_higgses, 3):
            score = sum(_combined_pair_score(pair, higgs, b_quarks) for pair, higgs in zip(pairs, ordered_higgses))
            if best_score is None or score < best_score:
                best_score = score
                best_pairs = list(pairs)
    if best_pairs is None or best_score is None or best_score > max_score:
        return None
    return best_pairs


def _higgs_pairs_by_mass(b_quarks, candidate_indices, higgs_mass=125.0):
    best_score = None
    best_pairs = None
    for pairs in _pair_partitions(candidate_indices, 3):
        score = sum(abs(_invariant_mass([b_quarks[index] for index in pair]) - higgs_mass) for pair in pairs)
        if best_score is None or score < best_score:
            best_score = score
            best_pairs = list(pairs)
    return best_pairs


def _higgs_pt_bin_observable(higgs_pt):
    for observable, lower, upper in HIGGS_PT_BINS:
        if lower is not None and higgs_pt < lower:
            continue
        if upper is not None and higgs_pt >= upper:
            continue
        return observable
    return None


def _append_higgs_observables(observables, b_quarks, higgs_pairs, source_higgses, weight):
    used_sources = set()
    for pair in higgs_pairs:
        pair_particles = [b_quarks[index] for index in pair]
        pair_vector = _combined_four_vector(pair_particles)
        source_higgs = None
        if source_higgses:
            best = None
            for source_index, candidate in enumerate(source_higgses):
                if source_index in used_sources:
                    continue
                score = _combined_pair_score(pair, candidate, b_quarks)
                if best is None or score < best[0]:
                    best = score, source_index, candidate
            if best is not None:
                used_sources.add(best[1])
                source_higgs = best[2]
                _append(observables, "h_pt", _pt(source_higgs), weight)
                _append(observables, "h_eta", _eta(source_higgs), weight)

        parent_vector = _four_vector(source_higgs) if source_higgs is not None else pair_vector
        higgs_pt = _pt_from_four_vector(parent_vector)
        delta_r = _delta_r(pair_particles[0], pair_particles[1])
        hpt_observable = _higgs_pt_bin_observable(higgs_pt)
        if hpt_observable is not None:
            _append(observables, hpt_observable, delta_r, weight)
        decay_b = next((particle for particle in pair_particles if particle.pid == 5), pair_particles[0])
        cos_theta_star = _cos_theta_star(decay_b, parent_vector)
        _append(observables, "cos_theta_star_higgs_b", cos_theta_star, weight)


def extract_lhe_8b_sample(path, label=None, source_lhe=None):
    """Extract weighted final-state 8b observables from an LHE file."""

    path = Path(path)
    events = parse_lhe_events(_read_lhe_text(path))
    source_lhe = Path(source_lhe) if source_lhe is not None else None
    source_events = parse_lhe_events(_read_lhe_text(source_lhe)) if source_lhe is not None else []
    observables = _empty_observables()
    accepted = 0
    skipped = 0
    weighted_event_sum = 0.0
    pair_classification = {
        "associated_source_match": 0,
        "associated_from_higgs_remainder": 0,
        "associated_unmatched": 0,
        "higgs_ancestry_pairs": 0,
        "source_higgs_match_pairs": 0,
        "higgs_mass_fallback_pairs": 0,
    }

    for event_index, event in enumerate(events):
        weight = _event_weight(event)
        b_quarks = [particle for particle in event.particles if particle.status == 1 and abs(particle.pid) == 5]
        if len(b_quarks) != 8:
            skipped += 1
            continue

        source_event, associated_indices = _find_source_context(source_events, b_quarks, preferred_index=event_index)
        if associated_indices is not None:
            pair_classification["associated_source_match"] += 1
        source_higgses = _source_higgs_bosons(source_event) if source_event is not None else []
        candidate_higgs_indices = set(range(len(b_quarks))) - (associated_indices or set())

        higgs_pairs = _higgs_pairs_from_ancestry(event, b_quarks, candidate_higgs_indices)
        if higgs_pairs is not None:
            pair_classification["higgs_ancestry_pairs"] += len(higgs_pairs)
        else:
            higgs_pairs = _higgs_pairs_from_source(source_higgses, b_quarks, candidate_higgs_indices)
            if higgs_pairs is not None:
                pair_classification["source_higgs_match_pairs"] += len(higgs_pairs)
            else:
                higgs_pairs = _higgs_pairs_by_mass(b_quarks, candidate_higgs_indices)
                pair_classification["higgs_mass_fallback_pairs"] += len(higgs_pairs or [])

        if associated_indices is None and higgs_pairs:
            higgs_indices = set().union(*higgs_pairs)
            associated_indices = set(range(len(b_quarks))) - higgs_indices
            if len(associated_indices) == 2:
                pair_classification["associated_from_higgs_remainder"] += 1
            else:
                associated_indices = set()
                pair_classification["associated_unmatched"] += 1

        if len(associated_indices or set()) != 2 or len(higgs_pairs or []) != 3:
            skipped += 1
            continue

        accepted += 1
        weighted_event_sum += weight
        ranked = sorted(b_quarks, key=_pt, reverse=True)
        higgs_pair_sets = {frozenset(pair) for pair in higgs_pairs}

        for b_quark in b_quarks:
            _append(observables, "b_pt_all", _pt(b_quark), weight)
        for rank, b_quark in enumerate(ranked, start=1):
            _append(observables, "b%d_pt" % rank, _pt(b_quark), weight)

        pair_delta_rs = []
        for first_index, second_index in combinations(range(len(b_quarks)), 2):
            first = b_quarks[first_index]
            second = b_quarks[second_index]
            pair = frozenset((first_index, second_index))
            delta_r = _delta_r(first, second)
            mass = _invariant_mass([first, second])
            pair_delta_rs.append(delta_r)
            _append(observables, "dr_bb_all", delta_r, weight)
            _append(observables, "m_bb_all", mass, weight)
            if pair == frozenset(associated_indices):
                _append(observables, "dr_associated_bb", delta_r, weight)
                _append(observables, "m_associated_bb", mass, weight)
            elif pair in higgs_pair_sets:
                _append(observables, "dr_higgs_bb", delta_r, weight)
                _append(observables, "m_higgs_bb", mass, weight)
            elif first_index in associated_indices or second_index in associated_indices:
                _append(observables, "dr_associated_higgs_cross_bb", delta_r, weight)
                _append(observables, "m_associated_higgs_cross_bb", mass, weight)
            else:
                _append(observables, "dr_inter_higgs_cross_bb", delta_r, weight)
                _append(observables, "m_inter_higgs_cross_bb", mass, weight)

        if pair_delta_rs:
            _append(observables, "dr_min_bb", min(pair_delta_rs), weight)
        _append(observables, "m_8b", _invariant_mass(b_quarks), weight)
        _append(observables, "ht_b", sum(_pt(b_quark) for b_quark in b_quarks), weight)
        _append_higgs_observables(observables, b_quarks, higgs_pairs, source_higgses, weight)

    xsec_pb, xsec_error_pb = _init_xsec(path)
    label = label or path.stem
    return {
        "label": label,
        "file": str(path),
        "observables": observables,
        "summary": {
            "label": label,
            "file": str(path),
            "xsec_pb": xsec_pb,
            "xsec_error_pb": xsec_error_pb,
            "event_count": int(len(events)),
            "accepted_8b_events": int(accepted),
            "skipped_events": int(skipped),
            "weighted_event_sum": float(weighted_event_sum),
            "source_file": "" if source_lhe is None else str(source_lhe),
            "source_event_count": int(len(source_events)),
            "pair_classification": pair_classification,
        },
    }


def _plain_table(headers, rows, right_aligned=None):
    right_aligned = set(right_aligned or [])
    text_rows = [[str(value) for value in row] for row in rows]
    widths = [max(len(str(header)), *(len(row[index]) for row in text_rows)) for index, header in enumerate(headers)]

    def separator():
        return "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def row_text(row):
        cells = []
        for index, value in enumerate(row):
            value = str(value)
            if index in right_aligned:
                cells.append(" " + value.rjust(widths[index]) + " ")
            else:
                cells.append(" " + value.ljust(widths[index]) + " ")
        return "|" + "|".join(cells) + "|"

    lines = [separator(), row_text(headers), separator()]
    lines.extend(row_text(row) for row in text_rows)
    lines.append(separator())
    return "\n".join(lines)


def _write_validation_table(path, samples, correction_summary=None):
    headers = [
        "Sample",
        "sigma_LHE [pb]",
        "dsigma [pb]",
        "events",
        "accepted 8b",
        "skipped",
        "sum w(8b)",
        "assoc source",
        "assoc remainder",
        "H ancestry pairs",
        "source H pairs",
        "mH fallback pairs",
    ]
    rows = []
    for sample in samples:
        summary = sample["summary"]
        pair_classification = summary.get("pair_classification", {})
        rows.append(
            [
                summary["label"],
                terminal_number(summary["xsec_pb"]) if summary["xsec_pb"] is not None else "--",
                terminal_number(summary["xsec_error_pb"]) if summary["xsec_error_pb"] is not None else "--",
                int(summary["event_count"]),
                int(summary["accepted_8b_events"]),
                int(summary["skipped_events"]),
                terminal_number(summary["weighted_event_sum"]),
                int(pair_classification.get("associated_source_match", 0)),
                int(pair_classification.get("associated_from_higgs_remainder", 0)),
                int(pair_classification.get("higgs_ancestry_pairs", 0)),
                int(pair_classification.get("source_higgs_match_pairs", 0)),
                int(pair_classification.get("higgs_mass_fallback_pairs", 0)),
            ]
        )
    lines = [
        "4H HEFT LHE 8b validation summary",
        "Higgs decays are treated as BR(h->bb)=1 for this validation.",
        _plain_table(headers, rows, right_aligned=set(range(1, len(headers)))),
    ]
    if correction_summary:
        lines.extend(["", "Forced-splitting correction:", json.dumps(correction_summary, indent=2, sort_keys=True)])
    Path(path).write_text("\n".join(lines) + "\n")


def write_lhe_8b_validation_report_from_samples(
    samples,
    output_dir,
    correction_summary=None,
    title=HHHBB_REPORT_TITLE,
    report_line=(
        "Final-state 8b LHE comparison in the HEFT/infinite-top-mass limit; "
        "BR(h->bb)=1; validation outputs only."
    ),
):
    """Write a validation report from extracted LHE 8b samples."""

    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_rows = []
    for observable in HHHBB_OBSERVABLE_ORDER:
        plot_path = plots_dir / ("%s.png" % safe_feature_filename(observable))
        plot_samples = []
        entries = 0
        for index, sample in enumerate(samples):
            values = sample["observables"][observable]["values"]
            weights = sample["observables"][observable]["weights"]
            entries += len(values)
            plot_samples.append(
                {
                    "label": sample["label"],
                    "values": values,
                    "weights": weights,
                    "style": sample_style(index),
                }
            )
        write_observable_shape_plot(
            plot_path,
            observable,
            plot_samples,
            min_bins=HHHBB_SHAPE_MIN_BINS,
            max_bins=HHHBB_SHAPE_MAX_BINS,
            entries_per_bin=HHHBB_SHAPE_ENTRIES_PER_BIN,
            title=HHHBB_PLOT_TITLE,
        )
        plot_rows.append(
            {
                "feature": HHHBB_OBSERVABLE_TITLES.get(observable, observable),
                "path": str(plot_path),
                "entries": int(entries),
            }
        )

    table_path = output_dir / "validation_table.txt"
    index_path = output_dir / "index.html"
    metadata_path = output_dir / "report_metadata.json"
    _write_validation_table(table_path, samples, correction_summary=correction_summary)

    metadata = {
        "validation_only": True,
        "title": title,
        "table": str(table_path),
        "index": str(index_path),
        "metadata": str(metadata_path),
        "plots": plot_rows,
        "samples": [sample["summary"] for sample in samples],
        "normalisation": {
            "histograms": "weighted LHE events, normalized per observable and sample",
            "shape_plot_binning": {
                "min_bins": HHHBB_SHAPE_MIN_BINS,
                "max_bins": HHHBB_SHAPE_MAX_BINS,
                "entries_per_bin": HHHBB_SHAPE_ENTRIES_PER_BIN,
            },
            "higgs_branching_ratio": "BR(h->bb)=1 validation",
            "pair_assignment": (
                "associated/direct bb pairs use source pre-decay LHE four-momentum matching; "
                "Higgs-decay bb pairs use Higgs ancestry, then source-Higgs four-vector matching, "
                "then a closest-mH fallback"
            ),
        },
        "correction_summary": correction_summary,
        "report_line": report_line,
    }
    write_report_index(
        index_path,
        plot_rows,
        table_path,
        metadata,
        title=title,
        table_label="Validation table",
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def write_lhe_8b_validation_report(
    split_lhe,
    direct_lhe,
    output_dir,
    split_label=HHHBB_SPLIT_LABEL,
    direct_label=HHHBB_DIRECT_LABEL,
    split_source_lhe=None,
    direct_source_lhe=None,
    correction_summary=None,
):
    """Write an LHE 8b comparison report with the standard sample webpage."""

    samples = [
        extract_lhe_8b_sample(split_lhe, label=split_label, source_lhe=split_source_lhe),
        extract_lhe_8b_sample(direct_lhe, label=direct_label, source_lhe=direct_source_lhe),
    ]
    return write_lhe_8b_validation_report_from_samples(
        samples=samples,
        output_dir=output_dir,
        correction_summary=correction_summary,
    )


def run_hhhbb_validation_chain(config, runner=None):
    """Run split/direct HEFT hhhbb Herwig validation and write the final 8b LHE report."""

    if runner is None:
        runner = _default_runner

    split_input_lhe = Path(config.split_input_lhe).resolve()
    direct_input_lhe = Path(config.direct_input_lhe).resolve()
    split_input_event_count = _validate_input_events(
        split_input_lhe,
        config.events,
        config.allow_input_oversampling,
    )
    direct_input_event_count = _validate_input_events(
        direct_input_lhe,
        config.events,
        config.allow_input_oversampling,
    )

    workdir = Path(config.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    run_name = config.run_name
    split_stage1_name = "%s_split_stage1" % run_name
    split_decay_name = "%s_split_final8b" % run_name
    direct_decay_name = "%s_direct_final8b" % run_name

    split_stage1_card = workdir / ("%s.in" % split_stage1_name)
    split_stage1_run = workdir / ("%s.run" % split_stage1_name)
    split_stage1_lhe = workdir / ("%s.lhe" % split_stage1_name)
    split_correction_file = workdir / ("%s.force_split.weights" % split_stage1_name)
    split_weighted_lhe = workdir / ("%s.weighted.lhe" % split_stage1_name)

    split_decay_card = workdir / ("%s.in" % split_decay_name)
    split_decay_run = workdir / ("%s.run" % split_decay_name)
    split_final_lhe = workdir / ("%s.lhe" % split_decay_name)

    direct_decay_card = workdir / ("%s.in" % direct_decay_name)
    direct_decay_run = workdir / ("%s.run" % direct_decay_name)
    direct_final_lhe = workdir / ("%s.lhe" % direct_decay_name)
    report_dir = workdir / "report"
    summary_file = workdir / ("%s_summary.json" % run_name)

    split_stage1_text = stage1_lhewriter_card(
        PROCESS_CONFIGS["gg_hhhg"],
        input_lhe=split_input_lhe,
        output_prefix=split_stage1_name,
        events=config.events,
        seed=config.seed_stage1,
        probe_trials=config.probe_trials,
        correction_file=split_correction_file.name,
        pdf_name=config.pdf_name,
    )
    _write_text(split_stage1_card, split_stage1_text, config.overwrite)

    commands = []
    commands.append(_run_herwig(config.herwig, "read", split_stage1_card.name, workdir, runner, config.dry_run))
    commands.append(_run_herwig(config.herwig, "run", split_stage1_run.name, workdir, runner, config.dry_run))

    split_decay_input = split_stage1_lhe
    normalization_message = None
    weight_check = None
    if not config.dry_run:
        if not split_stage1_lhe.exists():
            raise FileNotFoundError("Split Stage-1 LHE was not produced: %s" % split_stage1_lhe)
        _, normalization_message = normalize_lhe_file_process_ids(split_stage1_lhe)
        if config.probe_trials > 0:
            if not split_correction_file.exists():
                raise FileNotFoundError("ProbeTrials was nonzero but no correction sidecar exists: %s" % split_correction_file)
            apply_weights(
                split_stage1_lhe,
                split_correction_file,
                split_weighted_lhe,
                input_xsec_error=config.input_xsec_error,
            )
            weight_check = verify_weighted_lhe(split_stage1_lhe, split_correction_file, split_weighted_lhe)
            if not weight_check["ok"]:
                raise RuntimeError("Weighted LHE verification failed: %s" % weight_check)
            if weight_check["zero_success_rows"] and not config.allow_zero_probe_successes:
                raise RuntimeError(
                    "Correction sidecar has %d rows with probe_successes = 0; "
                    "increase --probe-trials or pass --allow-zero-probe-successes for a diagnostic run"
                    % weight_check["zero_success_rows"]
                )
            split_decay_input = split_weighted_lhe
    elif config.probe_trials > 0:
        split_decay_input = split_weighted_lhe

    split_decay_text = higgs_decay_lhewriter_card(
        input_lhe=split_decay_input.name,
        output_prefix=split_decay_name,
        events=config.events,
        seed=config.seed_split_decay,
        pdf_name=config.pdf_name,
    )
    direct_decay_text = higgs_decay_lhewriter_card(
        input_lhe=direct_input_lhe,
        output_prefix=direct_decay_name,
        events=config.events,
        seed=config.seed_direct_decay,
        pdf_name=config.pdf_name,
    )
    _write_text(split_decay_card, split_decay_text, config.overwrite)
    _write_text(direct_decay_card, direct_decay_text, config.overwrite)

    commands.append(_run_herwig(config.herwig, "read", split_decay_card.name, workdir, runner, config.dry_run))
    commands.append(_run_herwig(config.herwig, "run", split_decay_run.name, workdir, runner, config.dry_run))
    commands.append(_run_herwig(config.herwig, "read", direct_decay_card.name, workdir, runner, config.dry_run))
    commands.append(_run_herwig(config.herwig, "run", direct_decay_run.name, workdir, runner, config.dry_run))

    report_metadata = None
    if not config.dry_run:
        for output in (split_final_lhe, direct_final_lhe):
            if not output.exists():
                raise FileNotFoundError("Final validation LHE was not produced: %s" % output)
            normalize_lhe_file_process_ids(output)
        report_metadata = write_lhe_8b_validation_report(
            split_lhe=split_final_lhe,
            direct_lhe=direct_final_lhe,
            output_dir=report_dir,
            split_source_lhe=split_decay_input,
            direct_source_lhe=direct_input_lhe,
            correction_summary=weight_check,
        )

    summary = {
        "run_name": run_name,
        "split_input_lhe": str(split_input_lhe),
        "direct_input_lhe": str(direct_input_lhe),
        "split_input_event_count": int(split_input_event_count),
        "direct_input_event_count": int(direct_input_event_count),
        "events": int(config.events),
        "probe_trials": int(config.probe_trials),
        "pdf_name": str(config.pdf_name),
        "workdir": str(workdir),
        "split_stage1_card": str(split_stage1_card),
        "split_stage1_run": str(split_stage1_run),
        "split_stage1_lhe": str(split_stage1_lhe),
        "split_weighted_lhe": str(split_weighted_lhe) if config.probe_trials > 0 else "",
        "split_correction_file": str(split_correction_file),
        "split_decay_card": str(split_decay_card),
        "split_decay_source_lhe": str(split_decay_input),
        "split_decay_run": str(split_decay_run),
        "split_final_lhe": str(split_final_lhe),
        "direct_decay_source_lhe": str(direct_input_lhe),
        "direct_decay_card": str(direct_decay_card),
        "direct_decay_run": str(direct_decay_run),
        "direct_final_lhe": str(direct_final_lhe),
        "normalization_message": normalization_message,
        "weight_check": weight_check,
        "report": None if report_metadata is None else str(report_metadata["index"]),
        "commands": [" ".join(command) for command in commands],
        "dry_run": bool(config.dry_run),
        "summary": str(summary_file),
    }
    _write_text(summary_file, json.dumps(summary, indent=2, sort_keys=True) + "\n", True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="run split/direct HEFT hhhbb validation and write a report")
    run.add_argument("--split-lhe", type=Path, required=True)
    run.add_argument("--direct-lhe", type=Path, required=True)
    run.add_argument("--workdir", type=Path, required=True)
    run.add_argument("--events", type=int, required=True)
    run.add_argument("--probe-trials", type=int, default=0)
    run.add_argument("--run-name", default="hhhbb_validation")
    run.add_argument("--herwig", default="Herwig")
    run.add_argument("--seed-stage1", type=int, default=31122002)
    run.add_argument("--seed-split-decay", type=int, default=44071981)
    run.add_argument("--seed-direct-decay", type=int, default=44071982)
    run.add_argument("--pdf-name", default=DEFAULT_HERWIG_PDF_NAME)
    run.add_argument("--input-xsec-error", type=float)
    run.add_argument("--allow-zero-probe-successes", action="store_true")
    run.add_argument("--allow-input-oversampling", action="store_true")
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    compare = subparsers.add_parser("compare", help="write a validation report from existing final 8b LHE files")
    compare.add_argument("--split-lhe", type=Path, required=True)
    compare.add_argument("--direct-lhe", type=Path, required=True)
    compare.add_argument("--split-source-lhe", type=Path)
    compare.add_argument("--direct-source-lhe", type=Path)
    compare.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "run":
        summary = run_hhhbb_validation_chain(
            HHHBBValidationRunConfig(
                split_input_lhe=args.split_lhe,
                direct_input_lhe=args.direct_lhe,
                workdir=args.workdir,
                events=args.events,
                probe_trials=args.probe_trials,
                run_name=args.run_name,
                herwig=args.herwig,
                seed_stage1=args.seed_stage1,
                seed_split_decay=args.seed_split_decay,
                seed_direct_decay=args.seed_direct_decay,
                pdf_name=args.pdf_name,
                input_xsec_error=args.input_xsec_error,
                allow_zero_probe_successes=args.allow_zero_probe_successes,
                allow_input_oversampling=args.allow_input_oversampling,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
        )
        print("Validation summary:", summary["summary"])
        if summary["report"]:
            print("Validation report:", summary["report"])
        for line in weight_check_report_lines(summary):
            print(line)
        return 0
    if args.command == "compare":
        metadata = write_lhe_8b_validation_report(
            split_lhe=args.split_lhe,
            direct_lhe=args.direct_lhe,
            output_dir=args.output_dir,
            split_source_lhe=args.split_source_lhe,
            direct_source_lhe=args.direct_source_lhe,
        )
        print("Validation report:", metadata["index"])
        return 0
    parser.error("choose a command: run or compare")


if __name__ == "__main__":
    main()
