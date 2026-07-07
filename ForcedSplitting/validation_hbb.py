"""Validate forced g -> b bbar splitting against direct gg -> h b bbar LHEs."""

import argparse
import csv
import gzip
import json
import math
import shlex
import sys
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path
import subprocess

from .herwig_cards import DEFAULT_HERWIG_PDF_NAME, PROCESS_CONFIGS, higgs_decay_lhewriter_card, stage1_lhewriter_card
from .lhe_validation import normalize_lhe_file_process_ids, parse_lhe_events
from .lhe_weights import apply_weights, verify_weighted_lhe
from .run_chain import count_lhe_events


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


OBSERVABLE_ORDER = [
    "b_pt_all",
    "b1_pt",
    "b2_pt",
    "b3_pt",
    "b4_pt",
    "dr_bb_all",
    "dr_associated_bb",
    "dr_higgs_bb",
    "dr_higgs_bb_hpt_lt150",
    "dr_higgs_bb_hpt_150_300",
    "dr_higgs_bb_hpt_ge300",
    "dr_cross_bb",
    "dr_min_bb",
    "dr_associated_bb_0p3_0p8",
    "dr_associated_bb_0p8_1p5",
    "dr_associated_bb_ge1p5",
    "m_bb_all",
    "m_associated_bb",
    "m_higgs_bb",
    "m_4b",
    "h_pt",
    "h_eta",
    "cos_theta_star_higgs_b",
]

OBSERVABLE_TITLES = {
    "b_pt_all": "inclusive_b_pt",
    "b1_pt": "b1_pt",
    "b2_pt": "b2_pt",
    "b3_pt": "b3_pt",
    "b4_pt": "b4_pt",
    "dr_bb_all": "all_pair_deltaR_bb",
    "dr_associated_bb": "associated_pair_deltaR_bb",
    "dr_higgs_bb": "higgs_decay_deltaR_bb",
    "dr_higgs_bb_hpt_lt150": "higgs_decay_deltaR_bb_hpt_lt150",
    "dr_higgs_bb_hpt_150_300": "higgs_decay_deltaR_bb_hpt_150_300",
    "dr_higgs_bb_hpt_ge300": "higgs_decay_deltaR_bb_hpt_ge300",
    "dr_cross_bb": "cross_pair_deltaR_bb",
    "dr_min_bb": "min_deltaR_bb",
    "dr_associated_bb_0p3_0p8": "associated_pair_deltaR_bb_0p3_0p8",
    "dr_associated_bb_0p8_1p5": "associated_pair_deltaR_bb_0p8_1p5",
    "dr_associated_bb_ge1p5": "associated_pair_deltaR_bb_ge1p5",
    "m_bb_all": "all_pair_m_bb",
    "m_associated_bb": "associated_pair_m_bb",
    "m_higgs_bb": "higgs_decay_m_bb",
    "m_4b": "m_4b",
    "h_pt": "source_h_pt",
    "h_eta": "source_h_eta",
    "cos_theta_star_higgs_b": "cos_theta_star_higgs_b",
}


SOURCE_MATCH_MAX_SCORE = 1.0e-3

ASSOCIATED_DELTA_R_BINS = [
    ("0.3_0.8", "dr_associated_bb_0p3_0p8", 0.3, 0.8),
    ("0.8_1.5", "dr_associated_bb_0p8_1p5", 0.8, 1.5),
    ("ge1.5", "dr_associated_bb_ge1p5", 1.5, None),
]

HIGGS_PT_BINS = [
    ("dr_higgs_bb_hpt_lt150", None, 150.0),
    ("dr_higgs_bb_hpt_150_300", 150.0, 300.0),
    ("dr_higgs_bb_hpt_ge300", 300.0, None),
]


SHOWER_VARIATIONS = {
    "baseline": {
        "description": "nominal Herwig forced-splitting setup",
        "settings": [],
    },
    "renorm_0p5": {
        "description": "ShowerHandler renormalization scale factor 0.5",
        "settings": ["set ShowerHandler:RenormalizationScaleFactor 0.5"],
    },
    "renorm_2p0": {
        "description": "ShowerHandler renormalization scale factor 2.0",
        "settings": ["set ShowerHandler:RenormalizationScaleFactor 2.0"],
    },
    "hard_0p5": {
        "description": "ShowerHandler hard scale factor 0.5",
        "settings": ["set ShowerHandler:HardScaleFactor 0.5"],
    },
    "hard_2p0": {
        "description": "ShowerHandler hard scale factor 2.0",
        "settings": ["set ShowerHandler:HardScaleFactor 2.0"],
    },
    "gqq_scale_pt": {
        "description": "g->qqbar Sudakov scale choice pT",
        "settings": ["set /Herwig/Shower/GtoQQbarSplitFn:ScaleChoice pT"],
    },
    "gqq_scale_q2": {
        "description": "g->qqbar Sudakov scale choice Q2",
        "settings": ["set /Herwig/Shower/GtoQQbarSplitFn:ScaleChoice Q2"],
    },
    "gqq_scale_from_ao": {
        "description": "g->qqbar Sudakov scale choice FromAngularOrdering",
        "settings": ["set /Herwig/Shower/GtoQQbarSplitFn:ScaleChoice FromAngularOrdering"],
    },
    "gqq_ao_no": {
        "description": "g->qqbar AngularOrdered disabled",
        "settings": ["set /Herwig/Shower/GtoQQbarSplitFn:AngularOrdered No"],
    },
    "gqq_ao_yes": {
        "description": "g->qqbar AngularOrdered enabled; diagnostic for subsequent-emission studies",
        "settings": ["set /Herwig/Shower/GtoQQbarSplitFn:AngularOrdered Yes"],
    },
    "gqq_strictao_no": {
        "description": "g->qqbar StrictAO disabled",
        "settings": ["set /Herwig/Shower/GtoQQbarSplitFn:StrictAO No"],
    },
    "partner_scale": {
        "description": "PartnerFinder scale choice Partner",
        "settings": ["set /Herwig/Shower/PartnerFinder:ScaleChoice Partner"],
    },
}

SCAN_MODE_GROUPS = {
    "recommended": [
        "baseline",
        "renorm_0p5",
        "renorm_2p0",
        "hard_0p5",
        "hard_2p0",
        "gqq_scale_pt",
        "gqq_scale_q2",
        "gqq_scale_from_ao",
        "gqq_ao_no",
        "gqq_strictao_no",
        "partner_scale",
    ],
    "all": list(SHOWER_VARIATIONS),
}


@dataclass
class ValidationRunConfig(object):
    split_input_lhe: Path
    direct_input_lhe: Path
    workdir: Path
    events: int
    probe_trials: int = 0
    run_name: str = "hbb_validation"
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
    stage1_extra_settings: tuple = ()
    variation_label: str = "baseline"


def _default_runner(command, cwd):
    subprocess.run(command, cwd=str(cwd), check=True)


def _write_text(path, text, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError("%s already exists; pass --overwrite to replace it" % path)
    path.write_text(text)


def _read_lhe_text(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(str(path), "rt") as handle:
            return handle.read()
    return path.read_text(errors="replace")


def _line_columns(line):
    return line.split("#", 1)[0].split()


def _init_xsec(path):
    text = _read_lhe_text(path)
    lines = text.splitlines()
    init_start = None
    init_end = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "<init>":
            init_start = index
        elif stripped == "</init>" and init_start is not None:
            init_end = index
            break
    if init_start is None or init_end is None:
        return None, None

    data = [_line_columns(line) for line in lines[init_start + 1 : init_end] if _line_columns(line)]
    if len(data) < 2:
        return None, None
    process_row = data[1]
    if len(process_row) < 2:
        return None, None
    return float(process_row[0]), float(process_row[1])


def _event_weight(event):
    if len(event.header) >= 3:
        return float(event.header[2])
    return 1.0


def _pt(particle):
    return math.hypot(particle.px, particle.py)


def _momentum_abs(particle):
    return math.sqrt(particle.px * particle.px + particle.py * particle.py + particle.pz * particle.pz)


def _eta(particle):
    p_abs = _momentum_abs(particle)
    denominator = p_abs - particle.pz
    numerator = p_abs + particle.pz
    if denominator <= 0.0:
        return math.copysign(float("inf"), particle.pz if particle.pz != 0.0 else 1.0)
    if numerator <= 0.0:
        return math.copysign(float("inf"), particle.pz if particle.pz != 0.0 else -1.0)
    return 0.5 * math.log(numerator / denominator)


def _phi(particle):
    return math.atan2(particle.py, particle.px)


def _delta_phi(phi_a, phi_b):
    delta = phi_a - phi_b
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta <= -math.pi:
        delta += 2.0 * math.pi
    return delta


def _delta_r(a, b):
    eta_a = _eta(a)
    eta_b = _eta(b)
    if not math.isfinite(eta_a) or not math.isfinite(eta_b):
        return float("inf")
    return math.hypot(eta_a - eta_b, _delta_phi(_phi(a), _phi(b)))


def _invariant_mass(particles):
    energy = sum(p.energy for p in particles)
    px = sum(p.px for p in particles)
    py = sum(p.py for p in particles)
    pz = sum(p.pz for p in particles)
    mass2 = energy * energy - px * px - py * py - pz * pz
    return math.sqrt(max(0.0, mass2))


def _mother_indices(particle):
    indices = []
    for index in (particle.mother1, particle.mother2):
        if index > 0 and index not in indices:
            indices.append(index)
    return indices


def _has_ancestor_pid(event, particle, target_pid, seen=None):
    if seen is None:
        seen = set()
    for mother_index in _mother_indices(particle):
        if mother_index in seen:
            continue
        if mother_index < 1 or mother_index > len(event.particles):
            continue
        seen.add(mother_index)
        mother = event.particles[mother_index - 1]
        if abs(mother.pid) == abs(target_pid):
            return True
        if _has_ancestor_pid(event, mother, target_pid, seen=seen):
            return True
    return False


def _source_associated_b_quarks(source_event):
    return [
        particle
        for particle in source_event.particles
        if particle.status == 1 and abs(particle.pid) == 5 and not _has_ancestor_pid(source_event, particle, 25)
    ]


def _source_higgs_bosons(source_event):
    return [particle for particle in source_event.particles if particle.status == 1 and abs(particle.pid) == 25]


def _matching_source_event(source_events, source_pairs, final_b_quarks, associated_indices, preferred_index=None):
    if associated_indices is None:
        return None
    source_indices = []
    if preferred_index is not None and preferred_index < len(source_events):
        source_indices.append(preferred_index)
    source_indices.extend(index for index in range(len(source_events)) if index not in source_indices)
    for source_index in source_indices:
        matched = _match_source_b_indices(source_pairs[source_index], final_b_quarks)
        if matched == associated_indices:
            return source_events[source_index]
    return None


def _four_vector(particle):
    return particle.energy, particle.px, particle.py, particle.pz


def _combined_four_vector(particles):
    return (
        sum(particle.energy for particle in particles),
        sum(particle.px for particle in particles),
        sum(particle.py for particle in particles),
        sum(particle.pz for particle in particles),
    )


def _pt_from_four_vector(four_vector):
    _, px, py, _ = four_vector
    return math.hypot(px, py)


def _eta_from_four_vector(four_vector):
    _, px, py, pz = four_vector
    p_abs = math.sqrt(px * px + py * py + pz * pz)
    denominator = p_abs - pz
    numerator = p_abs + pz
    if denominator <= 0.0:
        return math.copysign(float("inf"), pz if pz != 0.0 else 1.0)
    if numerator <= 0.0:
        return math.copysign(float("inf"), pz if pz != 0.0 else -1.0)
    return 0.5 * math.log(numerator / denominator)


def _cos_theta_star(child, parent_four_vector):
    parent_energy, parent_px, parent_py, parent_pz = parent_four_vector
    parent_momentum = math.sqrt(parent_px * parent_px + parent_py * parent_py + parent_pz * parent_pz)
    if parent_energy <= 0.0 or parent_momentum <= 0.0:
        return None

    beta_x = parent_px / parent_energy
    beta_y = parent_py / parent_energy
    beta_z = parent_pz / parent_energy
    beta2 = beta_x * beta_x + beta_y * beta_y + beta_z * beta_z
    if beta2 <= 0.0 or beta2 >= 1.0:
        return None
    gamma = 1.0 / math.sqrt(1.0 - beta2)
    beta_dot_p = beta_x * child.px + beta_y * child.py + beta_z * child.pz
    factor = ((gamma - 1.0) * beta_dot_p / beta2) - gamma * child.energy
    rest_px = child.px + factor * beta_x
    rest_py = child.py + factor * beta_y
    rest_pz = child.pz + factor * beta_z
    rest_momentum = math.sqrt(rest_px * rest_px + rest_py * rest_py + rest_pz * rest_pz)
    if rest_momentum <= 0.0:
        return None

    axis_x = parent_px / parent_momentum
    axis_y = parent_py / parent_momentum
    axis_z = parent_pz / parent_momentum
    cos_theta = (rest_px * axis_x + rest_py * axis_y + rest_pz * axis_z) / rest_momentum
    return max(-1.0, min(1.0, cos_theta))


def _four_momentum_match_score(source, candidate):
    scale = max(
        1.0,
        abs(source.energy),
        abs(candidate.energy),
        _momentum_abs(source),
        _momentum_abs(candidate),
    )
    components = (
        source.px - candidate.px,
        source.py - candidate.py,
        source.pz - candidate.pz,
        source.energy - candidate.energy,
    )
    return sum((component / scale) ** 2 for component in components)


def _match_source_b_indices_with_score(source_b_quarks, final_b_quarks):
    if len(source_b_quarks) != 2 or len(final_b_quarks) != 4:
        return None

    best_score = None
    best_indices = None
    for candidate_indices in permutations(range(len(final_b_quarks)), len(source_b_quarks)):
        score = 0.0
        valid = True
        for source_b, candidate_index in zip(source_b_quarks, candidate_indices):
            candidate = final_b_quarks[candidate_index]
            if source_b.pid != candidate.pid:
                valid = False
                break
            score += _four_momentum_match_score(source_b, candidate)
        if valid and (best_score is None or score < best_score):
            best_score = score
            best_indices = set(candidate_indices)

    if best_indices is None or best_score is None:
        return None
    return best_indices, best_score


def _match_source_b_indices(source_b_quarks, final_b_quarks, max_score=SOURCE_MATCH_MAX_SCORE):
    match = _match_source_b_indices_with_score(source_b_quarks, final_b_quarks)
    if match is None:
        return None
    best_indices, best_score = match
    if best_score > max_score:
        return None
    return best_indices


def _source_pair_lookup_key(b_quarks):
    return tuple(sorted((particle.pid, round(particle.px, 6), round(particle.py, 6)) for particle in b_quarks))


def _build_source_pair_lookup(source_pairs):
    lookup = {}
    for source_pair in source_pairs:
        if len(source_pair) == 2:
            lookup.setdefault(_source_pair_lookup_key(source_pair), []).append(source_pair)
    return lookup


def _find_source_associated_indices(
    source_pairs,
    final_b_quarks,
    preferred_index=None,
    source_pair_lookup=None,
    max_score=SOURCE_MATCH_MAX_SCORE,
):
    if not source_pairs:
        return None

    if preferred_index is not None and preferred_index < len(source_pairs):
        associated_indices = _match_source_b_indices(source_pairs[preferred_index], final_b_quarks, max_score=max_score)
        if associated_indices is not None:
            return associated_indices

    if source_pair_lookup:
        for first_index, second_index in combinations(range(len(final_b_quarks)), 2):
            key = _source_pair_lookup_key([final_b_quarks[first_index], final_b_quarks[second_index]])
            for source_pair in source_pair_lookup.get(key, []):
                associated_indices = _match_source_b_indices(source_pair, final_b_quarks, max_score=max_score)
                if associated_indices is not None:
                    return associated_indices

    best_match = None
    for source_index, source_pair in enumerate(source_pairs):
        if source_index == preferred_index:
            continue
        match = _match_source_b_indices_with_score(source_pair, final_b_quarks)
        if match is None:
            continue
        associated_indices, score = match
        if score <= max_score and (best_match is None or score < best_match[1]):
            best_match = associated_indices, score
    if best_match is None:
        return None
    return best_match[0]


def _higgs_and_associated_index_sets(event, b_quarks, higgs_mass=125.0):
    from_higgs = [_has_ancestor_pid(event, b_quark, 25) for b_quark in b_quarks]
    if sum(1 for value in from_higgs if value) == 2:
        higgs_indices = {index for index, value in enumerate(from_higgs) if value}
        return higgs_indices, set(range(len(b_quarks))) - higgs_indices, "higgs_ancestry"

    best_pair = min(
        combinations(range(len(b_quarks)), 2),
        key=lambda pair: abs(_invariant_mass([b_quarks[pair[0]], b_quarks[pair[1]]]) - float(higgs_mass)),
    )
    higgs_indices = set(best_pair)
    return higgs_indices, set(range(len(b_quarks))) - higgs_indices, "higgs_mass_fallback"


def _empty_observables():
    return {
        name: {
            "values": [],
            "weights": [],
        }
        for name in OBSERVABLE_ORDER
    }


def _append(observables, name, value, weight):
    observables[name]["values"].append(float(value))
    observables[name]["weights"].append(float(weight))


def _higgs_pt_bin_observable(higgs_pt):
    if higgs_pt is None:
        return None
    for observable, lower, upper in HIGGS_PT_BINS:
        if lower is not None and higgs_pt < lower:
            continue
        if upper is not None and higgs_pt >= upper:
            continue
        return observable
    return None


def _associated_delta_r_bin(delta_r):
    for label, observable, lower, upper in ASSOCIATED_DELTA_R_BINS:
        if delta_r < lower:
            continue
        if upper is not None and delta_r >= upper:
            continue
        return label, observable
    return None, None


def _empty_associated_delta_r_bins():
    return {label: {"count": 0, "weighted": 0.0} for label, _, _, _ in ASSOCIATED_DELTA_R_BINS}


def extract_lhe_4b_sample(path, label=None, source_lhe=None):
    """Extract final-state 4b validation observables from an LHE file."""

    path = Path(path)
    text = _read_lhe_text(path)
    events = parse_lhe_events(text)
    source_lhe = Path(source_lhe) if source_lhe is not None else None
    source_events = parse_lhe_events(_read_lhe_text(source_lhe)) if source_lhe is not None else []
    source_pairs = [_source_associated_b_quarks(event) for event in source_events]
    source_pair_lookup = _build_source_pair_lookup(source_pairs)
    observables = _empty_observables()
    accepted = 0
    skipped = 0
    weighted_event_sum = 0.0
    pair_classification = {
        "source_lhe_match": 0,
        "higgs_ancestry": 0,
        "higgs_mass_fallback": 0,
        "source_lhe_unmatched": 0,
    }
    associated_delta_r_bins = _empty_associated_delta_r_bins()

    for event_index, event in enumerate(events):
        weight = _event_weight(event)
        b_quarks = [p for p in event.particles if p.status == 1 and abs(p.pid) == 5]
        if len(b_quarks) != 4:
            skipped += 1
            continue

        accepted += 1
        weighted_event_sum += weight
        ranked = sorted(b_quarks, key=_pt, reverse=True)

        for b_quark in b_quarks:
            _append(observables, "b_pt_all", _pt(b_quark), weight)
        for index, b_quark in enumerate(ranked, start=1):
            _append(observables, "b%d_pt" % index, _pt(b_quark), weight)
        associated_indices = _find_source_associated_indices(
            source_pairs,
            b_quarks,
            preferred_index=event_index,
            source_pair_lookup=source_pair_lookup,
        )
        if associated_indices is not None:
            higgs_indices = set(range(len(b_quarks))) - associated_indices
            classification_method = "source_lhe_match"
        else:
            higgs_indices, associated_indices, classification_method = _higgs_and_associated_index_sets(event, b_quarks)
        source_event = _matching_source_event(
            source_events,
            source_pairs,
            b_quarks,
            associated_indices,
            preferred_index=event_index,
        )
        source_higgs = None
        if source_event is not None:
            source_higgses = _source_higgs_bosons(source_event)
            if source_higgses:
                source_higgs = source_higgses[0]
                _append(observables, "h_pt", _pt(source_higgs), weight)
                _append(observables, "h_eta", _eta(source_higgs), weight)
        pair_classification[classification_method] += 1
        if source_lhe is not None and classification_method != "source_lhe_match":
            pair_classification["source_lhe_unmatched"] += 1
        pair_delta_rs = []
        higgs_pair = []
        associated_pair = []
        higgs_delta_r = None
        for first_index, second_index in combinations(range(len(b_quarks)), 2):
            first = b_quarks[first_index]
            second = b_quarks[second_index]
            delta_r = _delta_r(first, second)
            pair_delta_rs.append(delta_r)
            _append(observables, "dr_bb_all", delta_r, weight)
            _append(observables, "m_bb_all", _invariant_mass([first, second]), weight)
            if first_index in higgs_indices and second_index in higgs_indices:
                _append(observables, "dr_higgs_bb", delta_r, weight)
                higgs_delta_r = delta_r
                higgs_pair = [first, second]
                _append(observables, "m_higgs_bb", _invariant_mass(higgs_pair), weight)
            elif first_index in associated_indices and second_index in associated_indices:
                _append(observables, "dr_associated_bb", delta_r, weight)
                associated_pair = [first, second]
                _append(observables, "m_associated_bb", _invariant_mass(associated_pair), weight)
                bin_label, bin_observable = _associated_delta_r_bin(delta_r)
                if bin_label is not None:
                    associated_delta_r_bins[bin_label]["count"] += 1
                    associated_delta_r_bins[bin_label]["weighted"] += weight
                    _append(observables, bin_observable, delta_r, weight)
            else:
                _append(observables, "dr_cross_bb", delta_r, weight)
        if higgs_pair and higgs_delta_r is not None:
            parent_four_vector = _four_vector(source_higgs) if source_higgs is not None else _combined_four_vector(higgs_pair)
            higgs_pt = _pt_from_four_vector(parent_four_vector)
            hpt_observable = _higgs_pt_bin_observable(higgs_pt)
            if hpt_observable is not None:
                _append(observables, hpt_observable, higgs_delta_r, weight)
            decay_b = next((particle for particle in higgs_pair if particle.pid == 5), higgs_pair[0])
            cos_theta_star = _cos_theta_star(decay_b, parent_four_vector)
            if cos_theta_star is not None:
                _append(observables, "cos_theta_star_higgs_b", cos_theta_star, weight)
        if pair_delta_rs:
            _append(observables, "dr_min_bb", min(pair_delta_rs), weight)
        _append(observables, "m_4b", _invariant_mass(b_quarks), weight)

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
            "accepted_4b_events": int(accepted),
            "skipped_events": int(skipped),
            "weighted_event_sum": float(weighted_event_sum),
            "source_file": "" if source_lhe is None else str(source_lhe),
            "source_event_count": int(len(source_events)),
            "pair_classification": pair_classification,
            "associated_delta_r_bins": associated_delta_r_bins,
        },
    }


def _table_separator(widths):
    return "+" + "+".join("-" * (width + 2) for width in widths) + "+"


def _plain_table(headers, rows, right_aligned=None):
    right_aligned = set(right_aligned or [])
    text_rows = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in text_rows))
        for index, header in enumerate(headers)
    ]
    separator = _table_separator(widths)

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


def _write_validation_table(path, samples, correction_summary=None):
    headers = [
        "Sample",
        "sigma_LHE [pb]",
        "dsigma [pb]",
        "events",
        "accepted 4b",
        "skipped",
        "sum w(4b)",
        "source match",
        "ancestry",
        "mH fallback",
        "assoc dR 0.3-0.8",
        "assoc dR 0.8-1.5",
        "assoc dR >=1.5",
    ]
    rows = []
    for sample in samples:
        summary = sample["summary"]
        pair_classification = summary.get("pair_classification", {})
        delta_r_bins = summary.get("associated_delta_r_bins", {})
        rows.append(
            [
                summary["label"],
                terminal_number(summary["xsec_pb"]) if summary["xsec_pb"] is not None else "--",
                terminal_number(summary["xsec_error_pb"]) if summary["xsec_error_pb"] is not None else "--",
                int(summary["event_count"]),
                int(summary["accepted_4b_events"]),
                int(summary["skipped_events"]),
                terminal_number(summary["weighted_event_sum"]),
                int(pair_classification.get("source_lhe_match", 0)),
                int(pair_classification.get("higgs_ancestry", 0)),
                int(pair_classification.get("higgs_mass_fallback", 0)),
                "%d (%s)" % (
                    int(delta_r_bins.get("0.3_0.8", {}).get("count", 0)),
                    terminal_number(delta_r_bins.get("0.3_0.8", {}).get("weighted", 0.0)),
                ),
                "%d (%s)" % (
                    int(delta_r_bins.get("0.8_1.5", {}).get("count", 0)),
                    terminal_number(delta_r_bins.get("0.8_1.5", {}).get("weighted", 0.0)),
                ),
                "%d (%s)" % (
                    int(delta_r_bins.get("ge1.5", {}).get("count", 0)),
                    terminal_number(delta_r_bins.get("ge1.5", {}).get("weighted", 0.0)),
                ),
            ]
        )
    lines = [
        "4H LHE 4b validation summary",
        "Higgs decays are treated as BR(h->bb)=1 for this validation.",
        _plain_table(headers, rows, right_aligned=set(range(1, len(headers)))),
    ]
    if correction_summary:
        lines.extend(
            [
                "",
                "Forced-splitting correction:",
                json.dumps(correction_summary, indent=2, sort_keys=True),
            ]
        )
    Path(path).write_text("\n".join(lines) + "\n")


def write_lhe_validation_report_from_samples(
    samples,
    output_dir,
    correction_summary=None,
    title="4H LHE 4b Validation Observables",
    report_line="Final-state 4b LHE comparison; BR(h->bb)=1; validation outputs only.",
):
    """Write a validation report from already extracted LHE 4b samples."""

    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_rows = []
    for observable in OBSERVABLE_ORDER:
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
        write_observable_shape_plot(plot_path, observable, plot_samples)
        plot_rows.append(
            {
                "feature": OBSERVABLE_TITLES.get(observable, observable),
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
            "higgs_branching_ratio": "BR(h->bb)=1 validation",
            "pair_assignment": (
                "associated/higgs bb pairs use source pre-decay LHE four-momentum matching when provided; "
                "otherwise Higgs ancestry is used, with a closest-mH bb fallback only if ancestry is unavailable"
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


def write_lhe_validation_report(
    split_lhe,
    direct_lhe,
    output_dir,
    split_label="gg_hg_forced_split",
    direct_label="gg_hbb_direct",
    split_source_lhe=None,
    direct_source_lhe=None,
    correction_summary=None,
):
    """Write an LHE 4b comparison report with the standard sample webpage."""

    samples = [
        extract_lhe_4b_sample(split_lhe, label=split_label, source_lhe=split_source_lhe),
        extract_lhe_4b_sample(direct_lhe, label=direct_label, source_lhe=direct_source_lhe),
    ]
    return write_lhe_validation_report_from_samples(
        samples=samples,
        output_dir=output_dir,
        correction_summary=correction_summary,
    )


def _mg5_deck(process, process_dir, events, ebeam1=7000.0, ebeam2=7000.0, extra_run_settings=None):
    settings = [
        "set ebeam1 %g" % float(ebeam1),
        "set ebeam2 %g" % float(ebeam2),
        "set nevents %d" % int(events),
    ]
    settings.extend(extra_run_settings or [])
    return """import model sm
generate {process}
output {process_dir} -f
launch {process_dir}
{settings}
""".format(
        process=process,
        process_dir=process_dir,
        settings="\n".join(settings),
    )


def prepare_mg5_decks(
    output_dir,
    mg5_root,
    events,
    split_process_dir="gg_hg",
    direct_process_dir="gg_hbb",
    ptb=15,
    etab=3.0,
    drbb=0.3,
    ebeam1=7000.0,
    ebeam2=7000.0,
    overwrite=False,
):
    """Write MG5 command decks for the split/direct hbb validation samples."""

    output_dir = Path(output_dir)
    mg5_root = Path(mg5_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_deck = output_dir / "prepare_gg_hg.mg5"
    direct_deck = output_dir / "prepare_gg_hbb.mg5"

    split_text = _mg5_deck(
        "g g > h g [noborn=QCD]",
        mg5_root / split_process_dir,
        events,
        ebeam1=ebeam1,
        ebeam2=ebeam2,
    )
    direct_text = _mg5_deck(
        "g g > h b b~",
        mg5_root / direct_process_dir,
        events,
        ebeam1=ebeam1,
        ebeam2=ebeam2,
        extra_run_settings=[
            "set ptb %g" % float(ptb),
            "set etab %s" % float(etab),
            "set drbb %s" % float(drbb),
        ],
    )

    _write_text(split_deck, split_text, overwrite)
    _write_text(direct_deck, direct_text, overwrite)
    return {
        "gg_hg": str(split_deck),
        "gg_hbb": str(direct_deck),
    }


def _run_herwig(herwig, subcommand, filename, cwd, runner, dry_run):
    command = [herwig, subcommand, filename]
    if not dry_run:
        runner(command, cwd)
    return command


def _validate_input_events(path, events, allow_input_oversampling):
    path = Path(path)
    if path.is_dir():
        raise RuntimeError(
            "Input LHE expected an LHE file but got a directory: %s. "
            "Check that the corresponding shell variable is set, e.g. echo \"$SPLIT_LHE\"."
            % path
        )
    if not path.exists():
        raise FileNotFoundError("Input LHE does not exist: %s" % path)
    event_count = count_lhe_events(path)
    if event_count == 0:
        raise RuntimeError("Input LHE contains no <event> blocks: %s" % path)
    if events > event_count and not allow_input_oversampling:
        raise RuntimeError(
            "Requested %d events but input LHE %s only contains %d event%s. "
            "Generate more MG events, lower --events, or pass --allow-input-oversampling for diagnostics."
            % (events, path, event_count, "" if event_count == 1 else "s")
        )
    return event_count


def run_validation_chain(config, runner=None):
    """Run split/direct Herwig validation and write the final 4b LHE report."""

    if runner is None:
        runner = _default_runner

    split_input_lhe = Path(config.split_input_lhe)
    direct_input_lhe = Path(config.direct_input_lhe)
    if not split_input_lhe.exists():
        raise FileNotFoundError("Split input LHE does not exist: %s" % split_input_lhe)
    if not direct_input_lhe.exists():
        raise FileNotFoundError("Direct input LHE does not exist: %s" % direct_input_lhe)
    split_input_lhe = split_input_lhe.resolve()
    direct_input_lhe = direct_input_lhe.resolve()

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
    split_decay_name = "%s_split_final4b" % run_name
    direct_decay_name = "%s_direct_final4b" % run_name

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
        PROCESS_CONFIGS["gg_hg"],
        input_lhe=split_input_lhe,
        output_prefix=split_stage1_name,
        events=config.events,
        seed=config.seed_stage1,
        probe_trials=config.probe_trials,
        correction_file=split_correction_file.name,
        extra_shower_settings=config.stage1_extra_settings,
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
        report_metadata = write_lhe_validation_report(
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
        "variation_label": str(config.variation_label),
        "stage1_extra_settings": list(config.stage1_extra_settings or []),
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


def _resolve_scan_modes(modes):
    if modes is None:
        modes = ["recommended"]
    resolved = []
    for mode_entry in modes:
        for mode in str(mode_entry).split(","):
            mode = mode.strip()
            if not mode:
                continue
            if mode in SCAN_MODE_GROUPS:
                for grouped_mode in SCAN_MODE_GROUPS[mode]:
                    if grouped_mode not in resolved:
                        resolved.append(grouped_mode)
            elif mode in SHOWER_VARIATIONS:
                if mode not in resolved:
                    resolved.append(mode)
            else:
                raise ValueError(
                    "Unknown scan mode '%s'. Known modes: %s; groups: %s"
                    % (mode, ", ".join(sorted(SHOWER_VARIATIONS)), ", ".join(sorted(SCAN_MODE_GROUPS)))
                )
    if not resolved:
        raise ValueError("No scan modes selected")
    return resolved


def _write_scan_manifest(path, runs):
    fieldnames = [
        "mode",
        "description",
        "workdir",
        "run_name",
        "split_stage1_card",
        "split_decay_card",
        "direct_decay_card",
        "report",
        "settings",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "mode": run["variation_label"],
                    "description": SHOWER_VARIATIONS[run["variation_label"]]["description"],
                    "workdir": run["workdir"],
                    "run_name": run["run_name"],
                    "split_stage1_card": run["split_stage1_card"],
                    "split_decay_card": run["split_decay_card"],
                    "direct_decay_card": run["direct_decay_card"],
                    "report": run["report"] or "",
                    "settings": "; ".join(run.get("stage1_extra_settings", [])),
                }
            )


def _write_run_sequence(path, runs):
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    for run in runs:
        lines.append("# %s: %s" % (run["variation_label"], SHOWER_VARIATIONS[run["variation_label"]]["description"]))
        lines.append("cd %s" % shlex.quote(run["workdir"]))
        for command in run["commands"]:
            lines.append(command)
        lines.append("")
    Path(path).write_text("\n".join(lines))
    Path(path).chmod(0o755)


def _write_scan_aggregate_report(runs, output_dir, direct_source_lhe, correction_summary=None):
    completed_runs = [run for run in runs if run.get("report")]
    if not completed_runs:
        return None
    first_run = completed_runs[0]
    samples = [
        extract_lhe_4b_sample(
            first_run["direct_final_lhe"],
            label="gg_hbb_direct",
            source_lhe=direct_source_lhe,
        )
    ]
    for run in completed_runs:
        samples.append(
            extract_lhe_4b_sample(
                run["split_final_lhe"],
                label="gg_hg_forced_split__%s" % run["variation_label"],
                source_lhe=run["split_decay_source_lhe"],
            )
        )
    return write_lhe_validation_report_from_samples(
        samples=samples,
        output_dir=Path(output_dir) / "aggregate_report",
        correction_summary=correction_summary,
        title="4H Hbb Forced-Splitting Shower Scan",
        report_line="Overlay of validation-only gg->hg forced-splitting shower variations against gg->hbb direct.",
    )


def run_validation_scan(
    split_input_lhe,
    direct_input_lhe,
    workdir,
    events,
    probe_trials=0,
    modes=None,
    run_name_prefix="hbb_validation",
    herwig="Herwig",
    seed_stage1=31122002,
    seed_split_decay=44071981,
    seed_direct_decay=44071982,
    pdf_name=DEFAULT_HERWIG_PDF_NAME,
    input_xsec_error=None,
    allow_zero_probe_successes=False,
    allow_input_oversampling=False,
    overwrite=False,
    dry_run=False,
    runner=None,
):
    """Run or dry-run the hbb validation sequence for named shower variations."""

    resolved_modes = _resolve_scan_modes(modes)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    runs = []
    for mode_index, mode in enumerate(resolved_modes):
        variation = SHOWER_VARIATIONS[mode]
        mode_run = run_validation_chain(
            ValidationRunConfig(
                split_input_lhe=split_input_lhe,
                direct_input_lhe=direct_input_lhe,
                workdir=workdir / mode,
                events=events,
                probe_trials=probe_trials,
                run_name="%s_%s" % (run_name_prefix, mode),
                herwig=herwig,
                seed_stage1=seed_stage1 + mode_index,
                seed_split_decay=seed_split_decay + mode_index,
                seed_direct_decay=seed_direct_decay + mode_index,
                pdf_name=pdf_name,
                input_xsec_error=input_xsec_error,
                allow_zero_probe_successes=allow_zero_probe_successes,
                allow_input_oversampling=allow_input_oversampling,
                overwrite=overwrite,
                dry_run=dry_run,
                stage1_extra_settings=tuple(variation["settings"]),
                variation_label=mode,
            ),
            runner=runner,
        )
        runs.append(mode_run)

    scan_manifest = workdir / "scan_manifest.csv"
    run_sequence = workdir / "run_sequence.sh"
    scan_summary = workdir / "scan_summary.json"
    aggregate_metadata = None
    if not dry_run:
        aggregate_metadata = _write_scan_aggregate_report(
            runs,
            output_dir=workdir,
            direct_source_lhe=direct_input_lhe,
            correction_summary={run["variation_label"]: run.get("weight_check") for run in runs},
        )

    _write_scan_manifest(scan_manifest, runs)
    _write_run_sequence(run_sequence, runs)
    summary = {
        "validation_only": True,
        "split_input_lhe": str(Path(split_input_lhe).resolve()),
        "direct_input_lhe": str(Path(direct_input_lhe).resolve()),
        "workdir": str(workdir),
        "events": int(events),
        "probe_trials": int(probe_trials),
        "modes": resolved_modes,
        "mode_settings": {mode: SHOWER_VARIATIONS[mode] for mode in resolved_modes},
        "runs": runs,
        "scan_manifest": str(scan_manifest),
        "run_sequence": str(run_sequence),
        "aggregate_report": None if aggregate_metadata is None else str(aggregate_metadata["index"]),
        "dry_run": bool(dry_run),
        "scan_summary": str(scan_summary),
    }
    _write_text(scan_summary, json.dumps(summary, indent=2, sort_keys=True) + "\n", True)
    return summary


def weight_check_report_lines(summary):
    """Return human-readable probe-trial correction lines for a run or scan summary."""

    runs = summary.get("runs") if isinstance(summary, dict) else None
    if runs is None:
        runs = [summary]

    detail_lines = []
    total_zero_success_rows = 0
    for index, run in enumerate(runs, start=1):
        weight_check = run.get("weight_check") or {}
        if not weight_check:
            continue
        correction_rows = int(weight_check.get("correction_rows", 0))
        zero_success_rows = int(weight_check.get("zero_success_rows", 0))
        total_zero_success_rows += zero_success_rows
        label = run.get("variation_label") or run.get("run_name") or "run_%d" % index
        detail_lines.append(
            "  {label}: unsuccessful rows={zero}/{rows} nonzero_weight_rows={nonzero} mean_p_hat={mean:g}".format(
                label=label,
                zero=zero_success_rows,
                rows=correction_rows,
                nonzero=int(weight_check.get("nonzero_weight_rows", 0)),
                mean=float(weight_check.get("mean_p_hat", 0.0)),
            )
        )

    if not detail_lines:
        return []
    return ["Forced-splitting probe check:", "total unsuccessful rows: %d" % total_zero_success_rows] + detail_lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    prepare = subparsers.add_parser("prepare-mg5", help="write MG5 command decks for validation samples")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--mg5-root", type=Path, required=True)
    prepare.add_argument("--events", type=int, required=True)
    prepare.add_argument("--split-process-dir", default="gg_hg")
    prepare.add_argument("--direct-process-dir", default="gg_hbb")
    prepare.add_argument("--ptb", type=float, default=15.0)
    prepare.add_argument("--etab", type=float, default=3.0)
    prepare.add_argument("--drbb", type=float, default=0.3)
    prepare.add_argument("--ebeam1", type=float, default=7000.0)
    prepare.add_argument("--ebeam2", type=float, default=7000.0)
    prepare.add_argument("--overwrite", action="store_true")

    run = subparsers.add_parser("run", help="run split/direct Herwig validation and write a report")
    run.add_argument("--split-lhe", type=Path, required=True)
    run.add_argument("--direct-lhe", type=Path, required=True)
    run.add_argument("--workdir", type=Path, required=True)
    run.add_argument("--events", type=int, required=True)
    run.add_argument("--probe-trials", type=int, default=0)
    run.add_argument("--run-name", default="hbb_validation")
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

    compare = subparsers.add_parser("compare", help="write a validation report from existing final 4b LHE files")
    compare.add_argument("--split-lhe", type=Path, required=True)
    compare.add_argument("--direct-lhe", type=Path, required=True)
    compare.add_argument("--split-source-lhe", type=Path)
    compare.add_argument("--direct-source-lhe", type=Path)
    compare.add_argument("--output-dir", type=Path, required=True)

    scan = subparsers.add_parser("scan", help="run split/direct validation for a sequence of shower variations")
    scan.add_argument("--split-lhe", type=Path, required=True)
    scan.add_argument("--direct-lhe", type=Path, required=True)
    scan.add_argument("--workdir", type=Path, required=True)
    scan.add_argument("--events", type=int, required=True)
    scan.add_argument("--probe-trials", type=int, default=0)
    scan.add_argument(
        "--modes",
        default="recommended",
        help=(
            "Comma-separated modes or groups. Default: recommended. "
            "Groups: %s. Modes: %s"
            % (", ".join(sorted(SCAN_MODE_GROUPS)), ", ".join(sorted(SHOWER_VARIATIONS)))
        ),
    )
    scan.add_argument("--run-name-prefix", default="hbb_validation")
    scan.add_argument("--herwig", default="Herwig")
    scan.add_argument("--seed-stage1", type=int, default=31122002)
    scan.add_argument("--seed-split-decay", type=int, default=44071981)
    scan.add_argument("--seed-direct-decay", type=int, default=44071982)
    scan.add_argument("--pdf-name", default=DEFAULT_HERWIG_PDF_NAME)
    scan.add_argument("--input-xsec-error", type=float)
    scan.add_argument("--allow-zero-probe-successes", action="store_true")
    scan.add_argument("--allow-input-oversampling", action="store_true")
    scan.add_argument("--overwrite", action="store_true")
    scan.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "prepare-mg5":
        decks = prepare_mg5_decks(
            output_dir=args.output_dir,
            mg5_root=args.mg5_root,
            events=args.events,
            split_process_dir=args.split_process_dir,
            direct_process_dir=args.direct_process_dir,
            ptb=args.ptb,
            etab=args.etab,
            drbb=args.drbb,
            ebeam1=args.ebeam1,
            ebeam2=args.ebeam2,
            overwrite=args.overwrite,
        )
        print("MG5 validation decks:")
        print("  gg_hg:", decks["gg_hg"])
        print("  gg_hbb:", decks["gg_hbb"])
        return 0
    if args.command == "run":
        summary = run_validation_chain(
            ValidationRunConfig(
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
        metadata = write_lhe_validation_report(
            split_lhe=args.split_lhe,
            direct_lhe=args.direct_lhe,
            output_dir=args.output_dir,
            split_source_lhe=args.split_source_lhe,
            direct_source_lhe=args.direct_source_lhe,
        )
        print("Validation report:", metadata["index"])
        return 0
    if args.command == "scan":
        summary = run_validation_scan(
            split_input_lhe=args.split_lhe,
            direct_input_lhe=args.direct_lhe,
            workdir=args.workdir,
            events=args.events,
            probe_trials=args.probe_trials,
            modes=args.modes.split(","),
            run_name_prefix=args.run_name_prefix,
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
        print("Validation scan summary:", summary["scan_summary"])
        print("Validation scan manifest:", summary["scan_manifest"])
        print("Run sequence:", summary["run_sequence"])
        if summary["aggregate_report"]:
            print("Aggregate report:", summary["aggregate_report"])
        for line in weight_check_report_lines(summary):
            print(line)
        return 0
    parser.error("choose a command: prepare-mg5, run, compare, or scan")


if __name__ == "__main__":
    main()
