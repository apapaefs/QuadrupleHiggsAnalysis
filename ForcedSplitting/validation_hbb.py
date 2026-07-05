"""Validate forced g -> b bbar splitting against direct gg -> h b bbar LHEs."""

import argparse
import gzip
import json
import math
import sys
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path
import subprocess

from .herwig_cards import PROCESS_CONFIGS, higgs_decay_lhewriter_card, stage1_lhewriter_card
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
    "dr_cross_bb",
    "dr_min_bb",
    "m_bb_all",
    "m_4b",
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
    "dr_cross_bb": "cross_pair_deltaR_bb",
    "dr_min_bb": "min_deltaR_bb",
    "m_bb_all": "all_pair_m_bb",
    "m_4b": "m_4b",
}


SOURCE_MATCH_MAX_SCORE = 1.0e-3


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
    input_xsec_error: float = None
    allow_zero_probe_successes: bool = False
    allow_input_oversampling: bool = False
    overwrite: bool = False
    dry_run: bool = False


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
        pair_classification[classification_method] += 1
        if source_lhe is not None and classification_method != "source_lhe_match":
            pair_classification["source_lhe_unmatched"] += 1
        pair_delta_rs = []
        for first_index, second_index in combinations(range(len(b_quarks)), 2):
            first = b_quarks[first_index]
            second = b_quarks[second_index]
            delta_r = _delta_r(first, second)
            pair_delta_rs.append(delta_r)
            _append(observables, "dr_bb_all", delta_r, weight)
            _append(observables, "m_bb_all", _invariant_mass([first, second]), weight)
            if first_index in higgs_indices and second_index in higgs_indices:
                _append(observables, "dr_higgs_bb", delta_r, weight)
            elif first_index in associated_indices and second_index in associated_indices:
                _append(observables, "dr_associated_bb", delta_r, weight)
            else:
                _append(observables, "dr_cross_bb", delta_r, weight)
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
                int(summary["accepted_4b_events"]),
                int(summary["skipped_events"]),
                terminal_number(summary["weighted_event_sum"]),
                int(pair_classification.get("source_lhe_match", 0)),
                int(pair_classification.get("higgs_ancestry", 0)),
                int(pair_classification.get("higgs_mass_fallback", 0)),
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

    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        extract_lhe_4b_sample(split_lhe, label=split_label, source_lhe=split_source_lhe),
        extract_lhe_4b_sample(direct_lhe, label=direct_label, source_lhe=direct_source_lhe),
    ]

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
        "title": "4H LHE 4b Validation Observables",
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
        "report_line": "Final-state 4b LHE comparison; BR(h->bb)=1; validation outputs only.",
    }
    write_report_index(
        index_path,
        plot_rows,
        table_path,
        metadata,
        title="4H LHE 4b Validation Observables",
        table_label="Validation table",
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


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
    )
    direct_decay_text = higgs_decay_lhewriter_card(
        input_lhe=direct_input_lhe,
        output_prefix=direct_decay_name,
        events=config.events,
        seed=config.seed_direct_decay,
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
    parser.error("choose a command: prepare-mg5, run, or compare")


if __name__ == "__main__":
    main()
