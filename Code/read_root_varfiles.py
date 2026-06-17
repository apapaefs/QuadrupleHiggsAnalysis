"""Read variable ROOT files produced by FourHiggs8bAnalysis_smear_CMS."""

from __future__ import annotations

import math
from pathlib import Path

try:
    import ROOT
except ImportError as exc:  # pragma: no cover - depends on the local HEP stack.
    raise RuntimeError("PyROOT is required to read the 4H variable ROOT files") from exc

ROOT.gROOT.SetBatch(True)

VARIABLE_COUNT = 29

FEATURE_NAMES = [
    "bjet1_pt",
    "bjet2_pt",
    "bjet3_pt",
    "bjet4_pt",
    "bjet5_pt",
    "bjet6_pt",
    "bjet7_pt",
    "bjet8_pt",
    "m8b",
    "chi8",
    "delta_m_min",
    "delta_m_med1",
    "delta_m_med2",
    "delta_m_max",
    "higgs1_pt",
    "higgs2_pt",
    "higgs3_pt",
    "higgs4_pt",
    "dr_hh_12",
    "dr_hh_13",
    "dr_hh_14",
    "dr_hh_23",
    "dr_hh_24",
    "dr_hh_34",
    "dr_bb_h1",
    "dr_bb_h2",
    "dr_bb_h3",
    "dr_bb_h4",
]


def read_ROOT_varfile(filename, sample_id, xsec=1.0, max_events=None, include_weight_feature=False):
    """Return feature rows, labels, and weighted event weights from a Data2 ROOT tree.

    The C++ analysis writes ``variables[0]`` and ``weight`` as the event weight.
    By default the classifier features are ``variables[1:]`` so the target
    weight is not leaked into training.
    """

    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"ROOT variable file does not exist: {path}")

    root_file = ROOT.TFile.Open(str(path))
    if not root_file or root_file.IsZombie():
        raise OSError(f"Failed to open ROOT variable file: {path}")

    try:
        tree = root_file.Get("Data2")
        if not tree:
            raise KeyError(f"{path} does not contain a Data2 tree")
        if not tree.GetBranch("variables"):
            raise KeyError(f"{path}: Data2 tree does not contain a variables branch")

        n_entries = int(tree.GetEntries())
        if max_events is not None:
            n_entries = min(n_entries, int(max_events))

        features = []
        labels = []
        weights = []
        feature_start = 0 if include_weight_feature else 1

        for entry in range(n_entries):
            tree.GetEntry(entry)
            values = [float(tree.variables[i]) for i in range(VARIABLE_COUNT)]
            weight = float(getattr(tree, "weight", values[0]))
            row = values[feature_start:]

            if not math.isfinite(weight):
                continue
            if not all(math.isfinite(value) for value in row):
                continue

            features.append(row)
            labels.append(sample_id)
            weights.append(weight * xsec)

        return features, labels, weights
    finally:
        root_file.Close()
