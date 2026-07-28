"""Tests for the executable SM hhhh -> 8b XGBoost/pyhf tutorial."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = REPO_ROOT / "tutorials" / "sm_gg8b_xgboost_pyhf"
sys.path.insert(0, str(TUTORIAL_DIR))
try:
    import tutorial_helpers as tutorial
finally:
    sys.path.remove(str(TUTORIAL_DIR))


EXPECTED_HASHES = {
    "sm_hhhh_8b": {
        "root": "70688a574bb175e7a4a319209aa13b0335536417ebd8aa53a6ecc60b5fd9c6e1",
        "summary": "948f5048afaa2a3d6595b63203c54fb4a84eae0f190acc9490918eebae1f5247",
    },
    "gg8b": {
        "root": "b5514870553f792045465d3f893efe48ac9c1752d4f5de5000c416149766bef1",
        "summary": "6e60f04e87e9566e2675019e44e76e92906c0b4c4f5e6c9fe4bfb99dbaed7acd",
    },
}
INPUTS_AVAILABLE = all(
    path.is_file()
    for path in (
        REPO_ROOT
        / "Signals/events/HW-gg_hhhh_SM-extended-v2_var.smearCMS.root",
        REPO_ROOT
        / "Signals/events/HW-gg_hhhh_SM-extended-v2.analysis_summary.json",
        REPO_ROOT
        / "Backgrounds/events/HW-gg_to_8b-extended-v2_var.smearCMS.root",
        REPO_ROOT
        / "Backgrounds/events/HW-gg_to_8b-extended-v2.analysis_summary.json",
    )
)


@pytest.fixture(scope="module")
def config() -> tutorial.TutorialConfig:
    return tutorial.load_config(TUTORIAL_DIR / "config.json")


@pytest.fixture(scope="module")
def loaded_samples(
    config: tutorial.TutorialConfig,
) -> tuple[tutorial.LoadedSample, tutorial.LoadedSample]:
    if not INPUTS_AVAILABLE:
        pytest.skip("the Git-ignored tutorial ROOT inputs are not available")
    pytest.importorskip("ROOT")
    return tutorial.load_samples(config)


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    if not INPUTS_AVAILABLE:
        pytest.skip("the Git-ignored tutorial ROOT inputs are not available")
    pytest.importorskip("ROOT")
    pytest.importorskip("xgboost")
    pytest.importorskip("pyhf")
    output = tmp_path_factory.mktemp("sm_gg8b_tutorial")
    return tutorial.run_tutorial(
        TUTORIAL_DIR / "config.json",
        output_dir=output,
        make_plots=True,
    )


def test_configuration_records_the_declared_physics(config: tutorial.TutorialConfig):
    assert config.schema == "extended-91-v2"
    assert config.feature_profile == "corrected28"
    assert config.sqrt_s_tev == pytest.approx(14.0)
    assert config.luminosity_fb == pytest.approx(3000.0)
    assert config.sm_cross_section_fb == pytest.approx(1.24288e-4)
    assert config.branching_ratio_hbb == pytest.approx(0.5824)
    assert config.btag_efficiency == pytest.approx(0.85)
    assert config.n_folds == 5
    assert config.seed == 12345
    assert config.signal.k_factor == pytest.approx(2.0)
    assert config.background.k_factor == pytest.approx(2.0)
    assert config.background.cross_section_fb == pytest.approx(1.03875)
    assert config.background.generation_integration_error_fraction == pytest.approx(
        0.0254
    )
    assert config.background_normsys.name == "gg8b_norm"
    assert config.background_normsys.lo == pytest.approx(0.90)
    assert config.background_normsys.hi == pytest.approx(1.10)


def test_root_hash_schema_feature_and_normalization_closure(
    config: tutorial.TutorialConfig,
    loaded_samples: tuple[tutorial.LoadedSample, tutorial.LoadedSample],
):
    signal, background = loaded_samples
    assert signal.features.shape == (990, 28)
    assert background.features.shape == (9015, 28)
    assert signal.total_weight_in == pytest.approx(10000.0)
    assert background.total_weight_in == pytest.approx(7715.43)

    expected_signal_factor = (
        2.0 * config.branching_ratio_hbb**4 * config.btag_efficiency**8
    )
    expected_background_factor = 2.0 * config.btag_efficiency**8
    assert signal.rate_factor == pytest.approx(expected_signal_factor)
    assert background.rate_factor == pytest.approx(expected_background_factor)

    for sample in loaded_samples:
        assert sample.hashes == EXPECTED_HASHES[sample.spec.name]
        assert sample.file_contract["tree_name"] == "Data3"
        assert sample.file_contract["observable_schema"] == "extended-91-v2"
        assert sample.file_contract["feature_profile"] == "corrected28"
        names = tuple(sample.file_contract["feature_names"])
        assert len(names) == 28
        assert all("weight" not in name.lower() for name in names)
        assert len(np.unique(sample.event_indices)) == sample.entries

        expected_yield = (
            config.luminosity_fb
            * sample.spec.cross_section_fb
            * sample.rate_factor
            * float(np.sum(sample.raw_weights))
            / sample.total_weight_in
        )
        assert float(np.sum(sample.physical_weights)) == pytest.approx(
            expected_yield, rel=1e-13
        )


def test_crossfit_is_disjoint_and_scores_each_event_once(completed_run: dict):
    crossfit = completed_run["crossfit"]
    n_events = len(crossfit["oof_scores"])
    assert n_events == 10005
    assert np.all(np.isfinite(crossfit["oof_scores"]))
    assert np.all(np.isfinite(crossfit["validation_scores"]))

    test_multiplicity = np.zeros(n_events, dtype=int)
    validation_multiplicity = np.zeros(n_events, dtype=int)
    for rotation in crossfit["rotations"]:
        train = np.asarray(rotation["train_mask"], dtype=bool)
        validation = np.asarray(rotation["validation_mask"], dtype=bool)
        test = np.asarray(rotation["test_mask"], dtype=bool)
        assert not np.any(train & validation)
        assert not np.any(train & test)
        assert not np.any(validation & test)
        assert sorted(rotation["train_folds"]) == sorted(
            set(np.asarray(crossfit["arrays"]["folds"])[train].tolist())
        )
        assert rotation["validation_fold"] == (rotation["rotation"] + 1) % 5
        assert rotation["test_fold"] == rotation["rotation"]
        assert rotation["classifier_signal_weight"] == pytest.approx(
            rotation["classifier_background_weight"], rel=1e-13
        )
        test_multiplicity += test
        validation_multiplicity += validation
    np.testing.assert_array_equal(test_multiplicity, np.ones(n_events, dtype=int))
    np.testing.assert_array_equal(
        validation_multiplicity, np.ones(n_events, dtype=int)
    )

    model_dir = Path(completed_run["output_dir"]) / "fold_models"
    assert len(list(model_dir.glob("fold_*.json"))) == 5
    assert len(list(model_dir.glob("metadata_fold_*.json"))) == 5


def test_validation_binning_and_test_templates_close(completed_run: dict):
    result = completed_run
    crossfit = result["crossfit"]
    arrays = crossfit["arrays"]
    labels = np.asarray(arrays["labels"], dtype=int)
    physical = np.asarray(arrays["physical_weights"], dtype=float)
    unit = np.asarray(arrays["unit_xsec_weights"], dtype=float)

    total_signal = 0.0
    total_background = 0.0
    total_signal_variance = 0.0
    total_background_variance = 0.0
    for record, channel, rotation in zip(
        result["binning"]["records"],
        result["channels"],
        crossfit["rotations"],
    ):
        assert record["validation_scale"] == result["config"].n_folds
        assert record["validation_mcstat_relative_inflation_vs_full_sample"] == (
            pytest.approx(math.sqrt(result["config"].n_folds))
        )
        assert "conservative" in record["validation_mcstat_convention"]
        selected = record["validation"]["selected"]
        assert 2 <= selected["n_bins"] <= 5
        assert np.all(np.asarray(selected["background_yield"]) > 0.0)
        assert np.all(
            np.asarray(selected["background_raw_entries"])
            >= result["config"].binning.min_background_raw
        )
        assert np.all(
            np.asarray(selected["background_effective_entries"])
            >= result["config"].binning.min_background_neff
        )
        assert tuple(record["edges"]) in {
            tuple(level["edges"])
            for level in record["validation"]["fallback_hierarchy"]
        }
        assert np.all(np.asarray(channel["background"]) > 0.0)
        assert np.all(np.asarray(channel["signal"]) >= 0.0)

        test = np.asarray(rotation["test_mask"], dtype=bool)
        test_signal = test & (labels == 1)
        test_background = test & (labels == 0)
        assert float(np.sum(channel["signal"])) == pytest.approx(
            float(np.sum(unit[test_signal])), rel=1e-12, abs=1e-14
        )
        assert float(np.sum(channel["background"])) == pytest.approx(
            float(np.sum(physical[test_background])), rel=1e-12, abs=1e-14
        )
        assert float(np.sum(np.square(channel["signal_staterror"]))) == pytest.approx(
            float(np.sum(np.square(unit[test_signal]))), rel=1e-12, abs=1e-14
        )
        assert float(
            np.sum(np.square(channel["background_staterror"]))
        ) == pytest.approx(
            float(np.sum(np.square(physical[test_background]))),
            rel=1e-12,
            abs=1e-14,
        )
        total_signal += float(np.sum(channel["signal"]))
        total_background += float(np.sum(channel["background"]))
        total_signal_variance += float(
            np.sum(np.square(channel["signal_staterror"]))
        )
        total_background_variance += float(
            np.sum(np.square(channel["background_staterror"]))
        )

    assert total_signal == pytest.approx(
        float(np.sum(unit[labels == 1])), rel=1e-12
    )
    assert total_background == pytest.approx(
        float(np.sum(physical[labels == 0])), rel=1e-12
    )
    assert total_signal_variance == pytest.approx(
        float(np.sum(np.square(unit[labels == 1]))), rel=1e-12
    )
    assert total_background_variance == pytest.approx(
        float(np.sum(np.square(physical[labels == 0]))), rel=1e-12
    )


def test_workspace_has_shared_poi_staterrors_normsys_and_auxdata(
    completed_run: dict,
):
    pyhf = pytest.importorskip("pyhf")
    spec = completed_run["workspace"]
    workspace = pyhf.Workspace(spec)
    model = workspace.model(measurement_name=tutorial.MEASUREMENT_NAME)
    assert model.config.poi_name == "sigma_hhhh_fb"
    assert len(spec["channels"]) == 5

    signal_stat_names = []
    background_stat_names = []
    for channel, observation in zip(spec["channels"], spec["observations"]):
        signal, background = channel["samples"]
        signal_modifiers = signal["modifiers"]
        background_modifiers = background["modifiers"]
        assert signal_modifiers[0] == {
            "name": "sigma_hhhh_fb",
            "type": "normfactor",
            "data": None,
        }
        signal_stat_names.extend(
            modifier["name"]
            for modifier in signal_modifiers
            if modifier["type"] == "staterror"
        )
        background_stat_names.extend(
            modifier["name"]
            for modifier in background_modifiers
            if modifier["type"] == "staterror"
        )
        normsys = [
            modifier
            for modifier in background_modifiers
            if modifier["type"] == "normsys"
        ]
        assert normsys == [
            {
                "name": "gg8b_norm",
                "type": "normsys",
                "data": {"lo": 0.9, "hi": 1.1},
            }
        ]
        assert observation["data"] == background["data"]
    assert len(signal_stat_names) == len(set(signal_stat_names)) == 5
    assert len(background_stat_names) == len(set(background_stat_names)) == 5

    data = workspace.data(model)
    assert len(data) == model.config.nmaindata + model.config.nauxdata
    bounds = np.asarray(model.config.suggested_bounds(), dtype=float)
    assert np.all(np.isfinite(bounds))
    assert np.all(bounds[:, 1] > bounds[:, 0])

    control = completed_run["one_bin_workspace"]
    assert len(control["channels"]) == 1
    assert len(control["channels"][0]["samples"][0]["data"]) == 1
    assert len(control["channels"][0]["samples"][1]["data"]) == 1

    mcstat_only = completed_run["mcstat_only_workspace"]
    assert len(mcstat_only["channels"]) == 5
    assert all(
        modifier["type"] != "normsys"
        for channel in mcstat_only["channels"]
        for sample in channel["samples"]
        for modifier in sample["modifiers"]
    )


def test_asimov_fits_limits_and_shape_improvement(completed_run: dict):
    fits = completed_run["fit_results"]
    shape_limit = fits["headline_expected_limit"]["expected_median_fb"]
    mcstat_only_limit = fits["shape_mcstat_only"]["expected_median_fb"]
    control_limit = fits["one_bin_control"]["expected_median_fb"]
    assert math.isfinite(shape_limit) and shape_limit > 0.0
    assert math.isfinite(mcstat_only_limit) and mcstat_only_limit > 0.0
    assert math.isfinite(control_limit) and control_limit > 0.0
    assert mcstat_only_limit <= shape_limit
    assert shape_limit < control_limit
    assert fits["headline_expected_limit"]["expected_median_mu"] == pytest.approx(
        shape_limit / completed_run["config"].sm_cross_section_fb
    )

    background_fit = fits["background_only_fit"]
    assert abs(background_fit["poi_hat_fb"]) < 1.0e-3
    injection = fits["injected_asimov"]
    assert injection["is_observed_data"] is False
    assert "NOT DATA" in injection["label"]
    assert injection["sigma_hhhh_fb"] == pytest.approx(0.5 * shape_limit)
    assert injection["fit"]["poi_hat_fb"] == pytest.approx(
        injection["sigma_hhhh_fb"], rel=2.0e-5
    )
    parameter_names = injection["fit"]["parameter_names"]
    gg8b_index = parameter_names.index("gg8b_norm")
    assert injection["fit"]["bestfit_parameters"][gg8b_index] == pytest.approx(
        0.5, abs=1.0e-3
    )


def test_reference_limit_and_informative_two_bin_toy(
    config: tutorial.TutorialConfig,
):
    pyhf = pytest.importorskip("pyhf")

    def median_limit(signal, background):
        channel = {
            "name": "toy",
            "signal": signal,
            "background": background,
            "signal_staterror": np.zeros(len(signal)),
            "background_staterror": np.zeros(len(background)),
        }
        spec = tutorial.build_workspace(
            [channel],
            config,
            include_staterror=False,
            include_background_normsys=False,
            poi_upper=20.0,
        )
        workspace = pyhf.Workspace(spec)
        model = workspace.model(measurement_name=tutorial.MEASUREMENT_NAME)
        bounds = model.config.suggested_bounds()[model.config.poi_index]
        _, expected = pyhf.infer.intervals.upper_limits.toms748_scan(
            workspace.data(model),
            model,
            float(bounds[0]),
            float(bounds[1]),
            level=0.05,
            test_stat="qtilde",
        )
        return float(np.asarray(expected)[2])

    reference = median_limit([1.0], [2.9206089])
    assert reference == pytest.approx(4.73532, abs=0.03)
    one_bin = median_limit([2.0], [20.0])
    informative_shape = median_limit([0.1, 1.9], [19.0, 1.0])
    assert informative_shape < one_bin


def test_pyhf_confidence_level_is_honored(config: tutorial.TutorialConfig):
    pyhf = pytest.importorskip("pyhf")
    channel = {
        "name": "toy",
        "signal": [1.0],
        "background": [2.9206089],
        "signal_staterror": [0.0],
        "background_staterror": [0.0],
    }
    spec = tutorial.build_workspace(
        [channel],
        config,
        include_staterror=False,
        include_background_normsys=False,
        poi_upper=20.0,
    )
    workspace = pyhf.Workspace(spec)
    model = workspace.model(measurement_name=tutorial.MEASUREMENT_NAME)
    data = workspace.data(model)
    bounds = model.config.suggested_bounds()[model.config.poi_index]
    _, expected_90 = pyhf.infer.intervals.upper_limits.toms748_scan(
        data,
        model,
        float(bounds[0]),
        float(bounds[1]),
        level=0.10,
        test_stat="qtilde",
    )
    _, expected_95 = pyhf.infer.intervals.upper_limits.toms748_scan(
        data,
        model,
        float(bounds[0]),
        float(bounds[1]),
        level=0.05,
        test_stat="qtilde",
    )
    assert float(np.asarray(expected_90)[2]) < float(np.asarray(expected_95)[2])


def test_source_notebook_is_clean_and_contains_beginner_explanations():
    notebook_path = TUTORIAL_DIR / "sm_gg8b_xgboost_pyhf.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["cells"]
    assert all(
        not cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert all(
        cell.get("execution_count") is None
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "".join(notebook["cells"][-1]["source"]).strip()
    lesson = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
    ).lower()
    for concept in (
        "asimov",
        "out-of-fold",
        "fixed-poi",
        "profil",
        "confidence level",
        "fast-sm",
        "toms748_scan",
    ):
        assert concept in lesson

    beginner_guide = notebook_path.with_name("BEGINNER_GUIDE.md")
    assert beginner_guide.is_file()
    assert beginner_guide.stat().st_size > 10_000


def test_requested_artifacts_and_plot_pairs_exist(completed_run: dict):
    artifacts = completed_run["artifacts"]
    required_artifacts = {
        "input_hashes",
        "versions",
        "crossfit",
        "event_scores",
        "binning",
        "channels",
        "workspace",
        "workspace_mcstat_only",
        "workspace_one_bin",
        "one_bin_control",
        "fit_results",
        "plots",
        "summary",
    }
    assert required_artifacts <= set(artifacts)
    for path in artifacts.values():
        artifact = Path(path)
        assert artifact.is_file()
        assert artifact.stat().st_size > 0
        if artifact.suffix == ".json":
            json.loads(artifact.read_text(encoding="utf-8"))

    with Path(artifacts["event_scores"]).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10005
    assert len({(row["sample"], row["event_index"]) for row in rows}) == 10005
    assert all(math.isfinite(float(row["oof_score"])) for row in rows)

    required_plots = {
        "weighted_roc",
        "feature_importance",
        "score_normalized",
        "score_expected_yields",
        "unrolled_fit",
        "likelihood_scan",
        "cls_scan",
        "nuisance_pulls",
        "reduced_correlation",
    }
    assert required_plots <= set(completed_run["plots"])
    for plot_name in required_plots:
        pair = completed_run["plots"][plot_name]
        assert set(pair) == {"pdf", "png"}
        for kind, path in pair.items():
            image = Path(path)
            assert image.is_file(), (plot_name, kind)
            assert image.stat().st_size > 1000
