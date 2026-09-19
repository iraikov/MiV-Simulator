"""
Integration tests for optimize_network module using the new
_network_objectives API.

The real optimize_network and network_objectives modules are imported
canonically, with only the dmosopt package stubbed, since dmosopt is
not a miv_simulator dependency and its import requires spawning a
distributed worker pool.
"""

import sys
import types

import numpy as np

# ---------------------------------------------------------------------------
# Stub the ``dmosopt`` package so that importing optimize_network does not
# require a distributed worker environment.
# ---------------------------------------------------------------------------

dmosopt_pkg = types.ModuleType("dmosopt")
dmosopt_pkg.__path__ = []  # makes it a package

# Fake dmosopt.dmosopt submodule
dmosopt_dmo = types.ModuleType("dmosopt.dmosopt")

# Fake dmosopt.MOASMO submodule
dmosopt_moasmo = types.ModuleType("dmosopt.MOASMO")
dmosopt_moasmo.get_best = lambda *a, **k: None

# Wire them up
dmosopt_pkg.dmosopt = dmosopt_dmo
dmosopt_pkg.MOASMO = dmosopt_moasmo

stub_names = ["dmosopt", "dmosopt.dmosopt", "dmosopt.MOASMO"]
saved_modules = {name: sys.modules.get(name) for name in stub_names}

sys.modules["dmosopt"] = dmosopt_pkg
sys.modules["dmosopt.dmosopt"] = dmosopt_dmo
sys.modules["dmosopt.MOASMO"] = dmosopt_moasmo

from miv_simulator import network_objectives as mod_no  # noqa: E402
from miv_simulator import optimize_network as mod_on  # noqa: E402

# Restore the dmosopt modules that were shadowed by the stubs
for name, module in saved_modules.items():
    if module is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_pop_dict(n_total, n_active, rates_by_gid, n_bins=50, dt=2.0):
    time_bins = np.arange(0, n_bins * dt, dt, dtype=np.float32)
    spike_density_dict = {
        gid: {"rate": np.asarray(r, dtype=np.float32)}
        for gid, r in rates_by_gid.items()
    }
    return {
        "n_total": n_total,
        "n_active": n_active,
        "time_bins": time_bins,
        "spike_density_dict": spike_density_dict,
    }


# ---------------------------------------------------------------------------
# _merge_pop_features tests
# ---------------------------------------------------------------------------


def test_merge_two_workers():
    w0 = make_pop_dict(
        50, 40, {gid: np.full(50, 5.0, dtype=np.float32) for gid in range(50)}
    )
    w1 = make_pop_dict(
        50, 35, {gid: np.full(50, 5.0, dtype=np.float32) for gid in range(50, 100)}
    )
    merged = mod_on._merge_pop_features([w0, w1])
    assert merged["n_total"] == 100
    assert merged["n_active"] == 75
    assert len(merged["spike_density_dict"]) == 100
    print("  test_merge_two_workers passed")


def test_merge_single_worker():
    w0 = make_pop_dict(
        50, 40, {gid: np.full(50, 5.0, dtype=np.float32) for gid in range(50)}
    )
    merged = mod_on._merge_pop_features([w0])
    assert merged["n_total"] == 50
    assert merged["n_active"] == 40
    assert len(merged["spike_density_dict"]) == 50
    print("  test_merge_single_worker passed")


def test_merge_empty():
    merged = mod_on._merge_pop_features([])
    assert merged == {}
    print("  test_merge_empty passed")


# ---------------------------------------------------------------------------
# compute_objectives tests
# ---------------------------------------------------------------------------


def test_compute_objectives_steady_firing():
    steady_rate = np.full(50, 5.0, dtype=np.float32)
    features_dict = {
        "CA3": make_pop_dict(100, 80, {gid: steady_rate for gid in range(80)}),
    }
    local_features = [{0: features_dict}]

    operational_config = {
        "target_populations": ["CA3"],
        "temporal_resolution": 2.0,
    }
    opt_config = mod_no.NetworkOptimizationConfig(
        features=[
            mod_no.MeanFiringRateFeature(),
            mod_no.FractionActiveFeature(),
            mod_no.FiringRateStabilityFeature(temporal_resolution=2.0),
        ],
        objectives=[mod_no.TargetRateObjective("CA3", target_rate=5.0)],
        constraints=[mod_no.FiringRateBoundConstraint("CA3", min_rate=0.5)],
    )
    result = mod_on.compute_objectives(
        local_features, operational_config, {}, opt_config
    )
    objectives, features, constraints = result[0]
    assert objectives.shape == (1,)
    assert constraints.shape == (1,)
    assert constraints[0] > 0.0  # feasible
    # features array: one row, one column per NetworkFeature scalar across populations
    # MeanFiringRateFeature(1) + FractionActiveFeature(1) + FiringRateStabilityFeature(3) = 5 for CA3
    assert set(features.dtype.names) == {
        "CA3.mean_rate",
        "CA3.fraction_active",
        "CA3.mean_fraction_active_per_bin",
        "CA3.std_fraction_active_per_bin",
        "CA3.rate_cv",
    }
    assert abs(features[0]["CA3.mean_rate"] - 5.0) < 0.1
    print("  test_compute_objectives_steady_firing passed")


def test_compute_objectives_burst_then_silence():
    early_rate = np.array([20.0] * 10 + [0.0] * 40, dtype=np.float32)
    features_dict = {
        "CA3": make_pop_dict(100, 100, {gid: early_rate for gid in range(100)}),
    }
    local_features = [{0: features_dict}]

    operational_config = {
        "target_populations": ["CA3"],
        "temporal_resolution": 2.0,
    }
    opt_config = mod_no.NetworkOptimizationConfig(
        features=[
            mod_no.MeanFiringRateFeature(),
            mod_no.FiringRateStabilityFeature(temporal_resolution=2.0),
        ],
        objectives=[mod_no.SteadyFiringObjective("CA3")],
        constraints=[mod_no.SteadyFiringConstraint("CA3", max_cv=0.5)],
    )
    result = mod_on.compute_objectives(
        local_features, operational_config, {}, opt_config
    )
    objectives, features, constraints = result[0]
    assert constraints[0] <= 0.0  # infeasible (CV too high)
    # compute_objectives negates for dmosopt minimizer: -(-2.0) = 2.0
    assert objectives[0] > 1.0  # heavily penalized in dmosopt space
    # features: MeanFiringRateFeature(1) + FiringRateStabilityFeature(3) = 4 for CA3
    assert set(features.dtype.names) == {
        "CA3.mean_rate",
        "CA3.mean_fraction_active_per_bin",
        "CA3.std_fraction_active_per_bin",
        "CA3.rate_cv",
    }
    assert features[0]["CA3.rate_cv"] > 1.0  # high CV confirms burst-then-silence
    print("  test_compute_objectives_burst_then_silence passed")


def test_compute_objectives_silent_population():
    features_dict = {
        "CA3": make_pop_dict(100, 0, {}),
    }
    local_features = [{0: features_dict}]

    operational_config = {
        "target_populations": ["CA3"],
        "temporal_resolution": 2.0,
    }
    opt_config = mod_no.NetworkOptimizationConfig(
        features=[
            mod_no.MeanFiringRateFeature(),
            mod_no.FractionActiveFeature(),
            mod_no.FiringRateStabilityFeature(temporal_resolution=2.0),
        ],
        objectives=[mod_no.TargetRateObjective("CA3", target_rate=5.0)],
        constraints=[
            mod_no.FiringRateBoundConstraint("CA3", min_rate=0.5),
            mod_no.MinActiveFractionConstraint("CA3", min_fraction=0.1),
        ],
    )
    result = mod_on.compute_objectives(
        local_features, operational_config, {}, opt_config
    )
    objectives, features, constraints = result[0]

    # Both constraints should be infeasible for silent population
    assert constraints[0] <= 0.0  # rate too low
    assert constraints[1] <= 0.0  # fraction too low
    # features: MeanFiringRateFeature(1) + FractionActiveFeature(1) + FiringRateStabilityFeature(3) = 5
    assert "CA3.mean_rate" in features.dtype.names
    assert "CA3.fraction_active" in features.dtype.names
    assert features[0]["CA3.mean_rate"] == 0.0
    assert features[0]["CA3.fraction_active"] == 0.0
    print("  test_compute_objectives_silent_population passed")


def test_compute_objectives_cross_pop():
    steady_rate = np.full(50, 5.0, dtype=np.float32)
    features_dict = {
        "CA3": make_pop_dict(100, 80, {gid: steady_rate for gid in range(80)}),
        "DG": make_pop_dict(50, 40, {gid: steady_rate for gid in range(40)}),
    }
    local_features = [{0: features_dict}]

    operational_config = {
        "target_populations": ["CA3", "DG"],
        "temporal_resolution": 2.0,
    }
    opt_config = mod_no.NetworkOptimizationConfig(
        features=[
            mod_no.MeanFiringRateFeature(),
            mod_no.FiringRateStabilityFeature(temporal_resolution=2.0),
        ],
        objectives=[
            mod_no.MultiPopSteadyFiringObjective(["CA3", "DG"], name="network_steady")
        ],
        constraints=[
            mod_no.SteadyFiringConstraint("CA3", max_cv=0.5),
            mod_no.SteadyFiringConstraint("DG", max_cv=0.5),
        ],
    )
    result = mod_on.compute_objectives(
        local_features, operational_config, {}, opt_config
    )
    objectives, features, constraints = result[0]
    assert objectives.shape == (1,)
    assert constraints.shape == (2,)
    assert all(c > 0.0 for c in constraints)  # all feasible
    # features: MeanFiringRateFeature(1) + FiringRateStabilityFeature(3) per pop × 2 pops = 8
    assert len(features.dtype.names) == 8
    assert "CA3.mean_rate" in features.dtype.names
    assert "DG.mean_rate" in features.dtype.names
    assert "CA3.rate_cv" in features.dtype.names
    assert "DG.rate_cv" in features.dtype.names
    print("  test_compute_objectives_cross_pop passed")


def test_compute_objectives_multi_worker():
    steady_rate = np.full(50, 5.0, dtype=np.float32)
    local0 = {
        0: {
            "CA3": make_pop_dict(50, 40, {gid: steady_rate for gid in range(50)}),
        }
    }
    local1 = {
        0: {
            "CA3": make_pop_dict(50, 35, {gid: steady_rate for gid in range(50, 100)}),
        }
    }
    local_features = [local0, local1]

    operational_config = {
        "target_populations": ["CA3"],
        "temporal_resolution": 2.0,
    }
    opt_config = mod_no.NetworkOptimizationConfig(
        features=[mod_no.MeanFiringRateFeature()],
        objectives=[mod_no.TargetRateObjective("CA3", target_rate=5.0)],
        constraints=[mod_no.FiringRateBoundConstraint("CA3", min_rate=0.5)],
    )
    result = mod_on.compute_objectives(
        local_features, operational_config, {}, opt_config
    )
    objectives, features, constraints = result[0]
    assert constraints[0] > 0.0  # feasible
    print("  test_compute_objectives_multi_worker passed")


# ---------------------------------------------------------------------------
# YAML loading test
# ---------------------------------------------------------------------------


def test_yaml_loading():
    netclamp_config = {
        "Network Optimization": {
            "Features": [
                {"class": "miv_simulator.network_objectives.MeanFiringRateFeature"},
                {"class": "miv_simulator.network_objectives.FractionActiveFeature"},
                {
                    "class": "miv_simulator.network_objectives.FiringRateStabilityFeature",
                    "kwargs": {"temporal_resolution": 2.0},
                },
            ],
            "Objectives": [
                {
                    "class": "miv_simulator.network_objectives.TargetRateObjective",
                    "kwargs": {"pop_name": "CA3", "target_rate": 5.0},
                }
            ],
            "Constraints": [
                {
                    "class": "miv_simulator.network_objectives.FiringRateBoundConstraint",
                    "kwargs": {"pop_name": "CA3", "min_rate": 0.5},
                }
            ],
        }
    }
    opt_config = mod_no.load_network_opt_config(netclamp_config)
    assert len(opt_config.features) == 3
    assert len(opt_config.objectives) == 1
    assert len(opt_config.constraints) == 1
    print("  test_yaml_loading passed")


# ---------------------------------------------------------------------------
# Pickle validation tests
# ---------------------------------------------------------------------------


def test_pickle_validation():
    opt_config = mod_no.NetworkOptimizationConfig(
        features=[mod_no.MeanFiringRateFeature()],
        objectives=[mod_no.TargetRateObjective("CA3", target_rate=5.0)],
        constraints=[mod_no.FiringRateBoundConstraint("CA3", min_rate=0.5)],
    )
    opt_config.validate_picklable()  # should not raise
    print("  test_pickle_validation passed")


def test_pickle_validation_lambda_fails():
    try:
        opt_config = mod_no.NetworkOptimizationConfig(
            objectives=[mod_no.CustomNetworkObjective("bad", [], lambda x: 0.0)]
        )
        opt_config.validate_picklable()
        assert False, "Should have raised"
    except (TypeError, Exception):
        pass
    print("  test_pickle_validation_lambda_fails passed")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running optimize_network integration tests...")

    test_merge_two_workers()
    test_merge_single_worker()
    test_merge_empty()
    test_compute_objectives_steady_firing()
    test_compute_objectives_burst_then_silence()
    test_compute_objectives_silent_population()
    test_compute_objectives_cross_pop()
    test_compute_objectives_multi_worker()
    test_yaml_loading()
    test_pickle_validation()
    test_pickle_validation_lambda_fails()

    print("\nAll optimize_network integration tests passed!")
