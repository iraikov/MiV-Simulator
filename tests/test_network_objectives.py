"""
Integration tests for the network optimization objectives framework.
These tests verify the network_objectives module without importing
miv_simulator.
"""

import sys
import importlib.util
import numpy as np

# Load network_objectives module directly without triggering miv_simulator __init__
spec = importlib.util.spec_from_file_location(
    "network_objectives", "src/miv_simulator/network_objectives.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules["network_objectives"] = mod
spec.loader.exec_module(mod)


def make_pop_dict(n_total, n_active, rates_by_gid, n_bins=50, dt=2.0):
    """Helper: build a synthetic population raw feature dict."""
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
# Network Feature Tests
# ---------------------------------------------------------------------------


def test_mean_firing_rate_feature():
    f = mod.MeanFiringRateFeature()
    assert f.feature_names == ["mean_rate"]

    pop_dict = make_pop_dict(100, 80, {gid: np.full(50, 5.0) for gid in range(80)})
    result = f.compute("CA3", pop_dict)
    assert "mean_rate" in result
    assert abs(result["mean_rate"] - 5.0) < 0.001

    # Silent population
    pop_dict_silent = make_pop_dict(100, 0, {})
    result_silent = f.compute("CA3", pop_dict_silent)
    assert result_silent["mean_rate"] == 0.0

    print("  test_mean_firing_rate_feature passed")


def test_fraction_active_feature():
    f = mod.FractionActiveFeature()
    assert f.feature_names == ["fraction_active"]

    pop_dict = make_pop_dict(100, 80, {})
    result = f.compute("CA3", pop_dict)
    assert abs(result["fraction_active"] - 0.8) < 0.001

    # Silent population
    pop_dict_silent = make_pop_dict(100, 0, {})
    result_silent = f.compute("CA3", pop_dict_silent)
    assert result_silent["fraction_active"] == 0.0

    print("  test_fraction_active_feature passed")


def test_firing_rate_stability_feature():
    f = mod.FiringRateStabilityFeature(temporal_resolution=2.0)
    assert f.feature_names == [
        "mean_fraction_active_per_bin",
        "std_fraction_active_per_bin",
        "rate_cv",
    ]

    # Steady firing
    steady_rate = np.full(50, 5.0, dtype=np.float32)
    pop_dict = make_pop_dict(100, 80, {gid: steady_rate for gid in range(80)})
    result = f.compute("CA3", pop_dict)
    assert result["rate_cv"] < 0.05

    # Burst-then-silence
    early_rate = np.array([20.0] * 10 + [0.0] * 40, dtype=np.float32)
    pop_dict_burst = make_pop_dict(100, 100, {gid: early_rate for gid in range(100)})
    result_burst = f.compute("CA3", pop_dict_burst)
    assert result_burst["rate_cv"] > 1.0

    # Silent population
    pop_dict_silent = make_pop_dict(100, 0, {})
    result_silent = f.compute("CA3", pop_dict_silent)
    assert result_silent["rate_cv"] == 0.0

    print("  test_firing_rate_stability_feature passed")


# ---------------------------------------------------------------------------
# Network Objective Tests
# ---------------------------------------------------------------------------


def test_target_rate_objective():
    o = mod.TargetRateObjective("CA3", target_rate=5.0)
    assert o.name == "CA3_target_rate"
    assert o.required_features == ["CA3.mean_rate"]

    # Perfect match
    score = o.compute({"CA3.mean_rate": 5.0})
    assert abs(score) < 0.001

    # Off target
    score = o.compute({"CA3.mean_rate": 3.0})
    assert score < 0

    print("  test_target_rate_objective passed")


def test_steady_firing_objective():
    o = mod.SteadyFiringObjective("CA3")
    assert o.required_features == ["CA3.rate_cv"]

    # Low CV is good (higher score)
    score_low = o.compute({"CA3.rate_cv": 0.1})
    score_high = o.compute({"CA3.rate_cv": 2.0})
    assert score_low > score_high

    print("  test_steady_firing_objective passed")


def test_multi_pop_steady_firing_objective():
    o = mod.MultiPopSteadyFiringObjective(["CA3", "DG"], name="test")
    assert o.name == "test"
    assert o.required_features == ["CA3.rate_cv", "DG.rate_cv"]

    score = o.compute({"CA3.rate_cv": 0.05, "DG.rate_cv": 2.5})
    assert abs(score - -np.mean([0.05, 2.5])) < 0.001

    print("  test_multi_pop_steady_firing_objective passed")


def test_population_rate_ratio_objective():
    o = mod.PopulationRateRatioObjective("CA3", "DG", target_ratio=3.0)
    assert o.required_features == ["CA3.mean_rate", "DG.mean_rate"]

    # Perfect ratio
    score = o.compute({"CA3.mean_rate": 15.0, "DG.mean_rate": 5.0})
    assert abs(score) < 0.001

    print("  test_population_rate_ratio_objective passed")


def test_custom_network_objective():
    def custom_fn(feature_values):
        return -1.0

    o = mod.CustomNetworkObjective("custom", ["CA3.mean_rate"], custom_fn)
    assert o.name == "custom"
    assert o.compute({}) == -1.0

    # Lambda should raise
    try:
        mod.CustomNetworkObjective("bad", [], lambda x: 0.0)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

    print("  test_custom_network_objective passed")


# ---------------------------------------------------------------------------
# Network Constraint Tests
# ---------------------------------------------------------------------------


def test_firing_rate_bound_constraint():
    c = mod.FiringRateBoundConstraint("CA3", min_rate=0.5)
    assert c.name == "CA3_rate_bound"
    assert c.required_features == ["CA3.mean_rate"]

    # Below min -> infeasible (positive)
    val = c.compute({"CA3.mean_rate": 0.0})
    assert val > 0

    # Above min -> feasible (negative)
    val = c.compute({"CA3.mean_rate": 1.0})
    assert val <= 0

    print("  test_firing_rate_bound_constraint passed")


def test_steady_firing_constraint():
    c = mod.SteadyFiringConstraint("CA3", max_cv=0.5)
    assert c.required_features == ["CA3.rate_cv"]

    # Within bound -> feasible
    val = c.compute({"CA3.rate_cv": 0.1})
    assert val <= 0

    # Exceeds bound -> infeasible
    val = c.compute({"CA3.rate_cv": 1.0})
    assert val > 0

    print("  test_steady_firing_constraint passed")


# ---------------------------------------------------------------------------
# Config and Registry Tests
# ---------------------------------------------------------------------------


def test_feature_registry():
    registry = mod.FeatureRegistry
    registry.clear()

    f = mod.MeanFiringRateFeature()
    registry.register(f)
    assert "mean_rate" in registry.list()

    registry.clear()
    assert len(registry.list()) == 0

    print("  test_feature_registry passed")


def test_network_optimization_config():
    f = mod.MeanFiringRateFeature()
    o = mod.TargetRateObjective("CA3", target_rate=5.0)
    c = mod.FiringRateBoundConstraint("CA3", min_rate=0.5)

    config = mod.NetworkOptimizationConfig(
        features=[f], objectives=[o], constraints=[c]
    )

    assert config.objective_names() == ["CA3_target_rate"]
    assert config.constraint_names() == ["CA3_rate_bound"]
    assert len(config.feature_dtypes()) == 1

    # Should be picklable
    config.validate_picklable()

    print("  test_network_optimization_config passed")


def test_load_network_opt_config_default():
    """Default config when no 'Network Optimization' section is present returns empty config."""
    env_config = {}
    opt_config = mod.load_network_opt_config(env_config)
    assert len(opt_config.features) == 0
    assert len(opt_config.objectives) == 0
    assert len(opt_config.constraints) == 0
    assert opt_config.target_populations() == []
    print("  test_load_network_opt_config_default passed")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running network_objectives integration tests...")

    test_mean_firing_rate_feature()
    test_fraction_active_feature()
    test_firing_rate_stability_feature()
    test_target_rate_objective()
    test_steady_firing_objective()
    test_multi_pop_steady_firing_objective()
    test_population_rate_ratio_objective()
    test_custom_network_objective()
    test_firing_rate_bound_constraint()
    test_steady_firing_constraint()
    test_feature_registry()
    test_network_optimization_config()
    test_load_network_opt_config_default()

    print("\nAll integration tests passed!")
