"""
Integration tests for optimize_network module using the new
_network_objectives API.  These tests do not import miv_simulator.
"""

import sys
import types
import numpy as np
import importlib.util

# ---------------------------------------------------------------------------
# Build a fake ``dmosopt`` package with a real ``MOASMO`` subpackage so
# ``from dmosopt.MOASMO import get_best`` succeeds.
# ---------------------------------------------------------------------------

# Fake dmosopt package
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

sys.modules["dmosopt"] = dmosopt_pkg
sys.modules["dmosopt.dmosopt"] = dmosopt_dmo
sys.modules["dmosopt.MOASMO"] = dmosopt_moasmo

# Other external deps
sys.modules["neuron"] = types.ModuleType("neuron")
sys.modules["neuron"].h = None
sys.modules["click"] = types.ModuleType("click")
sys.modules["click"].get_current_context = lambda: None
sys.modules["mpi4py"] = types.ModuleType("mpi4py")
sys.modules["mpi4py"].MPI = types.ModuleType("mpi4py.MPI")
sys.modules["mpi4py"].MPI.COMM_WORLD = types.SimpleNamespace(size=1)

# ---------------------------------------------------------------------------
# Build a fake ``miv_simulator`` package tree so the real optimize_network
# module can execute its imports.
# ---------------------------------------------------------------------------

fake_miv = types.ModuleType("miv_simulator")

# env
fake_env_mod = types.ModuleType("miv_simulator.env")
fake_env_mod.Env = lambda **kw: None  # placeholder
fake_miv.env = fake_env_mod

# network
fake_network_mod = types.ModuleType("miv_simulator.network")
fake_miv.network = fake_network_mod

# mechanisms
fake_mechanisms_mod = types.ModuleType("miv_simulator.mechanisms")
fake_mechanisms_mod.compile_and_load = lambda **kw: None
fake_miv.mechanisms = fake_mechanisms_mod

# utils
fake_utils_mod = types.ModuleType("miv_simulator.utils")
fake_utils_mod.read_from_yaml = lambda x: {}
fake_utils_mod.write_to_yaml = lambda p, d: None
fake_utils_mod.get_module_logger = lambda name: types.SimpleNamespace(
    info=lambda *a, **k: None
)
fake_miv.utils = fake_utils_mod

# synapses
fake_synapses_mod = types.ModuleType("miv_simulator.synapses")
_syn = lambda **kw: None  # dummy SynParam-like object # noqa: E731
_syn._asdict = lambda: {}
fake_synapses_mod.syn_param_from_dict = lambda d: _syn
fake_synapses_mod.SynParam = type("SynParam", (), {})
fake_miv.synapses = fake_synapses_mod

# optimization
fake_optimization_mod = types.ModuleType("miv_simulator.optimization")
fake_optimization_mod.optimization_params = lambda *a, **k: None
fake_optimization_mod.update_network_params = lambda env, ptv: None
fake_optimization_mod.network_features = lambda env, t1, t2, pops: {}
fake_miv.optimization = fake_optimization_mod

# Load the real network_objectives module

spec_no = importlib.util.spec_from_file_location(
    "miv_simulator.network_objectives",
    "src/miv_simulator/network_objectives.py",
)
mod_no = importlib.util.module_from_spec(spec_no)
spec_no.loader.exec_module(mod_no)
fake_miv.network_objectives = mod_no

# Register the package tree
sys.modules["miv_simulator"] = fake_miv
sys.modules["miv_simulator.env"] = fake_miv.env
sys.modules["miv_simulator.network"] = fake_miv.network
sys.modules["miv_simulator.mechanisms"] = fake_miv.mechanisms
sys.modules["miv_simulator.utils"] = fake_miv.utils
sys.modules["miv_simulator.synapses"] = fake_miv.synapses
sys.modules["miv_simulator.optimization"] = fake_miv.optimization
sys.modules["miv_simulator.network_objectives"] = fake_miv.network_objectives

# Finally load the real optimize_network module
spec2 = importlib.util.spec_from_file_location(
    "miv_simulator.optimize_network",
    "src/miv_simulator/optimize_network.py",
)
mod_on = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(mod_on)


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
