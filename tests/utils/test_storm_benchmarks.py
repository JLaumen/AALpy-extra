"""Integration tests for model checking the AALpy benchmark MDPs with stormpy.

These tests download the benchmark MDPs (``DotModels/MDPs``) and their
reference property files (``Benchmarking/prism_eval_props``) from the AALpy
GitHub repository, model check each property with stormpy, and compare the
result against the ground-truth values provided by
:func:`aalpy.utils.get_correct_prop_values`.

Because they rely on network access and on stormpy (an optional dependency),
the whole module is skipped when either is unavailable.
"""

import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("stormpy", reason="stormpy is not installed")

from aalpy.utils import get_correct_prop_values, load_automaton_from_file

from aalpy_extra.utils.storm_interface import mdp_to_stormpy, stormpy_model_check

# Base URL for the raw files in the AALpy repository (master branch).
_AALPY_RAW = "https://raw.githubusercontent.com/DES-Lab/AALpy/master"

# Experiment name -> (MDP .dot file, property file), both relative to the
# repository root. ``faulty_car_alarm`` is intentionally excluded because it
# has neither a property file nor ground-truth values.
EXPERIMENTS = {
    "first_grid": (
        "DotModels/MDPs/first_grid.dot",
        "Benchmarking/prism_eval_props/first_eval.props",
    ),
    "second_grid": (
        "DotModels/MDPs/second_grid.dot",
        "Benchmarking/prism_eval_props/second_eval.props",
    ),
    "shared_coin": (
        "DotModels/MDPs/shared_coin.dot",
        "Benchmarking/prism_eval_props/shared_coin_eval.props",
    ),
    "slot_machine": (
        "DotModels/MDPs/slot_machine.dot",
        "Benchmarking/prism_eval_props/slot_machine_eval.props",
    ),
    "mqtt": (
        "DotModels/MDPs/mqtt.dot",
        "Benchmarking/prism_eval_props/emqtt_two_client.props",
    ),
    "tcp": (
        "DotModels/MDPs/tcp.dot",
        "Benchmarking/prism_eval_props/tcp_eval.props",
    ),
    "bluetooth": (
        "DotModels/MDPs/bluetooth.dot",
        "Benchmarking/prism_eval_props/bluetooth.props",
    ),
}

# Absolute tolerance when comparing to the reference values. The reference
# values are rounded to ~7-8 decimal places (and the .dot files store rounded
# transition probabilities), so the largest observed deviation is ~1e-6.
_TOLERANCE = 1e-4


def _download(url: str, dest: Path) -> None:
    urllib.request.urlretrieve(url, str(dest))


def _read_properties(path: Path):
    """Read a PRISM property file, returning one property string per entry."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@pytest.fixture(scope="session")
def benchmark_files(tmp_path_factory):
    """Download all benchmark files once per test session.

    Returns a dict mapping experiment name to ``(dot_path, props_path)``.
    Skips the whole session if the files cannot be downloaded (e.g. no
    network access).
    """
    cache_dir = tmp_path_factory.mktemp("aalpy_benchmarks")
    files = {}
    for exp_name, (dot_rel, props_rel) in EXPERIMENTS.items():
        dot_path = cache_dir / f"{exp_name}.dot"
        props_path = cache_dir / f"{exp_name}.props"
        try:
            if not dot_path.exists():
                _download(f"{_AALPY_RAW}/{dot_rel}", dot_path)
            if not props_path.exists():
                _download(f"{_AALPY_RAW}/{props_rel}", props_path)
        except Exception as exc:  # pragma: no cover - network dependent
            pytest.skip(f"Could not download benchmark files from GitHub: {exc}")
        files[exp_name] = (dot_path, props_path)
    return files


@pytest.mark.parametrize("exp_name", sorted(EXPERIMENTS))
def test_benchmark_mdp_matches_reference(exp_name, benchmark_files):
    """Each benchmark MDP's properties match the reference values."""
    dot_path, props_path = benchmark_files[exp_name]

    mdp = load_automaton_from_file(dot_path, automaton_type="mdp")
    stormpy_mdp = mdp_to_stormpy(mdp)

    properties = _read_properties(props_path)
    correct_values = get_correct_prop_values(exp_name)

    assert len(properties) == len(correct_values), (
        f"{exp_name}: {len(properties)} properties vs "
        f"{len(correct_values)} reference values"
    )

    for i, (prop_str, expected) in enumerate(zip(properties, correct_values), start=1):
        result = stormpy_model_check(stormpy_mdp, prop_str)
        assert result == pytest.approx(expected, abs=_TOLERANCE), (
            f"{exp_name} prop{i}: expected {expected}, got {result}"
        )
