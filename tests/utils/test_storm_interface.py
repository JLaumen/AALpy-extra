"""Tests for the storm_interface module.

These tests require stormpy to be installed.  If stormpy is not
available, all tests are automatically skipped via
:func:`pytest.importorskip`.
"""

import pytest

pytest.importorskip("stormpy", reason="stormpy is not installed")

from aalpy.automata import Mdp, MdpState, StochasticMealyMachine, StochasticMealyState
from aalpy_extra.utils.storm_interface import (
    mdp_to_stormpy,
    smm_to_stormpy_mdp,
    stormpy_model_check,
    stormpy_model_check_properties,
    stormpy_model_check_experiment,
)


class TestMdpToStormpy:
    """Tests for conversion of AALpy MDPs to stormpy models."""

    def test_simple_two_state_mdp(self):
        """Convert a simple 2-state MDP and verify its structure."""
        s0 = MdpState("s0", "start")
        s1 = MdpState("s1", "goal")
        s0.transitions["a"].append((s1, 1.0))

        mdp = Mdp(s0, [s0, s1])
        model = mdp_to_stormpy(mdp)

        assert model.nr_states == 2
        assert model.nr_choices > 0

    def test_probabilistic_transitions(self):
        """Convert an MDP with probabilistic branching."""
        s0 = MdpState("s0", "start")
        s1 = MdpState("s1", "left")
        s2 = MdpState("s2", "right")
        s0.transitions["a"].append((s1, 0.3))
        s0.transitions["a"].append((s2, 0.7))

        mdp = Mdp(s0, [s0, s1, s2])
        model = mdp_to_stormpy(mdp)

        assert model.nr_states == 3


class TestSmmToStormpyMdp:
    """Tests for conversion of Stochastic Mealy Machines to stormpy MDPs."""

    def test_simple_smm_conversion(self):
        """A simple SMM should convert to a stormpy MDP without error."""
        s0 = StochasticMealyState("s0")
        s1 = StochasticMealyState("s1")
        # SMM transition format: (target_state, output, probability)
        s0.transitions["a"].append((s1, "o1", 1.0))

        smm = StochasticMealyMachine(s0, [s0, s1])
        model = smm_to_stormpy_mdp(smm)

        assert model.nr_states > 0
        assert model.nr_choices > 0


class TestStormpyModelCheck:
    """Tests for model checking via stormpy."""

    def test_reachability_probability(self):
        """Check Pmax=? [F \"goal\"] on a simple MDP returns 1.0."""
        s0 = MdpState("s0", "start")
        s1 = MdpState("s1", "goal")
        s0.transitions["a"].append((s1, 1.0))

        mdp = Mdp(s0, [s0, s1])

        result = stormpy_model_check(mdp, 'Pmax=? [F "goal"]')
        assert pytest.approx(result, 0.01) == 1.0

    def test_reachability_with_branching(self):
        """Check reachability on a branching MDP."""
        s0 = MdpState("s0", "start")
        s1 = MdpState("s1", "goal")
        s2 = MdpState("s2", "fail")
        s0.transitions["a"].append((s1, 0.5))
        s0.transitions["a"].append((s2, 0.5))

        mdp = Mdp(s0, [s0, s1, s2])

        result = stormpy_model_check(mdp, 'Pmax=? [F "goal"]')
        assert pytest.approx(result, 0.01) == 0.5


class TestStormpyModelCheckProperties:
    """Tests for batch model checking."""

    def test_multiple_properties(self):
        """Check that multiple properties return correctly indexed results."""
        s0 = MdpState("s0", "start")
        s1 = MdpState("s1", "goal")
        s2 = MdpState("s2", "fail")
        s0.transitions["a"].append((s1, 0.6))
        s0.transitions["a"].append((s2, 0.4))

        mdp = Mdp(s0, [s0, s1, s2])

        results = stormpy_model_check_properties(
            mdp, ['Pmax=? [F "goal"]', 'Pmax=? [F "fail"]']
        )

        assert "prop1" in results
        assert "prop2" in results
        assert pytest.approx(results["prop1"], 0.01) == 0.6
        assert pytest.approx(results["prop2"], 0.01) == 0.4


class TestStormpyModelCheckExperiment:
    """Tests for the high-level experiment runner."""

    def test_experiment_without_correct_values(self):
        """Experiment without ground truth returns only results."""
        s0 = MdpState("s0", "start")
        s1 = MdpState("s1", "goal")
        s0.transitions["a"].append((s1, 1.0))

        mdp = Mdp(s0, [s0, s1])

        out = stormpy_model_check_experiment(
            mdp, ['Pmax=? [F "goal"]']
        )

        assert "results" in out
        assert "diff_to_correct" not in out
        assert "prop1" in out["results"]

    def test_experiment_with_correct_values(self):
        """Experiment with ground truth computes differences."""
        s0 = MdpState("s0", "start")
        s1 = MdpState("s1", "goal")
        s0.transitions["a"].append((s1, 1.0))

        mdp = Mdp(s0, [s0, s1])

        out = stormpy_model_check_experiment(
            mdp,
            ['Pmax=? [F "goal"]'],
            correct_values=[1.0],
            precision=4,
        )

        assert "results" in out
        assert "diff_to_correct" in out
        assert out["diff_to_correct"]["prop1"] == 0.0
