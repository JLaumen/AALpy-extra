"""Interface between AALpy automata models and the Storm model checker via stormpy.

This module provides functions to convert AALpy stochastic models (MDPs,
Stochastic Mealy Machines) to stormpy sparse models and to perform model
checking of PCTL/PRCT properties using stormpy.

Usage::

    from aalpy_extra.utils.storm_interface import (
        mdp_to_stormpy, stormpy_model_check
    )

    stormpy_model = mdp_to_stormpy(aalpy_mdp)
    result = stormpy_model_check(stormpy_model, 'Pmax=? [F "goal"]')

Note:
    stormpy is an optional dependency. Install it with::

        pip install aalpy-extra[stormpy]
"""

import os
import tempfile
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from aalpy.automata import Mdp, StochasticMealyMachine

if TYPE_CHECKING:
    import stormpy  # noqa: F401  (type-checking only; imported lazily at runtime)


# ---------------------------------------------------------------------------
# Lazy import helpers -- only trigger the stormpy import when actually called
# ---------------------------------------------------------------------------

def _get_stormpy():
    """Lazily import and return stormpy; raises a clear error if not installed."""
    try:
        import stormpy  # noqa: F811 (imported lazily)
        return stormpy
    except ImportError as e:
        raise ImportError(
            "stormpy is required for this function. "
            "Install it with:  pip install aalpy-extra[stormpy]"
        ) from e


# ---------------------------------------------------------------------------
# Model conversion
# ---------------------------------------------------------------------------

def mdp_to_stormpy(mdp: Mdp) -> "stormpy.storage.SparseMdp":
    """Convert an AALpy MDP to a stormpy SparseMdp model.

    The conversion is performed by first translating the AALpy MDP to a
    PRISM model string via :func:`aalpy.utils.mdp_2_prism_format`, and
    then parsing that string with stormpy's PRISM parser.  This reuses
    the well-tested PRISM export in AALpy and avoids fragile manual
    sparse-matrix construction.

    Args:
        mdp: AALpy Markov Decision Process.

    Returns:
        stormpy.storage.SparseMdp: Equivalent stormpy model.

    Raises:
        ImportError: If stormpy is not installed (see module docstring).
    """
    sp = _get_stormpy()

    from aalpy.utils import mdp_2_prism_format

    prism_str = mdp_2_prism_format(mdp, name="auto_mdp", output_path=None)

    # Write PRISM model to a temporary file (stormpy's parser reads files)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".prism", prefix="aalpy_extra_")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(prism_str)

        program = sp.parse_prism_program(tmp_path)
        options = sp.BuilderOptions(True, True)
        options.set_build_state_valuations()
        options.set_build_choice_labels()
        model = sp.build_sparse_model_with_options(program, options)
    finally:
        os.unlink(tmp_path)

    return model


def smm_to_stormpy_mdp(smm: StochasticMealyMachine) -> "stormpy.storage.SparseMdp":
    """Convert an AALpy Stochastic Mealy Machine to a stormpy SparseMdp model.

    The SMM is first converted to an AALpy MDP via
    :func:`aalpy.automata.StochasticMealyMachine.smm_to_mdp_conversion`,
    and then the MDP is converted to a stormpy model via
    :func:`mdp_to_stormpy`.

    Args:
        smm: AALpy Stochastic Mealy Machine.

    Returns:
        stormpy.storage.SparseMdp: Equivalent stormpy MDP model.

    Raises:
        ImportError: If stormpy is not installed.
    """
    from aalpy.automata.StochasticMealyMachine import smm_to_mdp_conversion

    mdp = smm_to_mdp_conversion(smm)
    return mdp_to_stormpy(mdp)


# ---------------------------------------------------------------------------
# Model checking
# ---------------------------------------------------------------------------

def stormpy_model_check(
    model: Union[Mdp, StochasticMealyMachine, "stormpy.storage.SparseMdp"],
    property_string: str,
    extract_scheduler: bool = False,
) -> float:
    """Model check a PCTL/PRCTL property on an AALpy or stormpy model.

    If *model* is an AALpy MDP or StochasticMealyMachine, it is first
    converted to a stormpy sparse model.

    Args:
        model: An AALpy MDP, an AALpy StochasticMealyMachine, or a
            stormpy sparse model obtained via :func:`mdp_to_stormpy`.
        property_string: A property formula in PRISM/PRCTL syntax,
            e.g. ``'Pmax=? [F "goal"]'``.
        extract_scheduler: If ``True``, the optimal scheduler is
            extracted and attached to the result object.

    Returns:
        The computed probability (or reward value) for the initial state.

    Raises:
        ImportError: If stormpy is not installed.
    """
    sp = _get_stormpy()

    # Convert AALpy models lazily
    if isinstance(model, StochasticMealyMachine):
        model = smm_to_stormpy_mdp(model)
    elif isinstance(model, Mdp):
        model = mdp_to_stormpy(model)

    properties = sp.parse_properties_without_context(property_string)
    result = sp.model_checking(model, properties[0], extract_scheduler=extract_scheduler)
    return result.at(model.initial_states[0])


def stormpy_model_check_properties(
    model: Union[Mdp, StochasticMealyMachine, "stormpy.storage.SparseMdp"],
    properties: List[str],
    extract_scheduler: bool = False,
) -> Dict[str, float]:
    """Model check multiple properties and return a dict of results.

    Args:
        model: An AALpy MDP, SMM, or a stormpy sparse model.
        properties: List of property strings in PRISM/PRCTL syntax.
        extract_scheduler: If ``True``, optimal schedulers are extracted.

    Returns:
        Dictionary mapping property index labels (``"prop1"``,
        ``"prop2"``, ...) to their computed values for the initial state.

    Raises:
        ImportError: If stormpy is not installed.
    """
    sp = _get_stormpy()

    if isinstance(model, StochasticMealyMachine):
        model = smm_to_stormpy_mdp(model)
    elif isinstance(model, Mdp):
        model = mdp_to_stormpy(model)

    results: Dict[str, float] = {}
    for i, prop_str in enumerate(properties):
        parsed = sp.parse_properties_without_context(prop_str)
        result = sp.model_checking(model, parsed[0], extract_scheduler=extract_scheduler)
        results[f"prop{i + 1}"] = result.at(model.initial_states[0])

    return results


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------

def stormpy_model_check_experiment(
    model: Union[Mdp, StochasticMealyMachine, "stormpy.storage.SparseMdp"],
    properties: List[str],
    correct_values: Optional[List[float]] = None,
    precision: int = 4,
) -> Dict:
    """Run a batch model checking experiment, optionally comparing against
    ground-truth values.

    This is the stormpy analogue of
    :func:`aalpy.utils.model_check_experiment`.

    Args:
        model: AALpy MDP, SMM, or stormpy sparse model.
        properties: List of property strings in PRISM/PRCTL syntax.
        correct_values: Optional list of correct (ground-truth) property
            values.  If provided, a ``diff_to_correct`` dict is included
            in the return value.
        precision: Decimal places to which results are rounded.

    Returns:
        A dictionary of the form::

            {
                "results": {"prop1": 0.1234, "prop2": 0.5678},
                "diff_to_correct": {"prop1": 0.0012, "prop2": 0.0034},  # optional
            }
    """
    results = stormpy_model_check_properties(model, properties)

    results_rounded = {
        key: round(val, precision) for key, val in results.items()
    }

    out: Dict = {"results": results_rounded}

    if correct_values is not None:
        diff: Dict[str, float] = {}
        for ind, val in enumerate(results.values()):
            diff[f"prop{ind + 1}"] = round(
                abs(correct_values[ind] - val), precision
            )
        out["diff_to_correct"] = diff

    return out
