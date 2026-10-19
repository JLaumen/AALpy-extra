"""Examples of using AALpy-extra.

Every example is a function without arguments that is registered in
``EXAMPLES``. To add an example, write a function (importing what it needs
inside the function, so that optional dependencies are only required for the
examples that use them) and add it to ``EXAMPLES``.

Usage::

    python Examples.py                 # run all examples
    python Examples.py lsharp_square   # run the named examples only
"""

import sys
from collections.abc import Callable
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data"


# ----------------------------------------------------------------------
# L#-square
# ----------------------------------------------------------------------

def _accepts(dfa, word: tuple[str, ...]) -> bool:
    """Run ``word`` on ``dfa`` and return whether it ends in an accepting state."""
    dfa.reset_to_initial()
    for symbol in word:
        dfa.step(symbol)
    return dfa.step(None)


def _report_lsharp_square(dfa, info: dict, alphabet: list[str], data: list[tuple[tuple[str, ...], bool]]) -> None:
    """Print the statistics and transitions of a learned DFA and check it against the known words."""
    print(f"Learned a DFA with {dfa.size} states in {info['learning_rounds']} rounds")
    print(f"Membership queries: {info['membership_queries']}, equivalence queries: {info['validity_query']}")
    print(f"Total time: {info['total_time']:.2f}s (SAT solver: {info['solver_time']:.2f}s)")
    print("Transitions:")
    for state in dfa.states:
        kind = "accepting" if state.is_accepting else "rejecting"
        moves = ", ".join(f"{symbol} -> {state.transitions[symbol].state_id}" for symbol in alphabet)
        print(f"  {state.state_id} ({kind}): {moves}")

    # The DFA must agree with every known answer.
    assert all(_accepts(dfa, word) == expected for word, expected in data)


def example_lsharp_square() -> None:
    """Learn a DFA from an incomplete teacher with L#-square.

    Requires the optional SAT dependency: ``pip install ".[python-sat]"``.

    L#-square learns a DFA when the teacher cannot answer every query. For
    some words the teacher does not know whether they are accepted; such
    answers are *don't-care* (DC) values. The learner searches (with a SAT
    solver) for a small DFA that agrees with every answer that *is* known.

    ``run_lsharp_square(alphabet, sul, eq_oracle, return_data=False, assume_prefix_closed=True)``
    takes:

    * ``alphabet``: list of input symbols, at least two. Symbols may be any
      hashable object (strings, ints, ...). The SUL and the oracle must use
      the same symbols.

    * ``sul`` (system under learning, i.e. the membership oracle): any object
      with

      - ``query(word) -> DCValue``. ``word`` is a *tuple* of input symbols
        (the empty tuple ``()`` is the empty word and is always queried
        first). The result must be a ``DCValue`` member:

          ``DCValue.TRUE``   the word is accepted
          ``DCValue.FALSE``  the word is rejected
          ``DCValue.DC``     the teacher does not know (don't care)

        Plain ``bool`` or ``None`` are *not* accepted: the learner calls
        ``.is_known()`` on the result. Compare ``DCValue`` members with
        ``is``; ``==`` treats DC as equal to everything and is not
        transitive.
      - ``num_queries``: int counter of ``query`` calls.
      - ``num_steps``: int counter of the total length of all queried words.

      ``aalpy_extra.SULs.IncompleteSUL`` is a ready-made SUL. It is backed by
      a list of ``(word, accepted)`` pairs and answers DC for any other word.

    * ``eq_oracle`` (equivalence oracle): an ``aalpy.base.Oracle`` subclass
      (or any object with the same interface) with

      - ``find_cex(hypothesis: Dfa) -> tuple | None``. It receives the
        current hypothesis, an ``aalpy.automata.Dfa``, and returns a
        counterexample, i.e. an input word (tuple or list of symbols) on
        which the hypothesis disagrees with the target, or ``None`` if it
        finds no disagreement. Learning only ends when it returns ``None``.
        The learner queries the SUL for the counterexample itself, so no
        output value is returned. A counterexample the SUL answers with DC
        adds no information, so only return words with a known answer.
        To run a word on a hypothesis: ``hypothesis.reset_to_initial()``,
        ``hypothesis.step(symbol)`` for each symbol, and then
        ``hypothesis.step(None)`` gives the boolean output of the state
        reached.
      - ``num_queries`` and ``num_steps`` counters.

      ``aalpy_extra.oracles.FiniteDataOracle`` is a ready-made oracle. It
      checks the hypothesis against a list of ``(word, accepted)`` pairs and
      returns the first word on which they disagree.

    Options:

    * ``return_data``: if ``False`` (default) only the learned ``Dfa`` is
      returned. If ``True``, a ``(Dfa, info)`` tuple is returned, with
      ``info`` holding ``learning_rounds``, ``automaton_size``,
      ``learning_time``, ``solver_time``, ``eq_oracle_time``, ``total_time``,
      ``membership_queries``, ``validity_query`` (number of equivalence
      queries), ``nodes``, ``informative_nodes`` (observation-tree size),
      ``sul_steps``, ``queries_eq_oracle`` and ``steps_eq_oracle``.

    * ``assume_prefix_closed``: if ``True`` (default), a DC answer for a word
      is assumed to hold for all its extensions, so the learner does not
      query them. This saves queries when the teacher's knowledge is
      prefix-closed. Set it to ``False`` if a word can be DC while some of
      its extensions are known.

    The learner logs through the ``logging`` module; enable ``logging.DEBUG``
    to follow it.
    """
    import random
    from itertools import product

    from aalpy_extra.SULs.IncompleteSUL import IncompleteSUL
    from aalpy_extra.learning_algs.incomplete_teacher.LSharpSquare import run_lsharp_square
    from aalpy_extra.oracles.FiniteDataOracle import FiniteDataOracle

    alphabet = ["a", "b"]

    def target(word: tuple[str, ...]) -> bool:
        """Hidden target language: the number of ``a`` symbols is a multiple of 3 (needs 3 states)."""
        return word.count("a") % 3 == 0

    # The teacher knows the label of a few random words of length up to 6 and
    # of all their prefixes. Every other word is a don't-care for the SUL.
    candidates = [word for length in range(7) for word in product(alphabet, repeat=length)]
    known = {word[:length] for word in random.Random(1).sample(candidates, 12) for length in range(len(word) + 1)}
    data = [(word, target(word)) for word in sorted(known, key=lambda word: (len(word), word))]

    # The known words are prefix-closed, so a DC answer holds for all
    # extensions of the word too. The learner can then skip querying them.
    sul = IncompleteSUL(data)
    eq_oracle = FiniteDataOracle(alphabet, sul, data)

    dfa, info = run_lsharp_square(alphabet, sul, eq_oracle, return_data=True, assume_prefix_closed=True)

    print(f"Known words: {len(data)} of {len(candidates)}")
    _report_lsharp_square(dfa, info, alphabet, data)


def example_lsharp_square_from_file() -> None:
    """Learn a 9-state DFA with L#-square from data loaded from a file.

    ``data/lsharp_square_9_states.csv`` holds one known word per line as
    ``<word>,<label>``, where ``<word>`` is a string of ``0`` and ``1``
    symbols (empty for the empty word) and ``<label>`` is ``+`` for accepted
    words and ``-`` for rejected ones. All other words are don't-care. The
    data is far from complete, so the first hypotheses are wrong and the
    equivalence oracle returns counterexamples: learning takes several
    rounds.

    Requires the optional SAT dependency: ``pip install ".[python-sat]"``.
    """
    from aalpy_extra.SULs.IncompleteSUL import IncompleteSUL
    from aalpy_extra.learning_algs.incomplete_teacher.LSharpSquare import run_lsharp_square
    from aalpy_extra.oracles.FiniteDataOracle import FiniteDataOracle

    alphabet = ["0", "1"]

    data = []
    with (DATA_PATH / "lsharp_square_9_states.csv").open() as file:
        for line in file:
            if line.strip():
                word, label = line.strip().rsplit(",", maxsplit=1)
                data.append((tuple(word), label == "+"))

    sul = IncompleteSUL(data)
    eq_oracle = FiniteDataOracle(alphabet, sul, data)

    # Assume prefix-closure (the default): an unknown word has only unknown extensions.
    dfa, info = run_lsharp_square(alphabet, sul, eq_oracle, return_data=True)

    print(f"Known words: {len(data)}")
    _report_lsharp_square(dfa, info, alphabet, data)


# ----------------------------------------------------------------------
# Registry and entry point
# ----------------------------------------------------------------------

EXAMPLES: dict[str, Callable[[], None]] = {
    "lsharp_square": example_lsharp_square,
    "lsharp_square_from_file": example_lsharp_square_from_file,
}


def main(names: list[str]) -> None:
    """Run the examples called ``names``, or all examples if ``names`` is empty."""
    unknown = [name for name in names if name not in EXAMPLES]
    if unknown:
        sys.exit(f"Unknown example(s): {', '.join(unknown)}. Available: {', '.join(EXAMPLES)}")

    for name in names or EXAMPLES:
        print(f"== {name} ==")
        EXAMPLES[name]()
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
