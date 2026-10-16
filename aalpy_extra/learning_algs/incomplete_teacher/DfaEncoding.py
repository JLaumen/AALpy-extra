"""CNF encoding of "a DFA with a given number of states into which the observation tree maps"."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .DCNode import DCNode
from .DCValue import DCValue

# (source node index, target node index, letter index) of an observation-tree edge.
Edge = tuple[int, int, int]
# Per-state, per-letter mapping from target state to the transition variable.
TransitionVariables = list[list[dict[int, int]]]
# Target state of each (state, letter), and whether each state is accepting.
HypothesisMapping = tuple[list[list[int]], list[bool]]


class CnfBuilder:
    """Collects clauses and hands out fresh SAT variables."""

    def __init__(self) -> None:
        self.clauses: list[list[int]] = []
        self.num_vars = 0

    def new_var(self) -> int:
        self.num_vars += 1
        return self.num_vars

    def new_vars(self, count: int) -> list[int]:
        start = self.num_vars + 1
        self.num_vars += count
        return list(range(start, start + count))

    def exactly_one(self, literals: Sequence[int]) -> None:
        """At least one literal, plus pairwise (<= 5 literals) or sequential-counter at-most-one."""
        clauses = self.clauses
        clauses.append(list(literals))
        count = len(literals)
        if count <= 1:
            return
        if count <= 5:
            clauses.extend([-literals[left], -literals[right]]
                           for left in range(count) for right in range(left + 1, count))
            return
        # Sinz's sequential counter: ``prefix[i]`` holds if one of ``literals[0..i]`` does.
        prefix = self.new_vars(count - 1)
        clauses.append([-literals[0], prefix[0]])
        for index in range(1, count - 1):
            clauses.append([-prefix[index - 1], prefix[index]])
            clauses.append([-literals[index], -prefix[index - 1]])
            clauses.append([-literals[index], prefix[index]])
        clauses.append([-literals[-1], -prefix[-1]])


class DfaEncoding:
    """CNF for a DFA with ``len(transition_domains)`` states into which the observation tree maps.

    Variables, allocated in this order:

    * ``map(u, s)``: tree node ``u`` maps to state ``s``, for ``s`` in the mapping domain of ``u``;
    * ``delta(s, a, t)``: the ``a``-transition of ``s`` goes to ``t``, for ``t`` in its transition domain;
    * ``accepting(s)``: state ``s`` is accepting;
    * auxiliary variables of the at-most-one, output-signature and symmetry-breaking constraints.

    Args:
        nodes: Informative tree nodes in breadth-first order; node ``0`` is the root.
        edges: Tree edges between ``nodes``, each listed after the edge into its source.
        mapping_domains: Allowed states for each node.
        transition_domains: Allowed targets for each state and letter.
        basis_size: Number of basis states, which come first and are fixed.
        alphabet_size: Number of input letters.
    """

    def __init__(self, nodes: Sequence[DCNode], edges: Sequence[Edge], mapping_domains: Sequence[set[int]],
                 transition_domains: Sequence[Sequence[set[int]]], basis_size: int, alphabet_size: int) -> None:
        self._cnf = CnfBuilder()
        self._edges = edges
        self.number_of_states = len(transition_domains)
        self.mapping_variables = [{state: self._cnf.new_var() for state in sorted(domain)}
                                  for domain in mapping_domains]
        self.transition_variables: TransitionVariables = [
            [{target: self._cnf.new_var() for target in sorted(domain)} for domain in row]
            for row in transition_domains]
        self.output_variables = self._cnf.new_vars(self.number_of_states)
        # For :meth:`phases`: the (state, suffix index) of every signature variable, each suffix as
        # (first letter, index of the rest or -1 for the empty word), and every symmetry-breaking
        # prefix variable as (variable, state, descriptor rank).
        self._signatures: dict[int, tuple[int, int]] = {}
        self._suffixes: list[tuple[int, int]] = []
        self._first_parent_prefixes: list[tuple[int, int, int]] = []

        self._encode_unique_choices()
        self._encode_functional_simulation(edges)
        self._encode_outputs(nodes)
        self._encode_output_signatures(nodes, edges)
        self._encode_bfs_symmetry_breaking(basis_size, alphabet_size)

    @property
    def clauses(self) -> list[list[int]]:
        return self._cnf.clauses

    @property
    def num_vars(self) -> int:
        return self._cnf.num_vars

    def _encode_unique_choices(self) -> None:
        """Every node maps to exactly one state and every transition has exactly one target."""
        for mapping in self.mapping_variables:
            self._cnf.exactly_one(list(mapping.values()))
        for row in self.transition_variables:
            for transition in row:
                self._cnf.exactly_one(list(transition.values()))

    def _encode_functional_simulation(self, edges: Sequence[Edge]) -> None:
        """map(u, s) & delta(s, a, t) -> map(ua, t); without a map(ua, t) variable the transition is excluded."""
        clauses = self._cnf.clauses
        for source_index, target_index, letter_index in edges:
            target_mapping = self.mapping_variables[target_index]
            for source_state, source_literal in self.mapping_variables[source_index].items():
                clauses.extend([-source_literal, -transition_literal, target_mapping[target_state]]
                               if target_state in target_mapping else [-source_literal, -transition_literal]
                               for target_state, transition_literal
                               in self.transition_variables[source_state][letter_index].items())

    def _encode_outputs(self, nodes: Sequence[DCNode]) -> None:
        """map(u, s) -> accepting(s) == output(u) for every node with a known output."""
        clauses = self._cnf.clauses
        output_variables = self.output_variables
        for node, mapping in zip(nodes, self.mapping_variables):
            if not node.has_known_output():
                continue
            sign = 1 if node.output is DCValue.TRUE else -1
            clauses.extend([-mapping_literal, sign * output_variables[state]]
                           for state, mapping_literal in mapping.items())

    def _encode_output_signatures(self, nodes: Sequence[DCNode], edges: Sequence[Edge]) -> None:
        """Redundant: map(u, s) -> sig(s, w) == output(uw) for every node ``uw`` with a known output.

        ``sig(s, w)`` is the output of state ``s`` after the non-empty suffix ``w``; ``accepting(s)``
        plays this role for the empty suffix. Two nodes whose outputs after some suffix differ thus
        cannot share a state, which unit propagation sees in two steps instead of only once all
        transitions along the suffix are fixed. Every DFA satisfies these clauses with ``sig`` set to
        its actual outputs, so satisfiability is unchanged.

        ``sig(s, w)`` is only created when nodes with both outputs after ``w`` may map to ``s``: with
        a single output the variable would be pure and its clauses could not cause any propagation.
        """
        parents: list[tuple[int, int] | None] = [None] * len(nodes)
        for source_index, target_index, letter_index in edges:
            parents[target_index] = (source_index, letter_index)

        # Number the suffixes through a trie of reversed words: ``letter + suffixes[rest]``.
        # ``occurrences[i]`` lists (ancestor, sign) for every known node ``ancestor + suffix i``.
        suffix_index: dict[tuple[int, int], int] = {}
        suffixes: list[tuple[int, int]] = []
        occurrences: list[list[tuple[int, int]]] = []
        for node_index, node in enumerate(nodes):
            if not node.has_known_output():
                continue
            sign = 1 if node.output is DCValue.TRUE else -1
            suffix = -1  # The empty suffix.
            parent = parents[node_index]
            while parent is not None:
                ancestor, letter = parent
                key = (letter, suffix)
                next_suffix = suffix_index.get(key)
                if next_suffix is None:
                    next_suffix = suffix_index[key] = len(suffixes)
                    suffixes.append(key)
                    occurrences.append([])
                suffix = next_suffix
                occurrences[suffix].append((ancestor, sign))
                parent = parents[ancestor]
        self._suffixes = suffixes

        domain_masks = [sum(1 << state for state in mapping) for mapping in self.mapping_variables]
        cnf = self._cnf
        clauses = cnf.clauses
        for suffix, entries in enumerate(occurrences):
            accepting_mask = rejecting_mask = 0
            for node_index, sign in entries:
                if sign > 0:
                    accepting_mask |= domain_masks[node_index]
                else:
                    rejecting_mask |= domain_masks[node_index]
            conflicting = accepting_mask & rejecting_mask
            if not conflicting:
                continue
            signature: dict[int, int] = {}
            for state in range(conflicting.bit_length()):
                if conflicting >> state & 1:
                    variable = signature[state] = cnf.new_var()
                    self._signatures[variable] = (state, suffix)
            for node_index, sign in entries:
                clauses.extend([-mapping_literal, sign * signature[state]]
                               for state, mapping_literal in self.mapping_variables[node_index].items()
                               if state in signature)

    def _encode_bfs_symmetry_breaking(self, basis_size: int, alphabet_size: int) -> None:
        """Order the non-basis states by the first (parent, letter) transition that reaches them.

        Descriptors (parent, letter) are ranked as ``parent * alphabet_size + letter``. Every
        non-basis state ``q`` needs an incoming transition from a smaller state, and the smallest
        descriptor into ``q + 1`` must exceed the smallest one into ``q``. With
        ``reached(q, r)`` meaning that some descriptor ``<= r`` of a parent below ``q`` enters
        ``q``, the order is ``reached(q + 1, r) -> reached(q, r - 1)`` for all ``r``.
        """
        cnf = self._cnf
        clauses = cnf.clauses
        number_of_states = self.number_of_states
        if basis_size >= number_of_states or alphabet_size == 0:
            return

        # ``reached[q][r]``: a literal for reached(q, r), or ``None`` when it is false.
        reached: dict[int, list[int | None]] = {}
        for state in range(basis_size, number_of_states):
            prefix: list[int | None] = []
            previous: int | None = None
            for parent in range(state):
                for letter in range(alphabet_size):
                    incoming = self.transition_variables[parent][letter].get(state)
                    if incoming is not None:
                        if previous is None:
                            previous = incoming
                        else:
                            current = cnf.new_var()
                            self._first_parent_prefixes.append((current, state, parent * alphabet_size + letter))
                            clauses.append([-current, previous, incoming])
                            clauses.append([-previous, current])
                            clauses.append([-incoming, current])
                            previous = current
                    prefix.append(previous)
            reached[state] = prefix
            # Without any possible incoming transition the empty clause makes the instance unsatisfiable.
            clauses.append([] if previous is None else [previous])

        for state in range(basis_size, number_of_states - 1):
            current, following = reached[state], reached[state + 1]
            # Ranks of parent ``state`` are not in ``current``; there, reached(state, r - 1) holds anyway.
            for rank in range(len(current) + 1):
                later = following[rank]
                if later is None or (rank > 0 and later == following[rank - 1]):
                    continue  # False, or implied by the clause for ``rank - 1``.
                earlier = current[rank - 1] if rank > 0 else None
                clauses.append([-later] if earlier is None else [-later, earlier])

    def phases(self, hint: HypothesisMapping) -> list[int]:
        """Literals describing ``hint``, a DFA over the first ``len(hint[0])`` states, as solver phases.

        Mapping, signature and symmetry-breaking variables follow the run of ``hint`` on the tree;
        variables of states beyond the hint, and transitions into them, are set to false.
        """
        transition_mapping, output_mapping = hint
        hinted_states = len(transition_mapping)
        literals: list[int] = []
        for state, row in enumerate(self.transition_variables[:hinted_states]):
            for letter, transition in enumerate(row):
                target = transition_mapping[state][letter]
                literals.extend(literal if successor == target else -literal
                                for successor, literal in transition.items())
        for row in self.transition_variables[hinted_states:]:
            for transition in row:
                literals.extend(-literal for literal in transition.values())
        literals.extend(literal if accepting else -literal
                        for literal, accepting in zip(self.output_variables, output_mapping))

        run = [0] * len(self.mapping_variables)
        for source_index, target_index, letter_index in self._edges:
            run[target_index] = transition_mapping[run[source_index]][letter_index]
        for mapping, state in zip(self.mapping_variables, run):
            literals.extend(literal if candidate == state else -literal for candidate, literal in mapping.items())

        # Output of every hinted state after every suffix, in suffix order (rest suffixes come first).
        suffix_outputs: list[list[bool]] = []
        for letter, rest in self._suffixes:
            rest_outputs = output_mapping if rest < 0 else suffix_outputs[rest]
            suffix_outputs.append([rest_outputs[transition_mapping[state][letter]] for state in range(hinted_states)])
        for variable, (state, suffix) in self._signatures.items():
            if state < hinted_states:
                literals.append(variable if suffix_outputs[suffix][state] else -variable)

        # reached(q, r) holds from the first descriptor rank of a hinted transition into ``q``.
        first_rank: dict[int, int] = {}
        for parent, targets in enumerate(transition_mapping):
            for letter, target in enumerate(targets):
                if target > parent and target not in first_rank:
                    first_rank[target] = parent * len(targets) + letter
        for variable, state, rank in self._first_parent_prefixes:
            literals.append(variable if first_rank.get(state, rank + 1) <= rank else -variable)
        return literals

    def decode(self, model: Iterable[int]) -> HypothesisMapping:
        """Read the transition targets and accepting states from a satisfying assignment."""
        true_literals = set(model)
        transition_mapping: list[list[int]] = []
        for state, row in enumerate(self.transition_variables):
            targets = []
            for letter, transition in enumerate(row):
                target = next((target for target, literal in transition.items() if literal in true_literals), None)
                if target is None:
                    raise RuntimeError(f"No transition selected for delta({state}, {letter}).")
                targets.append(target)
            transition_mapping.append(targets)
        return transition_mapping, [literal in true_literals for literal in self.output_variables]
