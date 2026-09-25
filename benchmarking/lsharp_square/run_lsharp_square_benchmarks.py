import argparse
import concurrent.futures
import logging
import re
import secrets
import sys
from multiprocessing.spawn import freeze_support
from pathlib import Path
from typing import Any

# The benchmark runner is kept outside the package and may be run directly
# from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from aalpy_extra.SULs.IncompleteSUL import IncompleteSUL
from aalpy_extra.learning_algs.incomplete_teacher.LSharpSquare import run_lsharp_square
from aalpy_extra.oracles.FiniteDataOracle import FiniteDataOracle

BENCHMARKS_PATH = PROJECT_ROOT / "benchmarking" / "lsharp_square" / "benchmarks" / "all"
RESULTS_PATH = PROJECT_ROOT / "benchmarking" / "lsharp_square" / "results"
TEST_NUMBER = re.compile(r"^randm(?P<number>\d{2})")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")


def get_possible_words(prefix: str, suffix: str, alphabet: list[str]) -> list[tuple[str, ...]]:
    """Expand ``X`` wildcards in a compact benchmark word."""
    if not suffix:
        return [tuple(prefix)]

    if suffix[0] == "X":
        return [word for letter in alphabet for word in get_possible_words(prefix + letter, suffix[1:], alphabet)]

    return get_possible_words(prefix + suffix[0], suffix[1:], alphabet)


def parse_file(filename: Path, horizon: int | None = None) -> tuple[list[tuple[tuple[str, ...], bool]], list[str]]:
    """Read a benchmark file and return observed words and alphabet."""
    alphabet = ["0", "1"]
    known_words: list[tuple[tuple[str, ...], bool]] = []
    observed_alphabet: list[str] = []

    with filename.open() as benchmark:
        for line in benchmark:
            line = line.strip()
            if not line:
                continue

            input_text, output_text = line.rsplit(",", maxsplit=1)
            output = output_text.strip() == "+"

            if all(symbol in {"0", "1", "X"} for symbol in input_text):
                words = get_possible_words("", input_text, alphabet)
            else:
                words = [tuple(input_text.split(";"))]

            for word in words:
                if horizon is None or len(word) <= horizon:
                    known_words.append((word, output))
                    for symbol in word:
                        if symbol not in observed_alphabet:
                            observed_alphabet.append(symbol)

    return known_words, observed_alphabet


def run_test_case(filename: Path, replace_basis: bool) -> dict[str, Any]:
    """Run one benchmark and return its measurements."""
    data, alphabet = parse_file(filename)
    sul = IncompleteSUL(data.copy())
    eq_oracle = FiniteDataOracle(alphabet, sul, data.copy())

    learned_dfa, info = run_lsharp_square(alphabet, sul, eq_oracle, return_data=True, replace_basis=replace_basis, )

    info["successful"] = learned_dfa is not None and eq_oracle.find_cex(learned_dfa) is None
    return info


def process_file(filename: Path, replace_basis: bool) -> str:
    """Run one benchmark and format its measurements as a CSV row."""
    logging.info("Testing %s", filename.name)
    info = run_test_case(filename, replace_basis)
    values = [filename.name, info["successful"], info["learning_rounds"], info["automaton_size"], info["learning_time"],
        info["solver_time"], info["eq_oracle_time"], info["total_time"], info["membership_queries"],
        info["validity_query"], info["nodes"], info["informative_nodes"], info["sul_steps"], info["queries_eq_oracle"],
        info["steps_eq_oracle"], ]
    return ",".join(map(str, values)) + "\n"


CSV_HEADER = ("file_name,successful,learning_rounds,automaton_size,learning_time,"
              "solver_time,eq_oracle_time,total_time,membership_queries,validity_query,"
              "nodes,informative_nodes,sul_steps,queries_eq_oracle,steps_eq_oracle\n")


def run_test_cases_sequential(filenames: list[Path], replace_basis: bool) -> list[str]:
    """Run benchmark files sequentially in the current process."""
    return [process_file(filename, replace_basis) for filename in filenames]


def run_test_cases_parallel(filenames: list[Path], replace_basis: bool, jobs: int) -> list[str]:
    """Run benchmark files in worker processes."""
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
        return list(executor.map(process_file, filenames, [replace_basis] * len(filenames)))


def run_test_cases(replace_basis: bool, jobs: int | None, until_test: int | None) -> None:
    """Run selected benchmark files and write their measurements."""
    filenames = sorted(BENCHMARKS_PATH.iterdir())
    if until_test is not None:
        filenames = [filename for filename in filenames if
            (match := TEST_NUMBER.match(filename.name)) and int(match.group("number")) <= until_test]

    if jobs is None:
        rows = run_test_cases_sequential(filenames, replace_basis)
        execution = "sequentially"
    else:
        rows = run_test_cases_parallel(filenames, replace_basis, jobs)
        execution = f"in parallel with {jobs} worker(s)"

    random = secrets.token_hex(8)

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_PATH / f"lsharp_square_r{replace_basis}_{random}.csv"
    with output_path.open("w") as output:
        output.write(CSV_HEADER)
        output.writelines(rows)

    logging.info("Completed %d benchmark(s) %s; results written to %s", len(filenames), execution, output_path)


def main(replace_basis: bool = False, jobs: int | None = None, until_test: int | None = None) -> None:
    """Parse benchmark options and run the selected tests."""
    run_test_cases(replace_basis, jobs, until_test)


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="Run the L#-square benchmarks.")
    parser.add_argument("-r", "--replace-basis", action="store_true",
                        help="Allow the observation tree to replace its basis.")
    parser.add_argument("-j", "--jobs", type=int, default=None, metavar="N",
                        help="Run in parallel with N worker processes. Without -j, run sequentially.")
    parser.add_argument("--until-test", type=int, choices=range(4, 24), metavar="N",
                        help="Run tests randm04 through randmN (inclusive). Default: all tests.")
    args = parser.parse_args()

    if args.jobs is not None and args.jobs < 1:
        parser.error("--jobs must be at least 1")

    main(replace_basis=args.replace_basis, jobs=args.jobs, until_test=args.until_test)
