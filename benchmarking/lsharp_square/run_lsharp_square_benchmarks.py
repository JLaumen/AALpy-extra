"""Run the L#-square benchmarks and write one CSV row of measurements per benchmark.

Usage::

    python benchmarking/lsharp_square/run_lsharp_square_benchmarks.py [-j N] [-t TIMEOUT] [--until-test N]

Each benchmark runs in its own process, which is stopped after ``TIMEOUT`` seconds when ``-t`` is given.
Results are written to ``benchmarking/lsharp_square/results/lsharp_square_<random id>.csv``.
"""

import argparse
import concurrent.futures
import functools
import logging
import multiprocessing
import re
import secrets
import sys
from multiprocessing.connection import Connection
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

# Keys of the ``run_lsharp_square`` info dict written after the file name, in CSV column order.
INFO_COLUMNS = ("successful", "timed_out", "learning_rounds", "automaton_size", "learning_time", "solver_time", "eq_oracle_time",
                "total_time", "membership_queries", "validity_query", "nodes", "informative_nodes", "sul_steps",
                "queries_eq_oracle", "steps_eq_oracle")
CSV_HEADER = ",".join(("file_name",) + INFO_COLUMNS) + "\n"
# Values of a benchmark stopped at the timeout; its measurements are left empty.
TIMED_OUT_INFO = {column: "" for column in INFO_COLUMNS} | {"successful": False, "timed_out": True}

# Benchmarks run in child processes, which are started from worker threads when running in parallel.
# Forking a multi-threaded process is unsafe, so children start from a fork server where available.
MP_CONTEXT = multiprocessing.get_context(
    "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn")

# Configured at import time so that child processes, which re-import this
# module, log their progress too.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")


# ----------------------------------------------------------------------
# Benchmark parsing
# ----------------------------------------------------------------------

def get_possible_words(prefix: str, suffix: str, alphabet: list[str]) -> list[tuple[str, ...]]:
    """Expand ``X`` wildcards in a compact benchmark word."""
    if not suffix:
        return [tuple(prefix)]

    if suffix[0] == "X":
        return [word for letter in alphabet for word in get_possible_words(prefix + letter, suffix[1:], alphabet)]

    return get_possible_words(prefix + suffix[0], suffix[1:], alphabet)


def parse_file(filename: Path, horizon: int | None = None) -> tuple[list[tuple[tuple[str, ...], bool]], list[str]]:
    """Read a benchmark file and return observed words and alphabet.

    Each line holds ``<word>,<label>``, where ``<label>`` is ``+`` for accepted
    words. Words over ``0``/``1`` may use ``X`` as a wildcard for either
    symbol; other words are ``;``-separated symbols. The alphabet lists the
    symbols in order of first appearance.
    """
    alphabet = ["0", "1"]
    known_words: list[tuple[tuple[str, ...], bool]] = []
    observed_alphabet: list[str] = []
    observed_symbols: set[str] = set()

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
                        if symbol not in observed_symbols:
                            observed_symbols.add(symbol)
                            observed_alphabet.append(symbol)

    return known_words, observed_alphabet


# ----------------------------------------------------------------------
# Running benchmarks
# ----------------------------------------------------------------------

def run_test_case(filename: Path) -> dict[str, Any]:
    """Run one benchmark and return its measurements."""
    data, alphabet = parse_file(filename)
    sul = IncompleteSUL(data.copy())
    eq_oracle = FiniteDataOracle(alphabet, sul, data.copy())

    learned_dfa, info = run_lsharp_square(alphabet, sul, eq_oracle, return_data=True)

    info["successful"] = learned_dfa is not None and eq_oracle.find_cex(learned_dfa) is None
    info["timed_out"] = False
    return info


def format_row(filename: Path, info: dict[str, Any]) -> str:
    """Format the measurements of one benchmark as a CSV row."""
    values = [filename.name] + [info[column] for column in INFO_COLUMNS]
    return ",".join(map(str, values)) + "\n"


def process_file(filename: Path, connection: Connection) -> None:
    """Run one benchmark in a child process and send its CSV row to the parent."""
    logging.info("Testing %s", filename.name)
    connection.send(format_row(filename, run_test_case(filename)))
    connection.close()


def run_benchmark(filename: Path, timeout: float | None) -> str:
    """Run one benchmark in a child process and return its CSV row.

    The child is killed if it has not finished after ``timeout`` seconds, and
    the row then records the benchmark as timed out. Without a timeout, the
    benchmark runs until it finishes.
    """
    receiver, sender = MP_CONTEXT.Pipe(duplex=False)
    process = MP_CONTEXT.Process(target=process_file, args=(filename, sender))
    process.start()
    sender.close()  # The child now holds the only sending end, so the pipe ends when the child exits.

    finished = False
    try:
        finished = receiver.poll(timeout)
        row = receiver.recv() if finished else None
    except EOFError:
        row = None  # The child exited without sending a row.
    finally:
        if not finished:
            process.kill()  # Timed out, or waiting was interrupted (e.g. by Ctrl-C).
        process.join()
        receiver.close()

    if not finished:
        logging.warning("%s timed out after %g seconds", filename.name, timeout)
        return format_row(filename, TIMED_OUT_INFO)
    if row is None:
        raise RuntimeError(f"Benchmark {filename.name} exited with code {process.exitcode} without a result")
    return row


def select_benchmarks(until_test: int | None) -> list[Path]:
    """Return the sorted benchmark files, optionally limited to ``randm04`` through ``randm<until_test>``."""
    filenames = sorted(BENCHMARKS_PATH.iterdir())
    if until_test is not None:
        filenames = [filename for filename in filenames
                     if (match := TEST_NUMBER.match(filename.name)) and int(match.group("number")) <= until_test]
    return filenames


def run_test_cases(jobs: int | None, until_test: int | None, timeout: float | None) -> None:
    """Run selected benchmark files, one or ``jobs`` at a time, and write their measurements."""
    filenames = select_benchmarks(until_test)
    run = functools.partial(run_benchmark, timeout=timeout)

    if jobs is None:
        rows = [run(filename) for filename in filenames]
        execution = "sequentially"
    else:
        # Each worker thread only waits for the child process running its benchmark.
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=jobs)
        try:
            rows = list(executor.map(run, filenames))  # Rows keep the input order.
        finally:
            executor.shutdown(cancel_futures=True)  # Start no further benchmarks after a failure or Ctrl-C.
        execution = f"in parallel with {jobs} worker(s)"

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_PATH / f"lsharp_square_{secrets.token_hex(8)}.csv"
    with output_path.open("w") as output:
        output.write(CSV_HEADER)
        output.writelines(rows)

    logging.info("Completed %d benchmark(s) %s; results written to %s", len(filenames), execution, output_path)


def main(jobs: int | None = None, until_test: int | None = None, timeout: float | None = None) -> None:
    """Run the selected benchmarks."""
    run_test_cases(jobs, until_test, timeout)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the L#-square benchmarks.")
    parser.add_argument("-j", "--jobs", type=int, default=None, metavar="N",
                        help="Run N benchmarks in parallel. Without -j, run them one at a time.")
    parser.add_argument("-t", "--timeout", type=float, default=None, metavar="TIMEOUT",
                        help="Stop each L#-square run after TIMEOUT seconds and record it as timed out. "
                             "Without -t, runs have no time limit.")
    parser.add_argument("--until-test", type=int, choices=range(4, 24), metavar="N",
                        help="Run tests randm04 through randmN (inclusive). Default: all tests.")
    args = parser.parse_args()

    if args.jobs is not None and args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


# Child processes re-import this module under the forkserver and spawn start methods, so the
# benchmarks must only start behind this guard.
if __name__ == "__main__":
    arguments = _parse_args()
    main(jobs=arguments.jobs, until_test=arguments.until_test, timeout=arguments.timeout)
