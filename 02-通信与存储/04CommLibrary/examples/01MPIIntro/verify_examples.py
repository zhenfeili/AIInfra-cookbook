"""Run the chapter's Python examples; optionally compile and cross-check C versions."""
import argparse
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
CHAPTER = HERE.parents[1] / "01MPIIntro.md"
CASES = [
    ("mpi_hello", 4, None),
    ("mpi_p2p", 2, ["rank 1 received: [1.0, 2.0, 3.0]"]),
    ("mpi_exchange", 2, ["rank 0 received [10, 11], local=100",
                          "rank 1 received [0, 1], local=101"]),
    ("mpi_collectives", 4, ["gathered: 0 10 20 30 40 50 60 70"]),
    ("mpi_gradient", 2, ["rank 0: sum=[4.0, 8.0], weights=[0.8, 0.6]",
                          "rank 1: sum=[4.0, 8.0], weights=[0.8, 0.6]"]),
]


def run_and_check_state(name):
    """Execute an unchanged example on each MPI rank, then inspect its final arrays."""
    import runpy

    import numpy as np

    state = runpy.run_path(str(HERE / f"{name}.py"), run_name="__main__")
    rank = state["rank"]

    def check(key, expected, dtype):
        actual = state[key]
        if not isinstance(actual, np.ndarray) or actual.dtype != np.dtype(dtype):
            raise AssertionError(f"{name} rank {rank}: wrong array type for {key}")
        if actual.shape != (len(expected),) or not actual.flags.c_contiguous:
            raise AssertionError(f"{name} rank {rank}: wrong layout for {key}")
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    if name == "mpi_p2p":
        check("x", [1, 2, 3], np.float32)
    elif name == "mpi_exchange":
        check("outgoing", [rank * 10, rank * 10 + 1], np.intc)
        peer = 1 - rank
        check("incoming", [peer * 10, peer * 10 + 1], np.intc)
    elif name == "mpi_collectives":
        check("factor", [10], np.intc)
        check("local", [rank * 20, rank * 20 + 10], np.intc)
        if rank == 0:
            check("input_data", list(range(8)), np.intc)
            check("output", list(range(0, 80, 10)), np.intc)
        elif state["input_data"] is not None or state["output"] is not None:
            raise AssertionError("Non-root must not allocate full input/output arrays")
    elif name == "mpi_gradient":
        check("grad", [1 + 2 * rank, 3 + 2 * rank], np.float32)
        check("grad_sum", [4, 8], np.float32)
        check("grad_mean", [2, 4], np.float32)
        check("weights", [0.8, 0.6], np.float32)


def run(command):
    """Bound execution time and clean up the local process group on timeout."""
    with subprocess.Popen(command, cwd=HERE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True,
                          start_new_session=True) as process:
        try:
            out, err = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise RuntimeError(f"Timed out: {command}") from None
        return process.returncode, out, err


def check_output(command, name, expected):
    rc, out, err = run(command)
    if rc:
        raise RuntimeError(f"{name}: run failed\n{out}{err}")
    lines = out.strip().splitlines()
    if expected is None:
        ranks = [re.fullmatch(r"Hello: rank (\d+)/4 on .+", line) for line in lines]
        valid = all(ranks) and sorted(int(m[1]) for m in ranks) == [0, 1, 2, 3]
    else:
        valid = sorted(lines) == sorted(expected)
    if not valid:
        raise RuntimeError(f"{name}: unexpected output\n{out}{err}")


def check_invalid_count(launcher, command, name, count):
    rc, out, err = run(launcher + ["-n", "1"] + command)
    if rc == 0 or f"Run with exactly {count} processes." not in out + err:
        raise RuntimeError(f"{name}: invalid process count was not rejected\n{out}{err}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-c", action="store_true", help="also compile and test C examples")
    args = parser.parse_args()
    chapter = CHAPTER.read_text(encoding="utf-8")
    blocks = re.findall(r"^```python\n(.*?)^```", chapter, re.MULTILINE | re.DOTALL)
    if len(blocks) != len(CASES):
        raise RuntimeError("The chapter must contain exactly five complete Python examples.")
    launcher = [os.environ.get("MPIEXEC", "mpiexec")]
    launcher += shlex.split(os.environ.get("MPIEXEC_FLAGS", ""))
    for (name, count, expected), block in zip(CASES, blocks):
        source = HERE / f"{name}.py"
        if source.read_text(encoding="utf-8").strip() != block.strip():
            raise RuntimeError(f"{source.name} differs from its chapter code block.")
        # The worker executes the actual source and verifies unrounded array values.
        worker = [sys.executable, "-B", "-c",
                  "from verify_examples import run_and_check_state; "
                  "import sys; run_and_check_state(sys.argv[1])", name]
        check_output(launcher + ["-n", str(count)] + worker, name, expected)
        print(f"PASS {name}.py: chapter matches; expected output; array state")
        if expected is not None:
            check_invalid_count(launcher, [sys.executable, str(source)], name, count)
            print(f"PASS {name}.py: wrong process count exits without hanging")
    print("All 5 Python examples and 4 invalid-count cases passed.")

    if args.with_c:
        compiler = os.environ.get("MPICC", "mpicc")
        with tempfile.TemporaryDirectory(prefix="mpi-intro-c-") as temp:
            for name, count, expected in CASES:
                executable = str(Path(temp) / name)
                rc, out, err = run([compiler, "-std=c11", "-Wall", "-Wextra", "-Werror",
                                    "-pedantic", str(HERE / f"{name}.c"), "-o", executable])
                if rc:
                    raise RuntimeError(f"{name}: compile failed\n{out}{err}")
                check_output(launcher + ["-n", str(count), executable], name, expected)
                if expected is not None:
                    check_invalid_count(launcher, [executable], name, count)
                print(f"PASS {name}.c: strict compile; expected output; process count")
        print("All 5 C examples and 4 invalid-count cases passed.")


if __name__ == "__main__":
    main()
