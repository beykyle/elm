"""Run walkers in parallel using MPI."""

import argparse
import lzma
import sys
from pathlib import Path

import dill as pickle
import numpy as np
from mpi4py import MPI


def main():
    """Read walkers in parallel, advance thier state, and overwrite them """

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    parser = argparse.ArgumentParser(description="Advance state of provided walkers")
    parser.add_argument(
        "--input",
        type=str,
        help="Input directory expected to containt one file called walker_{rank}.pkl.xz"
            " for each rank, each one being an LZMA-compressed walker object.",
    )
    parser.add_argument(
        "--steps", type=int,  help="Number of walking steps."
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size for walker."
    )
    parser.add_argument(
        "--burnin", type=int,  help="Number of burn-in steps."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode.")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_dir():
        print(f"Error: The input directory '{args.input}' does not exist.")
        sys.exit(1)

    # Read input file
    input_file = input_path / f"walkers_{rank}.pkl.xz"
    try:
        with lzma.open(input_file, "rb") as f:
            walker = pickle.load(f)
    except Exception as e:
        print(
            f"Error: Failed to read input file '{ininput_file}' on rank {rank}. Exception: {e}"
        )
        sys.exit(1)

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run the walker for the specified number of steps
    print(f"walking on rank {rank}...")
    try:
        acc_frac = walker.walk(
            n_steps=args.steps,
            batch_size=args.batch_size,
            burnin=args.burnin,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Error: Walker run failed on rank {rank}. Exception: {e}")
        sys.exit(1)

    # Gather acceptance fractions at rank 0
    try:
        acc_fracs = comm.gather(acc_frac, root=0)
    except Exception as e:
        print(f"Error: MPI gather operation failed on rank {rank}. Exception: {e}")
        sys.exit(1)

    # Rank 0 prints acceptance fractions
    if rank == 0 and acc_fracs is not None:
        acs = ", ".join([f"{a:1.2f}" for a in acc_fracs])
        print(f"acceptance fractions: {acs}")

    # Write walker object to disk from each rank
    try:
        with lzma.open(output_path / f"walker_{rank}.pkl.xz", "wb") as f:
            pickle.dump(walker, f)
    except Exception as e:
        print(f"Error: Failed to write walker to disk on rank {rank}. Exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
