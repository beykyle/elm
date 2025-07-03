"""Run walkers in parallel using MPI."""

import argparse
import lzma
import os
import sys

import dill as pickle
import numpy as np
from mpi4py import MPI


def main():
    """Main function to run the walkers in parallel, starting with state from a single
    walker provided as input"""

    parser = argparse.ArgumentParser(description="Run a parallel walker simulation.")
    parser.add_argument("--input", type=str, help="Input file path for walker data.")
    parser.add_argument(
        "--output",
        type=str,
        default="./",
        help="Output directory for results",
    )
    parser.add_argument(
        "--steps", type=int, default=10000, help="Number of walking steps."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1000, help="Batch size for walker."
    )
    parser.add_argument(
        "--burnin", type=int, default=1000, help="Number of burn-in steps."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode.")

    args = parser.parse_args()

    # Verify that the input file exists
    if not os.path.isfile(args.input):
        print(f"Error: The input file '{args.input}' does not exist.")
        sys.exit(1)

    # Ensure output directory exists
    if not os.path.isdir(args.output):
        print(f"Error: The output directory '{args.output}' does not exist.")
        sys.exit(1)

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Read input file
    try:
        with lzma.open(args.input, "rb") as f:
            walker = pickle.load(f)
    except Exception as e:
        print(
            f"Error: Failed to read input file '{args.input}' on rank {rank}. Exception: {e}"
        )
        sys.exit(1)

    # Set RNG seed for each walker
    try:
        walker.rng = np.random.default_rng(rank)
    except Exception as e:
        print(
            f"Error: Failed to initialize random number generator on rank {rank}. Exception: {e}"
        )
        sys.exit(1)

    # Run the walker for the specified number of steps
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

    # Write walker object to disk from each rank if requested
    try:
        with lzma.open(f"{args.output}/walkers_{rank}.pkl.xz", "wb") as f:
            pickle.dump(walker, f)
    except Exception as e:
        print(f"Error: Failed to write walker to disk on rank {rank}. Exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
