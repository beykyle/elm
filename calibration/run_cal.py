#!/usr/bin/env python3

import lzma
import dill as pickle
from mpi4py import MPI
import argparse
import numpy as np

gather = False


def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run a parallel walker simulation.")
    parser.add_argument("--input", type=str, help="Input file path for walker data.")
    parser.add_argument(
        "--output",
        type=str,
        default="./all_chain_walkers.xz",
        help="Output file path for results.",
    )
    parser.add_argument(
        "--steps", type=int, default=10000, help="Number of walking steps."
    )
    parser.add_argument(
        "--burnin", type=int, default=1000, help="Number of burn-in steps."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode.")

    args = parser.parse_args()

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # All ranks just read the input file
    # rather than broadcasting
    with lzma.open(args.input, "rb") as f:
        walker = pickle.load(f)

    # set rng seed for each walker
    walker.rng = np.random.default_rng(rank)

    # Each process performs its own independent walk
    acc_frac = walker.walk(
        n_steps=args.steps,
        burnin=args.burnin,
        verbose=args.verbose,
    )

    # Gather acceptance fractions at rank 0
    acc_fracs = comm.gather(acc_frac, root=0)

    # Rank0 prints acceptance fractions
    if rank == 0:
        acs = ", ".join([f"{a:1.2f}" for a in acc_fracs])
        print(f"acceptance fractions: {acs}")

    # Finally, just have each rank save its own walker
    # rather than gathering them all
    with lzma.open(f"./walker_{rank}.xz", "wb") as f:
        pickle.dump(walker, f)


if __name__ == "__main__":
    main()
