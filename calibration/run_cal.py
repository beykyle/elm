#!/usr/bin/env python3

import lzma
import dill as pickle
from mpi4py import MPI
import argparse


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

    # Rank 0 loads the initial walker data
    if rank == 0:
        with lzma.open(args.input, "rb") as f:
            walker = pickle.load(f)
    else:
        walker = None

    # Broadcast walker object to all processes
    walker = comm.bcast(walker, root=0)

    # Each process performs its walk
    acc_frac = walker.walk(
        n_steps=args.steps,
        burnin=args.burnin,
        verbose=args.verbose,
    )

    # Gather all walkers at rank 0
    all_walkers = comm.gather(walker, root=0)

    # Gather acceptance fractions at rank 0
    acc_fracs = comm.gather(acc_frac, root=0)

    # Rank 0 saves the results
    if rank == 0:
        acs = ", ".join([f"{a:1.2f}" for a in acc_fracs])
        print(f"acceptance fractions: {acs}")
        with lzma.open(args.output, "wb") as f:
            pickle.dump(all_walkers, f)


if __name__ == "__main__":
    main()
