"""Gather multiple walkers and write full results to disk with MPI"""

import argparse
import lzma
import os
import sys

import dill as pickle
import numpy as np
from mpi4py import MPI


def main():
    """Main function to gather walkers and write results to disk."""

    parser = argparse.ArgumentParser(
        description="Each rank looks for walker_{rank}.xz, reads it, "
        "and the resulting model chain and log likelihood aree gathered"
        " from all ranks, concatenated and written to disk"
    )
    parser.add_argument(
        "--input", type=str, default="./", help="Input directory to read walkers from."
    )
    args = parser.parse_args()

    # Verify that the input file exists
    if not os.path.isdir(args.input):
        print(f"Error: The input directory '{args.input}' does not exist.")
        sys.exit(1)

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Read input file
    try:
        with lzma.open(f"{args.input}/walker_{rank}.xz", "rb") as f:
            walker = pickle.load(f)
    except Exception as e:
        print(
            f"Error: Failed to read input file '{args.input}/walker_{rank}.xz' on rank {rank}. Exception: {e}"
        )
        sys.exit(1)

    # Gather model chains and log likelihoods
    n_params = len(walker.model_sample_conf.params)
    n_steps = walker.model_chain.shape[0]
    model_chain = np.array(walker.model_chain, dtype=float)
    logls = np.array(walker.log_posterior_record, dtype=float)
    try:
        mc_recvbuf = None
        logl_recvbuf = None
        if rank == 0:
            mc_recvbuf = np.empty([size, n_steps, n_params], dtype=float)
            logl_recvbuf = np.empty([size, n_steps], dtype=float)

        comm.Gather(model_chain, recvbuf=mc_recvbuf, root=0)
        comm.Gather(logls, recvbuf=logl_recvbuf, root=0)
    except Exception as e:
        print(f"Error: MPI Gather operation failed on rank {rank}. Exception: {e}")
        sys.exit(1)

    # Write combined results to disk at rank 0
    if rank == 0:
        try:
            np.savez_compressed(
                f"{args.input}/result.npz",
                log_likelihoods=logl_recvbuf,
                model_chains=mc_recvbuf,
            )
        except Exception as e:
            print(f"Error: Failed to write results to disk on rank 0. Exception: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
