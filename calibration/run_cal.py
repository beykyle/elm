#!/usr/bin/env python3

import lzma
import dill as pickle
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Rank 0 loads the initial walker data
if rank == 0:
    with lzma.open("./walker.xz", "rb") as f:
        walker = pickle.load(f)
else:
    walker = None

# Broadcast walker object to all processes
walker = comm.bcast(walker, root=0)

# Each process performs its walk
acc_frac = walker.walk(
    n_steps=10000,
    burnin=1000,
    verbose=False,
)

# Gather all walkers at rank 0
all_walkers = comm.gather(walker, root=0)

# Gather acceptance fractions at rank 0
acc_fracs = comm.gather(acc_frac, root=0)

# Rank 0 saves the results
if rank == 0:
    acs = ", ".join([f"{a:1.2f}" for a in acc_fracs])
    print(f"acceptance fractions: {acs}")
    with lzma.open("./all_chain_walkers.xz", "wb") as f:
        pickle.dump(all_walkers, f)
