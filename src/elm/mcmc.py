from pathlib import Path
import argparse
import pickle

import numpy as np
from scipy import stats

from mpi4py import MPI

from . import elm


def validate_pickle_file(path_str: str):
    path = Path(path_str)

    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File '{path}' does not exist.")

    try:
        with path.open("rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"File '{path}' is not a valid pickle file: {e}"
        )

    return obj, path


def metropolis_hastings(x0, n_steps, log_prob, propose, burn_in=1000):
    chain = np.zeros((n_steps, x0.size))
    logp = log_prob(x0)
    accepted = 0
    x = x0
    for i in range(n_steps):
        x_new = x + propose()
        logp_new = log_prob(x_new)
        accept_prob = np.exp(logp_new - logp)

        if np.random.rand() < accept_prob:
            x = x_new
            logp = logp_new
            accepted += 1

        if i >= burn_in:
            chain.append(x)

    return np.array(chain), accepted


def main():
    parser = argparse.ArgumentParser(
        description="Run MCMC with independent walkers each on their own MPI ranks"
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=1000,
        help="Total number of MCMC steps per chain (including burn-in).",
    )
    parser.add_argument(
        "--burnin",
        type=int,
        default=100,
        help="Total number of MCMC steps per chain (including burn-in).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Number of steps per batch to incrementally write to file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output .np file to save the chains.",
    )
    parser.add_argument(
        "--proposal_cov_scale_factor",
        type=int,
        default=100,
        help="ratio of diaginal elements in prior covariance to corresponding diagonal elements in proposal distribution",
    )
    parser.add_argument("--verbose", type=bool, default=False, help="Verbose printing")
    parser.add_argument(
        "--corpus_path",
        type=lambda p: validate_pickle_file(p),
        required=True,
        help="Path to constraint corpus pickle object",
    )
    parser.add_argument(
        "--prior_path",
        type=lambda p: validate_pickle_file(p),
        required=True,
        help="Path to prior distribution pickle object",
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # read in prior
    prior, path = args.prior_path
    if not hasattr(prior, "logpdf") or not callable(getattr(prior, "logpdf")):
        raise argparse.ArgumentTypeError(
            f"Object in '{path}' does not support logpdf (.logpdf() method missing or not callable)."
        )

    # read in corpus
    corpus, path = args.corpus_path
    if not hasattr(corpus, "logpdf") or not callable(getattr(corpus, "logpdf")):
        raise argparse.ArgumentTypeError(
            f"Object in '{path}' does not support logpdf (.logpdf() method missing or not callable)."
        )

    # proposal distribution
    proposal_cov = prior.cov / args.proposal_cov_scale_factor
    proposal = stats.multivariate_normal(np.zeros_like(prior.mean), proposal_cov)

    def log_prob(x):
        return prior.logpdf(x) + corpus.logpdf(elm.to_ordered_dict(x))

    x0 = prior.mean + proposal.rvs

    if args.batch_size is not None:
        rem = args.n_steps % args.batch_size
        n_full_batches = args.nsteps // args.batch_size
        batches = n_full_batches * [args.batch_size] + (rem > 0) * [rem]
    else:
        batches = [args.nsteps]

    chain = []
    accepted = 0
    for i, steps_in_batch in enumerate(batches):
        batch_chain, accepted_in_batch = metropolis_hastings(
            x0, steps_in_batch, log_prob, proposal.rvs, burn_in=args.burnin
        )
        accepted += accepted_in_batch
        chain = np.vstack([chain, batch_chain])
        x0 = chain[-1]
        if args.verbose:
            print(
                f"Rank: {rank}. Batch: {i}/{len(batches)}. Acceptance fraction: {accepted_in_batch/steps_in_batch}"
            )
        np.save(Path(f"./chain_{rank}.npy"), chain)

    all_chains = comm.gather(chain, root=0)
    accepted = comm.gather(accepted, root=0)

    if rank == 0:
        all_chains = np.array(all_chains)
        acc_fracs = np.array(accepted) / args.n_steps
        print(f"\nFinished sampling all {len(all_chains)} chains.")
        print(f"Chain shape: {all_chains.shape}")
        print(f"Average acceptance fraction: {np.mean(acc_fracs):.3f}")
        for i, af in enumerate(acc_fracs):
            print(f"  Chain {i}: acceptance fraction = {af:.3f}")

    if args.output:
        np.save(Path(args.output), all_chains)


if __name__ == "__main__":
    main()
