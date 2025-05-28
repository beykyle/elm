#!/usr/bin/env python3
from pathlib import Path
import argparse
import pickle

import numpy as np
from scipy import stats

from mpi4py import MPI

from elm import elm


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


def metropolis_hastings(
    x0,
    n_steps,
    log_prob,
    propose,
    burn_in=1000,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng(42)
    chain = np.zeros((n_steps - burn_in, x0.size))
    logp = np.zeros((n_steps - burn_in, x0.size))
    logp = log_prob(x0)
    accepted = 0
    x = x0
    for i in range(n_steps):
        x_new = propose(x)
        logp_new = log_prob(x_new)
        log_ratio = min(0, logp_new - logp)
        xi = np.log(rng.random())
        if xi < log_ratio:
            x = x_new
            logp = logp_new
            accepted += 1

        if i >= burn_in:
            chain[i - burn_in] = x
            logp[i - burn_in] = logp

    return np.array(chain), np.array(logp), accepted


def parse_options(comm):
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
        help="Number of steps to not log at the beginning of each chain",
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
        help="Directory to write output to. If it doesn't exist it will be created. All outputs will be written to it as .np files",
    )
    parser.add_argument(
        "--proposal_cov_scale_factor",
        type=float,
        default=100,
        help="ratio of diagonal elements in prior covariance to corresponding diagonal elements in proposal distribution",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for rank 0. All ranks will use seed + rank as their seed.",
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

    # TODO seeding

    args = None
    try:
        if comm.Get_rank() == 0:
            args = parser.parse_args()

            # output dir
            if args.output is None:
                args.output = "./"
            args.output = Path(args.output)
            if args.output.is_file():
                raise ValueError(f"--output ({args.output}) cannot be a file.")
            args.output.mkdir(parents=True, exist_ok=True)

            # read in prior
            prior, path = args.prior_path
            if not hasattr(prior, "logpdf") or not callable(getattr(prior, "logpdf")):
                raise argparse.ArgumentTypeError(
                    f"Object in '{path}' does not support logpdf "
                    "(.logpdf() method missing or not callable)."
                )
            if not hasattr(prior, "cov") or not hasattr(prior, "mean"):
                raise argparse.ArgumentTypeError(
                    f"Object in '{path}' does not have `mean` and `cov` attributes."
                )
            args.prior = prior

            # read in corpus
            corpus, path = args.corpus_path
            if not hasattr(corpus, "logpdf") or not callable(getattr(corpus, "logpdf")):
                raise argparse.ArgumentTypeError(
                    f"Object in '{path}' does not support logpdf "
                    "(.logpdf() method missing or not callable)."
                )
            args.corpus = corpus

    finally:
        args = comm.bcast(args, root=0)

    if args is None:
        exit(0)
    return args


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    args = parse_options(comm)

    prior = args.prior
    corpus = args.corpus

    seed = args.seed + rank
    rng = np.random.default_rng(seed)

    # proposal distribution
    proposal_cov = prior.cov / args.proposal_cov_scale_factor
    proposal_mean = np.zeros_like(prior.mean)

    def proposal(x):
        return x + stats.multivariate_normal.rvs(
            mean=proposal_mean, cov=proposal_cov, random_state=rng
        )

    def log_prob(x):
        return prior.logpdf(x) + corpus.logpdf(elm.to_ordered_dict(x))

    x0 = proposal(prior.mean)

    if args.batch_size is not None:
        rem = args.n_steps % args.batch_size
        n_full_batches = args.nsteps // args.batch_size
        batches = n_full_batches * [args.batch_size] + (rem > 0) * [rem]
    else:
        batches = [args.nsteps]

    chain = []
    logp = []
    accepted = 0
    for i, steps_in_batch in enumerate(batches):
        batch_chain, batch_logp, accepted_in_batch = metropolis_hastings(
            x0,
            steps_in_batch,
            log_prob,
            proposal,
            burn_in=args.burnin if i == 0 else 0,
            rng=rng,
        )

        # diagnostics
        accepted += accepted_in_batch
        chain.append(batch_chain)
        logp.append(batch_logp)
        x0 = batch_chain[-1]
        if args.verbose:
            print(
                f"Rank: {rank}. Batch: {i}/{len(batches)}. "
                f"Acceptance fraction: {accepted_in_batch/steps_in_batch}"
            )

        # update proposal distribution?
        # TODO

        # update unknown covariance factor estimate (Gibbs sampling)

        # write record of batch chain to disk
        np.save(Path(f"./chain_{rank}_{i}.npy"), batch_chain)

    logp = np.concatenate(logp, axis=0)
    chain = np.concatenate(chain, axis=0)

    # MPI Gather
    all_logp = comm.gather(logp, root=0)
    all_chains = comm.gather(chain, root=0)
    accepted = comm.gather(accepted, root=0)

    if rank == 0:
        all_chains = np.array(all_chains)
        all_logp = np.array(all_logp)
        acc_fracs = np.array(accepted) / args.nsteps
        print(f"\nFinished sampling all {len(all_chains)} chains.")
        print(f"Chain shape: {all_chains.shape}")
        print(f"Average acceptance fraction: {np.mean(acc_fracs):.3f}")
        for i, af in enumerate(acc_fracs):
            print(f"  Chain {i}: acceptance fraction = {af:.3f}")

    np.save(Path(args.output) / "all_chains.npy", all_chains)
    np.save(Path(args.output) / "log_likelihood.npy", all_logp)


if __name__ == "__main__":
    main()
