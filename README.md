# East Lansing Model
The East Lansing model is an uncertainty quantified global optical potential. This repo contains the code in the form of notebooks to generate the prior, curate the set of experimental constraints, formulate a likelihood model, run the calibration, and visualize and diagnose the results, all using the package [`rxmc`](https://github.com/beykyle/rxmc). Of course, it also contains the implementation of the physics of the model, in `src/elm/elm.py`.

Feel free to make your own branch or fork of the model to try out your own ideas! This setup should make it easy to implement and test any uncertainty quantified global optical potential.

## publication

The model is described in the following publications:

## install for development or modification

To modify the model, first clone and build
```bash
git clone git@github.com:beykyle/elm.git
cd elm
```

Then install an editable version locally like so:

```
pip install -ve .
```

Note that `pip` will install package dependencies listed in `requirements.txt`. It is **highly recommended** that you use an isolated virtual environment (e.g. using [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) or [conda/mamba](https://mamba.readthedocs.io/en/latest/)), as this action will install all of the dependencies in `requirements.txt`, at the specific version required.

If you don't want to create an isolated environment for `elm`, but also don't want `pip` to overwrite the package versions you have with the ones in `requirements.txt`, you can

```
pip install -ve . --no-deps
```
This will require that your current python environment satisfies the `requirements.txt`. 

## Setting up the calibration

Each of the sub-directories encapsulates choices that must be made to calibrate a model:
- `src/` contains the implementation of the model form, which is put into a form useable by `rxmc` in `model/`
- `prior/` contains the prior distribution, which is a multivariate Gaussian in the space of the model parameters
- `evidence/` contains the experimental data that will be used to constrain the model 
- `likelihood_model/` contains the likelihood model that will be used to compare the model predictions to the experimental observations
- `calibration/` contains the code to run the calibration, which is a Markov Chain Monte Carlo (MCMC) sampling of the posterior distribution of the model parameters given the prior and evidence
- `propagation/` contains the code to propagate a distribution of model parameters into the experimental data space
- `visualization/` contains the code to visualize the results of a calibration, including the posterior distribution of the model parameters and the model predictions compared to the experimental data


## Running a calibration


## visualizing the results

