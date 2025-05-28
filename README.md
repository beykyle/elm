# East Lansing Model
The East Lansing model is an uncertainty quantified global optical potential. This repo includes both parameter samples and its python implementation. The intent of providing it as a git package is so that updates or alterations to the model can be tracked with version control, just like any piece of software. This way, one is always aware of exactly which version of the model they are using, modifications can be tracked easily, and sets of parameter samples can be tied to the specific
version of the model that is used in the calibration that generates them.

Includes convenience functions for reading, writing and manipulating ensembles of parameter samples as `numpy` [structured array](https://numpy.org/doc/stable/user/basics.rec.html) or [`pandas.DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)s.

Feel free to make your own branch or fork of the model to try out your own ideas! This setup should make it easy to implement and test any uncertainty quantified global optical potential.

## publication

The model is described in the following publications:

## samples


## install for development or modification

To modify the model, first clone and build
```bash
git clone git@github.com:beykyle/elm.git
cd elm
python3 -m build
```

Then install an editable version locally like so:

```
pip install -ve .
```

It is highly recommended that you use an isolated virtual environment (e.g. using [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) or [conda/mamba](https://mamba.readthedocs.io/en/latest/)), as this action will install all of the dependencies in `requirements.txt`, at the specific version required.

## usage

Once you've installed, you can in python
```python
import elm
```

This allows you to run all of the notebooks, to set up and visualize the results of a calibration. 
For example, let's read a parameter sample file into memory:

```python
sample = elm.elm.read_sample_from_json( "path/to/sample.json" )
print(sample)
```

should print the `OrderedDict` with parameter names as keys and their values as values, something like:

```
{ 
    'V0' : 34.67, 
    'W0' 12.8, 
    'Wd0' : 3.2, 
    ... 
}
```

Now, for a given projectile-target system, we can calculate the depths, radii and things that go into the potential model:

```python
Ca48 = (48, 20)
neutron = (1,0)
com_frame_energy = 13.9
fermi_energy = -7.59
isoscalar_central_params, isovector_central_params, isoscalar_spin_orbit_params, isovector_spin_orbit_params, coulomb_params, asym = 
    elm.calculate_parameters(
        neutron, Ca48, com_frame_energy, fermi_energy, sample
    )
```

Now we can evaluate the different parts of the model on a radial coordinate grid $r$:

```python
r = np.linspace(0.01, 10, 200)
v0  = elm.isoscalar(r, *isoscalar_central_params)
```

and so on.

Once you've set up your calibration (e.g. by stepping through the notebooks in `prior/`, `data/` and `calibration/`), you can run `mcmc` to run the actual calibration:

```
mpirun -n 1 python -m mpi4py mcmc.py --help
```

For example, once you've run `calibration/setup_cal.ipynb` to generate the `Corpus` objects, you can then run, for example

```
mpirun -n 12 mcmc --nsteps 10000 --burnin 1000  --corpus_path ./calibration/nn_corpus.pkl --prior_path ./prior/prior_distribution.pickle

```
,

which will run Metropolis-Hastings with twelve independent chains, 10000 steps each using the supplied prior and corpus of constraints.

