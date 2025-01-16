# East Lansing Model
The East Lansing model is an uncertainty quantified global optical potential. This repo includes both parameter samples and its python implementation. The intent of providing it as a git package is so that updates or alterations to the model can be tracked with version control, just like any piece of software. This way, one is always aware of exactly which version of the model they are using, modifications can be tracked easily, and sets of parameter samples can be tied to the specific
version of the model that is used in the calibration that generates them.

Includes convenience functions for reading, writing and manipulating ensembles of parameter samples as `numpy` [structured array](https://numpy.org/doc/stable/user/basics.rec.html) or [`pandas.DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)s.

Feel free to make your own branch or fork of the model to try out your own ideas! This setup should make it easy to implement and test any uncertainty quantified global optical potential.

## publication

The model is described in the following publications:

## useage

```python
import elm
```

First, let's read a parameter sample file into memory:

```python
sample = elm.read_sample_from_json( "path/to/sample.json" )
print(sample)
```

should print the `dict` with parameter names as keys and their values as values, something like:

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
isoscalar_params, isovector_params, spin_orbit_params, coulomb_params, delta = 
    elm.calculate_parameters(
        neutron, Ca48, com_frame_energy, fermi_energy, sample
    )
```

Now we can evaluate the different parts of the model on a radial coordinate grid $r$:

```python
r = np.linspace(0.01, 10, 200)
v0  = elm.isoscalar(r, *isoscalar_params)
```

and so on.

## samples

## install

```bash
git clone git@github.com:beykyle/elm.git
cd elm
pip install -e .
```

Now you can
```python
import elm
```

and use the provided samples and implementation. 

## useage with git submodule

Another way to use the ELM in your project is to use git submodules like so (from within your project git directory):

```bash
git submodule add git@github.com:beykyle/elm.git
```

then you can include `elm` like so:

```python
from elm.src import elm
```

This way may be quick and easy way to use different versions of the ELM for different projects. To change the version, simply

```bash
cd elm
git checkout <desired branch or tag>
```
