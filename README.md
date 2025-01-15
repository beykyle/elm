# elm
The East Lansing model global optical potential, parameter samples and implementation, provided as a git package for version control. This way, the model version is baked in, making it easy to track alterations via new branches, and parameter samples calibrated using a specific version of the model are tied to that version. Future releases of the model will numbered just like any software package and released here.

Feel free to make your own branch or fork of the model to try out your own ideas! 

## publication

The model is described in the following publications:

## useage

```python
import elm
```

First, let's read a parameter sample file into memory:

```python
sample = read_sample_from_json( "path/to/sample.json" )
```

Now, for a given projectile-target system, we can calculate the depths, radii and things that go into the potential model:

```python
Ca48 = (48, 20)
neutron = (1,0)
com_frame_energy = 13.9
fermi_energy = -2.11
isoscalar_params, isovector_params, spin_orbit_params, coulomb_params, delta = elm.calculate_parameters(neutron, Ca48, com_frame_energy, fermi_energy, sample)
```

Now we can evaluate the different parts of the model at a given radial coordinate $r$:

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
pip install .
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
