# elm
The East Lansing model global optical potential, parameter samples and implementation, provided as a git package for version control. This way, the model version is baked in, making it easy to track alterations via new branches, and parameter samples calibrated using a specific version of the model are tied to that version. Future releases of the model will numbered just like any software package and released here.

Feel free to make your own branch or fork of the model to try out your own ideas! 

## publication

The model is described in the following publications:

## install

```
git clone git@github.com:beykyle/elm.git
cd elm
pip install .
```

Now you can
```
import elm
```

and use the provided samples and implementation.

## useage with git submodule

Another way to use the ELM in your project is to use git submodules like so (from within your project git directory):

```
git submodule add git@github.com:beykyle/elm.git
```

then you can include `elm` like so:

```
from elm.src import elm

```

This way may be quick and easy way to use different versions of the ELM for different projects. To change the version, simply

```
cd elm
git checkout <desired branch or tag>
```
