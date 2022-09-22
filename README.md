# PolarRoute: Long-horizon marine polar navigation route planner  

## Introduction
A long-distance polar route planner for marine navigation. This work is a refactor from the original Java code that can be found [here](https://github.com/foxm1/RoutePlanner). 

## Installation

### Development

The PolarRoute software can be installed by running the following commands:
```
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### Production

To build and deploy this project, it's as simple as:
```commandline
source venv/bin/activate
pip install --upgrade pip setuptools wheel
python setup.py build bdist_wheel
```
---

## Manual

The manual for this software can be installed by running:
```
source venv/bin/activate
pip install doc/requirements.txt
sphinx-build -b html ./doc/source ./doc/build
```
the html manual can then be found at ./doc/build.

## Developers
Jonathan Smith, Samuel Hall, George Coombs, James Byrne,  Michael Thorne, Maria Fox

## License
This software is licensed under a MIT license. For more information please see the attached license file.
