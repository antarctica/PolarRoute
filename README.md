# PolarRoute: Long-horizon marine polar navigation route planner  

## Introduction
A long-distance polar route planner for marine navigation. This work is a refactor from the original Java code that can be found [here](https://github.com/foxm1/RoutePlanner). 


## Installation
The PolarRoute software can be installed by running the following commands. Be careful and make sure that these are done in Anaconda.
```
  conda create -n PolarRoute python=3.9
  conda activate PolarRoute
  conda install geopandas ipykernel
  pip install sphinx tqdm rinohtype numpy==1.22
  python setup.py install
```
---

## Manual
The manual for this software can be installed by running:
```
  sphinx-build -b html ./doc/source ./doc/build
```
the html manual can then be found at ./doc/build.

## Developers
Jonathan Smith, Samuel Hall, George Coombs, James Byrne,  Michael Thorne, Maria Fox

## License
This software is licensed under a MIT license. For more information please see the attached license file.