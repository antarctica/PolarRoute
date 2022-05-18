# RoutePlanner: Long-distance marine navigation route planner
---
## Introduction
A long-distance route planner for marine navigation around Antartica. This work is a refactor from the original Java code at [link](https://github.com/foxm1/RoutePlanner). 

---
## Installation
For installation of the package we suggest that the use sets up their Python enviroments using Conda/MiniCoda distributions. Depending on the use case the additional information is supplied for the installation on the different operating systems

### Windows Installation
The RoutePlanner software can be installed by using the following. Be careful and make sure that the following lines are done in AnacondaComandPrompt
```
  conda create -n RoutePlanner python=3.9
  conda activate RoutePlanner
  pip install numpy
  conda install geopandas
  python setup.py install
```

### Mac/Linux Installation
The RoutePlanner software can be installed by using
```
  conda create -n RoutePlanner python=3.9
  conda activate RoutePlanner
  pip install numpy
  python setup.py install
```

---
## BAS Developers
Maria Fox, Jonathan Smith, Samuel Hall, James Byrne, George Coombs &  Michael Thorne


Documentation for the Code
**************************
.. toctree::
   :maxdepth: 2
   :caption: Contents:

Background
====================
The software package can be broadly separated into four main stages.


CellBox
=====================
Module Information
-----------------
.. automodule:: RoutePlanner.CellBox


Class Information
------------------
.. autoclass:: RoutePlanner.CellBox.CellBox
   :members:
