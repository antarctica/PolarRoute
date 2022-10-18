![](logo.jpg)
# PolarRoute
> PolarRoute is a long-distance maritime polar route planning, taking into account complex changing environmental conditions. The codebase allows the construction of optimised routes through three main stages: discrete modelling of the environmental conditions using a non-uniform mesh, the construction of mesh-optimal paths, and physics informed path smoothing. In order to account for different vehicle properties we construct a series of data driven functions that can be applied to the environmental mesh to determine the speed limitations and fuel requirements for a given vessel and mesh cell, representing these quantities graphically and geospatially.
---
## Installation
### Linux/MacOS
The PolarRoute software can be installed on Linux/MacOS by running one of the two following commands.

Github:
```
python setup.py install
```

Pip: 
```
pip install polar-route
```

### Windows
The PolarRoute software requires GDAL files to be installed. The PolarRoute software can be installed on Windows by running one of the two following commands.

Github:
```
pip install pipwin
pipwin install gdal
pipwin install fiona
python setup.py install
```

Pip: 
```
pip install pipwin
pipwin install gdal
pipwin install fiona
pip install polar-route
```

## Manual
The manual for this software can be installed by running:
```
source venv/bin/activate
pip install .[docs]
sphinx-build -b html ./docs/source ./docs/build
```
the html manual can then be found at ./docs/build.

## Developers
Jonathan Smith, Samuel Hall, George Coombs, James Byrne,  Michael Thorne, Maria Fox

## License
This software is licensed under a MIT license. For more information please see the attached  ``LICENSE`` file.

[version]: https://img.shields.io/PolarRoute/v/datadog-metrics.svg?style=flat-square
[downloads]: https://img.shields.io/PolarRoute/dm/datadog-metrics.svg?style=flat-square
