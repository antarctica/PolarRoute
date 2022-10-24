![](logo.jpg)

<a href="https://colab.research.google.com/drive/12D-CN10X7xAcXn_df0zNLHtdiiXxZVkz?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" alt="Colab">
<a href="https://antarctica.github.io/PolarRoute/"><img src="https://img.shields.io/badge/Manual%20-github.io%2FPolarRoute%2F-red" alt="Manual Page">
<a href="https://pypi.org/project/polar-route/"><img src="https://img.shields.io/pypi/v/polar-route" alt="PyPi">
<a href="https://github.com/antarctica/PolarRoute/tags"><img src="https://img.shields.io/github/v/tag/antarctica/PolarRoute" alt="Release Tag"></a>
<a href="https://github.com/antarctica/PolarRoute/issues"><img src="https://img.shields.io/github/issues/antarctica/PolarRoute" alt="Issues"></a>
<a href="https://github.com/antarctica/PolarRoute/blob/main/LICENSE"><img src="https://img.shields.io/github/license/antarctica/PolarRoute" alt="License"></a>

# PolarRoute
> PolarRoute is a long-distance maritime polar route planning, taking into account complex changing environmental conditions. The codebase allows the construction of optimised routes through three main stages: discrete modelling of the environmental conditions using a non-uniform mesh, the construction of mesh-optimal paths, and physics informed path smoothing. In order to account for different vehicle properties we construct a series of data driven functions that can be applied to the environmental mesh to determine the speed limitations and fuel requirements for a given vessel and mesh cell, representing these quantities graphically and geospatially.

## Installation
The PolarRoute software requires GDAL files to be installed. The PolarRoute software can be installed on Windows by running the required wheels for GDAL and FIONA. MOre information can be found in the manual pages linked above. Once these requirements are met then the software can be installed by:

Github:
```
python setup.py install
```

 Pip: 
```
pip install polar-route
```

## Required Data sources
Polar-route has been built to work with a variety of open-source climactic data sources. 
A list of supported data sources and there associated data-loaders is given in the 
'Data Loaders' section of the manual

## Developers
Jonathan Smith, Samuel Hall, George Coombs, James Byrne,  Michael Thorne, Maria Fox

## License
This software is licensed under a MIT license. For more information please see the attached ``LICENSE`` file.

[version]: https://img.shields.io/PolarRoute/v/datadog-metrics.svg?style=flat-square
[downloads]: https://img.shields.io/PolarRoute/dm/datadog-metrics.svg?style=flat-square