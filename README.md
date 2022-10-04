![](logo.jpg)
![Version][version] ![Downloads][downloads]

>PolarRoute is a long-distance polar route planner for marine navigation...
---

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
This software is licensed under a MIT license. For more information please see the attached license file.

[version]: https://img.shields.io/PolarRoute/v/datadog-metrics.svg?style=flat-square
[downloads]: https://img.shields.io/PolarRoute/dm/datadog-metrics.svg?style=flat-square