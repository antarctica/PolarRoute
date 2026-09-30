# PolarRoute

![](./docs/assets/logo.jpg)

![Dev Status](https://img.shields.io/badge/Status-Active-green)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12D-CN10X7xAcXn_df0zNLHtdiiXxZVkz?usp=sharing)
[![Documentation](https://img.shields.io/badge/Manual%20-github.io%2FPolarRoute%2F-red)](https://bas-amop.github.io/PolarRoute/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/polarroute)
[![Release Tag](https://img.shields.io/github/v/tag/bas-amop/PolarRoute)](https://github.com/bas-amop/polarroute/tags)
[![Issues](https://img.shields.io/github/issues/bas-amop/PolarRoute)](https://github.com/bas-amop/PolarRoute/issues)
[![License](https://img.shields.io/github/license/bas-amop/PolarRoute)](https://github.com/bas-amop/PolarRoute/blob/main/LICENSE)
[![Test Status](https://github.com/bas-amop/PolarRoute/actions/workflows/test.yml/badge.svg)](https://github.com/bas-amop/PolarRoute/actions/workflows/test.yml)
[![Coverage](https://img.shields.io/badge/coverage-report-blue)](https://github.com/bas-amop/PolarRoute/actions/workflows/test.yml)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-0C3A5C?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/450086048.svg)](https://doi.org/10.5281/zenodo.21134233)


PolarRoute is a long-distance maritime polar route planning package, able to take into account complex and changing environmental conditions. It allows the construction of optimised routes through three main stages: discrete modelling of the environmental conditions using a non-uniform mesh, the construction of mesh-optimal paths, and physics informed path smoothing. In order to account for different vehicle properties we construct a series of data-driven functions that can be applied to the environmental mesh to determine the speed limitations and fuel requirements for a given vessel and mesh cell. The environmental modelling component of this functionality is provided by the [MeshiPhi](https://github.com/bas-amop/MeshiPhi) library.

## Installation

PolarRoute is available from [PyPI](https://pypi.org/project/polar-route/) and the latest version can be installed by running:

```
pip install polar-route
```

Alternatively you can install PolarRoute by downloading the source code from GitHub:
```
git clone https://github.com/bas-amop/PolarRoute
cd PolarRoute
pip install -e .
```

Use of `-e` is optional, based on whether you want to be able to edit the installed copy of the package.

In order to run the test suite you will also need to include the `test` dependency group:

```
pip install --group test
```

> NOTE: Some features of the PolarRoute package require GDAL to be installed. Please consult the [documentation](https://bas-amop.github.io/PolarRoute) for further guidance.

## Usage

PolarRoute operates by creating an environmental mesh, adding vessel performance characteristics, and optimising routes between waypoints. Environmental meshes are created using [MeshiPhi](https://github.com/bas-amop/MeshiPhi), which is installed automatically when installing PolarRoute.

### Quick Start (CLI)

```bash
# Create environmental mesh
create_mesh examples/environment_config/grf_example.config.json -o mesh.json

# Add vessel performance model
add_vehicle examples/vessel_config/SDA.config.json mesh.json -o vessel_mesh.json

# Optimise routes
optimise_routes examples/route_config/traveltime.config.json vessel_mesh.json examples/waypoints_example.csv -o routes.json
```

### Quick Start (Python API)

```python
from meshiphi.mesh_generation.mesh_builder import MeshBuilder
from polar_route.vessel_performance.vessel_performance_modeller import VesselPerformanceModeller
from polar_route.route_planner.route_planner import RoutePlanner

# Create environmental mesh (using MeshiPhi)
mesh_builder = MeshBuilder(env_config)
mesh_json = mesh_builder.build_environmental_mesh().to_json()

# Add vessel performance to mesh
vpm = VesselPerformanceModeller(mesh_json, vessel_config)
vpm.model_accessibility()
vpm.model_performance()
vessel_mesh_json = vpm.to_json()

# Calculate routes
rp = RoutePlanner(vessel_mesh_json, route_config)
rp.compute_routes(waypoints_path)
routes_json = rp.to_json()
```

For more details, see the [CLI documentation](https://bas-amop.github.io/PolarRoute/cli/), [examples](https://bas-amop.github.io/PolarRoute/examples/), and the [examples/](examples/) directory.

## Required Data sources
PolarRoute has been built to work with a variety of open-source atmospheric and oceanographic data sources. For testing and demonstration purposes it is also possible to generate artificial Gaussian Random Field data.

A full list of supported data sources and their associated dataloaders is given in the  'Dataloader Overview' section of the [MeshiPhi manual](https://bas-amop.github.io/MeshiPhi/dataloaders/overview/)

## Developers
Samuel Hall, Harrison Abbot, Ayat Fekry, George Coombs, David Wyld, Thomas Zwagerman, Jonathan Smith, Maria Fox, and James Byrne

## License
This software is licensed under a MIT license, but request users cite our publication:


Jonathan D. Smith, Samuel Hall, George Coombs, Ayat Fekry, Michael A. S. Thorne, Derek Long, Maria Fox (2025) Path-Planning on a Spherical Surface with Disturbances and Exclusion Zones. Journal of Artificial Intelligence Research 82 (2025), <https://doi.org/10.1613/jair.1.16746>

For more information please see the attached ``LICENSE`` file.

[version]: https://img.shields.io/PolarRoute/v/datadog-metrics.svg?style=flat-square
[downloads]: https://img.shields.io/PolarRoute/dm/datadog-metrics.svg?style=flat-square
