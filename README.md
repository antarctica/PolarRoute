![](logo.jpg)
# PolarRoute
> PolarRoute is a long-distance maritime polar route planning, taking into account complex changing environmental conditions. The codebase allows the construction of optimised routes through three main stages: discrete modelling of the environmental conditions using a non-uniform mesh, the construction of mesh-optimal paths, and physics informed path smoothing. In order to account for different vehicle properties we construct a series of data driven functions that can be applied to the environmental mesh to determine the speed limitations and fuel requirements for a given vessel and mesh cell, representing these quantities graphically and geospatially.

Additional documentation can be found at the Github page https://antarctica.github.io/PolarRoute/

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

## Command-line Interface
Once installed, a mesh can be created from a config file using the command:
```
create_mesh <config.json>
```
optional arguments are
```
-v (verbose logging)
-o <output location> (set output location for mesh)
```

Vehicle specific information can be encoded into the mesh using
the command:
```
add_vehicle <mesh.json>
```
optional arguments are
```
-v (verbose logging)
-o <output location> (set output location for mesh)
```

Optimal routes through a mesh can be calculated using the command:
```
optimise_routes <vessel_mesh.json> <waypoints.csv>
```
optional arguments are
```
-v (verbose logging)
-o <output location> (set output location for mesh)
-p (output only the caculated path, not the entire mesh)
```

Meshes produced at any stage in the route planning process can be visualised using the GeoPlot library.
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
