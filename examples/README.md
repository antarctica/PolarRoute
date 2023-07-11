# Examples
This directory contains example configs that can be used with PolarRoute. Please make sure you have PolarRoute properly installed before attempting to run these examples. 
You will also need GeoPlot installed to run the `plot_mesh` command, however it is not necessary if all you need is the ability to generate meshes. 

To run PolarRoute, choose one config out of each directory and execute commands in this order:

1. create_mesh {environment_config}
2. add_vehicle {vessel_config} {create_mesh_output}
3. optimise_routes {route_config} {add_vehicle_output} {waypoints}
4. (optional) plot_mesh {any_output}

{environment_config}, {vessel_config}, {route_config}, and {waypoints} can be any of the JSON's in `environment_config/`, `vessel_config/`, `route_config/`, and `./` respectively. 

{create_mesh_output} is typically `create_mesh.output.json`

{add_vehicle_output} is typically `add_vehicle.output.json`

Both of these can be manually set by using the `-o` flag when running `create_mesh` and `add_vehicle` respectively.

{any_output} can be the output after any of the 3 prior stages. `plot_mesh` will output a HTML showing the mesh after any stage, which can be useful for debugging purposes.

## Environment Config
Contains 3 examples:
1. grf_example.json:
    - Generates fake data to mesh using Gaussian Random Fields (GRF)
    - Contains example all possible config parameters for GRF dataloader

2. grf_minimal_example.json
    - Generates fake data to mesh using GRF's
    - Contains bare minimum information required to be able to generate mesh from GRF dataloader

3. real_example.json
    - Relies on real data, see documentation for sources of each dataset
    - Contains example of typical config parameters used during development
    - File / folder paths are relative

## Vessel Config
Contains 1 example:
1. SDA.config.json
    - Parameters used to simulate the SDA research vessel.

## Route Config
Contains 2 examples:
1. fuel.config.json
    - For generating fuel optimised routes
2. traveltime.config.json
    - For generating travel-time optimised routes

## Waypoints
Contains 2 examples:
1. waypoints_real.csv
    - Contains points of interest around the world that the SDA may travel to.
2. waypoints_example.csv
    - Minimal working example that will work with the GRF meshes

## Common Errors
The most common source of errors is a malformed config. To ensure that this does not happen, here are some easy checks you can perform:
- Do the files/folders in your environment_config point to the correct location?
- Does the time range specified in your environment_config exclude all the data from one of the dataloaders?
- Do all of your waypoints lay outside of your initial mesh boundary?
- Are your waypoints inaccessible because of land/SIC?

## Known bugs
These are issues that may occur that are outside of your control
- Do any of your waypoints lay exactly on one of the cellbox boundaries?
- Do you not have a dataloader for `SIC` or `uC,vC`? (excluding one of these will crash `optimise_routes`)
