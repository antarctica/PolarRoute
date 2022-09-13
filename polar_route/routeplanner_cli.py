"""
    Command line interface for the PolarRoute route planner.

    The command line interface expects 3 arguments to be passed:

    Args:
         (1) config: The configuration file used to build the digital enviroment
         and the routes between waypoints
         (2) output_location: The file location the resulting file will be saved to
         (3) output_type: The type of output that will be saved
            < JSON | HTML | PATHS >

    An example for running this program:
        'python routeplanner_cli.py <config.json> <interactive.html> <HTML>'

"""

import sys
import pandas as pd
import json

from polar_route.mesh import Mesh
from polar_route.vessel_performance import VesselPerformance
from polar_route.route_planner import RoutePlanner
from geoplot.interactive import Map

def main():

    args = sys.argv[1:]
    config = args[0]
    output_location = args[1]
    output_type = args[2]

    with open(config, 'r') as f:
        info = json.load(f)

    print("Constructing Mesh...")
    cg = Mesh(info)
    info = cg.to_json()

    print("Calculating Vessel Performance...")
    vp = VesselPerformance(info)
    info = vp.to_json()

    print("Calculating Routes...")
    rp = RoutePlanner(info)
    rp.compute_routes()
    rp.compute_smoothed_routes()
    info = rp.to_json()

    if output_type == "JSON":
        with open(output_location, 'w') as f:
            json.dump(info, f)
    if output_type == "PATHS":
        with open(output_location, 'w') as f:
            json.dump(info['paths'], f)
    if output_type == "HTML":
        print("Building interactive map...")

        config    = info['config']
        mesh      = pd.DataFrame(info['cellboxes'])
        paths     = info['paths']
        waypoints = pd.DataFrame(info['waypoints'])

        mp = Map(title='Example Test 1')
        mp.Maps(mesh,'SIC',predefined='SIC')
        mp.Maps(mesh,'Extreme Ice',predefined='Extreme Sea Ice Conc')
        mp.Maps(mesh,'Land Mask',predefined='Land Mask')
        mp.Maps(mesh,'Fuel',predefined='Fuel',show=False)
        mp.Maps(mesh, 'speed', predefined = 'Speed', show = False)

        mp.Paths(paths,'Routes',predefined='Route Traveltime Paths')
        mp.Points(waypoints,'Waypoints',names={"font_size":10.0})
        mp.MeshInfo(mesh,'Mesh Info',show=False)
        mp.save(output_location)

    print("Done")

if __name__ == "__main__":
    main()
