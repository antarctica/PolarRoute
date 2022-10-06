import argparse
import json
import inspect
import logging

from polar_route import __version__ as version
from polar_route.utils import setup_logging, timed_call


@setup_logging
def get_args(
        default_output: str,
        config_arg: bool = True,
        info_arg: bool = False):
    """

    Parameters
    ----------
    config_arg
    info_arg

    Returns
    -------

    """
    ap = argparse.ArgumentParser()

    ap.add_argument("-o", "--output",
                    default=default_output,
                    help="Output file")
    ap.add_argument("-v", "--verbose",
                    default=False,
                    action="store_true",
                    help="Turn on DEBUG level logging")

    if config_arg:
        ap.add_argument("config", type=argparse.FileType("r"))

    if info_arg:
        ap.add_argument("info", type=argparse.FileType("r"))

    return ap.parse_args()


@timed_call
def create_mesh_cli():
    """

    """
    from polar_route.mesh import Mesh

    args = get_args("create_mesh.output.json")
    logging.info("{} {}".format(inspect.stack()[0][3][:-4], version))

    config = json.load(args.config)

    # Discrete Meshing
    cg = Mesh(config)
    info = cg.to_json()
    json.dump(info, open(args.output, "w"))


@timed_call
def add_vehicle_cli():
    """

    """
    from polar_route.vessel_performance import VesselPerformance

    args = get_args("add_vehicle.output.json", config_arg=False, info_arg=True)
    logging.info("{} {}".format(inspect.stack()[0][3][:-4], version))

    mesh = json.load(args.info)

    vp = VesselPerformance(mesh)
    info = vp.to_json()
    json.dump(info, open(args.output, "w"))


@timed_call
def optimise_routes_cli():
    """

    """
    from polar_route.route_planner import RoutePlanner

    args = get_args("optimise_routes.output.json",
                    config_arg=False, info_arg=True)
    logging.info("{} {}".format(inspect.stack()[0][3][:-4], version))

    vehicle_mesh = json.load(args.info)
    rp = RoutePlanner(vehicle_mesh)
    rp.compute_routes()
    rp.compute_smoothed_routes()
    info = rp.to_json()
    json.dump(info, open(args.output, "w"))


@timed_call
def route_plotting_cli():
    from geoplot.interactive import Map
    import pandas as pd

    args = get_args("routes.png", config_arg=False, info_arg=True)
    logging.info("{} {}".format(inspect.stack()[0][3][:-4], version))

    info = json.load(args.info)

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
    mp.save(args.output)
