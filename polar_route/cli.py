import argparse
import json


def get_args(
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

    ap.add_argument("--output", type=argparse.FileType("w"),
                    help="Output file")

    if config_arg:
        ap.add_argument("config", type=argparse.FileType("r"))

    if info_arg:
        ap.add_argument("info", type=argparse.FileType("r"))

    return ap.parse_args()


def create_mesh_cli():
    """

    """
    from polar_route.mesh import Mesh

    args = get_args()
    config = json.load(args.config)

    # Discrete Meshing
    cg = Mesh(config)
    info = cg.to_json()
    json.dump(info, args.output)


def add_vehicle_cli():
    """

    """
    from polar_route.vessel_performance import VesselPerformance

    args = get_args(config_arg=False, info_arg=True)

    vp = VesselPerformance(args.info)
    info = vp.to_json()
    json.dump(info, args.output)


def optimise_routes_cli():
    """

    """
    from polar_route.route_planner import RoutePlanner

    args = get_args(config_arg=False, info_arg=True)

    rp = RoutePlanner(args.info)
    rp.compute_routes()
    rp.compute_smoothed_routes()
    info = rp.to_json()
    json.dump(info, args.output)


def route_plotting_cli():
    from geoplot.interactive import Map
    import pandas as pd

    config = info_dict['config']
    mesh = pd.DataFrame(info_dict['cellboxes'])
    paths = info_dict['paths']
    waypoints = pd.DataFrame(info_dict['waypoints'])

    mp = Map(config,title='Example Test 1')
    mp.Maps(mesh,'SIC',predefined='SIC')
    mp.Maps(mesh,'Extreme Ice',predefined='Extreme Sea Ice Conc')
    mp.Maps(mesh,'Land Mask',predefined='Land Mask')
    mp.Paths(paths,'Routes',predefined='Route Traveltime Paths')
    mp.Points(waypoints,'Waypoints',names={"font_size":10.0})
    mp.MeshInfo(mesh,'Mesh Info',show=False)
    mp.show()