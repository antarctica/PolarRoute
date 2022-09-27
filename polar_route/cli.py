import argparse
import json


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

    ap.add_argument("--output",
                    default=default_output,
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

    args = get_args("create_mesh.output.json")
    config = json.load(args.config)

    # Discrete Meshing
    cg = Mesh(config)
    info = cg.to_json()
    json.dump(info, open(args.output, "w"))


def add_vehicle_cli():
    """

    """
    from polar_route.vessel_performance import VesselPerformance

    args = get_args("add_vehicle.output.json", config_arg=False, info_arg=True)

    vp = VesselPerformance(args.info)
    info = vp.to_json()
    json.dump(info, open(args.output, "w"))


def optimise_routes_cli():
    """

    """
    from polar_route.route_planner import RoutePlanner

    args = get_args("optimise_routes.output.json",
                    config_arg=False, info_arg=True)

    rp = RoutePlanner(args.info)
    rp.compute_routes()
    rp.compute_smoothed_routes()
    info = rp.to_json()
    json.dump(info, open(args.output, "w"))


def route_plotting_cli():
    from geoplot.interactive import Map
    import pandas as pd

    args = get_args("routes.png", config_arg=False, info_arg=True)
    config = args.info['config']
    mesh = pd.DataFrame(args.info['cellboxes'])
    paths = args.info['paths']
    waypoints = pd.DataFrame(args.info['waypoints'])

    mp = Map(config, title='Example Test 1')
    mp.Maps(mesh, 'SIC', predefined='SIC')
    mp.Maps(mesh, 'Extreme Ice', predefined='Extreme Sea Ice Conc')
    mp.Maps(mesh, 'Land Mask', predefined='Land Mask')
    mp.Paths(paths, 'Routes', predefined='Route Traveltime Paths')
    mp.Points(waypoints, 'Waypoints', names={"font_size": 10.0})
    mp.MeshInfo(mesh, 'Mesh Info', show=False)
    mp.save(args.output)
