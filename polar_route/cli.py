import argparse
import json
import inspect
import logging
from multiprocessing.connection import wait

from polar_route import __version__ as version
from polar_route.utils import setup_logging, timed_call


@setup_logging
def get_args(
        default_output: str,
        config_arg: bool = True,
        mesh_arg: bool = False,
        waypoints_arg: bool = False):
    """
    Adds required command line arguments to all CLI entry points.

    Args:
        config_arg (bool): True if the CLI entry point requires a <config.json> file. Default is True.
        mesh_arg (bool): True if the CLI entry point requires a <mesh.json> file. Default is False.
        waypoints_arg (bool): True if the CLI entry point requires a <waypoints.csv> file. Default is False.

    Returns:

    """
    ap = argparse.ArgumentParser()

    # Optinal arguments used in all CLI entry points
    ap.add_argument("-o", "--output",
                    default=default_output,
                    help="Output file")
    ap.add_argument("-v", "--verbose",
                    default=False,
                    action="store_true",
                    help="Turn on DEBUG level logging")

    if config_arg:
        ap.add_argument("config", type=argparse.FileType("r"), 
                    help="File location of a <config.json> file")

    if mesh_arg:
        ap.add_argument("mesh", type=argparse.FileType("r"),
                    help="File location of the enviromental mesh")

    if waypoints_arg:
        ap.add_argument("waypoints", type=argparse.FileType("r"))

        # Optional arguments used when route planning.
        ap.add_argument("-p", "--path_only",
                        default=False,
                        action = "store_true",
                        help="output only the calculated paths")

        ap.add_argument("-d", "--dijkstra",
                        default=False,
                        action = "store_true",
                        help="output only the calculated paths")


    return ap.parse_args()


@timed_call
def create_mesh_cli():
    """

    """
    from polar_route.MeshBuilder import MeshBuilder
    default_output = "create_mesh.output.json"
    args = get_args(default_output)
    logging.info("{} {}".format(inspect.stack()[0][3][:-4], version))

    config = json.load(args.config)

    # Discrete Meshing
    cg = MeshBuilder(config).build_environmental_mesh()

    logging.info("Saving mesh to {}".format(args.output))
    info = cg.to_json()
    json.dump(info, open(args.output, "w"))


@timed_call
def add_vehicle_cli():
    """

    """
    from polar_route.vessel_performance import VesselPerformance

    default_output = "add_vehicle.output.json"
    args = get_args(default_output, config_arg=True, mesh_arg=True)
    logging.info("{} {}".format(inspect.stack()[0][3][:-4], version))

    mesh = json.load(args.mesh)
    vessel = json.load(args.config)

    

    vp = VesselPerformance(mesh, vessel['Vessel'])

    logging.info("Saving mesh to {}".format(args.output))
    info = vp.to_json()
    json.dump(info, open(args.output, "w"))


@timed_call
def optimise_routes_cli():
    """

    """
    from polar_route.route_planner import RoutePlanner

    args = get_args("optimise_routes.output.json",
                    config_arg=True, mesh_arg=True ,waypoints_arg= True)
    logging.info("{} {}".format(inspect.stack()[0][3][:-4], version))

    if args.path_only:
        logging.info("outputting only path to {}".format(args.output))
    else: 
        logging.info("outputting full mesh to {}".format(args.output))

    rp = RoutePlanner(args.mesh.name, args.config.name, args.waypoints.name)
    rp.compute_routes()
    info_dijkstra = rp.to_json()
    rp.compute_smoothed_routes()
    info = rp.to_json()

    if args.path_only:
        if args.dijkstra:
             json.dump(info_dijkstra['paths'], open('{}_dijkstra.json'.format('.'.join(args.output.split('.')[:-1])), 'w'))
        json.dump(info['paths'], open(args.output, 'w'))
    else:
        if args.dijkstra:
             json.dump(info_dijkstra, open('{}_dijkstra.json'.format('.'.join(args.output.split('.')[:-1])), 'w'))
        json.dump(info, open(args.output, "w"))