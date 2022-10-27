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
        info_arg: bool = False,
        waypoints_arg: bool = False):
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
        ap.add_argument("config", type=argparse.FileType("r"), 
                    help="File location of configuration file used to build the mesh")

    if info_arg:
        ap.add_argument("info", type=argparse.FileType("r"),
                    help="File location of the enviromental mesh")

    if waypoints_arg:
        ap.add_argument("waypoints", type=argparse.FileType("r"))
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
    from polar_route.mesh import Mesh

    args = get_args("create_mesh.output.json")
    logging.info("{} {}".format(inspect.stack()[0][3][:-4], version))

    config = json.load(args.config)

    # Discrete Meshing
    cg = Mesh(config)

    logging.info("Saving mesh to {}".format(args.output))
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

    logging.info("Saving mesh to {}".format(args.output))
    info = vp.to_json()
    json.dump(info, open(args.output, "w"))


@timed_call
def optimise_routes_cli():
    """

    """
    from polar_route.route_planner import RoutePlanner

    args = get_args("optimise_routes.output.json",
                    config_arg=False, info_arg=True, waypoints_arg= True)
    logging.info("{} {}".format(inspect.stack()[0][3][:-4], version))

    if args.path_only:
        logging.info("outputting only path to {}".format(args.output))
    else: 
        logging.info("outputting full mesh to {}".format(args.output))

    vehicle_mesh = json.load(args.info)
    rp = RoutePlanner(vehicle_mesh, args.waypoints)
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