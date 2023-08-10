import argparse
import json
import inspect
import logging

from polar_route import __version__ as version
from polar_route.utils import setup_logging, timed_call, convert_decimal_days
from polar_route.mesh_generation.mesh_builder import MeshBuilder
from polar_route.vessel_performance.vessel_performance_modeller import VesselPerformanceModeller
from polar_route.route_planner import RoutePlanner
from polar_route.mesh_generation.environment_mesh import EnvironmentMesh
from polar_route.route_calc import route_calc


@setup_logging
def get_args(
        default_output: str,
        config_arg: bool = True,
        mesh_arg: bool = False,
        waypoints_arg: bool = False,
        format_arg: bool = False,
        format_conf: bool = False):
    """
    Adds required command line arguments to all CLI entry points.

    Args:
        default_output (str): The default output file location.
        config_arg (bool): True if the CLI entry point requires a <config.json> file. Default is True.
        mesh_arg (bool): True if the CLI entry point requires a <mesh.json> file. Default is False.
        waypoints_arg (bool): True if the CLI entry point requires a <waypoints.csv> file. Default is False.

    Returns:

    """
    ap = argparse.ArgumentParser()

    ap.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=version))

    # Optional arguments used in all CLI entry points
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
                    help="File location of the environmental mesh")

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
        
    if format_arg:
        ap.add_argument("format",
                        help = "Export format to transform a mesh into. Supported \
                        formats are JSON, GEOJSON, Tif")
        ap.add_argument( "-f", "--format_conf",
                        default = None,
                        help = "File location of Export to Tif configuration parameters")



    return ap.parse_args()

@timed_call
def rebuild_mesh_cli():
    """
        CLI entry point for rebuilding the mesh based on its encoded config files.
    """

    default_output = "rebuild_mesh.output.json"
    args = get_args(default_output, mesh_arg=True, config_arg=False)
    logging.info("{} {}".format(inspect.stack()[0][3][:-4], version))

    mesh_json = json.load(args.mesh)
    config = mesh_json['config']

    # rebuilding mesh...
    rebuilt_mesh = MeshBuilder(config).build_environmental_mesh()
    rebuilt_mesh_json = rebuilt_mesh.to_json()

    if 'vessel_info' in config.keys():
        vessel_config = config['vessel_info']
        vp = VesselPerformanceModeller(rebuilt_mesh_json, vessel_config)
        vp.model_accessibility()
        vp.model_performance()
        rebuilt_mesh_json = vp.to_json()

    logging.info("Saving mesh to {}".format(args.output))
    
    json.dump(rebuilt_mesh_json, open(args.output, "w"), indent=4)



@timed_call
def create_mesh_cli():
    """
        CLI entry point for the mesh construction
    """
    
    default_output = "create_mesh.output.json"
    args = get_args(default_output)
    logging.info("{} {}".format(inspect.stack()[0][3][:-4], version))

    config = json.load(args.config)

    # Discrete Meshing
    cg = MeshBuilder(config).build_environmental_mesh()

    logging.info("Saving mesh to {}".format(args.output))
    info = cg.to_json()
    json.dump(info, open(args.output, "w"), indent=4)


@timed_call
def add_vehicle_cli():
    """
        CLI entry point for the vessel performance modeller
    """

    default_output = "add_vehicle.output.json"
    args = get_args(default_output, config_arg=True, mesh_arg=True)
    logging.info("{} {}".format(inspect.stack()[0][3][:-4], version))

    mesh_json = json.load(args.mesh)
    vessel_config = json.load(args.config)

    vp = VesselPerformanceModeller(mesh_json, vessel_config)
    vp.model_accessibility()
    vp.model_performance()

    logging.info("Saving mesh to {}".format(args.output))
    info = vp.to_json()
    json.dump(info, open(args.output, "w"), indent=4)


@timed_call
def optimise_routes_cli():
    """
        CLI entry point for the route optimisation
    """

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
    
    if args.dijkstra:
        if args.path_only:
            json.dump(info_dijkstra['paths'], open('{}_dijkstra.json'.format('.'.join(args.output.split('.')[:-1])), 'w'), indent=4)
        else:
            json.dump(info_dijkstra, open('{}_dijkstra.json'.format('.'.join(args.output.split('.')[:-1])), 'w'), indent=4)
    
    rp.compute_smoothed_routes()
    info = rp.to_json()

    if args.path_only:
        json.dump(info['paths'], open(args.output, 'w'), indent=4)
    else:
        json.dump(info, open(args.output, "w"), indent=4)

@timed_call
def export_mesh_cli():
    """
        CLI entry point for exporting a mesh to standard formats.
        Currently supported formats are JSON, GEOJSON, TIF
    """
    # Default, used only by the Mesh Builder and PolarRoute
    args = get_args("mesh.json", 
                    config_arg = False, 
                    mesh_arg = True, 
                    format_arg = True)
        
    if args.format.upper() == "GEOJSON":
        args = get_args("mesh_geo.json", 
                    config_arg = False, 
                    mesh_arg = True, 
                    format_arg = True)
        
    elif args.format.upper() == "TIF":
        args = get_args("mesh.tif", 
                    config_arg = False, 
                    mesh_arg = True, 
                    format_arg = True)
    
    logging.info("{} {}".format(inspect.stack()[0][3][:-4], version))

    mesh = json.load(args.mesh)
    env_mesh = EnvironmentMesh.load_from_json(mesh)

    logging.info(f"exporting mesh to {args.output} in format {args.format}")

    env_mesh.save(args.output, args.format , args.format_conf)

@timed_call
def calculate_route_cli():
    """
        CLI entry point to calculate the cost of a manually defined route within an existing mesh.
    """
    args = get_args("calculated_route.json",
                    config_arg = False, mesh_arg = True, waypoints_arg = True)
    logging.info("{} {}".format(inspect.stack()[0][3][:-4], version))

    logging.info(f"Calculating the cost of route {args.waypoints.name} from mesh {args.mesh.name}")

    calc_route = route_calc(args.waypoints.name, args.mesh.name)

    if calc_route is not None:
        max_time = convert_decimal_days(calc_route["features"][0]["properties"]["traveltime"][-1])
        max_fuel = round(calc_route["features"][0]["properties"]["fuel"][-1],2)

        logging.info(f"Calculated route has travel time: {max_time} and fuel cost: {max_fuel} tons")

        logging.info(f"Saving calculated route to {args.output}")
        with open(args.output, "w") as f:
            json.dump(calc_route, f, indent=4)



