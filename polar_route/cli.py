import argparse
import json
import inspect
import logging
import geopandas as gpd

from meshiphi.mesh_generation.mesh_builder import MeshBuilder

from polar_route import __version__ as version
from polar_route.utils import setup_logging, timed_call, convert_decimal_days, to_chart_track_csv
from polar_route.vessel_performance.vessel_performance_modeller import VesselPerformanceModeller
from polar_route.route_planner import RoutePlanner
from polar_route.route_calc import route_calc


@setup_logging
def get_args(
        default_output: str,
        config_arg: bool = True,
        mesh_arg: bool = False,
        waypoints_arg: bool = False
        ):
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
        ap.add_argument("-p", "--path_geojson",
                        default=False,
                        action = "store_true",
                        help="Output the calculated paths as GeoJSON")

        ap.add_argument("-d", "--dijkstra",
                        default=False,
                        action = "store_true",
                        help="Output dijkstra paths")
        
        ap.add_argument("--chart_tracker",
                        default=False,
                        action = "store",
                        help="Output the calculated paths as CSV readable by ChartTracker")

    return ap.parse_args()


@timed_call
def resimulate_vehicle_cli():
    """
        CLI entry point for rebuilding the mesh based on its encoded config files.
    """

    default_output = "resimulate_vehicle_output.vessel.json"
    
    args = get_args(default_output, mesh_arg=True, config_arg=False)
    logging.info("{} {}".format(inspect.stack()[0][3][:-4], version))

    mesh_json = json.load(args.mesh)
    mesh_config = mesh_json['config']['mesh_info']

    # Rebuilding mesh, since pruned CB's don't exist in input file
    rebuilt_mesh = MeshBuilder(mesh_config).build_environmental_mesh()
    rebuilt_mesh_json = rebuilt_mesh.to_json()

    # Resimulating vessel
    vessel_config = mesh_json['config']['vessel_info']
    vp = VesselPerformanceModeller(rebuilt_mesh_json, vessel_config)
    vp.model_accessibility()
    vp.model_performance()
    rebuilt_mesh_json = vp.to_json()

    # Saving output
    logging.info(f"Saving mesh to {args.output}")
    with open(args.output, 'w+') as fp:
        json.dump(rebuilt_mesh_json, fp, indent=4)


@timed_call
def add_vehicle_cli():
    """
        CLI entry point for the vessel performance modeller
    """

    default_output = "add_vehicle_output.vessel.json"
    args = get_args(default_output, config_arg=True, mesh_arg=True)
    logging.info("{} {}".format(inspect.stack()[0][3][:-4], version))

    mesh_json = json.load(args.mesh)
    vessel_config = json.load(args.config)

    vp = VesselPerformanceModeller(mesh_json, vessel_config)
    vp.model_accessibility()
    vp.model_performance()

    info = vp.to_json()
    logging.info(f"Saving vp mesh to {args.output}")
    with open(args.output, 'w+') as fp:
        json.dump(info, fp, indent=4)


@timed_call
def optimise_routes_cli():
    """
        CLI entry point for the route optimisation
    """
    args = get_args("optimise_routes_output.route.json",
                    config_arg=True, mesh_arg=True ,waypoints_arg= True)
    logging.info("{} {}".format(inspect.stack()[0][3][:-4], version))

    rp = RoutePlanner(args.mesh.name, args.config.name, args.waypoints.name)
    
    output_file = args.output
    output_file_strs = output_file.split('.')
    
    logging.info("Calculating dijkstra routes")
    rp.compute_routes()
    info_dijkstra = rp.to_json()
    
    
    if args.dijkstra:
        # Form a unique name for the dijkstra output
        dijkstra_output_file_strs = output_file_strs
        dijkstra_output_file_strs[-2] += '_dijkstra'
        
        dijkstra_output_file = '.'.join(dijkstra_output_file_strs)
        logging.info(f"\tOutputting dijkstra path to {dijkstra_output_file}")
        with open(dijkstra_output_file, 'w+') as fp:
            json.dump(info_dijkstra, fp, indent=4)
        # Create GeoJSON filename
        if args.path_geojson:
            dijkstra_output_file_strs[-1] = 'geojson'
            dijkstra_output_file = '.'.join(dijkstra_output_file_strs)
            logging.info(f"\tExtracting standalone dijkstra path GeoJSON to {dijkstra_output_file}")
            with open(dijkstra_output_file, 'w+') as fp:
                json.dump(info_dijkstra['paths'], fp, indent=4)
    
    logging.info("Calculating smoothed routes")
    rp.compute_smoothed_routes()
    info = rp.to_json()
    # Resetting output file str in case dijkstra output run
    output_file = args.output
    output_file_strs = output_file.split('.')
    logging.info(f"\tOutputting smoothed path to {output_file}")
    with open(output_file, 'w+') as fp:
        json.dump(info, fp, indent=4)
    if args.path_geojson:
        # Create GeoJSON filename
        output_file_strs[-1] = 'geojson'
        output_file = '.'.join(output_file_strs)
        logging.info(f"\tExtracting standalone path GeoJSON to {output_file}")
        with open(output_file, 'w+') as fp:
                json.dump(info['paths'], fp, indent=4)
    # If want charttracker formatted csv
    if args.chart_tracker:
        # Extract each route as csv string
        route_name = 'PolarRoute' if args.chart_tracker == True else args.chart_tracker
        csv_strs = rp.to_charttracker_csv(route_name=route_name)
        # Format output filename
        output_file_strs[-1] = 'csv'
        output_file_strs.insert(1,'r0')
        # For each path generated, write to csv with unique name
        for i, csv_str in enumerate(csv_strs):
            output_file_strs[1] = f'r{i}'
            output_file = '.'.join(output_file_strs)
            logging.info(f"\tOutputting ChartTracker CSV to {output_file}")
            with open(output_file, 'w+') as fp:
                fp.write(csv_str)
        
@timed_call
def extract_routes_cli():
    """
        CLI entry point to extract individual routes from the output of optimise_routes
    """
    args = get_args("extracted_route.json", config_arg = False, mesh_arg = True)

    output_file = args.output
    output_file_strs = output_file.split('.')

    route_file = json.load(args.mesh)
    logging.info(f"Extracting routes from: {args.mesh.name} with base output: {args.output}")

    # Check if input is just a route file or if the routes are nested within a mesh
    if route_file.get("type") == "FeatureCollection":
        routes = route_file["features"]
    else:
        routes = route_file["paths"]["features"]

    if output_file_strs[-1] in ["json", "geojson"]:
        logging.info("Extracting routes in geojson format")
        for route in routes:
            geojson_wrapper = {"type": "FeatureCollection",
                               "features": []}
            from_wp = route["properties"]["from"].replace(" ", "_")
            to_wp = route["properties"]["to"].replace(" ", "_")
            geojson_output = geojson_wrapper
            geojson_output["features"].append(route)
            route_output_str = '.'.join(output_file_strs[:-1]) + "_" + from_wp + to_wp + "." + output_file_strs[-1]
            logging.info(f"Saving route to {route_output_str}")
            with open(route_output_str, "w") as f:
                json.dump(geojson_output, f, indent=4)


    elif output_file_strs[-1] == "gpx":
        logging.info("Extracting routes in gpx format")
        for route in routes:
            from_wp = route["properties"]["from"].replace(" ", "_")
            to_wp = route["properties"]["to"].replace(" ", "_")
            route_output_str = '.'.join(output_file_strs[:-1]) + "_" + from_wp + to_wp + ".gpx"
            gdf = gpd.GeoDataFrame.from_features([route])
            logging.info(f"Saving route to {route_output_str}")
            gdf['geometry'].to_file(route_output_str, "GPX")
    elif output_file_strs[-1] == "csv":
        logging.info("Extracting routes in ChartTrack csv format")
        for route in routes:
            from_wp = route["properties"]["from"].replace(" ", "_")
            to_wp = route["properties"]["to"].replace(" ", "_")
            route_output_str = '.'.join(output_file_strs[:-1]) + "_" + from_wp + to_wp + ".csv"
            csv_route = to_chart_track_csv(route)
            logging.info(f"Saving route to {route_output_str}")
            with open(route_output_str, "w") as f:
                f.write(csv_route)
    else:
        logging.warning("Unrecognised output type! No routes have been extracted!")
        

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
        with open(args.output, 'w+') as f:
            json.dump(calc_route, f, indent=4)