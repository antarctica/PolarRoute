"""
Shared utilities and helper functions for regression tests.

This module provides utilities for:
- Test file discovery (routes, meshes)
- Data extraction from JSON structures
- Route comparison functions
- Vessel/mesh comparison functions
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from polar_route.utils import round_to_sigfig
from polar_route.route_planner.route_planner import RoutePlanner
from polar_route.vessel_performance.vessel_performance_modeller import (
    VesselPerformanceModeller,
)

# Configure logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

SIG_FIG_TOLERANCE = 4


# Test File Discovery
def get_test_data_path(relative_path: str) -> str:
    """Convert relative path to absolute path based on test file location"""
    test_dir = Path(__file__).parent
    return str(test_dir / relative_path.lstrip("./").lstrip("/"))


def discover_test_files(pattern: str) -> List[str]:
    """Discover test files matching a glob pattern"""
    test_dir = Path(__file__).parent
    return [str(p) for p in test_dir.glob(pattern)]


def get_route_test_files(route_type: str) -> List[str]:
    """Get all test route files for a specific route type (dijkstra or smoothed)"""
    pattern = f"example_routes/{route_type}/**/*.json"
    return discover_test_files(pattern)


def get_mesh_test_files() -> Tuple[List[str], List[str]]:
    """Get input and output mesh files for vessel tests"""
    input_pattern = "example_meshes/env_meshes/*.json"
    output_pattern = "example_meshes/vessel_meshes/*.json"

    input_meshes = discover_test_files(input_pattern)
    output_meshes = discover_test_files(output_pattern)

    # Sort to ensure consistent pairing
    input_meshes.sort()
    output_meshes.sort()

    return input_meshes, output_meshes


# Route and Mesh Calculation Functions
def calculate_dijkstra_route(config: Dict, mesh: Dict) -> Dict:
    """
    Calculate optimised route using Dijkstra algorithm.

    Args:
        config: Route configuration dictionary
        mesh: Reference mesh including routes and waypoints

    Returns:
        Updated mesh with calculated Dijkstra paths
    """
    start = time.perf_counter()

    waypoints = pd.DataFrame(mesh["waypoints"])
    rp = RoutePlanner(mesh, config)
    routes = rp.compute_routes(waypoints)

    new_route = mesh
    new_route["paths"] = {"type": "FeatureCollection", "features": []}
    new_route["paths"]["features"] = [r.to_json() for r in routes]

    end = time.perf_counter()
    LOGGER.info(f"Route calculated in {end - start:.2f} seconds")

    return new_route


def calculate_smoothed_route(config: Dict, mesh: Dict) -> Dict:
    """
    Calculate smoothed route using Dijkstra followed by smoothing.

    Args:
        config: Route configuration dictionary
        mesh: Reference mesh including routes and waypoints

    Returns:
        Updated mesh with smoothed paths
    """
    start = time.perf_counter()

    waypoints = pd.DataFrame(mesh["waypoints"])
    rp = RoutePlanner(mesh, config)
    rp.compute_routes(waypoints)
    smoothed_route = rp.compute_smoothed_routes()

    new_route = mesh
    new_route["paths"] = smoothed_route

    end = time.perf_counter()
    LOGGER.info(f"Route smoothed in {end - start:.2f} seconds")

    return new_route


def calculate_vessel_mesh(config: Dict) -> Dict:
    """
    Create vessel performance mesh from environmental mesh and vessel config.

    Args:
        config: Combined configuration including environmental mesh and vessel_info

    Returns:
        Newly generated vessel mesh
    """
    start = time.perf_counter()

    # Extract vessel config from combined config
    vessel_config = config["config"]["vessel_info"]

    vessel_modeller = VesselPerformanceModeller(config, vessel_config)
    vessel_modeller.model_accessibility()
    vessel_modeller.model_performance()

    end = time.perf_counter()

    cellbox_count = len(vessel_modeller.env_mesh.agg_cellboxes)
    LOGGER.info(
        f"Vessel simulated against {cellbox_count} cellboxes in {end - start:.2f} seconds"
    )

    return vessel_modeller.to_json()


# Route Comparison Helper Functions
def _compare_property_arrays(
    route_a: Dict, route_b: Dict, property_name: str, should_round: bool = True
) -> None:
    """
    Generic function to compare numeric arrays in route properties.

    Args:
        route_a: First route
        route_b: Second route
        property_name: Name of the property to compare
        should_round: Whether to round values to significant figures

    Raises:
        AssertionError: If arrays differ beyond tolerance
    """
    path_a = route_a["paths"]["features"][0]["properties"]
    path_b = route_b["paths"]["features"][0]["properties"]

    values_a = path_a[property_name]
    values_b = path_b[property_name]

    if should_round:
        values_a = round_to_sigfig(values_a, sigfig=SIG_FIG_TOLERANCE)
        values_b = round_to_sigfig(values_b, sigfig=SIG_FIG_TOLERANCE)

    np.testing.assert_array_equal(
        values_a, values_b, err_msg=f'Difference in "{property_name}"'
    )


def _compare_optional_property(
    route_a: Dict, route_b: Dict, property_name: str
) -> None:
    """
    Compare property that may not exist in all route types.

    Args:
        route_a: First route
        route_b: Second route
        property_name: Name of the property to compare
    """
    path_a = route_a["paths"]["features"][0]["properties"]
    path_b = route_b["paths"]["features"][0]["properties"]

    # Skip if property not present in either route
    if property_name not in path_a or property_name not in path_b:
        return

    values_a = path_a[property_name]
    values_b = path_b[property_name]

    # Convert to int if CellIndices
    if property_name == "CellIndices":
        values_a = [int(x) for x in values_a]
        values_b = [int(x) for x in values_b]

    np.testing.assert_array_equal(
        values_a, values_b, err_msg=f'Difference in "{property_name}"'
    )


def compare_route_coordinates(route_a: Dict, route_b: Dict) -> None:
    """Compare route node coordinates."""
    coords_a = route_a["paths"]["features"][0]["geometry"]["coordinates"]
    coords_b = route_b["paths"]["features"][0]["geometry"]["coordinates"]

    assert len(coords_a) == len(coords_b), (
        f"Number of nodes different! Expected {len(coords_a)}, got {len(coords_b)}"
    )

    for axis in [0, 1]:  # x and y coordinates
        rounded_a = round_to_sigfig(coords_a[:][axis], sigfig=SIG_FIG_TOLERANCE)
        rounded_b = round_to_sigfig(coords_b[:][axis], sigfig=SIG_FIG_TOLERANCE)
        np.testing.assert_array_equal(
            rounded_a, rounded_b, err_msg='Difference in "route_coordinates"'
        )


def compare_waypoint_names(route_a: Dict, route_b: Dict) -> None:
    """Compare source and destination waypoint names."""
    path_a = route_a["paths"]["features"][0]["properties"]
    path_b = route_b["paths"]["features"][0]["properties"]

    assert path_a["from"] == path_b["from"], (
        f"Waypoint source names don't match! Expected {path_a['from']}, got {path_b['from']}"
    )
    assert path_a["to"] == path_b["to"], (
        f"Waypoint destination names don't match! Expected {path_a['to']}, got {path_b['to']}"
    )


# Simple property comparison wrappers
def compare_time(route_a, route_b):
    return _compare_property_arrays(route_a, route_b, "traveltime")


def compare_fuel(route_a, route_b):
    return _compare_property_arrays(route_a, route_b, "fuel")


def compare_battery(route_a, route_b):
    return _compare_property_arrays(route_a, route_b, "battery")


def compare_distance(route_a, route_b):
    return _compare_property_arrays(route_a, route_b, "distance")


def compare_speed(route_a, route_b):
    return _compare_property_arrays(route_a, route_b, "speed")


def compare_cell_indices(route_a, route_b):
    return _compare_optional_property(route_a, route_b, "CellIndices")


def compare_cases(route_a, route_b):
    return _compare_optional_property(route_a, route_b, "cases")


# Vessel/Mesh Comparison Helper Functions
def _compare_set_difference(set_a: set, set_b: set) -> Tuple[List, List]:
    """Helper to compute and return set differences for error messages."""
    missing_from_a = list(set_b - set_a)
    missing_from_b = list(set_a - set_b)
    return missing_from_a, missing_from_b


def compare_cellbox_count(mesh_a: Dict, mesh_b: Dict) -> None:
    """Compare number of cellboxes between meshes."""
    assert len(mesh_a["cellboxes"]) == len(mesh_b["cellboxes"]), (
        f"Incorrect number of cellboxes. Expected: {len(mesh_a['cellboxes'])}, got: {len(mesh_b['cellboxes'])}"
    )


def compare_cellbox_ids(mesh_a: Dict, mesh_b: Dict) -> None:
    """Compare cellbox IDs between meshes."""
    ids_a = {cb["id"] for cb in mesh_a["cellboxes"]}
    ids_b = {cb["id"] for cb in mesh_b["cellboxes"]}

    missing_from_a, missing_from_b = _compare_set_difference(ids_a, ids_b)

    assert ids_a == ids_b, (
        f"Mismatch in cellbox IDs. New IDs: {missing_from_a}, Missing IDs: {missing_from_b}"
    )


def compare_cellbox_values(mesh_a: Dict, mesh_b: Dict) -> None:
    """Compare cellbox attribute values between meshes."""
    df_a = pd.DataFrame(mesh_a["cellboxes"]).set_index("geometry")
    df_b = pd.DataFrame(mesh_b["cellboxes"]).set_index("geometry")

    # Extract only common boundaries, drop ID as it may differ
    bounds_a = [cb["geometry"] for cb in mesh_a["cellboxes"]]
    bounds_b = [cb["geometry"] for cb in mesh_b["cellboxes"]]
    common_bounds = [geom for geom in bounds_a if geom in bounds_b]
    df_a = df_a.loc[common_bounds].drop(columns=["id"])
    df_b = df_b.loc[common_bounds].drop(columns=["id"])

    # Find common columns only (exclude columns that only appear in one dataframe)
    common_cols = df_a.columns.intersection(df_b.columns)
    
    # Log which columns are being compared vs excluded
    excluded_a = set(df_a.columns) - set(common_cols)
    excluded_b = set(df_b.columns) - set(common_cols)
    if excluded_a or excluded_b:
        LOGGER.debug(f"Comparing only common columns. Excluded from old: {excluded_a}, from new: {excluded_b}")
    
    df_a = df_a[common_cols]
    df_b = df_b[common_cols]

    # Round float values and floats in lists
    for df in [df_a, df_b]:
        # Round float columns
        for col in df.select_dtypes(include=float).columns:
            df[col] = round_to_sigfig(df[col].to_numpy(), sigfig=SIG_FIG_TOLERANCE)

        # Round floats within list columns
        for col in df.select_dtypes(include=list).columns:
            df[col] = df[col].apply(
                lambda val: round_to_sigfig(val, sigfig=SIG_FIG_TOLERANCE)
                if isinstance(val, list) and all(isinstance(x, float) for x in val)
                else val
            )

    diff = df_a.compare(df_b).rename({"self": "old", "other": "new"})
    
    # Provide more informative error message
    if len(diff) > 0:
        # Count differences by column  
        diff_counts = {}
        diff_samples = {}
        for col in diff.columns.get_level_values(0).unique():
            col_diff = diff[col]
            non_null_mask = ~col_diff.isna().all(axis=1)
            diff_counts[col] = non_null_mask.sum()
            # Get sample of differences for this column
            if diff_counts[col] > 0:
                sample_rows = col_diff[non_null_mask].head(3)
                diff_samples[col] = sample_rows.to_dict('records')
        
        error_msg = f"\nMismatch in {len(diff)} cellboxes across {len(diff_counts)} columns:\n"
        error_msg += f"\nDifferences by column:\n"
        for col, count in sorted(diff_counts.items()):
            error_msg += f"  {col}: {count} differences\n"
            if col in diff_samples:
                error_msg += f"    Sample: {diff_samples[col][:2]}\n"
        
        error_msg += f"\nTo see full differences, check the test output or inspect the meshes directly."
        
        assert False, error_msg


def compare_cellbox_attributes(mesh_a: Dict, mesh_b: Dict) -> None:
    """Compare cellbox attribute names between meshes."""
    attrs_a = set(mesh_a["cellboxes"][0].keys())
    attrs_b = set(mesh_b["cellboxes"][0].keys())

    missing_from_a, missing_from_b = _compare_set_difference(attrs_a, attrs_b)

    assert attrs_a == attrs_b, (
        f"Mismatch in cellbox attributes. New: {missing_from_a}, Missing: {missing_from_b}"
    )


def compare_neighbour_graph_count(mesh_a: Dict, mesh_b: Dict) -> None:
    """Compare number of nodes in neighbour graphs."""
    assert len(mesh_a["neighbour_graph"]) == len(mesh_b["neighbour_graph"]), (
        f"Incorrect node count. Expected: {len(mesh_a['neighbour_graph'])}, got: {len(mesh_b['neighbour_graph'])}"
    )


def compare_neighbour_graph_ids(mesh_a: Dict, mesh_b: Dict) -> None:
    """Compare node IDs in neighbour graphs."""
    ids_a = set(mesh_a["neighbour_graph"].keys())
    ids_b = set(mesh_b["neighbour_graph"].keys())

    missing_from_a, missing_from_b = _compare_set_difference(ids_a, ids_b)

    assert ids_a == ids_b, (
        f"Mismatch in graph nodes. New: {len(missing_from_a)}, Missing: {len(missing_from_b)}"
    )


def compare_neighbour_graph_values(mesh_a: Dict, mesh_b: Dict) -> None:
    """Compare neighbour relationships in graphs."""
    graph_a = mesh_a["neighbour_graph"]
    graph_b = mesh_b["neighbour_graph"]

    mismatches = {}
    for node in graph_a.keys():
        if node in graph_b:  # Node existence checked by compare_neighbour_graph_ids
            # Sort neighbours as order doesn't matter
            sorted_a = {k: sorted(v) for k, v in graph_a[node].items()}
            sorted_b = {k: sorted(v) for k, v in graph_b[node].items()}

            if sorted_a != sorted_b:
                mismatches[node] = sorted_b

    assert len(mismatches) == 0, (
        f"Mismatch in neighbour relationships. {len(mismatches)} nodes changed."
    )
