import os
import json
import pytest
import pandas as pd
from io import StringIO
from jsonschema.exceptions import ValidationError

from polar_route.config_validation.config_validator import (
    validate_vessel_config,
    validate_route_config,
    validate_waypoints,
)

EXAMPLES_DIR = os.path.join("examples")
WAYPOINTS_COLUMNS = ["Name", "Lat", "Long", "Source", "Destination"]


def load_example(path, loader_func):
    """Load an example file or skip the test if it doesn't exist."""
    if not os.path.exists(path):
        pytest.skip(f"Example file not found: {path}")
    return loader_func(path)


def load_json_example(path):
    return load_example(path, lambda p: json.load(open(p)))


def load_csv_example(path):
    return load_example(path, pd.read_csv)


@pytest.fixture
def valid_vessel_config():
    """Fixture providing a minimal valid vessel config."""
    return {
        "vessel_class": "ship",
        "max_speed": 26.5,
        "unit": "km/hr",
        "consumption_model": {
            "type": "polynomial_fuel",
            "params": {
                "speed_coeffs": [0.001, -0.003, 0.25],
                "resistance_coeffs": [7.75e-11, 6.48e-06]
            }
        }
    }


@pytest.fixture
def valid_route_config():
    """Fixture providing a minimal valid route config."""
    return {
        "objective_function": "fuel_use",
        "path_variables": ["lat", "lon"],
        "vector_names": ["current_u", "current_v"],
        "time_unit": "hours",
    }


@pytest.fixture
def valid_waypoints_df():
    """Fixture providing a minimal valid waypoints DataFrame."""
    return pd.read_csv(
        StringIO(
            "Name,Lat,Long,Source,Destination\nWP1,60.0,-45.0,X,\nWP2,61.0,-44.0,,X"
        )
    )


# Vessel config validation
def test_validate_vessel_config_file():
    """Test that the example vessel config file validates successfully."""
    config = load_json_example(
        os.path.join(EXAMPLES_DIR, "vessel_config", "SDA.config.json")
    )
    validate_vessel_config(config)


@pytest.mark.parametrize(
    "config, match",
    [
        ({"vessel_class": "ship", "unit": "km/hr", "consumption_model": {"type": "polynomial_fuel", "params": {"speed_coeffs": [0.001, -0.003, 0.25], "resistance_coeffs": [7.75e-11, 6.48e-06]}}}, "max_speed"),
        ({"vessel_class": "ship", "unit": "km/hr", "max_speed": "fast", "consumption_model": {"type": "polynomial_fuel", "params": {"speed_coeffs": [0.001, -0.003, 0.25], "resistance_coeffs": [7.75e-11, 6.48e-06]}}}, "max_speed"),
        ({"vessel_class": None, "max_speed": 26.5, "unit": "km/hr", "consumption_model": {"type": "polynomial_fuel", "params": {"speed_coeffs": [0.001, -0.003, 0.25], "resistance_coeffs": [7.75e-11, 6.48e-06]}}}, "vessel_class"),
        ({"vessel_class": "ship", "max_speed": None, "unit": "km/hr", "consumption_model": {"type": "polynomial_fuel", "params": {"speed_coeffs": [0.001, -0.003, 0.25], "resistance_coeffs": [7.75e-11, 6.48e-06]}}}, "max_speed"),
        ({"vessel_class": "ship", "max_speed": 26.5, "unit": None, "consumption_model": {"type": "polynomial_fuel", "params": {"speed_coeffs": [0.001, -0.003, 0.25], "resistance_coeffs": [7.75e-11, 6.48e-06]}}}, "unit"),
        ({}, ""),  # Empty dict
    ],
)
def test_validate_vessel_config_invalid(config, match):
    """Test that ValidationError is raised for missing/invalid fields."""
    with pytest.raises(ValidationError, match=match):
        validate_vessel_config(config)


def test_validate_vessel_config_not_dict():
    """Test that a non-dictionary vessel config raises TypeError."""
    with pytest.raises(TypeError):
        validate_vessel_config(["not", "a", "dict"])


def test_validate_vessel_config_no_extra_keys(valid_vessel_config):
    """Test that unexpected fields in vessel config are NOT allowed (changed from legacy)."""
    config = valid_vessel_config.copy()
    config["additional_field"] = "some_value"
    # New schema has additionalProperties: False, so this should fail
    with pytest.raises(ValidationError, match="additional_field"):
        validate_vessel_config(config)


# Route config validation testing
def test_validate_route_config_file():
    """Test that the example route config file validates successfully."""
    config = load_json_example(
        os.path.join(EXAMPLES_DIR, "route_config", "all_options.config.json")
    )
    validate_route_config(config)


@pytest.mark.parametrize(
    "config, match",
    [
        ({"objective_function": "fuel_use"}, "path_variables"),
        (
            {
                "objective_function": "fuel_use",
                "path_variables": "lat,lon",
                "vector_names": ["current_u", "current_v"],
            },
            "path_variables",
        ),
        (
            {
                "objective_function": "fuel_use",
                "path_variables": ["lat", "lon"],
                "vector_names": ["current_u", "current_v"],
                "time_unit": "weeks",
            },
            "weeks",
        ),
        ({}, ""),  # Empty dict
    ],
)
def test_validate_route_config_invalid(config, match):
    """Test that ValidationError is raised for invalid structure or enums."""
    with pytest.raises(ValidationError, match=match):
        validate_route_config(config)


def test_validate_route_config_not_dict():
    """Test that a non-dictionary route config raises TypeError."""
    with pytest.raises(TypeError):
        validate_route_config(["not", "a", "dict"])


def test_validate_route_config_extra_keys(valid_route_config):
    """Test that unexpected fields in route config are allowed as expected."""
    config = valid_route_config.copy()
    config["additional_field"] = "some value"
    validate_route_config(config)


# Waypoint config validation
def test_validate_waypoints_file():
    """Test that the example waypoints CSV validates successfully."""
    df = load_csv_example(os.path.join(EXAMPLES_DIR, "waypoints_example.csv"))
    validate_waypoints(df)


@pytest.mark.parametrize(
    "csv_content, match",
    [
        ("Index,Name,Lat,Source\n0,WP1,60.0,X", "Expected the following columns"),
        (
            "Index,Name,Lat,Long,Source,Destination\n0,WP1,60.0,-45.0,,X",
            "No source waypoint defined!",
        ),
        (
            "Index,Name,Lat,Long,Source,Destination\n0,WP1,60.0,-45.0,X,",
            "No destination waypoint defined!",
        ),
        (
            "Index,Name,Lat,Long,Source,Destination\n0,WP1,sixty,-45.0,X,\n1,WP2,61.0,not_a_number,,X",
            'Non-numeric value in "Lat" column',
        ),
        (
            "Name,Lat,Long,Source,Destination",
            "No source waypoint defined!",
        ),  # Empty with columns
    ],
)
def test_validate_waypoints_invalid(csv_content, match):
    """Test that AssertionError is raised for invalid waypoint CSV content."""
    df = pd.read_csv(StringIO(csv_content))
    with pytest.raises(AssertionError, match=match):
        validate_waypoints(df)


def test_validate_waypoints_not_csv():
    """Test that a non-DataFrame input raises TypeError."""
    with pytest.raises(TypeError):
        validate_waypoints(["not", "a", "DataFrame"])


def test_validate_waypoints_empty_df():
    """Test that empty DataFrame raises AssertionError on missing columns."""
    with pytest.raises(AssertionError, match="Expected the following columns"):
        validate_waypoints(pd.DataFrame())


def test_validate_waypoints_extra_columns(valid_waypoints_df):
    """Test that waypoints validation ignores extra columns as expected."""
    df = valid_waypoints_df.copy()
    df["ExtraCol"] = ["foo", "bar"]
    validate_waypoints(df)
