import json
import jsonschema
import pandas as pd

from polar_route.config_validation.vessel_schema import vessel_schema
from polar_route.config_validation.route_schema import route_schema
from polar_route.config_validation.waypoints_schema import waypoints_columns

def flexi_json_input(config):
    """
    Allows flexible inputs. If a string is parsed, then assume it's a file path
    and read in as a json. If a dict is parsed, then assume it's already a 
    valid loaded json, and return it as is

    Args:
        config (str or dict): Input to translate into a dict.

    Raises:
        TypeError: If input is neither a str nor a dict, then wrong input type

    Returns:
        dict: Dictionary read from JSON
    """
    if type(config) is str:
        # If str, assume filename
        with open(config, 'r') as fp:
            config_json = json.load(fp)
    elif type(config) is dict:
        # If dict, assume it's the config
        config_json = config
    else:
        # Otherwise, can't deal with it
        raise TypeError(f"Expected 'str' or 'dict', instead got '{type(config)}'")
    
    return config_json
    
def validate_vessel_config(config):
    """
    Validates a vessel config

    Args:
        config (str or dict):
            Vessel config to be validated.
            If type 'str', tries to read in as a filename and open file as json
            If type 'dict', assumes it's already read in from a json file

    Raises:
        TypeError: Incorrect config parsed in. Must be 'str' or 'dict'
        FileNotFoundError: Could not read in file if 'str' parsed
        ValidationError: Malformed vessel config

    """
    # Deals with flexible input
    config_json = flexi_json_input(config)
    # Validate against schema
    jsonschema.validate(instance=config_json, schema=vessel_schema)

def validate_route_config(config):
    """
    Validates a route config

    Args:
        config (str or dict):
            route config to be validated.
            If type 'str', tries to read in as a filename and open file as json
            If type 'dict', assumes it's already read in from a json file

    Raises:
        TypeError: Incorrect config parsed in. Must be 'str' or 'dict'
        FileNotFoundError: Could not read in file if 'str' parsed
        ValidationError: Malformed route config

    """
    # Deals with flexible input
    config_json = flexi_json_input(config)
    # Validate against schema
    jsonschema.validate(instance=config_json, schema=route_schema)

def validate_waypoints(waypoints):
    """_summary_

    Args:
        waypoints (str or pd.DataFrame): _description_

    Raises:
        TypeError: Incorrect config parsed in. Must be 'str' or 'pd.DataFrame' 
        FileNotFoundError: Could not read in file if 'str' parsed
        AssertionError: Malformed waypoints config
    """

    if type(waypoints) is str:
        # If str, assume filename
        waypoints_df = pd.read_csv(waypoints).reset_index()
    elif type(waypoints) is pd.core.frame.DataFrame:
        # If dataframe, assume it's the loaded waypoints
        waypoints_df = waypoints.reset_index()
    else:
        # Otherwise, can't deal with it
        raise TypeError(
            f"Expected 'str' or 'dict', instead got '{type(waypoints)}'"
            )
    # Assert that all the required columns exist
    assert(all(col in waypoints_df.columns for col in waypoints_columns)), \
        f'Expected the following columns to exist: {waypoints_columns}'
    
    # Assert that at least one source and destination waypoint selected
    assert('X' in set(waypoints_df['Source'])), \
        'No source waypoint defined!'
    assert('X' in set(waypoints_df['Destination'])), \
        'No destination waypoint defined!'
        
    # Assert only numeric values given as coordinates
    assert(pd.to_numeric(waypoints_df['Lat'], errors='coerce').notnull().all()), \
        'Non-numeric value in "Lat" column of waypoints'
    assert(pd.to_numeric(waypoints_df['Long'], errors='coerce').notnull().all()), \
        'Non-numeric value in "Long" column of waypoints'

