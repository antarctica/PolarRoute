route_schema = {
    "type": "object",
    "required": [
            "objective_function",
            "path_variables",
            "vector_names"
        ],
    "additionalProperties": True,
    "properties": {
        "objective_function": {"type": "string"},
        "path_variables": {"type": "array", 
                           "items":{"type": "string"}},
        "vector_names": {"type": "array", 
                         "items":{"type":"string"},
                         "minItems": 2,
                         "maxItems": 2},
        "time_unit": {"type": "string",
                      "enum": ["days","hours","seconds"]},
        "adjust_waypoints":{"type": "boolean"},
        "zero_currents": {"type": "boolean"},
        "fixed_speed": {"type": "boolean"},
        "smoothing_blocked_sic": {"type": "number"},
        "smoothing_max_iterations": {"type": "integer"},
        "smoothing_merge_separation": {"type": "number"},
        "smoothing_converged_sep": {"type": "number"}    
    }
}