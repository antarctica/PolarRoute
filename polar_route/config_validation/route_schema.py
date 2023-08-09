route_schema = {
    "type": "object",
    "required": [
            "objective_function",
            "path_variables",
            "vector_names",
            "variable_speed",
            "time_unit",
            "early_stopping_criterion",
            "zero_currents"
        ],
    "additionalProperties": True,
    "properties": {
        "objective_function": {"type": "string"},
        "path_variables": {"type": "array", "items":"string"},
        "vector_names": {"type": "array", 
                         "items":"string",
                         "minContains": 2,
                         "maxContains": 2},
        "variable_speed": {"type":"boolean"},
        "time_unit": {"type": "string",
                      "enum": ["days","hours","seconds"]},
        "early_stopping_criterion":{"type": "boolean"},
        "zero_currents": {"type": "boolean"}
    }
}
