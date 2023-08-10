region_schema = {
    "type": "object",
    "required": ["latMin",  "latMax", 
                 "longMin", "longMax",
                 "startTime", "endTime",
                 "cellWidth", "cellHeight"],
    "additionalProperties": False,
    "properties":{    
        "latMin": {"type": "number"},
        "latMax": {"type": "number"},
        "longMin": {"type": "number"},
        "longMax": {"type": "number"},
        "startTime": {"type": "string"},
        "endTime": {"type": "string"},
        "cellWidth": {"type": "number"},
        "cellHeight": {"type": "number"}
    }
}

dataloader_schema = {
    "type": "object",
    "required": ["loader", "params"],
    "additionalProperties": False,
    "properties":{
        "loader": {"type": "string"},
        "params": {"type": "object"}
        
    }
}

splitting_schema = {
    "type": "object",
    "required": ["split_depth", "minimum_datapoints"],
    "additionalProperties": False,
    "properties":{
        "split_depth": {"type": "integer"},
        "minimum_datapoints": {"type": "integer"}
    } 
}

mesh_schema = {
    "type": "object",
    "required": ["Mesh_info"],
    "additionalProperties": False,
    "properties":{
        "Mesh_info": {
            "type": "object",
            "required": ["Region", "Data_sources", "splitting"],
            "additionalProperties": False,
            "properties":{
                "Region": {
                    "$ref": "#/region_schema"
                },
                "Data_sources": {
                    "type": "array",
                    "items": {
                        "$ref": "#/dataloader_schema"
                    },
                },
                "splitting": {
                    "$ref": "#/splitting_schema"
                }
            }
        }
    },
    "region_schema": region_schema,
    "dataloader_schema": dataloader_schema,
    "splitting_schema": splitting_schema
}