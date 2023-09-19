.. _outputs:

********************
Outputs - Data Types
********************

#########################
The Vessel_mesh.json file
#########################

Once a discrete mesh environment is contracted, it is then passed to the vessel performance modeller
which applies transformations which are specific to a given vehicle. These vehicle specific values 
are then encoded into the mesh json object and passed down-stream to the route planner.

::

    import json
    from polar_route.vessel_performance.vessel_performance_modeller import VesselPerformanceModeller

    with open('vessel_config.json', 'r') as f:
        vessel_config = json.load(f)

    vpm = VesselPerformanceModeller(mesh_json, vessel_config)

    vpm.model_accessibility()
    vpm.model_performance()

    vessel_mesh_json = vpm.to_json()

.. note::
    To make use of the full range of vessel performance transformations, a Mesh should be constructed with
    the following attributes:

    * elevation (available via data_loaders: *gebco*, *bsose_depth*)
    * SIC (available via data_loaders: *amsr*, *bsose_sic*, *baltic_sic*, *icenet*, *modis*)
    * thickness (available via data_loaders: *thickness*)
    * density (available via data_loaders: *density*)
    * u10, v10 (available via data_loaders: *era5_wind*)

    see section **Dataloader Overview** for more information on data_loaders

    The vessel performance modeller will still run without these attributes but will assign default values from the
    configuration file where any data is missing.


As an example, after running the vessel performance modeller with the SDA class and all relevant data each cellbox will
have a set of new attributes as follows:

* **speed** *(list)* : The speed of the vessel in that cell when travelling to each of its neighbours.
* **fuel** *(list)* : The rate of fuel consumption in that cell when travelling to each of its neighbours.
* **inaccessible** *(boolean)* : Whether the cell is considered inaccessible to the vessel for any reason.
* **land** *(boolean)* : Whether the cell is shallow enough to be considered land by the vessel.
* **ext_ice** *(boolean)* : Whether the cell has enough ice to be inaccessible to the vessel.
* **resistance** *(list)* : The total resistance force the vessel will encounter in that cell when travelling to each of its neighbours.
* **ice resistance** *(float)* : The resistance force due to ice.
* **wind resistance** *(list)* : The resistance force due to wind.
* **relative wind speed** *(list)* : The apparent wind speed acting on the vessel.
* **relative wind angle** *(list)* : The angle of the apparent wind acting on the vessel.


###################
The Route.json file
###################

During the route planning stage of the pipline information on the routes and the waypoints used are saved 
as outputs to the processing stage. Descriptions of the structure of the two outputs are given below:

=========
waypoints
=========

An entry in the json including all the information of the waypoints defined by the user from the `waypoints_path` 
file. It may be the case that ot all waypoints would have been used in the route construction, but all waypoints 
are returned to this entry. The structure of the entry follows:

:: 

    {\n
        "Name":{\n
            '0':"Falklands",\n
            '1':"Rothera",\n
            ...\n
        },\n
        "Lat":{\n
            '0':-52.6347222222,
            '1':-75.26722,\n
            ...\n
        },\n
        "Long":{\n
            ...\n
        },\n
        "index":{\n
            ...\n
        }\n
    }

where each of the values represent the following: 

* **<Name>** : The waypoint name defined by user
    * **0**  : The name of waypoint for index row '0'
    * **1**  : The name of waypoint for index row '1' etc
* **<Lat>** : The latitude of the waypoints in WGS84
    * **0**  : The latitude of waypoint for index row '0'
    * **1**  : The latitude of waypoint for index row '1' etc
* **<Long>** : The longitude of the waypoints in WGS84
    * **0**  : The longitude of waypoint for index row '0'
    * **1**  : The longitude of waypoint for index row '1' etc
* **<index>** : The index of the cellbox containing the waypoint
    * **0**  : The cellbox index of waypoint for index row '0'
    * **1**  : The cellbox index of waypoint for index row '1' etc
* **<...>** : Any additional column names defined in the original .csv that was loaded

This output can be converted to a pandas dataframe by running::
waypoints_dataframe = pd.DataFrame(waypoints) 


=====
paths
=====
An entry in the json, in a geojson format, including all the routes constructed between the user defined waypoints. The structure of this entry is as follows:

:: 

    {\n
        'types':'FeatureCollection',\n
        "features":{[\n
            'type':'feature',\n
            'geometry':{\n
                'type': 'LineString',

                'coordinates': [[-27.21694, -75.26722],\n
                                [-27.5, -75.07960297382266],\n
                                [-27.619238882768894, -75.0],\n
                                ...]\n
            },
            'properties':{\n
                'from': 'Halley',\n
                'to': 'Rothera',\n
                'traveltime': [0.0,\n
                               0.03531938671648596,\n
                               0.050310986633880575,\n
                               ...],\n
                'fuel': [0.0,\n
                         0.9648858923588642,\n
                         1.3745886107069096,\n
                         ...],\n
                'times': ['2017-01-01 00:00:00',
                          '2017-01-01 00:50:51.595036800',
                          '2017-01-01 01:12:26.869276800',
                          ...]\n
            }\n
        ]}\n
    }\n


where the output takes a GeoJSON standard form (more infor given at https://geojson.org) given by: 


* **<features>** : A list of the features representing each of the separate routes constructed
    * **geometry**  : The positioning of the route locations
        * **coordinates**  : A list of the Lat,Long position of all the route points
    * **<properties>** : A list of meta-information about the route
        * **from**  : Starting waypoint of route
        * **to**  : Ending waypoint of route
        * **traveltime** : A list of float values representing the cumulative travel time along the route. This entry was originally defined as a output in the configuration file by the `path_variables` definition.
        * **fuel** : A list of float values representing the cumulative fuel along the route. This entry was originally defined as a output in the configuration file by the `path_variables` definition.
        * **times** : A list of strings representing UTC Datetimes of the route points, given that the route started from `start_time` given in the configuration file.


