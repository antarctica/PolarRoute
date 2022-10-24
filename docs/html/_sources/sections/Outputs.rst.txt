********
Outputs - Data Types
********

#####
Mesh construction
#####

The first stage in the route planning pipeline is constructing a discrete 
mesh of the environment in which the route planner can operate. Once this
mesh is constructed, it can then be exported as a json object and passed 
down-stream to the vehicle specifics and route planner. An example 
of mesh construction and json object generation are as follows:

::

    from polar_route.mesh import Mesh

    with open('config.json', 'r') as f:
        config = json.load(f)

    mesh = Mesh(config)
    mesh_json = mesh.to_json()

.. note:: 
    Examples and a description of the configuration files can be found in
    the :ref:`Configuration` section of this document.


The json object outputted by the Mesh consists of 3 sections: **config**,
**cellboxes** and **neighbour_graph**.

::

    {
        "config": {
            ...
        },
        "cellboxes": [
            {...},
            ...
            {...}
        ],
        "neighbour_graph": [
            "<id_1>": {
                ...
            },
            ...
            "id_n": {
                ...
            }
        ]
    }

where the parts of the json object can be understood as follows:

* **config** : The configuration file used to generate the Mesh.
* **cellboxes** : A list of json representations of CellBox objects that form the Mesh.
* **neighbour_graph** : A graphical representation of the adjacency of CellBoxes within the Mesh.

=============
cellboxes
=============

Each CellBox object within *cellboxes* in the outputted json object is of
the following form:

::

    {
        "id" (string): ...,
        "geometry" (string): ...,
        "cx" (float): ...,
        "cy" (float): ...,
        "dcx" (float): ...,
        "dcy" (float): ...,
        "<value_1>" (float): ...,
        ...
        "<value_n>" (float): ...
    }

Where the values within the CellBox represent the following:

* **id** : The index of the CellBox within the Mesh.
* **geometry** : The spatial boundaries of the CellBox.
* **cx** : The x-position of the centroid of the CellBox, given in degrees latitude.
* **cy** : The y-position of the centroid of the CellBox, given in degrees longitude.
* **dcx** : The x-distance from the edge of the CellBox to the centroid of the CellBox. Given in degrees longitude.
* **dxy** : the y-distance from the edge of the CellBox to the centroid of the CellBox. Given in degrees latitude.

.. figure:: ./Figures/cellbox_json.png
   :align: center
   :width: 700


==================
neighbour_graph
==================

For each CellBox in the *cellboxes* section of the outputted json object, there will be a
corresponding entry in the *neighbour_graph*.

.. note::
    Once the vehicle accessibility conditions have been applied to the json object, this may no longer be true
    as inaccessible CellBoxes will be removed from *neighbour_graph* but will remain in *cellboxes*

Each entry in the *neighbour_graph* is of the following form:

:: 

    "<id>": {
        "1": [...],
        "2": [...],
        "3": [...],
        "4": [...],
        "-1": [...],
        "-2": [...],
        "-3": [...],
        "-4": [...]
    }

where each of the values represent the following: 

* **<id>** : The id of a CellBox within *cellboxes*
    * **1**  : A list of id's of CellBoxes within *cellboxes* to the North-East of the CellBox specified by 'id'.
    * **2**  : A list of id's of CellBoxes within *cellboxes* to the East of the CellBox specified by 'id'.
    * **3**  : A list of id's of CellBoxes within *cellboxes* to the South-East of the CellBox specified by 'id'.
    * **4**  : A list of id's of CellBoxes within *cellboxes* to the South-West of the CellBox specified by 'id'.
    * **-1** : A list of id's of CellBoxes within *cellboxes* to the South of the CellBox specified by 'id'.
    * **-2** : A list of id's of CellBoxes within *cellboxes* to the South-West of the CellBox specified by 'id'.
    * **-3** : A list of id's of CellBoxes within *cellboxes* to the North-West of the CellBox specified by 'id'.
    * **-4** : A list of id's of CellBoxes within *cellboxes* to the South of the CellBox specified by 'id'.

.. figure:: ./Figures/neighbour_graph_json.png
   :align: center
   :width: 700

#####
Vehicle specifics
#####

Once a discrete mesh environment is contracted, it is then passed to the vessel performance object
which applies transformations which are specific to a given vehicle.

:: 

    from polar_route.mesh import Mesh
    from polar_route.vessel_performance import VesselPerformance

    with open('config.json', 'r') as f:
        config = json.load(f)

    mesh = Mesh(config)
    mesh_json = mesh.to_json()

    vp = VesselPerformance(mesh_json)
    vessel_mesh_json = vp.to_json()

.. note::
    To be compatible with vessel performance transformations, a Mesh must be constructed with
    the following attributes:
    
    * SIC (available via data_loaders: *loader_amsr*, *load_bsose*, *load_modis*)
    * thickness (available via data_loaders: *load_thickness*)
    * density (available via data_loaders: *load_density*)

    see section **Multi Data Input** for more information on data_loaders


TODO - Description of transformation applied to the mesh json object by Vessel Performance.
................................................................................................................
................................................................................................................
................................................................................................................
................................................................................................................
................................................................................................................
................................................................................................................

#####
Route planning
#####

During the route planning stage of the pipline information on the routes and the waypoints used are saved as outputs to the processing stage. Descriptions of the structure of the two outputs are given below:

==================
waypoints
==================

An entry in the json including all the information of the waypoints defined by the user from the `waypoints_path` file. It may be the case that ot all waypoints would have been used in the route construction, but all waypoints are returned to this entry. The structure of the entry follows:

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

This output can be converted to a pandas dataframe by running
::
    waypoints_dataframe = pd.DataFrame(waypoints) 


==================
paths
==================
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
    * **<properties>** : A list of metainformation about the route
        * **from**  : Starting waypoint of route
        * **to**  : Ending waypoint of route
        * **traveltime** : A list of float values representing the cumulative travel time along the route. This entry was originally defined as a output in the configuration file by the `path_variables` definition.
        * **fuel** : A list of float values representing the cumulative fuel along the route. This entry was originally defined as a output in the configuration file by the `path_variables` definition.
        * **times** : A list of strings representing UTC Datetimes of the route points, given that the route started from `startTime` given in the configuration file.


