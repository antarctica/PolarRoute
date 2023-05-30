#############################
Mesh Construction - Classes
#############################

This section describes the main classes of the Mesh Construction module in detail. 
For an overview of the abstractions behind the Mesh Construction module, see the 
:ref:`Mesh Construction - Overview` section of the documentation.

MeshBuilder
************
The MeshBuilder object is the main class of the Mesh Construction module. It is used to build the 
EnvironmentalMesh object from a collection geospatial data. Features of the created EnvironmentalMesh 
as be set using a configuration file passed to the MeshBuilder object. For more information on the format
of the configuration file, see the :ref:`configuration - mesh construction` section of the documentation.

.. automodule:: polar_route.mesh_generation.mesh_builder

.. autoclass:: polar_route.mesh_generation.mesh_builder.MeshBuilder
   :special-members: __init__  
   :members: build_environmental_mesh , split_and_replace, split_to_depth, add_dataloader

EnvironmentMesh
*****************
The EnvironmentMesh object is a collection of geospatial boundaries containing an aggregated representation 
of the data contained within the boundaries (AggregatedCellBox objects). The EnvironmentMesh object is 
created by the MeshBuilder object, though the object is mutable and can be updated after construction.

.. automodule:: polar_route.mesh_generation.environment_mesh

.. autoclass:: polar_route.mesh_generation.environment_mesh.EnvironmentMesh
   :special-members: __init__  
   :members: load_from_json, update_cellbox , to_json, to_geojson, to_tif ,save

NeighbourGraph
***************
The NeighbourGraph object is used to store the connectivity information between the cells of the EnvironmentMesh.
The NeighbourGraph object is created by the MeshBuilder object and is encoded into the EnvironmentalMesh.

.. automodule:: polar_route.mesh_generation.neighbour_graph

.. autoclass:: polar_route.mesh_generation.neighbour_graph.NeighbourGraph
   :members: initialise_neighbour_graph, get_neighbour_case, update_neighbours

CellBox
***************
The CellBox object is used to store the data contained within a geospatial boundary in the MeshBuilder. 
The CellBox object is created by the MeshBuilder object and transformed into an AggregatedCellBox object 
when the MeshBuilder returns the EnvironmentalMesh object.

.. automodule:: polar_route.mesh_generation.cellbox

.. autoclass:: polar_route.mesh_generation.cellbox.CellBox
   :special-members: __init__
   :members:  set_data_source, should_split, split, set_parent, aggregate

MetaData
***********
The Metadata object is used to store the metadata associated with a CellBox object within the MeshBuilder. This includes 
associated DataLoaders, the depth of the CellBox within the MeshBuilder, and the parent CellBox of the CellBox among others.

.. automodule:: polar_route.mesh_generation.metadata

.. autoclass:: polar_route.mesh_generation.metadata.Metadata
   :special-members: __init__

AggregatedCellBox
******************
An aggregated representation of the data contained within a geospatial boundary. The AggregatedCellBox object is created 
by the CellBox object when the MeshBuilder returns the EnvironmentalMesh. 

.. automodule:: polar_route.mesh_generation.aggregated_cellbox

.. autoclass:: polar_route.mesh_generation.aggregated_cellbox.AggregatedCellBox
   :special-members: __init__
   :members:  contains_point, to_json


  






