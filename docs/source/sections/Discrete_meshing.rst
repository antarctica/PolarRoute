********************************
Methods - Mesh Construction
********************************

Throughout this section we will outline an overview of the Environment Mesh Construction module, describe the main classes that composes the module and illustrate a use case for the Discrete Meshing of the environment.

Mesh Construction Overview
##############################
A general overview of the method can be seen below:

.. figure:: ./Figures/FlowDiagram_MeshGraph.png
    :align: center
    :width: 700

    Overview figure of the Discrete Meshing from the multi-data input.


Mesh Construction Design
##############################
The below UML diagram describes how the Environment Mesh Construction module is designed. It depicts the classes of the module and how they interact with each other.

.. figure:: ./Figures/mesh-construct-UML.drawio.png
   :align: center
   :width: 1000
 



Mesh Construction Use case
###################################
The below sequence diagram illustrates a use case for the Discrete Meshing of the environment


Classes
##############
This section describes the main classes of the Mesh Construction module in details

CellBox
***************
.. automodule:: polar_route.cellbox

.. autoclass:: polar_route.cellbox.CellBox
   :special-members: __init__
   :members:  set_data_source, should_split, split, set_parent, aggregate

MetaData
***********

.. automodule:: polar_route.Metadata

.. autoclass:: polar_route.Metadata.Metadata
   :special-members: __init__
  

MeshBuilder
************

.. automodule:: polar_route.MeshBuilder

.. autoclass:: polar_route.MeshBuilder.MeshBuilder
   :special-members: __init__  
   :members: build_environmental_mesh , split_and_replace, split_to_depth

AggregatedCellBox
******************
.. automodule:: polar_route.AggregatedCellBox

.. autoclass:: polar_route.AggregatedCellBox.AggregatedCellBox
   :special-members: __init__
   :members:  contains_point, to_json

EnvironmentMesh
****************
.. automodule:: polar_route.EnvironmentMesh

.. autoclass:: polar_route.EnvironmentMesh.EnvironmentMesh
   :special-members: __init__  
   :members: load_from_json, update_cellbox , to_json, save


NeighbourGraph
***************

.. automodule:: polar_route.NeighbourGraph

.. autoclass:: polar_route.NeighbourGraph.NeighbourGraph
  
   :members: initialise_neighbour_graph, update_neighbours