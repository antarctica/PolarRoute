********************************
Methods - Mesh Construction
********************************

Mesh Construction Overview
##############################

Throughout this section we will outline the use case for the Discrete Meshing of the environment. The two main classes used are `Mesh` and `CellBox`, with a Mesh being composed of a series of CellBox objects.

A general overview of the method can be seen below:

.. figure:: ./Figures/FlowDiagram_MeshGraph.png
    :align: center
    :width: 700

    Overview figure of the Discrete Meshing from the multi-data input.


CellBox
##############

.. automodule:: polar_route.cellbox

.. autoclass:: polar_route.cellbox.CellBox
   :special-members: __init__
   :members:  get_bounds, should_split, split, contains_point, to_json

MeshBuilder
##############

.. automodule:: polar_route.MeshBuilder

.. autoclass:: polar_route.MeshBuilder.MeshBuilder
   :special-members: __init__  
   :members: build_environmental_mesh , split_and_replace, split_to_depth, to_json
