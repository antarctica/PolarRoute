.. _mesh_construction_overview:

********************************
Methods - Mesh Construction
********************************

Throughout this section we will outline an overview of the Environment Mesh Construction 
module, describe the main classes that composes the module and illustrate a use case for 
the Discrete Meshing of the environment.

Mesh Construction - Overview
##############################
A general overview of the method can be seen below:

.. figure:: ../Figures/FlowDiagram_MeshGraph.png
    :align: center
    :width: 700

    Overview figure of the Discrete Meshing from the multi-data input.


Mesh Construction Design
##############################
The below UML diagram describes how the Environment Mesh Construction module is designed. 
It depicts the classes of the module and how they interact with each other.

.. figure:: ../Figures/mesh-construct-UML.drawio.png
   :align: center
   :width: 1000
 
Mesh Construction Use case
###################################
This sequence diagram illustrates a use case for the Discrete Meshing of the environment, 
where the module's client starts by initializing the MeshBuilder with a certain mesh 
configuration (see Input-Configuration section for more details about the configuration format) 
then calls build_environment_mesh method.

.. figure:: ../Figures/mesh-build-sequence-diagram.drawio.png
   :align: center
   :width: 1000

The following diagram depicts the sequence of events that take place inside build_environment_mesh 
method into details

.. figure:: ../Figures/build-env-mesh.drawio.png
   :align: center
   :width: 1000

For a more in-depth explanation of the mesh construction methods, please refer to the :ref:`Mesh Construction - Classes`
section.

.. toctree::
   :maxdepth: 1

   ./Mesh_construction_classes
   ./Mesh_validation
