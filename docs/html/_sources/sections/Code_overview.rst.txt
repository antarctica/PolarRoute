**********
Background
**********

Code Overview
#############

The PolarRoute package provides an automated route-planning method for use by an ice-strengthened vessel operating in the polar regions. The long-distance route planning approach builds on the method developed for underwater vehicles reported in Fox et al (2021). We start with the same grid-based route construction approach to obtain routes that satisfy constraints on the performance of the ship in ice. We then apply a novel route-smoothing method to the resulting routes, shortening the grid-based routes and ensure that routes follow great circle arcs where possible. This two-stage process efficiently generates routes that follow standard navigation solutions in open water and optimise vessel performance in and around areas dominated by sea ice.  While we have focussed on navigation in and around polar ice, our methods are also applicable to shipping in a broader context (e.g.: commercial shipping) where route-planning must be responsive to changing local and weather conditions.


Code Structure
##############
The aim of this manual is to provide the user with all the tools that they need to run the software for a set of examples. We also hope that the background information supplied for each section allows the user to understand the methods used throughout this software package.

The PolarRoute package can be separated into the four main sections shown in the Figure below:

.. figure:: ./Figures/FlowDiagram_Overview.png
   :align: center
   :width: 700

   *Overview figure outlining the stages in planning a route with PolarRoute. Generation of the initial environmental mesh is performed by the MeshiPhi package.*

The separate stages can be broken down into:

1. :ref:`Vessel Performance <vessel-performance>` - Application of vehicle specific features applied to the discrete mesh. In this section we will supply the user with the knowledge of how vehicle specific features are applied to the discrete mesh or with variables applied to the computational graph of the mesh.
2. :ref:`Route Planner` - Generating grid-based dijkstra paths and data constrained path smoothing from the gridded solutions - In this section we will give the user the background to constructing paths between user defined waypoints that minimise a specific objective function (e.g. travel time, fuel). Once the gridded Dijkstra paths are formulated we outline a smoothing based procedure that uses the data information to generate non-gridded improved route paths.

.. figure:: ./Figures/PolarRoute_CodeFlowDiagram.png
   :align: center
   :width: 700

   *Overview figure outlining the Input/Output of all sections of the Route Planning pipeline*

Each stage of this pipeline makes use of a configuration file, found in the :ref:`Configuration Overview` section of the documentation
and produces an output file, the form of which can be found in the :ref:`outputs` section of this document.

In addition to the main section of the codebase we have also developed a series of plotting classes that allows the user to generate interactive maps and static figures for the Mesh Features and Route Paths. These can be found in the `Plotting` section later in the manual.