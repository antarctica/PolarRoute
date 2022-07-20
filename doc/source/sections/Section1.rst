********
Background
********

Overview
##############

We present an automated route-planning method for use by an ice-strengthened vessel operating as both a supply ship and a research vessel in polar regions. We build on the method developed for underwater vehicle long-distance route planning reported in Fox et al (2021). We start with the same grid-based route construction approach to obtain routes that satisfy constraints on the performance of the ship in ice. We then apply a novel route-smoothing method to the resulting routes, shortening the grid-based routes and ensuring that routes follow great circle arcs where possible. This two-stage process efficiently generates routes that follow standard navigation solutions in open water and optimise vessel performance in and around areas dominated by sea ice.  While we have focussed on navigation in and around polar ice, our methods are also applicable to shipping in a broader context (e.g.: commercial shipping) where route-planning must be responsive to changing local and weather conditions.


Code Structure
##############
The outline of this manual is to provide the user with all the tools that they need to run the software for a theory of examples. We also hope that the background information supplied for each section allows the user to understand the methods used throughout this toolkit. 

The outline of the toolkit can be separted into the four main sections demonstrated in the Figure below

.. figure:: ./Figures/FlowDiagram_Overview.png
   :align: center
   :width: 700

   *Overview figure outlining the stages in the RoutePlanner*

The separate stages can be broken down into:

1. **Multi Data Input** - Reading in different datasets of differening types. Throughout this section we will outline the form that the input datasets should take and useful tips for pre-processing your data of interest
2. **Discrete Meshing** - Generating a Digitial Twin of the environmental condtions. In this section we outline the different Python classes that are used to constucted a discretised represention across the user defined datasets, giving a coding background into the dynamic splitting of the mesh to finer resolution in regions of datasets that are spatially varying
3. **Vehicles Specifics** - Application of vehicle specific features applied to the discret mesh. In this section we will supply the user with the knowledge of how vehcile specific features are applied to the discret mesh or with varibles applied to the computational graph of the mesh. 
4. **Route Planning** - Generating grid-based dijkstra paths and data constrained path smoothing from the gridded solutions - In this section we will give the user the background to constructing paths between user defined waypoints that minimise a specific objective function (E.g. Traveltime, Fuel). Once the gridded Dijkstra paths are formulated we outline a smoothing based procedure that uses the data information to generate non-gridded improved route paths.

In addition to the main section of the codebase we have also developed a series of plotting classes that allows the user to generate Interactive maps and static figures for the Mesh Features and Route Paths. These can be found in the `Plotting` section later in the manual. 