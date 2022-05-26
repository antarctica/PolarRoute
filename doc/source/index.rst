Documentation for the Code
**************************
.. toctree::
   :maxdepth: 2
   :caption: Contents:

Background
====================
Overview
-----------------
We present an automated route-planning method for use by an ice-strengthened vessel operating as both a supply ship and a research vessel in polar regions. We build on the method developed for underwater vehicle long-distance route planning reported in (Fox *et al* 2021). We start with the same grid-based path construction approach to obtain routes that satisfy constraints on the performance of the ship in ice. We then apply a novel path-smoothing method to the resulting routes, shortening the grid-based paths and ensuring that routes follow great circle arcs where possible. This two-stage process efficiently generates routes that follow standard navigation solutions in open water and optimise vessel performance in and around areas dominated by sea ice.  While we have focussed on navigation in and around polar ice, our methods are also applicable to shipping in a broader context (e.g.: commercial shipping) where route-planning must be responsive to changing local and weather conditions.


..Gauge the improtance of this software package fin the wider role of the community


Outlined throught this manual we will supply the user with everything required to run this navigation software on your local computer, generating routes between user defined waypoints ...


Code Structure & Manual Outline
-----------------
The structure of the code

..This should be the big overview section of posing the paper to a wider audience


.. =====================================================================================================================
.. =====================================================================================================================
.. =====================================================================================================================

CellBox
=====================
Module Information
-----------------
.. automodule:: RoutePlanner.CellBox


Class Information
------------------
.. autoclass:: RoutePlanner.CellBox.CellBox
   :members:



.. =====================================================================================================================
.. =====================================================================================================================
.. =====================================================================================================================


Speed Functions
=====================
Module Information
-----------------
.. automodule:: RoutePlanner.speed



Class Information
-----------------
.. autoclass:: RoutePlanner.speed
..   :members:



.. =====================================================================================================================
.. =====================================================================================================================
.. =====================================================================================================================


Route Optimisation
=====================
In this section we will outline the construction of the route paths using the Mesh construction corrected to include the objective functions define and generated in the earlier section ...

This section relies on two classes `` 






Module Information
-----------------
.. automodule:: RoutePlanner.speed



Class Information
-----------------
.. autoclass:: RoutePlanner.speed
..   :members:
