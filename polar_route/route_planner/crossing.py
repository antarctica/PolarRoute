"""
    The python package `crossing` implements the optimisation of the crossing point for the Dijkstra path
    construction using the `NewtonianDistance` class.
    In the section below we will go through, stage by stage, how the crossing point is determined and the methods
    used within the class.
"""
import numpy as np
import logging
from polar_route.utils import unit_time, unit_speed
np.seterr(divide='ignore', invalid='ignore')


def traveltime_in_cell(xdist, ydist, u, v, s, tt_dist=None):
    """
        Determine the traveltime within a cell

        Args:
            xdist (float): Longitude distance between two points in km
            ydist (float): Latitude distance between two points in km
            u (float): U-Component for the forcing vector
            v (float): V-Component for the forcing vector
            s (float): Speed of the vehicle
            tt_dist (bool): Returns traveltime and distance if true, otherwise just traveltime
        Returns:
            traveltime (float): the travel time within the cell
            dist (float): the distance within the cell
    """
    dist = np.sqrt(xdist**2 + ydist**2)
    cval = np.sqrt(u**2 + v**2)

    dotprod = xdist*u + ydist*v
    diffsqrs = s**2 - cval**2

    if diffsqrs == 0.0:
        if dotprod == 0.0:
            traveltime = np.inf
        else:
            if ((dist**2)/(2*dotprod)) < 0:
                traveltime = np.inf
            else:
                traveltime = dist**2/(2*dotprod)
    else:
        traveltime = (np.sqrt(dotprod**2 + (dist**2)*diffsqrs) - dotprod)/diffsqrs

    if traveltime < 0:
        traveltime = np.inf

    if tt_dist:
        return traveltime, dist
    else:
        return traveltime


class NewtonianDistance:
    def __init__(self, node_id, neighbour_id, cellboxes, case=None, unit_shipspeed='km/hr', time_unit='days',
                 maxiter=1000, optimizer_tol=1e-3):
        """
            Class that uses the Newton method to find the crossing point between cells based on their environmental
            parameters.

            Args:
                node_id (str): the id of the initial cellbox
                neighbour_id (str): the id of the neighbouring cellbox
                cellboxes (dict): a dictionary with all cellboxes in the mesh indexed by their ids
                case (int): the case between the cellboxes
                unit_shipspeed (str): the speed unit to use
                time_unit (str): the time unit to use
                maxiter (int): the maximum number of iterations for the optimisation
                optimizer_tol (float): the tolerance value for the optimisation
        """
        # Cell information
        self.source_cellbox = cellboxes [node_id]
        self.neighbour_cellbox = cellboxes [neighbour_id]

        # Case indices
        direction = [1, 2, 3, 4, -1, -2, -3, -4]

        # Inside the code the base units are m/s. Changing the units of the inputs to match
        self.unit_shipspeed = unit_shipspeed
        self.unit_time = time_unit
        self.source_speed = unit_speed(self.source_cellbox.agg_data['speed'][direction.index(case)],
                                       self.unit_shipspeed)
        self.neighbour_speed = unit_speed(self.neighbour_cellbox.agg_data['speed'][direction.index(case)],
                                          self.unit_shipspeed)

        self.case = case

        # Optimisation Information
        self.maxiter = maxiter
        self.optimizer_tol = optimizer_tol

        # Optimisation Information
        self.m_long = 111.321*1000
        self.m_lat = 111.386*1000

        # Defining a small distance
        self.small_distance = 1e-4

    def _newton_optimisation(self, f, x, a, Y, u1, v1, u2, v2, s1, s2):
        """
            Newton Optimisation applied to the crossing point, to update its crossing position in-line
            with the environmental parameters of the start and end cell box.

            All inputs and outputs are given in SI units (m,m/s,s)

            Inputs:
                f  (func) -
                x  (float) - Start cell perpendicular distance from cell centre to crossing boundary
                a  (float) - End cell perpendicular distance from cell centre to crossing boundary
                Y  (float) - Parallel distance, to crossing boundary, between start cell and end cell centres
                u1 (float) - Start Cell perpendicular to crossing boundary forcing component
                v1 (float) - Start Cell parallel to crossing boundary forcing component
                u2 (float) - End Cell perpendicular to crossing boundary forcing component
                v2 (float) - End Cell parallel to crossing boundary forcing component
                s1 (float) - Start Cell max speed
                s1 (float) - End Cell max speed

            Returns:
                y0 (float) - updated crossing point represent as parallel distance along crossing boundary from
                start cell centre to crossing point
                t  (tuple, (float,float)) - Traveltime of the two segments from start cell centre to crossing
                 to end cell centre.
       """
        y0 = (Y * x)/(x + a)
        improving = True
        iteration_num = 0
        while improving:
            F, dF, X1, X2, t1, t2 = f(y0, x, a, Y, u1, v1, u2, v2, s1, s2)
            if F == np.inf:
                return np.nan, np.inf
            y0  = y0 - (F/dF)
            improving = abs((F/dF)/(X1*X2)) > self.optimizer_tol
            iteration_num += 1
            # Assume convergence impossible after 1000 iterations and exit
            if iteration_num > 1000:
                # Set crossing point to nan and travel times to infinity
                y0 = np.nan
                t1 = -1
                t2 = -1

        return y0, unit_time(np.array([t1, t2]), self.unit_time)

    def _F(self, y, x, a, Y, u1, v1, u2, v2, s1, s2):
        """
            Determining Newton Function and differential of newton function from adjacent cell pair information

            Args:
                y  (float) - Crossing point as a parallel distance along crossing boundary from start cell centre
                to the crossing point
                x  (float) - Start cell perpendicular distance from cell centre to crossing boundary
                a  (float) - End cell perpendicular distance from cell centre to crossing boundary
                Y  (float) - Parallel distance, to crossing boundary, between start cell and end cell centres
                u1 (float) - Start Cell perpendicular to crossing boundary forcing component
                v1 (float) - Start Cell parallel to crossing boundary forcing component
                u2 (float) - End Cell perpendicular to crossing boundary forcing component
                v2 (float) - End Cell parallel to crossing boundary forcing component
                s1 (float) - Start Cell max speed
                s1 (float) - End Cell max speed

            Outputs:
                F (float)  - Newton function value
                dF (float) - Differential of Newton function value
                X1 (float) - Characteristic X distance in start cell
                X2 (float) - Characteristic X distance in end cell
                t1 (float) - Traveltime leg from start cell centre to crossing point
                t1 (float) - Traveltime leg from crossing point to end cell centre
        """
        C1 = s1**2 - u1**2 - v1**2
        C2 = s2**2 - u2**2 - v2**2
        D1 = x*u1 + y*v1
        D2 = a*u2 + (Y-y)*v2
        X1ns = D1**2 + C1*(x**2 + y**2)
        if X1ns < 0:
            X1 = -1
        else:
            X1 = np.sqrt(X1ns)
        X2ns = D2**2 + C2*(a**2 + (Y-y)**2)
        if X2ns < 0:
            X2 = -1
        else:
            X2 = np.sqrt(X2ns)

        if X1 < 0 or X2 < 0:
            return np.inf, np.inf, np.inf, np.inf, np.inf, np.inf

        F  = X2*(y-((v1*(X1-D1))/C1)) + X1*(y-Y+((v2*(X2-D2))/C2))

        dD1 = v1
        dD2 = -v2
        if X1 == 0:
            dX1 = 0
        else:
            dX1 = (D1*v1 + C1*y)/X1
        if X2 ==0:
            dX2 = 0
        else:
            dX2 = (-D2*v2 - C2*(Y-y))/X2
        dF  = (X1+X2) + y*(dX1 + dX2) - (v1/C1)*(dX2*(X1-D1) + X2*(dX1-dD1))\
            + (v2/C2)*(dX1*(X2-D2)+X1*(dX2-dD2)) - Y*dX1

        t1 = (X1-D1)/C1
        t2 = (X2-D2)/C2

        return F, dF, X1, X2, t1, t2

    def waypoint_correction(self, Wp, Cp):
        """
            Determines the traveltime between two points within a given cell
            Args:
                Wp (tuple): Start Waypoint location (long,lat)
                Cp (tuple): End Waypoint location (long,lat)
            Returns:
                traveltime (float) - Traveltime between the two points within cell in unit_time
        """
        x = (Cp[0] - Wp[0]) * self.m_long * np.cos(Wp[1] * (np.pi/180))
        y = (Cp[1] - Wp[1]) * self.m_lat
        Su  = self.source_cellbox.agg_data['uC'] 
        Sv  = self.source_cellbox.agg_data['vC'] 
        Ssp = self.source_speed
        traveltime = traveltime_in_cell(x, y, Su, Sv, Ssp)
        return unit_time(traveltime, self.unit_time)

    def _longitude(self):
        """
            Applying an inplace longitude correction to the crossing point between the start and end cell boxes

            All outputs are given in SI units (m,m/s,s)

            Outputs:
                travel_time (tuple,[float,float])  - Traveltime legs from start cell centre to crossing to end cell centre
                cross_points (tuple,[float,float]) - Crossing Point in (long,lat)
                cell_points (tuple,[int,int])      - Start and End cell indices
        """

        if self.case==2:
            ptvl = 1.0
        else:
            ptvl = -1.0

        s_cx  = self.source_cellbox.get_bounds().getcx()
        s_cy  = self.source_cellbox.get_bounds().getcy()
        s_dcx = self.source_cellbox.get_bounds().getdcx()
        s_dcy = self.source_cellbox.get_bounds().getdcy()
        n_cx  = self.neighbour_cellbox.get_bounds().getcx()
        n_cy  = self.neighbour_cellbox.get_bounds().getcy()
        n_dcx = self.neighbour_cellbox.get_bounds().getdcx()
        n_dcy = self.neighbour_cellbox.get_bounds().getdcy()

        Su = ptvl*self.source_cellbox.agg_data['uC'] 
        Sv = ptvl*self.source_cellbox.agg_data['vC'] 
        Nu = ptvl*self.neighbour_cellbox.agg_data['uC'] 
        Nv = ptvl*self.neighbour_cellbox.agg_data['vC'] 

        Ssp = self.source_speed
        Nsp = self.neighbour_speed

        x = s_dcx*self.m_long*np.cos(s_cy*(np.pi/180))
        a = n_dcx*self.m_long*np.cos(n_cy*(np.pi/180))
        Y = ptvl * (n_cy - s_cy) * self.m_lat

        # Optimising to determine the y-value of the crossing point
        y, travel_time = self._newton_optimisation(self._F, x, a, Y, Su, Sv, Nu, Nv, Ssp, Nsp)
        if np.isnan(y) or travel_time[0] < 0 or travel_time[1] < 0:
            travel_time  = [np.inf, np.inf]
            cross_points = [np.nan, np.nan]
            cell_points  = [np.nan, np.nan]
            return travel_time, cross_points, cell_points

        cross_points = (s_cx+ptvl * s_dcx,
                       s_cy+ptvl * y/self.m_lat)
        cell_points  = [n_cx, n_cy]

        # Checking Crossing Point possible
        # Defining the min and max of the start and end cells
        smin = s_cy - s_dcy
        smax = s_cy + s_dcy
        emin = n_cy - n_dcy
        emax = n_cy + n_dcy
        vmin = np.max([smin, emin])
        vmax = np.min([smax, emax])
        if (cross_points [1] < vmin) or (cross_points[1] > vmax):
            cross_points = (cross_points[0], np.clip(cross_points[1], vmin, vmax))

        return travel_time, cross_points, cell_points

    def _latitude(self):
        """
            Applying an inplace latitude correction to the crossing point between the start and end cell boxes

            All outputs are given in SI units (m,m/s,s)

            Outputs:
                Traveltime (tuple,[float,float])  - Traveltime legs from start cell centre to crossing to end cell centre
                cross_points (tuple,[float,float]) - Crossing Point in (long,lat)
                cell_points (tuple,[int,int])      - Start and End cell indices
        """

        if self.case==4:
            ptvl = 1.0
        else:
            ptvl = -1.0

        s_cx  = self.source_cellbox.get_bounds().getcx()
        s_cy  = self.source_cellbox.get_bounds().getcy()
        s_dcx = self.source_cellbox.get_bounds().getdcx()
        s_dcy = self.source_cellbox.get_bounds().getdcy()
        n_cx  = self.neighbour_cellbox.get_bounds().getcx()
        n_cy  = self.neighbour_cellbox.get_bounds().getcy()
        n_dcx = self.neighbour_cellbox.get_bounds().getdcx()
        n_dcy = self.neighbour_cellbox.get_bounds().getdcy()

        Su = -1 * ptvl * self.source_cellbox.agg_data['vC']
        Sv = ptvl * self.source_cellbox.agg_data['uC']
        Nu = -1 * ptvl * self.neighbour_cellbox.agg_data['vC']
        Nv = ptvl * self.neighbour_cellbox.agg_data['uC']

        Ssp = self.source_speed
        Nsp = self.neighbour_speed

        x = s_dcy * self.m_lat
        a = n_dcy * self.m_lat
        Y = ptvl * (n_cx - s_cx) * self.m_long * np.cos((n_cy + s_cy) * (np.pi/180)/2.0)

        y, travel_time   = self._newton_optimisation(self._F, x, a, Y, Su, Sv, Nu, Nv, Ssp, Nsp)
        if np.isnan(y) or travel_time[0] < 0 or travel_time[1] < 0:
            travel_time  = [np.inf, np.inf]
            cross_points = [np.nan, np.nan]
            cell_points  = [np.nan, np.nan]
            return travel_time, cross_points, cell_points

        clon = s_cx + ptvl * y/(self.m_long * np.cos((n_cy + s_cy) * (np.pi/180)/2.0))
        clat = s_cy + -1 * ptvl * s_dcy

        cross_points = (clon, clat)
        cell_points  = [n_cx, n_cy]

        # Checking Crossing Point possible
        # Defining the min and max of the start and end cells
        smin = s_cx - s_dcx
        smax = s_cx + s_dcx
        emin = n_cx - n_dcx
        emax = n_cx + n_dcx
        vmin = np.max([smin, emin])
        vmax = np.min([smax, emax])
        if (cross_points [0] < vmin) or (cross_points[0] > vmax):
            cross_points = (np.clip(cross_points[0], vmin, vmax), cross_points[1])

        return travel_time, cross_points, cell_points

    def _corner(self):
        """
            Applying an inplace corner crossing point between the start and end cell boxes

            All outputs are given in SI units (m,m/s,s)

            Outputs:
                Traveltime (tuple,[float,float])  - Traveltime legs from start cell centre to crossing to end cell centre
                cross_points (tuple,[float,float]) - Crossing Point in (long,lat)
                cell_points (tuple,[int,int])      - Start and End cell indices
        """

        s_cx  = self.source_cellbox.get_bounds().getcx()
        s_cy  = self.source_cellbox.get_bounds().getcy()
        s_dcx = self.source_cellbox.get_bounds().getdcx()
        s_dcy = self.source_cellbox.get_bounds().getdcy()
        n_cx  = self.neighbour_cellbox.get_bounds().getcx()
        n_cy  = self.neighbour_cellbox.get_bounds().getcy()
        n_dcx = self.neighbour_cellbox.get_bounds().getdcx()
        n_dcy = self.neighbour_cellbox.get_bounds().getdcy()

        # Given the case determine the positive and negative position relative to centre
        if self.case == 1:
            ptvX = 1.0
            ptvY = 1.0
        elif self.case == -1:
            ptvX = -1.0
            ptvY = -1.0
        elif self.case == 3:
            ptvX = 1.0
            ptvY = -1.0
        elif self.case == -3:
            ptvX = -1.0
            ptvY = 1.0

        dx1 = ptvX * s_dcx * self.m_long*np.cos(s_cy * (np.pi/180))
        dx2 = ptvX * n_dcx * self.m_long*np.cos(n_cy * (np.pi/180))
        dy1 = ptvY * s_dcy * self.m_lat
        dy2 = ptvY * n_dcy * self.m_lat

        # Currents in Cells
        Su = self.source_cellbox.agg_data['uC']
        Sv = self.source_cellbox.agg_data['vC']
        Nu = self.neighbour_cellbox.agg_data['uC']
        Nv = self.neighbour_cellbox.agg_data['vC']

        # Vehicles Speeds in Cells
        Ssp = self.source_speed
        Nsp = self.neighbour_speed

        # Determining the crossing point as the corner of the case
        cross_points = [s_cx + ptvX * s_dcx,
                       s_cy + ptvY * s_dcy]
        cell_points  = [n_cx, n_cy]

        # Determining traveltime
        t1 = traveltime_in_cell(dx1, dy1, Su, Sv, Ssp)
        t2 = traveltime_in_cell(dx2, dy2, Nu, Nv, Nsp)
        travel_time  = unit_time(np.array([t1, t2]), self.unit_time)

        if travel_time[0] < 0 or travel_time[1] < 0:
            travel_time  = [np.inf, np.inf]
            cross_points = [np.nan, np.nan]
            cell_points  = [np.nan, np.nan]
            return travel_time, cross_points, cell_points

        return travel_time, cross_points, cell_points

    def value(self):
        """
            Applying a correction to determine the optimal crossing point between the start and end cell boxes

            All outputs are given in SI units (m,m/s,s)

            Outputs:
                Traveltime (tuple,[float,float])  - Traveltime legs from start cell centre to crossing to end cell centre
                cross_points (tuple,[float,float]) - Crossing Point in (long,lat)
                cell_points (tuple,[int,int])      - Start and End cell indices
                Case (int)                        - Adjacent cell-box case between start and end cell box
        """
        if abs(self.case)==2:
            travel_time, cross_points, cell_points = self._longitude()
        elif abs(self.case)==4:
            travel_time, cross_points, cell_points = self._latitude()
        elif abs(self.case)==1 or abs(self.case)==3:
            travel_time, cross_points, cell_points = self._corner()
        else:
            logging.debug('---> Issue with cell (Xsc,Ysc)={:.2f};{:.2f}'.\
                format(self.source_cellbox.get_bounds().getcx(),self.source_cellbox.get_bounds().getcy()))

            travel_time  = [np.inf, np.inf]
            cross_points = [np.nan, np.nan]
            cell_points  = [np.nan, np.nan]

        logging.debug(f"NewtonianDistance.value >> TravelTime >> {travel_time}" )
        return travel_time, cross_points, cell_points, self.case
