"""
    The python package `crossing` implement the optimisation for the crossing point for the unsmoothed and smoothed path
    construction. The package is separated into two classes `NewtonianDistance` and `NewtonianCurve`.
    In the section below we will go through, stage by stage, how the crossing point is determined and the methods
    used within the classes.

"""
import copy
import pandas as pd
import numpy as np
import pyproj
import logging
np.seterr(divide='ignore', invalid='ignore')


class NewtonianDistance:

    def __init__(self,source_graph=None,neighbour_graph=None,
                 case=None,unit_shipspeed='km/hr',unit_time='days',
                 zerocurrents=True,debugging=False,maxiter=1000,optimizer_tol=1e-3):
        """Example of docstring on the __init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1 (str): Description of `param1`.
            param2 (:obj:`int`, optional): Description of `param2`. Multiple
                lines are supported.
            param3 (:obj:`list` of :obj:`str`): Description of `param3`.

        """
        # Cell information
        self.source_graph     = source_graph
        self.neighbour_graph  = neighbour_graph

        # Case indices
        self.indx_type = np.array([1, 2, 3, 4, -1, -2, -3, -4])

        # Inside the code the base units are m/s. Changing the units of the inputs to match
        indx = np.where(self.indx_type==case)[0][0]
        self.unit_shipspeed  = unit_shipspeed
        self.unit_time       = unit_time
        self.source_speed    = self._unit_speed(self.source_graph['speed'][indx])
        self.neighbour_speed = self._unit_speed(self.neighbour_graph['speed'][indx])
        self.case            = case

        if zerocurrents:
            self.zero_current_factor = 0.0
        else:
            self.zero_current_factor = 1.0

        # Optimisation Information
        self.maxiter       = maxiter
        self.optimizer_tol = optimizer_tol

        # Optimisation Information
        self.m_long  = 111.321*1000
        self.m_lat   = 111.386*1000

        # Defining a small distance
        self.small_distance = 1e-4

        # For Debugging purposes
        self.debugging     = debugging

    def _newton_optimisation(self, f, x, a, Y, u1, v1, u2, v2, s1, s2):
        '''
            Newton Optimisation applied to the crossing point, to update its crossing position in-line 
            with environmental parameters of the starting and end cell box.

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
                y0 (float) - updated crossing point represent as parallel distance along crossing boundary from start cell centre to crossing point
                t  (tuple, (float,float)) - Traveltime of the two segments from start cell centre to crossing to end cell centre.
       '''
        y0 = (Y*x)/(x+a)
        improving = True
        iteration_num = 0
        while improving:
            F,dF,X1,X2,t1,t2  = f(y0,x,a,Y,u1,v1,u2,v2,s1,s2)
            if F==np.inf:
                return np.nan,np.inf  
            y0  = y0 - (F/dF)
            improving = abs((F/dF)/(X1*X2)) > self.optimizer_tol
            iteration_num+=1
            # Assume convergence impossible after 1000 iterations and exit
            if iteration_num>1000:
                # Set crossing point to nan and travel times to infinity
                y0 = np.nan
                t1 = -1
                t2 = -1
        return y0,self._unit_time(np.array([t1,t2]))

    def _unit_speed(self,val):
        '''
            Applying unit speed for an input type.
            
            Input:
                Val (float) - Input speed in m/s
            Output:
                Val (float) - Output speed in unit type unit_shipspeed

        '''
        if not isinstance(val,type(None)):
            if self.unit_shipspeed == 'km/hr':
                val = val*(1000/(60*60))
            if self.unit_shipspeed == 'knots':
                val = (val*0.51)
            return val
        else:
            return None

    def _unit_time(self,val):
        '''
            Applying unit time for an input type.
            
            Input:
                Val (float) - Input time in s
            Output:
                Val (float) - Output time in unit type unit_time

        '''
        if self.unit_time == 'days':
            val = val/(60*60*24.)
        elif self.unit_time == 'hr':
            val = val/(60*60.)
        elif self.unit_time == 'min':
            val = val/(60.)
        elif self.unit_time == 's':
            val = val
        return val


    def _F(self,y,x,a,Y,u1,v1,u2,v2,s1,s2):
        '''
            Determining Newton Function and differential of newton function from adjacent cell pair information

            Args:
                y  (float) - Crossing point as a parallel distance along crossing boundary from start cell centre to crossing point
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

        '''
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
            return np.inf,np.inf,np.inf,np.inf,np.inf,np.inf

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

        return F,dF,X1,X2,t1,t2

    def _traveltime_in_cell(self,xdist,ydist,U,V,S):
        '''
            Determines the travel-time within cell between two points

            Args:
                xdist (float) - Longitudinal distance in cell in m
                xdist (float) - Latitudinal distance in cell in m
                U (float) - Longitudinal component of forcing vector (m/s)
                V (float) - Latitudinal component of forcing vector (m/s)
                S (float) - Max speed in cell (m/s)


        '''
        dist  = np.sqrt(xdist**2 + ydist**2)
        cval  = np.sqrt(U**2 + V**2)

        dotprod  = xdist*U + ydist*V
        diffsqrs = S**2 - cval**2

        # if (dotprod**2 + diffsqrs*(dist**2) < 0)
        if diffsqrs == 0.0:
            if dotprod == 0.0:
                return np.inf
            else:
                if ((dist**2)/(2*dotprod))  <0:
                    return np.inf
                else:
                    traveltime = dist * dist / (2 * dotprod)
                    return traveltime

        traveltime = (np.sqrt(dotprod**2 + (dist**2)*diffsqrs) - dotprod)/diffsqrs
        if traveltime < 0:
            traveltime = np.inf
        return traveltime

    def waypoint_correction(self,Wp,Cp):
        '''
            Determines the traveltime between two points within a given cell
            Args:
                Wp (tuple): Start Waypoint location (long,lat)
                Cp (tuple): End Waypoint location (long,lat)
            Returns:
                traveltime (float) - Traveltime between the two points within cell in unit_time
        '''
        x = (Cp[0]-Wp[0])*self.m_long*np.cos(Wp[1]*(np.pi/180))
        y = (Cp[1]-Wp[1])*self.m_lat
        Su  = self.source_graph['Vector_x']*self.zero_current_factor
        Sv  = self.source_graph['Vector_y']*self.zero_current_factor
        Ssp = self.source_speed
        traveltime = self._traveltime_in_cell(x,y,Su,Sv,Ssp)
        return self._unit_time(traveltime)


    def _longitude(self):
        '''
            Applying an inplace longitude correction to the crossing point between the start and end cell boxes

            All outputs are given in SI units (m,m/s,s)

            Outputs:
                Traveltime (tuple,[float,float])  - Traveltime legs from start cell centre to crossing to end cell centre
                CrossPoints (tuple,[float,float]) - Crossing Point in (long,lat)
                CellPoints (tuple,[int,int])      - Start and End cell indices

        '''

        if self.case==2:
            ptvl = 1.0
        else:
            ptvl = -1.0

        s_cx  = self.source_graph['cx']
        s_cy  = self.source_graph['cy']
        s_dcx = self.source_graph['dcx']
        s_dcy = self.source_graph['dcy']
        n_cx  = self.neighbour_graph['cx']
        n_cy  = self.neighbour_graph['cy']
        n_dcx = self.neighbour_graph['dcx']
        n_dcy = self.neighbour_graph['dcy']


        Su = ptvl*self.source_graph['Vector_x']*self.zero_current_factor
        Sv = ptvl*self.source_graph['Vector_y']*self.zero_current_factor
        Nu = ptvl*self.neighbour_graph['Vector_x']*self.zero_current_factor
        Nv = ptvl*self.neighbour_graph['Vector_y']*self.zero_current_factor

        Ssp = self.source_speed
        Nsp = self.neighbour_speed


        x = s_dcx*self.m_long*np.cos(s_cy*(np.pi/180))
        a = n_dcx*self.m_long*np.cos(n_cy*(np.pi/180))
        Y = ptvl*(n_cy-s_cy)*self.m_lat

        # Optimising to determine the y-value of the crossing point
        y,TravelTime = self._newton_optimisation(self._F,x,a,Y,Su,Sv,Nu,Nv,Ssp,Nsp)
        if np.isnan(y) or TravelTime[0] < 0 or TravelTime[1] < 0:
            TravelTime  = [np.inf,np.inf]
            CrossPoints = [np.nan,np.nan]
            CellPoints  = [np.nan,np.nan]
            return TravelTime,CrossPoints,CellPoints

        
        CrossPoints = (s_cx+ptvl*s_dcx,\
                       s_cy+ptvl*y/self.m_lat)
        CellPoints  = [n_cx,n_cy]


        # Checking Crossing Point possible
        # Defining the min and max of the start and end cells
        smin = s_cy-s_dcy 
        smax = s_cy+s_dcy 
        emin = n_cy-n_dcy
        emax = n_cy+n_dcy
        vmin = np.max([smin,emin])
        vmax = np.min([smax,emax])
        if (CrossPoints [1] < vmin) or (CrossPoints[1] > vmax):
            CrossPoints = (CrossPoints[0],np.clip(CrossPoints[1],vmin,vmax))


        return TravelTime,CrossPoints,CellPoints

    def _latitude(self):
        '''
            Applying an inplace latitude correction to the crossing point between the start and end cell boxes

            All outputs are given in SI units (m,m/s,s)

            Outputs:
                Traveltime (tuple,[float,float])  - Traveltime legs from start cell centre to crossing to end cell centre
                CrossPoints (tuple,[float,float]) - Crossing Point in (long,lat)
                CellPoints (tuple,[int,int])      - Start and End cell indices

        '''

        if self.case==4:
            ptvl = 1.0
        else:
            ptvl = -1.0


        s_cx  = self.source_graph['cx']
        s_cy  = self.source_graph['cy']
        s_dcx = self.source_graph['dcx']
        s_dcy = self.source_graph['dcy']
        n_cx  = self.neighbour_graph['cx']
        n_cy  = self.neighbour_graph['cy']
        n_dcx = self.neighbour_graph['dcx']
        n_dcy = self.neighbour_graph['dcy']


        Su = -1*ptvl*self.source_graph['Vector_y']*self.zero_current_factor
        Sv = ptvl*self.source_graph['Vector_x']*self.zero_current_factor
        Nu = -1*ptvl*self.neighbour_graph['Vector_y']*self.zero_current_factor
        Nv = ptvl*self.neighbour_graph['Vector_x']*self.zero_current_factor


        Ssp=self.source_speed
        Nsp=self.neighbour_speed

        x = s_dcy*self.m_lat
        a = n_dcy*self.m_lat
        Y = ptvl*(n_cx-s_cx)*self.m_long*np.cos((n_cy+s_cy)*(np.pi/180)/2.0)
        
        
        y,TravelTime   = self._newton_optimisation(self._F,x,a,Y,Su,Sv,Nu,Nv,Ssp,Nsp)
        if np.isnan(y) or TravelTime[0] < 0 or TravelTime[1] < 0:
            TravelTime  = [np.inf,np.inf]
            CrossPoints = [np.nan,np.nan]
            CellPoints  = [np.nan,np.nan]
            return TravelTime,CrossPoints,CellPoints


        clon = s_cx  + ptvl*y/(self.m_long*np.cos((n_cy+s_cy)*(np.pi/180)/2.0))
        clat = s_cy + -1*ptvl*s_dcy

        CrossPoints = (clon,clat)
        CellPoints  = [n_cx,n_cy]


        # Checking Crossing Point possible
        # Defining the min and max of the start and end cells
        smin = s_cx-s_dcx 
        smax = s_cx+s_dcx 
        emin = n_cx-n_dcx
        emax = n_cx+n_dcx
        vmin = np.max([smin,emin])
        vmax = np.min([smax,emax])
        if (CrossPoints [0] < vmin) or (CrossPoints[0] > vmax):
            CrossPoints = (np.clip(CrossPoints[0],vmin,vmax),CrossPoints[1])

        return TravelTime,CrossPoints,CellPoints


    def _corner(self):
        '''
            Applying an inplace corner crossing point between the start and end cell boxes

            All outputs are given in SI units (m,m/s,s)

            Outputs:
                Traveltime (tuple,[float,float])  - Traveltime legs from start cell centre to crossing to end cell centre
                CrossPoints (tuple,[float,float]) - Crossing Point in (long,lat)
                CellPoints (tuple,[int,int])      - Start and End cell indices

        '''


        s_cx  = self.source_graph['cx']
        s_cy  = self.source_graph['cy']
        s_dcx = self.source_graph['dcx']
        s_dcy = self.source_graph['dcy']
        n_cx  = self.neighbour_graph['cx']
        n_cy  = self.neighbour_graph['cy']
        n_dcx = self.neighbour_graph['dcx']
        n_dcy = self.neighbour_graph['dcy']



        # Given the case determine the positive and negative position relative to centre
        if self.case==1:
            ptvX = 1.0
            ptvY = 1.0
        elif self.case==-1:
            ptvX = -1.0
            ptvY = -1.0
        elif self.case==3:
            ptvX = 1.0
            ptvY = -1.0
        elif self.case==-3:
            ptvX = -1.0
            ptvY = 1.0

        dx1 = ptvX*s_dcx*self.m_long*np.cos(s_cy*(np.pi/180))
        dx2 = ptvX*n_dcx*self.m_long*np.cos(n_cy*(np.pi/180))
        dy1 = ptvY*s_dcy*self.m_lat
        dy2 = ptvY*n_dcy*self.m_lat

        # Currents in Cells
        Su = self.source_graph['Vector_x']*self.zero_current_factor
        Sv = self.source_graph['Vector_y']*self.zero_current_factor
        Nu = self.neighbour_graph['Vector_x']*self.zero_current_factor
        Nv = self.neighbour_graph['Vector_y']*self.zero_current_factor



        # Vehicles Speeds in Cells
        Ssp = self.source_speed; Nsp = self.neighbour_speed

        # Determining the crossing point as the corner of the case
        CrossPoints = [s_cx+ptvX*s_dcx,\
                       s_cy+ptvY*s_dcy]
        CellPoints  = [n_cx,n_cy]

        # Determining traveltime
        t1 = self._traveltime_in_cell(dx1,dy1,Su,Sv,Ssp)
        t2 = self._traveltime_in_cell(dx2,dy2,Nu,Nv,Nsp)
        TravelTime  = self._unit_time(np.array([t1,t2]))

        if TravelTime[0] < 0 or TravelTime[1] < 0:
            TravelTime  = [np.inf,np.inf]
            CrossPoints = [np.nan,np.nan]
            CellPoints  = [np.nan,np.nan]
            return TravelTime,CrossPoints,CellPoints


        return TravelTime,CrossPoints,CellPoints


    def value(self):
        '''
            Applying a correction to determine the optimal crossing point between the start and end cell boxes

            All outputs are given in SI units (m,m/s,s)

            Outputs:
                Traveltime (tuple,[float,float])  - Traveltime legs from start cell centre to crossing to end cell centre
                CrossPoints (tuple,[float,float]) - Crossing Point in (long,lat)
                CellPoints (tuple,[int,int])      - Start and End cell indices
                Case (int)                        - Adjacent cell-box case between start and end cell box

        '''
        if abs(self.case)==2:
            TravelTime,CrossPoints,CellPoints = self._longitude()
        elif abs(self.case)==4:
            TravelTime,CrossPoints,CellPoints = self._latitude()
        elif abs(self.case)==1 or abs(self.case)==3:
            TravelTime,CrossPoints,CellPoints = self._corner()
        else:
            TravelTime  = [np.inf,np.inf]
            CrossPoints = [np.nan,np.nan]
            CellPoints  = [np.nan,np.nan]

        return TravelTime, CrossPoints, CellPoints, self.case