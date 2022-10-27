"""
    The python package `crossing` implement the optimisation for the crossing point for the unsmoothed and smoothed path
     construction. The package is separated into two classes `NewtonianDistance` and `NewtonianCurve`.
     In the section below we will go through, stage by stage, how the crossing point is determined and the methods
     used within the classes.

"""
import copy
import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


class NewtonianDistance:
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
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

        # Inside the code the base units are m/s. Changing the units of the inputs to match
        self.unit_shipspeed  = unit_shipspeed
        self.unit_time       = unit_time
        self.source_speed    = self._unit_speed(self.source_graph['Speed'])
        self.neighbour_speed = self._unit_speed(self.neighbour_graph['Speed'])
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
            FILL
        '''
        y0 = (Y*x)/(x+a)
        if self.debugging:
            print('---Initial y={:.2f}'.format(y0))
        improving = True
        iterartion_num = 0
        while improving:
            F,dF,X1,X2,t1,t2  = f(y0,x,a,Y,u1,v1,u2,v2,s1,s2)
            if self.debugging:
                print('---Iteration {}: y={:.2f}; F={:.5f}; dF={:.2f}'\
                      .format(iterartion_num,y0,F,dF))
            y0  = y0 - (F/dF)
            improving = abs((F/dF)/(X1*X2)) > self.optimizer_tol
            iterartion_num+=1
            if iterartion_num>1000:
                raise Exception('Newton not able to converge')
        return y0,self._unit_time(np.array([t1,t2]))

    def _unit_speed(self,val):
        '''
            FILL
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
            FILL
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

    def waypoint_correction(self,Wp,Cp):
        '''
            FILL
        '''
        x = (Cp[0]-Wp[0])*self.m_long*np.cos(Wp[1]*(np.pi/180))
        y = (Cp[1]-Wp[1])*self.m_lat
        Su  = self.source_graph['Vector_x']*self.zero_current_factor
        Sv  = self.source_graph['Vector_y']*self.zero_current_factor
        Ssp = self.source_speed
        traveltime = self._traveltime_in_cell(x,y,Su,Sv,Ssp)
        return self._unit_time(traveltime)

    def _F(self,y,x,a,Y,u1,v1,u2,v2,s1,s2):
        '''
            FILL
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
        '''
        dist  = np.sqrt(xdist**2 + ydist**2)
        cval  = np.sqrt(U**2 + V**2)

        dotprod  = xdist*U + ydist*V
        diffsqrs = S**2 - cval**2

        # if (dotprod**2 + diffsqrs*(dist**2) < 0)
        if diffsqrs == 0.0:
            if dotprod == 0.0:
                return np.inf
                #raise Exception(' ')
            else:
                if ((dist**2)/(2*dotprod))  <0:
                    return np.inf
                    #raise Exception(' ')
                else:
                    traveltime = dist * dist / (2 * dotprod)
                    return traveltime

        traveltime = (np.sqrt(dotprod**2 + (dist**2)*diffsqrs) - dotprod)/diffsqrs
        if traveltime < 0:
            traveltime = np.inf
            #raise Exception('Newton Corner Cases returning Zero Traveltime - ISSUE')
        return traveltime

    def _longitude(self):
        '''
            FILL
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
        CrossPoints = (s_cx+ptvl*s_dcx,\
                       s_cy+ptvl*y/self.m_lat)
        CellPoints  = [n_cx,n_cy]

        return TravelTime,CrossPoints,CellPoints

    def _latitude(self):
        '''
            FILL
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
        clon = s_cx  + ptvl*y/(self.m_long*np.cos((n_cy+\
               s_cy)*(np.pi/180)/2.0))
        clat = s_cy + -1*ptvl*s_dcy

        CrossPoints = (clon,clat)
        CellPoints  = [n_cx,n_cy]

        return TravelTime,CrossPoints,CellPoints


    def _corner(self):
        '''
            FILL
        '''


        s_cx  = self.source_graph['cx']
        s_cy  = self.source_graph['cy']
        s_dcx = self.source_graph['dcx']
        s_dcy = self.source_graph['dcy']
        n_cx  = self.neighbour_graph['cx']
        n_cy  = self.neighbour_graph['cy']
        n_dcx = self.neighbour_graph['dcx']
        n_dcy = self.neighbour_graph['dcy']



        # Given the determine the postive and negative position relative to centre
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

        dx1 = s_dcx*self.m_long*np.cos(s_cy*(np.pi/180))
        dx2 = n_dcx*self.m_long*np.cos(n_cy*(np.pi/180))
        dy1 = s_dcy*self.m_lat
        dy2 = n_dcy*self.m_lat

        # Currents in Cells
        Su = ptvX*self.source_graph['Vector_x']*self.zero_current_factor
        Sv = ptvY*self.source_graph['Vector_y']*self.zero_current_factor
        Nu = ptvX*self.neighbour_graph['Vector_x']*self.zero_current_factor
        Nv = ptvY*self.neighbour_graph['Vector_y']*self.zero_current_factor

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

        return TravelTime,CrossPoints,CellPoints


    def value(self):
        '''
            FILLE
        '''
        if self.debugging:
            print('============================================')
        if abs(self.case)==2:
            TravelTime,CrossPoints,CellPoints = self._longitude()
        elif abs(self.case)==4:
            TravelTime,CrossPoints,CellPoints = self._latitude()
        elif abs(self.case)==1 or abs(self.case)==3:
            TravelTime,CrossPoints,CellPoints = self._corner()
        else:
            print('---> Issue with cell (Xsc,Ysc)={:.2f};{:.2f}'.\
                format(self.source_graph['cx'],self.source_graph['cy']))
            TravelTime  = [np.inf,np.inf]
            CrossPoints = [np.nan,np.nan]
            CellPoints  = [np.nan,np.nan]

        return TravelTime, CrossPoints, CellPoints


# ===================================================================================================
# ===================================================================================================
# ===================================================================================================
# ===================================================================================================
# ===================================================================================================
# ===================================================================================================


class NewtonianCurve:
    def __init__(self,neighbour_graph,config,unit_shipspeed='km/hr',unit_time='days',debugging=False,maxiter=1000,pathIter=5,optimizer_tol=1e-3,minimumDiff=1e-3,zerocurrents=True):
        '''
            FILL
        '''
        # Passing the Dijkstra Graph
        self.neighbour_graph = copy.copy(neighbour_graph)

        # Passing the optional Information
        self.config = config

        
        # Inside the code the base units are m/s. Changing the units of the inputs to match
        self.unit_shipspeed = unit_shipspeed
        self.unit_time      = unit_time

        
        # Information for distance metrics
        self.R              = 6371.*1000

        # Optimisation Information
        self.maxiter       = maxiter
        self.pathIter      = pathIter
        self.optimizer_tol = optimizer_tol
        self.minimumDiff   = minimumDiff
        self._epsilon      = 1e-3


        # Optimisation Information
        self.m_long  = 111.321*1000
        self.m_lat   = 111.386*1000.

        # For Debugging purposes 
        self.debugging     = debugging

        if self.debugging:
            self.debugFile1 = open("debugFil.txt", "w")  # append mode
            self.debugFile1.write("Today \n")

        self.id = 0

        # zeroing currents if flagged
        if zerocurrents:
            self.zc = 0.0
        else:
            self.zc = 1.0

    def _unit_speed(self,Val):
        '''
            FILL
        '''
        if self.unit_shipspeed == 'km/hr':
            Val = Val*(1000/(60*60))
        if self.unit_shipspeed == 'knots':
            Val = (Val*0.51)
        return Val

    def _unit_time(self,Val):
        '''
            FILL
        '''
        if self.unit_time == 'days':
            Val = Val/(60*60*24)
        elif self.unit_time == 'hr':
            Val = Val/(60*60)
        elif self.unit_time == 'min':
            Val = Val/(60)
        elif self.unit_time == 's':
            Val = Val
        return Val


    def _calXDist(self,start_long,end_long):
        '''
            FILL
        '''
        return (end_long - start_long)*self.m_long#*np.cos(centralLat)
    def _calYDist(self,start_lat,end_lat):
        '''
            FILL
        '''
        return (end_lat-start_lat)*self.m_lat

    def _long_case(self):
        '''
            FILL
        '''
        def NewtonOptimisationLong(f,y0,x,a,Y,u1,v1,u2,v2,speed_s,speed_e,R,λ_s,φ_r):
                tryNum=1
                iter=0
                improving=True
                while improving:  
                    F,dF,X1,X2  = f(y0,x,a,Y,u1,v1,u2,v2,speed_s,speed_e,R,λ_s,φ_r)
                    if (F==0) or (dF==0):
                        dY = 0
                    else:
                        dY = (F/dF)
                    improving =  (abs(dY)>self._epsilon) or (abs(dY) > self._epsilon*(X1*X2) and (abs(dY)/iter) > self._epsilon)
                    y0  -= dY
                    iter+=1

                    if (iter>100 and tryNum == 1):
                        y0 = Y*x/(x+a)
                        tryNum+=1
                    if (iter > 200) and tryNum>= 2 and tryNum < 10:
                        tryNum+=1
                        iter-=100
                        if(Y < 0):
                            if v2>v1:
                                y0 = (tryNum-2)*Y
                            else:
                                y0 = (tryNum-3)*-Y
                        else:
                            if (v2<v1):
                                y0 = (tryNum-2)*Y
                            else:
                                y0 = (tryNum-3)*-Y
                    if iter > 1000:
                        raise Exception('Newton Curve Issue')
                return y0

        def _F(y,x,a,Y,u1,v1,u2,v2,speed_s,speed_e,R,λ_s,φ_r):
            θ  = (y/R + λ_s*(np.pi/180))
            zl = x*np.cos(θ)
            ψ  = (-(Y-y)/R + φ_r*(np.pi/180))
            zr = a*np.cos(ψ)

            C1  = speed_s**2 - u1**2 - v1**2
            C2  = speed_e**2 - u2**2 - v2**2
            D1  = zl*u1 + y*v1
            D2  = zr*u2 + (Y-y)*v2
            X1  = np.sqrt(D1**2 + C1*(zl**2 + y**2))
            X2  = np.sqrt(D2**2 + C2*(zr**2 + (Y-y)**2))

            dzr = -zr*np.sin(ψ)/R
            dzl = -zl*np.sin(θ)/R

            dD1 = dzl*u1 + v1
            dD2 = dzr*u2 - v2
            dX1 = (D1*v1 + C1*y + dzl*(D1*u1 + C1*zl))/X1
            dX2 = (-v2*D2 - C2*(Y-y) + dzr*(D2*u2 + C2*zr))/X2     

            zr_term = (zr - (X2 - D2)*u2/C2)
            zl_term = (zl - (X1 - D1)*u1/C1)

            F = (X1+X2)*y - v1*(X1-D1)*X2/C1 - (Y - v2*(X2-D2)/C2)*X1 + dzr*zr_term*X1 + dzl*zl_term*X2

            dF = (X1+X2) + y*(dX1 + dX2) - (v1/C1)*(dX2*(X1-D1) + X2*(dX1-dD1))\
                - Y*dX1 + (v2/C2)*(dX1*(X2-D2) + X1*(dX2-dD2))\
                - (zr/(R**2))*zr_term*X1\
                - (zl/(R**2))*zl_term*X2\
                + dzr*(dzr-u2*(dX2-dD2))/C2*X1\
                + dzl*(dzl-u1*(dX1-dD1))/C1*X2\
                + dzr*zr_term*dX1 + dzl*zl_term*dX2

            return F,dF,X1,X2

        Sp   = tuple(self.triplet[['cx','cy']].iloc[0])
        Cp   = tuple(self.triplet[['cx','cy']].iloc[1])
        Np   = tuple(self.triplet[['cx','cy']].iloc[2])

        cell_s_u = self.neighbour_graph.loc[self.triplet.iloc[1]['cellStart'].name,'Vector_x']
        cell_s_v = self.neighbour_graph.loc[self.triplet.iloc[1]['cellStart'].name,'Vector_y']
        cell_e_u = self.neighbour_graph.loc[self.triplet.iloc[1]['cellEnd'].name,'Vector_x']
        cell_e_v = self.neighbour_graph.loc[self.triplet.iloc[1]['cellEnd'].name,'Vector_y']

        if self.triplet.iloc[1].case == 2:   
            sgn  = 1
        else:
            sgn  = -1

        λ_s  = Sp[1]
        φ_r  = Np[1]

        x           = sgn*self._calXDist(Sp[0],Cp[0])
        a           = sgn*self._calXDist(Cp[0],Np[0])
        Y           = (Np[1]-Sp[1])*self.m_lat
        y0          = Y/2
        u1          = sgn*self.zc*cell_s_u; v1 = self.zc*cell_s_v
        u2          = sgn*self.zc*cell_e_u; v2 = self.zc*cell_e_v
        y           = NewtonOptimisationLong(_F,y0,x,a,Y,u1,v1,u2,v2,self.speed_s,self.speed_e,self.R,λ_s,φ_r)

        # Updating the crossing points
        self.triplet['cx'].iloc[1] = Cp[0]
        self.triplet['cy'].iloc[1] = Sp[1] + y/self.m_lat


    def _lat_case(self):
        '''
            FILL
        '''
        def NewtonOptimisationLat(f,y0,x,a,Y,u1,v1,u2,v2,speed_s,speed_e,R,λ,θ,ψ):
                tryNum=1
                iter=0
                improving=True
                while improving:  
                    F,dF,X1,X2  = f(y0,x,a,Y,u1,v1,u2,v2,speed_s,speed_e,R,λ,θ,ψ)
                    if (F==0) or (dF==0):
                        dY = 0
                    else:
                        dY = (F/dF)
                    improving =abs(dY) > 1 or (abs(dY) > self._epsilon*(X1*X2) and (abs(dY)/iter) > self._epsilon)
                    y0  -= dY
                    iter+=1

                    if (iter>100 and tryNum == 1):
                        y0 = Y*x/(x+a)
                        tryNum+=1
                    if (iter > 200) and tryNum== 2:
                        tryNum+=1
                        if(Y < 0):
                            if v2>v1:
                                y0 = Y
                            else:
                                y0 = 0
                        else:
                            if (v2<v1):
                                y0 = Y
                            else:
                                y0 = 0
                    if iter > 1000:
                        raise Exception('Newton Curve Issue')
                return y0

        def _F(y,x,a,Y,u1,v1,u2,v2,speed_s,speed_e,R,λ,θ,ψ):
            λ   = λ*(np.pi/180)
            ψ   = ψ*(np.pi/180)
            θ   = θ*(np.pi/180)
            r1  = np.cos(λ)/np.cos(θ)
            r2  = np.cos(ψ)/np.cos(θ)

            d1  = np.sqrt(x**2 + (r1*y)**2)
            d2  = np.sqrt(a**2 + (r2*(Y-y))**2)
            C1  = speed_s**2 - u1**2 - v1**2
            C2  = speed_e**2 - u2**2 - v2**2
            D1  = x*u1 + r1*v1*y
            D2  = a*u2 + r2*v2*(Y-y)
            X1  = np.sqrt(D1**2 + C1*(d1**2))
            X2  = np.sqrt(D2**2 + C2*(d2**2)) 

            dX1 = (r1*(D1*v1 + r1*C1*y))/X1

            # Cassing Warning message - FIX !!
            dX2 = (-r2*(D2*v2 + r2*C2*(Y-y)))/X2

            F = ((r2**2)*X1+(r1**2)*X2)*y - r1*v1*(X1-D1)*X2/C1 - r2*(r2*Y-v2*(X2-D2)/C2)*X1

            dF = ((r2**2)*X1 + (r1**2)*X2) + y*((r2**2)*dX1 + (r1**2)*dX2)\
                - r1*v1*((X1-D1)*dX2 + (dX1-r1*v1)*X2)/C1\
                - (r2**2)*Y*dX1\
                + r2*v2*((X2-D2)*dX1 + (dX2+r2*v2)*X1)/C2

            return F,dF,X1,X2


        Sp = tuple(self.triplet.iloc[0][['cx','cy']])
        Cp = tuple(self.triplet.iloc[1][['cx','cy']])
        Np = tuple(self.triplet.iloc[2][['cx','cy']])

        cell_s_u = self.neighbour_graph.loc[self.triplet.iloc[1]['cellStart'].name,'Vector_x']
        cell_s_v = self.neighbour_graph.loc[self.triplet.iloc[1]['cellStart'].name,'Vector_y']
        cell_e_u = self.neighbour_graph.loc[self.triplet.iloc[1]['cellEnd'].name,'Vector_x']
        cell_e_v = self.neighbour_graph.loc[self.triplet.iloc[1]['cellEnd'].name,'Vector_y']

        if self.triplet.iloc[1].case == 4:   
            sgn   = 1
        else:
            sgn   = -1

        λ=Sp[1]
        θ=Cp[1]   
        ψ=Np[1]  

        x     = sgn*self._calYDist(Sp[1],Cp[1])
        a     = sgn*self._calYDist(Cp[1],Np[1])
        Y     = sgn*(Np[0]-Sp[0])*self.m_long*np.cos(Cp[1]*(np.pi/180))
        Su    = -sgn*self.zc*cell_s_v; Sv = sgn*self.zc*cell_s_u
        Nu    = -sgn*self.zc*cell_e_v; Nv = sgn*self.zc*cell_e_u
        y0    = Y/2

        y     = NewtonOptimisationLat(_F,y0,x,a,Y,Su,Sv,Nu,Nv,self.speed_s,self.speed_e,self.R,λ,θ,ψ)

        self.triplet['cx'].iloc[1] = Sp[0] + sgn*y/(self.m_long*np.cos(Cp[1]*(np.pi/180)))
        self.triplet['cy'].iloc[1] = Cp[1]


    def _corner_case(self):
        '''
            FILL
        '''
        # Separting out the Long/Lat of each of the points
        Xs,Ys = tuple(self.triplet.iloc[0][['cx','cy']])
        Xc,Yc = tuple(self.triplet.iloc[1][['cx','cy']])
        Xe,Ye = tuple(self.triplet.iloc[2][['cx','cy']])

        # === 1. Assess the cells that are shared commonly in the corner case ====
        sourceNeighbourIndices = self.triplet.iloc[1]['cellStart']
        endNeighbourIndices    = self.triplet.iloc[1]['cellEnd']
    
        commonIndices = list(set(sourceNeighbourIndices['neighbourIndex']).intersection(endNeighbourIndices['neighbourIndex']))
        CornerCells   = self.neighbour_graph.loc[commonIndices]
        Y_line = ((Ye-Ys)/(Xe-Xs))*(Xc-Xs) + Ys

        try:
            if Yc >= Y_line:
                newCell = CornerCells.loc[CornerCells['cy'].idxmin()]
                if newCell.cy > Yc:
                    return
            elif Yc < Y_line:
                newCell = CornerCells.loc[CornerCells['cy'].idxmax()]
                if newCell.cy < Yc:
                    return
            # === 3. Return the path crossing points and cell indices

            firstCrossingPoint  = np.array(sourceNeighbourIndices['neighbourCrossingPoints'])[np.where(np.array(sourceNeighbourIndices['neighbourIndex'])==newCell.name)[0][0],:]
            secondCrossingPoint = np.array(newCell['neighbourCrossingPoints'])[np.where(np.array(newCell['neighbourIndex'])==endNeighbourIndices.name)[0][0],:]
        except:
            self.triplet = copy.deepcopy(self.org_triplet)
            return

        # Adding in the new crossing Point
        newP = pd.Series(name=self.triplet.iloc[1].name+1,dtype='object')
        newP['cx']        = secondCrossingPoint[0]
        newP['cy']        = secondCrossingPoint[1]
        newP['cellStart'] = newCell
        newP['cellEnd']   = copy.deepcopy(self.triplet['cellEnd'].iloc[1])
        newP['case']      = newP['cellStart']['case'][np.where(np.array(newP['cellStart']['neighbourIndex'])==newP['cellEnd'].name)[0][0]]


        # Updating the origional crossing point
        self.triplet['cx'].iloc[1]      = firstCrossingPoint[0]
        self.triplet['cy'].iloc[1]      = firstCrossingPoint[1]
        self.triplet['cellEnd'].iloc[1] = newCell 
        self.triplet['case'].iloc[1]    = self.triplet['cellStart'].iloc[1]['case'][np.where(np.array(self.triplet['cellStart'].iloc[1]['neighbourIndex'])==newCell.name)[0][0]]


        # Adding the new crossing point to the triplet
        self.newP = newP

        self.CrossingDF = pd.concat([self.CrossingDF,newP.to_frame().transpose()]).sort_index().reset_index(drop=True) #self.CrossingDF.append(newP,sort=True).sort_index().reset_index(drop=True)
        self.CrossingDF.index = np.arange(int(self.CrossingDF.index.min()),int(self.CrossingDF.index.max()*1e3 + 1e3),int(1e3))


    def _mergePoint(self):
        '''
            FILL
        '''
        def PtDist(Ser1,Ser2):
            #return np.sqrt(self._calXDist(Ser2['cx'],Ser1['cx'])**2 + self._calXDist(Ser2['cy'],Ser1['cy'])**2)
            return np.sqrt((Ser1['cx'] - Ser2['cx'])**2 + (Ser1['cy'] - Ser2['cy'])**2)

        id=0
        while id < len(self.CrossingDF)-3:
            triplet = self.CrossingDF.iloc[id:id+3]
            if PtDist(triplet.iloc[0],triplet.iloc[1]) < 2e-3:
                try:
                    neighbourIndex = np.where(np.array(triplet.iloc[0]['cellStart']['neighbourIndex'])==triplet.iloc[1]['cellEnd'].name)[0][0]
                except:
                    id+=1
                    continue
                case           = triplet['cellStart'].iloc[0]['case'][neighbourIndex]
                crossingPoint  = triplet['cellStart'].iloc[0]['neighbourCrossingPoints'][neighbourIndex]
                triplet['cx'].iloc[0]      = crossingPoint[0]
                triplet['cy'].iloc[0]      = crossingPoint[1]
                triplet['cellEnd'].iloc[0] = copy.deepcopy(triplet.iloc[1]['cellEnd'])
                triplet['case'].iloc[0]    = copy.deepcopy(case)
                self.CrossingDF           = self.CrossingDF.drop(triplet.iloc[1].name)
            if PtDist(triplet.iloc[1],triplet.iloc[2]) < 2e-3:
                try:
                    neighbourIndex = np.where(np.array(triplet.iloc[1]['cellStart']['neighbourIndex'])==triplet.iloc[2]['cellEnd'].name)[0][0]
                except:
                    id+=1
                    continue
                case           = triplet['cellStart'].iloc[1]['case'][neighbourIndex]
                crossingPoint  = triplet['cellStart'].iloc[1]['neighbourCrossingPoints'][neighbourIndex]
                triplet['cx'].iloc[1]      = crossingPoint[0]
                triplet['cy'].iloc[1]      = crossingPoint[1]
                triplet['cellEnd'].iloc[1] = copy.deepcopy(triplet.iloc[2]['cellEnd'])
                triplet['case'].iloc[1]    = copy.deepcopy(case)
                self.CrossingDF           = self.CrossingDF.drop(triplet.iloc[2].name)

            id+=1

        self.CrossingDF = self.CrossingDF.sort_index().reset_index(drop=True)
        self.CrossingDF.index = np.arange(int(self.CrossingDF.index.min()),int(self.CrossingDF.index.max()*1e3 + 1e3),int(1e3))



    def _horseshoe(self):
        '''
            FILL
        '''
        # Defining the case information
        Cp             = tuple(self.triplet[['cx','cy']].iloc[1])
        Sp             = tuple(self.triplet.iloc[0][['cx','cy']])
        Np             = tuple(self.triplet.iloc[2][['cx','cy']])
        
        cellStartGraph = self.triplet.iloc[1]['cellStart']
        cellEndGraph   = self.triplet.iloc[1]['cellEnd']        
        case           = self.triplet['case'].iloc[1]

        # Returning if corner horseshoe case type
        if abs(case)==1 or abs(case)==3 or abs(case)==0: 
            return
        elif abs(case) == 2:
            # Defining the min and max of the start and end cells
            smin = cellStartGraph['cy']-cellStartGraph['dcy'] 
            smax = cellStartGraph['cy']+cellStartGraph['dcy']
            emin = cellEndGraph['cy']-cellEndGraph['dcy']
            emax = cellEndGraph['cy']+cellEndGraph['dcy']

            # Defining the global min and max
            vmin = np.max([smin,emin])
            vmax = np.min([smax,emax])

            # Point crossingpoint on boundary between the two origional cells
            if (Cp[1] > vmin) and (Cp[1] < vmax):
                return

            # If Start and end cells share a edge for the horesshoe 
            if (Cp[1]<=smin) and (smin==emin):
                hrshCaseStart = 4
                hrshCaseEnd   = 4
            if (Cp[1]>=smax) and (smax==emax):
                hrshCaseStart = -4
                hrshCaseEnd   = -4

            # --- Cases where StartCell is Larger than end Cell ---
            if (Cp[1]>=emax) and (smax>emax):
                hrshCaseStart = case
                hrshCaseEnd   = (-4)                
            if (Cp[1]<=emin) and (smin<emin):
                hrshCaseStart = case
                hrshCaseEnd   = (4)                   

            # --- Cases where StartCell is smaller than end Cell ---
            if (Cp[1]>=smax) and (smax<emax):
                hrshCaseStart = -4
                hrshCaseEnd   = -case
            if (Cp[1]<=smin) and (emin<smin):
                hrshCaseStart = 4
                hrshCaseEnd   = -case                    

        elif abs(case) == 4:
            # Defining the min and max of the start and end cells
            smin = cellStartGraph['cx']-cellStartGraph['dcx']
            smax = cellStartGraph['cx']+cellStartGraph['dcx']
            emin = cellEndGraph['cx']-cellEndGraph['dcx']
            emax = cellEndGraph['cx']+cellEndGraph['dcx']

            # Defining the global min and max
            vmin = np.max([smin,emin])
            vmax = np.min([smax,emax])

            # Point crossingpoint on boundary between the two origional cells
            if (Cp[0] >= vmin) and (Cp[0] <= vmax):
                return

            # If Start and end cells share a edge for the horesshoe 
            if (Cp[0]<smin) and (smin==emin):
                hrshCaseStart = -2
                hrshCaseEnd   = -2
            if (Cp[0]>smax) and (smax==emax):
                hrshCaseStart = 2
                hrshCaseEnd   = 2

            # --- Cases where StartCell is Larger than end Cell ---
            if (Cp[0]>emax) and (smax>emax):
                hrshCaseStart = case
                hrshCaseEnd   = (2)                
            if (Cp[0]<emin) and (smin<emin):
                hrshCaseStart = case
                hrshCaseEnd   = (-2)                   

            # --- Cases where StartCell is smaller than end Cell ---
            if (Cp[0]>smax) and (smax<emax):
                hrshCaseStart = (2)
                hrshCaseEnd   = -case
            if (Cp[0]<smin) and (emin<smin):
                hrshCaseStart = (-2)
                hrshCaseEnd   = -case   


        # Determining the neighbours of the start and end cells that are the horseshoe case
        startGraphNeighbours = [cellStartGraph['neighbourIndex'][ii] for ii in list(np.where(np.array(cellStartGraph['case'])==hrshCaseStart)[0])]
        endGraphNeighbours   = [cellEndGraph['neighbourIndex'][ii] for ii in list(np.where(np.array(cellEndGraph['case'])==hrshCaseEnd)[0])]

        # If there is no neighbour trim back to the edge of the cell
        if (len(startGraphNeighbours)==0) or (len(endGraphNeighbours)==0):
            if abs(case) == 2:
                self.triplet['cy'].iloc[1] = np.clip(self.triplet.iloc[1]['cy'],vmin,vmax)
            if abs(case) == 4:
                self.triplet['cx'].iloc[1] = np.clip(self.triplet.iloc[1]['cx'],vmin,vmax)        
            return

        
        if abs(hrshCaseStart) == abs(hrshCaseEnd):
            for sGN in startGraphNeighbours:
                for eGN in endGraphNeighbours:
                    if (np.array(self.neighbour_graph.loc[sGN,'neighbourIndex'])==eGN).any() and (np.array(self.neighbour_graph.loc[eGN,'neighbourIndex'])==sGN).any():
                        sGNGraph = self.neighbour_graph.loc[sGN]
                        eGNGraph = self.neighbour_graph.loc[eGN]


                        try:

                            Crp1 = np.array(cellStartGraph['neighbourCrossingPoints'])[np.where(np.array(cellStartGraph['neighbourIndex']) == sGN)[0][0],:]
                            Crp2 = np.array(sGNGraph['neighbourCrossingPoints'])[np.where(np.array(sGNGraph['neighbourIndex']) == eGN)[0][0],:]
                            Crp3 = np.array(eGNGraph['neighbourCrossingPoints'])[np.where(np.array(eGNGraph['neighbourIndex']) == cellEndGraph.name)[0][0],:]
                        
                        except:
                            if abs(case) == 2:
                                self.triplet['cy'].iloc[1] = np.clip(self.triplet.iloc[1]['cy'],vmin,vmax)
                            if abs(case) == 4:
                                self.triplet['cx'].iloc[1] = np.clip(self.triplet.iloc[1]['cx'],vmin,vmax)        
                            return

                        # Trimminng back if the cell is worse off
                        if ('SIC' in sGNGraph) and ('SIC' in cellStartGraph) and (sGNGraph['SIC'] - cellStartGraph['SIC'])>=2:
                            if abs(case) == 2:
                                self.triplet['cy'].iloc[1] = np.clip(self.triplet.iloc[1]['cy'],vmin,vmax)
                            if abs(case) == 4:
                                self.triplet['cx'].iloc[1] = np.clip(self.triplet.iloc[1]['cx'],vmin,vmax)        
                            return


                        # Trimminng back if the cell is worse off fuel
                        if ('fuel' in sGNGraph) and ('fuel' in cellStartGraph) and (sGNGraph['fuel'] - cellStartGraph['fuel'])>=2:
                            if abs(case) == 2:
                                self.triplet['cy'].iloc[1] = np.clip(self.triplet.iloc[1]['cy'],vmin,vmax)
                            if abs(case) == 4:
                                self.triplet['cx'].iloc[1] = np.clip(self.triplet.iloc[1]['cx'],vmin,vmax)        
                            return


                        # Updating the origional crossing point
                        self.triplet['cx'].iloc[1]      = Crp1[0]
                        self.triplet['cy'].iloc[1]      = Crp1[1]
                        self.triplet['cellEnd'].iloc[1] = copy.deepcopy(sGNGraph)
                        self.triplet['case'].iloc[1]    = self.triplet['cellStart'].iloc[1]['case'][np.where(np.array(self.triplet['cellStart'].iloc[1]['neighbourIndex'])==sGNGraph.name)[0][0]]

                        # Crossing Point 2
                        Pcrp2 = pd.Series(name=self.triplet.iloc[1].name+1,dtype='object')
                        Pcrp2['cx']        = Crp2[0]
                        Pcrp2['cy']        = Crp2[1]
                        Pcrp2['cellStart'] = copy.deepcopy(sGNGraph)
                        Pcrp2['cellEnd']   = copy.deepcopy(eGNGraph)
                        Pcrp2['case']      = Pcrp2['cellStart']['case'][np.where(np.array(Pcrp2['cellStart']['neighbourIndex'])==Pcrp2['cellEnd'].name)[0][0]]

                        Pcrp3 = pd.Series(name=self.triplet.iloc[1].name+2,dtype='object')
                        Pcrp3['cx']        = Crp3[0]
                        Pcrp3['cy']        = Crp3[1]
                        Pcrp3['cellStart'] = copy.deepcopy(eGNGraph)
                        Pcrp3['cellEnd']   = copy.deepcopy(cellEndGraph)
                        Pcrp3['case']      = Pcrp3['cellStart']['case'][np.where(np.array(Pcrp3['cellStart']['neighbourIndex'])==Pcrp3['cellEnd'].name)[0][0]]
                        

                        self.CrossingDF = pd.concat([self.CrossingDF,Pcrp2.to_frame().transpose(),Pcrp3.to_frame().transpose()], sort=True).sort_index().reset_index(drop=True) #self.CrossingDF.append([Pcrp2,Pcrp3],sort=True).sort_index().reset_index(drop=True)

                        self.CrossingDF.index = np.arange(int(self.CrossingDF.index.min()),int(self.CrossingDF.index.max()*1e3 + 1e3),int(1e3))

                        break

            if abs(case) == 2:
                self.triplet['cy'].iloc[1] = np.clip(self.triplet.iloc[1]['cy'],vmin,vmax)
            if abs(case) == 4:
                self.triplet['cx'].iloc[1] = np.clip(self.triplet.iloc[1]['cx'],vmin,vmax)        
            return
        else:
            for sGN in startGraphNeighbours:
                for eGN in endGraphNeighbours:
                    if (np.array(sGN==eGN).any()):
                        
                        NeighGraph = self.neighbour_graph.loc[sGN]            
                        
                        
                        try:
                            Crp1 = np.array(cellStartGraph['neighbourCrossingPoints'])[np.where(np.array(cellStartGraph['neighbourIndex']) == sGN)[0][0],:]
                            Crp2 = np.array(NeighGraph['neighbourCrossingPoints'])[np.where(np.array(NeighGraph['neighbourIndex']) == cellEndGraph.name)[0][0],:]

                        except:
                            if abs(case) == 2:
                                self.triplet['cy'].iloc[1] = np.clip(self.triplet.iloc[1]['cy'],vmin,vmax)
                            if abs(case) == 4:
                                self.triplet['cx'].iloc[1] = np.clip(self.triplet.iloc[1]['cx'],vmin,vmax)        
                            return


                        # Trimminng back if the cell is worse off
                        if ('SIC' in NeighGraph) and ('SIC' in cellStartGraph) and (NeighGraph['SIC'] - cellStartGraph['SIC'])>=2:
                            if abs(case) == 2:
                                self.triplet['cy'].iloc[1] = np.clip(self.triplet.iloc[1]['cy'],vmin,vmax)
                            if abs(case) == 4:
                                self.triplet['cx'].iloc[1] = np.clip(self.triplet.iloc[1]['cx'],vmin,vmax)        
                            return

                        # Trimminng back if the cell is worse off fuel
                        if ('fuel' in NeighGraph) and ('fuel' in cellStartGraph) and (NeighGraph['fuel'] - cellStartGraph['fuel'])>=2:
                            if abs(case) == 2:
                                self.triplet['cy'].iloc[1] = np.clip(self.triplet.iloc[1]['cy'],vmin,vmax)
                            if abs(case) == 4:
                                self.triplet['cx'].iloc[1] = np.clip(self.triplet.iloc[1]['cx'],vmin,vmax)        
                            return



                        # Updating the origional crossing point
                        self.triplet['cx'].iloc[1]      = Crp1[0]
                        self.triplet['cy'].iloc[1]      = Crp1[1]
                        self.triplet['cellEnd'].iloc[1] = copy.deepcopy(NeighGraph)
                        self.triplet['case'].iloc[1]    = self.triplet['cellStart'].iloc[1]['case'][np.where(np.array(self.triplet['cellStart'].iloc[1]['neighbourIndex'])==NeighGraph.name)[0][0]]

                        Pcrp2 = pd.Series(name=self.triplet.iloc[1].name+2,dtype='object')
                        Pcrp2['cx']        = Crp2[0]
                        Pcrp2['cy']        = Crp2[1]
                        Pcrp2['cellStart'] = copy.deepcopy(NeighGraph)
                        Pcrp2['cellEnd']   = copy.deepcopy(cellEndGraph)
                        Pcrp2['case']      = Pcrp2['cellStart']['case'][np.where(np.array(Pcrp2['cellStart']['neighbourIndex'])==Pcrp2['cellEnd'].name)[0][0]]
                        
                        self.CrossingDF = pd.concat([self.CrossingDF,Pcrp2.to_frame().transpose()], sort=True).sort_index().reset_index(drop=True)#self.CrossingDF.append([Pcrp2],sort=True).sort_index().reset_index(drop=True)
                        
                        self.CrossingDF.index = np.arange(int(self.CrossingDF.index.min()),int(self.CrossingDF.index.max()*1e3 + 1e3),int(1e3))

                        break

            if abs(case) == 2:
                self.triplet['cy'].iloc[1] = np.clip(self.triplet.iloc[1]['cy'],vmin,vmax)
            if abs(case) == 4:
                self.triplet['cx'].iloc[1] = np.clip(self.triplet.iloc[1]['cx'],vmin,vmax)        
            return
                

    def _reverseCase(self):
        '''
            FILL
        '''
        # Removing Reverse Edge Type 1
        startIndex = np.array([row['cellStart'].name for idx,row in self.CrossingDF.iterrows()][1:-1])
        endIndex   = np.array([row['cellEnd'].name for idx,row in self.CrossingDF.iterrows()][1:-1] )
        boolReverseEdge  = np.logical_and((startIndex[:-1] == endIndex[1:]),(startIndex[1:] == endIndex[:-1]))
        if boolReverseEdge.any():
            indxReverseEdge = np.where(boolReverseEdge)[0]+1
            for id in indxReverseEdge:
                self.CrossingDF = self.CrossingDF.drop([self.CrossingDF.iloc[id].name]).sort_index().reset_index(drop=True)


        # Removing Reverse Edge Type 2
        startIndex = np.array([row['cellStart'].name for idx,row in self.CrossingDF.iterrows()][1:-1])
        endIndex   = np.array([row['cellEnd'].name for idx,row in self.CrossingDF.iterrows()][1:-1] )
        boolReverseEdge  = (endIndex[:-1] == endIndex[1:])
        if boolReverseEdge.any():
            indxReverseEdge = np.where(boolReverseEdge)[0]+2
            for id in indxReverseEdge:
                self.CrossingDF = self.CrossingDF.drop(self.CrossingDF.iloc[id].name).sort_index().reset_index(drop=True)

        # Removing Reverse Edge Type 3
        startIndex = np.array([row['cellStart'].name for idx,row in self.CrossingDF.iterrows()][1:-1])
        endIndex   = np.array([row['cellEnd'].name for idx,row in self.CrossingDF.iterrows()][1:-1] )
        boolReverseEdge  = (startIndex[:-1] == startIndex[1:])
        if boolReverseEdge.any():
            indxReverseEdge = np.where(boolReverseEdge)[0]+1
            for id in indxReverseEdge:
                self.CrossingDF = self.CrossingDF.drop(self.CrossingDF.iloc[id].name).sort_index().reset_index(drop=True)



        self.CrossingDF.index = np.arange(0,int(len(self.CrossingDF)*1e3),int(1e3))


    def _updateCrossingPoint(self):
        '''
            FILL
        '''

        self.org_triplet = copy.deepcopy(self.triplet) 
        self.speed_s = self._unit_speed(self.triplet.iloc[1]['cellStart']['Speed'])
        self.speed_e = self._unit_speed(self.triplet.iloc[1]['cellEnd']['Speed'])

        # ------ Case Deginitions & Dealing
        if self.debugging:
            print('===========================================================')
        if abs(self.triplet.iloc[1].case)==2:
            self._long_case()
        elif abs(self.triplet.iloc[1].case)==4:
            self._lat_case()
        elif (abs(self.triplet.iloc[1].case)==1) or (abs(self.triplet.iloc[1].case)==3):
            self._corner_case()

        if len(self.triplet) < 3:
            return




    def _traveltime_in_cell(self,xdist,ydist,U,V,S):
        '''
            FILL
        '''
        dist  = np.sqrt(xdist**2 + ydist**2)
        cval  = np.sqrt(U**2 + V**2)

        dotprod  = xdist*U + ydist*V
        diffsqrs = S**2 - cval**2

        # if (dotprod**2 + diffsqrs*(dist**2) < 0)
        if diffsqrs == 0.0:
            if dotprod == 0.0:
                return np.inf
                #raise Exception(' ')
            else:
                if ((dist**2)/(2*dotprod))  <0:
                    return np.inf
                    #raise Exception(' ')
                else:
                    traveltime = dist * dist / (2 * dotprod)
                    return traveltime

        traveltime = (np.sqrt(dotprod**2 + (dist**2)*diffsqrs) - dotprod)/diffsqrs
        if traveltime < 0:
            traveltime = np.inf
        return traveltime, dist

    def _waypoint_correction(self,source_graph,Wp,Cp):
        '''
            FILL
        '''
        m_long  = 111.321*1000
        m_lat   = 111.386*1000

        x = (Cp[0]-Wp[0])*m_long*np.cos(Wp[1]*(np.pi/180))
        y = (Cp[1]-Wp[1])*m_lat
        Su  = source_graph['Vector_x']*self.zc
        Sv  = source_graph['Vector_y']*self.zc
        Ssp = self._unit_speed(source_graph['Speed'])
        traveltime, distance = self._traveltime_in_cell(x,y,Su,Sv,Ssp)
        return traveltime, distance

    def objective_function(self):
        '''
            FILL
        '''
        TravelTime = np.zeros(len(self.CrossingDF))
        Distance   = np.zeros(len(self.CrossingDF))
        index      = np.zeros(len(self.CrossingDF))
        for ii in range(len(self.CrossingDF)-1):
            soruce_graph = self.CrossingDF.iloc[ii]['cellEnd']
            Wp = self.CrossingDF.iloc[ii][['cx','cy']].to_numpy()
            Cp = self.CrossingDF.iloc[ii+1][['cx','cy']].to_numpy()
            traveltime, distance = self._waypoint_correction(soruce_graph,Wp,Cp)
            TravelTime[ii+1] = self._unit_time(traveltime)
            Distance[ii+1]   = distance

            if ii ==0:
                index[ii] = soruce_graph.name
                index[ii+1] = soruce_graph.name
            else:
                index[ii+1] = soruce_graph.name
        return TravelTime,Distance,index

