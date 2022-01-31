import numpy as np
import math
from scipy import optimize

def sign(x):
    s = math.copysign(1, x)
    return s   


class _Euclidean_distance():
    """
    Replicating original route planner Euclidean distance 
    Inputs:
      origin      - tuple of floats e.g. (Long_orig,Lat_orig)
      destination - tuple of floats e.g. (Long_dest,Lat_dest)
      Optional: forward - Boolean True or False
    Output:
      Value - If 'forward' is True then returns Distance between 
              two points in 'km'. If 'False' then return the 
              Lat/Long position of a point.

    """

    def __init__(self,initialLat):
        self.m_per_longitude = 111.320*1000.
        self.m_per_latitude  = (110.574*1000.)/abs(np.cos(initialLat*(np.pi/180)))

    def value(self,origin,dest_dist,forward=True):
        lon1,lat1 = origin
        if forward:
            lon2,lat2 = dest_dist
            val = np.sqrt(((lat2-lat1)*self.m_per_latitude)**2 + ((lon2-lon1)*self.m_per_longitude)**2)
        else:
            dist_x,dist_y = dest_dist        
            val = [lon1+(dist_x/self.m_per_longitude),lat1+(dist_y/self.m_per_latitude)]
        return val


# ================================================================
# ================================================================
# ================================================================
# ================================================================
class NewtonianDistance:
    def __init__(self,Cell_s,Cell_n,s,unit_shipspeed='km/hr',unit_time='days',zerocurrents=False,debugging=False,maxiter=500,optimizer_tol=1e-7):
        # Cell information
        self.Cell_s         = Cell_s
        self.Cell_n         = Cell_n



        # Inside the code the base units are m/s. Changing the units of the inputs to match
        self.unit_shipspeed = unit_shipspeed
        self.unit_time      = unit_time


        self.s              = self._unit_speed(s)

        self.fdist          = _Euclidean_distance(self.Cell_s.lat+self.Cell_s.height)

        if zerocurrents:
            self.zx = 0.0
        else:
            self.zx = 1.0

        # Optimisation Information
        self.maxiter   = maxiter
        self.optimizer_tol = optimizer_tol

        # For Debugging purposes 
        self.debugging = debugging

        # Determining the distance between the cell centres to 
        #be used in defining the case
        self.df_x = (self.Cell_n.long+self.Cell_n.width/2) -  (self.Cell_s.long+self.Cell_s.width/2)
        self.df_y = (self.Cell_n.lat+self.Cell_n.height/2) -  (self.Cell_s.lat+self.Cell_s.height/2)

    def NewtonOptimisation(self,f,df,x,a,Y,u1,v1,u2,v2,s):
            y0 = (Y*x)/(x+a)
            if self.debugging:
                    print('---Initial y={:.2f}'.format(y0))
            if self.maxiter > 0:
                for iter in range(self.maxiter):
                    F  = f(y0,x,a,Y,u1,v1,u2,v2,s)
                    dF = df(y0,x,a,Y,u1,v1,u2,v2,s)
                    if self.debugging:
                        print('---Iteration {}: y={:.2f}; F={:.5f}; dF={:.2f}'.format(iter,y0,F,dF))
                    y0  = y0 - (F/dF)
                    if F < self.optimizer_tol:
                        break
            return y0

    def _unit_speed(self,Val):
        if self.unit_shipspeed == 'km/hr':
            Val = Val*(1000/(60*60))
        if self.unit_shipspeed == 'knots':
            Val = (Val*0.51)
        return Val

    def _unit_time(self,Val):
        if self.unit_time == 'days':
            Val = Val/(60*60*24)
        elif self.unit_time == 'hr':
            Val = Val/(60*60)
        elif self.unit_time == 'min':
            Val = Val/(60)
        elif self.unit_time == 's':
            Val = Val

        return Val
        

    def value(self):
        def _F(y,x,a,Y,u1,v1,u2,v2,s):
            C1 = s**2 - u1**2 - v1**2
            C2 = s**2 - u2**2 - v2**2
            D1 = x*u1 + y*v1
            D2 = a*u2 + (Y-y)*v2
            X1 = np.sqrt(D1**2 + C1*(x**2 + y**2))
            X2 = np.sqrt(D2**2 + C2*(a**2 + (Y-y)**2))
            F  = X2*(y-((v1*(X1-D1))/C1)) + X1*(y-Y+((v2*(X2-D2))/C2)) 
            return F

        def _dF(y,x,a,Y,u1,v1,u2,v2,s):
            C1  = s**2 - u1**2 - v1**2
            C2  = s**2 - u2**2 - v2**2
            D1  = x*u1 + y*v1
            D2  = a*u2 + (Y-y)*v2
            X1  = np.sqrt(D1**2 + C1*(x**2 + y**2))
            X2  = np.sqrt(D2**2 + C2*(a**2 + (Y-y)**2))
            dD1 = v1
            dD2 = -v2
            dX1 = (D1*v1 + C1*y)/X1
            dX2 = (-D2*v2 - C2*(Y-y))/X2
            dF  = (X1+X2) + y*(dX1 + dX2) - (v1/C1)*(dX2*(X1-D1) + X2*(dX1-dD1)) + (v2/C2)*(dX1*(X2-D2)+X1*(dX2-dD2)) - Y*dX1
            return dF

        def _T(y,x,a,Y,u1,v1,u2,v2,s):
            C1 = s**2 - u1**2 - v1**2
            C2 = s**2 - u2**2 - v2**2
            D1 = x*u1 + y*v1
            D2 = a*u2 + (Y-y)*v2
            X1 = np.sqrt(D1**2 + C1*(x**2 + y**2))
            X2 = np.sqrt(D2**2 + C2*(a**2 + (Y-y)**2))
            t1 = (X1-D1)/C1
            t2 = (X2-D2)/C2
            T  = t1+t2 
            return T


        def _positive_longitude(self):
            '''
                INCLUDE
            '''
            u1          = self.Cell_s.getuC()*self.zx; v1 = self.Cell_s.getvC()*self.zx
            u2          = self.Cell_n.getuC()*self.zx; v2 = self.Cell_n.getvC()*self.zx
            x           = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy), (self.Cell_s.cx + self.Cell_s.cx_ub,self.Cell_s.cy))
            a           = self.fdist.value((self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx - self.Cell_n.cx_lb,self.Cell_n.cy))
            Y           = sign(self.Cell_n.cy-self.Cell_s.cy)*self.fdist.value((self.Cell_s.cx+(abs(self.Cell_s.cx_ub)+abs(self.Cell_n.cx_lb)),self.Cell_s.cy),\
                                     (self.Cell_s.cx+(abs(self.Cell_s.cx_ub)+abs(self.Cell_n.cx_lb)),self.Cell_n.cy))
            y = self.NewtonOptimisation(_F,_dF,x,a,Y,u1,v1,u2,v2,self.s)
            if self.debugging:
                print('Positve Long: y={:.2f};x={:.2f};a={:.2f};Y={:.2f};u1={:.5f};v1={:.5f};u2={:.5f};v2={:.5f};s={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s))
            TravelTime  = self._unit_time(_T(y,x,a,Y,u1,v1,u2,v2,self.s))
            CrossPoints = self.fdist.value((self.Cell_s.cx + self.Cell_s.cx_ub,self.Cell_s.cy),(0.0,y),forward=False)
            CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
            if self.debugging:
                print('------ TravelTime={:.2f};CrossPoints=[{:.2f},{:.2f}];CellPoints=[{:.2f},{:.5f}]'.format(TravelTime,CrossPoints[0],CrossPoints[1],CellPoints[0],CellPoints[1]))    
            return TravelTime,CrossPoints,CellPoints

        def _negative_longitude(self):
            '''
                INCLUDE
            '''
            u1          = -self.Cell_s.getuC()*self.zx; v1 = self.Cell_s.getvC()*self.zx
            u2          = -self.Cell_n.getuC()*self.zx; v2 = self.Cell_n.getvC()*self.zx
            x           = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx-self.Cell_s.cx_lb,self.Cell_s.cy))
            a           = self.fdist.value((self.Cell_n.cx,self.Cell_n.cy),(self.Cell_n.cx+self.Cell_n.cx_ub,self.Cell_n.cy))
            Y           = sign(self.Cell_n.cy-self.Cell_s.cy)*\
                               self.fdist.value((self.Cell_s.cx-(abs(self.Cell_s.cx_lb)+abs(self.Cell_n.cx_ub)),self.Cell_s.cy),\
                                          (self.Cell_s.cx-(abs(self.Cell_s.cx_lb)+abs(self.Cell_n.cx_ub)),self.Cell_n.cy))
            y = self.NewtonOptimisation(_F,_dF,x,a,Y,u1,v1,u2,v2,self.s)
            if self.debugging:
                print('Negative Long: y={:.2f};x={:.2f};a={:.2f};Y={:.2f};u1={:.5f};v1={:.5f};u2={:.5f};v2={:.5f};s={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s))
            TravelTime  = self._unit_time(_T(y,x,a,Y,u1,v1,u2,v2,self.s))
            CrossPoints = self.fdist.value((self.Cell_s.cx - self.Cell_s.cx_lb,self.Cell_s.cy),(0.0,y),forward=False)
            CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
            if self.debugging:
                print('------ TravelTime={:.2f};CrossPoints=[{:.2f},{:.2f}];CellPoints=[{:.2f},{:.5f}]'.format(TravelTime,CrossPoints[0],CrossPoints[1],CellPoints[0],CellPoints[1]))    
            return TravelTime,CrossPoints,CellPoints

        def _positive_latitude(self):
            '''
                Case -4
            '''

            u1          = -self.Cell_s.getvC()*self.zx; v1 = self.Cell_s.getuC()*self.zx
            u2          = -self.Cell_n.getvC()*self.zx; v2 = self.Cell_n.getuC()*self.zx
            x           = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx,self.Cell_s.cy + self.Cell_s.cy_ub))
            a           = self.fdist.value((self.Cell_n.cx,self.Cell_n.cy),(self.Cell_n.cx,self.Cell_n.cy - self.Cell_n.cy_lb))
            Y           = sign(self.Cell_n.cx-self.Cell_s.cx)*\
                          self.fdist.value((self.Cell_s.cx, self.Cell_s.cy + (abs(self.Cell_s.cy_ub) + abs(self.Cell_n.cy_lb))),\
                                     (self.Cell_n.cx, self.Cell_s.cy + (abs(self.Cell_s.cy_ub) + abs(self.Cell_n.cy_lb))))
            y = self.NewtonOptimisation(_F,_dF,x,a,Y,u1,v1,u2,v2,self.s)
            if self.debugging:
                print('Postive Lat: y={:.2f};x={:.2f};a={:.2f};Y={:.2f};u1={:.5f};v1={:.5f};u2={:.5f};v2={:.5f};s={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s))
            TravelTime  = self._unit_time(_T(y,x,a,Y,u1,v1,u2,v2,self.s))
            CrossPoints = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy+self.Cell_s.cy_ub),(y,0.0),forward=False)
            CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
            if self.debugging:
                print('------ TravelTime={:.2f};CrossPoints=[{:.2f},{:.2f}];CellPoints=[{:.2f},{:.5f}]'.format(TravelTime,CrossPoints[0],CrossPoints[1],CellPoints[0],CellPoints[1]))    
            return TravelTime,CrossPoints,CellPoints


        def _negative_latitude(self):
            '''
                Case 4
            '''

            u1          = -self.Cell_s.getvC()*self.zx; v1 = -self.Cell_s.getuC()*self.zx
            u2          = -self.Cell_n.getvC()*self.zx; v2 = -self.Cell_n.getuC()*self.zx
            x           = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx,self.Cell_s.cy - self.Cell_s.cy_lb))
            a           = self.fdist.value((self.Cell_n.cx,self.Cell_n.cy),(self.Cell_n.cx,self.Cell_n.cy + self.Cell_n.cy_ub))
            Y           = sign(self.Cell_n.cx-self.Cell_s.cx)*\
                          self.fdist.value((self.Cell_s.cx, self.Cell_s.cy - (abs(self.Cell_s.cy_ub) + abs(self.Cell_n.cy_lb))),\
                                     (self.Cell_n.cx, self.Cell_s.cy - (abs(self.Cell_s.cy_ub) + abs(self.Cell_n.cy_lb))))
            y = self.NewtonOptimisation(_F,_dF,x,a,Y,u1,v1,u2,v2,self.s)
            if self.debugging:
                print('Negative Lat: y={:.2f};x={:.2f};a={:.2f};Y={:.2f};u1={:.5f};v1={:.5f};u2={:.5f};v2={:.5f};s={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s))
            TravelTime  = self._unit_time(_T(y,x,a,Y,u1,v1,u2,v2,self.s))
            CrossPoints = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy-self.Cell_s.cy_lb),(y,0.0),forward=False)
            CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
            if self.debugging:
                print('------ TravelTime={:.2f};CrossPoints=[{:.2f},{:.2f}];CellPoints=[{:.2f},{:.5f}]'.format(TravelTime,CrossPoints[0],CrossPoints[1],CellPoints[0],CellPoints[1]))    
            return TravelTime,CrossPoints,CellPoints

        def _top_right_corner(self):
            u1 = self.Cell_s.getuC()*self.zx; v1 = self.Cell_s.getvC()*self.zx
            u2 = self.Cell_n.getuC()*self.zx; v2 = self.Cell_n.getvC()*self.zx
            x  = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx + self.Cell_s.cx_ub,self.Cell_s.cy))
            y  = self.fdist.value((self.Cell_s.cx+self.Cell_s.cx_ub,self.Cell_s.cy),(self.Cell_s.cx+self.Cell_s.cx_ub,self.Cell_s.cy+self.Cell_s.cy_ub))
            a  = -self.fdist.value((self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx - self.Cell_n.cx_lb,self.Cell_n.cy))
            Y  = self.fdist.value((self.Cell_s.cx+(abs(self.Cell_s.cx_ub) + abs(self.Cell_n.cx_lb)),self.Cell_s.cy),\
                            (self.Cell_s.cx+(abs(self.Cell_s.cx_ub) + abs(self.Cell_n.cx_lb)),self.Cell_n.cy))
            if self.debugging:
                print('Top Right: y={:.2f};x={:.2f};a={:.2f};Y={:.2f};u1={:.5f};v1={:.5f};u2={:.5f};v2={:.5f};s={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s))
            TravelTime  = self._unit_time(_T(y,x,a,Y,u1,v1,u2,v2,self.s))
            CrossPoints = [self.Cell_s.cx+self.Cell_s.cx_ub,self.Cell_s.cy+self.Cell_s.cy_ub]
            CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
            if self.debugging:
                print('------ TravelTime={:.2f};CrossPoints=[{:.2f},{:.2f}];CellPoints=[{:.2f},{:.5f}]'.format(TravelTime,CrossPoints[0],CrossPoints[1],CellPoints[0],CellPoints[1]))    
            return TravelTime,CrossPoints,CellPoints

        def _bottom_right_corner(self):
            u1 = self.Cell_s.getuC()*self.zx; v1 = self.Cell_s.getvC()*self.zx
            u2 = self.Cell_n.getuC()*self.zx; v2 = self.Cell_n.getvC()*self.zx
            x  = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx + self.Cell_s.cx_ub,self.Cell_s.cy))
            y  = -self.fdist.value((self.Cell_s.cx+self.Cell_s.cx_ub,self.Cell_s.cy),(self.Cell_s.cx+self.Cell_s.cx_ub,self.Cell_s.cy-self.Cell_s.cy_lb))
            a  = -self.fdist.value((self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx - self.Cell_n.cx_lb,self.Cell_n.cy))
            Y  = -self.fdist.value((self.Cell_s.cx+(abs(self.Cell_s.cx_ub) + abs(self.Cell_n.cx_lb)),self.Cell_s.cy),\
                            (self.Cell_s.cx+(abs(self.Cell_s.cx_ub) + abs(self.Cell_n.cx_lb)),self.Cell_n.cy))
            if self.debugging:
                print('Bottom Right: y={:.2f};x={:.2f};a={:.2f};Y={:.2f};u1={:.5f};v1={:.5f};u2={:.5f};v2={:.5f};s={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s))
            TravelTime  = self._unit_time(_T(y,x,a,Y,u1,v1,u2,v2,self.s))
            CrossPoints = [self.Cell_s.cx+self.Cell_s.cx_ub,self.Cell_s.cy-self.Cell_s.cy_lb]
            CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
            if self.debugging:
                print('------ TravelTime={:.2f};CrossPoints=[{:.2f},{:.2f}];CellPoints=[{:.2f},{:.5f}]'.format(TravelTime,CrossPoints[0],CrossPoints[1],CellPoints[0],CellPoints[1]))    
            return TravelTime,CrossPoints,CellPoints

        def _bottom_left_corner(self):
            u1 = -self.Cell_s.getuC()*self.zx; v1 = self.Cell_s.getvC()*self.zx
            u2 = -self.Cell_n.getuC()*self.zx; v2 = self.Cell_n.getvC()*self.zx
            x  = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx - self.Cell_s.cx_lb,self.Cell_s.cy))
            y  = -self.fdist.value((self.Cell_s.cx-self.Cell_s.cx_lb,self.Cell_s.cy),(self.Cell_s.cx-self.Cell_s.cx_lb,self.Cell_s.cy-self.Cell_s.cy_lb))
            a  = -self.fdist.value((self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx + self.Cell_n.cx_ub,self.Cell_n.cy))
            Y  = -self.fdist.value((self.Cell_s.cx-(abs(self.Cell_s.cx_lb) + abs(self.Cell_n.cx_ub)),self.Cell_s.cy),\
                             (self.Cell_s.cx-(abs(self.Cell_s.cx_lb) + abs(self.Cell_n.cx_ub)),self.Cell_n.cy))
            if self.debugging:
                print('Bottom Left: y={:.2f};x={:.2f};a={:.2f};Y={:.2f};u1={:.5f};v1={:.5f};u2={:.5f};v2={:.5f};s={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s))    
            TravelTime  = self._unit_time(_T(y,x,a,Y,u1,v1,u2,v2,self.s))
            CrossPoints = [self.Cell_s.cx-self.Cell_s.cx_lb,self.Cell_s.cy-self.Cell_s.cy_lb]
            CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
            if self.debugging:
                print('------ TravelTime={:.2f};CrossPoints=[{:.2f},{:.2f}];CellPoints=[{:.2f},{:.5f}]'.format(TravelTime,CrossPoints[0],CrossPoints[1],CellPoints[0],CellPoints[1]))    
            return TravelTime,CrossPoints,CellPoints

        def _top_left_corner(self):
            u1 = -self.Cell_s.getuC()*self.zx; v1 = self.Cell_s.getvC()*self.zx
            u2 = -self.Cell_n.getuC()*self.zx; v2 = self.Cell_n.getvC()*self.zx
            x  = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx - self.Cell_s.cx_lb,self.Cell_s.cy))
            y  = self.fdist.value((self.Cell_s.cx-self.Cell_s.cx_lb,self.Cell_s.cy),(self.Cell_s.cx-self.Cell_s.cx_lb,self.Cell_s.cy+self.Cell_s.cy_ub))
            a  = -self.fdist.value((self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx + self.Cell_n.cx_ub,self.Cell_n.cy))
            Y  = self.fdist.value((self.Cell_s.cx-(abs(self.Cell_s.cx_lb) + abs(self.Cell_n.cx_ub)),self.Cell_s.cy),\
                            (self.Cell_s.cx-(abs(self.Cell_s.cx_lb) + abs(self.Cell_n.cx_ub)),self.Cell_n.cy))
            if self.debugging:
                print('Top Left: y={:.2f};x={:.2f};a={:.2f};Y={:.2f};u1={:.5f};v1={:.5f};u2={:.5f};v2={:.5f};s={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s))
            TravelTime  = self._unit_time(_T(y,x,a,Y,u1,v1,u2,v2,self.s))
            CrossPoints = [self.Cell_s.cx-self.Cell_s.cx_lb,self.Cell_s.cy+self.Cell_s.cy_ub]
            CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
            if self.debugging:
                print('------ TravelTime={:.2f};CrossPoints=[{:.2f},{:.2f}];CellPoints=[{:.2f},{:.5f}]'.format(TravelTime,CrossPoints[0],CrossPoints[1],CellPoints[0],CellPoints[1]))   
            return TravelTime,CrossPoints,CellPoints




        if self.debugging:
            print('============================================')

        # ======= Determining the Newton Value dependent on case
        # Case 2 - Positive Longitude
        if (self.df_x > self.Cell_s.width/2) and (abs(self.df_y) <= (self.Cell_s.height/2)):
            TravelTime,CrossPoints,CellPoints = _positive_longitude(self)
        # Case -2 - Negative Longitude
        elif (self.df_x < -(self.Cell_s.width/2)) and (abs(self.df_y) <= (self.Cell_s.height/2)):
            TravelTime,CrossPoints,CellPoints = _negative_longitude(self)
        # Case -4 - Positive Latitude
        elif (self.df_y > (self.Cell_s.height/2)) and (abs(self.df_x) <= (self.Cell_s.width/2)):
            TravelTime,CrossPoints,CellPoints = _positive_latitude(self)
        # Case 4 - Negative Latitude
        elif (self.df_y < -(self.Cell_s.height/2)) and (abs(self.df_x) <= (self.Cell_s.width/2)):
            TravelTime,CrossPoints,CellPoints = _negative_latitude(self)
        # Case 1 - Top Right Corner 
        elif (self.df_y > (self.Cell_s.height/2)) and (self.df_x > (self.Cell_s.width/2)):
            TravelTime,CrossPoints,CellPoints = _top_right_corner(self)    
        # Case 3 - Bottom Right Corner 
        elif (self.df_y < -(self.Cell_s.height/2)) and (self.df_x > (self.Cell_s.width/2)):
            TravelTime,CrossPoints,CellPoints = _bottom_right_corner(self)  
        # Case -1 - Bottom Left Corner 
        elif (self.df_y < -(self.Cell_s.height/2)) and (self.df_x < -(self.Cell_s.width/2)):
            TravelTime,CrossPoints,CellPoints = _bottom_left_corner(self)
        # Case -3 - Top Left Corner 
        elif (self.df_y > (self.Cell_s.height/2)) and (self.df_x < -(self.Cell_s.width/2)):
            TravelTime,CrossPoints,CellPoints = _top_left_corner(self)
        else:
            print('---> Issue with cell (Xsc,Ysc)={:.2f};{:.2f}; (dx,dy)={:.2f},{:.2f}; (diffX,diffY)={:.2f},{:.2f}'.format(self.Cell_s.cx,self.Cell_s.cy,self.Cell_s.width/2,self.Cell_s.height/2,self.df_x,self.df_y))
            
            TravelTime  = np.inf
            CrossPoints = [np.nan,np.nan]
            CellPoints  = [np.nan,np.nan]

        CrossPoints[0] = np.clip(CrossPoints[0],self.Cell_n.long,(self.Cell_n.long+self.Cell_n.width))
        CrossPoints[1] = np.clip(CrossPoints[1],self.Cell_n.lat,(self.Cell_n.lat+self.Cell_n.height))

        return TravelTime, CrossPoints, CellPoints



# class SmoothedNewtonianDistance:
#     def __init__(self,Mesh,Cell_s,Cell_n,s):

#         self.Mesh   = Mesh
#         self.Cell_s = Cell_s
#         self.Cell_n = Cell_n
#         self.s      = s
#         self.fdist.value  = _Euclidean_distance
#         self.R      = 6371

#         # Determining the distance between the cell centres to 
#         #be used in defining the case
#         self.df_x = (self.Cell_n.x+self.Cell_n.dx) -  (self.Cell_s.x+self.Cell_s.dx)
#         self.df_y = (self.Cell_n.y+self.Cell_n.dy) -  (self.Cell_s.y+self.Cell_s.dy)
        
#         # Determine the distance to the edge of the box. This
#         #is important for waypoint cases
#         if np.sign(self.df_x) == 1:
#             self.S_dx = self.Cell_s.cx_ub; self.N_dx = -self.Cell_n.cx_lb
#         else:
#             self.S_dx = -self.Cell_s.cx_lb; self.N_dx = self.Cell_n.cx_ub  
#         #dY       
#         if np.sign(self.df_y) == 1:
#             self.S_dy = self.Cell_s.cy_ub; self.N_dy = -self.Cell_n.cy_lb
#         else:
#             self.S_dy = -self.Cell_s.cy_lb; self.N_dy = self.Cell_n.cy_ub 


#     def value(self):

#         if ((abs(self.df_x) >= (self.Cell_s.dx/2)) and (abs(self.df_y) < (self.Cell_s.dy/2))):
#             TravelTime, CrossPoints, CellPoints = self._long_case()
#         elif (abs(self.df_x) < self.Cell_s.dx/2) and (abs(self.df_y) >= self.Cell_s.dy/2):
#             TravelTime, CrossPoints, CellPoints = self._lat_case()
#         elif (abs(self.df_x) >= self.Cell_s.dx/2) and (abs(self.df_y) >= self.Cell_s.dy/2):
#             TravelTime, CrossPoints, CellPoints = self._corner_case()
#         else:
#             TravelTime  = np.inf
#             CrossPoints = [np.nan,np.nan]
#             CellPoints  = [np.nan,np.nan]

#         CrossPoints[0] = np.clip(CrossPoints[0],self.Cell_n.x,(self.Cell_n.x+self.Cell_n.dx))
#         CrossPoints[1] = np.clip(CrossPoints[1],self.Cell_n.y,(self.Cell_n.y+self.Cell_n.dy))

#         return TravelTime, CrossPoints, CellPoints

#     def _long_case(self):

#         def _F(y,x,a,Y,u1,v1,u2,v2,s,R,λ_s,φ_r):
#             θ  = y/R + λ_s
#             zl = x*math.cos(θ)
#             ψ  = -(Y-y)/R + φ_r
#             zr = a*math.cos(ψ)

#             d1  = np.sqrt(zl**2 + y**2)
#             d2  = np.sqrt(zl**2 + (Y-y)**2)
#             C1  = s**2 - u1**2 - v1**2
#             D1  = zl*u1 + y*v1
#             X1  = np.sqrt(D1**2 + C1*(d1**2))
#             C2  = s**2 - u2**2 - v2**2
#             D2  = zr*u2 + (Y-y)*v2
#             X2  = np.sqrt(D2**2 + C2*(d2**2))

#             dzr = (-a*math.sin(ψ))/R
#             dzl = (-x*math.sin(θ))/R

#             F  = (X1+X2)*y - ((X1-D1)*X2*v1)/C1 + ((X2-D2)*X1*v2)/C2\
#                 - Y*X1 + dzr*(zr-((X2-D2)/C2))*X1 + dzl*(zl-((X1-D1)/C1))*X2
#             return F

#         def _dF(y,x,a,Y,u1,v1,u2,v2,s,R,λ_s,φ_r):
#             θ  = y/R + λ_s
#             zl = x*math.cos(θ)
#             ψ  = -(Y-y)/R + φ_r
#             zr = a*math.cos(ψ)

#             d1  = np.sqrt(zl**2 + y**2)
#             d2  = np.sqrt(zl**2 + (Y-y)**2)
#             C1  = s**2 - u1**2 - v1**2
#             D1  = zl*u1 + y*v1
#             X1  = np.sqrt(D1**2 + C1*(d1**2))
#             C2  = s**2 - u2**2 - v2**2
#             D2  = zr*u2 + (Y-y)*v2
#             X2  = np.sqrt(D2**2 + C2*(d2**2))

#             dzr = (-a*math.sin(ψ))/R
#             dzl = (-x*math.sin(θ))/R
#             dD1 = dzl*u1 + v1
#             dD2 = dzr*u2 - v2
#             dX1 = (D1*v1 + C1*y + dzl*(D1*u1 + C1*zl))/X1
#             dX2 = (-v2*D2 - C2*(Y-y) + dzr*(D2*u2 + C2*zr))/X2        

#             dF = (X1+X2) + y*(dX1 + dX2) - (v1/C1)*(dX2*(X1-D1) + X2*(dX1-dD1))\
#                 + (v2/C2)*(dX1*(X2-D2) + X1*(dX2-dD2))\
#                 - Y*dX1 - (zr/(R**2))*(zr-((X2-D2)*u2)/C2)*X1\
#                 - (zl/(R**2))*(zl-((X1-D1)*u1)/C1)*X2\
#                 + dzr*(dzr-(u2/C2)*(dX2-dD2))*X1\
#                 + dzl*(dzl-(u1/C1)*(dX1-dD1))*X2\
#                 + dzr(zr-((X2-D2)*u2)/C2)*dX1 + dzl*(zl-((X1-D1)*u1)/C1)*dX2
#             return dF 

#         def _T(y,x,a,Y,u1,v1,u2,v2,s,R,λ_s,φ_r):
#             θ  = y/R + λ_s
#             zl = x*math.cos(θ)
#             ψ  = -(Y-y)/R + φ_r
#             zr = a*math.cos(ψ)

#             d1  = np.sqrt(zl**2 + y**2)
#             d2  = np.sqrt(zl**2 + (Y-y)**2)
#             C1  = s**2 - u1**2 - v1**2
#             D1  = zl*u1 + y*v1
#             X1  = np.sqrt(D1**2 + C1*(d1**2))
#             t1  = (X1-D1)/C1
#             C2  = s**2 - u2**2 - v2**2
#             D2  = zr*u2 + (Y-y)*v2
#             X2  = np.sqrt(D2**2 + C2*(d2**2))
#             t2  = (X2-D2)/C2
#             TT  = t1 + t2
#             return TT

#         u1 = np.sign(self.df_x)*self.Cell_s.getuC(); v1 = self.Cell_s.getvC()
#         u2 = np.sign(self.df_x)*self.Cell_n.getuC(); v2 = self.Cell_n.getvC()
#         λ_s = self.Cell_s.cy
#         φ_r = self.Cell_n.cy
#         x     = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy), (self.Cell_s.cx + self.S_dx,self.Cell_s.cy))
#         a     = self.fdist.value((self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx + self.N_dx,self.Cell_n.cy))
#         Y     = self.fdist.value((self.Cell_s.cx + np.sign(self.df_x)*(abs(self.S_dx) + abs(self.N_dx)), self.Cell_s.cy), (self.Cell_s.cx + np.sign(self.df_x)*(abs(self.S_dx) + abs(self.N_dx)),self.Cell_n.cy))
#         ang   = np.arctan((self.Cell_n.cy - self.Cell_s.cy)/(self.Cell_n.cx - self.Cell_s.cx))
#         yinit = np.tan(ang)*(self.S_dx)
#         try:
#             y  = optimize.newton(_F,yinit,args=(x,a,Y,u1,v1,u2,v2,self.s,self.R,λ_s,φ_r),fprime=_dF)
#         except:
#             y  = yinit
#         TravelTime  = _T(y,x,a,Y,u1,v1,u2,v2,self.s,self.R,λ_s,φ_r)
#         CrossPoints = self.fdist.value((self.Cell_s.cx + self.S_dx,self.Cell_s.cy),(0.0,y),forward=False)
#         CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]

#         return TravelTime,CrossPoints,CellPoints


#     def _lat_case(self):
#         def _F(y,x,a,Y,u1,v1,u2,v2,s,R,λ,θ,ψ):
#             r1  = math.cos(λ)/math.cos(θ)
#             r2  = math.cos(ψ)/math.cos(θ)

#             d1  = np.sqrt(x**2 + (r1*y)**2)
#             d2  = np.sqrt(a**2 + (r2*(Y-y))**2)
#             C1  = s**2 - u1**2 - v1**2
#             C2  = s**2 - u2**2 - v2**2
#             D1  = x*u1 + r1*v1*Y
#             D2  = a*u2 + r2*v2*(Y-y)
#             X1  = np.sqrt(D1**2 + C1*(d1**2))
#             X2  = np.sqrt(D2**2 + C2*(d2**2)) 

#             F = ((r2**2)*X1 + (r1**2)*X2)*y - ((r1*(X1-D1)*X2*v2)/C1) + ((r2*(X2-D2)*X1*v2)/C2) - (r2**2)*Y*X1
#             return F

#         def _dF(y,x,a,Y,u1,v1,u2,v2,s,R,λ,θ,ψ):
#             r1  = math.cos(λ)/math.cos(θ)
#             r2  = math.cos(ψ)/math.cos(θ)

#             d1  = np.sqrt(x**2 + (r1*y)**2)
#             d2  = np.sqrt(a**2 + (r2*(Y-y))**2)
#             C1  = s**2 - u1**2 - v1**2
#             C2  = s**2 - u2**2 - v2**2
#             D1  = x*u1 + r1*v1*Y
#             D2  = a*u2 + r2*v2*(Y-y)
#             X1  = np.sqrt(D1**2 + C1*(d1**2))
#             X2  = np.sqrt(D2**2 + C2*(d2**2))   
            
#             dD1 = r1*v1
#             dD2 = -r2*v2
#             dX1 = (r1*(D1*v1 + r1*C1*y))/X1
#             dX2 = (-r2*(D2*v2 + r2*C2*(Y-y)))

#             dF = ((r2**2)*X1 + (r1**2)*X2) + y*((r2**2)*dX1 + (r1**2)*dX2)\
#                 - ((r1*v1)/C1)*((dX1-dD1)*X2 + (X1-D1)*dX2)\
#                 + ((r2*v2)/C2)*((dX2-dD2)*X1 + (X2-D2)*dX1)\
#                 - (r2**2)*Y*dX1

#             return dF

#         def _T(y,x,a,Y,u1,v1,u2,v2,s,R,λ,θ,ψ):
#             r1  = math.cos(λ)/math.cos(θ)
#             r2  = math.cos(ψ)/math.cos(θ)

#             d1  = np.sqrt(x**2 + (r1*y)**2)
#             d2  = np.sqrt(a**2 + (r2*(Y-y))**2)
#             C1  = s**2 - u1**2 - v1**2
#             C2  = s**2 - u2**2 - v2**2
#             D1  = x*u1 + r1*v1*Y
#             D2  = a*u2 + r2*v2*(Y-y)
#             X1  = np.sqrt(D1**2 + C1*(d1**2))
#             X2  = np.sqrt(D2**2 + C2*(d2**2))
#             t1  = (X1-D1)/C1
#             t2  = (X2-D2)/C2

#             TT  = t1+t2
#             return TT     

#         u1 = np.sign(self.df_y)*self.Cell_s.getvC(); v1 = self.Cell_s.getuC()
#         u2 = np.sign(self.df_y)*self.Cell_n.getvC(); v2 = self.Cell_n.getuC()

#         x  = self.fdist.value((self.Cell_s.cy,self.Cell_s.cx), (self.Cell_s.cy + self.S_dy,self.Cell_s.cx))
#         a  = self.fdist.value((self.Cell_n.cy,self.Cell_n.cx), (self.Cell_n.cy + self.N_dy,self.Cell_n.cx))
#         Y  = self.fdist.value((self.Cell_s.cy + np.sign(self.df_y)*(abs(self.S_dy) + abs(self.N_dy)), self.Cell_s.cx), (self.Cell_s.cy + np.sign(self.df_y)*(abs(self.S_dy) + abs(self.N_dy)),self.Cell_n.cx))
#         ang= np.arctan((self.Cell_n.cx - self.Cell_s.cx)/(self.Cell_n.cy - self.Cell_s.cy))
#         yinit  = np.tan(ang)*(self.S_dy)
        
#         λ=self.Cell_s.cy
#         θ=self.Cell_s.cy + self.S_dy
#         ψ=self.Cell_n.cy

#         try:
#             y  = optimize.newton(_F,yinit,args=(x,a,Y,u1,v1,u2,v2,self.s,self.R,λ,θ,ψ),fprime=_dF)
#         except:
#             y  = yinit
#         TravelTime  = _T(y,x,a,Y,u1,v1,u2,v2,self.s,self.R,λ,θ,ψ)
#         CrossPoints = self.fdist.value((self.Cell_s.cx + self.S_dx,self.Cell_s.cy),(0.0,y),forward=False)
#         CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]        

#         return TravelTime,CrossPoints,CellPoints


#     def _corner_case(self,start_p,crossing_p,end_p,start_index,end_index):
#         '''
#             Corner cases as outline in Part 4 of the latex formulations

#             Bug/Corrections
#                 - Return the new point locations
#         '''


#         # Defining the lat/long of the points
#         Xs,Ys = start_p
#         Xc,Yc = crossing_p
#         Xe,Ye = end_p 

#         # # Determine the intersection point on the edge where end_p is assuming a straight path through corner
#         Y_line = ((Yc-Ys)/(Xc-Xs))*(Xe-Xs) + Ys

#         CornerCells = []
#         for index in self.Mesh.NearestNeighbours(start_index):
#             cell = self.Mesh.cells[index]
#             if ((((np.array(cell._bounding_points)[::2,:] - np.array([Xc,Yc])[None,:])**2).sum(axis=1)) == 0).any() & (index!=end_index):
#                 CornerCells.append([index,cell.x+cell.dx/2,cell.y+cell.dy/2]) 
#         CornerCells = np.array(CornerCells)

#         # ====== Determining the crossing points & their corresponding index
#         # Case 1 - Top Right
#         if (np.sign(self.df_x) == 1) and (np.sign(self.df_y) == 1):
#             if Ye > Y_line:
#                 idx  = int(CornerCells[CornerCells[:,1].argmin(),0])
#                 cell = self.Mesh.cells[idx]
#                 Crp1_x,Crp1_y = cell.x+cell.dx/2, cell.y
#                 Crp2_x,Crp2_y = cell.x+cell.dx, cell.y+cell.dy/2
#             elif Ye < Y_line:
#                 idx  = int(CornerCells[CornerCells[:,1].argmax(),0])
#                 cell = self.Mesh.cells[idx]
#                 Crp1_x,Crp1_y = cell.x, cell.y+cell.dy/2
#                 Crp2_x,Crp2_y = cell.x+cell.dx/2, cell.y+cell.dy

#         # Case -3 - Top Left
#         if (np.sign(self.df_x) == -1) and (np.sign(self.df_y) == 1):
#             if Ye > Y_line:
#                 idx  = int(CornerCells[CornerCells[:,1].argmin(),0])
#                 cell = self.Mesh.cells[idx]
#                 Crp1_x,Crp1_y = cell.x+cell.dx/2, cell.y
#                 Crp2_x,Crp2_y = cell.x, cell.y+cell.dy/2
#             elif Ye < Y_line:
#                 idx  = int(CornerCells[CornerCells[:,1].argmax(),0])
#                 cell = self.Mesh.cells[idx]
#                 Crp1_x,Crp1_y = cell.x + cell.dx, cell.y+cell.dy/2
#                 Crp2_x,Crp2_y = cell.x+cell.dx/2, cell.y+cell.dy

#         # Case -1 - Bottom Left
#         if (np.sign(self.df_x) == -1) and (np.sign(self.df_y) == -1):
#             if Ye > Y_line:
#                 idx  = int(CornerCells[CornerCells[:,1].argmin(),0])
#                 cell = self.Mesh.cells[idx]
#                 Crp1_x,Crp1_y = cell.x+cell.dx, cell.y+cell.dy/2
#                 Crp2_x,Crp2_y = cell.x+cell.dx/2, cell.y
#             elif Ye < Y_line:
#                 idx  = int(CornerCells[CornerCells[:,1].argmax(),0])
#                 cell = self.Mesh.cells[idx]
#                 Xr1,Yr1 = cell.x + cell.dx/2, cell.y+cell.dy
#                 Xr2,Yr2 = cell.x, cell.y+cell.dy/2

#         # Case 3 - Bottom Right
#         if (np.sign(self.df_x) == -1) and (np.sign(self.df_y) == -1):
#             if Ye > Y_line:
#                 idx  = int(CornerCells[CornerCells[:,1].argmin(),0])
#                 cell = self.Mesh.cells[idx]
#                 Crp1_x,Crp1_y = cell.x, cell.y+cell.dy/2
#                 Crp2_x,Crp2_y = cell.x+cell.dx/2, cell.y
#             elif Ye < Y_line:
#                 idx  = int(CornerCells[CornerCells[:,1].argmax(),0])
#                 cell = self.Mesh.cells[idx]
#                 Crp1_x,Crp1_y = cell.x + cell.dx/2, cell.y+cell.dy
#                 Crp2_x,Crp2_y = cell.x + cell.dx, cell.y+cell.dy/2

#         # Appending the crossing points and their relative index
