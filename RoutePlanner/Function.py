import numpy as np
import math
from scipy import optimize

def sign(x):
    s = math.copysign(1, x)
    return s   


def _Euclidean_distance(origin, dest_dist,forward=True):
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

    m_per_deglat          = 111.386*1000
    m_per_deglonAtEquator = 111.321*1000
    lon1,lat1            = origin

    #kmperdeglonAtEquator = kmperdeglonAtEquator/(math.cos(lat1))

    if forward:
        lon2,lat2 = dest_dist
        val = np.sqrt(((lat2-lat1)*m_per_deglat)**2 + ((lon2-lon1)*m_per_deglonAtEquator)**2)
    else:
        dist_x,dist_y = dest_dist        
        val = [lon1+(dist_x/m_per_deglonAtEquator),lat1+(dist_y/m_per_deglat)]

    return val


# ================================================================
# ================================================================
# ================================================================
# ================================================================



class NewtonianDistance:
    def __init__(self,Cell_s,Cell_n,s,debugging=False,maxiter=500,optimizer_tol=1e-7):
        self.Cell_s = Cell_s
        self.Cell_n = Cell_n
        self.s      = s
        self.fdist  = _Euclidean_distance
        self.debugging = debugging
        self.maxiter   = maxiter
        self.optimizer_tol = optimizer_tol

        # Determining the distance between the cell centres to 
        #be used in defining the case
        self.df_x = (self.Cell_n.long+self.Cell_n.width) -  (self.Cell_s.long+self.Cell_s.width)
        self.df_y = (self.Cell_n.lat+self.Cell_n.height) -  (self.Cell_s.lat+self.Cell_s.height)

    def NewtonOptimisation(self,f,df,y0,x,a,Y,u1,v1,u2,v2,s):
            for iter in range(self.maxiter):
                F  = f(y0,x,a,Y,u1,v1,u2,v2,s,debugging=self.debugging)
                dF = df(y0,x,a,Y,u1,v1,u2,v2,s)
                if self.debugging:
                    print('---Iteration {}: y={:.2f}, F={:.5f}, dF={:.2f}'.format(iter,y0,F,dF))
                y0  = y0 - (F/dF)
                if F < self.optimizer_tol:
                    break
            return y0


    def value(self):
        # -- Initially defining Newton function & its derivative
        def _F(y,x,a,Y,u1,v1,u2,v2,s,debugging=False):
            d1 = x**2 + y**2
            d2 = a**2 + (Y-y)**2
            C1 = s**2 - u1**2 - v1**2
            C2 = s**2 - u2**2 - v2**2
            D1 = x*u1 + y*v1
            D2 = a*u2 + (Y-y)*v2
            X1 = math.sqrt(D1**2 + C1*(d1**2))
            X2 = math.sqrt(D2**2 + C2*(d2**2))
            F  = X2*(y-((v1*(X1-D1))/C1)) + X1*(y-Y+((v2*(X2-D2))/C2)) 

            if debugging:
                print('d1={},d2={},C1={},C2={},D1={},D2={},X1={},X2={},F={}'.format(d1,d2,C1,C2,D1,D2,X1,X2,F))

            return F

        def _dF(y,x,a,Y,u1,v1,u2,v2,s):
            d1 = x**2 + y**2
            d2 = a**2 + (Y-y)**2
            C1 = s**2 - u1**2 - v1**2
            C2 = s**2 - u2**2 - v2**2
            D1 = x*u1 + y*v1
            D2 = a*u2 + (Y-y)*v2
            X1 = math.sqrt(D1**2 + C1*(d1**2))
            X2 = math.sqrt(D2**2 + C2*(d2**2))
            dD1 = v1
            dD2 = -v2
            dX1 = (D1*v1 + C1*y)/X1
            dX2 = (D2*v2 - C1*(Y-y))/X1
            dF = (X1+X2) + y*(dX1 + dX2) - (v1/C1)*(dX2*(X1-D1) + X2*(dX1-dD1)) + (v2/C2)*(dX1*(X2-D2)+X1*(dX2-dD2)) - Y*dX1
            return dF

        def _T(y,x,a,Y,u1,v1,u2,v2,s):
            d1 = x**2 + y**2
            d2 = a**2 + (Y-y)**2
            C1 = s**2 - u1**2 - v1**2
            C2 = s**2 - u2**2 - v2**2
            D1 = x*u1 + y*v1
            D2 = a*u2 + (Y-y)*v2
            X1 = math.sqrt(D1**2 + C1*(d1**2))
            X2 = math.sqrt(D2**2 + C2*(d2**2))
            t1 = (X1-D1)/C1
            t2 = (X2-D2)/C2
            T  = t1+t2 
            return T

        def _degrenative(y,x,a,Y,u1,v1,u2,v2,s):
            '''
                Degrenative if Vehicle speed is unable to transect between centroid due to currents.
            '''

            d1 = x**2 + y**2
            d2 = a**2 + (Y-y)**2
            C1 = s**2 - u1**2 - v1**2
            C2 = s**2 - u2**2 - v2**2
            D1 = x*u1 + y*v1
            D2 = a*u2 + (Y-y)*v2

            if (D1<0) or (D2<0):
                return True
            else:
                return False


        def _positive_longitude(self):
            '''
                INCLUDE
            '''
            u1          = self.Cell_s.getuC(); v1 = self.Cell_s.getvC()
            u2          = self.Cell_n.getuC(); v2 = self.Cell_n.getvC()
            x           = self.fdist((self.Cell_s.cx,self.Cell_s.cy), (self.Cell_s.cx + self.Cell_s.cx_ub,self.Cell_s.cy))
            a           = self.fdist((self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx - self.Cell_n.cx_lb,self.Cell_n.cy))
            Y           = sign(self.Cell_n.cy-self.Cell_s.cy)*self.fdist((self.Cell_s.cx+(abs(self.Cell_s.cx_ub)+abs(self.Cell_n.cx_lb)),self.Cell_s.cy),\
                                     (self.Cell_s.cx+(abs(self.Cell_s.cx_ub)+abs(self.Cell_n.cx_lb)),self.Cell_n.cy))
            ang         = math.atan((self.Cell_n.cy - self.Cell_s.cy)/(self.Cell_n.cx - self.Cell_s.cx))
            yinit       = math.tan(ang)*(self.Cell_s.cx_ub)
            if self.debugging:
                print('Positive Long: Yinit={:.2f},x={:.2f},a={:.2f},Y={:.2f},u1={:.5f},v1={:.5f},u2={:.5f},v2={:.5f},s={:.2f}'.format(yinit,x,a,Y,u1,v1,u2,v2,self.s))
            y = self.NewtonOptimisation(_F,_dF,yinit,x,a,Y,u1,v1,u2,v2,self.s)
            TravelTime  = _T(y,x,a,Y,u1,v1,u2,v2,self.s)
            CrossPoints = self.fdist((self.Cell_s.cx + self.Cell_s.cx_ub,self.Cell_s.cy),(0.0,y),forward=False)
            CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]

            return TravelTime,CrossPoints,CellPoints

        def _negative_longitude(self):
            '''
                INCLUDE
            '''
            u1          = -self.Cell_s.getuC(); v1 = self.Cell_s.getvC()
            u2          = -self.Cell_n.getuC(); v2 = self.Cell_n.getvC()
            x           = self.fdist((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx-self.Cell_s.cx_lb,self.Cell_s.cy))
            a           = self.fdist((self.Cell_n.cx,self.Cell_n.cy),(self.Cell_n.cx+self.Cell_n.cx_ub,self.Cell_n.cy))
            Y           = sign(self.Cell_n.cy-self.Cell_s.cy)*\
                               self.fdist((self.Cell_s.cx-(abs(self.Cell_s.cx_lb)+abs(self.Cell_n.cx_ub)),self.Cell_s.cy),\
                                          (self.Cell_s.cx-(abs(self.Cell_s.cx_lb)+abs(self.Cell_n.cx_ub)),self.Cell_n.cy))
            ang         = np.arctan((self.Cell_n.cy - self.Cell_s.cy)/(self.Cell_n.cx - self.Cell_s.cx))
            yinit       = math.tan(ang)*(-self.Cell_s.cx_lb)
            if self.debugging:
                print('Negative Long: Yinit={:.2f},x={:.2f},a={:.2f},Y={:.2f},u1={:.5f},v1={:.5f},u2={:.5f},v2={:.5f},s={:.2f}'.format(yinit,x,a,Y,u1,v1,u2,v2,self.s))

            y = self.NewtonOptimisation(_F,_dF,yinit,x,a,Y,u1,v1,u2,v2,self.s)
            TravelTime  = _T(y,x,a,Y,u1,v1,u2,v2,self.s)
            CrossPoints = self.fdist((self.Cell_s.cx - self.Cell_s.cx_lb,self.Cell_s.cy),(0.0,y),forward=False)
            CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]


            return TravelTime,CrossPoints,CellPoints

        def _positive_latitude(self):
            '''
                Case -4
            '''

            u1          = self.Cell_s.getvC(); v1 = -self.Cell_s.getuC()
            u2          = self.Cell_n.getvC(); v2 = -self.Cell_n.getuC()
            x           = self.fdist((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx,self.Cell_s.cy + self.Cell_s.cy_ub))
            a           = -self.fdist((self.Cell_n.cx,self.Cell_n.cy),(self.Cell_n.cx,self.Cell_n.cy - self.Cell_n.cy_lb))
            Y           = -sign(self.Cell_n.cx-self.Cell_s.cx)*\
                          self.fdist((self.Cell_s.cx, self.Cell_s.cy + (abs(self.Cell_s.cy_ub) + abs(self.Cell_n.cy_lb))),\
                                     (self.Cell_n.cx, self.Cell_s.cy + (abs(self.Cell_s.cy_ub) + abs(self.Cell_n.cy_lb))))
            ang         = np.arctan(-(self.Cell_n.cx - self.Cell_s.cx)/(self.Cell_n.cy - self.Cell_s.cy))
            yinit       = np.tan(ang)*(self.Cell_s.cy_ub)
            if self.debugging:
                print('Postive Lat: Yinit={:.2f},x={:.2f},a={:.2f},Y={:.2f},u1={:.5f},v1={:.5f},u2={:.5f},v2={:.5f},s={:.2f}'.format(yinit,x,a,Y,u1,v1,u2,v2,self.s))
            y = self.NewtonOptimisation(_F,_dF,yinit,x,a,Y,u1,v1,u2,v2,self.s)        
            TravelTime  = _T(y,x,a,Y,u1,v1,u2,v2,self.s)
            CrossPoints = self.fdist((self.Cell_s.cx,self.Cell_s.cy+self.Cell_s.cy_ub),(-y,0.0),forward=False)
            CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]

            return TravelTime,CrossPoints,CellPoints


        def _negative_latitude(self):
            '''
                Case 4
            '''

            u1          = -self.Cell_s.getvC(); v1 = -self.Cell_s.getuC()
            u2          = -self.Cell_n.getvC(); v2 = -self.Cell_n.getuC()
            x           = self.fdist((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx,self.Cell_s.cy - self.Cell_s.cy_lb))
            a           = -self.fdist((self.Cell_n.cx,self.Cell_n.cy),(self.Cell_n.cx,self.Cell_n.cy + self.Cell_n.cy_ub))
            Y           = -sign(self.Cell_n.cx-self.Cell_s.cx)*\
                          self.fdist((self.Cell_s.cx, self.Cell_s.cy - (abs(self.Cell_s.cy_ub) + abs(self.Cell_n.cy_lb))),\
                                     (self.Cell_n.cx, self.Cell_s.cy - (abs(self.Cell_s.cy_ub) + abs(self.Cell_n.cy_lb))))
            ang         = np.arctan(-(self.Cell_n.cx - self.Cell_s.cx)/(self.Cell_n.cy - self.Cell_s.cy))
            yinit       = np.tan(ang)*(self.Cell_s.cy_lb)
            if self.debugging:
                print('Negative Lat: Yinit={:.2f},x={:.2f},a={:.2f},Y={:.2f},u1={:.5f},v1={:.5f},u2={:.5f},v2={:.5f},s={:.2f}'.format(yinit,x,a,Y,u1,v1,u2,v2,self.s))

            y = self.NewtonOptimisation(_F,_dF,yinit,x,a,Y,u1,v1,u2,v2,self.s)        
            TravelTime  = _T(y,x,a,Y,u1,v1,u2,v2,self.s)
            CrossPoints = self.fdist((self.Cell_s.cx,self.Cell_s.cy-self.Cell_s.cy_lb),(-y,0.0),forward=False)
            CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]

            return TravelTime,CrossPoints,CellPoints

        def _top_right_corner(self):
            u1 = self.Cell_s.getuC(); v1 = self.Cell_s.getvC()
            u2 = self.Cell_n.getuC(); v2 = self.Cell_n.getvC()
            x  = self.fdist((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx + self.Cell_s.cx_ub,self.Cell_s.cy))
            y  = self.fdist((self.Cell_s.cx+self.Cell_s.cx_ub,self.Cell_s.cy),(self.Cell_s.cx+self.Cell_s.cx_ub,self.Cell_s.cy+self.Cell_s.cy_ub))
            a  = -self.fdist((self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx - self.Cell_n.cx_lb,self.Cell_n.cy))
            Y  = self.fdist((self.Cell_s.cx+(abs(self.Cell_s.cx_ub) + abs(self.Cell_n.cx_lb)),self.Cell_s.cy),\
                            (self.Cell_s.cx+(abs(self.Cell_s.cx_ub) + abs(self.Cell_n.cx_lb)),self.Cell_n.cy))
            if self.debugging:
                print('Top Right Corner: y={:.2f},x={:.2f},a={:.2f},Y={:.2f},u1={:.5f},v1={:.5f},u2={:.5f},v2={:.5f},s={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s))

            TravelTime  = _T(y,x,a,Y,u1,v1,u2,v2,self.s)
            CrossPoints = [self.Cell_s.cx+self.Cell_s.cx_ub,self.Cell_s.cy+self.Cell_s.cy_ub]
            CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
            return TravelTime,CrossPoints,CellPoints

        def _bottom_right_corner(self):
            u1 = self.Cell_s.getuC(); v1 = self.Cell_s.getvC()
            u2 = self.Cell_n.getuC(); v2 = self.Cell_n.getvC()
            x  = self.fdist((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx + self.Cell_s.cx_ub,self.Cell_s.cy))
            y  = -self.fdist((self.Cell_s.cx+self.Cell_s.cx_ub,self.Cell_s.cy),(self.Cell_s.cx+self.Cell_s.cx_ub,self.Cell_s.cy-self.Cell_s.cy_lb))
            a  = -self.fdist((self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx - self.Cell_n.cx_lb,self.Cell_n.cy))
            Y  = -self.fdist((self.Cell_s.cx+(abs(self.Cell_s.cx_ub) + abs(self.Cell_n.cx_lb)),self.Cell_s.cy),\
                            (self.Cell_s.cx+(abs(self.Cell_s.cx_ub) + abs(self.Cell_n.cx_lb)),self.Cell_n.cy))

            if self.debugging:
                print('Bottom Right Corner: y={:.2f},x={:.2f},a={:.2f},Y={:.2f},u1={:.5f},v1={:.5f},u2={:.5f},v2={:.5f},s={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s))
                        
            TravelTime  = _T(y,x,a,Y,u1,v1,u2,v2,self.s)
            CrossPoints = [self.Cell_s.cx+self.Cell_s.cx_ub,self.Cell_s.cy-self.Cell_s.cy_lb]
            CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
            return TravelTime,CrossPoints,CellPoints

        def _bottom_left_corner(self):
            u1 = -self.Cell_s.getuC(); v1 = self.Cell_s.getvC()
            u2 = -self.Cell_n.getuC(); v2 = self.Cell_n.getvC()
            x  = self.fdist((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx - self.Cell_s.cx_lb,self.Cell_s.cy))
            y  = -self.fdist((self.Cell_s.cx-self.Cell_s.cx_lb,self.Cell_s.cy),(self.Cell_s.cx-self.Cell_s.cx_lb,self.Cell_s.cy-self.Cell_s.cy_lb))
            a  = -self.fdist((self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx + self.Cell_n.cx_ub,self.Cell_n.cy))
            Y  = -self.fdist((self.Cell_s.cx-(abs(self.Cell_s.cx_lb) + abs(self.Cell_n.cx_ub)),self.Cell_s.cy),\
                             (self.Cell_s.cx-(abs(self.Cell_s.cx_lb) + abs(self.Cell_n.cx_ub)),self.Cell_n.cy))

            if self.debugging:
                print('Bottom Left Corner: y={:.2f},x={:.2f},a={:.2f},Y={:.2f},u1={:.5f},v1={:.5f},u2={:.5f},v2={:.5f},s={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s))

            TravelTime  = _T(y,x,a,Y,u1,v1,u2,v2,self.s)
            CrossPoints = [self.Cell_s.cx-self.Cell_s.cx_lb,self.Cell_s.cy-self.Cell_s.cy_lb]
            CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
            return TravelTime,CrossPoints,CellPoints

        def _top_left_corner(self):
            u1 = -self.Cell_s.getuC(); v1 = self.Cell_s.getvC()
            u2 = -self.Cell_n.getuC(); v2 = self.Cell_n.getvC()
            x  = self.fdist((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx - self.Cell_s.cx_lb,self.Cell_s.cy))
            y  = self.fdist((self.Cell_s.cx-self.Cell_s.cx_lb,self.Cell_s.cy),(self.Cell_s.cx-self.Cell_s.cx_lb,self.Cell_s.cy+self.Cell_s.cy_ub))
            a  = -self.fdist((self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx + self.Cell_n.cx_ub,self.Cell_n.cy))
            Y  = self.fdist((self.Cell_s.cx-(abs(self.Cell_s.cx_lb) + abs(self.Cell_n.cx_ub)),self.Cell_s.cy),\
                            (self.Cell_s.cx-(abs(self.Cell_s.cx_lb) + abs(self.Cell_n.cx_ub)),self.Cell_n.cy))

            if self.debugging:
                print('Top Left Corner: y={:.2f},x={:.2f},a={:.2f},Y={:.2f},u1={:.5f},v1={:.5f},u2={:.5f},v2={:.5f},s={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s))                            

            TravelTime  = _T(y,x,a,Y,u1,v1,u2,v2,self.s)
            CrossPoints = [self.Cell_s.cx-self.Cell_s.cx_lb,self.Cell_s.cy+self.Cell_s.cy_ub]
            CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
            return TravelTime,CrossPoints,CellPoints




        if self.debugging:
            print('============================================')


        # ======= Determining the Newton Value dependent on case
        # Case 2 - Positive Longitude
        if (self.df_x > self.Cell_s.width/2) and (abs(self.df_y) < (self.Cell_s.height/2)):
            TravelTime,CrossPoints,CellPoints = _positive_longitude(self)
        # Case -2 - Negative Longitude
        elif (self.df_x < self.Cell_s.width/2) and (abs(self.df_y) < (self.Cell_s.height/2)):
            TravelTime,CrossPoints,CellPoints = _negative_longitude(self)
        # Case -4 - Positive Latitude
        elif (self.df_y > (self.Cell_s.height/2)) and (abs(self.df_x) < (self.Cell_s.width/2)):
            TravelTime,CrossPoints,CellPoints = _positive_latitude(self)
        # Case 4 - Negative Latitude
        elif (self.df_y < (self.Cell_s.height/2)) and (abs(self.df_x) < (self.Cell_s.width/2)):
            TravelTime,CrossPoints,CellPoints = _negative_latitude(self)
        # Case 1 - Top Right Corner 
        elif (self.df_y > (self.Cell_s.height/2)) and (self.df_x > (self.Cell_s.width/2)):
            TravelTime,CrossPoints,CellPoints = _top_right_corner(self)    
        # Case 3 - Bottom Right Corner 
        elif (self.df_y < (self.Cell_s.height/2)) and (self.df_x > (self.Cell_s.width/2)):
            TravelTime,CrossPoints,CellPoints = _bottom_right_corner(self)  
        # Case -1 - Bottom Left Corner 
        elif (self.df_y < (self.Cell_s.height/2)) and (self.df_x < (self.Cell_s.width/2)):
            TravelTime,CrossPoints,CellPoints = _bottom_left_corner(self)
        # Case -3 - Top Left Corner 
        elif (self.df_y > (self.Cell_s.height/2)) and (self.df_x < (self.Cell_s.width/2)):
            TravelTime,CrossPoints,CellPoints = _top_left_corner(self)
        else:
            
            TravelTime  = np.inf
            CrossPoints = [np.nan,np.nan]
            CellPoints  = [np.nan,np.nan]

        if self.debugging:
            print('---> (Xsc,Ysc)={:.2f},{:.2f}; TravelTime={:.2f}'.format(self.Cell_s.cx,self.Cell_s.cy,TravelTime))


        CrossPoints[0] = np.clip(CrossPoints[0],self.Cell_n.long,(self.Cell_n.long+self.Cell_n.width))
        CrossPoints[1] = np.clip(CrossPoints[1],self.Cell_n.lat,(self.Cell_n.lat+self.Cell_n.height))

        return TravelTime, CrossPoints, CellPoints



# class SmoothedNewtonianDistance:
#     def __init__(self,Mesh,Cell_s,Cell_n,s):

#         self.Mesh   = Mesh
#         self.Cell_s = Cell_s
#         self.Cell_n = Cell_n
#         self.s      = s
#         self.fdist  = _Euclidean_distance
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

#             d1  = math.sqrt(zl**2 + y**2)
#             d2  = math.sqrt(zl**2 + (Y-y)**2)
#             C1  = s**2 - u1**2 - v1**2
#             D1  = zl*u1 + y*v1
#             X1  = math.sqrt(D1**2 + C1*(d1**2))
#             C2  = s**2 - u2**2 - v2**2
#             D2  = zr*u2 + (Y-y)*v2
#             X2  = math.sqrt(D2**2 + C2*(d2**2))

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

#             d1  = math.sqrt(zl**2 + y**2)
#             d2  = math.sqrt(zl**2 + (Y-y)**2)
#             C1  = s**2 - u1**2 - v1**2
#             D1  = zl*u1 + y*v1
#             X1  = math.sqrt(D1**2 + C1*(d1**2))
#             C2  = s**2 - u2**2 - v2**2
#             D2  = zr*u2 + (Y-y)*v2
#             X2  = math.sqrt(D2**2 + C2*(d2**2))

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

#             d1  = math.sqrt(zl**2 + y**2)
#             d2  = math.sqrt(zl**2 + (Y-y)**2)
#             C1  = s**2 - u1**2 - v1**2
#             D1  = zl*u1 + y*v1
#             X1  = math.sqrt(D1**2 + C1*(d1**2))
#             t1  = (X1-D1)/C1
#             C2  = s**2 - u2**2 - v2**2
#             D2  = zr*u2 + (Y-y)*v2
#             X2  = math.sqrt(D2**2 + C2*(d2**2))
#             t2  = (X2-D2)/C2
#             TT  = t1 + t2
#             return TT

#         u1 = np.sign(self.df_x)*self.Cell_s.getuC(); v1 = self.Cell_s.getvC()
#         u2 = np.sign(self.df_x)*self.Cell_n.getuC(); v2 = self.Cell_n.getvC()
#         λ_s = self.Cell_s.cy
#         φ_r = self.Cell_n.cy
#         x     = self.fdist((self.Cell_s.cx,self.Cell_s.cy), (self.Cell_s.cx + self.S_dx,self.Cell_s.cy))
#         a     = self.fdist((self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx + self.N_dx,self.Cell_n.cy))
#         Y     = self.fdist((self.Cell_s.cx + np.sign(self.df_x)*(abs(self.S_dx) + abs(self.N_dx)), self.Cell_s.cy), (self.Cell_s.cx + np.sign(self.df_x)*(abs(self.S_dx) + abs(self.N_dx)),self.Cell_n.cy))
#         ang   = np.arctan((self.Cell_n.cy - self.Cell_s.cy)/(self.Cell_n.cx - self.Cell_s.cx))
#         yinit = np.tan(ang)*(self.S_dx)
#         try:
#             y  = optimize.newton(_F,yinit,args=(x,a,Y,u1,v1,u2,v2,self.s,self.R,λ_s,φ_r),fprime=_dF)
#         except:
#             y  = yinit
#         TravelTime  = _T(y,x,a,Y,u1,v1,u2,v2,self.s,self.R,λ_s,φ_r)
#         CrossPoints = self.fdist((self.Cell_s.cx + self.S_dx,self.Cell_s.cy),(0.0,y),forward=False)
#         CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]

#         return TravelTime,CrossPoints,CellPoints


#     def _lat_case(self):
#         def _F(y,x,a,Y,u1,v1,u2,v2,s,R,λ,θ,ψ):
#             r1  = math.cos(λ)/math.cos(θ)
#             r2  = math.cos(ψ)/math.cos(θ)

#             d1  = math.sqrt(x**2 + (r1*y)**2)
#             d2  = math.sqrt(a**2 + (r2*(Y-y))**2)
#             C1  = s**2 - u1**2 - v1**2
#             C2  = s**2 - u2**2 - v2**2
#             D1  = x*u1 + r1*v1*Y
#             D2  = a*u2 + r2*v2*(Y-y)
#             X1  = math.sqrt(D1**2 + C1*(d1**2))
#             X2  = math.sqrt(D2**2 + C2*(d2**2)) 

#             F = ((r2**2)*X1 + (r1**2)*X2)*y - ((r1*(X1-D1)*X2*v2)/C1) + ((r2*(X2-D2)*X1*v2)/C2) - (r2**2)*Y*X1
#             return F

#         def _dF(y,x,a,Y,u1,v1,u2,v2,s,R,λ,θ,ψ):
#             r1  = math.cos(λ)/math.cos(θ)
#             r2  = math.cos(ψ)/math.cos(θ)

#             d1  = math.sqrt(x**2 + (r1*y)**2)
#             d2  = math.sqrt(a**2 + (r2*(Y-y))**2)
#             C1  = s**2 - u1**2 - v1**2
#             C2  = s**2 - u2**2 - v2**2
#             D1  = x*u1 + r1*v1*Y
#             D2  = a*u2 + r2*v2*(Y-y)
#             X1  = math.sqrt(D1**2 + C1*(d1**2))
#             X2  = math.sqrt(D2**2 + C2*(d2**2))   
            
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

#             d1  = math.sqrt(x**2 + (r1*y)**2)
#             d2  = math.sqrt(a**2 + (r2*(Y-y))**2)
#             C1  = s**2 - u1**2 - v1**2
#             C2  = s**2 - u2**2 - v2**2
#             D1  = x*u1 + r1*v1*Y
#             D2  = a*u2 + r2*v2*(Y-y)
#             X1  = math.sqrt(D1**2 + C1*(d1**2))
#             X2  = math.sqrt(D2**2 + C2*(d2**2))
#             t1  = (X1-D1)/C1
#             t2  = (X2-D2)/C2

#             TT  = t1+t2
#             return TT     

#         u1 = np.sign(self.df_y)*self.Cell_s.getvC(); v1 = self.Cell_s.getuC()
#         u2 = np.sign(self.df_y)*self.Cell_n.getvC(); v2 = self.Cell_n.getuC()

#         x  = self.fdist((self.Cell_s.cy,self.Cell_s.cx), (self.Cell_s.cy + self.S_dy,self.Cell_s.cx))
#         a  = self.fdist((self.Cell_n.cy,self.Cell_n.cx), (self.Cell_n.cy + self.N_dy,self.Cell_n.cx))
#         Y  = self.fdist((self.Cell_s.cy + np.sign(self.df_y)*(abs(self.S_dy) + abs(self.N_dy)), self.Cell_s.cx), (self.Cell_s.cy + np.sign(self.df_y)*(abs(self.S_dy) + abs(self.N_dy)),self.Cell_n.cx))
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
#         CrossPoints = self.fdist((self.Cell_s.cx + self.S_dx,self.Cell_s.cy),(0.0,y),forward=False)
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
