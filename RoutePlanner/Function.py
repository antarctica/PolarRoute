import numpy as np
import math

def sign(x):
    s = math.copysign(1, x)
    return s   

def Intersection_BoxLine(Cell_s,Cell_n,type):
    X1,Y1 = Cell_s.long+Cell_s.width/2,Cell_s.lat+Cell_s.height/2
    X2,Y2 = Cell_n.long+Cell_n.width/2,Cell_n.lat+Cell_n.height/2
    if type == 'right':
        X3,Y3 = Cell_s.long+Cell_s.width,Cell_s.lat
        X4,Y4 = Cell_s.long+Cell_s.width,Cell_s.lat+Cell_s.height
    if type == 'top':
        X3,Y3 = Cell_s.long,Cell_s.lat+Cell_s.height
        X4,Y4 = Cell_s.long+Cell_s.width,Cell_s.lat+Cell_s.height
    if type == 'left':
        X3,Y3 = Cell_s.long,Cell_s.lat
        X4,Y4 = Cell_s.long,Cell_s.lat+Cell_s.height
    if type == 'bottom':
        X3,Y3 = Cell_s.long,Cell_s.lat
        X4,Y4 = Cell_s.long+Cell_s.width,Cell_s.lat
    D  = (X1-X2)*(Y3-Y4) - (Y1-Y2)*(X3-X4)
    

    Px = ((X1*Y2 - Y1*X2)*(X3-X4) - (X1-X2)*(X3*Y4-Y3*X4))/D
    Py = ((X1*Y2-Y1*X2)*(Y3-Y4)-(Y1-Y2)*(X3*Y4-Y3*X4))/D

    #print('XY1=({},{});XY2=({},{});XY3=({},{});XY4=({},{})\n type={};D={};Px={};Py={}'.format(X1,Y1,X2,Y2,X3,Y3,X4,Y4,type,D,Px,Py))


    return Px,Py



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

    def __init__(self,scaleLongitude=None):

        self.m_per_latitude  = 111.386*1000.

        if type(scaleLongitude) != type(None):
            self.m_per_longitude = 111.321*1000*np.cos(scaleLongitude*(np.pi/180))
        else:
            self.m_per_longitude = (111.321*1000.)

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
# class NewtonianDistance:
#     def __init__(self,Cell_S=None,Cell_N=None,Cell_S_Speed=None,Cell_N_Speed=None,unit_shipspeed='km/hr',unit_time='days',zerocurrents=False,debugging=False,maxiter=500,optimizer_tol=1e-7):
#         # Cell information
#         self.Cell_s         = Cell_S
#         self.Cell_n         = Cell_N

#         # Inside the code the base units are m/s. Changing the units of the inputs to match
#         self.unit_shipspeed = unit_shipspeed
#         self.unit_time      = unit_time

#         self.s1             = self._unit_speed(Cell_S_Speed)
#         self.s2             = self._unit_speed(Cell_N_Speed)
#         self.fdist          = _Euclidean_distance(scaleLongitude=(self.Cell_s.lat+self.Cell_s.height))

#         if zerocurrents:
#             self.zx = 0.0
#         else:
#             self.zx = 1.0

#         # Optimisation Information
#         self.maxiter       = maxiter
#         self.optimizer_tol = optimizer_tol

#         # For Debugging purposes 
#         self.debugging     = debugging


#     def NewtonOptimisation(self,f,df,x,a,Y,u1,v1,u2,v2,s1,s2):
#             y0 = (Y*x)/(x+a)
#             if self.debugging:
#                     print('---Initial y={:.2f}'.format(y0))
#             if self.maxiter > 0:
#                 for iter in range(self.maxiter):
#                     F  = f(y0,x,a,Y,u1,v1,u2,v2,s1,s2)
#                     dF = df(y0,x,a,Y,u1,v1,u2,v2,s1,s2)
#                     if self.debugging:
#                         print('---Iteration {}: y={:.2f}; F={:.5f}; dF={:.2f}'.format(iter,y0,F,dF))
#                     y0  = y0 - (F/dF)
#                     if F < self.optimizer_tol:
#                         break
#             return y0

#     def _unit_speed(self,Val):
#         if type(Val) != type(None):
#             if self.unit_shipspeed == 'km/hr':
#                 Val = Val*(1000/(60*60))
#             if self.unit_shipspeed == 'knots':
#                 Val = (Val*0.51)
#             return Val
#         else:
#             return None

#     def _unit_time(self,Val):
#         if self.unit_time == 'days':
#             Val = Val/(60*60*24)
#         elif self.unit_time == 'hr':
#             Val = Val/(60*60)
#         elif self.unit_time == 'min':
#             Val = Val/(60)
#         elif self.unit_time == 's':
#             Val = Val

#         return Val
        
#     def WaypointCorrection(self,Wp,Cp,s):
#         '''
        
        
#         '''
#         dS  = (self.Cell_s.width/2,self.Cell_s.height/2)
#         S   = (self.Cell_s.long+dS[0],self.Cell_s.lat+dS[1])
#         uS  = (self.Cell_s.getuC(),self.Cell_s.getvC())

#         x   = sign(Cp[0]-Wp[0])*self.fdist.value(Wp,(Wp[0]+(Cp[0]-Wp[0]),Wp[1]))
#         y   = sign(Cp[1]-Wp[1])*self.fdist.value(Wp,(Wp[0],Wp[1]+(Cp[0]-Wp[0])))

#         C1  = s**2 - uS[0]**2 - uS[1]**2
#         D1  = x*uS[0] + y*uS[1]
#         X1  = np.sqrt(D1**2 + C1*(x**2 + y**2))
#         return self._unit_time((X1-D1)/C1)


#     def value(self):
#         def _F(y,x,a,Y,u1,v1,u2,v2,s1,s2):
#             C1 = s1**2 - u1**2 - v1**2
#             C2 = s2**2 - u2**2 - v2**2
#             D1 = x*u1 + y*v1
#             D2 = a*u2 + (Y-y)*v2
#             X1 = np.sqrt(D1**2 + C1*(x**2 + y**2))
#             X2 = np.sqrt(D2**2 + C2*(a**2 + (Y-y)**2))
#             F  = X2*(y-((v1*(X1-D1))/C1)) + X1*(y-Y+((v2*(X2-D2))/C2)) 
#             return F

#         def _dF(y,x,a,Y,u1,v1,u2,v2,s1,s2):
#             C1  = s1**2 - u1**2 - v1**2
#             C2  = s2**2 - u2**2 - v2**2
#             D1  = x*u1 + y*v1
#             D2  = a*u2 + (Y-y)*v2
#             X1  = np.sqrt(D1**2 + C1*(x**2 + y**2))
#             X2  = np.sqrt(D2**2 + C2*(a**2 + (Y-y)**2))
#             dD1 = v1
#             dD2 = -v2
#             dX1 = (D1*v1 + C1*y)/X1
#             dX2 = (-D2*v2 - C2*(Y-y))/X2
#             dF  = (X1+X2) + y*(dX1 + dX2) - (v1/C1)*(dX2*(X1-D1) + X2*(dX1-dD1)) + (v2/C2)*(dX1*(X2-D2)+X1*(dX2-dD2)) - Y*dX1
#             return dF

#         def _T(y,x,a,Y,u1,v1,u2,v2,s1,s2):
#             C1 = s1**2 - u1**2 - v1**2
#             C2 = s2**2 - u2**2 - v2**2
#             D1 = x*u1 + y*v1
#             D2 = a*u2 + (Y-y)*v2
#             X1 = np.sqrt(D1**2 + C1*(x**2 + y**2))
#             X2 = np.sqrt(D2**2 + C2*(a**2 + (Y-y)**2))
#             t1 = (X1-D1)/C1
#             t2 = (X2-D2)/C2
#             return np.array([t1,t2])


#         def _positive_longitude(self):
#             '''
#                 INCLUDE
#             '''
#             u1          = self.Cell_s.getuC()*self.zx; v1 = self.Cell_s.getvC()*self.zx
#             u2          = self.Cell_n.getuC()*self.zx; v2 = self.Cell_n.getvC()*self.zx
#             x           = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy), (self.Cell_s.cx + self.Cell_s.dcx,self.Cell_s.cy))
#             a           = self.fdist.value((self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx - self.Cell_n.dcx,self.Cell_n.cy))
#             Y           = sign(self.Cell_n.cy-self.Cell_s.cy)*self.fdist.value((self.Cell_s.cx+(abs(self.Cell_s.dcx)+abs(self.Cell_n.dcx)),self.Cell_s.cy),\
#                                      (self.Cell_s.cx+(abs(self.Cell_s.dcx)+abs(self.Cell_n.dcx)),self.Cell_n.cy))
#             y = self.NewtonOptimisation(_F,_dF,x,a,Y,u1,v1,u2,v2,self.s1,self.s2)
#             if self.debugging:
#                 print('Positve Long: y={:.2f};x={:.2f};a={:.2f};Y={:.2f};u1={:.5f};v1={:.5f};u2={:.5f};v2={:.5f};s1={:.2f};s2={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
#             TravelTime  = self._unit_time(_T(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
#             CrossPoints = self.fdist.value((self.Cell_s.cx + self.Cell_s.dcx,self.Cell_s.cy),(0.0,y),forward=False)
#             CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
#             if self.debugging:
#                 print('------ TravelTime={:.2f};CrossPoints=[{:.2f},{:.2f}];CellPoints=[{:.2f},{:.5f}]'.format(TravelTime,CrossPoints[0],CrossPoints[1],CellPoints[0],CellPoints[1]))    
#             return TravelTime,CrossPoints,CellPoints

#         def _negative_longitude(self):
#             '''
#                 INCLUDE
#             '''
#             u1          = -self.Cell_s.getuC()*self.zx; v1 = self.Cell_s.getvC()*self.zx
#             u2          = -self.Cell_n.getuC()*self.zx; v2 = self.Cell_n.getvC()*self.zx
#             x           = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx-self.Cell_s.dcx,self.Cell_s.cy))
#             a           = self.fdist.value((self.Cell_n.cx,self.Cell_n.cy),(self.Cell_n.cx+self.Cell_n.dcx,self.Cell_n.cy))
#             Y           = sign(self.Cell_n.cy-self.Cell_s.cy)*\
#                                self.fdist.value((self.Cell_s.cx-(abs(self.Cell_s.dcx)+abs(self.Cell_n.dcx)),self.Cell_s.cy),\
#                                           (self.Cell_s.cx-(abs(self.Cell_s.dcx)+abs(self.Cell_n.dcx)),self.Cell_n.cy))
#             y = self.NewtonOptimisation(_F,_dF,x,a,Y,u1,v1,u2,v2,self.s1,self.s2)
#             if self.debugging:
#                 print('Negative Long: y={:.2f};x={:.2f};a={:.2f};Y={:.2f};u1={:.5f};v1={:.5f};u2={:.5f};v2={:.5f};s1={:.2f};s2={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
#             TravelTime  = self._unit_time(_T(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
#             CrossPoints = self.fdist.value((self.Cell_s.cx - self.Cell_s.dcx,self.Cell_s.cy),(0.0,y),forward=False)
#             CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
#             if self.debugging:
#                 print('------ TravelTime={:.2f};CrossPoints=[{:.2f},{:.2f}];CellPoints=[{:.2f},{:.5f}]'.format(TravelTime,CrossPoints[0],CrossPoints[1],CellPoints[0],CellPoints[1]))    
#             return TravelTime,CrossPoints,CellPoints

#         def _positive_latitude(self):
#             '''
#                 Case -4
#             '''

#             u1          = -self.Cell_s.getvC()*self.zx; v1 = self.Cell_s.getuC()*self.zx
#             u2          = -self.Cell_n.getvC()*self.zx; v2 = self.Cell_n.getuC()*self.zx
#             x           = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx,self.Cell_s.cy + self.Cell_s.dcy))
#             a           = self.fdist.value((self.Cell_n.cx,self.Cell_n.cy),(self.Cell_n.cx,self.Cell_n.cy - self.Cell_n.dcy))
#             Y           = sign(self.Cell_n.cx-self.Cell_s.cx)*\
#                           self.fdist.value((self.Cell_s.cx, self.Cell_s.cy + (abs(self.Cell_s.dcy) + abs(self.Cell_n.dcy))),\
#                                      (self.Cell_n.cx, self.Cell_s.cy + (abs(self.Cell_s.dcy) + abs(self.Cell_n.dcy))))
#             y = self.NewtonOptimisation(_F,_dF,x,a,Y,u1,v1,u2,v2,self.s1,self.s2)
#             if self.debugging:
#                 print('Postive Lat: y={:.2f};x={:.2f};a={:.2f};Y={:.2f};u1={:.5f};v1={:.5f};u2={:.5f};v2={:.5f};s1={:.2f};s2={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
#             TravelTime  = self._unit_time(_T(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
#             CrossPoints = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy+self.Cell_s.dcy),(y,0.0),forward=False)
#             CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
#             if self.debugging:
#                 print('------ TravelTime={:.2f};CrossPoints=[{:.2f},{:.2f}];CellPoints=[{:.2f},{:.5f}]'.format(TravelTime,CrossPoints[0],CrossPoints[1],CellPoints[0],CellPoints[1]))    
#             return TravelTime,CrossPoints,CellPoints


#         def _negative_latitude(self):
#             '''
#                 Case 4
#             '''

#             u1          = -self.Cell_s.getvC()*self.zx; v1 = -self.Cell_s.getuC()*self.zx
#             u2          = -self.Cell_n.getvC()*self.zx; v2 = -self.Cell_n.getuC()*self.zx
#             x           = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx,self.Cell_s.cy - self.Cell_s.dcy))
#             a           = self.fdist.value((self.Cell_n.cx,self.Cell_n.cy),(self.Cell_n.cx,self.Cell_n.cy + self.Cell_n.dcy))
#             Y           = sign(self.Cell_n.cx-self.Cell_s.cx)*\
#                           self.fdist.value((self.Cell_s.cx, self.Cell_s.cy - (abs(self.Cell_s.dcy) + abs(self.Cell_n.dcy))),\
#                                      (self.Cell_n.cx, self.Cell_s.cy - (abs(self.Cell_s.dcy) + abs(self.Cell_n.dcy))))
#             y = self.NewtonOptimisation(_F,_dF,x,a,Y,u1,v1,u2,v2,self.s1,self.s2)
#             if self.debugging:
#                 print('Negative Lat: y={:.2f};x={:.2f};a={:.2f};Y={:.2f};u1={:.5f};v1={:.5f};u2={:.5f};v2={:.5f};s1={:.2f};s2={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
#             TravelTime  = self._unit_time(_T(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
#             CrossPoints = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy-self.Cell_s.dcy),(y,0.0),forward=False)
#             CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
#             if self.debugging:
#                 print('------ TravelTime={:.2f};CrossPoints=[{:.2f},{:.2f}];CellPoints=[{:.2f},{:.5f}]'.format(TravelTime,CrossPoints[0],CrossPoints[1],CellPoints[0],CellPoints[1]))    
#             return TravelTime,CrossPoints,CellPoints

#         def _top_right_corner(self):
#             u1 = self.Cell_s.getuC()*self.zx; v1 = self.Cell_s.getvC()*self.zx
#             u2 = self.Cell_n.getuC()*self.zx; v2 = self.Cell_n.getvC()*self.zx
#             x  = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx + self.Cell_s.dcx,self.Cell_s.cy))
#             y  = self.fdist.value((self.Cell_s.cx+self.Cell_s.dcx,self.Cell_s.cy),(self.Cell_s.cx+self.Cell_s.dcx,self.Cell_s.cy+self.Cell_s.dcy))
#             a  = -self.fdist.value((self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx - self.Cell_n.dcx,self.Cell_n.cy))
#             Y  = self.fdist.value((self.Cell_s.cx+(abs(self.Cell_s.dcx) + abs(self.Cell_n.dcx)),self.Cell_s.cy),\
#                             (self.Cell_s.cx+(abs(self.Cell_s.dcx) + abs(self.Cell_n.dcx)),self.Cell_n.cy))
#             if self.debugging:
#                 print('Top Right: y={:.2f};x={:.2f};a={:.2f};Y={:.2f};u1={:.5f};v1={:.5f};u2={:.5f};v2={:.5f};s1={:.2f};s2={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
#             TravelTime  = self._unit_time(_T(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
#             CrossPoints = [self.Cell_s.cx+self.Cell_s.dcx,self.Cell_s.cy+self.Cell_s.dcy]
#             CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
#             if self.debugging:
#                 print('------ TravelTime={:.2f};CrossPoints=[{:.2f},{:.2f}];CellPoints=[{:.2f},{:.5f}]'.format(TravelTime,CrossPoints[0],CrossPoints[1],CellPoints[0],CellPoints[1]))    
#             return TravelTime,CrossPoints,CellPoints

#         def _bottom_right_corner(self):
#             u1 = self.Cell_s.getuC()*self.zx; v1 = self.Cell_s.getvC()*self.zx
#             u2 = self.Cell_n.getuC()*self.zx; v2 = self.Cell_n.getvC()*self.zx
#             x  = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx + self.Cell_s.dcx,self.Cell_s.cy))
#             y  = -self.fdist.value((self.Cell_s.cx+self.Cell_s.dcx,self.Cell_s.cy),(self.Cell_s.cx+self.Cell_s.dcx,self.Cell_s.cy-self.Cell_s.dcy))
#             a  = -self.fdist.value((self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx - self.Cell_n.dcx,self.Cell_n.cy))
#             Y  = -self.fdist.value((self.Cell_s.cx+(abs(self.Cell_s.dcx) + abs(self.Cell_n.dcx)),self.Cell_s.cy),\
#                             (self.Cell_s.cx+(abs(self.Cell_s.dcx) + abs(self.Cell_n.dcx)),self.Cell_n.cy))
#             if self.debugging:
#                 print('Bottom Right: y={:.2f};x={:.2f};a={:.2f};Y={:.2f};u1={:.5f};v1={:.5f};u2={:.5f};v2={:.5f};s1={:.2f};s2={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
#             TravelTime  = self._unit_time(_T(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
#             CrossPoints = [self.Cell_s.cx+self.Cell_s.dcx,self.Cell_s.cy-self.Cell_s.dcy]
#             CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
#             if self.debugging:
#                 print('------ TravelTime={:.2f};CrossPoints=[{:.2f},{:.2f}];CellPoints=[{:.2f},{:.5f}]'.format(TravelTime,CrossPoints[0],CrossPoints[1],CellPoints[0],CellPoints[1]))    
#             return TravelTime,CrossPoints,CellPoints

#         def _bottom_left_corner(self):
#             u1 = -self.Cell_s.getuC()*self.zx; v1 = self.Cell_s.getvC()*self.zx
#             u2 = -self.Cell_n.getuC()*self.zx; v2 = self.Cell_n.getvC()*self.zx
#             x  = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx - self.Cell_s.dcx,self.Cell_s.cy))
#             y  = -self.fdist.value((self.Cell_s.cx-self.Cell_s.dcx,self.Cell_s.cy),(self.Cell_s.cx-self.Cell_s.dcx,self.Cell_s.cy-self.Cell_s.dcy))
#             a  = -self.fdist.value((self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx + self.Cell_n.dcx,self.Cell_n.cy))
#             Y  = -self.fdist.value((self.Cell_s.cx-(abs(self.Cell_s.dcx) + abs(self.Cell_n.dcx)),self.Cell_s.cy),\
#                              (self.Cell_s.cx-(abs(self.Cell_s.dcx) + abs(self.Cell_n.dcx)),self.Cell_n.cy))
#             if self.debugging:
#                 print('Bottom Left: y={:.2f};x={:.2f};a={:.2f};Y={:.2f};u1={:.5f};v1={:.5f};u2={:.5f};v2={:.5f};s1={:.2f};s2={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))    
#             TravelTime  = self._unit_time(_T(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
#             CrossPoints = [self.Cell_s.cx-self.Cell_s.dcx,self.Cell_s.cy-self.Cell_s.dcy]
#             CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
#             if self.debugging:
#                 print('------ TravelTime={:.2f};CrossPoints=[{:.2f},{:.2f}];CellPoints=[{:.2f},{:.5f}]'.format(TravelTime,CrossPoints[0],CrossPoints[1],CellPoints[0],CellPoints[1]))    
#             return TravelTime,CrossPoints,CellPoints

#         def _top_left_corner(self):
#             u1 = -self.Cell_s.getuC()*self.zx; v1 = self.Cell_s.getvC()*self.zx
#             u2 = -self.Cell_n.getuC()*self.zx; v2 = self.Cell_n.getvC()*self.zx
#             x  = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx - self.Cell_s.dcx,self.Cell_s.cy))
#             y  = self.fdist.value((self.Cell_s.cx-self.Cell_s.dcx,self.Cell_s.cy),(self.Cell_s.cx-self.Cell_s.dcx,self.Cell_s.cy+self.Cell_s.dcy))
#             a  = -self.fdist.value((self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx + self.Cell_n.dcx,self.Cell_n.cy))
#             Y  = self.fdist.value((self.Cell_s.cx-(abs(self.Cell_s.dcx) + abs(self.Cell_n.dcx)),self.Cell_s.cy),\
#                             (self.Cell_s.cx-(abs(self.Cell_s.dcx) + abs(self.Cell_n.dcx)),self.Cell_n.cy))
#             if self.debugging:
#                 print('Top Left: y={:.2f};x={:.2f};a={:.2f};Y={:.2f};u1={:.5f};v1={:.5f};u2={:.5f};v2={:.5f};s1={:.2f};s2={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
#             TravelTime  = self._unit_time(_T(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
#             CrossPoints = [self.Cell_s.cx-self.Cell_s.dcx,self.Cell_s.cy+self.Cell_s.dcy]
#             CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
#             if self.debugging:
#                 print('------ TravelTime={:.2f};CrossPoints=[{:.2f},{:.2f}];CellPoints=[{:.2f},{:.5f}]'.format(TravelTime,CrossPoints[0],CrossPoints[1],CellPoints[0],CellPoints[1]))   
#             return TravelTime,CrossPoints,CellPoints


#         # Determining the distance between the cell centres to 
#         #be used in defining the case
#         self.df_x = (self.Cell_n.long+self.Cell_n.width/2) -  (self.Cell_s.long+self.Cell_s.width/2)
#         self.df_y = (self.Cell_n.lat+self.Cell_n.height/2) -  (self.Cell_s.lat+self.Cell_s.height/2)

#         if self.debugging:
#             print('============================================')

#         # ======= Determining the Newton Value dependent on case
#         # Case 2 - Positive Longitude
#         if (self.df_x > self.Cell_s.width/2) and (abs(self.df_y) <= (self.Cell_s.height/2)):
#             TravelTime,CrossPoints,CellPoints = _positive_longitude(self)
#         # Case -2 - Negative Longitude
#         elif (self.df_x < -(self.Cell_s.width/2)) and (abs(self.df_y) <= (self.Cell_s.height/2)):
#             TravelTime,CrossPoints,CellPoints = _negative_longitude(self)
#         # Case -4 - Positive Latitude
#         elif (self.df_y > (self.Cell_s.height/2)) and (abs(self.df_x) <= (self.Cell_s.width/2)):
#             TravelTime,CrossPoints,CellPoints = _positive_latitude(self)
#         # Case 4 - Negative Latitude
#         elif (self.df_y < -(self.Cell_s.height/2)) and (abs(self.df_x) <= (self.Cell_s.width/2)):
#             TravelTime,CrossPoints,CellPoints = _negative_latitude(self)
#         # Case 1 - Top Right Corner 
#         elif (self.df_y > (self.Cell_s.height/2)) and (self.df_x > (self.Cell_s.width/2)):
#             TravelTime,CrossPoints,CellPoints = _top_right_corner(self)    
#         # Case 3 - Bottom Right Corner 
#         elif (self.df_y < -(self.Cell_s.height/2)) and (self.df_x > (self.Cell_s.width/2)):
#             TravelTime,CrossPoints,CellPoints = _bottom_right_corner(self)  
#         # Case -1 - Bottom Left Corner 
#         elif (self.df_y < -(self.Cell_s.height/2)) and (self.df_x < -(self.Cell_s.width/2)):
#             TravelTime,CrossPoints,CellPoints = _bottom_left_corner(self)
#         # Case -3 - Top Left Corner 
#         elif (self.df_y > (self.Cell_s.height/2)) and (self.df_x < -(self.Cell_s.width/2)):
#             TravelTime,CrossPoints,CellPoints = _top_left_corner(self)
#         else:
#             print('---> Issue with cell (Xsc,Ysc)={:.2f};{:.2f}; (dx,dy)={:.2f},{:.2f}; (diffX,diffY)={:.2f},{:.2f}'.format(self.Cell_s.cx,self.Cell_s.cy,self.Cell_s.width/2,self.Cell_s.height/2,self.df_x,self.df_y))
            
#             TravelTime  = np.inf
#             CrossPoints = [np.nan,np.nan]
#             CellPoints  = [np.nan,np.nan]

#         CrossPoints[0] = np.clip(CrossPoints[0],self.Cell_n.long,(self.Cell_n.long+self.Cell_n.width))
#         CrossPoints[1] = np.clip(CrossPoints[1],self.Cell_n.lat,(self.Cell_n.lat+self.Cell_n.height))

#         return TravelTime, CrossPoints, CellPoints


class NewtonianDistance:
    def __init__(self,Mesh,Cell_S=None,Cell_N=None,Cell_S_Speed=None,Cell_N_Speed=None,unit_shipspeed='km/hr',unit_time='days',zerocurrents=False,debugging=False,maxiter=500,optimizer_tol=1e-7):
        # Cell information
        self.Cell_s         = Cell_S
        self.Cell_n         = Cell_N

        self.Mesh           = Mesh

        # Inside the code the base units are m/s. Changing the units of the inputs to match
        self.unit_shipspeed = unit_shipspeed
        self.unit_time      = unit_time

        self.s1             = self._unit_speed(Cell_S_Speed)
        self.s2             = self._unit_speed(Cell_N_Speed)
        self.fdist          = _Euclidean_distance(scaleLongitude=(self.Cell_s.lat+self.Cell_s.height))

        if zerocurrents:
            self.zx = 0.0
        else:
            self.zx = 1.0

        # Optimisation Information
        self.maxiter       = maxiter
        self.optimizer_tol = optimizer_tol

        # For Debugging purposes 
        self.debugging     = debugging


    def NewtonOptimisation(self,f,df,x,a,Y,u1,v1,u2,v2,s1,s2):
            y0 = (Y*x)/(x+a)
            if self.debugging:
                    print('---Initial y={:.2f}'.format(y0))
            if self.maxiter > 0:
                for iter in range(self.maxiter):
                    F  = f(y0,x,a,Y,u1,v1,u2,v2,s1,s2)
                    dF = df(y0,x,a,Y,u1,v1,u2,v2,s1,s2)
                    if self.debugging:
                        print('---Iteration {}: y={:.2f}; F={:.5f}; dF={:.2f}'.format(iter,y0,F,dF))
                    y0  = y0 - (F/dF)
                    if F < self.optimizer_tol:
                        break
            return y0

    def _unit_speed(self,Val):
        if type(Val) != type(None):
            if self.unit_shipspeed == 'km/hr':
                Val = Val*(1000/(60*60))
            if self.unit_shipspeed == 'knots':
                Val = (Val*0.51)
            return Val
        else:
            return None

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
        
    def WaypointCorrection(self,Wp,Cp,s):
        '''
        
        
        '''
        dS  = (self.Cell_s.width/2,self.Cell_s.height/2)
        S   = (self.Cell_s.long+dS[0],self.Cell_s.lat+dS[1])
        uS  = (self.Cell_s.getuC(),self.Cell_s.getvC())

        x   = sign(Cp[0]-Wp[0])*self.fdist.value(Wp,(Wp[0]+(Cp[0]-Wp[0]),Wp[1]))
        y   = sign(Cp[1]-Wp[1])*self.fdist.value(Wp,(Wp[0],Wp[1]+(Cp[0]-Wp[0])))

        C1  = s**2 - uS[0]**2 - uS[1]**2
        D1  = x*uS[0] + y*uS[1]
        X1  = np.sqrt(D1**2 + C1*(x**2 + y**2))
        return self._unit_time((X1-D1)/C1)


    def value(self):
        def _F(y,x,a,Y,u1,v1,u2,v2,s1,s2):
            C1 = s1**2 - u1**2 - v1**2
            C2 = s2**2 - u2**2 - v2**2
            D1 = x*u1 + y*v1
            D2 = a*u2 + (Y-y)*v2
            X1 = np.sqrt(D1**2 + C1*(x**2 + y**2))
            X2 = np.sqrt(D2**2 + C2*(a**2 + (Y-y)**2))
            F  = X2*(y-((v1*(X1-D1))/C1)) + X1*(y-Y+((v2*(X2-D2))/C2)) 
            return F

        def _dF(y,x,a,Y,u1,v1,u2,v2,s1,s2):
            C1  = s1**2 - u1**2 - v1**2
            C2  = s2**2 - u2**2 - v2**2
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

        def _T(y,x,a,Y,u1,v1,u2,v2,s1,s2):
            C1 = s1**2 - u1**2 - v1**2
            C2 = s2**2 - u2**2 - v2**2
            D1 = x*u1 + y*v1
            D2 = a*u2 + (Y-y)*v2
            X1 = np.sqrt(D1**2 + C1*(x**2 + y**2))
            X2 = np.sqrt(D2**2 + C2*(a**2 + (Y-y)**2))
            t1 = (X1-D1)/C1
            t2 = (X2-D2)/C2
            return np.array([t1,t2])


        def _positive_longitude(self):
            '''
                INCLUDE
            '''
            u1          = self.Cell_s.getuC()*self.zx; v1 = self.Cell_s.getvC()*self.zx
            u2          = self.Cell_n.getuC()*self.zx; v2 = self.Cell_n.getvC()*self.zx
            x           = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy), (self.Cell_s.cx + self.Cell_s.dcx,self.Cell_s.cy))
            a           = self.fdist.value((self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx - self.Cell_n.dcx,self.Cell_n.cy))
            Y           = sign(self.Cell_n.cy-self.Cell_s.cy)*self.fdist.value((self.Cell_s.cx+(abs(self.Cell_s.dcx)+abs(self.Cell_n.dcx)),self.Cell_s.cy),\
                                     (self.Cell_s.cx+(abs(self.Cell_s.dcx)+abs(self.Cell_n.dcx)),self.Cell_n.cy))
            y = self.NewtonOptimisation(_F,_dF,x,a,Y,u1,v1,u2,v2,self.s1,self.s2)
            if self.debugging:
                print('Positve Long: y={:.2f};x={:.2f};a={:.2f};Y={:.2f};u1={:.5f};v1={:.5f};u2={:.5f};v2={:.5f};s1={:.2f};s2={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
            TravelTime  = self._unit_time(_T(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
            CrossPoints = self.fdist.value((self.Cell_s.cx + self.Cell_s.dcx,self.Cell_s.cy),(0.0,y),forward=False)
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
            x           = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx-self.Cell_s.dcx,self.Cell_s.cy))
            a           = self.fdist.value((self.Cell_n.cx,self.Cell_n.cy),(self.Cell_n.cx+self.Cell_n.dcx,self.Cell_n.cy))
            Y           = sign(self.Cell_n.cy-self.Cell_s.cy)*\
                               self.fdist.value((self.Cell_s.cx-(abs(self.Cell_s.dcx)+abs(self.Cell_n.dcx)),self.Cell_s.cy),\
                                          (self.Cell_s.cx-(abs(self.Cell_s.dcx)+abs(self.Cell_n.dcx)),self.Cell_n.cy))
            y = self.NewtonOptimisation(_F,_dF,x,a,Y,u1,v1,u2,v2,self.s1,self.s2)
            if self.debugging:
                print('Negative Long: y={:.2f};x={:.2f};a={:.2f};Y={:.2f};u1={:.5f};v1={:.5f};u2={:.5f};v2={:.5f};s1={:.2f};s2={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
            TravelTime  = self._unit_time(_T(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
            CrossPoints = self.fdist.value((self.Cell_s.cx - self.Cell_s.dcx,self.Cell_s.cy),(0.0,y),forward=False)
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
            x           = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx,self.Cell_s.cy + self.Cell_s.dcy))
            a           = self.fdist.value((self.Cell_n.cx,self.Cell_n.cy),(self.Cell_n.cx,self.Cell_n.cy - self.Cell_n.dcy))
            Y           = sign(self.Cell_n.cx-self.Cell_s.cx)*\
                          self.fdist.value((self.Cell_s.cx, self.Cell_s.cy + (abs(self.Cell_s.dcy) + abs(self.Cell_n.dcy))),\
                                     (self.Cell_n.cx, self.Cell_s.cy + (abs(self.Cell_s.dcy) + abs(self.Cell_n.dcy))))
            y = self.NewtonOptimisation(_F,_dF,x,a,Y,u1,v1,u2,v2,self.s1,self.s2)
            if self.debugging:
                print('Postive Lat: y={:.2f};x={:.2f};a={:.2f};Y={:.2f};u1={:.5f};v1={:.5f};u2={:.5f};v2={:.5f};s1={:.2f};s2={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
            TravelTime  = self._unit_time(_T(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
            CrossPoints = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy+self.Cell_s.dcy),(y,0.0),forward=False)
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
            x           = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx,self.Cell_s.cy - self.Cell_s.dcy))
            a           = self.fdist.value((self.Cell_n.cx,self.Cell_n.cy),(self.Cell_n.cx,self.Cell_n.cy + self.Cell_n.dcy))
            Y           = sign(self.Cell_n.cx-self.Cell_s.cx)*\
                          self.fdist.value((self.Cell_s.cx, self.Cell_s.cy - (abs(self.Cell_s.dcy) + abs(self.Cell_n.dcy))),\
                                     (self.Cell_n.cx, self.Cell_s.cy - (abs(self.Cell_s.dcy) + abs(self.Cell_n.dcy))))
            y = self.NewtonOptimisation(_F,_dF,x,a,Y,u1,v1,u2,v2,self.s1,self.s2)
            if self.debugging:
                print('Negative Lat: y={:.2f};x={:.2f};a={:.2f};Y={:.2f};u1={:.5f};v1={:.5f};u2={:.5f};v2={:.5f};s1={:.2f};s2={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
            TravelTime  = self._unit_time(_T(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
            CrossPoints = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy-self.Cell_s.dcy),(y,0.0),forward=False)
            CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
            if self.debugging:
                print('------ TravelTime={:.2f};CrossPoints=[{:.2f},{:.2f}];CellPoints=[{:.2f},{:.5f}]'.format(TravelTime,CrossPoints[0],CrossPoints[1],CellPoints[0],CellPoints[1]))    
            return TravelTime,CrossPoints,CellPoints

        def _top_right_corner(self):
            u1 = self.Cell_s.getuC()*self.zx; v1 = self.Cell_s.getvC()*self.zx
            u2 = self.Cell_n.getuC()*self.zx; v2 = self.Cell_n.getvC()*self.zx
            x  = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx + self.Cell_s.dcx,self.Cell_s.cy))
            y  = self.fdist.value((self.Cell_s.cx+self.Cell_s.dcx,self.Cell_s.cy),(self.Cell_s.cx+self.Cell_s.dcx,self.Cell_s.cy+self.Cell_s.dcy))
            a  = -self.fdist.value((self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx - self.Cell_n.dcx,self.Cell_n.cy))
            Y  = self.fdist.value((self.Cell_s.cx+(abs(self.Cell_s.dcx) + abs(self.Cell_n.dcx)),self.Cell_s.cy),\
                            (self.Cell_s.cx+(abs(self.Cell_s.dcx) + abs(self.Cell_n.dcx)),self.Cell_n.cy))
            if self.debugging:
                print('Top Right: y={:.2f};x={:.2f};a={:.2f};Y={:.2f};u1={:.5f};v1={:.5f};u2={:.5f};v2={:.5f};s1={:.2f};s2={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
            TravelTime  = self._unit_time(_T(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
            CrossPoints = [self.Cell_s.cx+self.Cell_s.dcx,self.Cell_s.cy+self.Cell_s.dcy]
            CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
            if self.debugging:
                print('------ TravelTime={:.2f};CrossPoints=[{:.2f},{:.2f}];CellPoints=[{:.2f},{:.5f}]'.format(TravelTime,CrossPoints[0],CrossPoints[1],CellPoints[0],CellPoints[1]))    
            return TravelTime,CrossPoints,CellPoints

        def _bottom_right_corner(self):
            u1 = self.Cell_s.getuC()*self.zx; v1 = self.Cell_s.getvC()*self.zx
            u2 = self.Cell_n.getuC()*self.zx; v2 = self.Cell_n.getvC()*self.zx
            x  = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx + self.Cell_s.dcx,self.Cell_s.cy))
            y  = -self.fdist.value((self.Cell_s.cx+self.Cell_s.dcx,self.Cell_s.cy),(self.Cell_s.cx+self.Cell_s.dcx,self.Cell_s.cy-self.Cell_s.dcy))
            a  = -self.fdist.value((self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx - self.Cell_n.dcx,self.Cell_n.cy))
            Y  = -self.fdist.value((self.Cell_s.cx+(abs(self.Cell_s.dcx) + abs(self.Cell_n.dcx)),self.Cell_s.cy),\
                            (self.Cell_s.cx+(abs(self.Cell_s.dcx) + abs(self.Cell_n.dcx)),self.Cell_n.cy))
            if self.debugging:
                print('Bottom Right: y={:.2f};x={:.2f};a={:.2f};Y={:.2f};u1={:.5f};v1={:.5f};u2={:.5f};v2={:.5f};s1={:.2f};s2={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
            TravelTime  = self._unit_time(_T(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
            CrossPoints = [self.Cell_s.cx+self.Cell_s.dcx,self.Cell_s.cy-self.Cell_s.dcy]
            CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
            if self.debugging:
                print('------ TravelTime={:.2f};CrossPoints=[{:.2f},{:.2f}];CellPoints=[{:.2f},{:.5f}]'.format(TravelTime,CrossPoints[0],CrossPoints[1],CellPoints[0],CellPoints[1]))    
            return TravelTime,CrossPoints,CellPoints

        def _bottom_left_corner(self):
            u1 = -self.Cell_s.getuC()*self.zx; v1 = self.Cell_s.getvC()*self.zx
            u2 = -self.Cell_n.getuC()*self.zx; v2 = self.Cell_n.getvC()*self.zx
            x  = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx - self.Cell_s.dcx,self.Cell_s.cy))
            y  = -self.fdist.value((self.Cell_s.cx-self.Cell_s.dcx,self.Cell_s.cy),(self.Cell_s.cx-self.Cell_s.dcx,self.Cell_s.cy-self.Cell_s.dcy))
            a  = -self.fdist.value((self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx + self.Cell_n.dcx,self.Cell_n.cy))
            Y  = -self.fdist.value((self.Cell_s.cx-(abs(self.Cell_s.dcx) + abs(self.Cell_n.dcx)),self.Cell_s.cy),\
                             (self.Cell_s.cx-(abs(self.Cell_s.dcx) + abs(self.Cell_n.dcx)),self.Cell_n.cy))
            if self.debugging:
                print('Bottom Left: y={:.2f};x={:.2f};a={:.2f};Y={:.2f};u1={:.5f};v1={:.5f};u2={:.5f};v2={:.5f};s1={:.2f};s2={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))    
            TravelTime  = self._unit_time(_T(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
            CrossPoints = [self.Cell_s.cx-self.Cell_s.dcx,self.Cell_s.cy-self.Cell_s.dcy]
            CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
            if self.debugging:
                print('------ TravelTime={:.2f};CrossPoints=[{:.2f},{:.2f}];CellPoints=[{:.2f},{:.5f}]'.format(TravelTime,CrossPoints[0],CrossPoints[1],CellPoints[0],CellPoints[1]))    
            return TravelTime,CrossPoints,CellPoints

        def _top_left_corner(self):
            u1 = -self.Cell_s.getuC()*self.zx; v1 = self.Cell_s.getvC()*self.zx
            u2 = -self.Cell_n.getuC()*self.zx; v2 = self.Cell_n.getvC()*self.zx
            x  = self.fdist.value((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx - self.Cell_s.dcx,self.Cell_s.cy))
            y  = self.fdist.value((self.Cell_s.cx-self.Cell_s.dcx,self.Cell_s.cy),(self.Cell_s.cx-self.Cell_s.dcx,self.Cell_s.cy+self.Cell_s.dcy))
            a  = -self.fdist.value((self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx + self.Cell_n.dcx,self.Cell_n.cy))
            Y  = self.fdist.value((self.Cell_s.cx-(abs(self.Cell_s.dcx) + abs(self.Cell_n.dcx)),self.Cell_s.cy),\
                            (self.Cell_s.cx-(abs(self.Cell_s.dcx) + abs(self.Cell_n.dcx)),self.Cell_n.cy))
            if self.debugging:
                print('Top Left: y={:.2f};x={:.2f};a={:.2f};Y={:.2f};u1={:.5f};v1={:.5f};u2={:.5f};v2={:.5f};s1={:.2f};s2={:.2f}'.format(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
            TravelTime  = self._unit_time(_T(y,x,a,Y,u1,v1,u2,v2,self.s1,self.s2))
            CrossPoints = [self.Cell_s.cx-self.Cell_s.dcx,self.Cell_s.cy+self.Cell_s.dcy]
            CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]
            if self.debugging:
                print('------ TravelTime={:.2f};CrossPoints=[{:.2f},{:.2f}];CellPoints=[{:.2f},{:.5f}]'.format(TravelTime,CrossPoints[0],CrossPoints[1],CellPoints[0],CellPoints[1]))   
            return TravelTime,CrossPoints,CellPoints


        case,crossing_point = self.Mesh.getCase(self.Cell_s,self.Cell_n)

        if self.debugging:
            print('============================================')

        # ======= Determining the Newton Value dependent on case
        if case==2:
            TravelTime,CrossPoints,CellPoints = _positive_longitude(self)
        elif case==-2:
            TravelTime,CrossPoints,CellPoints = _negative_longitude(self)
        elif case==-4:
            TravelTime,CrossPoints,CellPoints = _positive_latitude(self)
        elif case==4:
            TravelTime,CrossPoints,CellPoints = _negative_latitude(self)
        elif case==1:
            TravelTime,CrossPoints,CellPoints = _top_right_corner(self)    
        elif case==3:
            TravelTime,CrossPoints,CellPoints = _bottom_right_corner(self)  
        elif case==-1:
            TravelTime,CrossPoints,CellPoints = _bottom_left_corner(self)
        elif case==-3:
            TravelTime,CrossPoints,CellPoints = _top_left_corner(self)
        else:
            print('---> Issue with cell (Xsc,Ysc)={:.2f};{:.2f}; (dx,dy)={:.2f},{:.2f}; (diffX,diffY)={:.2f},{:.2f}'.format(self.Cell_s.cx,self.Cell_s.cy,self.Cell_s.width/2,self.Cell_s.height/2,self.df_x,self.df_y))
            
            TravelTime  = np.inf
            CrossPoints = [np.nan,np.nan]
            CellPoints  = [np.nan,np.nan]

        CrossPoints[0] = np.clip(CrossPoints[0],self.Cell_n.long,(self.Cell_n.long+self.Cell_n.width))
        CrossPoints[1] = np.clip(CrossPoints[1],self.Cell_n.lat,(self.Cell_n.lat+self.Cell_n.height))

        return TravelTime, CrossPoints, CellPoints


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

class NewtonianCurve:
    def __init__(self,Mesh,Sp,Cp,Np,s,unit_shipspeed='km/hr',unit_time='days',debugging=0,maxiter=1000,optimizer_tol=1e-3,zerocurrents=False):
        self.Mesh = Mesh
        
        # Defining the Source Point (Sp), Crossing Point (Cp) and Neighbour Point(Np)
        self.Sp   = Sp
        self.Cp   = Cp
        self.Np   = Np


        # Inside the code the base units are m/s. Changing the units of the inputs to match
        self.unit_shipspeed = unit_shipspeed
        self.unit_time      = unit_time
        self.s              = self._unit_speed(s)
        
        # Information for distance metrics
        self.R              = 6371*1000.
        self.fdist          = _Euclidean_distance()

        # Optimisation Information
        self.maxiter       = maxiter
        self.optimizer_tol = optimizer_tol

        # For Debugging purposes 
        self.debugging     = debugging
        
        if zerocurrents:
            self.zc = 0.0
        else:
            self.zc = 1.0


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


    def _long_case(self):
            def NewtonOptimisationLong(f,df,x,a,Y,u1,v1,u2,v2,s,R,λ_s,φ_r):
                    y0 = (Y*x)/(x+a)
                    if self.debugging>=2:
                            print('---Initial y={:.2f}'.format(y0))
                    if self.maxiter > 0:
                        for iter in range(self.maxiter):
                            F  = f(y0,x,a,Y,u1,v1,u2,v2,s,R,λ_s,φ_r)
                            dF = df(y0,x,a,Y,u1,v1,u2,v2,s,R,λ_s,φ_r)
                            if self.debugging>=2:
                                print('---Iteration {}: y={:.2f}; F={:.5f}; dF={:.2f}'.format(iter,y0,F,dF))
                            y0  = y0 - (F/dF)
                            if abs(F) < self.optimizer_tol:
                                break

                        if self.debugging>=2:
                            if abs(F) > self.optimizer_tol:
                                print('--- Unusal Opt y={:.2f}; F={:.5f}; dF={:.2f}'.format(y0,F,dF))
                    return y0

            def _F(y,x,a,Y,u1,v1,u2,v2,s,R,λ_s,φ_r):
                θ  = (y/R + λ_s)*(np.pi/180)
                zl = x*np.cos(θ)
                ψ  = (-(Y-y)/R + φ_r)*(np.pi/180)
                zr = a*np.cos(ψ)

                C1  = s**2 - u1**2 - v1**2
                C2  = s**2 - u2**2 - v2**2
                D1  = zl*u1 + y*v1
                D2  = zr*u2 + (Y-y)*v2
                X1  = np.sqrt(D1**2 + C1*(zl**2 + y**2))
                X2  = np.sqrt(D2**2 + C2*(zr**2 + (Y-y)**2))

                dzr = (-a*np.sin(ψ))/R
                dzl = (-x*np.sin(θ))/R

                F  = (X1+X2)*y - ((X1-D1)*X2*v1)/C1 + ((X2-D2)*X1*v2)/C2\
                    - Y*X1 + dzr*(zr-((X2-D2)/C2))*X1 + dzl*(zl-((X1-D1)/C1))*X2
                return F

            def _dF(y,x,a,Y,u1,v1,u2,v2,s,R,λ_s,φ_r):
                θ  = (y/R + λ_s)*(np.pi/180)
                zl = x*np.cos(θ)
                ψ  = (-(Y-y)/R + φ_r)*(np.pi/180)
                zr = a*np.cos(ψ)

                C1  = s**2 - u1**2 - v1**2
                C2  = s**2 - u2**2 - v2**2
                D1  = zl*u1 + y*v1
                D2  = zr*u2 + (Y-y)*v2
                X1  = np.sqrt(D1**2 + C1*(zl**2 + y**2))
                X2  = np.sqrt(D2**2 + C2*(zr**2 + (Y-y)**2))

                dzr = (-a*np.sin(ψ))/R
                dzl = (-x*np.sin(θ))/R

                dD1 = dzl*u1 + v1
                dD2 = dzr*u2 - v2
                dX1 = (D1*v1 + C1*y + dzl*(D1*u1 + C1*zl))/X1
                dX2 = (-v2*D2 - C2*(Y-y) + dzr*(D2*u2 + C2*zr))/X2        

                dF = (X1+X2) + y*(dX1 + dX2) - (v1/C1)*(dX2*(X1-D1) + X2*(dX1-dD1))\
                    + (v2/C2)*(dX1*(X2-D2) + X1*(dX2-dD2))\
                    - Y*dX1 - (zr/(R**2))*(zr-((X2-D2)*u2)/C2)*X1\
                    - (zl/(R**2))*(zl-((X1-D1)*u1)/C1)*X2\
                    + dzr*(dzr-(u2/C2)*(dX2-dD2))*X1\
                    + dzl*(dzl-(u1/C1)*(dX1-dD1))*X2\
                    + dzr*(zr-((X2-D2)*u2)/C2)*dX1 + dzl*(zl-((X1-D1)*u1)/C1)*dX2
                return dF 

            def _T(y,x,a,Y,u1,v1,u2,v2,s,R,λ_s,φ_r):
                θ  = (y/R + λ_s)*(np.pi/180)
                zl = x*np.cos(θ)
                ψ  = (-(Y-y)/R + φ_r)*(np.pi/180)
                zr = a*np.cos(ψ)

                C1  = s**2 - u1**2 - v1**2
                C2  = s**2 - u2**2 - v2**2
                D1  = zl*u1 + y*v1
                D2  = zr*u2 + (Y-y)*v2
                X1  = np.sqrt(D1**2 + C1*(zl**2 + y**2))
                X2  = np.sqrt(D2**2 + C2*(zr**2 + (Y-y)**2))

                t1  = (X1-D1)/C1
                t2  = (X2-D2)/C2
                TT  = t1 + t2
                return TT


            Cp = self.Cp
            if sign(self.Np[0]-self.Sp[0]) == 1:   
                Sp   = self.Sp
                Np   = self.Np
                Box1 = self.Box1
                Box2 = self.Box2
            else:
                Sp   = self.Np
                Np   = self.Sp
                Box1 = self.Box2
                Box2 = self.Box1    
            λ_s  = Sp[1]
            φ_r  = Np[1]

            x     = self.fdist.value(Sp,(Cp[0],Sp[1]))
            a     = self.fdist.value(Np, (Cp[0],Np[1]))
            Y     = sign(Np[1]-Sp[1])*self.fdist.value((Sp[0]+(Np[0]-Sp[0]),Sp[1]),\
                                                       (Sp[0]+(Np[0]-Sp[0]),Np[1]))
            u1    = self.zc*Box1.getuC(); v1 = self.zc*Box1.getvC()
            u2    = self.zc*Box2.getuC(); v2 = self.zc*Box2.getvC()
            y     = NewtonOptimisationLong(_F,_dF,x,a,Y,u1,v1,u2,v2,self.s,self.R,λ_s,φ_r)
            TravelTime  = _T(y,x,a,Y,u1,v1,u2,v2,self.s,self.R,λ_s,φ_r)
            CrossPoint  = self.fdist.value((Cp[0],Sp[1]),(0.0,y),forward=False)
            #CrossPoint[0] = np.clip(CrossPoint[0],Box1.long,(Box1.long+Box1.width)) 
            #CrossPoint[1] = np.clip(CrossPoint[1],Box1.lat,(Box1.lat+Box1.height))
            return TravelTime,np.array(CrossPoint)[None,:]



    def _lat_case(self):
        def NewtonOptimisationLat(f,df,x,a,Y,u1,v1,u2,v2,s,R,λ,θ,ψ):
                y0 = (Y*x)/(x+a)
                if self.debugging>=2:
                        print('---Initial y={:.2f}'.format(y0))
                if self.maxiter > 0:
                    for iter in range(self.maxiter):
                        F  = f(y0,x,a,Y,u1,v1,u2,v2,s,R,λ,θ,ψ)
                        dF = df(y0,x,a,Y,u1,v1,u2,v2,s,R,λ,θ,ψ)
                        if self.debugging>=2:
                            print('---Iteration {}: y={:.2f}; F={:.5f}; dF={:.2f}'.format(iter,y0,F,dF))
                        y0  = y0 - (F/dF)
                        if F < self.optimizer_tol:
                            break
                        if self.debugging>=2:
                            if abs(F) > self.optimizer_tol:
                                print('--- Unusal Opt y={:.2f}; F={:.5f}; dF={:.2f}'.format(y0,F,dF))

                return y0

        def _F(y,x,a,Y,u1,v1,u2,v2,s,R,λ,θ,ψ):
            λ   = λ*(np.pi/180)
            ψ   = ψ*(np.pi/180)
            θ   = θ*(np.pi/180)
            r1  = np.cos(λ)/np.cos(θ)
            r2  = np.cos(ψ)/np.cos(θ)

            d1  = np.sqrt(x**2 + (r1*y)**2)
            d2  = np.sqrt(a**2 + (r2*(Y-y))**2)
            C1  = s**2 - u1**2 - v1**2
            C2  = s**2 - u2**2 - v2**2
            D1  = x*u1 + r1*v1*Y
            D2  = a*u2 + r2*v2*(Y-y)
            X1  = np.sqrt(D1**2 + C1*(d1**2))
            X2  = np.sqrt(D2**2 + C2*(d2**2)) 

            F = ((r2**2)*X1 + (r1**2)*X2)*y - ((r1*(X1-D1)*X2*v1)/C1) + ((r2*(X2-D2)*X1*v2)/C2) - (r2**2)*Y*X1
            return F

        def _dF(y,x,a,Y,u1,v1,u2,v2,s,R,λ,θ,ψ):
            λ   = λ*(np.pi/180)
            ψ   = ψ*(np.pi/180)
            θ   = θ*(np.pi/180)
            r1  = np.cos(λ)/np.cos(θ)
            r2  = np.cos(ψ)/np.cos(θ)


            d1  = np.sqrt(x**2 + (r1*y)**2)
            d2  = np.sqrt(a**2 + (r2*(Y-y))**2)
            C1  = s**2 - u1**2 - v1**2
            C2  = s**2 - u2**2 - v2**2
            D1  = x*u1 + r1*v1*Y
            D2  = a*u2 + r2*v2*(Y-y)
            X1  = np.sqrt(D1**2 + C1*(d1**2))
            X2  = np.sqrt(D2**2 + C2*(d2**2))   
            
            dD1 = r1*v1
            dD2 = -r2*v2
            dX1 = (r1*(D1*v1 + r1*C1*y))/X1
            dX2 = (-r2*(D2*v2 + r2*C2*(Y-y)))/X2

            dF = ((r2**2)*X1 + (r1**2)*X2) + y*((r2**2)*dX1 + (r1**2)*dX2)\
                - ((r1*v1)/C1)*((dX1-dD1)*X2 + (X1-D1)*dX2)\
                + ((r2*v2)/C2)*((dX2-dD2)*X1 + (X2-D2)*dX1)\
                - (r2**2)*Y*dX1

            return dF

        def _T(y,x,a,Y,u1,v1,u2,v2,s,R,λ,θ,ψ):
            λ   = λ*(np.pi/180)
            ψ   = ψ*(np.pi/180)
            θ   = θ*(np.pi/180)
            r1  = np.cos(λ)/np.cos(θ)
            r2  = np.cos(ψ)/np.cos(θ)


            d1  = np.sqrt(x**2 + (r1*y)**2)
            d2  = np.sqrt(a**2 + (r2*(Y-y))**2)
            C1  = s**2 - u1**2 - v1**2
            C2  = s**2 - u2**2 - v2**2
            D1  = x*u1 + r1*v1*Y
            D2  = a*u2 + r2*v2*(Y-y)
            X1  = np.sqrt(D1**2 + C1*(d1**2))
            X2  = np.sqrt(D2**2 + C2*(d2**2))
            t1  = (X1-D1)/C1
            t2  = (X2-D2)/C2

            TT  = t1+t2
            return TT     


        Cp = self.Cp
        if sign(self.Np[1]-self.Sp[1]) == 1:   
            Sp   = self.Sp
            Np   = self.Np
            Box1 = self.Box1
            Box2 = self.Box2
        else:
            Sp   = self.Np
            Np   = self.Sp
            Box1 = self.Box2
            Box2 = self.Box1            

        θ   = Cp[1]
        λ   = Sp[1]
        ψ   = Np[1]        

        x     = self.fdist.value(Sp,(Sp[0],Cp[1]))
        a     = self.fdist.value(Np, (Np[0],Cp[1]))
        Y     = sign(Np[0]-Sp[0])*self.fdist.value((Sp[0],Cp[1]),\
                                 (Np[0],Cp[1]))
        u1    = self.zc*Box1.getvC(); v1 = self.zc*Box1.getuC()
        u2    = self.zc*Box2.getvC(); v2 = self.zc*Box2.getuC()

        y = NewtonOptimisationLat(_F,_dF,x,a,Y,u1,v1,u2,v2,self.s,self.R,λ,θ,ψ)
        TravelTime  = _T(y,x,a,Y,u1,v1,u2,v2,self.s,self.R,λ,θ,ψ)
        CrossPoint  = self.fdist.value((Sp[0],Cp[1]),(y,0.0),forward=False)
        #CrossPoint[0] = np.clip(CrossPoint[0],Box1.long,(Box1.long+Box1.width))   
        #CrossPoint[1] = np.clip(CrossPoint[1],Box1.lat,(Box1.lat+Box1.height))     
        return TravelTime,np.array(CrossPoint)[None,:]


    def _corner_case(self,case,crossing_point):
        '''
            Corner cases as outline in Part 4 of the latex formulations

            Bug/Corrections
                - Issue sometimes the Cp is not on the corner
        '''


        # Defining the lat/long of the points
        Xs,Ys = self.Sp
        Xc,Yc = crossing_point
        Xe,Ye = self.Np

        #print('Cell Indices',self.Mesh.getIndex(self.Box1),self.Mesh.getIndex(self.Box2))


        # # Determine the intersection point on the edge where end_p is assuming a straight path through corner
        Y_line = ((Yc-Ys)/(Xc-Xs))*(Xe-Xs) + Ys

        # Determining the cells in contact with the corner point
        CornerCells = []
        neighbours,neighbours_idx = self.Mesh.getNeightbours(self.Box1)
        for idx in neighbours_idx:
            cell = self.Mesh.cellBoxes[idx]
            if ((((np.array(cell.getBounds()) - np.array([Xc,Yc])[None,:])**2).sum(axis=1))==0).any() and (idx!=self.Mesh.getIndex(self.Box1)[0]) and (idx!=self.Mesh.getIndex(self.Box2)[0]):
                CornerCells.append([idx,cell.long+cell.width/2,cell.lat+cell.height/2]) 
        CornerCells = np.array(CornerCells)
        #print(CornerCells)

        if len(CornerCells) == 0:
            print('Issue - No Corner Cp={};'.format(self.Cp))
            print('Sp=({},{});Cp=({},{});Np=({},{})'.format(self.Sp[0],self.Sp[1],self.Cp[0],self.Cp[1],self.Np[0],self.Np[1]))
            print('Box1=({},{});Box2=({},{})'.format(self.Box1.cx,self.Box1.cy,self.Box2.cx,self.Box2.cy))
            print('No-Case: df_X={},df_Y={}'.format(self.df_x,self.df_y))
            print('Neighbour Index {}'.format(neighbours_idx))
        #     TravelTime = np.nan
        #     CrossPoint = np.array([[np.nan,np.nan]])
        #     return TravelTime,np.array(CrossPoint)           

        # ====== Determining the crossing points & their corresponding index
        # Case 1 - Top Right
        if case==1:
            if self.debugging>=1:
                print('--Corner Case TopRight: Xs=[{:.2f},{:.2f}]; Xc=[{:.2f},{:.2f}]; Xe=[{:.2f},{:.2f}];'.format(Xs,Ys,Xc,Yc,Xe,Ye))   
                print('----- df_x={}, df_y={}'.format(self.df_x,self.df_y))
                print('---- Ye={}; Y_line={}'.format(Ye,Y_line))
                print('-- Corner Cells={}'.format(CornerCells))

            if Ye >= Y_line:
                idx           = int(CornerCells[CornerCells[:,1].argmin(),0])
                cell          = self.Mesh.cellBoxes[idx]
                Crp1_x,Crp1_y = Intersection_BoxLine(self.Box1,cell,'top')
                Crp2_x,Crp2_y = Intersection_BoxLine(self.Box2,cell,'left')
                
            elif Ye < Y_line:
                idx           = int(CornerCells[CornerCells[:,1].argmax(),0])
                cell          = self.Mesh.cellBoxes[idx]

                Crp1_x,Crp1_y = Intersection_BoxLine(self.Box1,cell,'right')
                Crp2_x,Crp2_y = Intersection_BoxLine(self.Box2,cell,'bottom')

        # Case -3 - Top Left
        if case==-3:
            if self.debugging>=1:
                print('--Corner Case TopLeft: Xs=[{:.2f},{:.2f}]; Xc=[{:.2f},{:.2f}]; Xe=[{:.2f},{:.2f}];'.format(Xs,Ys,Xc,Yc,Xe,Ye))   
                print('----- df_x={}, df_y={}'.format(self.df_x,self.df_y))
                print('---- Ye={}; Y_line={}'.format(Ye,Y_line))
                print('-- Corner Cells={}'.format(CornerCells)) 

            if Ye >= Y_line:
                idx  = int(CornerCells[CornerCells[:,1].argmax(),0])
                cell = self.Mesh.cellBoxes[idx]
                Crp1_x,Crp1_y = Intersection_BoxLine(self.Box1,cell,'top')
                Crp2_x,Crp2_y = Intersection_BoxLine(self.Box2,cell,'right')

            elif Ye < Y_line:
                idx  = int(CornerCells[CornerCells[:,1].argmin(),0])
                cell = self.Mesh.cellBoxes[idx]
                Crp1_x,Crp1_y = Intersection_BoxLine(self.Box1,cell,'left')
                Crp2_x,Crp2_y = Intersection_BoxLine(self.Box2,cell,'bottom')

        # Case -1 - Bottom Left
        if case==-1:
            if self.debugging>=1:
                print('--Corner Case BottomLeft: Xs=[{:.2f},{:.2f}]; Xc=[{:.2f},{:.2f}]; Xe=[{:.2f},{:.2f}];'.format(Xs,Ys,Xc,Yc,Xe,Ye))   
                print('----- df_x={}, df_y={}'.format(self.df_x,self.df_y)) 
                print('---- Ye={}; Y_line={}'.format(Ye,Y_line))
                print('-- Corner Cells={}'.format(CornerCells))

            if Ye >= Y_line:
                idx  = int(CornerCells[CornerCells[:,1].argmin(),0])
                cell = self.Mesh.cellBoxes[idx]
                Crp1_x,Crp1_y = Intersection_BoxLine(self.Box1,cell,'left')
                Crp2_x,Crp2_y = Intersection_BoxLine(self.Box2,cell,'top') 
            elif Ye < Y_line:
                idx  = int(CornerCells[CornerCells[:,1].argmax(),0])
                cell = self.Mesh.cellBoxes[idx]
                Crp1_x,Crp1_y = Intersection_BoxLine(self.Box1,cell,'bottom')
                Crp2_x,Crp2_y = Intersection_BoxLine(self.Box2,cell,'right')

        # Case 3 - Bottom Right
        if case==3:
            if self.debugging>=1:
                print('--Corner Case BottomRight: Xs=[{:.2f},{:.2f}]; Xc=[{:.2f},{:.2f}]; Xe=[{:.2f},{:.2f}];'.format(Xs,Ys,Xc,Yc,Xe,Ye))   
                print('----- df_x={}, df_y={}'.format(self.df_x,self.df_y))
                print('---- Ye={}; Y_line={}'.format(Ye,Y_line))
                print('-- Corner Cells={}'.format(CornerCells)) 

            if Ye >= Y_line:
                idx  = int(CornerCells[CornerCells[:,1].argmax(),0])
                cell = self.Mesh.cellBoxes[idx]
                Crp1_x,Crp1_y = Intersection_BoxLine(self.Box1,cell,'right')
                Crp2_x,Crp2_y = Intersection_BoxLine(self.Box2,cell,'top')

            elif Ye < Y_line:
                idx  = int(CornerCells[CornerCells[:,1].argmin(),0])
                cell = self.Mesh.cellBoxes[idx]
                Crp1_x,Crp1_y = Intersection_BoxLine(self.Box1,cell,'bottom')
                Crp2_x,Crp2_y = Intersection_BoxLine(self.Box2,cell,'left')

        # Appending the crossing points and their relative index
        CrossPoint = [[Crp1_x,Crp1_y],[Crp2_x,Crp2_y]]
        TravelTime = np.nan
        return TravelTime,np.array(CrossPoint)

    def value(self):
        '''
            Bug - Some corner cases are actually Longitude. 
                  Need to correct to select corner case based on the angle of the line between the two cell centres.
        
                - Large issue that loading the box information is currently incorrect and need to



        '''

        # Determining the cells to use
        print('Sp=({},{});Cp=({},{});Np=({},{})'.format(self.Sp[0],self.Sp[1],self.Cp[0],self.Cp[1],self.Np[0],self.Np[1]))
        print('Box1=({},{});Box2=({},{})'.format((self.Sp[1]+self.Cp[1])/2 , (self.Sp[0]+self.Cp[0])/2, (self.Np[1]+self.Cp[1])/2 , (self.Np[0]+self.Cp[0])/2))

        self.Box1 = self.Mesh.getCellBox( (self.Sp[1]+self.Cp[1])/2 , (self.Sp[0]+self.Cp[0])/2 )
        self.Box2 = self.Mesh.getCellBox( (self.Np[1]+self.Cp[1])/2 , (self.Np[0]+self.Cp[0])/2 ) 
        self.df_x = (self.Box2.long+self.Box2.width/2) -  (self.Box1.long+self.Box1.width/2)
        self.df_y = (self.Box2.lat+self.Box2.height/2) -  (self.Box1.lat+self.Box1.height/2)

        if self.Box1.containsLand() or self.Box2.containsLand():
            TravelTime = np.nan
            CrossPoint = np.array([[np.nan,np.nan]])
            return TravelTime, self.Cp,self.Box1,self.Box2
        case,crossing_point = self.Mesh.getCase(self.Box1,self.Box2)


        if self.debugging>0:
            print('===========================================================')
        if abs(case)==2:
            TravelTime, CrossPoint = self._long_case()
        elif abs(case)==4:
            TravelTime, CrossPoint = self._lat_case()
        elif (abs(case)==1) or (abs(case)==3):
            TravelTime, CrossPoint = self._corner_case(case,crossing_point)
        else:
            #TravelTime, CrossPoint = self._corner_case()
            print('Sp=({},{});Cp=({},{});Np=({},{})'.format(self.Sp[0],self.Sp[1],self.Cp[0],self.Cp[1],self.Np[0],self.Np[1]))
            print('Box1=({},{});Box2=({},{})'.format((self.Sp[1]+self.Cp[1])/2 , (self.Sp[0]+self.Cp[0])/2, (self.Np[1]+self.Cp[1])/2 , (self.Np[0]+self.Cp[0])/2))
            print('No-Case: df_X={},df_Y={}'.format(self.df_x,self.df_y))
            TravelTime = np.nan
            CrossPoint = self.Cp
        
        return TravelTime, CrossPoint,self.Box1,self.Box2
