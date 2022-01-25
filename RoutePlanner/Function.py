import numpy as np
import math
from scipy import optimize

# ================================================================
# ================================================================
# ================================================================
# ================================================================

def _Haversine_distance(origin, destination,forward=True):
    """
    Calculate the Haversine distance between two points 
    Inputs:
      origin      - tuple of floats e.g. (Lat_orig,Long_orig)
      destination - tuple of floats e.g. (Lat_dest,Long_dest)
    Output:
      Distance - Distance between two points in 'km'

    """
    R = 6371  # Radius of earth
    def haversine(theta):
        return math.sin(theta/2) ** 2

    def deg2rad(deg,forward=True):
        if forward:
            d = deg * (math.pi/180)
        else:
            d = deg * (180/math.pi)
        return d

    def distance(pa,pb):
        a_long,a_lat = pa
        b_long,b_lat = pb

        lat1  = deg2rad(a_lat)
        lat2  = deg2rad(b_lat)
        dLat  = deg2rad(a_lat - b_lat)
        dLong = deg2rad(a_long - b_long)
        x     =  haversine(dLat) + math.cos(lat1)*math.cos(lat2)*haversine(dLong)
        c     = 2*math.atan2(math.sqrt(x), math.sqrt(1-x))
        return R*c  

    def point(pa,dist):
        # Determining the latituted difference in Long & Lat
        a_long,a_lat = pa
        distX,distY  = dist

        lat1  = deg2rad(a_lat)
        dLat   = deg2rad(distX/R,forward=False)
        dLong  = deg2rad(2*math.asin(math.sqrt(haversine(distY/R)/(math.cos(lat1)**2))),forward=False)
        b_long = a_long + dLong
        b_lat  = a_lat + dLat

        return [b_long,b_lat]

    if forward:
        val = distance(origin,destination)
    else:
        val = point(origin,destination)
    return val


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

    kmperdeglat          = 111.386
    kmperdeglonAtEquator = 111.321
    if forward:
        lon1,lat1 = origin
        lon2,lat2 = dest_dist
        val = np.sqrt(((lat2-lat1)*kmperdeglat)**2 + ((lon2-lon1)*kmperdeglonAtEquator)**2)
    else:
        lon1,lat1     = origin
        dist_x,dist_y = dest_dist        
        val = [lon1+(dist_x/kmperdeglonAtEquator),lat1+(dist_y/kmperdeglat)]

    return val


# ================================================================
# ================================================================
# ================================================================
# ================================================================

# # === New Scripting
class SmoothedNewtonianDistance:
    def __init__(self,Cell_s,Cell_n,s):

        self.Cell_s = Cell_s
        self.Cell_n = Cell_n
        self.s      = s
        self.fdist  = _Euclidean_distance
        self.R      = 6371

        # Determining the distance between the cell centres to 
        #be used in defining the case
        self.df_x = (self.Cell_n.x+self.Cell_n.dx) -  (self.Cell_s.x+self.Cell_s.dx)
        self.df_y = (self.Cell_n.y+self.Cell_n.dy) -  (self.Cell_s.y+self.Cell_s.dy)
        
        # Determine the distance to the edge of the box. This
        #is important for waypoint cases
        if np.sign(self.df_x) == 1:
            self.S_dx = self.Cell_s.dxp; self.N_dx = -self.Cell_n.dxm
        else:
            self.S_dx = -self.Cell_s.dxm; self.N_dx = self.Cell_n.dxp  
        #dY       
        if np.sign(self.df_y) == 1:
            self.S_dy = self.Cell_s.dyp; self.N_dy = -self.Cell_n.dym
        else:
            self.S_dy = -self.Cell_s.dym; self.N_dy = self.Cell_n.dyp 


    def value(self):

        if ((abs(self.df_x) >= (self.Cell_s.dx/2)) and (abs(self.df_y) < (self.Cell_s.dy/2))):
            TravelTime, CrossPoints, CellPoints = self._long_case()
        elif (abs(self.df_x) < self.Cell_s.dx/2) and (abs(self.df_y) >= self.Cell_s.dy/2):
            TravelTime, CrossPoints, CellPoints = self._lat_case()
        elif (abs(self.df_x) >= self.Cell_s.dx/2) and (abs(self.df_y) >= self.Cell_s.dy/2):
            TravelTime, CrossPoints, CellPoints = self._corner_case()
        else:
            TravelTime  = np.inf
            CrossPoints = [np.nan,np.nan]
            CellPoints  = [np.nan,np.nan]

        CrossPoints[0] = np.clip(CrossPoints[0],self.Cell_n.x,(self.Cell_n.x+self.Cell_n.dx))
        CrossPoints[1] = np.clip(CrossPoints[1],self.Cell_n.y,(self.Cell_n.y+self.Cell_n.dy))

        return TravelTime, CrossPoints, CellPoints

    def _long_case(self):

        def _F(y,x,a,Y,u1,v1,u2,v2,s,R,λ_s,φ_r):
            θ  = y/R + λ_s
            zl = x*math.cos(θ)
            ψ  = -(Y-y)/R + φ_r
            zr = a*math.cos(ψ)

            d1  = math.sqrt(zl**2 + y**2)
            d2  = math.sqrt(zl**2 + (Y-y)**2)
            C1  = s**2 - u1**2 - v1**2
            D1  = zl*u1 + y*v1
            X1  = math.sqrt(D1**2 + C1*(d1**2))
            C2  = s**2 - u2**2 - v2**2
            D2  = zr*u2 + (Y-y)*v2
            X2  = math.sqrt(D2**2 + C2*(d2**2))

            dzr = (-a*math.sin(ψ))/R
            dzl = (-x*math.sin(θ))/R

            F  = (X1+X2)*y - ((X1-D1)*X2*v1)/C1 + ((X2-D2)*X1*v2)/C2\
                - Y*X1 + dzr*(zr-((X2-D2)/C2))*X1 + dzl*(zl-((X1-D1)/C1))*X2
            return F

        def _dF(y,x,a,Y,u1,v1,u2,v2,s,R,λ_s,φ_r):
            θ  = y/R + λ_s
            zl = x*math.cos(θ)
            ψ  = -(Y-y)/R + φ_r
            zr = a*math.cos(ψ)

            d1  = math.sqrt(zl**2 + y**2)
            d2  = math.sqrt(zl**2 + (Y-y)**2)
            C1  = s**2 - u1**2 - v1**2
            D1  = zl*u1 + y*v1
            X1  = math.sqrt(D1**2 + C1*(d1**2))
            C2  = s**2 - u2**2 - v2**2
            D2  = zr*u2 + (Y-y)*v2
            X2  = math.sqrt(D2**2 + C2*(d2**2))

            dzr = (-a*math.sin(ψ))/R
            dzl = (-x*math.sin(θ))/R
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
                + dzr(zr-((X2-D2)*u2)/C2)*dX1 + dzl*(zl-((X1-D1)*u1)/C1)*dX2
            return dF 

        def _T(y,x,a,Y,u1,v1,u2,v2,s,R,λ_s,φ_r):
            θ  = y/R + λ_s
            zl = x*math.cos(θ)
            ψ  = -(Y-y)/R + φ_r
            zr = a*math.cos(ψ)

            d1  = math.sqrt(zl**2 + y**2)
            d2  = math.sqrt(zl**2 + (Y-y)**2)
            C1  = s**2 - u1**2 - v1**2
            D1  = zl*u1 + y*v1
            X1  = math.sqrt(D1**2 + C1*(d1**2))
            t1  = (X1-D1)/C1
            C2  = s**2 - u2**2 - v2**2
            D2  = zr*u2 + (Y-y)*v2
            X2  = math.sqrt(D2**2 + C2*(d2**2))
            t2  = (X2-D2)/C2
            TT  = t1 + t2
            return TT

        u1 = np.sign(self.df_x)*self.Cell_s.vector[0]; v1 = self.Cell_s.vector[1]
        u2 = np.sign(self.df_x)*self.Cell_n.vector[0]; v2 = self.Cell_n.vector[1]
        λ_s = self.Cell_s.cy
        φ_r = self.Cell_n.cy
        x     = self.fdist((self.Cell_s.cx,self.Cell_s.cy), (self.Cell_s.cx + self.S_dx,self.Cell_s.cy))
        a     = self.fdist((self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx + self.N_dx,self.Cell_n.cy))
        Y     = self.fdist((self.Cell_s.cx + np.sign(self.df_x)*(abs(self.S_dx) + abs(self.N_dx)), self.Cell_s.cy), (self.Cell_s.cx + np.sign(self.df_x)*(abs(self.S_dx) + abs(self.N_dx)),self.Cell_n.cy))
        ang   = np.arctan((self.Cell_n.cy - self.Cell_s.cy)/(self.Cell_n.cx - self.Cell_s.cx))
        yinit = np.tan(ang)*(self.S_dx)
        try:
            y  = optimize.newton(_F,yinit,args=(x,a,Y,u1,v1,u2,v2,self.s,self.R,λ_s,φ_r),fprime=_dF)
        except:
            y  = yinit
        TravelTime  = _T(y,x,a,Y,u1,v1,u2,v2,self.s,self.R,λ_s,φ_r)
        CrossPoints = self.fdist((self.Cell_s.cx + self.S_dx,self.Cell_s.cy),(0.0,y),forward=False)
        CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]

        return TravelTime,CrossPoints,CellPoints


    def _lat_case(self):
        def _F(y,x,a,Y,u1,v1,u2,v2,s,R,λ,θ,ψ):
            r1  = math.cos(λ)/math.cos(θ)
            r2  = math.cos(ψ)/math.cos(θ)

            d1  = math.sqrt(x**2 + (r1*y)**2)
            d2  = math.sqrt(a**2 + (r2*(Y-y))**2)
            C1  = s**2 - u1**2 - v1**2
            C2  = s**2 - u2**2 - v2**2
            D1  = x*u1 + r1*v1*Y
            D2  = a*u2 + r2*v2*(Y-y)
            X1  = math.sqrt(D1**2 + C1*(d1**2))
            X2  = math.sqrt(D2**2 + C2*(d2**2)) 

            F = ((r2**2)*X1 + (r1**2)*X2)*y - ((r1*(X1-D1)*X2*v2)/C1) + ((r2*(X2-D2)*X1*v2)/C2) - (r2**2)*Y*X1
            return F

        def _dF(y,x,a,Y,u1,v1,u2,v2,s,R,λ,θ,ψ):
            r1  = math.cos(λ)/math.cos(θ)
            r2  = math.cos(ψ)/math.cos(θ)

            d1  = math.sqrt(x**2 + (r1*y)**2)
            d2  = math.sqrt(a**2 + (r2*(Y-y))**2)
            C1  = s**2 - u1**2 - v1**2
            C2  = s**2 - u2**2 - v2**2
            D1  = x*u1 + r1*v1*Y
            D2  = a*u2 + r2*v2*(Y-y)
            X1  = math.sqrt(D1**2 + C1*(d1**2))
            X2  = math.sqrt(D2**2 + C2*(d2**2))   
            
            dD1 = r1*v1
            dD2 = -r2*v2
            dX1 = (r1*(D1*v1 + r1*C1*y))/X1
            dX2 = (-r2*(D2*v2 + r2*C2*(Y-y)))

            dF = ((r2**2)*X1 + (r1**2)*X2) + y*((r2**2)*dX1 + (r1**2)*dX2)\
                - ((r1*v1)/C1)*((dX1-dD1)*X2 + (X1-D1)*dX2)\
                + ((r2*v2)/C2)*((dX2-dD2)*X1 + (X2-D2)*dX1)\
                - (r2**2)*Y*dX1

            return dF

        def _T(y,x,a,Y,u1,v1,u2,v2,s,R,λ,θ,ψ):
            r1  = math.cos(λ)/math.cos(θ)
            r2  = math.cos(ψ)/math.cos(θ)

            d1  = math.sqrt(x**2 + (r1*y)**2)
            d2  = math.sqrt(a**2 + (r2*(Y-y))**2)
            C1  = s**2 - u1**2 - v1**2
            C2  = s**2 - u2**2 - v2**2
            D1  = x*u1 + r1*v1*Y
            D2  = a*u2 + r2*v2*(Y-y)
            X1  = math.sqrt(D1**2 + C1*(d1**2))
            X2  = math.sqrt(D2**2 + C2*(d2**2))
            t1  = (X1-D1)/C1
            t2  = (X2-D2)/C2

            TT  = t1+t2
            return TT     

        u1 = np.sign(self.df_y)*self.Cell_s.vector[1]; v1 = self.Cell_s.vector[0]
        u2 = np.sign(self.df_y)*self.Cell_n.vector[1]; v2 = self.Cell_n.vector[0]

        x  = self.fdist((self.Cell_s.cy,self.Cell_s.cx), (self.Cell_s.cy + self.S_dy,self.Cell_s.cx))
        a  = self.fdist((self.Cell_n.cy,self.Cell_n.cx), (self.Cell_n.cy + self.N_dy,self.Cell_n.cx))
        Y  = self.fdist((self.Cell_s.cy + np.sign(self.df_y)*(abs(self.S_dy) + abs(self.N_dy)), self.Cell_s.cx), (self.Cell_s.cy + np.sign(self.df_y)*(abs(self.S_dy) + abs(self.N_dy)),self.Cell_n.cx))
        ang= np.arctan((self.Cell_n.cx - self.Cell_s.cx)/(self.Cell_n.cy - self.Cell_s.cy))
        yinit  = np.tan(ang)*(self.S_dy)
        
        λ=self.Cell_s.cy
        θ=self.Cell_s.cy + self.S_dy
        ψ=self.Cell_n.cy

        try:
            y  = optimize.newton(_F,yinit,args=(x,a,Y,u1,v1,u2,v2,self.s,self.R,λ,θ,ψ),fprime=_dF)
        except:
            y  = yinit
        TravelTime  = _T(y,x,a,Y,u1,v1,u2,v2,self.s,self.R,λ,θ,ψ)
        CrossPoints = self.fdist((self.Cell_s.cx + self.S_dx,self.Cell_s.cy),(0.0,y),forward=False)
        CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]        

        return TravelTime,CrossPoints,CellPoints

    def _corner_case(self):
        '''
            Corner cases as outline in Part 4 of the latex formulations
        '''
        
        # Include Definition here

class NewtonianDistance:
    def __init__(self,Cell_s,Cell_n,s):

        
        self.Cell_s = Cell_s
        self.Cell_n = Cell_n
        self.s      = s
        self.fdist  = _Euclidean_distance

        # Determining the distance between the cell centres to 
        #be used in defining the case
        self.df_x = (self.Cell_n.x+self.Cell_n.dx) -  (self.Cell_s.x+self.Cell_s.dx)
        self.df_y = (self.Cell_n.y+self.Cell_n.dy) -  (self.Cell_s.y+self.Cell_s.dy)
        
        # Determine the distance to the edge of the box. This
        #is important for waypoint cases
        if np.sign(self.df_x) == 1:
            self.S_dx = self.Cell_s.dxp; self.N_dx = -self.Cell_n.dxm
        else:
            self.S_dx = -self.Cell_s.dxm; self.N_dx = self.Cell_n.dxp  
        #dY       
        if np.sign(self.df_y) == 1:
            self.S_dy = self.Cell_s.dyp; self.N_dy = -self.Cell_n.dym
        else:
            self.S_dy = -self.Cell_s.dym; self.N_dy = self.Cell_n.dyp 


    def value(self):
        # -- Initially defining Newton function & its derivative
        def _F(y,x,a,Y,u1,v1,u2,v2,s):
            d1 = x**2 + y**2
            d2 = a**2 + (Y-y)**2
            C1 = s**2 - u1**2 - v1**2
            C2 = s**2 - u2**2 - v2**2
            D1 = x*u1 + y*v1
            D2 = a*u2 + (Y-y)*v2
            X1 = math.sqrt(D1**2 + C1*(d1**2))
            X2 = math.sqrt(D2**2 + C2*(d2**2))
            F  = X2*(y-((v1*(X1-D1))/C1)) + X1*(y-Y+((v2*(X2-D2))/C2)) 
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
            X1 = np.sqrt(D1**2 + C1*(d1**2))
            X2 = np.sqrt(D2**2 + C2*(d2**2))
            t1 = (X1-D1)/C1
            t2 = (X2-D2)/C2
            T  = t1+t2 
            return T

        # Longitude Cases - 2,-2
        if ((abs(self.df_x) >= (self.Cell_s.dx/2)) and (abs(self.df_y) < (self.Cell_s.dy/2))):
            u1 = np.sign(self.df_x)*self.Cell_s.vector[0]; v1 = self.Cell_s.vector[1]
            u2 = np.sign(self.df_x)*self.Cell_n.vector[0]; v2 = self.Cell_n.vector[1]

            x  = self.fdist( (self.Cell_s.cx,self.Cell_s.cy), (self.Cell_s.cx + self.S_dx,self.Cell_s.cy))
            a  = self.fdist( (self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx + self.N_dx,self.Cell_n.cy))
            Y  = self.fdist((self.Cell_s.cx + np.sign(self.df_x)*(abs(self.S_dx) + abs(self.N_dx)), self.Cell_s.cy), (self.Cell_s.cx + np.sign(self.df_x)*(abs(self.S_dx) + abs(self.N_dx)),self.Cell_n.cy))
            ang= np.arctan((self.Cell_n.cy - self.Cell_s.cy)/(self.Cell_n.cx - self.Cell_s.cx))
            yinit  = np.tan(ang)*(self.S_dx)
            try:
                y  = optimize.newton(_F,yinit,args=(x,a,Y,u1,v1,u2,v2,self.s),fprime=_dF)
            except:
                y  = yinit
            TravelTime  = _T(y,x,a,Y,u1,v1,u2,v2,self.s)
            CrossPoints = self.fdist((self.Cell_s.cx + self.S_dx,self.Cell_s.cy),(0.0,y),forward=False)
            CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]

        # Latitude Cases - 4,-4
        elif (abs(self.df_x) < self.Cell_s.dx/2) and (abs(self.df_y) >= self.Cell_s.dy/2):
            u1 = np.sign(self.df_y)*self.Cell_s.vector[1]; v1 = self.Cell_s.vector[0]
            u2 = np.sign(self.df_y)*self.Cell_n.vector[1]; v2 = self.Cell_n.vector[0]

            x  = self.fdist((self.Cell_s.cy,self.Cell_s.cx), (self.Cell_s.cy + self.S_dy,self.Cell_s.cx))
            a  = self.fdist((self.Cell_n.cy,self.Cell_n.cx), (self.Cell_n.cy + self.N_dy,self.Cell_n.cx))
            Y  = self.fdist((self.Cell_s.cy + np.sign(self.df_y)*(abs(self.S_dy) + abs(self.N_dy)), self.Cell_s.cx), (self.Cell_s.cy + np.sign(self.df_y)*(abs(self.S_dy) + abs(self.N_dy)),self.Cell_n.cx))
            ang= np.arctan((self.Cell_n.cx - self.Cell_s.cx)/(self.Cell_n.cy - self.Cell_s.cy))
            yinit  = np.tan(ang)*(self.S_dy)
            try:
                y = optimize.newton(_F,yinit,args=(x,a,Y,u1,v1,u2,v2,self.s),fprime=_dF)
            except:
                y = yinit
            TravelTime   = _T(y,x,a,Y,u1,v1,u2,v2,self.s)
            CrossPoints  = self.fdist((self.Cell_s.cx,self.Cell_s.cy + self.S_dy),(-y,0.0),forward=False)
            CellPoints   = [self.Cell_n.cx,self.Cell_n.cy]    
        
        # Diagonal Cases - 1,-1,3,-3
        elif (abs(self.df_x) >= self.Cell_s.dx/2) and (abs(self.df_y) >= self.Cell_s.dy/2):
            u1 = np.sign(self.df_x)*self.Cell_s.vector[0]; v1 = self.Cell_s.vector[1]
            u2 = np.sign(self.df_x)*self.Cell_n.vector[0]; v2 = self.Cell_n.vector[1]
            
            x  = self.fdist((self.Cell_s.cx,self.Cell_s.cy),(self.Cell_s.cx + self.S_dx,self.Cell_s.cy))
            y  = self.fdist((self.Cell_s.cx+self.S_dx,self.Cell_s.cy),(self.Cell_s.cx + self.S_dx,self.Cell_s.cy+self.S_dy))
            a  = self.fdist((self.Cell_n.cx,self.Cell_n.cy), (self.Cell_n.cx + self.N_dx,self.Cell_n.cy))
            Y  = self.fdist((self.Cell_s.cx + np.sign(self.df_x)*(abs(self.S_dx) + abs(self.N_dx)), self.Cell_s.cy), (self.Cell_s.cx + np.sign(self.df_x)*(abs(self.S_dx) + abs(self.N_dx)),self.Cell_n.cy))
            
            TravelTime  = _T(y,x,a,Y,u1,v1,u2,v2,self.s)
            CrossPoints = self.fdist((self.Cell_s.cx+self.S_dx,self.Cell_s.cy),(0.0,y),forward=False)
            CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]

        # Inside Cell
        else:
            TravelTime  = np.inf
            CrossPoints = [np.nan,np.nan]
            CellPoints  = [np.nan,np.nan]

        CrossPoints[0] = np.clip(CrossPoints[0],self.Cell_n.x,(self.Cell_n.x+self.Cell_n.dx))
        CrossPoints[1] = np.clip(CrossPoints[1],self.Cell_n.y,(self.Cell_n.y+self.Cell_n.dy))


        return TravelTime, CrossPoints, CellPoints