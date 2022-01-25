import numpy as np
import math

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

def _F(y,x,a,Y,u1,v1,u2,v2,s):
    '''
        Minimisation function of ...

        Variable definitions correspond to those given in the
        paper ??? Figure 4

        Inputs:
          x     - Left cell horizontal distance to right edge from centroid
          y     - Crossing point right of box relative to left cell position
          a     - Right cell horizontal distance to left edge from centroid
          u1,v1 - Current vector in left cell
          u2,v2 - Current vector in right cell
          s     - vesal speed in cells
          Y     - Vertical difference between left and right cell

        Outputs:
    '''
    d1 = x**2 + y**2
    d2 = a**2 + (Y-y)**2
    C1 = s**2 - u1**2 - v1**2
    C2 = s**2 - u2**2 - v2**2
    D1 = x*u1 + y*v1
    D2 = a*u2 + (Y-y)*v2
    X1 = np.sqrt(D1**2 + C1*(d1**2))
    X2 = np.sqrt(D2**2 + C2*(d2**2))
    # Minimisation Function
    F  = X2*(y-((v1*(X1-D1))/C1)) + X1*(y-Y+((v2*(X2-D2))/C2)) 
    return F

def _dF(y,x,a,Y,u1,v1,u2,v2,s):
    '''
        Analytical Differentiation function of ...

        Variable definitions correspond to those given in the
        paper ??? Figure 4

        Inputs:
          x     - Left cell horizontal distance to right edge from centroid
          y     - Crossing point right of box relative to left cell position
          a     - Right cell horizontal distance to left edge from centroid
          u1,v1 - Current vector in left cell
          u2,v2 - Current vector in right cell
          s     - vesal speed in cells
          Y     - Vertical difference between left and right cell

        Outputs:
    '''
    d1 = x**2 + y**2
    d2 = a**2 + (Y-y)**2
    C1 = s**2 - u1**2 - v1**2
    C2 = s**2 - u2**2 - v2**2
    D1 = x*u1 + y*v1
    D2 = a*u2 + (Y-y)*v2
    X1 = np.sqrt(D1**2 + C1*(d1**2))
    X2 = np.sqrt(D2**2 + C2*(d2**2))
    # Analytical Derivatives
    dD1 = v1
    dD2 = -v2
    dX1 = (D1*v1 + C1*y)/X1
    dX2 = (D2*v2 - C1*(Y-y))/X1
    # Derivative Function
    dF = (X1+X2) + y*(dX1 + dX2) - (v1/C1)*(dX2*(X1-D1) + X2*(dX1-dD1)) + (v2/C2)*(dX1*(X2-D2)+X1*(dX2-dD2)) - Y*dX1
    return dF

def _T(y,x,a,Y,u1,v1,u2,v2,s):
    '''
        Indivdual Travel-time between two adjacent Cells given the current field

        Variable definitions correspond to those given in the
        paper ??? Figure 4

        Inputs:
          x     - Left cell horizontal distance to right edge from centroid
          y     - Crossing point right of box relative to left cell position
          a     - Right cell horizontal distance to left edge from centroid
          u1,v1 - Current vector in left cell
          u2,v2 - Current vector in right cell
          s     - vesal speed in cells
          Y     - Vertical difference between left and right cell

        Outputs:
    '''
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

