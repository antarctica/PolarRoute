import numpy as np
import pyproj
import logging
from polar_route.route_planner.crossing import traveltime_in_cell
from polar_route.utils import unit_time, unit_speed, case_from_angle


def dist_around_globe(start_point,crossing_point):
    """
        Determining the longitude distance around the globe between two points
    
        Args:
            start_point    (tuple): Start Waypoint (long,lat)
            crossing_point (tuple): End Waypoint (long,lat)
        Returns:
            a (float): longitude distance between the two points in degrees
    """
    a1 = np.sign(crossing_point-start_point)*(np.max([start_point,crossing_point])-np.min([start_point,crossing_point]))
    a2 = -(360-(np.max([start_point,crossing_point])-np.min([start_point,crossing_point])))*np.sign(crossing_point-start_point)

    dist = [a1,a2]
    indx = np.argmin(abs(np.array(dist)))

    a = dist[indx]
    return a


def rhumb_line_distance(start_waypoint, end_waypoint):
    """
        Defining the rhumb line distance from a given waypoint start and end point

        Args:
            start_waypoint (list([Long,lat])): Start Waypoint location with long lat
            end_waypoint (list([Long,lat])): End Waypoint location with long lat

        Returns:
            distance (float): Calculated rhumb line distance
    """

    # Defining a corrected distance based on rhumb lines
    x_s, y_s  = start_waypoint
    x_e, y_e  = end_waypoint

    r = 6371.1*1000.
    dy_corrected = np.log(np.tan((np.pi/4) + (y_e * (np.pi/180)) / 2) / np.tan((np.pi/4) + (y_s * (np.pi/180)) / 2))
    dx = (x_e - x_s) * (np.pi / 180)
    dy = (y_e - y_s) * (np.pi / 180)
    if dy_corrected==0 and dy==0:
        q = np.cos(y_e * (np.pi / 180))
    else:
        q = dy/dy_corrected

    distance = np.sqrt(dy**2 + (q**2)*(dx**2))*r

    return distance


class FindEdge:
    """
        Class to return characteristics information about the edge connecting two
        cells. This information includes:

        crossing (tuple) - Crossing point (long,lat)
        case (int)       - Case type connecting the two cells
        start (dict)     - Dictionary containing the environmental parameters of the start cell
        end   (dict)     - Dictionary containing the environmental parameters of the end cell

    """
    def __init__(self, cell_a, cell_b, case):
        self.crossing, self.case, self.start, self.end = self._find_edge(cell_a, cell_b, case)

    def _find_edge(self, cell_a, cell_b, case):
        """
            Function that returns the edge connecting to cells, cell_a and cell_b. If there is no edge
            connecting the two then it returns None

            Args:
                cell_a (dict): Dictionary of cell_a information
                cell_b (dict): Dictionary of cell_b information

            Returns
                crossing (tuple) - Crossing point (long,lat) connecting the two cells
                case (int)       - Case type connecting the two cells
                start (dict)     - Dictionary containing the environmental parameters of the start cell
                end   (dict)     - Dictionary containing the environmental parameters of the end cell
        """

        # Determining the crossing point between the two cells
        crossing_points = []
        for indx in range(len(cell_a['neighbourCrossingPoints'])):
            if (cell_a['neighbourIndex'][indx] == cell_b['id']) and (cell_a['case'][indx] == case):
                crossing_points += [np.array(cell_a['neighbourCrossingPoints'][indx])]
        if len(crossing_points) == 1:
            crossing  = crossing_points[0]
            case      = case
            start     = cell_a
            end       = cell_b
        else:
            crossing  = None
            case      = None
            start     = None
            end       = None

        return crossing,case,start,end
    
# =====================================================
class PathValues:
    """
        A class that returns attributes along a given path intersecting the environmental/vessel mesh.

        Attributes:
            path_requested_variables (dict) - Dictionary of the required path variables and the processing method
                                            e.g.{'distance':{'processing':'cumsum'},
                                                'traveltime':{'processing':'cumsum'},
                                                'datetime':{'processing':'cumsum'},
                                                'cell_index':{'processing':None},
                                                'fuel':{'processing':'cumsum'}}

            unit_shipspeed (string) - unit speed type. This is a string of type: 'km/hr','knots'
            unit_time (string) - unit time format. This is a string of type: 'days','hr','min','s

        Functions:
            objective_function - For a list of adjacent cell pairs, start and end waypoints compute path attributes
    """
    def __init__(self, path_vars):
        """
        Args:
            path_vars: The path variables specified in the route config
        """
        # Determining the important variables to return for the paths
        required_path_variables = {'distance':{'processing':'cumsum'},
                                   'traveltime':{'processing':'cumsum'},
                                   'speed':{'processing':None},
                                   'datetime':{'processing':'cumsum'},
                                   'cell_index':{'processing':None}
                                   }
        for var in path_vars:
            required_path_variables[var] = {'processing':'cumsum'}
        self.path_requested_variables = required_path_variables

        self.unit_shipspeed='km/hr'
        self.unit_time='days'


    def _waypoint_correction(self, path_requested_variables, source_graph, Wp, Cp):
        """
            Applies an in-cell correction to a path segments to determine 'path_requested_variables'
            defined by the use (e.g. total distance, total traveltime, total fuel usage)

            Input:
                path_requested_variable (dict) - A dictionary of the path requested variables
                source_graph (dict) - Dictionary of the cell in which the vessel is transiting
                Wp (tuple) - Start Waypoint location (long,lat)
                Cp (tuple) - End Waypoint location (long,lat)

            Returns:
                segment_values (dict) - Dictionary of the segment value information
                case (int) - Adjacency case type connecting the two points
        """
        # Determine the travel-time and distance between start and end waypoint given
        #environmental forcing variables
        m_long  = 111.321*1000
        m_lat   = 111.386*1000
        x = dist_around_globe(Cp[0], Wp[0]) * m_long * np.cos(Wp[1] * (np.pi / 180))
        y = (Cp[1]-Wp[1]) * m_lat
        case = case_from_angle(Cp, Wp)
        Su  = source_graph['Vector_x']
        Sv  = source_graph['Vector_y']
        Ssp = unit_speed(source_graph['speed'][case], self.unit_shipspeed)
        traveltime = traveltime_in_cell(x, y, Su, Sv, Ssp)
        traveltime = unit_time(traveltime, self.unit_time)
        distance = rhumb_line_distance(Wp, Cp)

        # Given the traveltime and distance between the two waypoints
        # determine the path related variables (e.g. fuel usage, traveltime)
        segment_values = {}
        for var in path_requested_variables.keys():
            if var=='distance':
                segment_values[var] = distance
            elif var=='traveltime':
                segment_values[var] = traveltime
            elif var=='cell_index':
                segment_values[var] = int(source_graph['id'])
            else:
                if var in source_graph.keys():
                    # Determining the objective value information. Apply an inplace
                    # metric along the path e.g. cumulative sum of values
                    if type(source_graph[var]) == list:
                        objective_rate_value = source_graph[var][case]
                    else:
                        objective_rate_value = source_graph[var]
                    if path_requested_variables[var]['processing'] is None:
                        segment_values[var] = objective_rate_value
                    else:
                        segment_values[var] = traveltime * objective_rate_value

        return segment_values, case

    def objective_function(self, adjacent_pairs, start_waypoint, end_waypoint):
        """
            Given a  list of adjacent pairs determine the path related information
            apply waypoint_correction to get path related information along the path

            Inputs:
                adjacent_pairs (list of type find_edge) - A list of the adjacent cell pairs in the form of find_edge
                start_waypoint (tuple) - Start waypoint (long,lat)
                end_waypoint (tuple) - End waypoint (long,lat)
        """
        # Initialising zero arrays for the path variables 
        variables =  {}    
        for var in self.path_requested_variables:
            variables[var] ={}
            variables[var]['path_values'] = np.zeros(len(adjacent_pairs)+2)

        # Path point
        path_points = [start_waypoint]

        # Looping over the path and determining the variable information
        for ii in range(len(adjacent_pairs)+1):
            if ii == 0:
                Wp = start_waypoint
                Cp = adjacent_pairs[ii].crossing
                cellbox = adjacent_pairs[ii].start
            elif ii == (len(adjacent_pairs)):
                Wp = adjacent_pairs[ii-1].crossing
                Cp = end_waypoint
                cellbox = adjacent_pairs[ii-1].end
            else:
                Wp = adjacent_pairs[ii-1].crossing
                Cp = adjacent_pairs[ii].crossing
                cellbox = adjacent_pairs[ii].start

            # Adding End point
            path_points += [Cp]

            # Determining the value for the variable for the segment of the path and the corresponding case
            segment_variable, segment_case = self._waypoint_correction(self.path_requested_variables, cellbox, Wp, Cp)

            # Adding that value for the segment along the paths
            for var in segment_variable:
                if type(segment_variable[var]) == np.ndarray:
                    variables[var]['path_values'][ii+1] = segment_variable[var][segment_case]
                else:
                    variables[var]['path_values'][ii+1] = segment_variable[var]

        # Applying processing to all path values
        for var in variables.keys():
            processing_type = self.path_requested_variables[var]['processing']
            if type(processing_type) == type(None):
                continue
            elif processing_type == 'cumsum':
                variables[var]['path_values'] = np.cumsum(variables[var]['path_values'])

        path_info = dict()
        path_info['path']      = np.array(path_points)
        path_info['variables'] = variables

        return path_info

#======================================================
class Smoothing:
    def __init__(self, dijkstra_graph, adjacent_pairs, start_waypoint, end_waypoint, blocked_metric='SIC',
                 max_iterations=2000, blocked_sic=10.0, merge_separation=1e-3, converged_sep=1e-3):
        """
            Class construct that has all the operations required for path smoothing. Including: Relationship of adjacent pairs,
            edge finding new edges to add and returns a list of the adjacent pairs for the constructed path


            Args:
                dijkstra_graph (dict)           - Dictionary comprising all the environmental mesh information and dijkstra graph information.
                                                  This includes but is not limited to: path crossing points to cell centre, neighbour crossing


                adjacent_pairs (list,find_edge) - An initial list of adjacent cell pairs as 'find_edge' objects comprising: .start, the start cell environmental mesh dictionary;
                                                  .end, the end environmental cell information; .crossing, a tuple of the crossing point on the edge (long,lat); and,
                                                  .case, the adjacent cell case between the two cell boxes.

                start_waypoint (tuple)          - Start Waypoint (long,lat)
                end_waypoint (tuple)            - End Waypoint (long,lat)

        """
        self.dijkstra_graph = dijkstra_graph
        self.aps = adjacent_pairs
        self.start_waypoint = start_waypoint
        self.end_waypoint = end_waypoint
        self.blocked_metric = blocked_metric
        self.max_iterations = max_iterations
        self.blocked_sic    = blocked_sic
        self.merge_separation = merge_separation
        self.converged_sep    = converged_sep
        self._g = pyproj.Geod(ellps='WGS84')

        for key in self.dijkstra_graph.keys():
            cell = self.dijkstra_graph[key]
            if len(cell['neighbourTravelLegs']) > 0:
                accessible_edges = np.where(np.isfinite(np.sum(cell['neighbourTravelLegs'], axis=1)))[0]
                cell['case'] = cell['case'][accessible_edges]
                cell['neighbourIndex'] = cell['neighbourIndex'][accessible_edges]
                cell['neighbourCrossingPoints'] = cell['neighbourCrossingPoints'][accessible_edges]
                cell['neighbourTravelLegs'] = cell['neighbourTravelLegs'][accessible_edges]

                self.dijkstra_graph[key] = cell

    def _long_case(self, start, end, case, Sp, Cp, Np):
        """
            Longitude based smoothing updating the crossing point given the conditions
            of the adjacency pair

            Input:
                start (dict) - Dictionary of the start cell information
                end (dict)   - Dictionary of the end  cell information
                case (int)   - Adjacency case type connecting the two cells
                Sp (tuple)   - Start Point (long,lat)
                Cp (tuple)   - Crossing Point (long,lat)
                Np (tuple)   - End Point (long,lat)

            Returns:
                Cp (tuple)   - Updated Crossing Point (long,lat)
        """
        def newton_optimisation_long(f, y0, x, a, Y, u1, v1, u2, v2, speed_s, speed_e, R, λ_s, φ_r):
            """
                Apply newton optimisation to determine an update to the crossing point.

                All information must be considered in tandem to scientific publication
                https://arxiv.org/pdf/2209.02389

                Args:
                    y0  (float)      - Current Crossing point as a parallel distance along crossing boundary from start cell centre to crossing point
                    x  (float)       - Perpendicular distance from first-point to crossing boundary
                    a  (float)       - Perpendicular distance from crossing boundary to end-point
                    Y  (float)       - Parallel distance, along crossing boundary, between start-point and end-point
                    u1 (float)       - Start Cell perpendicular to crossing boundary forcing component
                    v1 (float)       - Start Cell parallel to crossing boundary forcing component
                    u2 (float)       - End Cell perpendicular to crossing boundary forcing component
                    v2 (float)       - End Cell parallel to crossing boundary forcing component
                    speed_s (float)  - Start Cell max speed
                    speed_e (float)  - End Cell max speed
                    R (float)        - Radius of the Earth
                    λ_s (float)      - Start point latitude in radians
                    φ_r (float)      - End point latitude in radians

                Outputs:
                    y0  (float)      - Updated Crossing point as a parallel distance along crossing boundary from start cell centre to crossing point

            """
            try_num = 1
            iter_number = 0
            improving = True
            _epsilon = 1e-4
            while improving:
                F, dF, X1, X2  = f(y0,x,a,Y,u1,v1,u2,v2,speed_s,speed_e,R,λ_s,φ_r)
                if (F==0) or (dF==0):
                    dY = 0
                else:
                    dY = (F/dF)
                if iter_number != 0:
                    improving =  (abs(dY) > _epsilon) or (abs(dY) > _epsilon*(X1*X2)
                                                          and (abs(dY)/iter_number) > _epsilon)
                else:
                    improving = True
                y0  -= dY
                iter_number += 1

                if iter_number > 100 and try_num == 1:
                    y0 = Y*x/(x+a)
                    try_num += 1
                if (iter_number > 200) and 2 <= try_num < 10:
                    try_num += 1
                    iter_number -= 100
                    if Y < 0:
                        if v2 > v1:
                            y0 = (try_num - 2) * Y
                        else:
                            y0 = (try_num - 3) * -Y
                    else:
                        if v2 < v1:
                            y0 = (try_num - 2) * Y
                        else:
                            y0 = (try_num - 3) * -Y
                if iter_number > 1000:
                    raise Exception('Newton Curve Issue - Longitude Case')
            return y0

        def _F(y,x,a,Y,u1,v1,u2,v2,speed_s,speed_e,R,λ_s,φ_r):
            """
                Determining Newton Function and differential of newton function from the longitude crossing point optimisation

                Args:
                    y  (float)      - Current Crossing point as a parallel distance along crossing boundary from start cell centre to crossing point
                    x  (float)       - Perpendicular distance from first-point to crossing boundary
                    a  (float)       - Perpendicular distance from crossing boundary to end-point
                    Y  (float)       - Parallel distance, along crossing boundary, between start-point and end-point
                    u1 (float)       - Start Cell perpendicular to crossing boundary forcing component
                    v1 (float)       - Start Cell parallel to crossing boundary forcing component
                    u2 (float)       - End Cell perpendicular to crossing boundary forcing component
                    v2 (float)       - End Cell parallel to crossing boundary forcing component
                    speed_s (float)  - Start Cell max speed
                    speed_e (float)  - End Cell max speed
                    R (float)        - Radius of the Earth
                    λ_s (float)      - Start point latitude in radians
                    φ_r (float)      - End point latitude in radians

                Outputs:
                    F (float)  - Newton function value
                    dF (float) - Differential of Newton function value
                    X1 (float) - Characteristic X distance in start cell
                    X2 (float) - Characteristic X distance in end cell

            """
            ρ = (λ_s+φ_r)/2.0
            ϕ_min = min(λ_s,φ_r) 
            if λ_s > φ_r:
                ϕ_l   = ρ
                ϕ_r   = (ϕ_min+ρ)/2
            else:
                ϕ_l   = (ϕ_min+ρ)/2
                ϕ_r   = ρ

            θ  = (y/(2*R) + ϕ_l) #(y/R + λ_s)
            zl = x*np.cos(θ)
            ψ  = ((Y-y)/(2*R) + ϕ_r) #((Y-y)/R + φ_r)
            zr = a*np.cos(ψ)

            C1  = speed_s**2 - u1**2 - v1**2
            C2  = speed_e**2 - u2**2 - v2**2
            D1  = zl*u1 + y*v1
            D2  = zr*u2 + (Y-y)*v2
            X1  = np.sqrt(D1**2 + C1*(zl**2 + y**2))
            X2  = np.sqrt(D2**2 + C2*(zr**2 + (Y-y)**2))

            dzr = -a*np.sin(ψ)/(2*R) #-zr*np.sin(ψ)/R
            dzl = -x*np.sin(θ)/(2*R) #-zl*np.sin(θ)/R

            dD1 = dzl*u1 + v1
            dD2 = dzr*u2 - v2
            dX1 = (D1*v1 + C1*y + dzl*(D1*u1 + C1*zl))/X1
            dX2 = (-v2*D2 - C2*(Y-y) + dzr*(D2*u2 + C2*zr))/X2     

            zr_term = (zr - (X2 - D2)*u2/C2)
            zl_term = (zl - (X1 - D1)*u1/C1)

            F = (X1+X2)*y - v1*(X1-D1)*X2/C1 - (Y - v2*(X2-D2)/C2)*X1 + dzr*zr_term*X1 + dzl*zl_term*X2

            dF = (X1+X2) + y*(dX1 + dX2) - (v1/C1)*(dX2*(X1-D1) + X2*(dX1-dD1))\
                - Y*dX1 + (v2/C2)*(dX1*(X2-D2) + X1*(dX2-dD2))\
                - (zr/(4*(R**2)))*zr_term*X1\
                - (zl/(4*(R**2)))*zl_term*X2\
                + dzr*(dzr-u2*(dX2-dD2))/C2*X1\
                + dzl*(dzl-u1*(dX1-dD1))/C1*X2\
                + dzr*zr_term*dX1 + dzl*zl_term*dX2

            return F,dF,X1,X2

        Sp = list(Sp)
        Cp = list(Cp)
        Np = list(Np)

        cell_s_u = start['Vector_x']
        cell_s_v = start['Vector_y']
        cell_e_u = end['Vector_x']
        cell_e_v = end['Vector_y']
        speed_s = start['speed'][0]*(1000/(60*60))
        speed_e = end['speed'][0]*(1000/(60*60))
        Rd = 6371.*1000

        if case == 2:   
            sgn  = 1
        else:
            sgn  = -1

        λ_s  = Sp[1]*(np.pi/180)
        φ_r  = Np[1]*(np.pi/180)

        x = dist_around_globe(Cp[0], Sp[0]) * 111.321 * 1000.
        a = dist_around_globe(Np[0], Cp[0]) * 111.321 * 1000.

        Y           = (Np[1]-Sp[1])*111.386*1000.
        y0          = Y/2
        u1          = sgn*cell_s_u; v1 = cell_s_v
        u2          = sgn*cell_e_u; v2 = cell_e_v
        y           = newton_optimisation_long(_F, y0, x, a, Y, u1, v1, u2, v2, speed_s, speed_e, Rd, λ_s, φ_r)

        # Updating the crossing points
        Cp = (Cp[0],
              Sp[1] + y/(111.386*1000.))
        
        return Cp

    def _lat_case(self, start, end, case, Sp, Cp, Np):
        """
            Latitude based smoothing updating the crossing point given the conditions
            of the adjacency pair

            Input:
                start (dict) - Dictionary of the start cell information
                end (dict)   - Dictionary of the end  cell information
                case (int)   - Adjacency case type connecting the two cells
                Sp (tuple)   - Start Point (long,lat)
                Cp (tuple)   - Crossing Point (long,lat)
                Np (tuple)   - End Point (long,lat)

            Returns:
                Cp (tuple)   - Updated crossing Point (long,lat)
        """
        def newton_optimisation_lat(f, y0, x, a, Y, u1, v1, u2, v2, speed_s, speed_e, R, λ, θ, ψ):
            """
                Apply newton optimisation to determine an update to the crossing point in a latitude.

                All information must be considered in tandem to scientific publication.

                Args:
                    y0  (float)      - Current Crossing point as a parallel distance along crossing boundary from start cell centre to crossing point
                    x  (float)       - Perpendicular distance from first-point to crossing boundary
                    a  (float)       - Perpendicular distance from crossing boundary to end-point
                    Y  (float)       - Parallel distance, along crossing boundary, between start-point and end-point
                    u1 (float)       - Start Cell perpendicular to crossing boundary forcing component
                    v1 (float)       - Start Cell parallel to crossing boundary forcing component
                    u2 (float)       - End Cell perpendicular to crossing boundary forcing component
                    v2 (float)       - End Cell parallel to crossing boundary forcing component
                    speed_s (float)  - Start Cell max speed
                    speed_e (float)  - End Cell max speed
                    R (float)        - Radius of the Earth
                    λ (float)        - Start point latitude in radians
                    θ (float)        - Crossing point latitude in radians
                    φ (float)        - End point latitude in radians

                Outputs:
                    y0  (float)      - Updated Crossing point as a parallel distance along crossing boundary from start cell centre to crossing point

            """
            try_num = 1
            iter_number = 0
            improving = True
            _epsilon = 1e-4

            while improving:
                F, dF, X1, X2  = f(y0,x,a,Y,u1,v1,u2,v2,speed_s,speed_e,R,λ,θ,ψ)
                if (F == 0) or (dF == 0):
                    dY = 0
                else:
                    dY = (F/dF)
                if iter_number != 0:
                    improving =abs(dY) > 1 or (abs(dY) > _epsilon*(X1*X2) and (abs(dY)/iter_number) > _epsilon)
                else:
                    improving = True
                y0  -= dY
                iter_number += 1

                if iter_number > 100 and try_num == 1:
                    y0 = Y*x/(x+a)
                    try_num += 1
                if (iter_number > 200) and try_num == 2:
                    try_num += 1
                    if Y < 0:
                        if v2 > v1:
                            y0 = Y
                        else:
                            y0 = 0
                    else:
                        if v2 < v1:
                            y0 = Y
                        else:
                            y0 = 0
                if iter_number > 1000:
                    raise Exception('Newton Curve Issue - Latitude Case')
            return y0

        def _F(y, x, a, Y, u1, v1, u2, v2, speed_s, speed_e, R, λ, θ, ψ):
            """
                Determining Newton Function and differential of newton function from the longitude crossing point optimisation

                Args:
                    y  (float)      - Current Crossing point as a parallel distance along crossing boundary from start cell centre to crossing point
                    x  (float)       - Perpendicular distance from first-point to crossing boundary
                    a  (float)       - Perpendicular distance from crossing boundary to end-point
                    Y  (float)       - Parallel distance, along crossing boundary, between start-point and end-point
                    u1 (float)       - Start Cell perpendicular to crossing boundary forcing component
                    v1 (float)       - Start Cell parallel to crossing boundary forcing component
                    u2 (float)       - End Cell perpendicular to crossing boundary forcing component
                    v2 (float)       - End Cell parallel to crossing boundary forcing component
                    speed_s (float)  - Start Cell max speed
                    speed_e (float)  - End Cell max speed
                    R (float)        - Radius of the Earth
                    λ (float)        - Start point latitude in degrees
                    θ (float)        - Crossing point latitude in degrees
                    φ (float)        - End point latitude in degrees

                Outputs:
                    F (float)  - Newton function value
                    dF (float) - Differential of Newton function value
                    X1 (float) - Characteristic X distance in start cell
                    X2 (float) - Characteristic X distance in end cell

            """
            λ   = λ*(np.pi/180)
            ψ   = ψ*(np.pi/180)
            θ   = θ*(np.pi/180)
            #r1  = np.cos(λ)/np.cos(θ)
            r1  = np.cos((θ + 3*λ)/4)/np.cos(θ)
            #r2  = np.cos(ψ)/np.cos(θ)
            r2  = np.cos((θ + 3*ψ)/4)/np.cos(θ)

            d1  = np.sqrt(x**2 + (r1*y)**2)
            d2  = np.sqrt(a**2 + (r2*(Y-y))**2)
            C1  = speed_s**2 - u1**2 - v1**2
            C2  = speed_e**2 - u2**2 - v2**2
            D1  = x*u1 + r1*v1*y
            D2  = a*u2 + r2*v2*(Y-y)
            X1  = np.sqrt(D1**2 + C1*(d1**2))
            X2  = np.sqrt(D2**2 + C2*(d2**2)) 

            dX1 = (r1*(D1*v1 + r1*C1*y))/X1
            dX2 = (-r2*(D2*v2 + r2*C2*(Y-y)))/X2

            F = ((r2**2)*X1+(r1**2)*X2)*y - r1*v1*(X1-D1)*X2/C1 - r2*(r2*Y-v2*(X2-D2)/C2)*X1

            dF = ((r2**2)*X1 + (r1**2)*X2) + y*((r2**2)*dX1 + (r1**2)*dX2)\
                - r1*v1*((X1-D1)*dX2 + (dX1-r1*v1)*X2)/C1\
                - (r2**2)*Y*dX1\
                + r2*v2*((X2-D2)*dX1 + (dX2+r2*v2)*X1)/C2

            return F, dF, X1, X2

        Sp = list(Sp)
        Cp = list(Cp)
        Np = list(Np)

        cell_s_u = start['Vector_x']
        cell_s_v = start['Vector_y']
        cell_e_u = end['Vector_x']
        cell_e_v = end['Vector_y']

        speed_s = start['speed'][0]*(1000/(60*60))
        speed_e = end['speed'][0]*(1000/(60*60))
        Rd = 6371.*1000

        if case == 4:   
            sgn   = 1
        else:
            sgn   = -1

        λ=Sp[1]; θ=Cp[1]; ψ=Np[1]  

        x     = sgn*(Cp[1]-Sp[1])*111.386*1000.
        a     = -sgn*(Np[1]-Cp[1])*111.386*1000.
        Y     = sgn*(Np[0]-Sp[0])*111.321*1000*np.cos(Cp[1]*(np.pi/180))
        Su    = -sgn*cell_s_v; Sv = sgn*cell_s_u
        Nu    = -sgn*cell_e_v; Nv = sgn*cell_e_u
        y0    = Y/2

        y     = newton_optimisation_lat(_F,y0,x,a,Y,Su,Sv,Nu,Nv,speed_s,speed_e,Rd,λ,θ,ψ)

        Cp = (Sp[0] + sgn*y/(111.321*1000*np.cos(Cp[1]*(np.pi/180))), Cp[1])
        
        return Cp

    def newton_smooth(self, start, end, case, firstpoint, midpoint, lastpoint):
        """
            Given an adjacent cell pair that are non-diagonal determine the
            update to the crossing point/midpoint given the environmental
            conditions

            Input:
                start (dict) - Dictionary of the start cell information
                end (dict)   - Dictionary of the end  cell information
                case (int)   - Adjacency case type connecting the two cells
                firstpoint (tuple) - First Point (long,lat)
                midpoint (tuple)   - Midpoint Point (long,lat)
                lastpoint (tuple)  - Last Point (long,lat)
            Return:
                midpoint (tuple)   - Updated midpoint (long,lat)
        """
        if abs(case) == 2:
            midpoint = self._long_case(start, end, case, firstpoint, midpoint, lastpoint)
        elif abs(case) == 4:
            midpoint = self._lat_case(start, end, case, firstpoint, midpoint, lastpoint)

        return midpoint

    def remove(self, index):
        """
            Removing an adjacent cell pair

            Args:
                index - index in the adjacent cell pair list (.ap) to remove the index for
        """
        self.aps.pop(index)

    def add(self, index, ap_list):
        """
            Adding in a new adjacent cell pair

            Args:
                index   - the index to add the adjacent cell pair
                ap_list - a list of adjacent cell pair objects to add
        """
        for i in range(len(ap_list)):
            self.aps.insert(i + index, ap_list[i])
        
    def _neighbour_case(self, cell_a, cell_b, x, case):
        """
            Checks to determine if the crossing point has moved outside the domain
            connecting the two cells in the adjacency case

            Args:
                cell_a (dict) - Start cell environmental info as dictionary
                cell_b (dict) - End cell environmental info as dictionary
                x (tuple) - Updated crossing point that could lie outside the connection of the cell boxes (long,lat)
                case (int) - Adjacency case type connecting the two cells

            Output
                case_a (int or None) - Possible additional case edge relative to start cell to add, if None no edge to add
                case_b (int or None) - Possible additional case edge relative to start cell to add, if None no edge to add
        """

        # Returning if corner horseshoe case type
        if abs(case) == 1 or abs(case) == 3 or abs(case) == 0:
            return None, None
        elif abs(case) == 2:
            # Defining the min and max of the start and end cells
            smin = cell_a['cy'] - cell_a['dcy']
            smax = cell_a['cy'] + cell_a['dcy']
            emin = cell_b['cy'] - cell_b['dcy']
            emax = cell_b['cy'] + cell_b['dcy']

            # Defining the global min and max
            vmin = np.max([smin, emin])
            vmax = np.min([smax, emax])

            # Point lies on the boundary connecting up the
            #two adjacent cell pairs, the start and end cell.
            if (x[1] >= vmin) and (x[1] <= vmax):
                return None, None

            # If Start and end cells share an edge for the horseshoe
            if (x[1] < smin) and (smin == emin):
                case_a = 4
                case_b = 4
                return case_a, case_b
            if (x[1] > smax) and (smax == emax):
                case_a = -4
                case_b = -4
                return case_a, case_b

            # --- Cases where StartCell is Larger than end Cell ---
            if (x[1] > emax) and (smax > emax):
                case_a = case
                case_b = -4
                return case_a, case_b
            if (x[1] < emin) and (smin < emin):
                case_a = case
                case_b = 4
                return case_a, case_b

            # --- Cases where StartCell is smaller than end Cell ---
            if (x[1] > smax) and (smax < emax):
                case_a = -4
                case_b = -case
                return case_a,case_b
            if (x[1] < smin) and (emin < smin):
                case_a = 4
                case_b = -case
                return case_a,case_b      

        elif abs(case) == 4:
            # Defining the min and max of the start and end cells
            smin = cell_a['cx'] - cell_a['dcx']
            smax = cell_a['cx'] + cell_a['dcx']
            emin = cell_b['cx'] - cell_b['dcx']
            emax = cell_b['cx'] + cell_b['dcx']

            # Defining the global min and max
            vmin = np.max([smin, emin])
            vmax = np.min([smax, emax])

            # Point lies on the boundary connecting up the
            # two adjacent cell pairs, the start and end cell.
            if (x[0] >= vmin) and (x[0] <= vmax):
                return None, None

            # If Start and end cells share an edge for the horseshoe
            if (x[0] < smin) and (smin == emin):
                case_a = -2
                case_b = -2
                return case_a, case_b
            if (x[0] > smax) and (smax == emax):
                case_a = 2
                case_b = 2
                return case_a, case_b

            # --- Cases where StartCell is Larger than end Cell ---
            if (x[0] > emax) and (smax > emax):
                case_a = case
                case_b = 2
                return case_a, case_b
            if (x[0] < emin) and (smin < emin):
                case_a = case
                case_b = -2
                return case_a, case_b
            # --- Cases where StartCell is smaller than end Cell ---
            if (x[0] > smax) and (smax < emax):
                case_a = 2
                case_b = -case
                return case_a, case_b
            if (x[0] < smin) and (emin < smin):
                case_a = -2
                case_b = -case
                return case_a, case_b

        raise Exception('Path Smoothing - Failure - Adding additional cases unknown in neighbour_case')

    def _neighbour_indices(self, cell_a, cell_b, case, add_case_a, add_case_b):
        """
            For a given adjacency cell pair, and possible cases to add to start and end
            cell, determine the index of the new cell/cells to add into the adjacency
            list

            Args:
                cell_a (dict) - Start cell environmental info as dictionary
                cell_b (dict) - End cell environmental info as dictionary
                case (int) - Adjacency case type connecting the two cells
                add_case_a (int) - Possible additional case edge relative to start cell to add
                add_case_b (int) - Possible additional case edge relative to start cell to add

            Returns
                additional_indices (list) - A list of possible cell indices to add. None if no index added.
                additional_cases (list) - A list of the cases connecting the additional cell indices. None if no index added.
        """
        cell_a_neighbours = cell_a['neighbourIndex'][cell_a['case']==add_case_a]
        cell_b_neighbours = cell_b['neighbourIndex'][cell_b['case']==add_case_b]

        # Determining possible v-connections
        v_connections = set(cell_a_neighbours).intersection(cell_b_neighbours)
        if len(v_connections) != 0:
            if len(v_connections) == 1:
                return list(v_connections), [add_case_a, -add_case_b]

        # Determining possible u-connections
        for cell_a_neighbour in cell_a_neighbours:
            _possible_cell = self.dijkstra_graph[cell_a_neighbour]
            _connections = set(_possible_cell['neighbourIndex']).intersection(cell_b_neighbours)
            if len(_connections) == 1 and (abs(add_case_a) == abs(add_case_b)):
               u_connections = [cell_a_neighbour,list(_connections)[0]]
               return list(u_connections), [add_case_a, case, -add_case_b]

        return None, None

    def _neighbour_cells(self, cell_a, cell_b, case, add_case_a, add_case_b):
        """
            Adding in the neighbour cell information as a dict and case types of the neighbour cells that must be
            added. If the add_indices is None then this means that the case need to change relating the adjacency
            cell pair, but no additional cells need to be added

            Args:
                cell_a (dict) - Start cell environmental info as dictionary
                cell_b (dict) - End cell environmental info as dictionary
                case (int) - Adjacency case type connecting the two cells
                add_case_a (int) - Possible additional case edge relative to start cell to add
                add_case_b (int) - Possible additional case edge relative to start cell to add

            Returns
                additional_indices (list) - A list of possible cell dictionary info. None if no index added.
                additional_cases (list) - A list of the cases connecting the additional cell indices. None if no index added.
        """
        add_indices, add_cases = self._neighbour_indices(cell_a, cell_b, case, add_case_a, add_case_b)

        if add_indices is None:
            return None, add_cases
        
        else:
            return [self.dijkstra_graph[ii] for ii in add_indices], add_cases

    def nearest_neighbour(self, start, end, case, x):
        """
            Returns the cell in the mesh that shares a boundary with cellA and has an edge on the line that extends the common
            boundary of cellA and cellB (and on which the point x lies) in the direction of x.
            If x lies inside cellA or there is no cell that satisfies these requirements, it returns null.

            Args:
                start (dict) - Start cell environmental info as dictionary
                end (dict)   - End cell environmental info as dictionary
                case (int)   - Adjacency case type connecting the two cells
                x (tuple)    - Updated crossing point (long,lat)

            Returns
                additional_indices (list) - A list of possible cell dictionary info. None if no index added.
                additional_cases (list) - A list of the cases connecting the additional cell indices. None if no index added.
        """
        # Determine the neighbour cases, if any
        target_a_case, target_b_case = self._neighbour_case(start, end, x, case)
        add_indices, add_edges = self._neighbour_cells(start, end, case, target_a_case, target_b_case)

        return add_indices, add_edges

    def diagonal_case(self, case):
        """
            Function that determines if the adjacent cell pair is a diagonal case

            Args:
                case (int) - Adjacency case type connecting the two cells
            Returns
                True is diagonal case, false if not
        """
        if (abs(case) == 1) or (abs(case) == 3):
            return True
        else:
            return False

    def blocked(self, new_cell, cell_a, cell_b):
        """
            Function that determines if the new cell being introduced is worse off than the original two cells.
            Currently, this is hard encoded to not enter a cell 5% worse off in Sea-Ice-Concentration

            Args:
                new_cell (dict) - New cell to add environmental parameters as dict
                cell_a (dict)   - Start cell to add environmental parameters as dict
                cell_b (dict)   - End cell to add environmental parameters as dict

            Return:
                True if the cell cannot be entered, False if the cell can
        """
        start = cell_a['SIC']
        end   = cell_b['SIC']
        max_new = new_cell['SIC']

        percentage_diff1  = (max_new-start)*100
        percentage_diff2  = (max_new-end)*100
        if ((percentage_diff1 <= self.blocked_sic*start) or (percentage_diff2 <= self.blocked_sic*end)
                or max_new <= self.blocked_sic):
            return False
        else:
            return True

    def clip(self, cell_a, cell_b, case, x):
        """
            Given two cell boxes clip point to within the cell boxes

            Function that clips back the crossing point so that it is only on the intersection
            between the two cell boxes in the adjacent cell pair

            Args:
                cell_a (dict) - Start cell environmental info as dictionary
                cell_b (dict)   - End cell environmental info as dictionary
                case (int)   - Adjacency case type connecting the two cells
                x (tuple)    - Updated crossing point (long,lat)

            Return:
                x (tuple) - Updated crossing point clipped to cell intersection (long,lat)
        """
        if abs(case) == 2:
            # Defining the min and max of the start and end cells
            smin = cell_a['cy'] - cell_a['dcy']
            smax = cell_a['cy'] + cell_a['dcy']
            emin = cell_b['cy'] - cell_b['dcy']
            emax = cell_b['cy'] + cell_b['dcy']

            # Defining the global min and max
            vmin = np.max([smin, emin])
            vmax = np.min([smax, emax])

            x = (x[0], np.clip(x[1], vmin, vmax))

        elif abs(case) == 4:

            # Defining the min and max of the start and end cells
            smin = cell_a['cx'] - cell_a['dcx']
            smax = cell_a['cx'] + cell_a['dcx']
            emin = cell_b['cx'] - cell_b['dcx']
            emax = cell_b['cx'] + cell_b['dcx']

            # Defining the global min and max
            vmin = np.max([smin, emin])
            vmax = np.min([smax, emax])

            x = (np.clip(x[0], vmin, vmax), x[1])

        return x

    def diagonal_select_side(self, cell_a, cell_b, case, firstpoint, midpoint, lastpoint):
        """
            Assuming that midpoint is the common corner of the two cells in the diagonal edge ap. Then
            this function returns the cell that shares a boundary with both ap.start and ap.end on the same side
            of midpoint as the shorter great circle arc (using pyproj with default projection 'WGS84')
            passing between firstpoint and lastpoint.

            If that cell is not in the neighbourhood graph then this returns None

            Args:
                cell_a (dict) - Start cell environmental info as dictionary
                cell_b (dict)   - End cell environmental info as dictionary
                case (int)   - Adjacency case type connecting the two cells
                firstpoint (tuple) - First Point (long,lat)
                midpoint (tuple)   - Midpoint Point (long,lat)
                lastpoint (tuple)  - Last Point (long,lat)

            Returns
                additional_indices (list) - A list of possible cell dictionary info. None if no index added.
                additional_cases (list) - A list of the cases connecting the additional cell indices. None if no index added.
        """
        fp_lon, fp_lat = firstpoint
        mp_lon, mp_lat = midpoint
        lp_lon, lp_lat = lastpoint

        # Approximate great-circle to 50000 point and determine point with closest misfit
        _lonlats     = np.array(self._g.npts(fp_lon, fp_lat, lp_lon, lp_lat,50000))
        mp_lat_misfit = _lonlats[:,0] - mp_lon
        mp_lat_diff   = _lonlats[np.argmin(abs(mp_lat_misfit)), 1] - mp_lat

        # Straight Line Connecting points
        # mp_line = ((lp_lat-fp_lat)/(lp_lon-fp_lon))*(mp_lon-fp_lon) + fp_lat
        # mp_lat_diff = mp_line-mp_lat
       
        # Using the initial case identify the new cell to introduce. Return to after doing
        # nearest neighbour section
        if case == 1:
            if mp_lat_diff > 0:
                target_a_case = -4
                target_b_case = -2
            if mp_lat_diff < 0:
                target_a_case = 2
                target_b_case = 4
        elif case == 3:
            if mp_lat_diff > 0:
                target_a_case = 2
                target_b_case = -4
            if mp_lat_diff < 0:
                target_a_case = 4
                target_b_case = -2
        elif case == -1:
            if mp_lat_diff > 0:
                target_a_case = -2
                target_b_case = -4
            if mp_lat_diff < 0:
                target_a_case = 4
                target_b_case = 2
        elif case == -3:
            if mp_lat_diff > 0:
                target_a_case = -4
                target_b_case = 2
            if mp_lat_diff < 0:
                target_a_case = -2
                target_b_case = 4
        else:
            return None, None

        # Determining the additional cell to include
        add_indices, add_edges = self._neighbour_cells(cell_a, cell_b, case, target_a_case, target_b_case)

        return add_indices, add_edges

    def dist(self, start_point, end_point):
        """
            Determining the absolute distance between two points using pyproj and the
            reference project (default: WGS84)

            Inputs:
                start_point (tuple) - Start Point (long,lat)
                end_point (tuple) - End Point (long,lat)
            Outputs:
                distance (float) - Distance between the two points in km

        """
        sp_lon, sp_lat = start_point
        ep_lon, ep_lat = end_point

        distance = np.sqrt(((sp_lon-ep_lon)*111.321)**2 + ((sp_lat-ep_lat)*111.386)**2)

        # azimuth1, azimuth2, distance = self._g.inv(sp_lon, sp_lat, ep_lon, ep_lat)
        return distance

    def previous_vs(self, edge_a, edge_b, midpoint_prime):
        """
            For a V-additional case determine if we have already seen this edge added in the
            same situation. If a common past has been seen return True, otherwise add this
            v-additional case to a global list and return False

            Args:
                edge_a (find_edge)     - First-edge connecting start cell to new cell
                edge_b (find_edge)     - First-edge connecting new cell to end cell
                midpoint_prime (tuple) - midpoint that triggered the v-additional case addition (long,lat)

            Return:
                True if this v-additional case has been seen before, or false if not

        """
        edge_a_start_index = edge_a.start['id']
        edge_b_start_index = edge_b.start['id']
        edge_a_end_index   = edge_a.end['id']
        edge_b_end_index   = edge_b.end['id']

        current_v = [edge_a_start_index, edge_a_end_index,
                     edge_b_start_index, edge_b_end_index,
                     midpoint_prime]

        if len(self.previous_vs_info) == 0:
            self.previous_vs_info += [current_v]
            return False

        if current_v[:-1] in np.array(self.previous_vs_info, dtype=object)[:,:-1].tolist():
            previous_vs_info_np = np.array(self.previous_vs_info, dtype=object)
            similar_midpoint_primes = previous_vs_info_np[(previous_vs_info_np[:,:-1] == current_v[:-1]).all(axis=1),-1]

            if np.min([self.dist(c, current_v[-1]) for c in similar_midpoint_primes]) <= self.merge_separation:
                return True
            else:
                self.previous_vs_info += [current_v]
                return False

        else:
            self.previous_vs_info += [current_v]
            return False

    def previous_us(self, edge_a, edge_b, edge_c, midpoint_prime):
        """
            For a U-additional case determine if we have already seen these edges added in the
            same situation and the same crossing point. If a common past has been seen return True,
            otherwise add this U-additional case to a global list and return False

            Input:
                edge_a (find_edge)     - First-edge connecting start cell to new cell 1
                edge_b (find_edge)     - First-edge connecting new cell 1 to new cell 2
                edge_c (find_edge)     - First-edge connecting new cell 2 to end cell
                midpoint_prime (tuple) - midpoint that triggered the u-additional case addition (long,lat)

            Return:
                True if this U-additional case has been seen before, or false if not

        """
        edge_a_start_index = edge_a.start['id']
        edge_b_start_index = edge_b.start['id']
        edge_c_start_index = edge_c.start['id']
        edge_a_end_index   = edge_a.end['id']
        edge_b_end_index   = edge_b.end['id']
        edge_c_end_index   = edge_c.end['id']

        current_u = [edge_a_start_index, edge_a_end_index,
                     edge_b_start_index, edge_b_end_index,
                     edge_c_start_index, edge_c_end_index,
                     midpoint_prime]

        if len(self.previous_us_info) == 0:
            self.previous_us_info += [current_u]
            return False

        if current_u[:-1] in np.array(self.previous_us_info, dtype=object)[:,:-1].tolist():
            previous_us_info_np = np.array(self.previous_us_info, dtype=object)
            similar_midpoint_primes = previous_us_info_np[(previous_us_info_np[:,:-1] == current_u[:-1]).all(axis=1),-1]

            if np.min([self.dist(c, current_u[-1]) for c in similar_midpoint_primes]) <= self.merge_separation:
                return True
            else:
                self.previous_us_info += [current_u]
                return False

        else:
            self.previous_us_info += [current_u]
            return False
        
    def previous_diagonals(self, edge_a, edge_b, firstpoint, lastpoint):
        """
            For a diagonal-additional case determine if we have already seen these edges added in the
            same situation and the same first and last points. If a common past has been seen return True,
            otherwise add an additional case to of the diagonal to the global list and return False

            Input:
                edge_a (find_edge)     - First-edge connecting start cell to new cell
                edge_b (find_edge)     - First-edge connecting new cell  to end cell
                firstpoint (tuple)     - firstpoint in the adjacent cell triplet of points (long,lat)
                lastpoint (tuple)      - lastpoint in the adjacent cell triplet of points (long,lat)

            Return:
                True if this diagonal case has been seen before, or false if not
        """

        edge_a_start_index = edge_a.start['id']
        edge_b_start_index = edge_b.start['id']
        edge_a_end_index   = edge_a.end['id']
        edge_b_end_index   = edge_b.end['id']

        current_diagonal = [edge_a_start_index, edge_a_end_index,
                     edge_b_start_index, edge_b_end_index,
                     firstpoint, lastpoint]
        

        if len(self.previous_diagonal_info) == 0:
            self.previous_diagonal_info += [current_diagonal]
            return False

        if current_diagonal[:-2] in np.array(self.previous_diagonal_info, dtype=object)[:,:-2].tolist():
            previous_diagonal_info_np = np.array(self.previous_diagonal_info, dtype=object)
            similar_start_end = previous_diagonal_info_np[(previous_diagonal_info_np[:,:-2] == current_diagonal[:-2]).all(axis=1),-2:]
            if np.any([self.dist(c[0], current_diagonal[-2]) <= self.merge_separation and
                       self.dist(c[1],current_diagonal[-1]) <= self.merge_separation for c in similar_start_end]):
                return True
            else:
                self.previous_diagonal_info += [current_diagonal]
                return False

        else:
            self.previous_diagonal_info += [current_diagonal]
            return False
                
    def forward(self):
        """
            Applies inplace this function conducts a forward pass over the adjacent cell pairs, updating the crossing
            points between adjacent cell pairs for the given environmental conditions and great-circle characteristics.
            This is applied as a forward pass across the path moving out in adjacent cell pairs (triplets of crossing
            points with the cell adjacency).

            Key features of forward pass include
                reverse edges - Removal of adjacent cell edges that enter and exit a cell on subsequent
                                iterations. e.g. routes going back on themselves
                merging     - When two crossing points are close, merge points and determine new
                                common edge between start and end point
                diagonal case - If the middle point is a diagonal edge between cells, introduce a new cell box
                                dependent on start and end crossing points. If cell is inaccessible 'blocked' then
                                remain on corner for a later iteration.

                                If exact diagonal, with same start and end crossing point, has be seen before
                                then skip.

                newton smooth - If adjacency is not diagonal then smooth the midpoint crossing point on the boundary given a
                                horizontal or vertical smoothing. Returns a new midpoint that can either lie on the boundary
                                between the two cells or outside the boundary

                                If lies on the boundary then check if similar to previous seen case of this crossing point
                                else continue and not converged

                v shaped add  - If the crossing point lies outside the boundary in the newton smoothing stage the addition
                                cell/cells must be included.

                                Determine the new edges that need to be included if only a single cell (two edges) then do
                                a v-shaped addition. If blocked then trim back. If exact v-shaped seen before, with same
                                midpoint prime and possible edge additions, then skip. If blocked or seen before and crossing
                                point hasn't changed within converge_sep then the crossing point has converged

                u shaped add - Identical to v-shaped add but now with the addition of 2 cells (3 edges). If blocked then trim back. If exact v-shaped seen before, with same
                                midpoint prime and possible edge additions, then skip. If blocked or seen before and crossing
                                point hasn't changed within converge_sep then the crossing point has converged.

            This code should be read relative to the pseudocode outlined in the paper.
            https://arxiv.org/pdf/2209.02389

        """
        self.jj = 0
        self.previous_aps = []
        converged = False
        self.all_aps = []
        self.previous_vs_info = []
        self.previous_us_info = []
        self.previous_diagonal_info = []
        while not converged:
            # Early stopping criterion
            if self.jj == self.max_iterations:
                break

            path_length = len(self.aps)
            firstpoint = self.start_waypoint
            midpoint   = None 
            lastpoint  = None
            converged  = True

            ii = 0
            self.jj += 1
            while ii < path_length:
                ap       = self.aps[ii]
                midpoint = ap.crossing

                # Determine the next adjacency pair and the last point
                if ii+1 < path_length:
                    app = self.aps[ii+1]
                    lastpoint = app.crossing
                else:
                    app = self.aps[ii]
                    lastpoint  = self.end_waypoint

                # Remove the reverse edges
                if ap.start['id'] == app.end['id']:
                    self.remove(ii)
                    self.remove(ii)
                    path_length -= 2
                    converged = False
                    continue

                # Merging the first and last point close move to next iteration
                if self.dist(firstpoint, midpoint) < self.merge_separation:
                    firstpoint = midpoint
                    ii += 1
                    continue

                # Merging the mid and last point if separation close, determine new edge for adjacency
                if self.dist(midpoint, lastpoint) < self.merge_separation:
                    start_cell  = ap.start
                    end_cell    = app.end

                    common_cell = np.where(np.array(start_cell['neighbourIndex']) == end_cell['id'])[0]
                    if len(common_cell) == 1:
                        _merge_case = start_cell['case'][np.where(np.array(start_cell['neighbourIndex']) == end_cell['id'])[0][0]]
                        new_edge = FindEdge(start_cell, end_cell, _merge_case)
                        self.remove(ii) #Removing ap
                        self.remove(ii) #Removing app
                        self.add(ii,[new_edge])
                        path_length -= 1
                        converged = False
                        continue

                # Relationship is a diagonal case
                if self.diagonal_case(ap.case):
                    add_indices, add_cases = self.diagonal_select_side(ap.start, ap.end, ap.case, firstpoint, midpoint,
                                                                       lastpoint)
                    if add_indices is None:
                        ii += 1
                        firstpoint = midpoint
                        continue

                    if len(add_indices) == 1:
                        target = add_indices[0]
                        case_a = add_cases[0]
                        case_b = add_cases[1]
                        if self.blocked(target, ap.start, ap.end):
                            ii += 1
                            firstpoint = midpoint
                            continue
                        else:
                            edge_a = FindEdge(ap.start, target, case_a)
                            edge_b = FindEdge(target, ap.end, case_b)
                            if self.previous_diagonals(edge_a, edge_b, firstpoint, lastpoint):
                                ii += 1
                                firstpoint = midpoint
                                continue
                            self.remove(ii)
                            self.add(ii,[edge_a, edge_b])
                            path_length += 1
                            converged = False
                            continue
                        
                # Updating crossing point
                midpoint_prime = self.newton_smooth(ap.start, ap.end, ap.case, firstpoint, midpoint, lastpoint)
                if type(midpoint_prime) == type(None) or np.isnan(midpoint_prime[0]) or np.isnan(midpoint_prime[1]):
                    raise Exception('Newton call failed to converge or recover')

                # Determining if additional cases need to be added
                add_indices, add_cases = self.nearest_neighbour(ap.start, ap.end, ap.case, midpoint_prime)

                # No additional cells to add
                if add_indices is None:
                    midpoint_prime = self.clip(ap.start, ap.end, ap.case, midpoint_prime)
                    if self.dist(midpoint, midpoint_prime) > self.converged_sep:
                        converged = False
                    self.aps[ii].crossing = midpoint_prime
                    ii += 1
                    firstpoint = midpoint_prime
                    continue

                # Introduction of a v-shape
                if len(add_indices) == 1:
                        target = add_indices[0]
                        case_a = add_cases[0]
                        case_b = add_cases[1]
                        if self.blocked(target, ap.start, ap.end):
                            midpoint_prime = self.clip(ap.start, ap.end, ap.case, midpoint_prime)
                            if self.dist(midpoint, midpoint_prime) > self.converged_sep:
                                converged = False
                            self.aps[ii].crossing = midpoint_prime
                            ii += 1
                            firstpoint = midpoint_prime
                        else:
                            edge_a = FindEdge(ap.start, target, case_a)
                            edge_b = FindEdge(target, ap.end, case_b)

                            if self.previous_vs(edge_a, edge_b, midpoint_prime):
                                midpoint_prime = self.clip(ap.start, ap.end, ap.case, midpoint_prime)
                                if self.dist(midpoint, midpoint_prime) > self.converged_sep:
                                    converged = False
                                self.aps[ii].crossing = midpoint_prime
                                ii += 1
                                firstpoint = midpoint_prime
                            else:
                                self.remove(ii)
                                self.add(ii,[edge_a, edge_b])
                                path_length += 1
                                converged = False

                # Introduction of a U-shape
                if len(add_indices) == 2:
                        logging.debug('--- Adding in U-shape ---')
                        target_a = add_indices[0]
                        target_b = add_indices[1]
                        case_a = add_cases[0]
                        case_b = add_cases[1]
                        case_c = add_cases[2]

                        if (not self.blocked(target_a, ap.start, ap.end) and
                                not self.blocked(target_b, ap.start, ap.end)):
                            edge_a = FindEdge(ap.start, target_a, case_a)
                            edge_b = FindEdge(target_a, target_b, case_b)
                            edge_c = FindEdge(target_b, ap.end, case_c)

                            if self.previous_us(edge_a, edge_b, edge_c, midpoint_prime):
                                midpoint_prime = self.clip(ap.start, ap.end, ap.case, midpoint_prime)
                                if self.dist(midpoint, midpoint_prime) > self.converged_sep:
                                    converged = False
                                self.aps[ii].crossing = midpoint_prime
                                ii += 1
                                firstpoint = midpoint_prime
                            else:
                                self.remove(ii)
                                self.add(ii,[edge_a, edge_b, edge_c])
                                path_length += 2
                                converged = False

                        else:
                            midpoint_prime = self.clip(ap.start, ap.end, ap.case, midpoint_prime)
                            if self.dist(midpoint, midpoint_prime) > self.converged_sep:
                                converged = False
                            self.aps[ii].crossing = midpoint_prime
                            ii += 1
                            firstpoint = midpoint_prime

