import copy
import pandas as pd
import numpy as np
import pyproj



class find_edge:
    def __init__(self,cell_a,cell_b,case):
        self.crossing,self.case,self.start,self.end = self._find_edge(cell_a,cell_b,case)

    def _find_edge(self,cell_a,cell_b,case):
        '''
            Function that returns the edge connecting to cells, cell_a and cell_b. If there is no edge 
            connecting the two then it returns None

            Input:
                cell_a : 
                cell_b : 
        '''

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
    

class Smoothing:
    def __init__(self,dijkstra_graph,dijkstra_route,config=None):
        '''
            Class construct that has all the operations requried for path smoothing. Including: Relationship of adjacent pairs,
            edge finding ..

        '''
        self._initialise_config(config)
        self._initialise_dijkstra_graph(dijkstra_graph)
        self._initialise_dijkstra_route(dijkstra_route)



    def _initialise_config(self,config):
        '''    
            Initialising configuration information. If None return a list of standards
        '''

        self.merge_separation = 1e-3
        self.converged_sep = 1e-3

        self._g = pyproj.Geod(ellps='WGS84')

    def _initialise_dijkstra_graph(self,dijkstra_graph):
        '''
            Initialising dijkstra graph information into a standard form
        '''

        dijkstra_graph_dict = {}
        for idx,cell in dijkstra_graph.iterrows():
            dijkstra_graph_dict[cell.name] = {}
            dijkstra_graph_dict[cell.name]['id'] = cell.name
            for key in cell.keys():
                entry = cell[key]
                if type(entry) == list:
                    entry = np.array(entry)
                dijkstra_graph_dict[cell.name][key] = entry
        self.dijkstra_graph = dijkstra_graph_dict
        

    def _initialise_dijkstra_route(self,dijkstra_route):
        '''
            Initialising dijkstra route into a standard path form
        '''

        org_path_points = np.array(dijkstra_route['geometry']['coordinates'])
        org_cellindices = np.array(dijkstra_route['properties']['CellIndices'])
        org_cellcases= np.array(dijkstra_route['properties']['cases'])

        # -- Generating a dataframe of the case information -- 
        Points      = np.concatenate([org_path_points[0,:][None,:],org_path_points[1:-1:2],org_path_points[-1,:][None,:]])
        cellIndices = np.concatenate([[org_cellindices[0]],[org_cellindices[0]],org_cellindices[1:-1:2],[org_cellindices[-1]],[org_cellindices[-1]]])
        cellcases = np.concatenate([[org_cellcases[0]],[org_cellcases[0]],org_cellcases[1:-1:2],[org_cellcases[-1]],[org_cellcases[-1]]])

        cellDijk    = [self.dijkstra_graph[ii] for ii in cellIndices]
        cells  = cellDijk[1:-1]
        cases  = cellcases[1:-1]
        aps = []
        for ii in range(len(cells)-1):
            aps += [find_edge(cells[ii],cells[ii+1],cases[ii+1])]

        # #-- Setting some backend information
        self.aps = aps
        self.start_waypoint = Points[0,:]
        self.end_waypoint   = Points[-1,:]



        # self.aps = aps[-29:]
        # self.start_waypoint = self.aps[1].crossing
        # self.end_waypoint   = self.aps[-1].crossing

        # self.aps = self.aps

    def _long_case(self,start,end,case,Sp,Cp,Np):
        '''
            Longitude based smoothing
        '''
        def NewtonOptimisationLong(f,y0,x,a,Y,u1,v1,u2,v2,speed_s,speed_e,R,λ_s,φ_r):
                tryNum=1
                iter=0
                improving=True
                _epsilon = 1e-4
                while improving:  
                    F,dF,X1,X2  = f(y0,x,a,Y,u1,v1,u2,v2,speed_s,speed_e,R,λ_s,φ_r)
                    if (F==0) or (dF==0):
                        dY = 0
                    else:
                        dY = (F/dF)
                    improving =  (abs(dY)>_epsilon) or (abs(dY) > _epsilon*(X1*X2) and (abs(dY)/iter) > _epsilon)
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
                        raise Exception('Newton Curve Issue - Longitude Case')
                return y0

        def _F(y,x,a,Y,u1,v1,u2,v2,speed_s,speed_e,R,λ_s,φ_r):
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

            dzr = -a*np.sin(ψ)/(2*R)#-zr*np.sin(ψ)/R
            dzl = -x*np.sin(θ)/(2*R)#-zl*np.sin(θ)/R

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

        x           = sgn*(Cp[0] - Sp[0])*111.321*1000.
        a           = sgn*(Np[0] - Cp[0])*111.321*1000.
        Y           = (Np[1]-Sp[1])*111.386*1000.
        y0          = Y/2
        u1          = sgn*cell_s_u; v1 = cell_s_v
        u2          = sgn*cell_e_u; v2 = cell_e_v
        y           = NewtonOptimisationLong(_F,y0,x,a,Y,u1,v1,u2,v2,speed_s,speed_e,Rd,λ_s,φ_r)

        # Updating the crossing points
        Cp = (Cp[0],
              Sp[1] + y/(111.386*1000.))
        
        return Cp


    def _lat_case(self,start,end,case,Sp,Cp,Np):
        '''
            Latitude based smoothing
        '''
        def NewtonOptimisationLat(f,y0,x,a,Y,u1,v1,u2,v2,speed_s,speed_e,R,λ,θ,ψ):
                tryNum=1
                iter=0
                improving=True
                _epsilon      = 1e-4
                
                while improving:  
                    F,dF,X1,X2  = f(y0,x,a,Y,u1,v1,u2,v2,speed_s,speed_e,R,λ,θ,ψ)
                    if (F==0) or (dF==0):
                        dY = 0
                    else:
                        dY = (F/dF)
                    improving =abs(dY) > 1 or (abs(dY) > _epsilon*(X1*X2) and (abs(dY)/iter) > _epsilon)
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
                        raise Exception('Newton Curve Issue - Latitude Case')
                return y0

        def _F(y,x,a,Y,u1,v1,u2,v2,speed_s,speed_e,R,λ,θ,ψ):
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

            # Cassing Warning message - FIX !!
            dX2 = (-r2*(D2*v2 + r2*C2*(Y-y)))/X2

            F = ((r2**2)*X1+(r1**2)*X2)*y - r1*v1*(X1-D1)*X2/C1 - r2*(r2*Y-v2*(X2-D2)/C2)*X1

            dF = ((r2**2)*X1 + (r1**2)*X2) + y*((r2**2)*dX1 + (r1**2)*dX2)\
                - r1*v1*((X1-D1)*dX2 + (dX1-r1*v1)*X2)/C1\
                - (r2**2)*Y*dX1\
                + r2*v2*((X2-D2)*dX1 + (dX2+r2*v2)*X1)/C2

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

        y     = NewtonOptimisationLat(_F,y0,x,a,Y,Su,Sv,Nu,Nv,speed_s,speed_e,Rd,λ,θ,ψ)

        Cp = (Sp[0] + sgn*y/(111.321*1000*np.cos(Cp[1]*(np.pi/180))),
              Cp[1])
        
        return Cp

    def newton_smooth(self,start,end,case,firstpoint,midpoint,lastpoint):      

        if abs(case)==2:
            midpoint = self._long_case(start,end,case,firstpoint,midpoint,lastpoint)
        elif abs(case)==4:
            midpoint = self._lat_case(start,end,case,firstpoint,midpoint,lastpoint)

        return midpoint


    def remove(self,index):
        '''
            Removing a adjacent cell pair

            INPUT:
                index - index in the adjacent cell pair list (.ap) to romove the index for
        '''

        self.aps.pop(index)

    def add(self,index,ap_list):
        '''
            Adding in a new adjacent cell pair

            INPUT
                index   - the index to add the adjacent cell pair
                ap_list - a list of adjacent cell pair objects to add 
        '''
        for i in range(len(ap_list)):
            self.aps.insert(i + index, ap_list[i])

        
    def _neighbour_case(self,cell_a,cell_b,x,case):
        '''
            Checks to determine if the crossing point has moved outside the domain.
        '''   

        # Returning if corner horseshoe case type
        if abs(case)==1 or abs(case)==3 or abs(case)==0:
            return None,None
        elif abs(case) == 2:
            # Defining the min and max of the start and end cells
            smin = cell_a['cy']-cell_a['dcy'] 
            smax = cell_a['cy']+cell_a['dcy']
            emin = cell_b['cy']-cell_b['dcy']
            emax = cell_b['cy']+cell_b['dcy']

            # Defining the global min and max
            vmin = np.max([smin,emin])
            vmax = np.min([smax,emax])

            # Point crossingpoint on boundary between the two origional cells
            if (x[1] >= vmin) and (x[1] <= vmax):
                return None,None

            # If Start and end cells share a edge for the horesshoe 
            if (x[1]<smin) and (smin==emin):
                case_a = 4
                case_b   = 4
            if (x[1]>smax) and (smax==emax):
                case_a = -4
                case_b   = -4

            # --- Cases where StartCell is Larger than end Cell ---
            if (x[1]>emax) and (smax>emax):
                case_a = case
                case_b   = (-4)                
            if (x[1]<emin) and (smin<emin):
                case_a = case
                case_b   = (4)                   

            # --- Cases where StartCell is smaller than end Cell ---
            if (x[1]>smax) and (smax<emax):
                case_a = -4
                case_b   = -case
            if (x[1]<smin) and (emin<smin):
                case_a = 4
                case_b   = -case      

        elif abs(case) == 4:
            # Defining the min and max of the start and end cells
            smin = cell_a['cx']-cell_a['dcx']
            smax = cell_a['cx']+cell_a['dcx']
            emin = cell_b['cx']-cell_b['dcx']
            emax = cell_b['cx']+cell_b['dcx']

            # Defining the global min and max
            vmin = np.max([smin,emin])
            vmax = np.min([smax,emax])

            # Point crossingpoint on boundary between the two origional cells
            if (x[0] >= vmin) and (x[0] <= vmax):
                return None,None

            # If Start and end cells share a edge for the horesshoe 
            if (x[0]<smin) and (smin==emin):
                case_a = -2
                case_b   = -2
            if (x[0]>smax) and (smax==emax):
                case_a = 2
                case_b   = 2

            # --- Cases where StartCell is Larger than end Cell ---
            if (x[0]>emax) and (smax>emax):
                case_a = case
                case_b   = (2)                
            if (x[0]<emin) and (smin<emin):
                case_a = case
                case_b   = (-2)                   

            # --- Cases where StartCell is smaller than end Cell ---
            if (x[0]>smax) and (smax<emax):
                case_a = (2)
                case_b   = -case
            if (x[0]<smin) and (emin<smin):
                case_a = (-2)
                case_b   = -case   

            

        if ('case_a' in locals()) and ('case_b' in locals()):
            return case_a,case_b
        else:
            raise Exception ('Case addition {} not specified')

    def _neighbour_indices(self,cell_a,cell_b,case,add_case_a,add_case_b):
        '''
            Apply's set theory to determine the indicies of the additional cells to add
        '''

        cell_a_neighbours = cell_a['neighbourIndex'][cell_a['case']==add_case_a]
        cell_b_neighbours = cell_b['neighbourIndex'][cell_b['case']==add_case_b]

        # # Determining if the cell_a and cell_b share a new edge that should be added
        # new_edge = set(cell_a_neighbours).intersection([cell_b['id']])
        # if len(new_edge) == 1:
        #     return None,[add_case_a]
        # new_edge = set(cell_b_neighbours).intersection([cell_a['id']])
        # if len(new_edge) == 1:
        #     return None,[-add_case_b]

        # Determining possible v-connections
        v_connections = set(cell_a_neighbours).intersection(cell_b_neighbours)
        if len(v_connections) != 0:
            if len(v_connections) == 1:
                return list(v_connections),[add_case_a,-add_case_b]


        # Determining possible u-connections
        for cell_a_neighbour in cell_a_neighbours:
            _possible_cell = self.dijkstra_graph[cell_a_neighbour]
            _connections = set(_possible_cell['neighbourIndex']).intersection(cell_b_neighbours)
            if len(_connections) == 1 and (abs(add_case_a) == abs(add_case_b)):
               u_connections = [cell_a_neighbour,list(_connections)[0]]
               return list(u_connections),[add_case_a,case,-add_case_b]





        return None,None
                

    def _neighbour_cells(self,cell_a,cell_b,case,add_case_a,add_case_b):
        '''
            Apply's set theory to determine the indicies of the additional cells to add
        '''

        add_indices,add_cases = self._neighbour_indices(cell_a,cell_b,case,add_case_a,add_case_b)

        if add_indices == None:
            return None,add_cases
        
        else:
            return [self.dijkstra_graph[ii] for ii in add_indices],add_cases

    def nearest_neighbour(self,start,end,case,x):
        '''
            Returns the cell in the mesh that shares a boundary with cellA and has an edge on the line that extends the common 
            boundary of cellA and cellB (and on which the point x lies) in the direction of x. 
            If x lies inside cellA or there is no cell that satisfies these requirements, it returns null.
        '''

        # Determine the neighbour cases if any
        target_a_case,target_b_case = self._neighbour_case(start,end,x,case)   
        add_indicies,add_edges = self._neighbour_cells(start,end,case,target_a_case,target_b_case)

        return add_indicies,add_edges  

        

    def diagonal_case(self,cell_a,cell_b,case):
        '''
            Function that determines if the adjacent cell pair is a diagonal case
        '''
        if (abs(case)==1) or (abs(case)==3):
            return True
        else:
            return False
        

    def blocked(self,new_cell,cell_a,cell_b):
        '''
            Function that determines if the new cell being introducted is worse off that the origional two cells

            Initially this is only dependent on the Sea-Ice Concentration
        
        '''
        # if type(new_cell) == type(None):
        #     return True

        # Sea-Ice Determinination
        max_org = np.max([cell_a['SIC'],cell_b['SIC']])
        try:
            max_new = new_cell['SIC']
        except:
            print(new_cell)
        
        
        # Percentage difference
        percentage_diff = (max_new-max_org)
        if percentage_diff < 10:
            return False
        else:
            return True


    def clip(self,cell_a,cell_b,case,x):
        '''
            Given two cell boxes clip point to within the cell boxes
        '''
        if abs(case) == 2:
            # Defining the min and max of the start and end cells
            smin = cell_a['cy']-cell_a['dcy'] 
            smax = cell_a['cy']+cell_a['dcy']
            emin = cell_b['cy']-cell_b['dcy']
            emax = cell_b['cy']+cell_b['dcy']

            # Defining the global min and max
            vmin = np.max([smin,emin])
            vmax = np.min([smax,emax])

            x = (x[0],
                np.clip(x[1],vmin+1e-9,vmax-1e-9))
        elif abs(case) == 4:

            # Defining the min and max of the start and end cells
            smin = cell_a['cx']-cell_a['dcx']
            smax = cell_a['cx']+cell_a['dcx']
            emin = cell_b['cx']-cell_b['dcx']
            emax = cell_b['cx']+cell_b['dcx']

            # Defining the global min and max
            vmin = np.max([smin,emin])
            vmax = np.min([smax,emax])

            x = (np.clip(x[0],vmin+1e-9,vmax-1e-9),
                    x[1])

        return x


        

    def diagonal_select_side(self,cell_a,cell_b,case,firstpoint,midpoint,lastpoint):
        ''' 
            Assuming that midpoint is the common corner of the two cells in the diagonal edge ap. Then
            this function returns the cell that shares a boundary with both ap.start and ap.end on the same side 
            of midpoint as the shorter great circle arc (using pyproj with default projection 'WGS84') 
            passing between firstpoint and lastpoint. 

            In the case that midpoint is within CONVERGESEP of the arc, then it returns null
        '''

        fp_lon,fp_lat = firstpoint
        mp_lon,mp_lat = midpoint
        lp_lon,lp_lat = lastpoint


        # Approximate great-circle to 50000 point and determine point with closest misfit
        # _lonlats     = np.array(self._g.npts(fp_lon, fp_lat, lp_lon, lp_lat,50000))
        # mp_lat_misfit = _lonlats[:,0]-mp_lon
        # mp_lat_diff   = _lonlats[np.argmin(abs(mp_lat_misfit)),1] - mp_lat


        mp_line = ((lp_lat-fp_lat)/(lp_lon-fp_lon))*(mp_lon-fp_lon) + fp_lat
        mp_lat_diff = mp_line-mp_lat

        
        # Return if difference below the coverged_sep
        if abs(mp_lat_diff) < self.converged_sep:
            return None,None
        
        # Using the initial case identify the new cell to introduce. Return to after doing
        #nearest neighbour section
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
            return None,None


        # Determining the additional cell to include
        add_indicies,add_edges = self._neighbour_cells(cell_a,cell_b,case,target_a_case,target_b_case)




        return add_indicies,add_edges

    def dist(self,start_point,end_point):
        '''
            Determining the absolute distance between two points using pyproj and the 
            reference project (default: WGS84)
        '''
        sp_lon,sp_lat = start_point
        ep_lon,ep_lat = end_point

        distance = np.sqrt(((sp_lon-ep_lon)*(111.321*1000.))**2 + ((sp_lat-ep_lat)*(111.386*1000.))**2)

        #azimuth1, azimuth2, distance = self._g.inv(sp_lon, sp_lat, ep_lon, ep_lat)
        return distance

    def inside(self):
        '''
            Determining if a point is within a cell box
        '''

    def forward(self):
        converged = False
        jj = 0
        while not converged:
            path_length = len(self.aps)

            firstpoint = self.start_waypoint
            midpoint   = None 
            lastpoint  = None

            jj+=1
            print('--Iteration=',jj)
            converged = True
            ii=0
            while ii < path_length:
                ap       = self.aps[ii]
                midpoint = ap.crossing

                if ii+1 < path_length:
                    app = self.aps[ii+1]
                    lastpoint = app.crossing
                else:
                    app = None
                    lastpoint  = self.end_waypoint
                    break

                # Removing reverse edges
                if ap.start['id'] == app.end['id']:
                    self.remove(ii)
                    self.remove(ii)
                    path_length -= 2
                    converged = False
                    continue

                # see figure 7
                if self.dist(firstpoint,midpoint) < self.merge_separation:
                    firstpoint = midpoint
                    ii += 1
                    continue



                if self.dist(midpoint,lastpoint) < self.merge_separation and self.dist(firstpoint,midpoint) != 0.0:
                    start_cell  = ap.start
                    end_cell    = app.end
                    _merge_case = start_cell['case'][np.where(np.array(start_cell['neighbourIndex']) == end_cell['id'])[0][0]]
                    new_edge = find_edge(start_cell,end_cell,_merge_case)
                    self.remove(ii) #Removing ap
                    self.remove(ii) #Removing app
                    self.add(ii,[new_edge])
                    path_length -= 1
                    converged = False
                    continue

                # == Diagonal cases == 
                if self.diagonal_case(ap.start,ap.end,ap.case):
                    add_indicies,add_cases = self.diagonal_select_side(ap.start,ap.end,ap.case,firstpoint,midpoint,lastpoint)
                    if add_indicies is None:
                        if add_cases == None:
                            ii += 1
                            firstpoint=midpoint
                            continue
                        # else:
                        #     edge = find_edge(ap.start,ap.end,add_cases[0])
                        #     self.remove(ii)
                        #     self.add(ii,[edge])
                        #     converged=False
                        #     continue
                    
                    if len(add_indicies) == 1:
                        target = add_indicies[0]
                        case_a = add_cases[0]
                        case_b = add_cases[1]
                        if self.blocked(target,ap.start,ap.end):
                            ii += 1
                            firstpoint=midpoint
                            continue
                        else:
                            print('Diagonal')
                            edge_a = find_edge(ap.start,target,case_a)
                            edge_b = find_edge(target,ap.end,case_b)
                            self.remove(ii)
                            self.add(ii,[edge_a,edge_b])
                            path_length += 1
                            ii+=2
                            firstpoint = lastpoint
                            converged = False
                            continue
                        


                midpoint_prime = self.newton_smooth(ap.start,ap.end,ap.case,firstpoint,midpoint,lastpoint)
                if type(midpoint_prime) == type(None):
                    raise Exception('Newton call failed to converge or recover')
                
                add_indicies,add_cases = self.nearest_neighbour(ap.start,ap.end,ap.case,midpoint_prime)
                if add_indicies == None:
                    midpoint_prime = self.clip(ap.start,ap.end,ap.case,midpoint_prime)
                    if self.dist(midpoint,midpoint_prime) > 10:
                        converged = False
                    ap.crossing = midpoint_prime
                    ii += 1
                    firstpoint = midpoint_prime
                    continue

                # Introduction of a v-shape
                if len(add_indicies) == 1:
                        target = add_indicies[0]
                        case_a = add_cases[0]
                        case_b = add_cases[1]
                        if self.blocked(target,ap.start,ap.end):
                            midpoint_prime = self.clip(ap.start,ap.end,ap.case,midpoint_prime)
                            ap.crossing = midpoint_prime
                            ii += 1
                            firstpoint = midpoint_prime
                            continue
                        else:
                            print('V-Shaphed')
                            edge_a = find_edge(ap.start,target,case_a)
                            edge_b = find_edge(target,ap.end,case_b)
                            self.remove(ii)
                            self.add(ii,[edge_a,edge_b])
                            path_length += 1
                            ii += 2
                            firstpoint = lastpoint
                            converged = False
                            continue


                # Introduction of a U-shape
                if len(add_indicies) == 2:
                        target_a = add_indicies[0]
                        target_b = add_indicies[1]
                        case_a = add_cases[0]
                        case_b = add_cases[1]
                        case_c = add_cases[2]

                        if not self.blocked(target_a,ap.start,ap.end) and not self.blocked(target_b,ap.start,ap.end):
                            print('U-Shaphed')
                            edge_a = find_edge(ap.start,target_a,case_a)
                            edge_b = find_edge(target_a,target_b,case_b)
                            edge_c = find_edge(target_b,ap.end,case_c)
                            self.remove(ii)
                            self.add(ii,[edge_a,edge_b,edge_c])
                            path_length += 2
                            ii += 3
                            firstpoint = lastpoint
                            converged = False

                        else:
                            midpoint_prime = self.clip(ap.start,ap.end,ap.case,midpoint_prime)
                            ap.crossing = midpoint_prime
                            ii += 1
                            firstpoint = midpoint_prime
                            continue
            if jj == 50000:
                print('Early termination at 5000 epochs')
                break
