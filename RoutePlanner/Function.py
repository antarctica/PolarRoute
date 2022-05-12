import numpy as np
import copy
import pandas as pd
import numpy as np

class NewtonianDistance:
    def __init__(self,Mesh,Sc=None,Nc=None,Sc_Speed=None,Nc_Speed=None,Case=None,unit_shipspeed='km/hr',unit_time='days',zerocurrents=True,debugging=False,maxiter=1000,optimizer_tol=1e-3):
        '''
           BUG - Zero Currents not working ! 
        '''
        # Cell information
        self.Cell_s         = Sc
        self.Cell_n         = Nc

        self.Mesh           = Mesh

        self.R              = 6371.*1000

        # Inside the code the base units are m/s. Changing the units of the inputs to match
        self.unit_shipspeed = unit_shipspeed
        self.unit_time      = unit_time
        self.s1             = self._unit_speed(Sc_Speed)
        self.s2             = self._unit_speed(Nc_Speed)
        self.case           = Case
        self.splitLevels    = 0

        if zerocurrents:
            self.zx = 0.0
        else:
            self.zx = 1.0

        # Optimisation Information
        self.maxiter       = maxiter
        self.optimizer_tol = optimizer_tol

        # Optimisation Information
        self.mLon  = 111.321*1000
        self.mLat  = 111.386*1000.
        

        # Defining a small distance 
        self.smallDist = 1e-4

        # For Debugging purposes 
        self.debugging     = debugging


    def _dist(self,origin,dest_dist,cell,forward=True):
        mLonScaled=self.mLon*np.cos(cell.cy*(np.pi/180))
        lon1,lat1 = origin
        if forward:
            lon2,lat2 = dest_dist
            # lon2 = lon2+360
            # lon1 = lon1+360
            val = np.sqrt(((lat2-lat1)*self.mLat)**2 + ((lon2-lon1)*mLonScaled)**2)
        else:
            dist_x,dist_y = dest_dist        
            val = [lon1+(dist_x/self.mLat),lat1+(dist_y/mLonScaled)]
        return val


    def NewtonOptimisation(self,f,x,a,Y,u1,v1,u2,v2,s1,s2):
            y0 = (Y*x)/(x+a)
            if self.debugging:
                    print('---Initial y={:.2f}'.format(y0))
            improving = True
            iter = 0
            while improving:
                F,dF,X1,X2,t1,t2  = f(y0,x,a,Y,u1,v1,u2,v2,s1,s2)
                if self.debugging:
                    print('---Iteration {}: y={:.2f}; F={:.5f}; dF={:.2f}'.format(iter,y0,F,dF))
                y0  = y0 - (F/dF)
                improving = abs((F/dF)/(X1*X2)) > self.optimizer_tol
                iter+=1
                if (iter>1000):
                    raise Exception('Newton not able to converge')
            
            return y0,self._unit_time(np.array([t1,t2]))

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
            Val = Val/(60*60*24.)
        elif self.unit_time == 'hr':
            Val = Val/(60*60.)
        elif self.unit_time == 'min':
            Val = Val/(60.)
        elif self.unit_time == 's':
            Val = Val

        return Val
        
    def WaypointCorrection(self,Wp,Cp,s):
        '''
        '''
        uS  = (self.Cell_s.getuC(),self.Cell_s.getvC())
        x   = np.sign(Cp[0]-Wp[0])*self.fdist.value(Wp,(Wp[0]+(Cp[0]-Wp[0]),Wp[1]))
        y   = np.sign(Cp[1]-Wp[1])*self.fdist.value(Wp,(Wp[0],Wp[1]+(Cp[0]-Wp[0])))
        C1  = s**2 - uS[0]**2 - uS[1]**2
        D1  = x*uS[0] + y*uS[1]
        X1  = np.sqrt(D1**2 + C1*(x**2 + y**2))
        return self._unit_time((X1-D1)/C1)

    def _F(self,y,x,a,Y,u1,v1,u2,v2,s1,s2):
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
        dF  = (X1+X2) + y*(dX1 + dX2) - (v1/C1)*(dX2*(X1-D1) + X2*(dX1-dD1)) + (v2/C2)*(dX1*(X2-D2)+X1*(dX2-dD2)) - Y*dX1
    
        t1 = (X1-D1)/C1
        t2 = (X2-D2)/C2

        return F,dF,X1,X2,t1,t2


    def _TCell(self,xdist,ydist,U,V,S):
        '''
            Determines the travel-time within cell between two points
        '''
        dist  = np.sqrt(xdist**2 + ydist**2)
        cval  = np.sqrt(U**2 + V**2)

        dotprod  = xdist*U + ydist*V 
        diffsqrs = S**2 - cval**2

        # if (dotprod**2 + diffsqrs*(dist**2) < 0)
        if (diffsqrs == 0.0):
            if (dotprod == 0.0):
                raise Exception(' ')
            else:
                if ((dist**2)/(2*dotprod))  <0:
                    raise Exception(' ')
                else:
                    traveltime = dist * dist / (2 * dotprod)
                    return traveltime

        traveltime = (np.sqrt(dotprod**2 + (dist**2)*diffsqrs) - dotprod)/diffsqrs
        if (traveltime<0):
            raise Exception('Newton Corner Cases returning Zero Traveltime - ISSUE')
        return traveltime

    def _longitude(self):
        '''
            Longitude cases ...
        '''

        if self.case==2:
            ptvl = 1.0
        else:
            ptvl = -1.0

        Su = ptvl*self.Cell_s.getuC()*self.zx
        Sv = ptvl*self.Cell_s.getvC()*self.zx
        Nu = ptvl*self.Cell_n.getuC()*self.zx
        Nv = ptvl*self.Cell_n.getvC()*self.zx

        Ssp = self.s1
        Nsp = self.s2

        x = self.Cell_s.dcx*self.mLon*np.cos(self.Cell_s.cy*(np.pi/180))
        a = self.Cell_n.dcx*self.mLon*np.cos(self.Cell_n.cy*(np.pi/180))
        Y = ptvl*(self.Cell_n.cy-self.Cell_s.cy)*self.mLat

        # Optimising to determine the y-value of the crossing point
        y,TravelTime = self.NewtonOptimisation(self._F,x,a,Y,Su,Sv,Nu,Nv,Ssp,Nsp)
        CrossPoints = (self.Cell_s.cx+ptvl*self.Cell_s.dcx,self.Cell_s.cy+ptvl*y/self.mLat)        
        CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]

        return TravelTime,CrossPoints,CellPoints

    def _latitude(self):
        '''
            Latitude cases ....

        '''

        if self.case==4:
            ptvl = 1.0
        else:
            ptvl = -1.0

        Su = -1*ptvl*self.Cell_s.getvC()*self.zx
        Sv = ptvl*self.Cell_s.getuC()*self.zx
        Nu = -1*ptvl*self.Cell_n.getvC()*self.zx
        Nv = ptvl*self.Cell_n.getuC()*self.zx

        Ssp=self.s1
        Nsp=self.s2

        x = self.Cell_s.dcy*self.mLat
        a = self.Cell_n.dcy*self.mLat
        Y = ptvl*(self.Cell_n.cx-self.Cell_s.cx)*self.mLon*np.cos((self.Cell_n.cy+self.Cell_s.cy)*(np.pi/180)/2.0)

        y,TravelTime   = self.NewtonOptimisation(self._F,x,a,Y,Su,Sv,Nu,Nv,Ssp,Nsp)
        clon = self.Cell_s.cx  + ptvl*y/(self.mLon*np.cos((self.Cell_n.cy+self.Cell_s.cy)*(np.pi/180)/2.0))
        clat = self.Cell_s.cy + -1*ptvl*self.Cell_s.dcy    
        CrossPoints = (clon,clat)
        CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]

        return TravelTime,CrossPoints,CellPoints


    def _corner(self):
        '''
            Corner Cases ....
        '''

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



        dx1 = self.Cell_s.dcx*self.mLon*np.cos(self.Cell_s.cy*(np.pi/180))
        dx2 = self.Cell_n.dcx*self.mLon*np.cos(self.Cell_n.cy*(np.pi/180))
        dy1 = self.Cell_s.dcy*self.mLat
        dy2 = self.Cell_n.dcy*self.mLat    

        # Currents in Cells
        Su = ptvX*self.Cell_s.getuC()*self.zx; Sv = ptvY*self.Cell_s.getvC()*self.zx
        Nu = ptvX*self.Cell_n.getuC()*self.zx; Nv = ptvY*self.Cell_n.getvC()*self.zx

        # Vehicles Speeds in Cells
        Ssp = self.s1; Nsp = self.s2

        # Determining the crossing point as the corner of the case        
        CrossPoints = [self.Cell_s.cx+ptvX*self.Cell_s.dcx,self.Cell_s.cy+ptvY*self.Cell_s.dcy]
        CellPoints  = [self.Cell_n.cx,self.Cell_n.cy]

        # Determining traveltime
        t1 = self._TCell(dx1,dy1,Su,Sv,Ssp)
        t2 = self._TCell(dx2,dy2,Nu,Nv,Nsp)
        TravelTime  = self._unit_time(np.array([t1,t2]))

        return TravelTime,CrossPoints,CellPoints


    def value(self):
        if self.debugging:
            print('============================================')
        if abs(self.case)==2:
            TravelTime,CrossPoints,CellPoints = self._longitude()
        elif abs(self.case)==4:
            TravelTime,CrossPoints,CellPoints = self._latitude()
        elif abs(self.case)==1 or abs(self.case)==3:
            TravelTime,CrossPoints,CellPoints = self._corner()    
        else:
            print('---> Issue with cell (Xsc,Ysc)={:.2f};{:.2f}'.format(self.Cell_s.cx,self.Cell_s.cy))            
            TravelTime  = [np.inf,np.inf]
            CrossPoints = [np.nan,np.nan]
            CellPoints  = [np.nan,np.nan]

        return TravelTime, CrossPoints, CellPoints


class NewtonianCurve:
    def __init__(self,Mesh,DijkstraInfo,config,unit_shipspeed='km/hr',unit_time='days',debugging=False,maxiter=1000,pathIter=5,optimizer_tol=1e-3,minimumDiff=1e-3,zerocurrents=True):
        '''
    
            BUG:
                - Currently the speed is fixed. Move the construction of the cellBox speed to a function of the cellBox
        
        '''

        # Passing the Mesh information
        self.Mesh = Mesh

        # Passing the Dijkstra Graph
        self.DijkstraInfo = copy.copy(DijkstraInfo)

        # Passing the optional Information
        self.config = config

        
        # Inside the code the base units are m/s. Changing the units of the inputs to match
        self.unit_shipspeed = unit_shipspeed
        self.unit_time      = unit_time
        self.s              = self._unit_speed(26.5)
        
        # Information for distance metrics
        self.R              = 6371.*1000

        # Optimisation Information
        self.maxiter       = maxiter
        self.pathIter      = pathIter
        self.optimizer_tol = optimizer_tol
        self.minimumDiff   = minimumDiff
        self._epsilon      = 1e-3


        # Optimisation Information
        self.mLon  = 111.321*1000
        self.mLat  = 111.386*1000.

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


    def calXDist(self,start_long,end_long):#,centralLat):
        return (end_long - start_long)*self.mLon#*np.cos(centralLat)
    def calYDist(self,start_lat,end_lat):
        return (end_lat-start_lat)*self.mLat

    def _long_case(self):
            def NewtonOptimisationLong(f,y0,x,a,Y,u1,v1,u2,v2,s,R,λ_s,φ_r):
                    tryNum=1
                    iter=0
                    improving=True
                    while improving:  
                        F,dF,X1,X2  = f(y0,x,a,Y,u1,v1,u2,v2,s,R,λ_s,φ_r)
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

            def _F(y,x,a,Y,u1,v1,u2,v2,s,R,λ_s,φ_r):
                θ  = (y/R + λ_s*(np.pi/180))
                zl = x*np.cos(θ)
                ψ  = (-(Y-y)/R + φ_r*(np.pi/180))
                zr = a*np.cos(ψ)

                C1  = s**2 - u1**2 - v1**2
                C2  = s**2 - u2**2 - v2**2
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

            Sp   = tuple(self.triplet[['cX','cY']].iloc[0])
            Cp   = tuple(self.triplet[['cX','cY']].iloc[1])
            Np   = tuple(self.triplet[['cX','cY']].iloc[2])
            Box1 = self.Mesh.cellBoxes[self.triplet.iloc[1]['cellStart'].name]
            Box2 = self.Mesh.cellBoxes[self.triplet.iloc[1]['cellEnd'].name]

            if self.triplet.iloc[1].case == 2:   
                sgn  = 1
            else:
                sgn  = -1

            λ_s  = Sp[1]
            φ_r  = Np[1]

            x           = sgn*self.calXDist(Sp[0],Cp[0])
            a           = sgn*self.calXDist(Cp[0],Np[0])
            Y           = (Np[1]-Sp[1])*self.mLat
            y0          = Y/2
            u1          = sgn*self.zc*Box1.getuC(); v1 = self.zc*Box1.getvC()
            u2          = sgn*self.zc*Box2.getuC(); v2 = self.zc*Box2.getvC()
            y           = NewtonOptimisationLong(_F,y0,x,a,Y,u1,v1,u2,v2,self.s,self.R,λ_s,φ_r)

            # Updating the crossing points
            self.triplet['cX'].iloc[1] = Cp[0]
            self.triplet['cY'].iloc[1] = Sp[1] + y/self.mLat


    def _lat_case(self):
        def NewtonOptimisationLat(f,y0,x,a,Y,u1,v1,u2,v2,s,R,λ,θ,ψ):
                tryNum=1
                iter=0
                improving=True
                while improving:  
                    F,dF,X1,X2  = f(y0,x,a,Y,u1,v1,u2,v2,s,R,λ,θ,ψ)
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

            return F,dF,X1,X2


        Sp = tuple(self.triplet.iloc[0][['cX','cY']])
        Cp = tuple(self.triplet.iloc[1][['cX','cY']])
        Np = tuple(self.triplet.iloc[2][['cX','cY']])
        Box1   = self.Mesh.cellBoxes[self.triplet.iloc[1]['cellStart'].name]
        Box2   = self.Mesh.cellBoxes[self.triplet.iloc[1]['cellEnd'].name]

        if self.triplet.iloc[1].case == 4:   
            sgn   = 1
        else:
            sgn   = -1

        λ=Sp[1]
        θ=Cp[1]   
        ψ=Np[1]  

        x     = sgn*self.calYDist(Sp[1],Cp[1])
        a     = sgn*self.calYDist(Cp[1],Np[1])
        Y     = sgn*(Np[0]-Sp[0])*self.mLon*np.cos(Cp[1]*(np.pi/180))
        Su    = -sgn*self.zc*Box1.getvC(); Sv = sgn*self.zc*Box1.getuC()
        Nu    = -sgn*self.zc*Box2.getvC(); Nv = sgn*self.zc*Box2.getuC()
        y0    = Y/2

        y     = NewtonOptimisationLat(_F,y0,x,a,Y,Su,Sv,Nu,Nv,self.s,self.R,λ,θ,ψ)

        self.triplet['cX'].iloc[1] = Sp[0] + sgn*y/(self.mLon*np.cos(Cp[1]*(np.pi/180)))
        self.triplet['cY'].iloc[1] = Cp[1]


    def _corner_case(self):
        '''
        '''
        # Separting out the Long/Lat of each of the points
        Xs,Ys = tuple(self.triplet.iloc[0][['cX','cY']])
        Xc,Yc = tuple(self.triplet.iloc[1][['cX','cY']])
        Xe,Ye = tuple(self.triplet.iloc[2][['cX','cY']])

        # === 1. Assess the cells that are shared commonly in the corner case ====
        sourceNeighbourIndices = self.triplet.iloc[1]['cellStart']
        endNeighbourIndices    = self.triplet.iloc[1]['cellEnd']
    
        commonIndices = list(set(sourceNeighbourIndices['neighbourIndex']).intersection(endNeighbourIndices['neighbourIndex']))
        CornerCells   = self.DijkstraInfo.loc[commonIndices]
        Y_line = ((Ye-Ys)/(Xe-Xs))*(Xc-Xs) + Ys

        # if np.sign(self.triplet['case'].iloc[1]) == -1:

        if Yc >= Y_line:
            newCell = CornerCells.loc[CornerCells['cY'].idxmin()]
            if newCell.cY > Yc:
                return
        elif Yc < Y_line:
            newCell = CornerCells.loc[CornerCells['cY'].idxmax()]
            if newCell.cY < Yc:
                return

        # === 3. Return the path crossing points and cell indices
        try:
            firstCrossingPoint  = np.array(sourceNeighbourIndices['neighbourCrossingPoints'])[np.where(np.array(sourceNeighbourIndices['neighbourIndex'])==newCell.name)[0][0],:]
            secondCrossingPoint = np.array(newCell['neighbourCrossingPoints'])[np.where(np.array(newCell['neighbourIndex'])==endNeighbourIndices.name)[0][0],:]
        except:
            self.triplet = copy.deepcopy(self.org_triplet)
            return

        # Adding in the new crossing Point
        newP = pd.Series(name=self.triplet.iloc[1].name+1)
        newP['cX']        = secondCrossingPoint[0]
        newP['cY']        = secondCrossingPoint[1]
        newP['cellStart'] = newCell
        newP['cellEnd']   = copy.deepcopy(self.triplet['cellEnd'].iloc[1])
        newP['case']      = newP['cellStart']['case'][np.where(np.array(newP['cellStart']['neighbourIndex'])==newP['cellEnd'].name)[0][0]]


        # Updating the origional crossing point
        self.triplet['cX'].iloc[1]      = firstCrossingPoint[0]
        self.triplet['cY'].iloc[1]      = firstCrossingPoint[1]
        self.triplet['cellEnd'].iloc[1] = newCell 
        self.triplet['case'].iloc[1]    = self.triplet['cellStart'].iloc[1]['case'][np.where(np.array(self.triplet['cellStart'].iloc[1]['neighbourIndex'])==newCell.name)[0][0]]


        # Adding the new crossing point to the triplet
        self.CrossingDF = self.CrossingDF.append(newP,sort=True).sort_index().reset_index(drop=True)
        self.CrossingDF.index = np.arange(int(self.CrossingDF.index.min()),int(self.CrossingDF.index.max()*1e3 + 1e3),int(1e3))


    def _mergePoint(self):
        '''
            Function to merge point if on the corner 
        '''

        def PtDist(Ser1,Ser2):
            return np.sqrt((Ser1['cX'] - Ser2['cX'])**2 + (Ser1['cY'] - Ser2['cY'])**2)

        id=0
        while id < len(self.CrossingDF)-3:
            triplet = self.CrossingDF.iloc[id:id+3]
            if PtDist(triplet.iloc[0],triplet.iloc[1]) < 1e-2:
                neighbourIndex = np.where(np.array(triplet.iloc[0]['cellStart']['neighbourIndex'])==triplet.iloc[1]['cellEnd'].name)[0][0]
                case           = triplet['cellStart'].iloc[0]['case'][neighbourIndex]
                crossingPoint  = triplet['cellStart'].iloc[0]['neighbourCrossingPoints'][neighbourIndex]
                triplet['cX'].iloc[0]      = crossingPoint[0]
                triplet['cY'].iloc[0]      = crossingPoint[1]
                triplet['cellEnd'].iloc[0] = copy.deepcopy(triplet.iloc[1]['cellEnd'])
                triplet['case'].iloc[0]    = copy.deepcopy(case)
                self.CrossingDF           = self.CrossingDF.drop(triplet.iloc[1].name)
            if PtDist(triplet.iloc[1],triplet.iloc[2]) < 1e-2:
                neighbourIndex = np.where(np.array(triplet.iloc[1]['cellStart']['neighbourIndex'])==triplet.iloc[2]['cellEnd'].name)[0][0]
                case           = triplet['cellStart'].iloc[1]['case'][neighbourIndex]
                crossingPoint  = triplet['cellStart'].iloc[1]['neighbourCrossingPoints'][neighbourIndex]
                triplet['cX'].iloc[1]      = crossingPoint[0]
                triplet['cY'].iloc[1]      = crossingPoint[1]
                triplet['cellEnd'].iloc[1] = copy.deepcopy(triplet.iloc[2]['cellEnd'])
                triplet['case'].iloc[1]    = copy.deepcopy(case)
                self.CrossingDF           = self.CrossingDF.drop(triplet.iloc[2].name)
            
            id+=1

        self.CrossingDF = self.CrossingDF.sort_index().reset_index(drop=True)
        self.CrossingDF.index = np.arange(int(self.CrossingDF.index.min()),int(self.CrossingDF.index.max()*1e3 + 1e3),int(1e3))



    def _horseshoe(self):
        '''

        '''

        # Defining the case information
        Cp             = tuple(self.triplet[['cX','cY']].iloc[1])
        Sp             = tuple(self.triplet.iloc[0][['cX','cY']])
        Np             = tuple(self.triplet.iloc[2][['cX','cY']])
        cellStart      = self.Mesh.cellBoxes[self.triplet.iloc[1]['cellStart'].name]
        cellStartGraph = self.triplet.iloc[1]['cellStart']
        cellEnd        = self.Mesh.cellBoxes[self.triplet.iloc[1]['cellEnd'].name]
        cellEndGraph   = self.triplet.iloc[1]['cellEnd']
        case           = self.triplet['case'].iloc[1]

        # Returning if corner horseshoe case type
        if abs(case)==1 or abs(case)==3 or abs(case)==0: 
            return
        elif abs(case) == 2:

            # Defining the global min and max
            vmin = np.max([cellStart.cy-cellStart.dcy,cellEnd.cy-cellEnd.dcy])
            vmax = np.min([cellStart.cy+cellStart.dcy,cellEnd.cy+cellEnd.dcy])

            # Point crossingpoint on boundary between the two origional cells
            if (Cp[1] >= vmin) and (Cp[1] <= vmax):
                return

            # Defining the min and max of the start and end cells
            smin = cellStart.cy-cellStart.dcy   
            smax = cellStart.cy+cellStart.dcy
            emin = cellEnd.cy-cellEnd.dcy
            emax = cellEnd.cy+cellEnd.dcy

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
                hrshCaseEnd   = 4                
            if (Cp[1]<=emin) and (smin<emin):
                hrshCaseStart = case
                hrshCaseEnd   = -4                   

            # --- Cases where StartCell is smaller than end Cell ---
            if (Cp[1]>=smax) and (smax<emax):
                hrshCaseStart = -4
                hrshCaseEnd   = case
            if (Cp[1]<=smin) and (emin<smin):
                hrshCaseStart = 4
                hrshCaseEnd   = case                    

        elif abs(case) == 4:

            # Defining the global min and max
            gmin = np.min([cellStart.cx-cellStart.dcx,cellEnd.cx-cellEnd.dcx])
            gmax = np.max([cellStart.cx+cellStart.dcx,cellEnd.cx+cellEnd.dcx])
            vmin = np.max([cellStart.cx-cellStart.dcx,cellEnd.cx-cellEnd.dcx])
            vmax = np.min([cellStart.cx+cellStart.dcx,cellEnd.cx+cellEnd.dcx])

            # Point crossingpoint on boundary between the two origional cells
            if (Cp[0] >= vmin) and (Cp[0] <= vmax):
                return

            # Defining the min and max of the start and end cells
            smin = cellStart.cx-cellStart.dcx   
            smax = cellStart.cx+cellStart.dcx
            emin = cellEnd.cx-cellEnd.dcx
            emax = cellEnd.cx+cellEnd.dcx


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
                hrshCaseEnd   = -2                
            if (Cp[1]<emin) and (smin<emin):
                hrshCaseStart = case
                hrshCaseEnd   = 2                   

            # --- Cases where StartCell is smaller than end Cell ---
            if (Cp[0]>smax) and (smax<emax):
                hrshCaseStart = 2
                hrshCaseEnd   = case
            if (Cp[0]<smin) and (emin<smin):
                hrshCaseStart = -2
                hrshCaseEnd   = case   


        # Determining the neighbours of the start and end cells that are the horseshoe case
        startGraphNeighbours = [cellStartGraph['neighbourIndex'][ii] for ii in list(np.where(np.array(cellStartGraph['case'])==hrshCaseStart)[0])]
        endGraphNeighbours   = [cellEndGraph['neighbourIndex'][ii] for ii in list(np.where(np.array(cellEndGraph['case'])==hrshCaseEnd)[0])]

        if (len(startGraphNeighbours)==0) or (len(endGraphNeighbours)==0):
            if abs(case) == 2:
                self.triplet['cY'].iloc[1] = np.clip(self.triplet.iloc[1]['cY'],vmin,vmax)
            if abs(case) == 4:
                self.triplet['cX'].iloc[1] = np.clip(self.triplet.iloc[1]['cX'],vmin,vmax)        
            return
        
        if abs(hrshCaseStart) == abs(hrshCaseEnd):
            for sGN in startGraphNeighbours:
                for eGN in endGraphNeighbours:
                    if (np.array(self.DijkstraInfo.loc[sGN,'neighbourIndex'])==eGN).any() and (np.array(self.DijkstraInfo.loc[eGN,'neighbourIndex'])==sGN).any():
                        sGNGraph = self.DijkstraInfo.loc[sGN]
                        eGNGraph = self.DijkstraInfo.loc[eGN]

                        Crp1 = np.array(cellStartGraph['neighbourCrossingPoints'])[np.where(np.array(cellStartGraph['neighbourIndex']) == sGN)[0][0],:]
                        Crp2 = np.array(sGNGraph['neighbourCrossingPoints'])[np.where(np.array(sGNGraph['neighbourIndex']) == eGN)[0][0],:]
                        Crp3 = np.array(eGNGraph['neighbourCrossingPoints'])[np.where(np.array(eGNGraph['neighbourIndex']) == cellEndGraph.name)[0][0],:]
                        


                        # Updating the origional crossing point
                        self.triplet['cX'].iloc[1]      = Crp1[0]
                        self.triplet['cY'].iloc[1]      = Crp1[1]
                        self.triplet['cellEnd'].iloc[1] = copy.deepcopy(sGNGraph)
                        self.triplet['case'].iloc[1]    = self.triplet['cellStart'].iloc[1]['case'][np.where(np.array(self.triplet['cellStart'].iloc[1]['neighbourIndex'])==sGNGraph.name)[0][0]]

                        # Crossing Point 2
                        Pcrp2 = pd.Series(name=self.triplet.iloc[1].name+1)
                        Pcrp2['cX']        = Crp2[0]
                        Pcrp2['cY']        = Crp2[1]
                        Pcrp2['cellStart'] = copy.deepcopy(sGNGraph)
                        Pcrp2['cellEnd']   = copy.deepcopy(eGNGraph)
                        Pcrp2['case']      = Pcrp2['cellStart']['case'][np.where(np.array(Pcrp2['cellStart']['neighbourIndex'])==Pcrp2['cellEnd'].name)[0][0]]

                        Pcrp3 = pd.Series(name=self.triplet.iloc[1].name+2)
                        Pcrp3['cX']        = Crp3[0]
                        Pcrp3['cY']        = Crp3[1]
                        Pcrp3['cellStart'] = copy.deepcopy(eGNGraph)
                        Pcrp3['cellEnd']   = copy.deepcopy(cellEndGraph)
                        Pcrp3['case']      = Pcrp3['cellStart']['case'][np.where(np.array(Pcrp3['cellStart']['neighbourIndex'])==Pcrp3['cellEnd'].name)[0][0]]
                        

                        self.CrossingDF = self.CrossingDF.append([Pcrp2,Pcrp3],sort=True).sort_index().reset_index(drop=True)
                        self.CrossingDF.index = np.arange(int(self.CrossingDF.index.min()),int(self.CrossingDF.index.max()*1e3 + 1e3),int(1e3))

                        self.id=-1
        else:
            for sGN in startGraphNeighbours:
                for eGN in endGraphNeighbours:
                    if (np.array(sGN==eGN).any()):
                        NeighGraph = self.DijkstraInfo.loc[sGN]               
                        Crp1 = np.array(cellStartGraph['neighbourCrossingPoints'])[np.where(np.array(cellStartGraph['neighbourIndex']) == sGN)[0][0],:]
                        Crp2 = np.array(NeighGraph['neighbourCrossingPoints'])[np.where(np.array(NeighGraph['neighbourIndex']) == cellEndGraph.name)[0][0],:]


                        # Updating the origional crossing point
                        self.triplet['cX'].iloc[1]      = Crp1[0]
                        self.triplet['cY'].iloc[1]      = Crp1[1]
                        self.triplet['cellEnd'].iloc[1] = copy.deepcopy(NeighGraph)
                        self.triplet['case'].iloc[1]    = self.triplet['cellStart'].iloc[1]['case'][np.where(np.array(self.triplet['cellStart'].iloc[1]['neighbourIndex'])==NeighGraph.name)[0][0]]

                        Pcrp2 = pd.Series(name=self.triplet.iloc[1].name+2)
                        Pcrp2['cX']        = Crp2[0]
                        Pcrp2['cY']        = Crp2[1]
                        Pcrp2['cellStart'] = copy.deepcopy(NeighGraph)
                        Pcrp2['cellEnd']   = copy.deepcopy(cellEndGraph)
                        Pcrp2['case']      = Pcrp2['cellStart']['case'][np.where(np.array(Pcrp2['cellStart']['neighbourIndex'])==Pcrp2['cellEnd'].name)[0][0]]
                        
                        self.CrossingDF = self.CrossingDF.append([Pcrp2],sort=True).sort_index().reset_index(drop=True)
                        self.CrossingDF.index = np.arange(int(self.CrossingDF.index.min()),int(self.CrossingDF.index.max()*1e3 + 1e3),int(1e3))

                        self.id=-1

    def _reverseCase(self):

        # Removing Reverse Edge Type 1
        startIndex = np.array([row['cellStart'].name for idx,row in self.CrossingDF.iterrows()][1:-1])
        endIndex   = np.array([row['cellEnd'].name for idx,row in self.CrossingDF.iterrows()][1:-1] )
        boolReverseEdge  = np.logical_and((startIndex[:-1] == endIndex[1:]),(startIndex[1:] == endIndex[:-1]))
        if boolReverseEdge.any():
            indxReverseEdge = np.where(boolReverseEdge)[0]+1
            for id in indxReverseEdge:
                self.CrossingDF = self.CrossingDF.drop(self.CrossingDF.iloc[id].name).sort_index().reset_index(drop=True)


        # Removing Reverse Edge Type 2
        startIndex = np.array([row['cellStart'].name for idx,row in self.CrossingDF.iterrows()][1:-1])
        endIndex   = np.array([row['cellEnd'].name for idx,row in self.CrossingDF.iterrows()][1:-1] )
        boolReverseEdge  = (endIndex[:-1] == endIndex[1:])
        if boolReverseEdge.any():
            indxReverseEdge = np.where(boolReverseEdge)[0]+2
            for id in indxReverseEdge:
                self.CrossingDF = self.CrossingDF.drop(self.CrossingDF.iloc[id].name).sort_index().reset_index(drop=True)


        self.CrossingDF.index = np.arange(0,int(len(self.CrossingDF)*1e3),int(1e3))


    def _updateCrossingPoint(self):
        '''
            COMPLETE:
                - Unsmoothed _long_case_ Path, Unsplit & No Currents - 
        '''


        self.org_triplet = copy.deepcopy(self.triplet) 


        # ------ Case Deginitions & Dealing
        if self.debugging:
            print('===========================================================')
        if abs(self.triplet.iloc[1].case)==2:
            self._long_case()
            self.id=0
        elif abs(self.triplet.iloc[1].case)==4:
            self._lat_case()
            self.id=0
        elif (abs(self.triplet.iloc[1].case)==1) or (abs(self.triplet.iloc[1].case)==3):
            self._corner_case()
            #self.id=-1

        if len(self.triplet) < 3:
            return


