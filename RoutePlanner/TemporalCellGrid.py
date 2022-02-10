import pandas as pd
import xarray as xr
from RoutePlanner.CellGrid import CellGrid

class TemporalCellGrid:

    def __init__(self, longMin, longMax, latMin, latMax, cellWidth, cellHeight):
        self._longMin = longMin
        self._longMax = longMax
        self._latMin = latMin
        self._latMax = latMax

        self._cellWidth = cellWidth
        self._cellHeight = cellHeight

        self.cellGrids = []

    def addIcePoints(self, icePointsPath):
        """
            takes a netCDF file containing sea-ice data given by parameter 'icePointsPath' and converts it to a
            dataframe to be stored within this object.
        """
        ds = xr.open_dataset(icePointsPath)
        df = ds.to_dataframe()

        self._icePoints = df

    def getGrid(self, time):
        """
            Returns a cellGrid for a selected time given by parameter 'time'
        """
        # get all icePoints for the given time
        df = self._icePoints[self._icePoints.index.get_level_values('time') == time]

        # flatten the dataframe so as it may be used by the 'cellGrid' object.
        # TODO - rework flattening as part of the optimization work-package
        df = df.reset_index()
        df = df.drop(['iter'], axis=1)
        df = df.rename(columns={"XC": "long", "YC": "lat", "Depth": "depth", "SIarea": "iceArea"})

        # Load current points
        # TODO - replace with daily current data once it is available
        currentPoints = pd.read_json('resources/currentPoints.json')
        currentPoints = pd.DataFrame.from_records(currentPoints.currentPoints)

        # create a cellGrid using datapoints for the given day
        cellGrid = CellGrid(self._longMin, self._longMax, self._latMin, self._latMax, self._cellWidth, self._cellHeight)
        cellGrid.addCurrentPoints(currentPoints)
        cellGrid.addIcePoints(df)

        return cellGrid

    def getMeanGrid(self, startTime, endTime):
        # get all icePoints for the given time

        # TODO requires rework in optimization work package
        df = self._icePoints[self._icePoints.index.get_level_values('time') > startTime]
        df = df[df.index.get_level_values('time') < endTime]

        df = df.groupby(['XC', 'YC']).mean()

        df = df.reset_index()
        df = df.drop(['iter'], axis=1)
        df = df.rename(columns={"XC": "long", "YC": "lat", "Depth": "depth", "SIarea": "iceArea"})

        # Load current points
        # TODO - replace with daily current data once it is available
        currentPoints = pd.read_json('resources/currentPoints.json')
        currentPoints = pd.DataFrame.from_records(currentPoints.currentPoints)

        # create a cellGrid using datapoints for the given day
        cellGrid = CellGrid(self._longMin, self._longMax, self._latMin, self._latMax, self._cellWidth, self._cellHeight)
        cellGrid.addCurrentPoints(currentPoints)
        cellGrid.addIcePoints(df)

        return cellGrid

import numpy as np
from netCDF4 import Dataset
import pandas as pd
from RoutePlanner.CellGrid import CellGrid

class TimeCellGrid:

    def __init__(self, OptInfo):
        self.OptInfo     = OptInfo

        self._longMin    = self.OptInfo['Bounds Longitude'][0] 
        self._longMax    = self.OptInfo['Bounds Longitude'][1]
        self._latMin     = self.OptInfo['Bounds Latitude'][0]
        self._latMax     = self.OptInfo['Bounds Latitude'][1]
        
        self._cellWidth  = self.OptInfo['Grid Spacing (dx,dy)'][0]
        self._cellHeight = self.OptInfo['Grid Spacing (dx,dy)'][1]

        self._icePoints     = Dataset(self.OptInfo['Ice Data Path'])
        self._CurrentPoints = Dataset(self.OptInfo['Current Data Path'])

        self.cellGrids   = []

    def value(self, time):
        """
            Returns a cellGrid for a selected time given by parameter 'time'
        """
        Dates = pd.to_datetime('2012-12-01') + pd.to_timedelta(self._icePoints['time'][:],unit='S')

        DateRangeIndx = np.argmin(abs(Dates - pd.to_datetime(time)))

        XC,YC = np.meshgrid(self._icePoints['XC'][:].data,self._icePoints['YC'][:].data)
        icePoints = pd.DataFrame({'long':XC.flatten(),'lat':YC.flatten(),
                                'iceArea':self._icePoints['SIarea'][DateRangeIndx,...].flatten(),
                                'depth':self._icePoints['Depth'][:,:].data.flatten()})

        icePoints['time'] = Dates[DateRangeIndx]

        currentPoints = pd.DataFrame({'long':self._CurrentPoints['lon'][...].data.flatten(),'lat':self._CurrentPoints['lat'][...].data.flatten(),'uC':self._CurrentPoints['uC'][...].data.flatten(),'vC':self._CurrentPoints['vC'][...].data.flatten()})
        currentPoints['time'] = ''

        # create a cellGrid using datapoints for the given day
        cellGrid = CellGrid(self.OptInfo)
        cellGrid.addCurrentPoints(currentPoints)
        cellGrid.addIcePoints(icePoints)

        return cellGrid

    def range(self, startTime, endTime):
        # get all icePoints for the given time

        Dates = pd.to_datetime('2012-12-01') + pd.to_timedelta(self._icePoints['time'][:],unit='S')

        DateRange     = [startTime,endTime]
        DateRangeIndx = [np.argmin(abs(Dates - pd.to_datetime(DateRange[0]))),np.argmin(abs(Dates - pd.to_datetime(DateRange[1])))] 

        XC,YC     = np.meshgrid(self._icePoints['XC'][:].data,self._icePoints['YC'][:].data)
        IceData   = self._icePoints['SIarea'][DateRangeIndx[0]:DateRangeIndx[1],...].data
        TimeInfo  = np.ones(IceData.shape)*self._icePoints['time'][DateRangeIndx[0]:DateRangeIndx[1]].data[:,None,None]
        Depth     = np.ones(IceData.shape)*self._icePoints['Depth'][:,:].data[None,:,:]
        Xct       = np.ones(IceData.shape)*XC[None,:,:]
        Yct       = np.ones(IceData.shape)*YC[None,:,:]
        icePoints = pd.DataFrame({'time':TimeInfo.flatten(),
                                'long':Xct.flatten(),'lat':Yct.flatten(),
                                        'iceArea':IceData.flatten(),
                                        'depth':Depth.flatten()})

        currentPoints = pd.DataFrame({'long':self._CurrentPoints['lon'][...].data.flatten(),'lat':self._CurrentPoints['lat'][...].data.flatten(),'uC':self._CurrentPoints['uC'][...].data.flatten(),'vC':self._CurrentPoints['vC'][...].data.flatten()})
        currentPoints['time'] = ''

        # create a cellGrid using datapoints for the given day
        cellGrid = CellGrid(self.OptInfo)
        cellGrid.addCurrentPoints(currentPoints)
        cellGrid.addIcePoints(icePoints)

        return cellGrid


