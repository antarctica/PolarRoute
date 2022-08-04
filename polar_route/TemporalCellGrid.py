import pandas as pd
import xarray as xr
from RoutePlanner.CellGrid import CellGrid
from netCDF4 import Dataset
import numpy as np
import datetime

class TemporalCellGrid:

    def __init__(self, OptInfo):
        self.OptInfo = OptInfo['Mesh']

        self._longMin = self.OptInfo['Longitude Bounds (Min,Max,Width)'][0]
        self._longMax = self.OptInfo['Longitude Bounds (Min,Max,Width)'][1]
        self._latMin  = self.OptInfo['Latitude Bounds (Min,Max,Width)'][0]
        self._latMax  = self.OptInfo['Latitude Bounds (Min,Max,Width)'][1]

        self._cellWidth = self.OptInfo['Longitude Bounds (Min,Max,Width)'][2]
        self._cellHeight = self.OptInfo['Latitude Bounds (Min,Max,Width)'][2]

        self.cellGrids = []


        self.addIcePoints(self.OptInfo['Ice Data Path'], self.OptInfo['Date Range (Min,Max,dT)'][0], self.OptInfo['Date Range (Min,Max,dT)'][1])
        self.addCurrentPoints(self.OptInfo['Current Data Path'])



    def _loadDailyIce(self, icePointsPath ,time):
        bsos = Dataset(icePointsPath)

        Dates = pd.to_datetime('2012-12-01') + pd.to_timedelta(bsos['time'][:], unit='S')

        timeindx = np.argmin(abs(Dates - pd.to_datetime(time)))

        XC, YC = np.meshgrid(bsos['XC'][:].data, bsos['YC'][:].data)

        icePoints = pd.DataFrame({'time': pd.to_datetime(Dates[timeindx]),
                                  'long': XC.flatten(),
                                  'lat': YC.flatten(),
                                  'iceArea': bsos['SIarea'][timeindx, ...].data.flatten(),
                                  'depth': bsos['Depth'][...].data.flatten()})
        return icePoints

    #def _loadDailyWind(self, era5windPathU, era5windPathv, time):


    def addIcePoints(self, icePointsPath, startDate, endDate):
        self._icePointsPath = icePointsPath
        """
        startDate = pd.to_datetime(startDate)
        endDate = pd.to_datetime(endDate)

        delta = endDate - startDate

        icePoints = []
        for i in range(delta.days + 1):
            day = startDate + datetime.timedelta(days=i)

            icePoints.append(self._loadDailyIce(icePointsPath, day))

        icePoints = pd.concat(icePoints)

        icePoints['long'] = icePoints['long'].apply(lambda x: x if x <= 180 else x - 360)

        self._icePoints =  icePoints
        """

    def addCurrentPoints(self, currentPointsPath):
        sose = Dataset(currentPointsPath)

        currentPoints = pd.DataFrame({'long': sose['lon'][...].data.flatten(),
                                      'lat': sose['lat'][...].data.flatten(),
                                      'uC': sose['uC'][...].data.flatten(),
                                      'vC': sose['vC'][...].data.flatten()})

        currentPoints['time'] = ''

        currentPoints['uC'] = currentPoints['uC'] * 3.6
        currentPoints['vC'] = currentPoints['vC'] * 3.6
        currentPoints['long'] = currentPoints['long'].apply(lambda x: x if x <= 180 else x - 360)
        self._currentPoints = currentPoints

    # def getGrid(self, time):
    #     """
    #         Returns a cellGrid for a selected time given by parameter 'time'
    #     """
    #     icePoints = self._icePoints.loc[self._icePoints['time'] == time]

    #     # create a cellGrid using datapoints for the given day
    #     cellGrid = CellGrid(self._longMin, self._longMax, self._latMin, self._latMax, self._cellWidth, self._cellHeight)
    #     cellGrid.addCurrentPoints(self._currentPoints)
    #     cellGrid.addIcePoints(icePoints)

    #     return cellGrid

    def range(self, startTime, endTime, j_grid=False):
        """
         # get all icePoints for the given time
        startDate = pd.to_datetime(startTime)
        endDate = pd.to_datetime(endTime)

        delta = endDate - startDate

        icePoints = []
        for i in range(delta.days + 1):
            day = startDate + datetime.timedelta(days=i)

            icePoints.append(self._loadDailyIce(self._icePointsPath, day))

        icePoints = pd.concat(icePoints)

        # convert long from 0:360 to -180:180
        icePoints['long'] = icePoints['long'].apply(lambda x: x if x <= 180 else x - 360)
        """

        icePointsAll = xr.open_dataset(self._icePointsPath, autoclose=True)
        icePoints = icePointsAll.sel(time=slice(startTime, endTime))

        # create a cellGrid using datapoints for the given day
        cellGrid = CellGrid(self._longMin, self._longMax, self._latMin, self._latMax, self._cellWidth, self._cellHeight, j_grid)
        cellGrid.add_current_points(self._currentPoints)
        cellGrid.addIcePoints(icePoints)

        icePoints.close()

        return cellGrid

    def getGrids(self, startTime, endTime, step, j_grid=False):
        cellGrids = []
        endTime = pd.to_datetime(endTime)

        tempStart = pd.to_datetime(startTime)
        tempEnd = tempStart + pd.to_timedelta(step, unit='D')

        while(tempEnd <= endTime):
            cellGrids.append(self.range(tempStart, tempEnd, j_grid))
            tempStart = tempEnd + pd.to_timedelta(1, unit='D')
            tempEnd = tempStart + pd.to_timedelta(step, unit='D')

        return cellGrids