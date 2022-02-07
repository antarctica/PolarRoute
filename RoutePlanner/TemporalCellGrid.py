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
