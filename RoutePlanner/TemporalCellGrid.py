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
        ds = xr.open_dataset(icePointsPath)
        df = ds.to_dataframe()

        self._icePoints = df

    def getGrid(self, time):
        df = self._icePoints[self._icePoints.index.get_level_values('time') == time]
        df = df.reset_index()
        df = df.drop(['iter'], axis=1)
        df = df.rename(columns={"XC": "long", "YC": "lat", "Depth": "depth", "SIarea": "iceArea"})

        currentPoints = pd.read_json('resources/currentPoints.json')
        currentPoints = pd.DataFrame.from_records(currentPoints.currentPoints)

        cellGrid = CellGrid(self._longMin, self._longMax, self._latMin, self._latMax, self._cellWidth, self._cellHeight)
        cellGrid.addCurrentPoints(currentPoints)
        cellGrid.addIcePoints(df)

        return cellGrid
