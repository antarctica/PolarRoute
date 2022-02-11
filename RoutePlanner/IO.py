from netCDF4 import Dataset
import numpy as np
import pandas as pd


def LoadIcePoints(NetCDF,startDate,endDate):
    bsos = Dataset(NetCDF)

    Dates = pd.to_datetime('2012-12-01') + pd.to_timedelta(bsos['time'][:],unit='S')

    DateRange     = [startDate,endDate]
    DateRangeIndx = [np.argmin(abs(Dates - pd.to_datetime(DateRange[0]))),np.argmin(abs(Dates - pd.to_datetime(DateRange[1])))] 

    XC,YC = np.meshgrid(bsos['XC'][:].data,bsos['YC'][:].data)

    icePoints = pd.DataFrame({'time':pd.to_datetime(DateRangeIndx[0]),
                            'long':XC.flatten(),'lat':YC.flatten(),
                            'iceArea':np.mean(bsos['SIarea'][DateRangeIndx[0]:DateRangeIndx[1],...].data,axis=0).flatten(),
                            'depth':bsos['Depth'][:,:].data.flatten()})
    return icePoints

def LoadCurrentPoints(NetCDF):
    sose = Dataset(NetCDF)
    currentPoints = pd.DataFrame({'long':sose['lon'][...].data.flatten(),'lat':sose['lat'][...].data.flatten(),'uC':sose['uC'][...].data.flatten(),'vC':sose['vC'][...].data.flatten()})
    currentPoints['time'] = ''
    return currentPoints
