'''
    Python scripts used to determine the optimal pathway of transients across the cell-grid system.
    This module can then be interchanged with alternative trasient forecasts or the route paths from alternative
    ships/drones
'''


class Icebergs:
    '''
        Simple module for processing and updating the iceberge locations through time determining the location on the same time interval as
        the RoutePlanner cellGrid, and returning the indices of the iceberg locations. 
    '''
    def __init__(self,cellGrid,object_locations):
        self.cellGrid         = cellGrid
        self.object_locations = object_locations

    def _cellGridResample(self):
        '''
            Function that takes the super resolution of the iceberg tracking and resamples on the temporal sampling of the 
            Routeplanner cellGrid structure
        '''
    
    def _physicsModel(self):
        '''
            The physics based model describing the updating of the iceberge locations. 
            After better methods are determined then this is the main area to update.

            INPUTS:


            OUTPUTS:
        '''

    def forward(self):
        '''
            Given the cellGrid and the iceberg locations forward propogate to determine the expected locations given a physics model describing the dynamics of the icebergs.
        '''

        



