**********************
BSOSE Depth Dataloader
**********************

B-SOSE (Biogeochemical Southern Ocean State Estimate solution) provide a publically available dataset that
hosts (amongst other products) sea ice concentration (SIC) of the southern ocean. Their SIC product provides 
a 'depth' value, which this dataloader ingests.
BSOSE is an extension of the SOSE project led by Mazloff at the Scripps Institution of Oceanography.

From their website:
   The Southern Ocean State Estimate (SOSE) is a model-generated best fit to Southern Ocean 
   observations. As such, it provides a quantitatively useful climatology of the mean-state 
   of the Southern Ocean. 

Data can be downloaded from `here <http://sose.ucsd.edu/BSOSE_iter105_solution.html>`_

Note: This dataloader may not work as is for new data downloaded, it has been internally collated into 
a more easily ingestable format.

.. automodule:: polar_route.dataloaders.scalar.bsoseDepth
   :special-members: __init__
   :members: