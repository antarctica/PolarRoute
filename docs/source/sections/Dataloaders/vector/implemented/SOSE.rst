************************
SOSE Currents Dataloader
************************

Southern Ocean State Estimate (SOSE) is a publicly available dataset that provides (amongst other products)
ocean current vectors of the southern ocean. It is a project led by Mazloff at the Scripps Institution of Oceanography.

From their website:
   The Southern Ocean State Estimate (SOSE) is a model-generated best fit to Southern Ocean 
   observations. As such, it provides a quantitatively useful climatology of the mean-state 
   of the Southern Ocean. 

Data can be downloaded from `here <http://sose.ucsd.edu/sose_stateestimation_data_05to10.html>`_

Note: This dataloader may not work as is for new data downloaded, it has been internally collated into 
a more easily ingestable format.

.. automodule:: polar_route.dataloaders.vector.sose
   :special-members: __init__
   :members: