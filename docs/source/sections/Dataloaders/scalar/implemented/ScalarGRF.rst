.. _scalar-grf:

*********************
Scalar GRF Dataloader
*********************

Produces a gaussian random field of scalar values, useful for producing 
artificial, yet somewhat realistic values for real-world variables.

Can be used to generate :ref:`binary masks<binary-grf>`.

For vector fields, see  the :ref:`Vector GRF page<vector-grf>`.

.. code-block::
    :caption: Default parameters for scalar GRF dataloader. 

    {
        "loader": "scalar_grf",
        "params":{
            "data_name": "data",    # - Name of the data column
            "seed":       None,     # - Seed for random number generator. Must
                                    #   be int or None. None sets a random seed
            "size":       512,      # - Number of datapoints per lat/long axis
            "alpha":      3,        # - Power of the power-law momentum 
                                    #   distribution used to generate GRF
            "binary":     False,    # - Flag specifying this GRF isn't a binary mask
            "threshold":  [0, 1],   # - Caps of min/max values to ensure normalising 
                                    #   not skewed by outlier in randomised GRF
            "min":        -10,      # - Minimum value of GRF
            "max":        10,       # - Maximum value of GRF
            "multiplier": 1,        # - Multiplier for entire dataset
            "offset":     0         # - Offset for entire dataset
        }
    }

NOTE: min/max are set BEFORE multiplier and offset are used. The actual values for
the min and max are 

| :code:`actual_min = multiplier * min + offset`
| :code:`actual_max = multiplier * max + offset` 

The dataloader is implemented as follows:

.. automodule:: polar_route.dataloaders.scalar.scalar_grf
   :special-members: __init__
   :members: