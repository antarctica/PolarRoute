.. _scalar-grf:

*********************
Scalar GRF Dataloader
*********************

Produces a gaussian random field of scalar values, useful for producing 
artificial, yet somewhat realistic values for real-world variables.

Can be used to generate :ref:`binary masks<binary-grf>`.

For vector fields, see  the :ref:`Vector GRF page<vector-grf>`.

.. code-block::
    :caption: Default parameters for scalar GRF dataloader 

    {
        "loader": "scalar_grf",
        "params":{
            "data_name": "data",
            "seed":      None,
            "size":      512,
            "alpha":     3,
            "min":       1,
            "max":       10,
            "binary":    False,
            "threshold": [0, 1]
        }
    }
    
The dataloader is implemented as follows:

.. automodule:: polar_route.dataloaders.scalar.scalarGRF
   :special-members: __init__
   :members: