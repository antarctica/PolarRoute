.. _vector-grf:

*********************
Vector GRF Dataloader
*********************

Produces a gaussian random field of vector values, useful for producing 
artificial, yet somewhat realistic values for real-world variables. 
Values are broken down into `x` and `y` components, and saved in two
columns in the final dataframe.

Can be used to generate :ref:`binary masks<binary-grf>`.

For scalar fields, see  the :ref:`Vector GRF page<scalar-grf>`.

.. code-block::
    :caption: Default parameters for vector GRF dataloader 

    {
        "loader": "vector_grf",
        "params":{
            "vec_x":     "uC",      # - Name of the first data column
            "vec_y":     "vC",      # - Name of the second data column
            "seed":       None,     # - Seed for random number generator. Must
                                    #   be int or None. None sets a random seed
            "size":       512,      # - Number of datapoints per lat/long axis
            "alpha":      3,        # - Power of the power-law momentum 
                                    #   distribution used to generate GRF
            "min":        0,        # - Minimum value of vector magnitude
            "max":        10        # - Maximum value of vector magnitude
        }
    }

The dataloader is implemented as follows:

.. automodule:: polar_route.dataloaders.vector.vector_grf
   :special-members: __init__
   :members: