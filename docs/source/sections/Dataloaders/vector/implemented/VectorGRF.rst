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
            "data_name": "data",
            "seed":      None,
            "size":      512,
            "alpha":     3,
            "min":       1,
            "max":       10,
            "vec_x":     "uC",
            "vec_y":     "vC"
        }
    }
    
The dataloader is implemented as follows:

.. automodule:: polar_route.dataloaders.vector.vectorGRF
   :special-members: __init__
   :members: