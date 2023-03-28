.. _binary-grf:

*********************
Binary GRF Dataloader
*********************

The binary GRF dataloader is the same as the :ref:`Scalar GRF<scalar-grf>`
The only difference is that instead of returning a dataframe that consists
of values between the min/max set in the config, this dataframe will contain
only True/False. It is useful for generating land masks.

.. code-block:: 
    :caption: Default parameters for binary/mask GRF dataloader 

    {
        "loader": "binary_grf",
        "params":{
            "data_name": "data",    # - Name of the data column
            "seed":       None,     # - Seed for random number generator. Must
                                    #   be int or None. None sets a random seed
            "size":       512,      # - Number of datapoints per lat/long axis
            "alpha":      3,        # - Power of the power-law momentum 
                                    #   distribution used to generate GRF
            "min":        0,        # - Minimum value of GRF
            "max":        1,        # - Maximum value of GRF
            "binary":     True,     # - Flag specifiying this GRF is a binary mask
            "threshold":  0.5       # - Value around which mask values are set.
                                    #   Below this, values are set to False 
                                    #   Above this, values are set to True
        }
    }

See the :ref:`Scalar GRF page<scalar-grf>` for documentation on the dataloader