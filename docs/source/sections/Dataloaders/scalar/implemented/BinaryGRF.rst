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
            "data_name": "data",
            "seed":      None,
            "size":      512,
            "alpha":     3,
            "min":       1,
            "max":       10,
            "binary":    True,
            "threshold": 0.5
        }
    }
    
See the :ref:`Scalar GRF page<scalar-grf>` for documentation on the dataloader