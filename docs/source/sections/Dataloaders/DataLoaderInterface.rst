.. _dataloader-interface:
********************
Dataloader Interface
********************

Shows how the mesh generation code may interact with the dataloaders. In operation,
only get_hom_condition() and get_value() are needed realistically. Other methods are
implemented in the :ref:`abstractScalar<abstract-scalar-dataloader>`, 
:ref:`abstractVector<abstract-vector-dataloader>`, 
and :ref:`abstractLUT<abstract-lut-dataloader>` dataloaders.

.. automodule:: polar_route.Dataloaders.DataLoaderInterface
   :members: