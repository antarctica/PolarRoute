******************
Dataloader Factory
******************

The dataloader factory produces dataloader objects based off of parameter 
inputs provided in the config file. The parameters needed in the config are
defined in the :code:`get_dataloader()` method of the factory. At the very
least, a name must be provided to select the dataloader from all those that 
are available.



^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Adding New Dataloader to Factory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two actions must be performed to add a new dataloader to the Factory object.
Optionally, a third may be added if you want to add a new default value for 
a parameter the dataloader requires. The actions are:

#. Import the dataloader
#. Add an entry to the :code:`dataloader_requirements` dictionary
#. (OPTIONAL) Add a default value to :code:`set_default_params()`

^^^^^^^
Example
^^^^^^^
In this example, a new scalar dataloader `myScalarDataloader` has been created, and
is located at :code:`polar_route/Dataloaders/Scalar/myScalarDataloader.py`.
The only parameter required by this dataloader is a file to read data from::

   # Add new import statement for Factory to read
   from polar_route.Dataloaders/Scalar/myScalarDataloader import myScalarDataloader

   ...

   class DataLoaderFactory:
      ...
      def get_dataloader(self, name, bounds, params, min_dp=5):
         ...
         dataloader_requirements = {
            ...
            # Add new dataloader
            'myscalar':    (myScalarDataloader, ['file'])
            ...
         ...
      ...


To call this dataloader, add an entry in the :code:`config.json` 
file used to generate the mesh::
   {
         "loader": "myscalar",
         "params": {
            "file": "PATH_TO_DATA_FILE",
         }
   }

^^^^^^^^^^^^^^^^^^^^^^^^^
Dataloader Factory Object
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: polar_route.Dataloaders.Factory
   :members: