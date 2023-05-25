.. _dataloader-factory:

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

^^^^^^^
Example
^^^^^^^
In this example, a new scalar dataloader `myScalarDataloader` has been created, and
is located at :code:`polar_route/Dataloaders/Scalar/myScalarDataloader.py`.

The only parameter required by this dataloader is a file to read data from. 'files' 
is passed as a mandatory parameter, as 'file' and 'folder' both get translated into 
a list of files, and stored in params under the key 'files'::

   # Add new import statement for Factory to read
   from polar_route.Dataloaders.Scalar.myScalarDataloader import myScalarDataloader

   ...

   class DataLoaderFactory:
      ...
      def get_dataloader(self, name, bounds, params, min_dp=5):
         ...
         dataloader_requirements = {
            ...
            # Add new dataloaders
            'myscalar':    (myScalarDataloader, ['files'])
            ...
         ...
      ...


To call this dataloader, add an entry in the :code:`config.json` 
file used to generate the mesh. Alternatively, add a folder, or a list of 
individual files::
   
   {
         "loader": "myscalar",
         "params": {
            "file": "PATH_TO_DATA_FILE"   # For a single file
            "folder": "PATH_TO_FOLDER"    # For a folder, must have trailing '/'
            "files":[                     # For a list of individual files
               "PATH_TO_FILE_1",
               "PATH_TO_FILE_2",
               ...
            ]
         }
   }

^^^^^^^^^^^^^^^^^^^^^^^^^
Dataloader Factory Object
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: polar_route.dataloaders.factory
   :members: