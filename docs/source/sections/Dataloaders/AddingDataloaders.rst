.. _adding-dataloaders:

Adding New Dataloaders
============================

Adding to the repository
------------------------

Each dataloader is to be implemented as a separate object for the Environmental mesh to interface with. 
The general workflow for creating a new dataloader is as follows:

#. Choose an approriate dataloader type (see :ref:`Dataloader Types`).
#. Create a new file under :code:`polar_route/DataLoaders/{dataloader-type}` with an appropriate name.
#. Create :code:`import_data()` and (optianally) :code:`add_default_params()` methods. Examples of how to do this are shown on the :ref:`abstractScalar<abstract-scalar-dataloader-index>` and :ref:`abstractVector<abstract-vector-dataloader-index>` pages.
#. Add a new entry to the dataloader factory object, within :code:`polar_route/Dataloaders/Factory.py`. Instructions on how to do so are shown in :ref:`dataloader-factory`

After performing these actions, the dataloader should be ready to go. It is useful for debugging purposes 
to create the dataloader object from within :code:`polar_route/Dataloaders/Factory.py` (e.g. within
:code:`if __name__=='__main__':` ) and test its functionality before deploying it.



Adding within iPython Notebooks
-------------------------------

If you do not wish to modify the repo to add a dataloader, you may add one into the mesh by calling the 
:code:`add_dataloader()` method of :ref:`MeshBuilder`.

An example of how to do this is detailed below. Assuming you're working out of a Jupyter notebook, the 
basic steps would be to

#. Create a dataloader
   ::
      
      # Import the abstract dataloader as the base class
      from polar_route.dataloaders.scalar.abstract_scalar import ScalarDataLoader
      
      # Set up dataloader in the same way as the existing dataloaders
      class MyDataLoader(ScalarDataLoader):
         # Only user defined function required
         def import_data(self, bounds):
            # Read in data
            if len(self.files) == 1:    data = xr.open_dataset(self.files[0])
            else:                       data = xr.open_mfdataset(self.files)
            # Trim data to boundary
            data = self.trim_datapoints(bounds, data=data)

            return data
   
#. Create a dictionary of parameters to initialise the dataloader
   ::
      
      # Params formatted same way as dataloaders in config
      params = {
         'files': [  
            'PATH_TO_FILE_1',
            'PATH_TO_FILE_2',
            ... # Populate with as many files as you need
         ],
         'data_name': 'my_data',
         'splitting_conditions':[
            {
            'my_data':{
               'threshold': 0.5,
               'upper_bound': 0.9,
               'lower_bound': 0.1
               }
            }
         ]
      }

#. Initialise an Environmental Mesh
   ::

      import json
      from polar_route import MeshBuilder

      # Config to initialise mesh from
      with open('config.json', 'r') as fp:
         config = json.load(fp)

      # Build a mesh from the config
      mesh_builder = MeshBuilder(config)
      env_mesh = mesh_builder.build_environmental_mesh()

#. Add dataloader to mesh
   ::

      # Set up bounds of data in dataloader
      from polar_route import Boundary
      bounds = Boundary.from_json(config)

      # Add dataloader to mesh builder and regenerate mesh
      modified_builder = mesh_builder.add_dataloader(MyDataLoader, params, bounds)
      modified_mesh = modified_builder.build_environmental_mesh()


