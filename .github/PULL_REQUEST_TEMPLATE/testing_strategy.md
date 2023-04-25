# Testing Strategy
When updating any files within the PolarRoute repository, tests must be run to ensure that the core functionality of the software remains unchanged. To allow for validation of changes, a suite of regression tests have been provided in the folder `tests/regression_tests/...`. These tests attempt to rebuild existing test cases using the changed code and compares these rebuilt outputs to the reference test files. If any differences are found, the tests will fail. 

Evidence that all the required regression tests have passed needs to be submitted as part of a pull request. This should be in the form of a `pytest_output.txt` attached the pull request. 

Pull requests will not be accepted unless all required regression tests pass. 

## Mesh Construction

### Files
`aggregated_cellBox.py`  
`boundary.py`  
`cellbox.py`  
`direction.py`  
`environment_mesh.py`  
`mesh_builder.py`  
`mesh.py`  
`metadata.py`  
`neighbour_graph.py`  

### Tests
`tests/regression_tests/test_mesh_construcion.py`

## Vessel Performance Modelling

### Files
`abstract_vessel.py`  
`vessel_factory.py`  
`vessel_performance_modeller.py`  

### Tests
`tests/regression_tests/test_vessel_performance.py`  

## Route Planning

### Files
`crossing.py`  
`route_planner.py`  

### Tests
`tests/regression_tests/test_route_planner.py`  

## Testing files
Some updates to PolarRoute may result in changes to meshes calculated in our tests suite (*such as adding additional attributes to the cellbox object*). These changes will cause the test suite to fail, though the mode of failure should be predictable. 

Details of these failed tests should be submitted as part of the pull request in the form of a `pytest_failures.txt` file, as well as reasoning for a cause of the failures.

If the changes made are valid, the test files should be updated so-as the tests pass again, and evidence of the updated tests passing also submitted with the pull request. 

### Files
`test_regression.py`  
`test_boundary.py`  
`test_cellbox.py`  
`test_env_mesh.py`  
`test_neighbour_graph.py`  
`test_vessel_performance.py`  

`create_mesh.output2013_4_80.py`  
`create_mesh.output2016_6_80.py`  
`create_mesh.output2019_6_80.py`  

`add_vehicle.output2013_4_80.py`  
`add_vehicle.output2016_6_80.py`  
`add_vehicle.output2019_6_80.py`  