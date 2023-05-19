# Testing Strategy
When updating any files within the PolarRoute repository, tests must be run to ensure that the core functionality of the software remains unchanged. To allow for validation of changes, a suite of regression tests have been provided in the folder `tests/regression_tests/...`. These tests attempt to rebuild existing test cases using the changed code and compares these rebuilt outputs to the reference test files. If any differences are found, the tests will fail. 

Evidence that all the required regression tests have passed needs to be submitted as part of a pull request. This should be in the form of a `pytest_output.txt` attached to the pull request. 

Pull requests will not be accepted unless all required regression tests pass. 

## Mesh Construction

| **Files altered**          | **Tests**                             |
|----------------------------|---------------------------------------|
| `mesh_builder.py`          | `tests/regression_tests/test_mesh.py` |
| `mesh.py`                  |                                       |
| `neighbour_graph.py`       |                                       |
| `metadata.py`              |                                       |
| `aggregated_cellbox.py`    |                                       |
| `boundary.py`              |                                       |
| `cellbox.py`               |                                       |
| `direction.py`             |                                       |
| `environment_mesh.py`      |                                       |
|                            |                                       |



## Vessel Performance Modelling
| **Files altered**                | **Tests**                              |
|----------------------------------|-----------------------------------------|
| `abstract_vessel.py`             | `tests/regression_tests/test_vessel.py` |
| `vessel_factory.py`              |                                         |
| `vessel_performance_modeller.py` |                                         |
| `SDA.py`                         |                                         |
| `underwater_vessel.py`           |                                         |
| `abstract_ship.py`               |                                         |



## Route Planning

| **Files altered**    | **Tests**                                           |
|----------------------|-----------------------------------------------------|
| `crossing.py`        | `tests/regression_tests/test_routes_dijkstra.py`    |
| `route_planner.py`   | `tests/regression_tests/test_routes_smoothed.py`    |
|                      |                                                     |



## Testing files
Some updates to PolarRoute may result in changes to meshes calculated in our tests suite (*such as adding additional attributes to the cellbox object*). These changes will cause the test suite to fail, though the mode of failure should be predictable. 

Details of these failed tests should be submitted as part of the pull request in the form of a `pytest_failures.txt` file, as well as reasoning for a cause of the failures.

If the changes made are valid, the test files should be updated so-as the tests pass again, and evidence of the updated tests passing also submitted with the pull request. 

### Files
`tests/regression_tests/exmaple_meshes/*` 
`tests/regression_tests/exmaple_routes/*` 
