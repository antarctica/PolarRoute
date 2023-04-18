# PolarRoute Pull Request Template

Date: <!--- Include date PR was created -->   
Version Number: <!--- Include version number of PolarRoute PR was made against (0.0.17) -->   
 
## Description of change
<!--- Describe your changes in detail -->

Fixes # (issue)
<!--- If this PR adds functionality/ resolved problems associated with an issue on GitHub, please include a link to the issue -->

# Testing.
To ensure that operational functionality of the PolarRoute codebase remain throughout the development cycle, a testing strategy has been developed which can be viewed in the document `.github/PULL_REQUEST_TEMPLATE/testing_strategy.md`. This includes a collection of test files which should be run dependant of which part of the codebase have been alterd in a pull request. Please consult the testing stragegy to determine which tests need to be run. 

- [ ] My changes result in all required regression tests passing without need to update test files.  

> *include pytest.txt file showing all tests passing.*  

- [ ] My changes require one or more test files to be updated for all regression tests to pass.   

> *include pytest.txt file showing which tests fail.*  
> *include reasoning as to why changes cause these tests to fail.* 
>
> Should these changes be valid, relevant test files should be updated.  
> *include pytest.txt file of test passing after test files have been updated.*

# Checklist.

- [ ] My code follows [pep8](https://peps.python.org/pep-0008/) style guidelines.  
- [ ] I have commented my code, particuarly in hard-to-understand areas.  
- [ ] I have updated the documentation of the codebase where required.  
- [ ] My changes generate no new warnings.   
- [ ] My PR has been made to the `intergration_testing` branch (**DO NOT SUMBIT A PR TO MAIN**)  

   
