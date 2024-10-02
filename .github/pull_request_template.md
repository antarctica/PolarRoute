# PolarRoute Pull Request Template

Date: <!--- Include date PR was created -->   
Version Number: <!--- Include version number of PolarRoute the PR will be included in (e.g. 1.0 -> 1.1) -->   
 
## Description of change
<!--- Describe your changes in detail -->

## Fixes # (issue)
<!--- If this PR adds functionality or resolves problems associated with an issue on GitHub, please include a link to the issue -->

# Testing
To ensure that the functionality of the PolarRoute codebase remains consistent throughout the development cycle a testing strategy has been developed, which can be viewed in the document `test/testing_strategy.md`. 
This includes a collection of test files which should be run according to which part of the codebase has been altered in a pull request. Please consult the testing strategy to determine which tests need to be run. 

- [ ] My changes have not altered any of the files listed in the testing strategy

- [ ] My changes result in all required regression tests passing without the need to update test files.  
  
> *list which files have been altered and include a pytest.txt file for each of
> the tests required to be run*
>
> The files which have been changed during this PR can be listed using the command

    git diff --name-only 0.7.x

- [ ] My changes require one or more test files to be updated for all regression tests to pass.   

> *include pytest.txt file showing which tests fail.*  
> *include reasoning as to why your changes cause these tests to fail.* 
>
> Should these changes be valid, relevant test files should be updated.  
> *include pytest.txt file of test passing after test files have been updated.*

# Checklist

- [ ] My code follows [pep8](https://peps.python.org/pep-0008/) style guidelines.  
- [ ] I have commented my code, particularly in hard-to-understand areas.  
- [ ] I have updated the documentation of the codebase where required.  
- [ ] My changes generate no new warnings.   
- [ ] My PR has been made to the `1.1.x` branch (**DO NOT SUBMIT A PR TO MAIN**)  

   
