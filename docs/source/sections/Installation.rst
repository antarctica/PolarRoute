********
Installation
********

In this section we will outline the installation steps for installing the software package on a corresponding OS. 

The first stage is installing a version of Python 3.9, if you don't have a working version. We suggest installing a working Anaconda distribution from https://www.anaconda.com/products/distribution#macos following the instructions on that page.

Windows
##############
The PolarRoute software requires GDAL files to be installed. The PolarRoute software can be installed on Windows by running one of the two following commands.

Github:
::
    pip install pipwin
    pipwin install gdal
    pipwin install fiona
    python setup.py install

Pip: 
::
    pip install pipwin
    pipwin install gdal
    pipwin install fiona
    pip install polar-route


Linux/MacOS
##############

The PolarRoute software can be installed on Linux/MacOS by running one of the two following commands.

Github:
::
    python setup.py install

Pip: 
::
    pip install polar-route
