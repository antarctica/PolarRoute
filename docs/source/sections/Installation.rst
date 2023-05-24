************
Installation
************

In this section we will outline the installation steps for installing the software package on a corresponding OS. 

The first stage is installing a version of Python 3.9, if you don't have a working version. We suggest installing a working Anaconda distribution from https://www.anaconda.com/products/distribution#macos following the instructions on that page.

Windows
#######
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

**NOTE**

For GDAL (one of the required dependencies for the project), you may try the following commands to avoid any installation issues:
::
   
    pipwin install GDAL
    pip install -r requirements.txt
    python setup.py install


Linux/MacOS
###########

The PolarRoute software can be installed on Linux/MacOS by running one of the two following commands.

Github:
::

    python setup.py install

Pip: 
::

    pip install polar-route


**NOTE**

For GDAL (one of the required dependencies for the project), you may try the following commands to avoid any installation issues:
::

   
    sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update
    sudo apt-get update
    sudo apt-get install gdal-bin
    sudo apt-get install libgdal-dev
    export CPLUS_INCLUDE_PATH=/usr/include/gdal
    export C_INCLUDE_PATH=/usr/include/gdal
    pip install GDAL==$(gdal-config --version)

