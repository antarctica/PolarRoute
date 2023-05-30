************
Installation
************

In this section we will outline the installation steps for installing the software package on a corresponding OS. 

The first stage is installing a version of Python 3.9, if you don't have a working version. We suggest installing a working Anaconda distribution from https://www.anaconda.com/products/distribution#macos following the instructions on that page.

Windows
#######
The PolarRoute software requires GDAL files to be installed. The PolarRoute software can be installed on Windows by running one of the two following commands.

.. note:: 
    We assume a version of Windows 10 or higher, with a working version of Python 3.9 including pip installed. 
    We recommend installing PolarRoute into a virtual environment.

Requirements:
::

    pip install pipwin # pipwin is a package that allows for easy installation of windows binaries
    pipwin install gdal
    pipwin install fiona

    pip install -r requirements.txt

Github:
::

    git clone https://https://github.com/antarctica/PolarRoute.git
    python setup.py install

Pip: 
::

    pip install polar-route

Linux/MacOS
###########


Installing GDAL
***************

Requirements:

.. note::
    For GDAL (one of the required dependencies for the project), you may try the following 
    commands to avoid any installation issues:

Ubuntu/Debian:

::
   
    sudo add-apt-repository ppa:ubuntugis/ppa
    sudo apt-get update
    sudo apt-get install gdal-bin libgdal-dev
    export CPLUS_INCLUDE_PATH=/usr/include/gdal
    export C_INCLUDE_PATH=/usr/include/gdal
    pip install GDAL==$(gdal-config --version)


Fedora:

::

    sudo dnf update
    sudo dnf install gdal gdal-devel
    export CPLUS_INCLUDE_PATH=/usr/include/gdal
    export C_INCLUDE_PATH=/usr/include/gdal
    pip install GDAL==$(gdal-config --version)


MacOS (with HomeBrew):

::

    brew install gdal --HEAD
    brew install gdal
    pip install GDAL==$(gdal-config --version)



Installing PolarRoute
*********************

The PolarRoute software can be installed on Linux/MacOS by running one of the two following commands.

Github:
::

    git clone https://https://github.com/antarctica/PolarRoute.git
    python setup.py install

Pip: 
::

    pip install polar-route





