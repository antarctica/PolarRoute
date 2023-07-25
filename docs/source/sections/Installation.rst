************
Installation
************

In this section we will outline the installation steps for installing the software package on a corresponding OS. 

The first stage is installing a version of Python 3.9, if you don't have a working version. We suggest installing a working Anaconda distribution from https://www.anaconda.com/products/distribution#macos following the instructions on that page.

Installing PolarRoute
#####################

The PolarRoute software can be installed on Windows/Linux/MacOS by running one of the two following commands.

Github:
::

    git clone https://https://github.com/antarctica/PolarRoute.git
    python setup.py install

Pip: 
::

    pip install polar-route


Installing GDAL
###############

The PolarRoute software has GDAL as an optional requirement. It is only used when exporting TIFF images, 
so if this is not useful to you, we would recommend steering clear. It is not trivial and is a common source of problems.
With that said, below are instructions for various operating systems.

Windows
*******

.. note:: 
    We assume a version of Windows 10 or higher, with a working version of Python 3.9 including pip installed. 
    We recommend installing PolarRoute into a virtual environment.

Windows:

::

    pip install pipwin # pipwin is a package that allows for easy installation of windows binaries
    pipwin install gdal
    pipwin install fiona


Linux/MacOS
***********

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