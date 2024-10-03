************
Installation
************

In this section we will outline the necessary steps for installing the PolarRoute software package. Any installation of PolarRoute will also include `MeshiPhi <https://github.com/antarctica/MeshiPhi>`_; a package designed to
discretise the world from heterogeneous data sources. PolarRoute requires a pre-existing installation of Python 3.8 or higher.

Installing PolarRoute
#####################

PolarRoute is available from PyPI and can be installed by running:
::

    pip install polar-route

For development purposes you can install PolarRoute by downloading the source code from GitHub:
::

    git clone https://github.com/Antarctica/PolarRoute
    pip install -e ./PolarRoute

Use of :code:`-e` is optional, depending on whether you want to be able to edit the installed copy of the package.

In order to run the test suite you will also need to include the `[test]` flag to install the optional test dependencies:
::

    pip install -e ./PolarRoute[test]

Installing GeoPlot
#####################

Plotting functionality for the outputs of PolarRoute is provided by the `GeoPlot <https://github.com/antarctica/GeoPlot>`_ package, also developed at BAS.

Geoplot is available from PyPI and can be installed by running:
::

    pip install bas-geoplot


Installing GDAL (Optional)
##########################

The PolarRoute software has GDAL as an optional requirement. It is only used when exporting TIFF images, 
so if this is not useful to you, you can skip this step. It is not always trivial and is a common source of problems.
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
    sudo apt update
    sudo apt install gdal-bin libgdal-dev
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
