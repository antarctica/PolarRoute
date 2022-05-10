import codecs
import glob
import inspect
import os
import re
from setuptools import setup
import sys
import time
import numpy.distutils.misc_util

# Directory of the current file
SETUP_DIRECTORY = os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe())))

LOCAL_PATH = os.path.join(SETUP_DIRECTORY, "setup.py")

NAME    = "RoutePlanner"
VERSION = '0.0.4'

INCLUDE_DIRS = numpy.distutils.misc_util.get_numpy_include_dirs()

META_PATH = os.path.join("RoutePlanner", "__init__.py")
KEYWORDS = ["BAS", "SDT", "DT"]

CLASSIFIERS = [
    "Development Status :: Beta",
    "Intended Audience :: BAS",
    "Natural Language :: English",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.9",
    "Topic :: Scientific/Engineering",
]

INSTALL_REQUIRES = [
    'shapely',
    'ipykernel',
    'geopandas',
    'xarray',
    'netCDF4',
    'matplotlib',
    'folium']


def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(SETUP_DIRECTORY, *parts),
                     "rb", "utf-8") as f:
        return f.read()


def find_packages():
    """
    Simple function to find all modules under the current folder.
    """
    modules = []
    for dirpath, _, filenames in os.walk(os.path.join(SETUP_DIRECTORY,
                                                      "RoutePlanner")):
        if "__init__.py" in filenames:
            modules.append(os.path.relpath(dirpath, SETUP_DIRECTORY))
    return [_i.replace(os.sep, ".") for _i in modules]


META_FILE = read(META_PATH)


def find_meta(meta):
    """
    Extract __*meta*__ from META_FILE.
    """
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta),
        META_FILE, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


def setup_package():
    """Setup package"""
    setup(
        name="RoutePlanner",
        version=VERSION,
        description=find_meta("description"),
        long_description=read("README.md"),
        author=find_meta("author"),
        author_email=find_meta("email"),
        maintainer=find_meta("author"),
        maintainer_email=find_meta("email"),
        classifiers=CLASSIFIERS,
        keywords=KEYWORDS,
        packages=find_packages(),
        zip_safe=False,
        install_requires=INSTALL_REQUIRES,
        include_package_data=True,
        include_dirs=INCLUDE_DIRS,
        package_data={"RoutePlanner": ["lib/*.so"]})


    # os.system('make ./.docs/html')
    # os.system('sphinx-build -b rinoh ./.docs/source ./.docs/_build/rinoh')
    # os.system('cp ./.docs/_build/rinoh/pyRoutePlanner.pdf .')
    # os.system('rm -rf ./.docs/_build')
    # os.system('rm -rf ./.docs/build')

if __name__ == "__main__":
    # clean --all does not remove extensions automatically
    if 'clean' in sys.argv and '--all' in sys.argv:
        import shutil
        # delete complete build directory
        path = os.path.join(SETUP_DIRECTORY, 'build')
        try:
            shutil.rmtree(path)
        except Exception:
            pass
        # delete all shared libs from lib directory
        path = os.path.join(SETUP_DIRECTORY, 'RoutePlanner', 'lib')
        for filename in glob.glob(path + os.sep + '*.pyd'):
            try:
                os.remove(filename)
            except Exception:
                pass
        for filename in glob.glob(path + os.sep + '*.so'):
            try:
                os.remove(filename)
            except Exception:
                pass
    else:
        setup_package()