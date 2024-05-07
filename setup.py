from setuptools import setup, find_packages

import polar_route
import polar_route.vessel_performance

def get_content(filename):
    with open(filename, "r") as fh:
        return fh.read()


requirements = get_content("requirements.txt")

setup(
    name=polar_route.__name__,
    version=polar_route.__version__,
    description=polar_route.__description__,
    license=polar_route.__license__,
    long_description=get_content("README.md"),
    long_description_content_type="text/markdown",
    author=polar_route.__author__,
    author_email=polar_route.__email__,
    maintainer=polar_route.__author__,
    maintainer_email=polar_route.__email__,
    url="https://www.github.com/antarctica",
    project_urls={
    },
    classifiers=[el.lstrip() for el in """Development Status :: 3 - Alpha
        Intended Audience :: Science/Research
        Intended Audience :: System Administrators
        License :: OSI Approved :: MIT License
        Natural Language :: English
        Operating System :: OS Independent
        Programming Language :: Python
        Programming Language :: Python :: 3
        Programming Language :: Python :: 3.7
        Topic :: Scientific/Engineering""".split('\n')],
    entry_points={
        'console_scripts': [
            "add_vehicle=polar_route.cli:add_vehicle_cli",
            "optimise_routes=polar_route.cli:optimise_routes_cli",
            "calculate_route=polar_route.cli:calculate_route_cli",
            "resimulate_vehicle=polar_route.cli:resimulate_vehicle_cli",
            "extract_routes=polar_route.cli:extract_routes_cli"],
    },
    keywords=[],
    packages=find_packages(),
    install_requires=requirements,
    tests_require=["pytest"],
    extras_require={
        "tests": get_content("tests/requirements.txt"),
    },
    zip_safe=False,
    include_package_data=True)
