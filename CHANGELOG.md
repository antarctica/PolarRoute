# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed
* Replaced `fiona` with `pyogrio` for KML/GPX/GeoPackage I/O via GeoPandas (requires `geopandas>=1.0`). `pyogrio` ships wheels with GDAL bundled, so no system GDAL is needed to install PolarRoute on Python 3.14.
* Declare dependencies in `pyproject.toml` instead of `requirements.txt` and remove `requirements.txt`.
* Adjust `pip install -e .[dev]` command in CONTRIBUTING.md to use the `dev` dependency group.
* `route_calc()` can now accept a route as a GeoJSON in addition to a DataFrame.
* Extracted geojson-to-DataFrame parsing in `load_route()` into a shared `route_geojson_to_df()` helper, reused by both the json and gpx.

## [1.1.11] - 2026-07-02


### Added

### Changed
* compare keys separately to avoid pytest.approx issue by @thomaszwagerman in https://github.com/bas-logist/PolarRoute/pull/346
* removed Python 3.9, 3.10 support

## [1.1.10] - 2026-04-07

### Added

### Changed
* Replacing logging calls with module logger by @thomaszwagerman in https://github.com/bas-logist/PolarRoute/pull/334
* Simplify testing suite by @thomaszwagerman in https://github.com/bas-logist/PolarRoute/pull/333
* Migrate docs from sphinx to mkdocs material by @davidwyld in https://github.com/bas-logist/PolarRoute/pull/339
* Docs fixes and give polarroute docs their own colour: light blue by @davidwyld in https://github.com/bas-logist/PolarRoute/pull/341
* Linting and pre-commit hooks by @thomaszwagerman in https://github.com/bas-logist/PolarRoute/pull/342
* v1.1.9 by @thomaszwagerman in https://github.com/bas-logist/PolarRoute/pull/332

## New Contributors
* @davidwyld made their first contribution in https://github.com/bas-logist/PolarRoute/pull/339


## [1.1.9] - 2026-02-10

### Added
- Convert docs from Sphinx to MkDocs (#321).
- Loosen MeshiPhi version strictness in requirements.
- Add Slocum and Boaty McBoatface vessel configs.
- Add module logger, replaces logging.*() calls (#331).
- Enable the running of pytest in the main directory (#260).
- Enable running of tests in parallel using pytest-xdist (#260).
- Add usage section to README.md (#321).
- Add CHANGELOG.md, CONTRIBUTING.md, CITATION.cff, Makefile.
- Linting with ruff.
- Pre-commit hooks which checks linting, along with other formatting issues.
- GitHub Actions test.yml with GDAL installation and caching.

### Changed
- Account for SIC values being None, rather than absent in various vessel performance models (#245).
- Parameterise repetitive tests (#260).
- Mark smoothing tests as slow, enable running "not slow" tests (#260).
- Consolidate add tests to dependency group, remove requirements.txt.
- Reduce the amount of duplicate filepaths by adding dynamic discovery of relevant files needed for the tests (#260).

### Fixed
- Fix test failures on Python 3.14 caused by NumPy float64 type comparisons in vessel performance tests.

[Unreleased]: https://github.com/bas-amop/PolarRoute/compare/v1.1.11...HEAD
[1.1.11]: https://github.com/bas-amop/PolarRoute/compare/v1.1.10...v1.1.11
[1.1.10]: https://github.com/bas-amop/PolarRoute/compare/v1.1.9...v1.1.10
[1.1.9]: https://github.com/bas-amop/PolarRoute/releases/tag/v1.1.9
