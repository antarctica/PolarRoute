"""
Vessel performance modeling for polar route planning.

This package provides tools for calculating vessel performance characteristics
including resistance forces and fuel/battery consumption based on environmental conditions.
"""

# Ensure submodules are importable
from polar_route.vessel_performance import models
from polar_route.vessel_performance import vessels

__all__ = ['models', 'vessels']
