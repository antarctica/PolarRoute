"""
Unit tests for the generic vessel classes and the vessel factory.

Individual resistance and consumption model behaviour is covered in
test_vessel_performance_models.py.
"""

import pytest
import json
from polar_route.vessel_performance.vessels.generic_vessels import Ship, Glider, AUV, Aircraft
from polar_route.vessel_performance.vessel_factory import VesselFactory


# Mock cellbox for testing
class MockCellbox:
    """Lightweight mock cellbox for testing without MeshiPhi dependency."""
    def __init__(self, agg_data=None):
        self.id = "test_cell_0"
        self.agg_data = agg_data or {}


@pytest.fixture
def ice_cellbox():
    """Cellbox with ice conditions."""
    return MockCellbox({
        "SIC": 50.0,
        "thickness": 1.0,
        "density": 900.0,
        "elevation": -100.0
    })


# ============================================================================
# Generic Vessel Tests
# ============================================================================

class TestShip:
    """Tests for generic Ship class."""
    
    @pytest.fixture
    def ship_config(self):
        """SDA-equivalent ship configuration."""
        with open('examples/vessel_config/SDA.config.json', 'r') as f:
            return json.load(f)
    
    def test_ship_initialization(self, ship_config):
        """Test that ship initializes with models."""
        ship = Ship(ship_config)
        assert ship.max_speed == 26.5
        assert len(ship.resistance_models) == 2
        assert ship.consumption_model is not None
    
    def test_ship_model_performance(self, ship_config, ice_cellbox):
        """Test that ship calculates performance values."""
        ship = Ship(ship_config)
        ice_cellbox.agg_data["speed"] = ship.max_speed
        performance = ship.model_performance(ice_cellbox)
        
        assert "speed" in performance
        assert "resistance" in performance
        assert "fuel" in performance
        assert len(performance["speed"]) == 8
    
    def test_ship_model_accessibility(self, ship_config):
        """Test ship accessibility checks."""
        ship = Ship(ship_config)
        
        # Test land detection
        land_cellbox = MockCellbox({"elevation": 10.0})
        access = ship.model_accessibility(land_cellbox)
        assert access["land"] is True
        assert access["inaccessible"] is True
        
        # Test navigable water
        water_cellbox = MockCellbox({"elevation": -100.0, "SIC": 0.0})
        access = ship.model_accessibility(water_cellbox)
        assert access["inaccessible"] is False


class TestGlider:
    """Tests for generic Glider class."""
    
    @pytest.fixture
    def glider_config(self):
        """Slocum-equivalent glider configuration."""
        with open('examples/vessel_config/Slocum.config.json', 'r') as f:
            return json.load(f)
    
    def test_glider_initialization(self, glider_config):
        """Test that glider initializes correctly."""
        glider = Glider(glider_config)
        assert glider.max_speed == 1.25
        assert glider.consumption_model is not None
    
    def test_glider_model_performance(self, glider_config):
        """Test that glider calculates battery consumption."""
        glider = Glider(glider_config)
        cellbox = MockCellbox({"elevation": -50.0, "speed": 1.0})
        performance = glider.model_performance(cellbox)
        
        assert "speed" in performance
        assert "battery" in performance
        assert len(performance["speed"]) == 8
        assert len(performance["battery"]) == 8


# ============================================================================
# Vessel Factory Tests
# ============================================================================

class TestVesselFactory:
    """Tests for vessel factory."""
    
    def test_create_ship_from_config(self):
        """Test creating ship from config file."""
        with open('examples/vessel_config/SDA.config.json', 'r') as f:
            config = json.load(f)
        vessel = VesselFactory.get_vessel(config)
        assert isinstance(vessel, Ship)
    
    def test_create_glider_from_config(self):
        """Test creating glider from config file."""
        with open('examples/vessel_config/Slocum.config.json', 'r') as f:
            config = json.load(f)
        vessel = VesselFactory.get_vessel(config)
        assert isinstance(vessel, Glider)
    
    def test_invalid_vessel_class_raises_error(self):
        """Test that invalid vessel_class raises ValueError."""
        config = {"vessel_class": "invalid", "max_speed": 10, "unit": "km/hr"}
        with pytest.raises(ValueError, match="Unknown vessel_class"):
            VesselFactory.get_vessel(config)
