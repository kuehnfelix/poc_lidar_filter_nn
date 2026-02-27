import numpy as np
import pytest

from lidar_sim.core.ray import Ray
from lidar_sim.core.types import ObjectType
from lidar_sim.geometry.box import BoxObject
from lidar_sim.geometry.cylinder import CylinderObject
from lidar_sim.geometry.cone import ConeObject
from lidar_sim.geometry.ground import GroundPlane


# ============================================================================
# BoxObject Tests
# ============================================================================

def test_box_simple_intersection():
    """Test basic box intersection with ray along x-axis."""
    box = BoxObject(
        object_id=1,
        center=np.array([5.0, 0.0, 0.0]),
        size=np.array([2.0, 2.0, 2.0]),
        orientation=np.eye(3)
    )
    ray = Ray(np.array([0.0, 0.0, 0.0]), 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    hit = box.intersect(ray)
    
    assert hit.hit is True
    assert hit.object_id == 1
    assert hit.object_type == ObjectType.BOX
    assert np.isclose(hit.distance, 4.0)


def test_box_no_intersection():
    """Test that ray missing the box returns no hit."""
    box = BoxObject(
        object_id=1,
        center=np.array([5.0, 0.0, 0.0]),
        size=np.array([2.0, 2.0, 2.0]),
        orientation=np.eye(3)
    )
    ray = Ray(np.array([0.0, 10.0, 0.0]), 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    hit = box.intersect(ray)
    
    assert hit.hit is False


def test_box_ray_from_inside():
    """Test that ray originating from inside the box returns no hit."""
    box = BoxObject(
        object_id=1,
        center=np.array([0.0, 0.0, 0.0]),
        size=np.array([2.0, 2.0, 2.0]),
        orientation=np.eye(3)
    )
    ray = Ray(np.array([0.0, 0.0, 0.0]), 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    hit = box.intersect(ray)
    
    assert hit.hit is False


def test_box_intersection_normal():
    """Test that normal vector is computed correctly."""
    box = BoxObject(
        object_id=1,
        center=np.array([0.0, 0.0, 0.0]),
        size=np.array([2.0, 2.0, 2.0]),
        orientation=np.eye(3)
    )
    ray = Ray(np.array([-5.0, 0.0, 0.0]), 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    hit = box.intersect(ray)
    
    assert hit.hit is True
    assert np.allclose(hit.normal, [-1.0, 0.0, 0.0]) or np.allclose(hit.normal, [1.0, 0.0, 0.0])


def test_box_with_rotation():
    """Test box intersection with rotated orientation."""
    # Rotation 45 degrees around z-axis
    angle = np.pi / 4
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    box = BoxObject(
        object_id=1,
        center=np.array([0.0, 0.0, 0.0]),
        size=np.array([2.0, 2.0, 2.0]),
        orientation=rotation
    )
    ray = Ray(np.array([-5.0, 0.0, 0.0]), 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    hit = box.intersect(ray)
    
    assert hit.hit is True


# ============================================================================
# CylinderObject Tests
# ============================================================================

def test_cylinder_simple_intersection():
    """Test basic cylinder intersection."""
    cylinder = CylinderObject(
        object_id=2,
        base_center=np.array([5.0, 0.0, 0.0]),
        radius=1.0,
        height=2.0
    )
    ray = Ray(np.array([0.0, 0.0, 1.0]), 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    hit = cylinder.intersect(ray)
    
    assert hit.hit is True
    assert hit.object_id == 2
    assert hit.object_type == ObjectType.CYLINDER


def test_cylinder_no_intersection():
    """Test ray missing the cylinder."""
    cylinder = CylinderObject(
        object_id=2,
        base_center=np.array([5.0, 0.0, 0.0]),
        radius=1.0,
        height=2.0
    )
    ray = Ray(np.array([0.0, 10.0, 1.0]), 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    hit = cylinder.intersect(ray)
    
    assert hit.hit is False


def test_cylinder_above_height():
    """Test ray passing above cylinder height."""
    cylinder = CylinderObject(
        object_id=2,
        base_center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        height=1.0
    )
    ray = Ray(np.array([-5.0, 0.0, 2.0]), 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    hit = cylinder.intersect(ray)
    
    assert hit.hit is False


def test_cylinder_below_base():
    """Test ray passing below cylinder base."""
    cylinder = CylinderObject(
        object_id=2,
        base_center=np.array([0.0, 0.0, 1.0]),
        radius=1.0,
        height=1.0
    )
    ray = Ray(np.array([-5.0, 0.0, 0.0]), 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    hit = cylinder.intersect(ray)
    
    assert hit.hit is False


def test_cylinder_normal_vector():
    """Test that cylinder normal vector is computed correctly."""
    cylinder = CylinderObject(
        object_id=2,
        base_center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        height=2.0
    )
    ray = Ray(np.array([-5.0, 0.0, 1.0]), 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    hit = cylinder.intersect(ray)
    
    assert hit.hit is True
    # Normal should point radially outward (z component should be 0)
    assert np.isclose(hit.normal[2], 0.0)


# ============================================================================
# ConeObject Tests
# ============================================================================

def test_cone_simple_intersection():
    """Test basic cone intersection."""
    cone = ConeObject(
        object_id=3,
        position=np.array([5.0, 0.0, 0.0]),
        height=2.0,
        radius_base=1.0
    )
    ray = Ray(np.array([0.0, 0.0, 1.0]), 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    hit = cone.intersect(ray)
    
    assert hit.hit is True
    assert hit.object_id == 3
    assert hit.object_type == ObjectType.CONE


def test_cone_no_intersection():
    """Test ray missing the cone."""
    cone = ConeObject(
        object_id=3,
        position=np.array([5.0, 0.0, 0.0]),
        height=2.0,
        radius_base=1.0
    )
    ray = Ray(np.array([0.0, 10.0, 1.0]), 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    hit = cone.intersect(ray)
    
    assert hit.hit is False


def test_cone_above_apex():
    """Test ray passing above cone apex."""
    cone = ConeObject(
        object_id=3,
        position=np.array([0.0, 0.0, 0.0]),
        height=1.0,
        radius_base=1.0
    )
    ray = Ray(np.array([-5.0, 0.0, 2.0]), 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    hit = cone.intersect(ray)
    
    assert hit.hit is False


def test_cone_below_base():
    """Test ray passing below cone base."""
    cone = ConeObject(
        object_id=3,
        position=np.array([0.0, 0.0, 1.0]),
        height=1.0,
        radius_base=1.0
    )
    ray = Ray(np.array([-5.0, 0.0, 0.0]), 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    hit = cone.intersect(ray)
    
    assert hit.hit is False


# ============================================================================
# GroundPlane Tests
# ============================================================================

def test_ground_simple_intersection():
    """Test basic ground plane intersection."""
    ground = GroundPlane(z=0.0)
    ray = Ray(np.array([0.0, 0.0, 5.0]), 0.0, 0.0, np.array([0.0, 0.0, -1.0]))
    hit = ground.intersect(ray)
    
    assert hit.hit is True
    assert hit.object_id == 0
    assert hit.object_type == ObjectType.GROUND
    assert np.isclose(hit.distance, 5.0)


def test_ground_intersection_position():
    """Test that ground intersection position is correct."""
    ground = GroundPlane(z=0.0)
    ray = Ray(np.array([3.0, 4.0, 10.0]), 0.0, 0.0, np.array([0.0, 0.0, -1.0]))
    hit = ground.intersect(ray)
    
    assert hit.hit is True
    assert np.allclose(hit.position, [3.0, 4.0, 0.0])


def test_ground_no_intersection_parallel():
    """Test that ray parallel to ground doesn't intersect."""
    ground = GroundPlane(z=0.0)
    ray = Ray(np.array([0.0, 0.0, 5.0]), 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    hit = ground.intersect(ray)
    
    assert hit.hit is False


def test_ground_behind_ray():
    """Test that ground behind ray origin returns no hit."""
    ground = GroundPlane(z=0.0)
    ray = Ray(np.array([0.0, 0.0, -5.0]), 0.0, 0.0, np.array([0.0, 0.0, -1.0]))
    hit = ground.intersect(ray)
    
    assert hit.hit is False


def test_ground_normal_vector():
    """Test that ground normal is always pointing up."""
    ground = GroundPlane(z=5.0)
    ray = Ray(np.array([0.0, 0.0, 10.0]), 0.0, 0.0, np.array([0.0, 0.0, -1.0]))
    hit = ground.intersect(ray)
    
    assert hit.hit is True
    assert np.allclose(hit.normal, [0.0, 0.0, 1.0])


def test_ground_elevated():
    """Test intersection with elevated ground plane."""
    ground = GroundPlane(z=5.0)
    ray = Ray(np.array([2.0, 3.0, 10.0]), 0.0, 0.0, np.array([0.0, 0.0, -1.0]))
    hit = ground.intersect(ray)
    
    assert hit.hit is True
    assert np.isclose(hit.distance, 5.0)
    assert np.allclose(hit.position, [2.0, 3.0, 5.0])


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================

def test_box_intersection_at_edge():
    """Test intersection precisely at box edge."""
    box = BoxObject(
        object_id=1,
        center=np.array([0.0, 0.0, 0.0]),
        size=np.array([2.0, 2.0, 2.0]),
        orientation=np.eye(3)
    )
    ray = Ray(np.array([-10.0, 1.0, 1.0]), 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    hit = box.intersect(ray)
    
    assert hit.hit is True


def test_cylinder_parallel_to_axis():
    """Test cylinder intersection with ray parallel to axis."""
    cylinder = CylinderObject(
        object_id=2,
        base_center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        height=2.0
    )
    ray = Ray(np.array([0.5, 0.0, -5.0]), 0.0, 0.0, np.array([0.0, 0.0, 1.0]))
    hit = cylinder.intersect(ray)
    
    assert hit.hit is True
