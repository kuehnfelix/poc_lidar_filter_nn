import numpy as np
import pytest

from lidar_sim.core.ray import Ray
from lidar_sim.core.types import ObjectType
from lidar_sim.geometry.box import BoxObject
from lidar_sim.geometry.cylinder import CylinderObject
from lidar_sim.geometry.ground import GroundPlane
from lidar_sim.lidar.lidar_model import LiDARModel
from lidar_sim.lidar.scan_pattern import ScanPattern
from lidar_sim.scene.scene import Scene


# ============================================================================
# Custom ScanPattern for Testing
# ============================================================================

class FixedScanPattern(ScanPattern):
    """A simple scan pattern that yields a fixed set of angles."""
    
    def __init__(self, angles: list):
        """
        Parameters
        ----------
        angles : list of tuples
            List of (azimuth, elevation) angle pairs in radians.
        """
        self.angles = angles
        self.index = 0
    
    def __iter__(self):
        self.index = 0
        return self
    
    def __next__(self):
        if self.index >= len(self.angles):
            raise StopIteration
        result = self.angles[self.index]
        self.index += 1
        return result


# ============================================================================
# Scene Tests
# ============================================================================

def test_scene_empty():
    """Test empty scene."""
    scene = Scene()
    assert len(scene.objects) == 0


def test_scene_add_object():
    """Test adding objects to scene."""
    scene = Scene()
    ground = GroundPlane(z=0.0)
    scene.add_object(ground)
    
    assert len(scene.objects) == 1
    assert scene.objects[0] == ground


def test_scene_add_multiple_objects():
    """Test adding multiple objects to scene."""
    scene = Scene()
    ground = GroundPlane(z=0.0)
    box = BoxObject(1, np.array([0.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0]), np.eye(3))
    cylinder = CylinderObject(2, np.array([3.0, 0.0, 0.0]), 1.0, 2.0)
    
    scene.add_object(ground)
    scene.add_object(box)
    scene.add_object(cylinder)
    
    assert len(scene.objects) == 3


def test_scene_intersect_no_objects():
    """Test intersection in empty scene."""
    scene = Scene()
    ray = Ray(np.array([0.0, 0.0, 5.0]), 0.0, 0.0, np.array([0.0, 0.0, -1.0]))
    hit = scene.intersect(ray)
    
    assert hit.hit is False


def test_scene_intersect_single_object():
    """Test intersection with single object."""
    scene = Scene()
    ground = GroundPlane(z=0.0)
    scene.add_object(ground)
    
    ray = Ray(np.array([0.0, 0.0, 5.0]), 0.0, 0.0, np.array([0.0, 0.0, -1.0]))
    hit = scene.intersect(ray)
    
    assert hit.hit is True
    assert hit.object_id == ground.object_id


def test_scene_intersect_closest_object():
    """Test that scene returns closest hit."""
    scene = Scene()
    
    # Add ground at z=0
    ground = GroundPlane(z=0.0)
    scene.add_object(ground)
    
    # Add box at z=2
    box = BoxObject(1, np.array([0.0, 0.0, 2.0]), np.array([1.0, 1.0, 1.0]), np.eye(3))
    scene.add_object(box)
    
    ray = Ray(np.array([0.0, 0.0, 10.0]), 0.0, 0.0, np.array([0.0, 0.0, -1.0]))
    hit = scene.intersect(ray)
    
    assert hit.hit is True
    # Box should be closer than ground
    assert hit.object_id == box.object_id


def test_scene_intersect_miss_all():
    """Test ray that misses all objects."""
    scene = Scene()
    ground = GroundPlane(z=0.0)
    scene.add_object(ground)
    
    ray = Ray(np.array([0.0, 0.0, 5.0]), 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    hit = scene.intersect(ray)
    
    assert hit.hit is False


# ============================================================================
# LiDARModel Tests
# ============================================================================

def test_lidar_generate_rays_identity_pose():
    """Test ray generation with identity pose."""
    pattern = FixedScanPattern([(0.0, 0.0)])
    model = LiDARModel(pattern)
    
    pose = np.eye(4)
    rays = model.generate_rays(pose)
    
    # Should generate 5 rays (one for each channel)
    assert len(rays) == 5
    
    # All rays should originate at origin
    for ray in rays:
        assert np.allclose(ray.origin, [0.0, 0.0, 0.0])
    
    # First ray (central channel at 0 elevation) should point along +X
    assert np.allclose(rays[2].direction, [1.0, 0.0, 0.0])


def test_lidar_generate_rays_translated_pose():
    """Test ray generation with translated pose."""
    pattern = FixedScanPattern([(0.0, 0.0)])
    model = LiDARModel(pattern)
    
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    rays = model.generate_rays(pose)
    
    # All rays should originate at translated position
    for ray in rays:
        assert np.allclose(ray.origin, [1.0, 2.0, 3.0])


def test_lidar_generate_rays_rotated_pose():
    """Test ray generation with rotated pose."""
    pattern = FixedScanPattern([(0.0, 0.0)])
    model = LiDARModel(pattern)
    
    # 90-degree rotation around z-axis
    angle = np.pi / 2
    pose = np.array([
        [np.cos(angle), -np.sin(angle), 0.0, 0.0],
        [np.sin(angle), np.cos(angle), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    rays = model.generate_rays(pose)
    
    # Central ray should point along +Y after rotation
    assert np.allclose(rays[2].direction, [0.0, 1.0, 0.0], atol=1e-5)


def test_lidar_generate_rays_multiple_samples():
    """Test ray generation with multiple azimuth samples."""
    angles = [(0.0, 0.0), (np.pi / 2, 0.0)]
    pattern = FixedScanPattern(angles)
    model = LiDARModel(pattern)
    
    pose = np.eye(4)
    rays = model.generate_rays(pose)
    
    assert len(rays) == 5


def test_lidar_generate_rays_with_elevation():
    """Test ray generation with non-zero elevation."""
    pattern = FixedScanPattern([(0.0, np.pi / 4)])
    model = LiDARModel(pattern)
    
    pose = np.eye(4)
    rays = model.generate_rays(pose)
    
    assert len(rays) == 5
    
    # Central ray should have upward z-component
    for ray in rays:
        assert ray.direction[2] > 0.0


def test_lidar_invalid_pose_shape():
    """Test that invalid pose shape raises error."""
    pattern = FixedScanPattern([(0.0, 0.0)])
    model = LiDARModel(pattern)
    
    # 3x3 matrix instead of 4x4
    pose = np.eye(3)
    
    with pytest.raises(ValueError):
        model.generate_rays(pose)


def test_lidar_pose_is_not_modified():
    """Test that generate_rays does not modify the input pose."""
    pattern = FixedScanPattern([(0.0, 0.0)])
    model = LiDARModel(pattern)
    
    pose = np.eye(4)
    pose_copy = pose.copy()
    
    model.generate_rays(pose)
    
    assert np.allclose(pose, pose_copy)


# ============================================================================
# LiDARModel Integration Tests
# ============================================================================

def test_lidar_measure_single_with_ground():
    """Test LiDAR measurement hitting ground."""
    scene = Scene()
    ground = GroundPlane(z=0.0)
    scene.add_object(ground)
    
    pattern = FixedScanPattern([(0.0, -np.pi / 4)])  # Downward pointing
    model = LiDARModel(pattern)
    
    pose = np.eye(4)
    pose[:3, 3] = [0.0, 0.0, 5.0]
    
    hits = model.measure_single(scene, pose)
    
    assert len(hits) == 5
    # At least some rays should hit the ground
    hit_count = sum(1 for hit in hits if hit.hit)
    assert hit_count > 0


def test_lidar_measure_single_no_hits():
    """Test LiDAR measurement with no intersections."""
    scene = Scene()
    ground = GroundPlane(z=0.0)
    scene.add_object(ground)
    
    pattern = FixedScanPattern([(0.0, np.pi / 4)])  # Upward pointing
    model = LiDARModel(pattern)
    
    pose = np.eye(4)
    pose[:3, 3] = [0.0, 0.0, 5.0]
    
    hits = model.measure_single(scene, pose)
    
    assert len(hits) == 5
    # All rays pointing up should miss the ground
    for hit in hits:
        assert hit.hit is False


def test_lidar_measure_single_box():
    """Test LiDAR measurement hitting a box."""
    scene = Scene()
    box = BoxObject(1, np.array([10.0, 0.0, 2.0]), np.array([2.0, 2.0, 2.0]), np.eye(3))
    scene.add_object(box)
    
    pattern = FixedScanPattern([(0.0, 0.0)])
    model = LiDARModel(pattern)
    
    pose = np.eye(4)
    pose[:3, 3] = [0.0, 0.0, 2.0]
    
    hits = model.measure_single(scene, pose)
    
    assert len(hits) == 5
    # Central ray should hit the box
    hit_count = sum(1 for hit in hits if hit.hit)
    assert hit_count > 0


# ============================================================================
# Ray Storage Tests
# ============================================================================

def test_ray_stores_azimuth_elevation():
    """Test that Ray stores azimuth and elevation angles."""
    ray = Ray(np.array([0.0, 0.0, 0.0]), 0.5, 0.3, np.array([1.0, 0.0, 0.0]))
    
    assert np.isclose(ray.azimuth, 0.5)
    assert np.isclose(ray.elevation, 0.3)


def test_ray_normalized_direction():
    """Test that Ray normalizes direction."""
    ray = Ray(np.array([0.0, 0.0, 0.0]), 0.0, 0.0, np.array([3.0, 4.0, 0.0]))
    
    norm = np.linalg.norm(ray.direction)
    assert np.isclose(norm, 1.0)
    assert np.allclose(ray.direction, [0.6, 0.8, 0.0])


# ============================================================================
# Multi-Object Scene Tests
# ============================================================================

def test_scene_with_multiple_object_types():
    """Test scene with multiple object types."""
    scene = Scene()
    
    ground = GroundPlane(z=0.0)
    box = BoxObject(1, np.array([5.0, 0.0, 1.0]), np.array([2.0, 2.0, 2.0]), np.eye(3))
    cylinder = CylinderObject(2, np.array([10.0, 0.0, 0.0]), 1.0, 2.0)
    
    scene.add_object(ground)
    scene.add_object(box)
    scene.add_object(cylinder)
    
    ray = Ray(np.array([0.0, 0.0, 5.0]), 0.0, 0.0, np.array([1.0, 0.0, -0.5]))
    hit = scene.intersect(ray)
    
    assert hit.hit is True


def test_scene_intersect_returns_closest():
    """Test that scene.intersect returns the closest hit."""
    scene = Scene()
    
    box1 = BoxObject(1, np.array([5.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), np.eye(3))
    box2 = BoxObject(2, np.array([10.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), np.eye(3))
    
    scene.add_object(box1)
    scene.add_object(box2)
    
    ray = Ray(np.array([0.0, 0.0, 0.0]), 0.0, 0.0, np.array([1.0, 0.0, 0.0]))
    hit = scene.intersect(ray)
    
    assert hit.hit is True
    assert hit.object_id == 1  # Closer box


# ============================================================================
# Edge Cases
# ============================================================================

def test_lidar_channels_have_correct_angles():
    """Test that LiDAR channels have the expected elevation angles."""
    pattern = FixedScanPattern([(0.0, 0.0)])
    model = LiDARModel(pattern)
    
    pose = np.eye(4)
    rays = model.generate_rays(pose)
    
    assert len(rays) == 5
    
    # Verify the channel angles are encoded in the rays
    expected_azimuths = [
        np.deg2rad(-48),
        np.deg2rad(-24),
        np.deg2rad(0.0),
        np.deg2rad(24),
        np.deg2rad(48),
    ]
    
    for ray, expected_azimuth in zip(rays, expected_azimuths):
        assert np.isclose(ray.azimuth, expected_azimuth, atol=1e-5)
        assert np.isclose(ray.elevation, 0.0, atol=1e-5)
