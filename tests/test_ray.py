import numpy as np
import pytest

from lidar_sim.core.ray import Ray


def test_ray_normalizes_direction():
    r = Ray([0, 0, 0], [0, 0, 10])
    assert np.allclose(r.direction, [0, 0, 1])


def test_ray_origin_and_direction_shapes():
    r = Ray([1, 2, 3], [4, 5, 6])
    assert r.origin.shape == (3,)
    assert r.direction.shape == (3,)


def test_point_at_computes_correct_location():
    r = Ray([1, 2, 3], [1, 0, 0])
    assert np.allclose(r.point_at(5.0), [6.0, 2.0, 3.0])


def test_invalid_direction_raises():
    with pytest.raises(ValueError):
        Ray([0, 0, 0], [0, 0, 0])


def test_invalid_shapes_raise():
    with pytest.raises(ValueError):
        Ray([0, 0], [1, 0, 0])
    with pytest.raises(ValueError):
        Ray([0, 0, 0], [1, 0])


def test_lidar_model_generate_rays_identity_pose():
    # simple pattern: two beams along x and y axes
    from lidar_sim.lidar.lidar_model import LiDARModel

    angles = [(0.0, 0.0), (np.pi / 2, 0.0)]
    model = LiDARModel(angles, noise_params=None)
    # identity pose => origin at (0,0,0) and world dirs equal to local dirs
    pose = np.eye(4)
    rays = model.generate_rays(pose)
    assert len(rays) == 2
    assert np.allclose(rays[0].direction, [1.0, 0.0, 0.0])
    assert np.allclose(rays[1].direction, [0.0, 1.0, 0.0])
    assert np.allclose(rays[0].origin, [0.0, 0.0, 0.0])


def test_lidar_model_generate_rays_translated_pose():
    from lidar_sim.lidar.lidar_model import LiDARModel

    angles = [(0.0, 0.0)]
    model = LiDARModel(angles, noise_params=None)
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    rays = model.generate_rays(pose)
    assert len(rays) == 1
    assert np.allclose(rays[0].origin, [1.0, 2.0, 3.0])
