import numpy as np
import pytest

from lidar_sim.core.ray import Ray


def test_ray_normalizes_direction():
    # provide explicit azimuth/elevation along with a non-normalized direction
    r = Ray([0, 0, 0], azimuth=0.0, elevation=0.0, direction=[0, 0, 10])
    assert np.allclose(r.direction, [0, 0, 1])
    assert r.azimuth == 0.0
    assert r.elevation == 0.0


def test_ray_origin_and_direction_shapes():
    r = Ray([1, 2, 3], azimuth=1.0, elevation=0.5, direction=[4, 5, 6])
    assert r.origin.shape == (3,)
    assert r.direction.shape == (3,)


def test_point_at_computes_correct_location():
    r = Ray([1, 2, 3], azimuth=0.0, elevation=0.0, direction=[1, 0, 0])
    assert np.allclose(r.point_at(5.0), [6.0, 2.0, 3.0])


def test_invalid_direction_raises():
    with pytest.raises(ValueError):
        Ray([0, 0, 0], azimuth=0.0, elevation=0.0, direction=[0, 0, 0])


def test_invalid_shapes_raise():
    # invalid origin shape
    with pytest.raises(ValueError):
        Ray([0, 0], azimuth=0.0, elevation=0.0, direction=[1, 0, 0])
    # invalid direction shape
    with pytest.raises(ValueError):
        Ray([0, 0, 0], azimuth=0.0, elevation=0.0, direction=[1, 0])


def test_lidar_model_generate_rays_identity_pose():
    # pattern with two successive azimuths; override channels to simplify output
    from lidar_sim.lidar.lidar_model import LiDARModel

    angles = [(0.0, 0.0), (np.pi / 2, 0.0)]
    model = LiDARModel(angles)
    # only use a single channel so we generate one ray per pattern element
    model.channels = [0.0]

    pose = np.eye(4)
    r1 = model.generate_rays(pose)[0]
    r2 = model.generate_rays(pose)[0]

    assert np.allclose(r1.direction, [1.0, 0.0, 0.0])
    assert np.allclose(r2.direction, [0.0, 1.0, 0.0])
    assert np.allclose(r1.origin, [0.0, 0.0, 0.0])


def test_lidar_model_generate_rays_translated_pose():
    from lidar_sim.lidar.lidar_model import LiDARModel

    angles = [(0.0, 0.0)]
    model = LiDARModel(angles)
    model.channels = [0.0]

    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    ray = model.generate_rays(pose)[0]

    assert np.allclose(ray.origin, [1.0, 2.0, 3.0])
