import numpy as np
import pytest

from lidar_sim.core.hit import Hit
from lidar_sim.utils.visualization import LidarVisualizer, visualize_hits


def test_visualizer_basic_usage():
    """Basic operations on the visualizer should not raise errors."""
    viz = LidarVisualizer(show_hits=True, show_scene=True, point_size=2.0)
    viz.set_hits([])
    viz.set_scene([])
    viz.toggle_hits()
    viz.toggle_scene()
    viz.show(block=False)

    hits = [Hit(True, 1.0, np.array([0.0, 0.0, 0.0]))]
    viz.set_hits(hits)
    viz.set_scene([])
    assert viz.hits == hits


def test_visualize_hits_forwarding(monkeypatch):
    """`visualize_hits` should construct and use LidarVisualizer correctly."""
    called = {}

    class DummyViz:
        def __init__(self, *args, **kwargs):
            called['init'] = kwargs
        def set_hits(self, hits):
            called['hits'] = hits
        def set_scene(self, scene):
            called['scene'] = scene
        def show(self):
            called['shown'] = True

    monkeypatch.setattr('lidar_sim.utils.visualization.LidarVisualizer', DummyViz)
    hits = [Hit(True, 1.0, np.array([1.0, 2.0, 3.0]))]
    scene = ['foo']
    visualize_hits(hits, scene_objects=scene, show_hits=False, point_size=5)

    assert called['hits'] is hits
    assert called['scene'] is scene
    assert called.get('shown', False)
    assert called['init']['show_hits'] is False
    assert called['init']['point_size'] == 5
