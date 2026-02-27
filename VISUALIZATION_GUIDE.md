# LiDAR Visualization (Vispy)

Minimal 3‑D viewer for LiDAR hits and scene objects.

```python
from lidar_sim.core.hit import Hit
from lidar_sim.geometry.ground import GroundPlane
from lidar_sim.utils.visualization import LidarVisualizer, visualize_hits
import numpy as np

hits = [Hit(True, 10, np.array([1,2,3]))]
scene = [GroundPlane(z=0.0)]

# quick one-liner
visualize_hits(hits, scene_objects=scene)

# or full API
viz = LidarVisualizer(point_size=5, hit_color="red")
viz.set_hits(hits)
viz.set_scene(scene)
viz.show()  # use block=False on headless machines
```

The visualizer handles:
- point clouds, axis-aligned boxes, cones, cylinders, ground plane
- toggling hits/scene (`toggle_hits()`, `toggle_scene()`)
- dynamic updates (`update()`, `set_hits()`, `set_scene()`)

**Tip:** on a system without a GUI the `show()` call emits a warning and
returns; you can still grab an image via `viz.canvas.render()` or drive
`app.run()` yourself.

See `tests/playground.py` for a longer example.