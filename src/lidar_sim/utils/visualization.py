"""
LiDAR visualization using Vispy for fast 3D rendering.
Supports visualization of hits and scene objects.
"""
import numpy as np
from typing import List, Dict
from vispy import app, scene
app.use_app("pyside6")  #
from vispy.scene import visuals
from vispy.color import Color

from lidar_sim.core.hit import Hit
from lidar_sim.core.types import ObjectType
from lidar_sim.geometry.scene_object import SceneObject
from lidar_sim.geometry.box import BoxObject
from lidar_sim.geometry.cone import ConeObject
from lidar_sim.geometry.cylinder import CylinderObject
from lidar_sim.geometry.ground import GroundPlane


class LidarVisualizer:
    """
    Fast 3D visualizer for LiDAR hits and scene geometry using Vispy.
    
    Parameters
    ----------
    show_hits : bool
        Display hit points
    show_scene : bool
        Display scene objects
    point_size : float
        Size of hit points in pixels
    hit_color : str or tuple
        Color of hit points (vispy color specification)
    title : str
        Window title
    """
    
    def __init__(
        self,
        show_hits: bool = True,
        show_scene: bool = True,
        point_size: float = 3.0,
        hit_color: str = "white",
        title: str = "LiDAR Visualization"
    ):
        self.show_hits = show_hits
        self.show_scene = show_scene
        self.point_size = point_size
        self.hit_color = hit_color
        self.title = title
        
        # Create vispy canvas
        self.canvas = scene.SceneCanvas(title=title, keys='interactive', show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera()
        
        # Storage for visuals
        self.points_visual = None
        self.scene_visuals: Dict[int, visuals.Mesh] = {}
        self.hits: List[Hit] = []
        self.scene_objects: List[SceneObject] = []
        
        # Setup grid and axis
        self._setup_grid()
        
    def _setup_grid(self):
        """Setup grid and axis visuals."""
        # Add axis
        visuals.XYZAxis(parent=self.view.scene)
        
        # Add grid
        visuals.GridLines(parent=self.view.scene)
        
    def set_hits(self, hits: List[Hit]):
        """
        Update visualization with hit points.
        
        Parameters
        ----------
        hits : List[Hit]
            List of Hit objects to visualize
        """
        self.hits = hits
        
        if not self.show_hits:
            return
        
        # Extract valid hit positions
        positions = np.array([
            hit.position for hit in hits 
            if hit.hit and hit.position is not None
        ], dtype=np.float32)
        
        if positions.size == 0:
            # Remove existing points visual
            if self.points_visual is not None:
                self.points_visual.parent = None
                self.points_visual = None
            return
        
        # Remove old points visual if exists
        if self.points_visual is not None:
            self.points_visual.parent = None
        
        # Create new points visual
        # "color" arg isn't supported by newer vispy versions; use face_color
        self.points_visual = visuals.Markers(
            pos=positions,
            size=self.point_size,
            face_color=self.hit_color,
            edge_color=None,
            parent=self.view.scene
        )
        
    def set_scene(self, scene_objects: List[SceneObject]):
        """
        Update visualization with scene objects.
        
        Parameters
        ----------
        scene_objects : List[SceneObject]
            List of scene objects to visualize
        """
        self.scene_objects = scene_objects
        
        if not self.show_scene:
            return
        
        # Clear existing scene visuals
        for visual in self.scene_visuals.values():
            visual.parent = None
        self.scene_visuals.clear()
        
        # Add new scene objects
        for obj in scene_objects:
            self._add_scene_object(obj)
    
    def _add_scene_object(self, obj: SceneObject):
        """Add a single scene object to visualization."""
        obj_id = obj.object_id
        obj_type = obj.object_type
        
        if obj_type == ObjectType.GROUND:
            self._add_ground(obj, obj_id)
        elif obj_type == ObjectType.BOX:
            self._add_box(obj, obj_id)
        elif obj_type == ObjectType.CONE:
            self._add_cone(obj, obj_id)
        elif obj_type == ObjectType.CYLINDER:
            self._add_cylinder(obj, obj_id)
    
    def _add_ground(self, ground: GroundPlane, obj_id: int):
        """Visualize ground plane as a large rectangle."""
        z = ground.z
        size = 100
        
        # Create a flat mesh at the ground plane
        positions = np.array([
            [-size, -size, z],
            [size, -size, z],
            [size, size, z],
            [-size, size, z],
        ], dtype=np.float32)
        
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=np.uint32)
        
        mesh = visuals.Mesh(
            vertices=positions,
            faces=faces,
            color=Color("gray", alpha=0.3),
            parent=self.view.scene
        )
        self.scene_visuals[obj_id] = mesh
        
    def _add_box(self, box: BoxObject, obj_id: int):
        """Visualize box as a wireframe or mesh."""
        center = box.center
        size = box.size
        orientation = box.orientation
        
        # Create box vertices centered at origin
        dx, dy, dz = size[0] / 2, size[1] / 2, size[2] / 2
        vertices = np.array([
            [-dx, -dy, -dz],
            [dx, -dy, -dz],
            [dx, dy, -dz],
            [-dx, dy, -dz],
            [-dx, -dy, dz],
            [dx, -dy, dz],
            [dx, dy, dz],
            [-dx, dy, dz],
        ], dtype=np.float32)
        
        # Apply rotation and translation
        vertices = (orientation @ vertices.T).T + center
        
        # Define box faces
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
            [0, 4, 5], [0, 5, 1],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2],  # right
        ], dtype=np.uint32)
        
        mesh = visuals.Mesh(
            vertices=vertices,
            faces=faces,
            color=Color("cyan", alpha=0.5),
            parent=self.view.scene
        )
        self.scene_visuals[obj_id] = mesh
        
    def _add_cone(self, cone: ConeObject, obj_id: int):
        """Visualize cone (aligned with z-axis)."""
        # Cone properties
        position = cone.position
        height = cone.height
        radius = cone.radius
        
        if height < 1e-6:
            return
        
        # Create cone vertices
        n_segments = 16
        angles = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
        
        # Base circle vertices (at z=0)
        base_x = radius * np.cos(angles)
        base_y = radius * np.sin(angles)
        base_z = np.zeros(n_segments)
        
        base_vertices = np.column_stack([base_x, base_y, base_z])
        
        # Tip vertex (at z=height)
        tip = np.array([[0, 0, height]])
        
        # Translate to position
        base_vertices = base_vertices + position
        tip_pos = tip[0] + position
        
        vertices = np.vstack([base_vertices, tip_pos])
        
        # Create faces
        tip_idx = n_segments
        faces = []
        for i in range(n_segments):
            next_i = (i + 1) % n_segments
            faces.append([i, next_i, tip_idx])
        # Add base
        for i in range(n_segments - 2):
            faces.append([0, i + 1, i + 2])
        
        faces = np.array(faces, dtype=np.uint32)
        
        mesh = visuals.Mesh(
            vertices=vertices,
            faces=faces,
            color=Color("yellow", alpha=0.5),
            parent=self.view.scene
        )
        self.scene_visuals[obj_id] = mesh
        
    def _add_cylinder(self, cylinder: CylinderObject, obj_id: int):
        """Visualize cylinder (aligned with z-axis)."""
        # Cylinder properties
        base_center = cylinder.base
        radius = cylinder.radius
        height = cylinder.height
        
        if height < 1e-6:
            return
        
        # Create cylinder vertices
        n_segments = 16
        angles = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
        
        # Bottom circle (at z=0)
        bottom_x = radius * np.cos(angles)
        bottom_y = radius * np.sin(angles)
        bottom_z = np.zeros(n_segments)
        
        bottom = np.column_stack([bottom_x, bottom_y, bottom_z])
        
        # Top circle (at z=height)
        top_x = radius * np.cos(angles)
        top_y = radius * np.sin(angles)
        top_z = np.full(n_segments, height)
        
        top = np.column_stack([top_x, top_y, top_z])
        
        # Translate to base center
        bottom = bottom + base_center
        top = top + base_center
        
        vertices = np.vstack([bottom, top])
        
        # Create faces
        faces = []
        # Side faces
        for i in range(n_segments):
            next_i = (i + 1) % n_segments
            faces.append([i, next_i, n_segments + next_i])
            faces.append([i, n_segments + next_i, n_segments + i])
        # Bottom cap
        for i in range(n_segments - 2):
            faces.append([0, i + 1, i + 2])
        # Top cap
        for i in range(n_segments - 2):
            faces.append([n_segments, n_segments + i + 2, n_segments + i + 1])
        
        faces = np.array(faces, dtype=np.uint32)
        
        mesh = visuals.Mesh(
            vertices=vertices,
            faces=faces,
            color=Color("magenta", alpha=0.5),
            parent=self.view.scene
        )
        self.scene_visuals[obj_id] = mesh
    
    def toggle_hits(self):
        """Toggle hits visualization."""
        self.show_hits = not self.show_hits
        if self.show_hits:
            self.set_hits(self.hits)
        elif self.points_visual is not None:
            self.points_visual.parent = None
            self.points_visual = None
    
    def toggle_scene(self):
        """Toggle scene visualization."""
        self.show_scene = not self.show_scene
        if self.show_scene:
            self.set_scene(self.scene_objects)
        else:
            for visual in self.scene_visuals.values():
                visual.parent = None
            self.scene_visuals.clear()
    
    def show(self, block: bool = True):
        """Display the visualization window.

        Parameters
        ----------
        block : bool
            Whether to block execution by calling ``app.run()``. Set to
            ``False`` if you only want to create the window and will call
            the run loop yourself or are running headless.
        """
        self.canvas.show()
        if block:
            try:
                app.run()
            except Exception as exc:  # pragma: no cover - behavior depends on env
                # Likely running in headless environment with no X display.
                print(
                    "Warning: could not start vispy application loop -",
                    exc,
                    "(headless display?)"
                )
    
    def update(self, hits: List[Hit] = None, scene_objects: List[SceneObject] = None):
        """
        Update visualization with new hits and/or scene objects.
        
        Parameters
        ----------
        hits : List[Hit], optional
            New hits to visualize
        scene_objects : List[SceneObject], optional
            New scene objects to visualize
        """
        if hits is not None:
            self.set_hits(hits)
        if scene_objects is not None:
            self.set_scene(scene_objects)


def visualize_hits(
    hits: List[Hit],
    scene_objects: List[SceneObject] = None,
    show_hits: bool = True,
    show_scene: bool = True,
    point_size: float = 3.0,
    title: str = "LiDAR Visualization"
):
    """
    Convenience function to quickly visualize hits and scene objects.
    
    Parameters
    ----------
    hits : List[Hit]
        List of hits to visualize
    scene_objects : List[SceneObject], optional
        List of scene objects to visualize
    show_hits : bool
        Display hits
    show_scene : bool
        Display scene objects
    point_size : float
        Size of hit points
    title : str
        Window title
    
    Examples
    --------
    >>> hits = [Hit(True, 10.0, np.array([1, 2, 3]))]
    >>> visualize_hits(hits, scene_objects=scene.objects)
    """
    viz = LidarVisualizer(
        show_hits=show_hits,
        show_scene=show_scene,
        point_size=point_size,
        title=title
    )
    viz.set_hits(hits)
    if scene_objects is not None:
        viz.set_scene(scene_objects)
    try:
        viz.show()
    except Exception as exc:  # pragma: no cover
        print("Visualization failed to start (maybe headless):", exc)
        print("You can still retrieve `viz.canvas` or run `app.run()` manually.")
