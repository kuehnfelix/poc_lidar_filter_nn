
import numpy as np
from lidar_sim.scene.scene import Scene
from lidar_sim.geometry.ground import GroundPlane
from lidar_sim.geometry.box import BoxObject

from lidar_sim.scene.track import Track, TrackGenerationError, Color
from lidar_sim.geometry.cone import ConeObject
from lidar_sim.geometry.cylinder import CylinderObject


class SceneGenerator:
    def __init__(self):
        self._track = None

    @property
    def track(self):
        if self._track is None:
            raise TrackGenerationError("Track has not been generated yet.")
        return self._track

    def generate_scene(self) -> Scene:
        scene = Scene()
        self._add_ground(scene)
        self._add_track_cones(scene)
        self._add_obstacles(scene)
        return scene


    def _add_ground(self, scene: Scene):
        scene.add_object(GroundPlane())
        
    def _add_track_cones(self, scene: Scene):
        track = Track()
        track.generate_random_track()
        self._track: Track = track
        
        for c in track.cones:
            
            height, radius = (.505, 0.1425) if c.color == Color.ORANGE else (.325, 0.114)
            
            distance = np.linalg.norm([c.x, c.y])
            if distance > 100:
                continue
            cone = ConeObject(-1, np.array([c.x, c.y, 0]), height, radius)
            scene.add_object(cone)

    
    def _add_obstacles(self, scene: Scene):
        i = 0
        track_bounds = self._track.track_bounds

        
        while i < 50:
            x = np.random.uniform(track_bounds[0], track_bounds[1])
            y = np.random.uniform(track_bounds[2], track_bounds[3])
            
            centerline_dist = self._track.distance_to_centerline(x, y)

            
            if i < 25:
                # Add a box obstacle
                size = np.random.uniform(0.5, 6, size=3)
                
                diagonal = np.linalg.norm(size[0:2])
                if centerline_dist < diagonal*0.5 + 0.5:
                    continue  # Skip if too close to the track centerline
                
                center=np.array([x, y, size[2]/2])
                z_angle = np.random.uniform(0, 360)
                cos_angle = np.cos(np.radians(z_angle))
                sin_angle = np.sin(np.radians(z_angle))
                orientation = np.array([[cos_angle, -sin_angle, 0],
                                        [sin_angle,  cos_angle, 0],
                                        [0,          0,         1]])
                box = BoxObject(i, center, size, orientation)
                scene.add_object(box)
            else:
                # Add a cylinder obstacle
                height = np.random.uniform(0.1, 6)
                radius = np.random.uniform(0.05, 3)
                
                if centerline_dist < radius + 0.5:
                    continue  # Skip if too close to the track centerline
                
                center = np.array([x, y, 0])
                cylinder = CylinderObject(i, center, height, radius)
                scene.add_object(cylinder)
            i += 1
            
    def sample_lidar_pose(self,  
                             progress_along_track: float = 0.0, 
                             lateral_offset: float = 0.0, 
                             rotation_offset: float = 0.0, 
                             height: float = 1.0, 
                             lidar_rotation_deg:float = 10):
        """Sample a LiDAR pose along the track."""
        if self._track is None:
            raise TrackGenerationError("Track must be generated before sampling vehicle poses.")
        
        track_length = self._track.total_spline_length()
        arc_length = progress_along_track * track_length
        
        x, y, car_angle = self._track.pose_at_arc_length(arc_length)
        car_angle += rotation_offset
        x += lateral_offset * np.cos(car_angle + np.pi/2)
        y += lateral_offset * np.sin(car_angle + np.pi/2)
        
        z = height
        
        lidar_pose = np.eye(4)
        lidar_pose[0, 3] = x
        lidar_pose[1, 3] = y    
        lidar_pose[2, 3] = z
        
        # Rotate lidar down around y-axis
        lidar_angle = np.radians(lidar_rotation_deg)
        cos_l = np.cos(lidar_angle)
        sin_l = np.sin(lidar_angle)
        lidar_rotation = np.array([[cos_l, 0, sin_l],
                                [0,     1, 0],
                                [-sin_l,0, cos_l]])
        lidar_pose[0:3, 0:3] = lidar_rotation
        
        # rotate to align with track tangent
        cos_a = np.cos(car_angle)
        sin_a = np.sin(car_angle)
        rotation_matrix = np.array([[cos_a, -sin_a, 0],
                                    [sin_a,  cos_a, 0],
                                    [0,      0,     1]])
        lidar_pose[0:3, 0:3] = rotation_matrix @ lidar_pose[0:3, 0:3]
        
        return lidar_pose