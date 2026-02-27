import csv
import math
import random
import numpy as np
from scipy import interpolate, signal
from scipy.interpolate import BSpline
from enum import Enum

from shapely import Polygon, Point

from lidar_sim.scene.track_utils import *

import matplotlib.pyplot as plt


class TrackGenerationError(Exception):
    """Custom exception for track generation errors."""


class Color(Enum):
    """Preset cone colors"""
    YELLOW = 'y'
    BLUE = 'b'
    ORANGE = 'darkorange'


class Cone:
    """Class representing a cone on the track in 2D."""
    def __init__(self, color: Color, x: float, y: float) -> None:
        self.color = color
        self.x = x
        self.y = y

class Track:
    """Class representing a track, which consists of a centerline, cones and triangulation."""
    def __init__(self, spline: BSpline = None, 
                 track_width: float = 3., 
                 min_bound: float = 0., 
                 max_bound: float = 150., 
                 n_points: int = 1000,
                 n_regions: int = 30):
        self.generation_error = False
                 
        self._spline = spline
        self._centerline_points: Polygon = None
        self._cones = []
        
        orange_cone1 = Cone(Color.ORANGE, 5.85, 1.6)
        orange_cone2 = Cone(Color.ORANGE, 5.85, -1.6)
        orange_cone3 = Cone(Color.ORANGE, 6.15, 1.6)
        orange_cone4 = Cone(Color.ORANGE, 6.15, -1.6)
        self._orange_cones = [orange_cone1, orange_cone2, orange_cone3, orange_cone4]

        self._subdivisions = 500
        self._track_width = track_width
        self._straight_threshold = 1./100.
        self._length_start_area = 5.
        self._min_bound = min_bound
        self._max_bound = max_bound   
        self._n_points = n_points  
        self._n_regions = n_regions

        # if spline not None calculate rest
        if self._spline:
            self.calculate_track()
            
        
    
    @property
    def spline(self):
        return self._spline

    @spline.setter
    def spline(self, value):
        self._spline = value
        self._subdivisions = int(self.total_spline_length()*4)
        self.calculate_track()

    @property
    def cones(self):
        return self._cones

    @property
    def centerline_points(self):
        return self._centerline_points
    
    def generate_random_track(self, num_tries = 100):
        """Generates a random track, with a maximum number of tries to find a valid track."""
        for _ in range(num_tries):
            self._generate_random_spline()
            try:
                self.calculate_track()
                return
            except TrackGenerationError as e:
                print(f"Track generation failed with error: {e}. Retrying...")
        raise TrackGenerationError("Failed to generate a valid track after multiple attempts.")
        
        

    def _generate_random_spline(self):
        """Generates a random track using Voronoi diagrams and B-spline interpolation."""
        while True:

            # Create bounded Voronoi diagram
            input_points: np.ndarray = np.random.uniform(self._min_bound, self._max_bound, (self._n_points, 2))
            vor = bounded_voronoi(input_points, np.array([self._min_bound, self._max_bound] * 2))
            
            while True:
                # Select regions randomly
                random_point_indices = np.random.randint(0, self._n_points, self._n_regions)
                
                # From the Voronoi regions, get the regions belonging to the randomly selected points
                regions = np.array([np.array(region) for region in vor.regions],dtype=object)
                random_region_indices = vor.point_region[random_point_indices]
                random_regions = np.concatenate(regions[random_region_indices])

                # Get the vertices belonging to the random regions
                random_vertices = np.unique(vor.vertices[random_regions], axis=0)

                # Sort vertices
                sorted_vertices = clockwise_sort(random_vertices)
                sorted_vertices = np.vstack([sorted_vertices, sorted_vertices[0]])

                while True:
                    # Interpolate
                    spacing = np.linspace(0, 1, 500)
                    (t,c,k), _ = interpolate.splprep([sorted_vertices[:,0], sorted_vertices[:,1]], s=0, per=True)
                    spline = BSpline(t, np.asarray(c).T, k)

                    derivative = spline.derivative()
                    derivative2 = derivative.derivative()
                    x, y = spline(spacing).T
                    dx_dt, dy_dt = derivative(spacing).T
                    d2x_dt2, d2y_dt2 = derivative2(spacing).T
                    
                
                    # Calculate curvature
                    k = calculate_curvature(dx_dt, d2x_dt2, dy_dt, d2y_dt2)
                    abs_curvature = np.abs(k)

                    # Check if curvature exceeds threshold
                    peaks, _ = signal.find_peaks(abs_curvature)
                    exceeded_peaks = abs_curvature[peaks] > 1./3.
                    max_peak_index = abs_curvature[peaks].argmax()
                    is_curvature_exceeded = exceeded_peaks[max_peak_index]

                    if is_curvature_exceeded:
                        # Find vertex where curvature is exceeded and delete vertice from sorted vertices. Reiterate
                        max_peak = peaks[max_peak_index]
                        peak_coordinate = (x[max_peak], y[max_peak])
                        vertex = closest_node(peak_coordinate, sorted_vertices, k=0)
                        sorted_vertices = np.delete(sorted_vertices, vertex, axis=0)
                        
                        # Make sure that first and last coordinate are the same for periodic interpolation
                        if not np.array_equal(sorted_vertices[0], sorted_vertices[-1]):
                            sorted_vertices = np.vstack([sorted_vertices, sorted_vertices[0]])                
                    else:
                        break

                # Create track boundaries
                track = Polygon(zip(x,y))
                track_left = track.buffer(3 / 2)
                track_right = track.buffer(-3 / 2)


                
                # Check if track does not cross itself
                if track.is_valid and track_left.is_valid and track_right.is_valid:
                    if track.geom_type == track_left.geom_type == track_right.geom_type == 'Polygon':
                        break
                print("track is invalid Polygon! Removing vertex with max curvature")
            try:
                self._spline = spline
            except TrackGenerationError:
                continue

            return self



    def calculate_track(self):
        """ 
        Calculates centerline_points, cones_left, cones_right, triangulation from self._spline 
        """
        assert self._spline, 'track needs to have a spline in order to calculate the track. Set the spline manually or use generate_random_spline()'
        t = np.linspace(0, 1, self._subdivisions)
        points = self._spline(t)
        poly_centerline = Polygon(points.tolist())
        poly_left = poly_centerline.buffer(self._track_width/2)
        poly_right = poly_centerline.buffer(-self._track_width/2)

        cone_spacing_left = np.linspace(0, poly_left.length, np.ceil(poly_left.length / self._track_width).astype(int) + 1)[:-1]
        cone_spacing_right= np.linspace(0, poly_right.length, np.ceil(poly_right.length / self._track_width).astype(int) + 1)[:-1]
        centerline_spacing= np.linspace(0, poly_centerline.length, np.ceil(poly_centerline.length / self._track_width).astype(int) + 1)[:-1]
        
        cones_left = np.asarray([np.asarray(poly_left.exterior.interpolate(sp).xy).flatten() for sp in cone_spacing_left])
        cones_right = np.asarray([np.asarray(poly_right.exterior.interpolate(sp).xy).flatten() for sp in cone_spacing_right])
        centerline_points = np.asarray([np.asarray(poly_centerline.exterior.interpolate(sp).xy).flatten() for sp in centerline_spacing])

        # Calculate curvature
        abs_curvature = np.abs(self.curvature())

        # Find straight section in track that is at least the length of the start area
        # If such a section cannot be found, adjust the straight_threshold and length_start_area variables
        # There is only a chance of this happening if n_regions == 1 
        straight_threshold = self._straight_threshold if abs_curvature.min() < self._straight_threshold else abs_curvature.min() + 0.1
        straight_sections = abs_curvature[:-1] <= straight_threshold
        distances = arc_lengths(*self._spline(t).T, 1 / abs_curvature)
        length_straights = distances * straight_sections

        # Find cumulative length of straight sections
        for i in range(1, len(length_straights)):
            if length_straights[i]:
                length_straights[i] += length_straights[i-1]

        # Find start line and start pose
        length_start_area = self._length_start_area if length_straights.max() > self._length_start_area else length_straights.max()
        try:
            start_line_index = np.where(length_straights > length_start_area)[0][0]
        except IndexError:
            self.generation_error = True
            raise TrackGenerationError("Unable to find suitable starting position. Try to decrease the length of the starting area or different input parameters.")
        start_position = np.asarray(poly_centerline.exterior.interpolate(np.sum(distances[:start_line_index]) - length_start_area)).flatten() 
        start_line = np.asarray(poly_centerline.exterior.interpolate(np.sum(distances[:start_line_index]))).flatten() 

        rel_x = start_line[0].x - start_position[0].x
        rel_y = start_line[0].y - start_position[0].y
        start_heading = -math.atan2(rel_y, rel_x)

        # Filter out cones that are on the starting line
        filter_array_left = [((c[0]-start_line[0].x)**2 + (c[1]-start_line[0].y)**2) > 1.8**2 for c in cones_left]
        cones_left = cones_left[filter_array_left]
        filter_array_right = [((c[0]-start_line[0].x)**2 + (c[1]-start_line[0].y)**2) > 1.8**2  for c in cones_right]
        cones_right = cones_right[filter_array_right]

        # Translate and rotate track to origin
        M = transformation_matrix((-start_position[0].x, -start_position[0].y), start_heading)

        self._cones = []

        def check_near_orange_cone(c):
            for oc in self._orange_cones:
                if np.linalg.norm(np.array([c[0]-oc.x, c[1]-oc.y])) < 1.0:
                    return True
            return False

        for c in (M.dot(np.c_[cones_left, np.ones(len(cones_left))].T)[:-1].T).tolist():
            if not check_near_orange_cone(c):
                self._cones.append(Cone(Color.BLUE, c[0], c[1]))
        for c in (M.dot(np.c_[cones_right, np.ones(len(cones_right))].T)[:-1].T).tolist():
            if not check_near_orange_cone(c):
                self._cones.append(Cone(Color.YELLOW, c[0], c[1]))

        # Triangulation
        outer_polygon = Polygon( (M.dot(np.c_[cones_left, np.ones(len(cones_left))].T)[:-1].T).tolist())
        inner_polygon = Polygon((M.dot(np.c_[cones_right, np.ones(len(cones_right))].T)[:-1].T).tolist())

        if inner_polygon.contains(outer_polygon):
            inner_polygon, outer_polygon = outer_polygon, inner_polygon

        def find_inner_or_outer_point(polygon, inner: bool = True):
            (minx, miny, maxx, maxy) = outer_polygon.bounds
            while True:
                pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
                if polygon.contains(pnt) == inner:
                    return([pnt.x, pnt.y])


        
        inner_len = len(cones_left)
        outer_len = len(cones_right)

        print(f'innerlen: {inner_len}  outerlen: {outer_len}  outerraw: {len(cones_left)}  len: {len(self._cones)}')
        outer_segments = [[i,(i+1)%inner_len] for i in range(inner_len)]
        inner_segments = [[i+inner_len, ((i+outer_len+1)%outer_len)+inner_len] for i in range(outer_len)]
        
        points = {
            'vertices': [[cone.x, cone.y] for cone in self._cones],
            'segments': inner_segments
                      + outer_segments,
            'holes': [find_inner_or_outer_point(inner_polygon, True), find_inner_or_outer_point(outer_polygon, False)]
        }
        
        for c in self._orange_cones:
             self._cones.append(c)

        self._centerline_points = Polygon((M.dot(np.c_[centerline_points, np.ones(len(centerline_points))].T)[:-1].T).tolist())

        return self


    def curvature(self):
        """
        calculates an array with the curvature of the spline
        """
        t = np.linspace(0, 1, self._subdivisions)
        derivative = self._spline.derivative()
        derivative2 = derivative.derivative()
        x, y = self._spline(t).T
        dx_dt, dy_dt = derivative(t).T        
        d2x_dt2, d2y_dt2 = derivative2(t).T
        return calculate_curvature(dx_dt, d2x_dt2, dy_dt, d2y_dt2)

    def total_spline_length(self):
        """
        Calculates the total length of the spline
        """
        assert self._spline, 'No track provided!'
        t = np.linspace(0, 1, 500)
        x, y = self._spline(t).T
        derivative = self._spline.derivative()
        derivative2 = derivative.derivative()
        dx_dt, dy_dt = derivative(t).T        
        d2x_dt2, d2y_dt2 = derivative2(t).T
        k = calculate_curvature(dx_dt, d2x_dt2, dy_dt, d2y_dt2)
        abs_curvature = np.abs(k)
        distances = arc_lengths(x, y, 1 / abs_curvature)
        return np.sum(distances)
    
    def plot(self):
        """
        Plots itself
        """
        plt.figure()

        for cone in self._cones:
            plt.plot(cone.x, cone.y, '.', color=cone.color.value)

        plt.plot(*self.centerline_points.exterior.xy, 'k')


        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')
        plt.grid()
        plt.show(block=True)

