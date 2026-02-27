import numpy as np
from scipy.spatial import Voronoi

def bounded_voronoi(input_points, bounding_box):
    """
    Creates a Voronoi diagram bounded by the bounding box.
    
    Args:
        input_points (numpy.ndarray): Coordinates of input points for Voronoi diagram.
        bounding_box (numpy.ndarray): Specifies the boundaries of the Voronoi diagram, [x_min, x_max, y_min, y_max].
    
    Returns:
        scipy.spatial.qhull.Voronoi: Voronoi diagram object.
    """
    
    def _mirror(boundary, axis):
        mirrored = np.copy(points_center)
        mirrored[:, axis] = 2 * boundary - mirrored[:, axis]
        return mirrored
    
    x_min, x_max, y_min, y_max = bounding_box
    
    # Mirror points around each boundary
    points_center = input_points
    points_left = _mirror(x_min, axis=0) 
    points_right = _mirror(x_max, axis=0) 
    points_down = _mirror(y_min, axis=1)
    points_up = _mirror(y_max, axis=1)
    points = np.concatenate([points_center, points_left, points_right, points_down, points_up])
    
    # Compute Voronoi
    vor = Voronoi(points)
    
    # We only need the section of the Voronoi diagram that is inside the bounding box
    vor.filtered_points = points_center
    vor.filtered_regions = np.array(vor.regions, dtype=object)[vor.point_region[:vor.npoints//5]]
    return vor


def clockwise_sort(p):
    """
    Sorts nodes in clockwise order.
    
    Args:
        p (numpy.ndarray): Points to sort.
    
    Returns:
        numpy.ndarray: Clockwise sorted points.
    """
    d = p - np.mean(p, axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def calculate_curvature(dx_dt, d2x_dt2, dy_dt, d2y_dt2):
    """
    Calculates the curvature along a line.
    
    Args:
        dx_dt (numpy.ndarray): First derivative of x.
        d2x_dt2 (numpy.ndarray): Second derivative of x.
        dy_dt (numpy.ndarray): First derivative of y.
        d2y_dt2 (numpy.ndarray): Second derivative of y.
    
    Returns:
        np.ndarray: Curvature along line.
    """
    return (dx_dt**2 + dy_dt**2)**-1.5 * (dx_dt * d2y_dt2 - dy_dt * d2x_dt2)

def arc_lengths(x, y, R):
    """
    Calculates the arc length between two points based on the radius of curvature of the path segment.
    
    Args:
        x (numpy.ndarray): X-coordinates.
        y (numpy.ndarray): Y-coordinates.
        R (numpy.ndarray): Radius of curvature of track segment in meters.
    Returns:
        (float): Arc length in meters.
    """
    x0, x1 = x[:-1], x[1:]
    y0, y1 = y[:-1], y[1:]   
    R = R[:-1]
    
    distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    theta = 2 * np.arcsin(0.5 * distance / R)
    arc_length = R * theta
    return arc_length

def transformation_matrix(displacement, angle):
    """
    Translate, then rotate around origin.
    
    Args:
        displacement (tuple): Distance to translate along both axes.
        angle (float): Angle in radians to rotate.
    
    Returns:
        numpy.ndarray: 3x3 transformation matrix.
    """
    h, k = displacement
    c, s = np.cos(angle), np.sin(angle)
    
    M = np.array([
        [c,    -s,      h * c - k * s],
        [s,     c,      h * s + k * c],
        [0,     0,            1      ]
    ])
    return M

def closest_node(node, nodes, k):
    """
    Returns the index of the k-th closest node.
    
    Args:
        node (numpy.ndarray): Node to find k-th closest node to.
        nodes (numpy.ndarray): Available nodes.
        k (int): Number which determines which closest node to return.
    
    Returns:
        int: Index of k-th closest node.
    """
    deltas = nodes - node
    distance = np.einsum('ij,ij->i', deltas, deltas)
    return np.argpartition(distance, k)[k]
