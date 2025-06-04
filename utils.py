import coal
import numpy as np
import pinocchio as pin
import meshcat.geometry as mg
from meshcat.transformations import translation_matrix, rotation_matrix


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def compute_robot_radius(geom_model):
    """Compute the maximum distance from the robot base to any part of its body."""
    max_dist = 0.0
    for go in geom_model.geometryObjects:
        placement = go.placement  # SE3 from base to geometry

        # Position of geometry center
        center = placement.translation

        # Estimate shape size (very conservative for any geometry)
        if hasattr(go.geometry, 'halfSide'):  # box
            size = np.array(go.geometry.halfSide) * 2  # full size
            shape_radius = np.linalg.norm(size) / 2
        elif hasattr(go.geometry, 'radius') and hasattr(go.geometry, 'length'):  # cylinder
            shape_radius = go.geometry.radius + go.geometry.length / 2
        elif hasattr(go.geometry, 'radius'):  # sphere or capsule
            shape_radius = go.geometry.radius
        else:
            shape_radius = 0.0  # fallback

        total_dist = np.linalg.norm(center) + shape_radius
        max_dist = max(max_dist, total_dist)

    return max_dist


def get_q_from_pos(pos):
    x, y, theta = pos
    q = np.zeros(7)
    q[0:3] = [x, y, 0.0]  # translation in x, y, no z offset
    rot = pin.utils.rotate('z', theta)
    quat = pin.Quaternion(rot).coeffs()
    q[3:] = quat  # Fill quaternion (x, y, z, w)
    return q

def point_to_cylinders(p, centers, cyl_radius=0.2):
    return np.min(np.linalg.norm(centers - p, axis=1) - cyl_radius)
    

def closest_distance(conn_obstacles, cylinder_centers, x, y, cylinder_radius=0.2):
    """Compute closest distance from coordenates x, y to any obstacle"""
    radius = 0.01
    sphere = coal.Sphere(radius)
    placement = pin.SE3(np.eye(3), np.array([x, y, 0.0]))
    sphere_obj = coal.CollisionObject(sphere, placement)
    
    request = coal.DistanceRequest()
    result = coal.DistanceResult()
    
    min_distance = float('inf')
    
    for obstacle in conn_obstacles:
        coal.distance(sphere_obj, obstacle, request, result)
        if result.min_distance < min_distance:
            min_distance = result.min_distance
    min_distance += radius
            
    d_cyl = point_to_cylinders((x,y), cylinder_centers)

    return min(min_distance, d_cyl)


def point_segment_distance(p, a, b):
    """
    Compute the minimum distance between point p and line segment ab in 2D.
    """
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t * ab
    return np.linalg.norm(p - closest)


def is_inside_wall(position, wall_width, robot_radius):
    return np.abs(position) < (wall_width / 2 + robot_radius)
    

def is_inside_cylinder(position, center_cylinder, radius, robot_radius):
    return np.linalg.norm(center_cylinder - position) < (radius + robot_radius)
    

def is_outside_limits(position, xlim, ylim, robot_radius):
    outside_x = (position[0] < xlim[0] + robot_radius) or (position[0] > xlim[1] - robot_radius)
    outside_y = (position[1] < ylim[0] + robot_radius) or (position[1] > ylim[1] - robot_radius)
    return outside_x or outside_y


def show_sample_point(viewer, name, pos, color=[0, 0.6, 1.0, 1.0], radius=0.03):
    mat = mg.MeshLambertMaterial(color=color[:3], transparency=1.0 - color[3])
    viewer[name].set_object(mg.Sphere(radius), mat)
    T = translation_matrix([pos[0], pos[1], 0.01])  # z slightly above floor
    viewer[name].set_transform(T)


def erase_sample_points(viewer):
    for i in range(10000):
        try:
            viewer[f"sample_{i}"].delete()
        except:
            pass


def show_robot(x, y, theta, viz):
    quat = pin.Quaternion(pin.utils.rotate('z', theta)).coeffs()
    pos = np.array([x,y,0.1])

    viz.display(np.append(pos,quat))