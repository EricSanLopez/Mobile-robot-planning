import coal
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as mg
from meshcat.transformations import translation_matrix, rotation_matrix
import time

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

# ────────────────────────────────────────────────────────────────
#  1. show_robot  — now works for 3- or 4-state vectors
# ────────────────────────────────────────────────────────────────
def show_robot_(robot: pin.RobotWrapper,
                viz: MeshcatVisualizer,
                x: float,
                y: float,
                theta: float,
                phi: float = 0.0):
    """
    Display either a differential-drive (x,y,θ) or a bicycle (x,y,θ,φ) pose.

    • (x, y, θ)      – position & chassis heading  
    • φ (optional)   – steering angle; ignored if the model has no steer_joint
    """
    # build a fresh configuration vector --------------------------
    q = pin.neutral(robot.model).copy()          # length = model.nq
    # base position & orientation (floating joint = 7 DoF)
    q[0:3] = [x, y, 0.0]
    q[3:7] = pin.Quaternion(pin.utils.rotate('z', theta)).coeffs()

    # steering joint, if present ---------------------------------
    try:
        idx = robot.model.getJointId("steer_joint") - 1   # −1 → q-index
        q[idx] = phi
    except KeyError:
        pass                                              # diff-drive model

    viz.display(q)


# ────────────────────────────────────────────────────────────────
# 2. simple trajectory player that accepts 3- or 4-state arrays
# ────────────────────────────────────────────────────────────────
def sim_trajectory(viz, robot: pin.RobotWrapper, delta_t: float, X: np.ndarray):
    """
    Visualise a state trajectory X of shape (N,3) or (N,4).
    """
    is_bicycle = X.shape[1] == 4
    for x in X:
        if is_bicycle:
            px, py, th, phi = x
            show_robot(robot, viz, px, py, th, phi)
        else:
            px, py, th = x
            show_robot(robot, viz, px, py, th)
        
        time.sleep(delta_t)


# Define robot dynamics
def discrete_dynamics(state, control_input, dt, model_mismatch=False):
    """
    Update the robot's state based on its dynamics.

    Parameters:
    - state: Current state [x, y, theta]
    - control_input: Control input [linear_velocity, angular_velocity]
    - dt: Time step

    Returns:
    - Updated state [x, y, theta]
    """
    x, y, theta = state
    linear_velocity, angular_velocity = control_input
    if model_mismatch:
        # Introduce model mismatch by adding noise to the control input
        linear_velocity += np.random.normal(0, 1.0)
        angular_velocity += np.random.normal(0, 1.0)

    # Update state using differential drive kinematics
    x += linear_velocity * np.cos(theta) * dt
    y += linear_velocity * np.sin(theta) * dt
    theta += angular_velocity * dt

    # Normalize theta to keep it within [-pi, pi]
    theta = np.arctan2(np.sin(theta), np.cos(theta))

    return np.array([x, y, theta])
