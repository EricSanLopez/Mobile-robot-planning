import coal
import utils
import numpy as np
import pinocchio as pin
import meshcat.geometry as mg
from scipy.spatial.distance import cdist
from pinocchio.visualize import MeshcatVisualizer
from meshcat.transformations import translation_matrix, rotation_matrix


def init_obstacles(robot, geom_model, params, visualization=False, viz=None):
    """Initialize the collision objects for all obstacles and the collision pairs with all robot parts"""
    robot_parts = len(geom_model.geometryObjects)

    obstacle_positions = params.obstacle_positions
    if visualization:
        viz.viewer.delete()
        viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
        viz.initViewer(loadModel=True)
    
    # Walls
    wall_thickness = params.wall_thickness
    wall_width = params.wall_width
    wall_height = params.wall_height
    
    for i in range(4):
        # Position and orientation
        x_pos = (-1)**(i-1) if (i < 2) else 0
        y_pos = (-1)**(i-1) if (i > 1) else 0
        wall_position = np.array([
            x_pos * (5 + wall_thickness / 2), 
            y_pos * (5 + wall_thickness / 2), 
            wall_height / 2
        ])
    
        wall_rotation = np.eye(3) if (y_pos == 0) else pin.utils.rotate('z', np.pi / 2)
        wall_placement = pin.SE3(wall_rotation, wall_position)
        
        # Definition
        wall = coal.Box(wall_thickness, wall_width, wall_height)
        
        # Object creation
        wall_object = pin.GeometryObject(
            f"wall_{i}", 
            robot.model.getFrameId("universe"), 
            wall, 
            wall_placement
        )
        geom_model.addGeometryObject(wall_object)
    
        if visualization:
            T_world_obs = translation_matrix(wall_position)
            T_world_obs[:3, :3] = wall_rotation
            viz.viewer[f"/Wall/wall_{i}"].set_object(
                mg.Box([wall_thickness, wall_width, wall_height]),
                mg.MeshLambertMaterial(color=0xff0000)
            )
            viz.viewer[f"/Wall/wall_{i}"].set_transform(T_world_obs)
    
    # Towers
    tower_height = params.tower_height
    tower_radius = params.tower_radius
    
    for i in range(4):
        # Position
        x_pos = (-1)**i
        y_pos = (-1)**(i // 2)  # changes every 2
        tower_position = np.array([
            x_pos * (5 + wall_thickness / 2), 
            y_pos * (5 + wall_thickness / 2), 
            tower_height / 2
        ])
        tower_rotation = pin.utils.rotate('x', np.pi / 2)  # by default is defined in y axis
        tower_placement = pin.SE3(tower_rotation, tower_position)
    
        # Definition
        tower_geom = coal.Cylinder(tower_radius, tower_height)
    
        # Object creation
        tower_object = pin.GeometryObject(
            f"tower_{i}",
            robot.model.getFrameId("universe"),
            tower_geom,
            tower_placement
        )
        geom_model.addGeometryObject(tower_object)
    
        if visualization:
            T_world_obs = translation_matrix(tower_position)
            T_world_obs[:3, :3] = pin.utils.rotate('x', np.pi / 2)
            viz.viewer[f"/Wall/tower_{i}"].set_object(
                mg.Cylinder(tower_height, tower_radius), 
                mg.MeshLambertMaterial(color=0xff0000)
            )
            viz.viewer[f"/Wall/tower_{i}"].set_transform(T_world_obs)
    
    # Obstacle cylinders
    obstacle_height = params.obstacle_height
    obstacle_radius = params.obstacle_radius
    
    for i, pos in enumerate(obstacle_positions):
        # Position and orientation
        obstacle_rotation = pin.utils.rotate('x', np.pi / 2)  # by default is defined in y axis
        obstacle_placement = pin.SE3(obstacle_rotation, pos)
    
        # Definition
        obstacle_geom = coal.Cylinder(obstacle_radius, obstacle_height)
    
        # Object creation
        obstacle = pin.GeometryObject(
            f"cylinder_obstacle_{i}", 
            robot.model.getFrameId("universe"), 
            obstacle_geom, 
            obstacle_placement
        )
        geom_model.addGeometryObject(obstacle)
    
        if visualization:
            T_world_obs = translation_matrix(pos)
            T_world_obs[:3, :3] = pin.utils.rotate('x', np.pi / 2)
            viz.viewer[f"/Obstacle/cylinder_{i}"].set_object(
                mg.Cylinder(obstacle_height, obstacle_radius), 
                mg.MeshLambertMaterial(color=0xff0000)
            )
            viz.viewer[f"/Obstacle/cylinder_{i}"].set_transform(T_world_obs)
    
    # Obstacle walls (connected between obstacle pairs)
    conn_wall_width = params.conn_wall_width
    conn_wall_height = params.conn_wall_height
    
    for i, pos in enumerate(obstacle_positions[::2]):
        conn_wall_length = np.linalg.norm(obstacle_positions[2*i][:2] - obstacle_positions[2*i+1][:2])
    
        # Absolute position 
        conn_wall_position = (obstacle_positions[2*i][:2] + obstacle_positions[2*i+1][:2]) / 2
        conn_wall_translation = np.array([conn_wall_position[0], conn_wall_position[1], conn_wall_height / 2])
    
        # Orientation
        angle = np.arctan2(
            obstacle_positions[2*i+1][1] - obstacle_positions[2*i][1],
            obstacle_positions[2*i+1][0] - obstacle_positions[2*i][0]
        )
        conn_wall_rotation = pin.utils.rotate('z', angle)
    
        # Definition
        conn_wall_placement = pin.SE3(conn_wall_rotation, conn_wall_translation)
        conn_wall = coal.Box(conn_wall_length, conn_wall_width, conn_wall_height)
    
        # Object creation
        obstacle = pin.GeometryObject(
            f"obstacle_connection_{i}", 
            robot.model.getFrameId("universe"), 
            conn_wall, 
            conn_wall_placement
        )
        geom_model.addGeometryObject(obstacle)
    
        if visualization:
            viz.viewer[f"/Obstacle/connection_{i}"].set_object(
                mg.Box([conn_wall_length, conn_wall_width, conn_wall_height]), 
                mg.MeshLambertMaterial(color=0xff0000)
            )
            viz.viewer[f"/Obstacle/connection_{i}"].set_transform(conn_wall_placement.homogeneous)
    
    if visualization:
        # Show robot in a basic pose
        x,y,theta = 0,0,0
        utils.show_robot(x,y,theta, viz)
    
    # Add a collision pair: robot vs obstacle
    # We check for collision between each robot body and the environment
    for i, rob in enumerate(geom_model.geometryObjects[:robot_parts]):
        for j, obs in enumerate(geom_model.geometryObjects[robot_parts:]):
            geom_model.addCollisionPair(pin.CollisionPair(i, j+robot_parts))
    geom_data = pin.GeometryData(geom_model)

    return geom_model, geom_data


def is_in_collision(geom_model, geom_data):
    """Collision checker with pinnochio"""
    pin.computeCollisions(geom_model, geom_data)
    for i, cr in enumerate(geom_data.collisionResults):
        if cr.isCollision():
            return True
    return False


def is_collision_free(timesteps, model, data, geom_model, geom_data):
    """Checks trajectory's collisions"""
    for t, x, u in zip(*timesteps):
        q = utils.get_q_from_pos(x)
        pin.forwardKinematics(model, data, q)
        pin.updateGeometryPlacements(model, data, geom_model, geom_data)
        if is_in_collision(geom_model, geom_data):
            return False
    return True


def is_collision_free_approx(timesteps, geom_model, robot_radius, cylinder_radius):
    """
    Approximate 2D collision checking against cylindrical and box obstacles (without pinocchio)

    Parameters
    ----------
    timesteps : tuple of arrays
        Typically (times, positions, velocities) from a trajectory; positions are Nx3.
    geom_model : pinocchio GeometryModel
        Contains GeometryObjects with names 'cylinder' or 'connection'.
    robot_radius : float
        Radius of the robot used for approximation.
    cylinder_radius : float
        Radius of cylindrical obstacles.

    Returns
    -------
    bool
        True if all timesteps are collision-free, False otherwise.
    """
    # Extract cylinder obstacle centers
    cylinder_centers = np.array([
        g.placement.translation[:2]
        for g in geom_model.geometryObjects
        if 'cylinder' in g.name
    ])

    # Extract wall segments using halfSide from Box geometry
    walls = []  # list of (p1, p2, half_thickness)
    for g in geom_model.geometryObjects:
        if 'connection' in g.name:
            # Box half-sides (half-lengths) in local x, y, z
            half_side = np.array(g.geometry.halfSide)
            half_len = half_side[0]
            half_thick = half_side[1]

            center = g.placement.translation[:2]
            R = g.placement.rotation[:2, :2]
            # Endpoints along the box's local x-axis
            p1 = center + R.dot(np.array([ half_len, 0.0]))
            p2 = center + R.dot(np.array([-half_len, 0.0]))
            walls.append((p1, p2, half_thick))

    # Loop through trajectory points
    for _, x, _ in zip(*timesteps):
        p = x[:2]
        # Cylinder collision
        if cylinder_centers.size:
            dists = cdist(cylinder_centers, p.reshape(1,2)).flatten()
            if np.any(dists < (cylinder_radius + robot_radius)):
                return False
        # Wall collision
        for p1, p2, half_thick in walls:
            d = utils.point_segment_distance(p, p1, p2)
            if d < (half_thick + robot_radius):
                return False
    return True
