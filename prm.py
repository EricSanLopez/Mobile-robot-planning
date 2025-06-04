import utils
import coal
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
from meshcat.transformations import translation_matrix, rotation_matrix
from scipy.spatial.distance import cdist
from localPlanner import LocalOptimalControlPlanner, DifferentialDriveModel


class PRM_Graph:
    def __init__(self, samples):
        self.nodes = samples
        self.edges = list()
        self._no_edges = list()
        
    def add_edge(self, p1, p2, collision=False):
        if not collision:
            self.edges.append([p1, p2])
        else:
            self._no_edges.append([p1, p2])        

    def exists_edge(self, p1, p2):
        exists_real_edge = ([p1, p2] in self.edges) or ([p2, p1] in self.edges)
        exists_no_edge = ([p1, p2] in self._no_edges) or ([p2, p1] in self._no_edges)
        return exists_real_edge or exists_no_edge

    def knn(self, p, k=10):
        distances = cdist(self.nodes, [self.nodes[p]]).flatten()
        return np.argsort(distances)[1:k+1]


    def local_path(self, p1, p2, n_intervals=30):
        p1 = self.nodes[p1]
        p2 = self.nodes[p2]
        
        # OCP also needs theta for each point
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        theta = np.arctan2(dy, dx)
        p1, p2 = np.array([*p1, theta]), np.array([*p2, theta])

        planner = LocalOptimalControlPlanner(
            model = DifferentialDriveModel(),
            t_final = 5.0,
            n_intervals = n_intervals,
            ipopt_options={"ipopt.print_level":0, "print_time":False}
        )
                
        planner.build_ocp(p1, p2)
    
        # straight‑line initial guess in configuration space 
        t_lin = np.linspace(0, 1, planner.n_intervals + 1) 
        x_guess = np.outer(1 - t_lin, p1) + np.outer(t_lin, p2) # (n_intervals, 3)
        planner.set_initial_guess(x_guess=x_guess.T)  # rockit expects column‑wise (3; n_intervals)
    
        planner.solve()
        return planner.sample() 

    def show_graph(self, vis):
        def draw_edge_cylinder(vis, name, p1, p2, radius=0.01, color=0x0000ff):
            """Draw a cylinder between two points at z=0.5"""
            p1 = np.array([*p1, 0.01])
            p2 = np.array([*p2, 0.01])
            # Midpoint
            mid = (p1 + p2) / 2
            # Vector between points
            v = p2 - p1
            length = np.linalg.norm(v)
            if length == 0:
                return  # Avoid degenerate case
        
            # Normalize
            v /= length
        
            # Compute rotation matrix from z-axis to v
            y_axis = np.array([0, 1, 0])  # cylinders are aligned along Y-axis in meshcat
            axis = np.cross(y_axis, v)
            angle = np.arccos(np.clip(np.dot(y_axis, v), -1.0, 1.0))
            if np.linalg.norm(axis) < 1e-8:
                R = np.eye(3) if np.dot(z_axis, v) > 0 else -np.eye(3)
            else:
                axis /= np.linalg.norm(axis)
                R = rotation_matrix(angle, axis)[:3, :3]
        
            # Build 4x4 transform
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = mid
        
            # Create and show cylinder
            vis[name].set_object(
                mg.Cylinder(length, radius),
                mg.MeshLambertMaterial(color=color)
            )
            vis[name].set_transform(T)
        
        # Draw nodes
        for i, pos in enumerate(self.nodes):
            vis.viewer[f"nodes/{i}"].set_object(
                mg.Sphere(0.05),
                mg.MeshLambertMaterial(color=0xff0000)
            )
            pos = np.array([*pos, 0.01])
            vis.viewer[f"nodes/{i}"].set_transform(translation_matrix(pos))
        
        # Draw edges as cylinders
        for i, (a, b) in enumerate(self.edges):
            draw_edge_cylinder(vis.viewer, f"edges/{i}", self.nodes[a], self.nodes[b])


def sample_near_wall_full(wall_obj, xlim, ylim, robot_radius, tower_radius=0.2, std=0.2):
    """Biased sampling around obstacles"""
    placement = wall_obj.placement
    center = placement.translation[:2]
    R = placement.rotation[:2, :2]
    x_axis = R[:, 0]  # Local X (wall direction)
    y_axis = R[:, 1]  # Local Y (width direction)

    wall_length, wall_width, _ = 2 * wall_obj.geometry.halfSide

    # Sample from the side with a prob proportional to the perimeter difference
    side_prob = wall_length / (wall_length + 2*np.pi*(tower_radius + robot_radius))

    if np.random.rand() < side_prob:
        # Side sampling (parallel to long edge)
        s = np.random.uniform(-wall_length/2, wall_length/2)

        # Outside the wall
        offset = 0
        pos = [0,0]
        
        while utils.is_inside_wall(offset, wall_width, robot_radius) or \
                utils.is_outside_limits(pos, xlim, ylim, robot_radius):
            offset = np.random.normal(0, std)
            pos = center + s * x_axis + offset * y_axis
    
    else:
        # End sampling (near cylinder caps)
        sign = np.random.choice([-1, 1])
        cap_center = center + sign * (wall_length/2) * x_axis

        # Outside the cylinder
        offset = [0,0]
        pos = cap_center
        while utils.is_inside_cylinder(pos, cap_center, tower_radius, robot_radius) or \
                utils.is_outside_limits(pos, xlim, ylim, robot_radius):
            offset = np.random.normal(0, std, size=2)
            
            # Ensure that the sampling is towards the "out" direction
            dot = np.dot(offset, x_axis)
            if dot*sign < 0:
                offset -= 2 * dot * x_axis
            pos = cap_center + offset
        
    return pos
    

def sample_pose(xlim, ylim):
    """Uniform sampling over all space"""
    x = np.random.uniform(*xlim)
    y = np.random.uniform(*ylim)
    return np.array([x, y])


def biased_sample_pose(xlim, ylim, geom_model, robot_radius, bias_prob=0.7, std=0.2):
    """Sampler wrapper"""
    wall_objects = [
        g for g in geom_model.geometryObjects if "connection" in g.name
    ]
    wall_lengths = np.array([wall.geometry.halfSide[0] * 2 for wall in wall_objects])
    wall_lengths /= np.sum(wall_lengths)
    
    # Around obstacles
    if np.random.rand() < bias_prob:
        # Weighted choice between obstacles to have proportional sampling
        wall = np.random.choice(wall_objects, p=wall_lengths)
        return sample_near_wall_full(wall, xlim, ylim, robot_radius, std=std)
    # Random
    return sample_pose(xlim, ylim)


def get_samples(N, xlim, ylim, geom_model, robot_radius, bias_prob=0.6, std=0.2):
    """Generate samples according to geom_model and hyperparams"""
    conn_obstacles = [
        coal.CollisionObject(g.geometry, g.placement) for g in geom_model.geometryObjects if "connection" in g.name
    ]
    cylinder_centers = np.array([
        g.placement.translation[:2] for g in geom_model.geometryObjects if "cylinder" in g.name
    ])
    
    samples = list()
    
    for _ in range(N):
        while True:
            sample = biased_sample_pose(xlim, ylim, geom_model, robot_radius, bias_prob=bias_prob, std=std)
            if utils.closest_distance(conn_obstacles, cylinder_centers, *sample) > robot_radius:
                break
        samples.append(sample)
    return samples
