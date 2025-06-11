import utils
import coal
import meshcat.geometry as mg
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from meshcat.transformations import translation_matrix, rotation_matrix
from scipy.spatial.distance import cdist
from localPlanner import LocalOptimalControlPlanner, DifferentialDriveModel
import networkx as nx
import math
import heapq


class PRM_Graph:
    def __init__(self, samples):
        self.nodes = samples
        self.edges = list()
        self._no_edges = list()

    def add_nodes(self, nodes):
        self.nodes.extend(nodes)
        idx = len(self.nodes)
        return list(range(idx - len(nodes), idx))
        
    def add_edge(self, p1, p2, collision_free=False):
        if collision_free:
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

    def local_path(self, p1, p2, n_intervals=10, t_final=5.0):
        p1 = self.nodes[p1]
        p2 = self.nodes[p2]
        
        # OCP also needs theta for each point
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        theta = np.arctan2(dy, dx)
        p1, p2 = np.array([*p1, theta]), np.array([*p2, theta])

        planner = LocalOptimalControlPlanner(
            model = DifferentialDriveModel(),
            t_final = t_final,
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

    def _edge_weight(self, u, v):
        # Compute Euclidean distance
        x1, y1 = self.nodes[u]
        x2, y2 = self.nodes[v]
        return math.hypot(x2 - x1, y2 - y1)

    def dijkstra(self, start, goal, return_dist=False):
        np_edges = np.array(self.edges)
        n = len(self.nodes)
        dist = [math.inf] * n
        prev = [None] * n
        dist[start] = 0

        # min-heap of (distance, node)
        heap = [(0, start)]

        while heap:
            current_dist, u = heapq.heappop(heap)
            # If we popped a stale distance, skip
            if current_dist > dist[u]:
                continue
            # Early exit if we reached the goal
            if u == goal:
                break
            # Relax edges
            edges = np_edges[np.where((np_edges == u).any(axis=1))[0]]
            for ed in edges:
                v = ed[0] if ed[0] != u else ed[1]
                weight = self._edge_weight(u, v)
                alt = current_dist + weight
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(heap, (alt, v))

        # Reconstruct path
        path = []
        u = goal
        if prev[u] is not None or u == start:
            while u is not None:
                path.append(int(u))
                u = prev[u]
            path.reverse()
            
        if return_dist:
            return path, [dist[u] for u in path]
        return path

    def plot_graph(self):
        nodes = {i:v for i, v in enumerate(self.nodes)}
        G = nx.Graph()
        G.add_edges_from(self.edges)
        
        # Use actual positions for layout
        nx.draw(G, pos=nodes, node_color='lightgreen', node_size=50)
        plt.title("PRM Graph")
        plt.show()

    def plot_path(self, path):
        nodes = {i:v for i, v in enumerate(self.nodes)}
        path_nodes = {i:v for i, v in zip(path, np.array(self.nodes)[path])}
        path_edges = list(zip(path, path[1:]))
        
        G = nx.Graph()
        G.add_edges_from(self.edges)
        
        # Use actual positions for layout
        nx.draw(G, pos=nodes, node_color='lightgreen', node_size=50)
        nx.draw_networkx_edges(G, pos=nodes, edgelist=path_edges, edge_color='red', width=2.5)
        nx.draw_networkx_nodes(G, pos=nodes, nodelist=path, node_color='red', node_size=80)
        plt.title("Path over PRM Graph")
        plt.show()

    def show_graph(self, vis):
        utils.erase_graph_vis(vis)
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
    tries = 2000

    c = np.random.rand()

    if c < side_prob:
        # Side sampling (parallel to long edge)
        s = np.random.uniform(-wall_length/2, wall_length/2)

        # Outside the wall
        offset = 0
        pos = [0,0]

        wall = True
        limits = True
        
        while (wall or limits) and (tries > 0):
            offset = np.random.normal(0, std)
            pos = center + s * x_axis + offset * y_axis

            wall = utils.is_inside_wall(offset, wall_width, robot_radius)
            limits = utils.is_outside_limits(pos, xlim, ylim, robot_radius)
            tries -= 1
    
    else:
        # End sampling (near cylinder caps)
        sign = np.random.choice([-1, 1])
        cap_center = center + sign * (wall_length/2) * x_axis

        # Outside the cylinder
        offset = [0,0]
        pos = cap_center

        cylinder = True
        limits = True
        
        while (cylinder or limits) and (tries > 0):
            offset = np.random.normal(0, std, size=2)
            
            # Ensure that the sampling is towards the "out" direction
            dot = np.dot(offset, x_axis)
            if dot*sign < 0:
                offset -= 2 * dot * x_axis
            pos = cap_center + offset

            cylinder = utils.is_inside_cylinder(pos, cap_center, tower_radius, robot_radius)
            limits = utils.is_outside_limits(pos, xlim, ylim, robot_radius)
            tries -= 1
    
    return pos if tries > 0 else None
    

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
        sample = None
        while sample is None:
            wall = np.random.choice(wall_objects, p=wall_lengths)
            sample = sample_near_wall_full(wall, xlim, ylim, robot_radius, std=std)
        return sample
    # Random
    sample = sample_pose(xlim, ylim)
    return sample


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
