
import numpy as np
from scipy.sparse.linalg import lobpcg as LOBPCG
from scipy.spatial import cKDTree
import open3d as o3d
from matplotlib import cm
import matplotlib
import copy

class MyPointCloud():
    def __init__(self, points=None, normals=None):
        self.points = points
        if points != None:
            self.n_point = points.shape[0]
        else:
            self.n_point = 0
        self.normals = normals
        self.kdtree = None
        self.adjacency = None

    def build_kdtree(self):
        self.kdtree = cKDTree(self.points, leafsize=16)

    def compute_nieghbors(self, n_neighbor):
        if self.kdtree == None:
            self.build_kdtree()

        self.neighbor_idx = np.zeros((self.n_point, n_neighbor), dtype=int)
        for i, point in enumerate(self.points):
            self.nieghbor_idx[i] = self.kdtree.query(points, range(2,n_neighbor+2))[1]

    def estimate_normals(self, n_neighbor):
        if self.normals != None:
            return
        if self.neighbor_idx == None:
            self.compute_neighbors(n_neighbor)

        self.normals = np.zeros_like(self.points)
        x = np.random.randn(3,1)
        for i in range(self.n_point):
            local_points = np.concatenate([self.points[i], self.points[self.neighbor_idx[i]] ], axis=0)
            local_mean = np.mean(local_points, 0)
            local_cov = (local_points - local_mean).T @ (local_points - local_mean)
            _, eigvecs = LOBPCG(local_cov, x, largest=False)
            self.normals[i] = eigvecs[:,0]

def compute_neighbors(points, n_neighbor, kdtree=None):
    if kdtree is None:
        kdtree = cKDTree(points, leafsize=16)
    n_point = points.shape[0]
    neighbor_idx = np.zeros((points.shape[0], n_neighbor), dtype=int)
    neighbor_count = np.zeros((n_point), dtype=int)
    neighbor_dist = np.zeros((n_point, n_neighbor), dtype=int)
    for i, point in enumerate(points):
        neighbor_info = kdtree.query(point, range(2,n_neighbor+2))
        neighbor_idx[i] = neighbor_info[1]
        neighbor_dist[i] = neighbor_info[0]
        neighbor_count[neighbor_idx[i]] = neighbor_count[neighbor_idx[i]] + 1
    return neighbor_idx, neighbor_dist, neighbor_count

def compute_normals(points, neighbor_idx, init_normals=None):

    if init_normals is None:
        init_normals = np.random.randn(*points.shape)

    normals = np.zeros(points.shape)

    for i, point in enumerate(points):
        local_points = np.concatenate([ [point], points[neighbor_idx[i]] ], axis=0)
        local_mean = np.mean(local_points, 0)
        local_cov = (local_points - local_mean).T @ (local_points - local_mean)
        _, eigvecs = LOBPCG(local_cov, np.reshape(init_normals[i],(3,1)), largest=False)
        normals[i] = eigvecs[:,0]

    return normals

def farthest_point_sampling(points, n_samples, return_idx=True):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N
    """
    points = np.array(points)

    # Represent the points by their indices in points
    points_left = np.arange(len(points)) # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int') # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf') # [P]

    # Select a point from points by its index, save it
    selected = np.random.choice(n_samples)
    sample_inds[0] = points_left[selected]

    # Delete selected
    points_left = np.delete(points_left, selected) # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i-1]

        dist_to_last_added_point = (
            (points[last_added] - points[points_left])**2).sum(-1) # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point,
                                        dists[points_left]) # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    if return_idx:
        return sample_inds
    return points[sample_inds]

def compute_distance_between_pointclouds(ref, pcl):
    ref_kdtree = cKDTree(np.array(ref.points), leafsize=16)
    pcl_points = np.array(pcl.points)
    distance = np.zeros((pcl_points.shape[0]))
    for i, point in enumerate(pcl_points):
        dist, _ = ref_kdtree.query([point], 1)
        distance[i] = dist
    return distance

def compute_distance_between_points(ref, points, ref_is_kdtree=False):
    if not ref_is_kdtree:
        ref = cKDTree(np.array(ref), leafsize=16)
    distance = np.zeros((points.shape[0]))
    for i, point in enumerate(points):
        dist, _ = ref.query([point], 1)
        distance[i] = dist
    return distance

def surface_reconstruction(pcd, method='poisson'):

    if not pcd.has_normals():
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)

    if method == 'poisson':
        rec_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)
        densities = np.asarray(densities)
        vertices_to_remove = densities < np.quantile(densities, 0.008)
        rec_mesh.remove_vertices_by_mask(vertices_to_remove)
    elif method == 'alpha':
        tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.01, tetra_mesh, pt_map)

    return rec_mesh

def color_mesh(mesh, vals, valmax=None):

    mesh = copy.deepcopy(mesh)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=valmax)
    cmap = cm.bwr
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = m.to_rgba(vals)
    colors = colors[:,:3]
    mesh.compute_vertex_normals()
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return mesh
