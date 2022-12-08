
import open3d as o3d
import numpy as np
import transforms3d
from scipy.spatial.transform import *
import matplotlib.pyplot as plt
import json
from PIL import Image
import argparse
from matplotlib import cm
import matplotlib
import scipy.spatial
from scipy.sparse.linalg import lobpcg as LOBPCG
import copy
import warnings
warnings.filterwarnings('error')

import geometry
import optim

parser = argparse.ArgumentParser()
parser.add_argument('-sigma', default=0.005, type=float)
parser.add_argument('-eval', type=str, default=None)
args = parser.parse_args()

def my_denoise_algo(points, n_iter, sample_rate, n_neighbor, sigma_d, sigma_n):

    point_kdtree = scipy.spatial.cKDTree(points)
    n_point = points.shape[0]
    normals = np.random.randn(*points.shape)
    n_sample = int(sample_rate * n_point)
    neighbor_idx = np.zeros((n_point, n_neighbor), dtype=int)
    neighbor_dist = np.zeros((n_point, n_neighbor), dtype=int)
    neighbor_count = np.zeros((n_point), dtype=int)

    for it in range(n_iter):

        print('{:d}/{:d}'.format(it+1, n_iter))
        directions = np.zeros((n_point, 3))
        neighbor_count = np.zeros((n_point), dtype=int)

        print('computing normals')
        for i, i_point in enumerate(points):
            neighbor_info = point_kdtree.query(i_point, n_neighbor+1)
            neighbor_idx[i] = neighbor_info[1][1:]
            neighbor_dist[i] = neighbor_info[0][1:]
            neighbor_count[neighbor_idx[i]] = neighbor_count[neighbor_idx[i]] + 1
            neighbor_points = np.concatenate([ [i_point], points[neighbor_idx[i]] ], axis=0)
            neighbor_mu = np.mean(neighbor_points, axis=0)
            X = neighbor_points - neighbor_mu
            _, eigvecs = LOBPCG(X.T @ X, np.reshape(normals[i], (3,1)), largest=False)
            # _, eigvecs = LOBPCG(X.T @ X, np.random.randn(3,1), largest=False)
            normals[i] = eigvecs[:,0] / np.linalg.norm(eigvecs[:,0], ord=2)

        #         for i in range(self.n_point):
        #             patch = self.points[self.adjacency[i]]
        #             patch_mean = np.mean(patch, 0)
        #             patch_cov = (patch - patch_mean).T @ (patch - patch_mean)
        #             _, eigvecs = LOBPCG(patch_cov, x, largest=False)
        #             self.normals[i] = eigvecs[:,0]

        sample_idx = np.random.choice(range(n_point), size=(n_sample), replace=False)
        for i, i_sample_idx in enumerate(sample_idx):
            i_point = points[i_sample_idx]
            i_normal = normals[i_sample_idx]
            i_neighbor_idx = neighbor_idx[i_sample_idx]
            neighbor_points = points[i_neighbor_idx]
            neighbor_normals = normals[i_neighbor_idx]
            neighbor_normal_cos = neighbor_normals @ i_normal
            neighbor_normal_dist = np.sqrt( np.abs(2 - 2 * np.abs(neighbor_normal_cos)) )
            neighbor_plane_dist = (neighbor_points - i_point) - np.expand_dims((neighbor_points - i_point) @ i_normal, axis=1) * np.expand_dims(i_normal, axis=0)
            neighbor_plane_dist = np.linalg.norm(neighbor_plane_dist, ord=2, axis=1)
            # weights = np.exp( - neighbor_dist[i_sample_idx]**2 / sigma_d**2 / 2 - neighbor_normal_dist**2 / sigma_n**2 / 2 )
            # weights = np.exp( - neighbor_plane_dist**2 / sigma_d**2 / 2 - neighbor_normal_dist**2 / sigma_n**2 / 2 )
            weights = np.exp( - neighbor_plane_dist**2 / sigma_d**2 / 2 - neighbor_normal_dist**2 / sigma_n**2 / 2 )
            directions[i_neighbor_idx] = directions[i_neighbor_idx] + np.expand_dims( weights * ( (neighbor_points - i_point) @ i_normal ), axis=1) * np.expand_dims(i_normal, axis=0)

        nonzero_neighbor = neighbor_count != 0
        directions[nonzero_neighbor] = directions[nonzero_neighbor] / np.expand_dims(neighbor_count[nonzero_neighbor], axis=1)
        points = points - directions

    return points


bunny = o3d.data.BunnyMesh()
bun_mesh = o3d.io.read_triangle_mesh(bunny.path)
gt_mesh = bun_mesh
gt_points = np.array(bun_mesh.vertices)
pcd = bun_mesh.sample_points_poisson_disk(5000)
points = np.array(pcd.points)
points_noisy = points + np.random.randn(*(points.shape)) * args.sigma

# xx, yy = np.meshgrid(np.linspace(-1,1,100), np.linspace(-1,1,100))
# points = np.stack([xx.flatten(), yy.flatten(), np.zeros((xx.size))], axis=1)
# points_noisy = points + np.random.randn(*(points.shape)) * args.sigma

pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
pcd_noisy = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_noisy))

# noisy_distance = geometry.compute_distance_between_points(np.array(bun_mesh.vertices), bun_pcl_s_noisy_points)

# n_patch = 3000
# patch_size = 9
# n_patch_neighbor = 10
# n_iter = 7
# optim_M_iter = 10
# m_offdiag_n_block_iter = 10
# m_offdiag_n_pg_iter = 10
# m_diag_n_pg_iter = 10
# m_offdiag_pg_step_size = 1e-6
# m_diag_pg_step_size = 1e-6
# max_trace = 3
# pcd_denoised = optim.denoise_point_cloud(pcd_noisy, n_patch, patch_size, n_patch_neighbor,
#                                          n_iter, optim_M_iter, m_offdiag_n_block_iter, m_offdiag_n_pg_iter, m_diag_n_pg_iter,
#                                          m_offdiag_pg_step_size, m_diag_pg_step_size,
#                                          max_trace=max_trace)
# points_denoised = my_denoise_algo(points_noisy, 5, 1, 25, 0.01, 0.2)
# points_denoised = optim.my_denoise_algo(points_noisy, 20, 1, 50, 0.03, 0.5, 1, ground_truth=np.array(bun_mesh.vertices))
points_denoised = optim.my_denoise_algo(points_noisy, 20, 1, 50, 0.25, 0.25, 1, ground_truth=np.array(bun_mesh.vertices))
# points_denoised = optim.my_denoise_algo(points_noisy, 20, 1, 50, 500, 500, 1, ground_truth=np.array(bun_mesh.vertices))
pcd_denoised = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_denoised))
# pcd_denoised = pcd_noisy
# print(np.sum(points_noisy[:,2]**2))
# print(np.sum(points_denoised[:,2]**2))

noisy_distance = geometry.compute_distance_between_points(gt_points, points_noisy)
rec_distance = geometry.compute_distance_between_points(gt_points, points_denoised)

print(np.mean(noisy_distance))
print(np.mean(rec_distance))

# bun_pcl_s_denoised, _ = bun_pcl_s_denoised.remove_statistical_outlier(10, std_ratio=2)
# bun_pcl_s_noisy.estimate_normals()
# bun_pcl_s_noisy.orient_normals_consistent_tangent_plane(10)
#
# points = bun_pcl_s_noisy_points
# normals = np.array(bun_pcl_s_noisy.normals)
# new_normals = np.copy(normals)
#
# w = 1000
# pcd_kdtree = scipy.spatial.cKDTree(points)
# for i in range(points.shape[0]):
#     nghb_dist, nghb_idx = pcd_kdtree.query(points[i], 2)
#     weights = np.exp(-nghb_dist/(2*w**2))
#     new_normal = np.sum(np.expand_dims(weights, axis=1) * normals[nghb_idx], axis=0)
#     new_normal = new_normal / np.linalg.norm(new_normal, ord=2)
#     new_normals[i] = new_normal
#
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# pcd.normals = o3d.utility.Vector3dVector(new_normals)

mesh_noisy = geometry.surface_reconstruction(pcd_noisy, method='poisson')
mesh_noisy_distance = geometry.compute_distance_between_points(gt_points, np.array(mesh_noisy.vertices))

mesh_denoised = geometry.surface_reconstruction(pcd_denoised, method='poisson')
mesh_denoised_distance = geometry.compute_distance_between_points(gt_points, np.array(mesh_denoised.vertices))

max_distance = np.max([np.amax(mesh_noisy_distance), np.amax(mesh_denoised_distance)]) / 8
mesh_noisy = geometry.color_mesh(mesh_noisy, mesh_noisy_distance, valmax=max_distance)
mesh_denoised = geometry.color_mesh(mesh_denoised, mesh_denoised_distance, valmax=max_distance)

print(np.mean(mesh_noisy_distance))
print(np.mean(mesh_denoised_distance))

# if args.eval != None:
#     pcd_denoised.estimate_normals()
#     pcd_denoised.orient_normals_consistent_tangent_plane(10)
#
# if args.eval == 'poisson':
#     rec_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_denoised, depth=5)
#     densities = np.asarray(densities)
#     vertices_to_remove = densities < np.quantile(densities, 0.0002)
#     rec_mesh.remove_vertices_by_mask(vertices_to_remove)
# elif args.eval == 'alpha':
#     tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd_denoised)
#     rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_denoised, 0.12, tetra_mesh, pt_map)
#
# if args.eval != None:
#     distance = geometry.compute_distance_between_points(np.array(bun_mesh.vertices), np.array(rec_mesh.vertices))
#     distance_normalized = distance / np.amax(distance)
#     norm = matplotlib.colors.Normalize(vmin=0, vmax=1/3)
#     cmap = cm.plasma
#     m = cm.ScalarMappable(norm=norm, cmap=cmap)
#     colors = m.to_rgba(distance_normalized)
#     colors = colors[:,:3]
#     #
#     rec_mesh.compute_vertex_normals()
#     rec_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
#     # rec_mesh.paint_uniform_color([0.5,0.5,0.5])

pcd_noisy.paint_uniform_color([0,0,1])
pcd_denoised.paint_uniform_color([0,0,1])
mesh_denoised2 = copy.deepcopy(mesh_denoised)
mesh_denoised2.paint_uniform_color([0,0,1])
gt_mesh.paint_uniform_color([1,0,0])

# viewer = o3d.visualization.Visualizer()
# viewer.create_window()
# viewer.add_geometry(pcd_noisy)
# viewer.add_geometry(pcd_denoised)
# if args.eval != None:
#     viewer.add_geometry(rec_mesh)
# viewer.run()

vis = o3d.visualization.Visualizer()
vis.create_window(window_name='TopLeft', width=960, height=540, left=0, top=0)
vis.add_geometry(pcd_noisy)

vis2 = o3d.visualization.Visualizer()
vis2.create_window(window_name='TopRight', width=960, height=540, left=960, top=0)
vis2.add_geometry(pcd_denoised)

vis3 = o3d.visualization.Visualizer()
vis3.create_window(window_name='BottomLeft', width=960, height=540, left=0, top=640)
vis3.add_geometry(mesh_noisy)

vis4 = o3d.visualization.Visualizer()
vis4.create_window(window_name='BottomRight', width=960, height=540, left=960, top=640)
vis4.add_geometry(mesh_denoised)

vis5 = o3d.visualization.Visualizer()
vis5.create_window(window_name='BottomBottomRight', width=960, height=540, left=960, top=1280)
vis5.add_geometry(gt_mesh)
vis5.add_geometry(mesh_denoised2)

while True:
    vis.update_geometry(pcd_noisy)
    if not vis.poll_events():
        break
    vis.update_renderer()

    vis2.update_geometry(pcd_denoised)
    if not vis2.poll_events():
        break
    vis2.update_renderer()

    vis3.update_geometry(mesh_noisy)
    if not vis3.poll_events():
        break
    vis3.update_renderer()

    vis4.update_geometry(mesh_denoised)
    if not vis4.poll_events():
        break
    vis4.update_renderer()

    vis5.update_geometry(mesh_denoised2)
    vis5.update_geometry(gt_mesh)
    if not vis5.poll_events():
        break
    vis5.update_renderer()

vis.destroy_window()
vis2.destroy_window()
vis3.destroy_window()
vis4.destroy_window()
vis5.destroy_window()

# intrinsic_matrix = np.array([[935.30743608719399, 0.0, 0.0],
#                              [0.0, 935.30743608719399, 0.0],
#                              [959.5, 539.5, 1.0]])
# extrinsic_matrix = np.eye(4)
#
# scene = o3d.t.geometry.RaycastingScene()
# scene.add_triangles(gt_mesh)
# rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(o3d.core.Tensor(intrinsic_matrix), o3d.core.Tensor(extrinsic_matrix), 1000, 1000)
# rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
#     fov_deg=70,
#     center=center,
#     eye=center+np.array([0,0.1,0.2]),
#     up=[0, 0, 1],
#     width_px=640,
#     height_px=480,
# )
# ans = scene.cast_rays(rays)
# depth_map = ans['t_hit'].numpy()
# plt.imshow(depth_map + np.random.randn(*depth_map.shape)*0.01, cmap='gray')
# plt.show()
