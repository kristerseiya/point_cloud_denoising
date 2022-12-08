
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

n_neighbor = 50

bunny = o3d.data.BunnyMesh()
bun_mesh = o3d.io.read_triangle_mesh(bunny.path)
gt_mesh = bun_mesh
gt_points = np.array(bun_mesh.vertices)
pcd = bun_mesh.sample_points_poisson_disk(5000)
points = np.array(pcd.points)
# points_noisy = points + np.random.randn(*(points.shape)) * args.sigma
#
# # xx, yy = np.meshgrid(np.linspace(-1,1,100), np.linspace(-1,1,100))
# # points = np.stack([xx.flatten(), yy.flatten(), np.zeros((xx.size))], axis=1)
# # points_noisy = points + np.random.randn(*(points.shape)) * args.sigma

point_kdtree = scipy.spatial.cKDTree(points)

n_point = points.shape[0]
normals = np.random.randn(*points.shape)
neighbor_idx = np.zeros((n_point, n_neighbor), dtype=int)
neighbor_dist = np.zeros((n_point, n_neighbor), dtype=int)
neighbor_count = np.zeros((n_point), dtype=int)

for i, i_point in enumerate(points):
    neighbor_info = point_kdtree.query(i_point, n_neighbor+1)
    neighbor_idx[i] = neighbor_info[1][1:]
    neighbor_dist[i] = neighbor_info[0][1:]
    neighbor_count[neighbor_idx[i]] = neighbor_count[neighbor_idx[i]] + 1
    neighbor_points = np.concatenate([ [i_point], points[neighbor_idx[i]], 2*i_point - points[neighbor_idx[i]]  ], axis=0)
    neighbor_mu = np.copy(i_point)
    # neighbor_points = np.concatenate([ [i_point], points[neighbor_idx[i]] ], axis=0)
    # neighbor_mu = np.mean(neighbor_points, axis=0)
    X = neighbor_points - neighbor_mu
    _, eigvecs = LOBPCG(X.T @ X, np.reshape(normals[i], (3,1)), largest=False)
    # _, eigvecs = LOBPCG(X.T @ X, np.random.randn(3,1), largest=False)
    normals[i] = eigvecs[:,0] / np.linalg.norm(eigvecs[:,0], ord=2)

project_dist = np.zeros((n_neighbor*n_point))
ortho_dist = np.zeros((n_neighbor*n_point))

for i, i_point in enumerate(points):
    i_normal = normals[i]
    i_neighbor_points = points[neighbor_idx[i]]
    diff_vec = i_neighbor_points - i_point
    ortho_vec = np.expand_dims(diff_vec @ i_normal, axis=1) * np.expand_dims(i_normal, axis=0)
    project_vec = diff_vec - ortho_vec
    project_dist[(i*n_neighbor):(i*n_neighbor+n_neighbor)] = np.linalg.norm(diff_vec, ord=2, axis=1)
    ortho_dist[(i*n_neighbor):(i*n_neighbor+n_neighbor)] = np.linalg.norm(ortho_vec, ord=2, axis=1)

bin_count = np.zeros((100,), dtype=int)
data = np.zeros((len(bin_count),len(ortho_dist)))
bin_val = np.linspace(0.003, 0.012, len(bin_count)+1)
bin_min = bin_val[0]
bin_max = bin_val[-1]
bin_incr = bin_val[1] - bin_val[0]
for i in range(len(ortho_dist)):
    d = project_dist[i]
    if d > bin_max or d < bin_min:
        continue
    idx = int(np.floor((d - bin_min) / bin_incr))
    data[idx, bin_count[idx]] = ortho_dist[i]
    bin_count[idx] += 1

var_data = np.zeros((len(bin_count)))
for i in range(len(bin_count)):
    var_data[i] = np.std(np.concatenate([ data[i,:bin_count[i]], -data[i,:bin_count[i]] ]) )

plt.figure(1)
plt.scatter(project_dist, ortho_dist, s=0.1)

plt.figure(2)
plt.scatter(bin_val[:-1]+bin_incr, var_data)
# plt.scatter(np.arange(100), np.arange(100), s=1)
plt.show()
exit()
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

max_distance = np.max([np.amax(mesh_noisy_distance), np.amax(mesh_denoised_distance)]) / 10
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


exit()
