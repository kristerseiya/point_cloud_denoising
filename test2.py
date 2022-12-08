
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

import geometry
import optim

parser = argparse.ArgumentParser()
parser.add_argument('-sigma', default=0.005, type=float)
parser.add_argument('-eval', type=str, default=None)
args = parser.parse_args()

xx, yy = np.meshgrid(np.linspace(-1,1,50), np.linspace(-1,1,50))
points = np.stack([xx.flatten(), yy.flatten(), np.zeros((xx.size))], axis=1)
points_noisy = points + np.random.randn(*(points.shape)) * args.sigma

pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
pcd_noisy = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_noisy))

n_patch = 400
patch_size = 9
n_patch_neighbor = 10
n_iter = 7
optim_M_iter = 10
m_offdiag_n_block_iter = 10
m_offdiag_n_pg_iter = 10
m_diag_n_pg_iter = 10
m_offdiag_pg_step_size = 1e-6
m_diag_pg_step_size = 1e-6
max_trace = 3
points_denoised = optim.denoise_point_cloud_2(points_noisy, n_patch, patch_size, n_patch_neighbor,
                                          n_iter, optim_M_iter, m_offdiag_n_block_iter, m_offdiag_n_pg_iter, m_diag_n_pg_iter,
                                          m_offdiag_pg_step_size, m_diag_pg_step_size,
                                          max_trace=max_trace)
pcd_denoised = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_denoised))

noisy_distance = np.sum(points_noisy[:,2]**2)
denoised_distance = np.sum(points_denoised[:,2]**2)

print(noisy_distance)
print(denoised_distance)

if args.eval != None:
    pcd_denoised.estimate_normals()
    pcd_denoised.orient_normals_consistent_tangent_plane(10)

if args.eval == 'poisson':
    mesh_denoised, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_denoised, depth=9)
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.0002)
    mesh_denoised.remove_vertices_by_mask(vertices_to_remove)
elif args.eval == 'alpha':
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd_denoised)
    mesh_denoised = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_denoised, 0.12, tetra_mesh, pt_map)
#
# if args.eval != None:
#     distance = geometry.compute_distance_between_points(np.array(bun_mesh.vertices), np.array(mesh_denoised.vertices))
#     distance_normalized = distance / np.amax(distance)
#     norm = matplotlib.colors.Normalize(vmin=0, vmax=1/3)
#     cmap = cm.plasma
#     m = cm.ScalarMappable(norm=norm, cmap=cmap)
#     colors = m.to_rgba(distance_normalized)
#     colors = colors[:,:3]
#
#     mesh_denoised.compute_vertex_normals()
#     mesh_denoised.vertex_colors = o3d.utility.Vector3dVector(colors)

pcd_noisy.paint_uniform_color([1,0,0])
pcd_denoised.paint_uniform_color([0,0,1])


viewer = o3d.visualization.Visualizer()
viewer.create_window()
viewer.add_geometry(pcd_noisy)
viewer.add_geometry(pcd_denoised)
if args.eval != None:
    viewer.add_geometry(mesh_denoised)
viewer.run()

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
