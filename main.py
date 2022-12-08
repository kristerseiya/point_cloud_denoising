
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

import geometry
import optim

parser = argparse.ArgumentParser()
parser.add_argument('-sigma', default=0.005, type=float)
parser.add_argument('-eval', type=str, default=None)
args = parser.parse_args()

bunny = o3d.data.BunnyMesh()
bun_mesh = o3d.io.read_triangle_mesh(bunny.path)
# bun_pcl = o3d.geometry.PointCloud(bun_mesh.vertices)

bun_pcl_s = bun_mesh.sample_points_poisson_disk(50000)
bun_pcl_s_points = np.array(bun_pcl_s.points)
bun_pcl_s_noisy_points = bun_pcl_s_points + np.random.randn(*(bun_pcl_s_points.shape)) * args.sigma
bun_pcl_s_noisy = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bun_pcl_s_noisy_points))

noisy_distance = geometry.compute_distance_between_points(np.array(bun_mesh.vertices), bun_pcl_s_noisy_points)

# bun_pcl_s_denoised = bun_pcl_s_noisy
n_patch = 15000
patch_size = 9
n_patch_neighbor = 10
n_iter = 7
optim_M_iter = 10
# m_offdiag_n_block_iter = 5
# m_offdiag_n_pg_iter = 100
# m_diag_n_pg_iter = 100
m_offdiag_n_block_iter = 10
m_offdiag_n_pg_iter = 10
m_diag_n_pg_iter = 10
m_offdiag_pg_step_size = 1e-6
m_diag_pg_step_size = 1e-6
max_trace = 3
bun_pcl_s_denoised = optim.denoise_point_cloud(bun_pcl_s_noisy, n_patch, patch_size, n_patch_neighbor,
                                               n_iter, optim_M_iter, m_offdiag_n_block_iter, m_offdiag_n_pg_iter, m_diag_n_pg_iter,
                                               m_offdiag_pg_step_size, m_diag_pg_step_size,
                                               max_trace=max_trace)

rec_distance = geometry.compute_distance_between_points(np.array(bun_mesh.vertices), np.array(bun_pcl_s_denoised.points))

print(np.mean(noisy_distance))
print(np.mean(rec_distance))

# bun_pcl_s_denoised, _ = bun_pcl_s_denoised.remove_statistical_outlier(10, std_ratio=2)

if args.eval != None:
    bun_pcl_s_denoised.estimate_normals()
    bun_pcl_s_denoised.orient_normals_consistent_tangent_plane(10)

if args.eval == 'poisson':
    rec_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(bun_pcl_s_denoised, depth=9)
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.0002)
    rec_mesh.remove_vertices_by_mask(vertices_to_remove)
elif args.eval == 'alpha':
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(bun_pcl_s_denoised)
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(bun_pcl_s_denoised, 0.12, tetra_mesh, pt_map)

if args.eval != None:
    distance = geometry.compute_distance_between_points(np.array(bun_mesh.vertices), np.array(rec_mesh.vertices))
    distance_normalized = distance / np.amax(distance)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1/3)
    cmap = cm.plasma
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = m.to_rgba(distance_normalized)
    colors = colors[:,:3]

    rec_mesh.compute_vertex_normals()
    rec_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

bun_pcl_s_noisy.paint_uniform_color([1,0,0])
bun_pcl_s_denoised.paint_uniform_color([0,0,1])

viewer = o3d.visualization.Visualizer()
viewer.create_window()
viewer.add_geometry(bun_pcl_s_denoised)
viewer.add_geometry(bun_pcl_s_noisy)
if args.eval != None:
    viewer.add_geometry(rec_mesh)
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
