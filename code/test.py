
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
# parser.add_argument('-sigma', default=0.005, type=float)
# parser.add_argument('-eval', type=str, default=None)
parser.add_argument('-example', type=int, default=1)
parser.add_argument('-method', type=int, default=1)
args = parser.parse_args()

if args.example == 1:
    bunny = o3d.data.BunnyMesh()
    gt_mesh = o3d.io.read_triangle_mesh(bunny.path)
    sigma = 0.005
    max_error_map = 0.02
    # knot_mesh = o3d.data.KnotMesh()
    # gt_mesh = o3d.io.read_triangle_mesh(knot_mesh.path)
    # armadillo_mesh = o3d.data.ArmadilloMesh()
    # gt_mesh = o3d.io.read_triangle_mesh(armadillo_mesh.path)
    gt_points = np.array(gt_mesh.vertices)
    pcd = gt_mesh.sample_points_poisson_disk(5000)
    points = np.array(pcd.points)
    points_noisy = points + np.random.randn(*(points.shape)) * sigma
elif args.example == 2:
    # gt_mesh = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    # sigma = 0.03
    gt_mesh = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.5)
    sigma = 0.01
    max_error_map = 1.0
    # knot_mesh = o3d.data.KnotMesh()
    # gt_mesh = o3d.io.read_triangle_mesh(knot_mesh.path)
    # armadillo_mesh = o3d.data.ArmadilloMesh()
    # gt_mesh = o3d.io.read_triangle_mesh(armadillo_mesh.path)
    gt_points = np.array(gt_mesh.vertices)
    pcd = gt_mesh.sample_points_poisson_disk(5000)
    points = np.array(pcd.points)
    points_noisy = points + np.random.randn(*(points.shape)) * sigma
elif args.example == 3:
    # gt_mesh = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    # sigma = 0.03
    gt_mesh = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.5)
    sigma = 0.02
    max_error_map = 1.2
    # knot_mesh = o3d.data.KnotMesh()
    # gt_mesh = o3d.io.read_triangle_mesh(knot_mesh.path)
    # armadillo_mesh = o3d.data.ArmadilloMesh()
    # gt_mesh = o3d.io.read_triangle_mesh(armadillo_mesh.path)
    gt_points = np.array(gt_mesh.vertices)
    pcd = gt_mesh.sample_points_poisson_disk(5000)
    points = np.array(pcd.points)
    points_noisy = points + np.random.randn(*(points.shape)) * sigma


pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
pcd_noisy = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_noisy))

if args.method == 1:
    if args.example == 1:
        # n_patch = 15000
        # n_patch = 4000
        n_patch = 4000
        # patch_size = 50
        patch_size = 10
        n_patch_neighbor = 10
        n_iter = 5
        optim_M_iter = 10
        m_offdiag_n_block_iter = 10
        m_offdiag_n_pg_iter = 10
        m_diag_n_pg_iter = 10
        m_offdiag_pg_step_size = 1e-6
        m_diag_pg_step_size = 1e-6
        max_trace = 3
        gamma_arr = 0.7**(np.arange(n_iter)+1) * (np.exp(1)-1)**(1-np.arange(n_iter))
        gamma_arr = 0.1*np.exp(-np.linspace(0,30,101))
    if args.example == 2:
        n_patch = 4000
        patch_size = 10
        n_patch_neighbor = 10
        n_iter = 5
        optim_M_iter = 10
        m_offdiag_n_block_iter = 10
        m_offdiag_n_pg_iter = 10
        m_diag_n_pg_iter = 10
        m_offdiag_pg_step_size = 1e-6
        m_diag_pg_step_size = 1e-6
        max_trace = 3
        gamma_arr = 0.7**(np.arange(n_iter)+1) * (np.exp(1)-1)**(1-np.arange(n_iter))
        gamma_arr = 0.1*np.exp(-np.linspace(0,30,101))
        # gamma_arr = 25.0 * (np.exp((np.arange(30)) / 20.) - 1)
    points_denoised, M = optim.denoise_point_cloud(points_noisy, n_patch, patch_size, n_patch_neighbor,
                                              n_iter, optim_M_iter, m_offdiag_n_block_iter, m_offdiag_n_pg_iter, m_diag_n_pg_iter,
                                              m_offdiag_pg_step_size, m_diag_pg_step_size,
                                              max_trace=max_trace, gamma=gamma_arr, ground_truth=gt_points)
elif args.method == 2:
    if args.example == 1:
        n_iter = 10
        sample_rate = 1
        n_neighbor = 15
        sigma_d = 0.25
        sigma_n = 0.25
        lambd = 1
        lambd_arr = 50*np.exp(-np.linspace(0,50,101))
    elif args.example == 2:
        n_iter = 10
        sample_rate = 1
        n_neighbor = 15
        sigma_d = 0.25
        sigma_n = 0.25
        lambd = 1
        lambd_arr = 50*np.exp(-np.linspace(0,50,101))
    points_denoised = optim.my_denoise_algo(points_noisy, n_iter, sample_rate, n_neighbor, sigma_d, sigma_n, lambd_arr, ground_truth=np.array(gt_mesh.vertices))

pcd_denoised = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_denoised))

noisy_distance = geometry.compute_distance_between_points(gt_points, points_noisy)
denoised_distance = geometry.compute_distance_between_points(gt_points, points_denoised)

print('    Noisy PCD MSE: {:.3e}'.format(np.mean(noisy_distance)))
print(' Denoised PCD MSE: {:.3e}'.format(np.mean(denoised_distance)))

mesh_noisy = geometry.surface_reconstruction(pcd_noisy, method='poisson')
mesh_noisy_distance = geometry.compute_distance_between_points(gt_points, np.array(mesh_noisy.vertices))

mesh_denoised = geometry.surface_reconstruction(pcd_denoised, method='poisson')
mesh_denoised_distance = geometry.compute_distance_between_points(gt_points, np.array(mesh_denoised.vertices))

max_distance = np.max([np.amax(mesh_noisy_distance), np.amax(mesh_denoised_distance)]) / 8
# max_distance = 40
mesh_noisy = geometry.color_mesh(mesh_noisy, 1-np.exp(-mesh_noisy_distance/max_error_map*4), valmax=1)
mesh_denoised = geometry.color_mesh(mesh_denoised, 1-np.exp(-mesh_denoised_distance/max_error_map*4), valmax=1)

print('   Noisy MESH MSE: {:.3e}'.format(np.mean(mesh_noisy_distance)))
print('Denoised MESH MSE: {:.3e}'.format(np.mean(mesh_denoised_distance)))

if args.method == 1:
    plt.pcolormesh(M[::-1])
    plt.colorbar()
    plt.show()

pcd_noisy.paint_uniform_color([0,0,1])
pcd_denoised.paint_uniform_color([0,0,1])
mesh_denoised2 = copy.deepcopy(mesh_denoised)
mesh_denoised2.paint_uniform_color([0,0,1])
gt_mesh.paint_uniform_color([0.5,0.5,0.5])
gt_mesh.compute_vertex_normals()

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


exit()
