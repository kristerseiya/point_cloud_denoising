import open3d as o3d
import numpy as np
import transforms3d
from scipy.spatial.transform import *
import matplotlib.pyplot as plt
import json
from PIL import Image


bunny = o3d.data.BunnyMesh()
gt_mesh = o3d.io.read_triangle_mesh(bunny.path)
pcd = gt_mesh.sample_points_poisson_disk(5000)
points = np.array(pcd.points)
# noise = np.random.randn(points.shape[0]) * 0.004
# points[:,2] = points[:,2] + noise
noise = np.random.randn(*points.shape) * 0.004 * 0
points = points + noise
pcd.points = o3d.utility.Vector3dVector(points)
pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals

pcd.estimate_normals()
pcd.orient_normals_consistent_tangent_plane(100)

mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=5)

densities = np.asarray(densities)
density_colors = plt.get_cmap('plasma')(
    (densities - densities.min()) / (densities.max() - densities.min()))
density_colors = density_colors[:, :3]
density_mesh = o3d.geometry.TriangleMesh()
density_mesh.vertices = mesh.vertices
density_mesh.triangles = mesh.triangles
density_mesh.triangle_normals = mesh.triangle_normals
density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)

vertices_to_remove = densities < np.quantile(densities, 0.015)
mesh.remove_vertices_by_mask(vertices_to_remove)

mesh.compute_vertex_normals()
mesh.paint_uniform_color([1, 0.706, 0])

viewer = o3d.visualization.Visualizer()
viewer.create_window()
viewer.add_geometry(mesh)
viewer.run()

# control = viewer.get_view_control()
# control.convert_from_pinhole_camera_parameters(camera_parameters)

# print("show depth")
# print(np.asarray(depth))
# plt.imshow(np.asarray(depth))
# plt.imsave("testing_depth.png", np.asarray(depth), dpi = 1)
# plt.show()
