from open3d import *
import numpy as np
import transforms3d
from scipy.spatial.transform import *
import matplotlib.pyplot as plt
import json
from PIL import Image


original_pose = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 2],
                             [0, 0, 0, 1]])

rotations_eul = np.random.randint(-50, 50, size=(4000, 3)) / 100.0

point_list = []

# for i, eu in enumerate(rotations_eul):
#     pose = np.dot(transforms3d.affines.compose([0, 0, 0],
#                                                   Rotation.from_euler("xyz", eu).as_matrix(),
#                                                   [1, 1, 1]),
#                      np.asarray(original_pose))
#     point = np.dot(pose, np.array([0,0,0,1]).T).T[0:3]
#     point_list.append(point)

img_pil = Image.open('bun_depth_8bit.png')
img = np.array(img_pil) / 255.
img = img + np.random.randn(*img.shape) * 0.0 * (img > 0.01)
img = np.clip(img, 0, None)
img = img.astype(np.float32)

# depth_image = io.read_image('bun_depth.png')
depth_image = geometry.Image(img)

f = open('bun_depth.json')
config = json.load(f)
f.close()

intrinsic_matrix = np.array(config['intrinsic']['intrinsic_matrix']).reshape((3,3)).T
extrinsic_matrix = np.array(config['extrinsic']).reshape((4,4)).T
print(intrinsic_matrix)
print(extrinsic_matrix)

cam = camera.PinholeCameraIntrinsic()
cam.intrinsic_matrix = intrinsic_matrix
pcd = geometry.PointCloud.create_from_depth_image(depth_image, cam, extrinsic_matrix)
pcd.estimate_normals(search_param=geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(100)
mesh, densities = geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
mesh.compute_vertex_normals()

densities = np.asarray(densities)
density_colors = plt.get_cmap('plasma')(
    (densities - densities.min()) / (densities.max() - densities.min()))
density_colors = density_colors[:, :3]
density_mesh = geometry.TriangleMesh()
density_mesh.vertices = mesh.vertices
density_mesh.triangles = mesh.triangles
density_mesh.triangle_normals = mesh.triangle_normals
density_mesh.vertex_colors = utility.Vector3dVector(density_colors)

vertices_to_remove = densities < np.quantile(densities, 0.01)
mesh.remove_vertices_by_mask(vertices_to_remove)

# pcd.compute_vertex_normals()
# pcd = geometry.PointCloud()
# pcd.points = utility.Vector3dVector(point_list)

camera_parameters = camera.PinholeCameraParameters()
camera_parameters.extrinsic = np.array([[1,0,0,1],
                                           [0,1,0,0],
                                           [0,0,1,2],
                                           [0,0,0,1]])
camera_parameters.intrinsic.set_intrinsics(width=1920, height=1080, fx=1000, fy=1000, cx=959.5, cy=539.5)

viewer = visualization.Visualizer()
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
