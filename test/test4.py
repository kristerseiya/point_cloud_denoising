import open3d as o3d
import numpy as np
import transforms3d
from scipy.spatial.transform import *
import matplotlib.pyplot as plt
import json
from PIL import Image

bunny = o3d.data.BunnyMesh()
gt_mesh = o3d.io.read_triangle_mesh(bunny.path)
gt_mesh.compute_vertex_normals()
# f = open('bun_depth.json')
# config = json.load(f)
# f.close()

# intrinsic_matrix = np.array(config['intrinsic']['intrinsic_matrix']).reshape((3,3)).T
# extrinsic_matrix = np.array(config['extrinsic']).reshape((4,4)).T
intrinsic_matrix = np.array([[935.30743608719399, 0.0, 0.0],
                             [0.0, 935.30743608719399, 0.0],
                             [959.5, 539.5, 1.0]])
# intrinsic_matrix = intrinsic_matrix.T
extrinsic_matrix = np.eye(4)

R = extrinsic_matrix[:3,:3]
R = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
t = extrinsic_matrix[3,:3]
t = np.array([0.016840399999999998,0.11015419999999999, 0.18723805508136665])
print(intrinsic_matrix)
print(extrinsic_matrix)

x = np.array([[600,400,1],[800,1000,1],[1200,1000,1],[1000,200,1]])
w = np.linspace(0, 0.3, 100)
wx = np.kron(np.expand_dims(w,axis=1), x)

xyz = (wx @ np.linalg.inv(intrinsic_matrix) - t) @ np.linalg.inv(R)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
# mesh.paint_uniform_color([1, 0.706, 0])

vertices = np.array(gt_mesh.vertices)
xy = (vertices @ R + t) @ intrinsic_matrix
xy = xy[:,:2] / np.expand_dims(xy[:,2], axis=1)

# plt.scatter(xy[:,0], xy[:,1])
# plt.show()
#
# exit()

viewer = o3d.visualization.Visualizer()
viewer.create_window()
viewer.add_geometry(gt_mesh)
# ctrl = viewer.get_view_control()
# ctrl.convert_from_pinhole_camera_parameters(o3d.utility.Vector3dVector(intrinsic_matrix))
viewer.add_geometry(pcd)

viewer.run()

# control = viewer.get_view_control()
# control.convert_from_pinhole_camera_parameters(camera_parameters)

# print("show depth")
# print(np.asarray(depth))
# plt.imshow(np.asarray(depth))
# plt.imsave("testing_depth.png", np.asarray(depth), dpi = 1)
# plt.show()
