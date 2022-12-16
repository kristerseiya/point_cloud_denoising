from open3d import *
import numpy as np
import transforms3d
from scipy.spatial.transform import *
import matplotlib.pyplot as plt
from PIL import Image

def uniform_quantizer(x, max, n_bit=8):
    x = x / max * 2**n_bit
    x = np.floor(x).astype(int)
    return x

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

pcd = io.read_triangle_mesh('bunny/reconstruction/bun_zipper.ply')
pcd.compute_vertex_normals()
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
viewer.add_geometry(pcd)
viewer.run()

# control = viewer.get_view_control()
# control.convert_from_pinhole_camera_parameters(camera_parameters)

depth = np.array(viewer.capture_depth_float_buffer())
# plt.hist(np.array(depth).flatten(), bins=300)
# plt.show()
depth_q = uniform_quantizer(depth, 0.25, n_bit=8)

Image.fromarray(depth_q.astype(np.uint8)).save('bun_depth_6bit.png')

# depth = viewer.capture_depth_float_buffer()
# print("show depth")
# print(np.asarray(depth))
# plt.imshow(np.asarray(depth))
# plt.imsave("testing_depth.png", np.asarray(depth), dpi = 1)
# plt.show()
