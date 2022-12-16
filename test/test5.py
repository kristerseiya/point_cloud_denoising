import open3d as o3d
import numpy as np
import transforms3d
from scipy.spatial.transform import *
import matplotlib.pyplot as plt
import json
from PIL import Image

bunny = o3d.data.BunnyMesh()
bun_mesh = o3d.io.read_triangle_mesh(bunny.path)
center = bun_mesh.get_center()
center = np.array(center)
bound = bun_mesh.get_axis_aligned_bounding_box()
bound = np.array(bound)
gt_mesh = o3d.t.geometry.TriangleMesh()
gt_mesh = gt_mesh.from_legacy(bun_mesh)
# gt_mesh.compute_vertex_normals()
# f = open('bun_depth.json')
# config = json.load(f)
# f.close()
intrinsic_matrix = np.array([[935.30743608719399, 0.0, 0.0],
                             [0.0, 935.30743608719399, 0.0],
                             [959.5, 539.5, 1.0]])
extrinsic_matrix = np.eye(4)

scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(gt_mesh)
rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(o3d.core.Tensor(intrinsic_matrix), o3d.core.Tensor(extrinsic_matrix), 1000, 1000)
rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
    fov_deg=70,
    center=center,
    eye=center+np.array([0,0.1,0.2]),
    up=[0, 0, 1],
    width_px=640,
    height_px=480,
)
ans = scene.cast_rays(rays)
depth_map = ans['t_hit'].numpy()
plt.imshow(depth_map + np.random.randn(*depth_map.shape)*0.01, cmap='gray')
plt.show()


exit()
