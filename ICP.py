import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
import open3d
from utils import compare_points
import copy


class Viewer:
    def __init__(self) -> None:
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window()
        self.pcd = open3d.geometry.PointCloud()

    def __del__(self):
        self.vis.destroy_window()

    def add(self, source_pcd: np.array, target_pcd: np.array):
        tmp = np.hstack((source_pcd, target_pcd))
        pts = open3d.utility.Vector3dVector(tmp.reshape([-1, 3]))
        self.pcd.points = pts
        self.vis.add_geometry(self.pcd)

    def update(self, source_pcd: np.array, target_pcd: np.array):
        tmp = np.hstack((source_pcd, target_pcd))
        pts = open3d.utility.Vector3dVector(tmp.reshape([-1, 3]))
        self.pcd.points = pts
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
        time.sleep(0.1)


class ICP:
    def __init__(self) -> None:
        self._viewer_handle = Viewer()

    def __del__(self):
        self._viewer_handle.__del__()

    def __call__(self, source_pcd, target_pcd):

        source_init = copy.deepcopy(source_pcd)

        T = np.eye(4)
        orig_centroid = np.mean(source_pcd, axis=0)
        R = np.array(Rotation.random().as_matrix())
        t= np.random.rand(3,1)
        source_pcd = source_pcd @ R.T + t.T

        self._viewer_handle.add(source_pcd, target_pcd)

        nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        centroid_tgt = np.mean(target_pcd, axis=0)

        for i in range(50):
            nn.fit(target_pcd)
            indices = nn.kneighbors(source_pcd, return_distance=False)
            indices = indices.ravel()
            matched_dst_pts = target_pcd[indices, :]

            centroid_src = np.mean(source_pcd, axis=0)

            P = source_pcd - centroid_src
            Q = matched_dst_pts - centroid_tgt

            # rotation matrix
            M = Q.T @ P
            U, S, Vt = np.linalg.svd(M)
            R_int = U @ Vt
            # special reflection case
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
            R_int = U @ Vt
            t_int = centroid_tgt - R_int@centroid_src

            source_pcd = source_pcd @ R_int.T + t_int.T

            R = R_int@R
            t = centroid_tgt - R@orig_centroid

            tmp = source_init @ R.T + t
            self._viewer_handle.update(tmp, target_pcd)

        # Implement your own algorithm here.
        T[:3, :3] = R
        T[:3, 3] = t.squeeze()
        return T
