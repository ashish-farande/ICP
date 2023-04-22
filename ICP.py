
import copy
import numpy as np

from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors

from Viewer import Viewer


class ICP:
    def __init__(self) -> None:
        self._viewer_handle: Viewer = Viewer()

    def __call__(self, source_pcd: np.ndarray, target_pcd: np.ndarray) -> np.ndarray:

        source_init = copy.deepcopy(source_pcd)

        T = np.eye(4)
        orig_centroid = np.mean(source_pcd, axis=0)
        R = np.array(Rotation.random().as_matrix())
        t = np.random.rand(3, 1)
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
