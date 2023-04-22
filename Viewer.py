import time
import open3d
import numpy as np


class Viewer:
    SLEEP_INTERVAL_S: float = 0.1

    def __init__(self) -> None:
        self._vis = open3d.visualization.Visualizer()
        self._vis.create_window()
        self._pcd = open3d.geometry.PointCloud()

    def __del__(self):
        self._vis.destroy_window()

    def add(self, source_pcd: np.array, target_pcd: np.array):
        tmp = np.hstack((source_pcd, target_pcd))
        pts = open3d.utility.Vector3dVector(tmp.reshape([-1, 3]))
        self._pcd.points = pts
        self._vis.add_geometry(self._pcd)

    def update(self, source_pcd: np.array, target_pcd: np.array):
        tmp = np.hstack((source_pcd, target_pcd))
        pts = open3d.utility.Vector3dVector(tmp.reshape([-1, 3]))
        self._pcd.points = pts
        self._vis.update_geometry(self._pcd)
        self._vis.poll_events()
        self._vis.update_renderer()
        time.sleep(Viewer.SLEEP_INTERVAL_S)
