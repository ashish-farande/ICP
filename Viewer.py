import time
import open3d
import numpy as np


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
