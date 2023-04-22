import trimesh
import numpy as np
from ICP import ICP
from utils import compare_points, compute_rre, compute_rte


"""Load data."""
source_pcd = trimesh.load("data/banana.source.ply").vertices
target_pcd = trimesh.load("data/banana.target.ply").vertices
gt_T = np.loadtxt("data/banana.pose.txt")

# Run ICP
icp = ICP()
T = icp(source_pcd, target_pcd)

# Visualization
rre = np.rad2deg(compute_rre(T[:3, :3], gt_T[:3, :3]))
rte = compute_rte(T[:3, 3], gt_T[:3, 3])
print(f"rre={rre}, rte={rte}")
compare_points(source_pcd @ T[:3, :3].T + T[:3, 3], target_pcd)