import os
import numpy as np
import open3d as o3d
import sys
from os.path import join
from tqdm import tqdm

kitti_base = sys.argv[1]
output_base = sys.argv[2]

seqs = sorted(os.listdir(kitti_base))

voxel_size = 0.25
point_num = 25000

for seq in seqs:
    raw_path = join(kitti_base, seq, 'velodyne')
    files = sorted(os.listdir(raw_path))
    out_path = join(output_base, seq)
    os.makedirs(out_path) if not os.path.isdir(out_path) else None
    for file in tqdm(files[:100]):
        raw_dat = np.fromfile(join(raw_path, file), dtype=np.float32).reshape((-1, 4))[:,:3]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(raw_dat))
        downpcd = pcd.voxel_down_sample(voxel_size)

        point_set = np.asarray(downpcd.points)
        if len(point_set) < point_num:
            add_index = np.random.choice(
                len(point_set), point_num - len(point_set))
            add_pts = point_set[add_index, ...]
            sample_pc = np.vstack([point_set, add_pts])
        else:
            tree = o3d.geometry.KDTreeFlann(downpcd)
            k, ind, _ = tree.search_knn_vector_3d([0, 0, 0], point_num)
            sample_pc = point_set[ind]
        
        idx = np.arange(point_num)
        np.random.shuffle(idx)
        sample_pc = sample_pc[idx]
        out_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample_pc))
        out_pcd.paint_uniform_color([1, 1, 0])
        o3d.io.write_point_cloud(join(out_path, file[:-4] + '.pcd'), out_pcd)
        

