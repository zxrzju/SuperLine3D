
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from os.path import join
import os
from tqdm import tqdm
from multiprocessing import Pool
import random
import Geometry3D as g3d
from liegroups.numpy import SO3
import sys

def get_labels(pcd, kps, k):
    out_dat = np.zeros((len(pcd.points), 4))
    out_dat[:, :3] = np.asarray(pcd.points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for i in range(kps.shape[0]):
        # k, idx, _ = pcd_tree.search_hybrid_vector_3d(kps[i, :], 0.2, 10)
        # k, idx, _ = pcd_tree.search_radius_vector_3d(kps[i, :], 0.05)
        k, idx, _ = pcd_tree.search_knn_vector_3d(kps[i, :], k)
        out_dat[idx, 3] = 1

    return out_dat

def gen_one_pole(points_num, label_line=False):
    np.random.seed(int(random.random()*100))
    all_np = []
    pts = []
    line_idx_all = []

    length = np.random.rand(1)*0.8 + 0.2
    length = length[0]
    # print(length)

    points = np.array([[0, 0, 0], [0, 0.03, 0], [0, 0, length], [0, 0.03, length],
                       [0, 0, 0], [0, 1, 0], [1, 0, 0], [-0.8, -0.8, 0],
                       [0, 0, 0], [0.03, 0, 0], [0, 0, length], [0.03, 0, length]]).astype(np.float)

    faces = np.array(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]).astype(np.int)
    triangles = np.zeros((12, 3)).astype(np.int)
    triangles[:4, :] = faces
    triangles[4:8, :] = faces + 4
    triangles[8:, :] = faces + 8
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.8, 0.8, 0.8])
    mesh.paint_uniform_color([0, 0.651, 0.929])

    pts_g3d = []
    for i in [0, 1, 2, 3, 8, 9, 10, 11]:
        pts_g3d.append(g3d.Point(points[i]))

    poly_g3d = []
    for i in range(len(pts_g3d)//4):
        poly_g3d.append(g3d.ConvexPolygon(
            [pts_g3d[4*i], pts_g3d[4*i+1], pts_g3d[4*i+2], pts_g3d[4*i+3]]))

    inter_poly = []
    for i in range(len(poly_g3d)-1):
        for k in range(i+1, len(poly_g3d)):
            inter_poly.append(g3d.intersection(poly_g3d[i], poly_g3d[k]))

    inter_line = []
    for i in range(len(inter_poly)-1):
        for k in range(i+1, len(inter_poly)):
            inter_line.append(g3d.intersection(inter_poly[i], inter_poly[k]))

    inter_pts = []
    for i in range(len(inter_poly)):
        inter_pts.append([inter_poly[i][0][0], inter_poly[i]
                         [0][1], inter_poly[i][0][2]])
        inter_pts.append([inter_poly[i][1][0], inter_poly[i]
                         [1][1], inter_poly[i][1][2]])

    pcds = mesh.sample_points_uniformly(int(points_num*10))
    pcds = pcds.voxel_down_sample(0.04)

    pcd_pts = np.array(pcds.points)
    pcd_all = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_pts.copy()))

    line_noise = []
    line_idx = []

    if label_line:
        inte = np.linspace(0, points[1, 1], 51)
        line0 = np.zeros((51, 3))
        line0[:, 0] = inte.copy()
        line1 = np.zeros((51, 3))
        line1[:, 1] = inte.copy()
        line2 = np.zeros((51, 3))
        inte = np.linspace(0, length, 51).reshape(-1)
        line2[:, 2] = inte.copy()
        line_all = np.vstack((line0, line1, line2))
        line_all = np.expand_dims(line_all, 1)
        pts_all = np.expand_dims(pcd_pts, 0)
        dist = np.sum(np.abs(line_all - pts_all), -1)

        line_idx = np.argmin(dist, 1)
        line_pts = pcd_pts[line_idx, :]

        line_noise = get_labels(pcd_all, line_pts[-48:], 5)

    pts_labels = line_noise

    pcd_all.paint_uniform_color([0, 0.651, 0.929])
    colors_all = np.asarray(pcd_all.colors)
    colors_all[np.where(pts_labels[:, -1] > 0)[0]] = [1, 0, 0]
    # o3d.visualization.draw_geometries([pcd_all])
    # o3d.io.write_point_cloud('data/test_face_pole.pcd', pcd_all)

    shuffle_idx = np.arange(pts_labels.shape[0])
    np.random.shuffle(shuffle_idx)
    pts_labels = pts_labels[shuffle_idx]
    pts_labels[:,:3] += (np.random.rand(pts_labels.shape[0], 3)-0.5)*0.01
    return pts_labels

def gen_one_triface(points_num, label_line=False):

    np.random.seed(int(random.random()*100))

    points = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1],
                       [0, 0, 0], [0, 1, 0], [1, 0, 0], [-0.8, -0.8, 0],
                       [0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]]).astype(np.float)

    noise = np.random.rand(3, 2)*0.2-0.1

    points[3, 1:] += noise[0, :]
    points[7, :2] += noise[1, :]
    points[11, [0, 2]] += noise[2, :]

    faces = np.array(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]).astype(np.int)
    triangles = np.zeros((12, 3)).astype(np.int)
    triangles[:4, :] = faces
    triangles[4:8, :] = faces + 4
    triangles[6:8, :] = triangles[4:6, :]
    triangles[8:, :] = faces + 8
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    # mesh.paint_uniform_color([0.8, 0.8, 0.8])
    mesh.paint_uniform_color([0, 0.651, 0.929])
    # o3d.io.write_triangle_mesh('data/test_face.ply', mesh)
    pts_g3d = []
    for i in range(len(points)):
        pts_g3d.append(g3d.Point(points[i]))

    poly_g3d = []
    for i in range(len(points)//4):
        poly_g3d.append(g3d.ConvexPolygon(
            [pts_g3d[4*i], pts_g3d[4*i+1], pts_g3d[4*i+2], pts_g3d[4*i+3]]))

    inter_poly = []
    for i in range(len(poly_g3d)-1):
        for k in range(i+1, len(poly_g3d)):
            inter_poly.append(g3d.intersection(poly_g3d[i], poly_g3d[k]))

    inter_line = []
    for i in range(len(inter_poly)-1):
        for k in range(i+1, len(inter_poly)):
            inter_line.append(g3d.intersection(inter_poly[i], inter_poly[k]))

    inter_pts = []
    for i in range(len(inter_poly)):
        inter_pts.append([inter_poly[i][0][0], inter_poly[i]
                         [0][1], inter_poly[i][0][2]])
        inter_pts.append([inter_poly[i][1][0], inter_poly[i]
                         [1][1], inter_poly[i][1][2]])

    inter_pts.append([inter_line[0][0], inter_line[0][1], inter_line[0][2]])

    # o3d.visualization.draw_geometries([mesh])
    pcds = mesh.sample_points_uniformly(int(points_num*10))
    # pcds = mesh.sample_points_uniformly(points_num*10)
    pcds = pcds.voxel_down_sample(0.052)
    pcd_pts = np.array(pcds.points)
    # print(len(pcd_pts))


    if len(pcd_pts) > points_num:
        # idx = np.random.choice(len(pcd_pts), points_num)
        pcd_pts = pcd_pts[:points_num, :]

    noise_all = (np.random.rand(pcd_pts.shape[0], 3)-0.5)*(0.01)
    pcd_pts += noise_all
    pcd_all = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_pts.copy()))

    line_noise = []
    line_idx = []
    if label_line:
        inte = np.linspace(0, 1, 51)
        line0 = np.zeros((51, 3))
        line0[:, 0] = inte.copy()
        line1 = np.zeros((51, 3))
        line1[:, 1] = inte.copy()
        line2 = np.zeros((51, 3))
        line2[:, 2] = inte.copy()
        line_all = np.vstack((line0, line1, line2))
        line_all = np.expand_dims(line_all, 1)
        pts_all = np.expand_dims(pcd_pts, 0)
        dist = np.sum(np.abs(line_all - pts_all), -1)

        line_idx = np.argmin(dist, 1)
        line_pts = pcd_pts[line_idx, :]

        line_noise = get_labels(pcd_all, line_pts, 1)

    pts_labels = line_noise

    pcd_all.paint_uniform_color([0, 0.651, 0.929])
    colors_all = np.asarray(pcd_all.colors)
    colors_all[np.where(pts_labels[:,-1]>0)[0]] = [1, 0, 0]
    # o3d.visualization.draw_geometries([pcd_all])

    # o3d.io.write_point_cloud('data/test_face_pc.pcd', pcd_all)
    shuffle_idx = np.arange(pts_labels.shape[0])
    np.random.shuffle(shuffle_idx)
    pts_labels = pts_labels[shuffle_idx]

    return pts_labels

def gen_one_lpn(train_all, train_pcd, noise_path, noise_files, points_num, j):
    file_name = '%04d' % j
    trans = np.array([[1.3,1.3], [1.3, -1.3], [-1.3, 1.3], [-1.3, -1.3], [4., 4.]])
    np.random.seed(int(random.random()*100))

    list_pts = []
    
    list_pts.append(gen_one_triface(points_num//5, True))
    list_pts.append(gen_one_triface(points_num//5, True))
    pts_all = gen_one_pole(points_num//5, True)
    pts_all[:, -1] *= 2
    list_pts.append(pts_all)

    lp = int(np.random.rand(1)*100)%2
    if lp == 1:
        list_pts.append(gen_one_triface(points_num//5, True))
    else:
        pts_all = gen_one_pole(points_num//5, True)
        pts_all[:, -1] *= 2
        list_pts.append(pts_all)
    
    noise_idx = j%len(noise_files)

    noise_pcd = o3d.io.read_point_cloud(noise_path + noise_files[noise_idx])
    noise_pcd.paint_uniform_color([0.8, 0.8, 0.8])
    noise_pcd.paint_uniform_color([1, 0.706, 0])

    # o3d.visualization.draw_geometries([noise_pcd])
    noise = np.asarray(noise_pcd.points)
    # print(noise.shape)
    noise_all = np.hstack([noise, np.zeros((noise.shape[0], 1))])
    
    list_pts.append(noise_all)
    all_np = []

    pcd_all = o3d.geometry.PointCloud()
    for i in range(len(list_pts)):

        transform = np.identity(4)
        transform[:2, 3] = trans[i]
        transform[:3, :3] = SO3.from_rpy(0, 0, np.random.rand(1)*2*np.pi).as_matrix()

        pts_labels = list_pts[i]
        pts_h = np.zeros(pts_labels.shape)
        pts_h[:,3] = 1
        pts_h[:,:3] = pts_labels[:,:3].copy()
        # print(rot_angle)
        pts_h = np.matmul(pts_h, transform.T)
        pts_labels[:, :3] = pts_h[:,:3]

        cur_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_h[:, :3]))
        if i < 4:
            cur_pcd.paint_uniform_color([0, 0.651, 0.929])
        else:
            cur_pcd.paint_uniform_color([1, 0.706, 0])

        pcd_all += cur_pcd
        if len(all_np) == 0:
            all_np = pts_labels.copy()
        else:
            all_np = np.vstack((all_np, pts_labels))

    colors = np.asarray(pcd_all.colors)
    colors[np.where(all_np[:, 3] > 0)[0]] = [1, 0, 0]
    
    all_idx = np.arange(len(all_np))
    if len(all_np) > points_num:
        reserved_idx = all_idx[:points_num]
   
    else:
        choice_id = np.random.choice(len(all_np), points_num-len(all_np))
        reserved_idx = np.concatenate((all_idx, choice_id))

    
    all_np = all_np[reserved_idx]
    all_np[:, :3] += (np.random.rand(points_num, 3)-0.5)*0.005

    shuffle_idx = np.arange(len(all_np))
    np.random.shuffle(shuffle_idx)
    all_np = all_np[shuffle_idx]

    transform = np.identity(4)
    transform[:3, :3] = SO3.from_rpy(0, 0, np.random.rand(1)*2*np.pi).as_matrix()

    pts_h = np.zeros(all_np.shape)
    pts_h[:, 3] = 1
    pts_h[:, :3] = all_np[:, :3]
    pts_h = np.matmul(pts_h, transform.T)
    all_np[:, :3] = pts_h[:, :3]

    all_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_np[:, :3]))
    all_pcd.colors = o3d.utility.Vector3dVector(colors[reserved_idx][shuffle_idx])


    # o3d.visualization.draw_geometries([all_pcd])

    o3d.io.write_point_cloud(join(train_pcd, file_name+'.pcd'), all_pcd)
    np.save(join(train_all, file_name+'.npy'), all_np)


base_path = '/home/miyun/dataset/lpn_2_class_5k/'

base_path = sys.argv[1]
noise_path = './kitti_noise/'
noise_files = os.listdir(noise_path)

class_name = 'line_pole_noise'
pts_num = 5000

if 1:
    train = join(base_path, class_name, 'train')
    test = join(base_path, class_name, 'test')
    train_pcd = join(train, 'pcd')
    train_all = join(train, 'npy')
    test_pcd = join(test, 'pcd')
    test_all = join(test, 'npy')
    if not os.path.isdir(train_all):
        os.makedirs(train_pcd)
        os.makedirs(train_all)
        os.makedirs(test_pcd)
        os.makedirs(test_all)

    for ii in range(10):
        gen_one_lpn(train_all, train_pcd, noise_path, noise_files, pts_num, ii)
    
    train_num = 100
    test_num = 20


    pool = Pool(processes=4, maxtasksperchild=50)
    for ii in range(train_num):
        pool.apply_async(gen_one_lpn, (train_all, train_pcd, noise_path, noise_files, 5000, ii))
    pool.close()
    pool.join()

    # pool = Pool(processes=4, maxtasksperchild=50)

    # for ii in range(test_num):
    #     pool.apply_async(gen_one_lpn, (test_all, test_pcd, noise_path, noise_files, 5000, ii))
    # pool.close()
    # pool.join()
