import numpy as np
import open3d as o3d
import os
import h5py
from torch_cluster import fps
import torch
from lrf import lrf

'''
Notes:
1. LRF and patch preprocessing assumes that correspondences are given or preprocessed
2. to preprocess correspondences use preprocess_3dmatch_correspondences_train.py
'''

# directory of the 3DMatch training dataset
root_dir = 'path-to-dataset-root/3DMatch_train'

# destination directory of preprocessed 3DMatch training data
dest_dir = 'path-to-dataset-root/3DMatch_train_pre'
if not os.path.isdir(dest_dir):
    os.mkdir(dest_dir)

def get_T(file):
    with open(file) as f:
        lines = f.readlines()
    T = np.empty((4, 4), dtype=float)
    j = 0
    for i, l in enumerate(lines):
        if i < 1:
            continue
        T[j, :] = np.fromstring(l, dtype=float, sep='\t')
        j += 1
    return T

do_save = False

# training and testing parameters for 3DMatch
batch_size = 256
patch_size = 256
lrf_kernel = .3 * np.sqrt(3)
voxel_size = .01

root_dirs = os.listdir(root_dir)
root_dirs.sort()

# to specify which sequences to preprocess
# root_dirs = ['7-scenes-chess', '7-scenes-fire']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

for d in root_dirs:
    if os.path.isdir(os.path.join(root_dir, d)):

        # load preprocessed correspondences, i.e. set of indices of the corresponding 3D points between two point clouds
        hf_corrs = h5py.File(os.path.join(dest_dir, 'correspondences', '{}.hdf5'.format(d)), 'r')
        corrs_to_test = np.asarray(list(hf_corrs.keys()))

        if do_save:
            if not os.path.isdir(os.path.join(dest_dir, 'patches_lrf')):
                os.mkdir(os.path.join(dest_dir, 'patches_lrf'))
            if not os.path.isdir(os.path.join(dest_dir, 'points_lrf')):
                os.mkdir(os.path.join(dest_dir, 'points_lrf'))
            if not os.path.isdir(os.path.join(dest_dir, 'rotations_lrf')):
                os.mkdir(os.path.join(dest_dir, 'rotations_lrf'))
            if not os.path.isdir(os.path.join(dest_dir, 'lrfs')):
                os.mkdir(os.path.join(dest_dir, 'lrfs'))

            hf_patches = h5py.File(os.path.join(dest_dir, 'patches_lrf', '{}.hdf5'.format(d)), 'w')
            hf_points = h5py.File(os.path.join(dest_dir, 'points_lrf', '{}.hdf5'.format(d)), 'w')
            hf_rotations = h5py.File(os.path.join(dest_dir, 'rotations_lrf', '{}.hdf5'.format(d)), 'w')
            hf_lrfs = h5py.File(os.path.join(dest_dir, 'lrfs', '{}.hdf5'.format(d)), 'w')

        for j in range(len(corrs_to_test)):

            pcd1_id = int(corrs_to_test[j].split(',')[0])
            pcd2_id = int(corrs_to_test[j].split(',')[1])

            fpcd1 = os.path.join(root_dir, d, '01_Data', 'cloud_bin_{}.ply'.format(pcd1_id))
            fpcd2 = os.path.join(root_dir, d, '01_Data', 'cloud_bin_{}.ply'.format(pcd2_id))

            fT1 = os.path.join(root_dir, d, 'cloud_bin_{}.info.txt'.format(pcd1_id))
            fT2 = os.path.join(root_dir, d, 'cloud_bin_{}.info.txt'.format(pcd2_id))

            pcd1 = o3d.io.read_point_cloud(fpcd1)
            pcd2 = o3d.io.read_point_cloud(fpcd2)

            T1 = get_T(fT1)
            T2 = get_T(fT2)

            corrs = np.asarray(hf_corrs['{},{}'.format(pcd1_id, pcd2_id)])

            # select only corresponding points
            pcd1_corr = pcd1.select_by_index(corrs[:, 0])
            pcd2_corr = pcd2.select_by_index(corrs[:, 1])

            pcd1 = pcd1.voxel_down_sample(voxel_size)
            pcd2 = pcd2.voxel_down_sample(voxel_size)

            # apply ground truth transformation to bring them in the same reference frame
            pcd1_corr.transform(T1)
            pcd2_corr.transform(T2)

            # FPS
            tensor_pcd1_frag = torch.Tensor(np.asarray(pcd1_corr.points)).to(device)
            fps_pcd1_idx = fps(tensor_pcd1_frag,
                               ratio=batch_size / tensor_pcd1_frag.shape[0],
                               random_start=True)

            _pcd2_frag_tree = o3d.geometry.KDTreeFlann(pcd2_corr)

            fps_pcd1_pts = np.asarray(pcd1_corr.points)[fps_pcd1_idx.cpu()]

            fps_pcd2_idx = torch.empty(fps_pcd1_idx.shape, dtype=int)

            # find nearest neighbors on the other point cloud
            for i, pt in enumerate(fps_pcd1_pts):
                _, patch_idx, _ = _pcd2_frag_tree.search_knn_vector_xd(pt, 1)
                fps_pcd2_idx[i] = patch_idx[0]

            # visualise point clouds with FPS + NN result overlaid
            # to_viz = []
            # pcd1_corr.estimate_normals()
            # pcd1_corr.paint_uniform_color([1, 1, 0])
            # pcd2_corr.estimate_normals()
            # pcd2_corr.paint_uniform_color([0, 1, 1])
            # to_viz.append(pcd1_corr)
            # to_viz.append(pcd2_corr)
            # for c in fps_pcd1_idx:
            #     to_viz.append(o3d.geometry.TriangleMesh.create_sphere(radius=.025))
            #     to_viz[-1].compute_vertex_normals()
            #     to_viz[-1].paint_uniform_color([1, 0, 0])
            #     to_viz[-1].translate(np.asarray(pcd1_corr.points)[c])
            # o3d.visualization.draw_geometries(to_viz)

            # transform point clouds back to their reference frame using ground truth
            # (this is important because the network must learn when point clouds are in their original reference frame)
            pcd1_corr.transform(np.linalg.inv(T1))
            pcd2_corr.transform(np.linalg.inv(T2))

            # extract patches and compute LRFs
            patches1_batch = np.empty((batch_size, 3, patch_size))
            patches2_batch = np.empty((batch_size, 3, patch_size))

            lrfs1_batch = np.empty((batch_size, 4, 4))
            lrfs2_batch = np.empty((batch_size, 4, 4))

            frag1_lrf = lrf(pcd=pcd1,
                            pcd_tree=o3d.geometry.KDTreeFlann(pcd1),
                            patch_size=patch_size,
                            lrf_kernel=lrf_kernel,
                            viz=False)

            frag2_lrf = lrf(pcd=pcd2,
                            pcd_tree=o3d.geometry.KDTreeFlann(pcd2),
                            patch_size=patch_size,
                            lrf_kernel=lrf_kernel,
                            viz=False)

            # GET PATCHES USING THE FPS POINTS
            for i in range(len(fps_pcd1_idx)):

                pt1 = np.asarray(pcd1_corr.points)[fps_pcd1_idx.cpu()[i]]
                pt2 = np.asarray(pcd2_corr.points)[fps_pcd2_idx.cpu()[i]]

                frag1_lrf_pts, _, lrf1 = frag1_lrf.get(pt1)
                frag2_lrf_pts, _, lrf2 = frag2_lrf.get(pt2)

                patches1_batch[i] = frag1_lrf_pts.T
                patches2_batch[i] = frag2_lrf_pts.T

                lrfs1_batch[i] = lrf1
                lrfs2_batch[i] = lrf2

            if do_save:
                hf_lrfs.create_dataset('{},{}'.format(pcd1_id, pcd2_id),
                                            data=np.asarray([lrfs1_batch, lrfs2_batch]),
                                            compression='gzip')

                hf_patches.create_dataset('{},{}'.format(pcd1_id, pcd2_id),
                                          data=np.asarray([patches1_batch, patches2_batch]),
                                          compression='gzip')

                hf_points.create_dataset('{},{}'.format(pcd1_id, pcd2_id),
                                         data=np.asarray([np.asarray(pcd1_corr.points)[fps_pcd1_idx.cpu()],
                                                          np.asarray(pcd2_corr.points)[fps_pcd2_idx.cpu()]]),
                                         compression='gzip')

                hf_rotations.create_dataset('{},{}'.format(pcd1_id, pcd2_id),
                                            data=np.asarray([T1[:3, :3], T2[:3, :3]]),
                                            compression='gzip')

            # visualise patch points before and after applying LRF

            # p1 = patches1_batch[0]
            # p2 = patches2_batch[0]
            #
            # p1_rot = np.dot(T1[:3, :3].T, p1)
            # p2_rot = np.dot(T2[:3, :3].T, p2)
            #
            # # before applying LRF
            # pcd1 = o3d.geometry.PointCloud()
            # pcd1.points = o3d.utility.Vector3dVector(p1_rot.T)
            # pcd1.paint_uniform_color([1, 0, 0])
            # pcd2 = o3d.geometry.PointCloud()
            # pcd2.points = o3d.utility.Vector3dVector(p2_rot.T)
            # pcd2.paint_uniform_color([0, 0, 1])
            # o3d.visualization.draw_geometries([pcd1, pcd2])
            #
            # # after applying LRF
            # pcd1 = o3d.geometry.PointCloud()
            # pcd1.points = o3d.utility.Vector3dVector(p1.T)
            # pcd1.paint_uniform_color([1, 0, 0])
            # pcd2 = o3d.geometry.PointCloud()
            # pcd2.points = o3d.utility.Vector3dVector(p2.T)
            # pcd2.paint_uniform_color([0, 0, 1])
            # o3d.visualization.draw_geometries([pcd1, pcd2])

            print('{}/{} - {},{}'.format(j, len(corrs_to_test), pcd1_id, pcd2_id))

        if do_save:
            hf_patches.close()
            hf_points.close()
            hf_rotations.close()
            hf_lrfs.close()

        print(d + ' -> Done')



