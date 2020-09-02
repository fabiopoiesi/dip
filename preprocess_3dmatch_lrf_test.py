import numpy as np
import open3d as o3d
import os
import h5py
from lrf import lrf
from scipy.spatial.transform import Rotation as rotation
import random

# this is to create 3DMatchRotated
do_rotated = False

# directory of the 3DMatch testing dataset
root_dir = 'path-to-dataset-root/3DMatch_test'

# destination directory of preprocessed 3DMatch testing data
dest_dir = 'path-to-dataset-root/3DMatch_test_pre'
if not os.path.isdir(dest_dir):
    os.mkdir(dest_dir)

# 5000 points sampled from the point clouds as in Gojcic et al. CVPR 2019
pts_to_sample = 5000
patch_size = 256
voxel_size = .01
lrf_kernel = .3 * np.sqrt(3)

root_dirs = ['7-scenes-redkitchen',
'sun3d-home_at-home_at_scan1_2013_jan_1',
'sun3d-home_md-home_md_scan9_2012_sep_30',
'sun3d-hotel_uc-scan3',
'sun3d-hotel_umd-maryland_hotel1',
'sun3d-hotel_umd-maryland_hotel3',
'sun3d-mit_76_studyroom-76-1studyroom2',
'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika']

print('*****************************')
print('datasets to preprocess:')
for i, d in enumerate(root_dirs):
    gt_file = open(os.path.join(root_dir, d, '01_Data', 'gt.log'), 'r')
    gt = gt_file.readlines()
    nfrag = int(len(gt)/5)
    print('{}: {}'.format(d, nfrag))
print('*****************************')
print('*****************************')

for d in root_dirs:

    if do_rotated:
        if not os.path.isdir(os.path.join(dest_dir, 'patches_lrf_rot')):
            os.mkdir(os.path.join(dest_dir, 'patches_lrf_rot'))
        if not os.path.isdir(os.path.join(dest_dir, 'points_lrf_rot')):
            os.mkdir(os.path.join(dest_dir, 'points_lrf_rot'))
        if not os.path.isdir(os.path.join(dest_dir, 'lrfs_rot')):
            os.mkdir(os.path.join(dest_dir, 'lrfs_rot'))
        if not os.path.isdir(os.path.join(dest_dir, 'rotations_lrf_rot')):
            os.mkdir(os.path.join(dest_dir, 'rotations_lrf_rot'))

        hf_patches = h5py.File(os.path.join(dest_dir, 'patches_lrf_rot', '{}.hdf5'.format(d)), 'w')
        hf_points = h5py.File(os.path.join(dest_dir, 'points_lrf_rot', '{}.hdf5'.format(d)), 'w')
        hf_lrfs = h5py.File(os.path.join(dest_dir, 'lrfs_rot', '{}.hdf5'.format(d)), 'w')
        hf_rotations = h5py.File(os.path.join(dest_dir, 'rotations_lrf_rot', '{}.hdf5'.format(d)), 'w')

    else:
        if not os.path.isdir(os.path.join(dest_dir, 'patches_lrf')):
            os.mkdir(os.path.join(dest_dir, 'patches_lrf'))
        if not os.path.isdir(os.path.join(dest_dir, 'points_lrf')):
            os.mkdir(os.path.join(dest_dir, 'points_lrf'))
        if not os.path.isdir(os.path.join(dest_dir, 'lrfs')):
            os.mkdir(os.path.join(dest_dir, 'lrfs'))

        hf_patches = h5py.File(os.path.join(dest_dir, 'patches_lrf', '{}.hdf5'.format(d)), 'w')
        hf_points = h5py.File(os.path.join(dest_dir, 'points_lrf', '{}.hdf5'.format(d)), 'w')
        hf_lrfs = h5py.File(os.path.join(dest_dir, 'lrfs', '{}.hdf5'.format(d)), 'w')

    gt_file = open(os.path.join(root_dir, d, '01_Data', 'gt.log'), 'r')
    gt = gt_file.readlines()

    nfrag = int(len(gt)/5)

    print('Processing: {}'.format(d))

    frag_counter = 0

    for frag in range(nfrag):

        frag_ptr = frag * 5
        info = gt[frag_ptr].split('\t')
        pcd1_id = int(info[0])
        pcd2_id = int(info[1])

        fpcd1 = os.path.join(root_dir, d, '01_Data', 'cloud_bin_{}.ply'.format(pcd1_id))
        fpcd2 = os.path.join(root_dir, d, '01_Data', 'cloud_bin_{}.ply'.format(pcd2_id))

        pcd1 = o3d.io.read_point_cloud(fpcd1)
        pcd2 = o3d.io.read_point_cloud(fpcd2)

        pcd1 = pcd1.voxel_down_sample(voxel_size)
        pcd2 = pcd2.voxel_down_sample(voxel_size)

        if do_rotated:
            rot_int = 2 * np.pi
            R1 = rotation.from_euler('zyx', [random.uniform(0, rot_int),
                                             random.uniform(0, rot_int),
                                             random.uniform(0, rot_int)]).as_matrix()

            R2 = rotation.from_euler('zyx', [random.uniform(0, rot_int),
                                             random.uniform(0, rot_int),
                                             random.uniform(0, rot_int)]).as_matrix()

            np.asarray(pcd1.points)[:] = np.dot(R1.T, np.asarray(pcd1.points).T).T
            np.asarray(pcd2.points)[:] = np.dot(R2.T, np.asarray(pcd2.points).T).T

            hf_rotations.create_dataset('{},{}'.format(pcd1_id, pcd2_id),
                                        data=np.asarray([R1, R2]),
                                        compression='gzip')

        # pick 5000 random points from the two point clouds
        inds1 = np.random.choice(np.asarray(pcd1.points).shape[0], pts_to_sample, replace=False)
        inds2 = np.random.choice(np.asarray(pcd2.points).shape[0], pts_to_sample, replace=False)

        pcd1_pts = np.asarray(pcd1.points)[inds1]
        pcd2_pts = np.asarray(pcd2.points)[inds2]

        # COMPUTE PATCHES
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

        patches1_batch = np.empty((pcd1_pts.shape[0], 3, patch_size))
        patches2_batch = np.empty((pcd2_pts.shape[0], 3, patch_size))

        lrfs1_batch = np.empty((pcd1_pts.shape[0], 4, 4))
        lrfs2_batch = np.empty((pcd2_pts.shape[0], 4, 4))

        for i in range(pcd1_pts.shape[0]):

            frag1_lrf_pts, _, lrf1 = frag1_lrf.get(pcd1_pts[i])
            frag2_lrf_pts, _, lrf2 = frag2_lrf.get(pcd2_pts[i])

            patches1_batch[i] = frag1_lrf_pts.T
            patches2_batch[i] = frag2_lrf_pts.T

            lrfs1_batch[i] = lrf1
            lrfs2_batch[i] = lrf2

        hf_patches.create_dataset('{},{}'.format(pcd1_id, pcd2_id),
                                  data=np.asarray([patches1_batch, patches2_batch]),
                                  compression='gzip')

        hf_points.create_dataset('{},{}'.format(pcd1_id, pcd2_id),
                                 data=np.asarray([pcd1_pts, pcd2_pts]),
                                 compression='gzip')

        hf_lrfs.create_dataset('{},{}'.format(pcd1_id, pcd2_id),
                               data=np.asarray([lrfs1_batch, lrfs2_batch]),
                               compression='gzip')

        frag_counter += 1
        print('{}/{} - {},{}'.format(frag_counter, nfrag, pcd1_id, pcd2_id))

    hf_patches.close()
    hf_points.close()
    hf_lrfs.close()
    if do_rotated:
        hf_rotations.close()

    print(d + ' -> Done')