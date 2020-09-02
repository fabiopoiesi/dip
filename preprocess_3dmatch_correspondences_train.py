import numpy as np
import open3d as o3d
import os
import h5py

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

# directory of the 3DMatch training dataset
root_dir = 'path-to-dataset-root/3DMatch_train'
root_dirs = os.listdir(root_dir)
root_dirs.sort()

# destination directory of preprocessed 3DMatch training data
dest_dir = 'path-to-dataset-root/3DMatch_train_pre'
if not os.path.isdir(dest_dir):
    os.mkdir(dest_dir)

if not os.path.isdir(os.path.join(dest_dir, 'correspondences')):
    os.mkdir(os.path.join(dest_dir, 'correspondences'))

for d in root_dirs:
    if os.path.isdir(os.path.join(root_dir, d)):

        hf_corr = h5py.File(os.path.join(dest_dir, 'correspondences', '{}.hdf5'.format(d)), 'w')

        # these are the correspondences computed by Gojcic et al. CVPR 2019
        # (we will recompute them in this code using ICP)
        corr_dir = os.path.join(root_dir, d, 'Correspondences')
        files = os.listdir(corr_dir)
        files.sort()

        print('Processing {}'.format(d))
        file_counter = 0
        for f in files:
            if f[:9] == 'cloud_bin':

                _f = f.split('_')
                pcd1_id = int(_f[2].split('-')[0])
                pcd2_id = int(_f[4])

                fpcd1 = os.path.join(root_dir, d, '01_Data', 'cloud_bin_{}.ply'.format(pcd1_id))
                fpcd2 = os.path.join(root_dir, d, '01_Data', 'cloud_bin_{}.ply'.format(pcd2_id))

                fT1 = os.path.join(root_dir, d, 'cloud_bin_{}.info.txt'.format(pcd1_id))
                fT2 = os.path.join(root_dir, d, 'cloud_bin_{}.info.txt'.format(pcd2_id))

                if not os.path.isfile(fpcd1):
                    continue
                if not os.path.isfile(fpcd2):
                    continue
                if not os.path.isfile(fT1):
                    continue
                if not os.path.isfile(fT2):
                    continue

                pcd1 = o3d.io.read_point_cloud(fpcd1)
                pcd2 = o3d.io.read_point_cloud(fpcd2)

                T1 = get_T(fT1)
                T2 = get_T(fT2)

                pcd1.transform(T1)
                pcd2.transform(T2)

                # find correspondences
                result = o3d.registration.registration_icp(pcd1, pcd2, .02, np.eye(4),
                                                           o3d.registration.TransformationEstimationPointToPoint())

                pcd1_overlap_idx = np.asarray(result.correspondence_set)[:, 0]
                pcd2_overlap_idx = np.asarray(result.correspondence_set)[:, 1]
                hf_corr.create_dataset('{},{}'.format(pcd1_id, pcd2_id),
                                  data=np.asarray(result.correspondence_set),
                                  compression='gzip')

                file_counter += 1
                print('{}/{} - {},{}'.format(file_counter, len(files), pcd1_id, pcd2_id))

        hf_corr.close()
        print(d + ' -> Done')