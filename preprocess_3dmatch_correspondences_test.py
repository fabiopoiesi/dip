import numpy as np
import open3d as o3d
import os
import h5py
import pickle

do_viz = False

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

# directory of the 3DMatch testing dataset
root_dir = 'path-to-dataset-root/3DMatch_test'
root_dirs = os.listdir(root_dir)
root_dirs.sort()

# destination directory of preprocessed 3DMatch testing data
dest_dir = 'path-to-dataset-root/3DMatch_test_pre'
if not os.path.isdir(dest_dir):
    os.mkdir(dest_dir)

for d in root_dirs:
    if os.path.isdir(os.path.join(root_dir, d)):

        hf = h5py.File(os.path.join(dest_dir, '{}.hdf5'.format(d)), 'w')

        # load ground-truth transformations
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

            T = np.empty((4, 4))
            T[0, :] = np.asarray(gt[frag_ptr + 1].split('\t')[:-1], dtype=np.float)
            T[1, :] = np.asarray(gt[frag_ptr + 2].split('\t')[:-1], dtype=np.float)
            T[2, :] = np.asarray(gt[frag_ptr + 3].split('\t')[:-1], dtype=np.float)
            T[3, :] = np.asarray(gt[frag_ptr + 4].split('\t')[:-1], dtype=np.float)

            fpcd1 = os.path.join(root_dir, d, '01_Data', 'cloud_bin_{}.ply'.format(pcd1_id))
            fpcd2 = os.path.join(root_dir, d, '01_Data', 'cloud_bin_{}.ply'.format(pcd2_id))

            pcd1 = o3d.io.read_point_cloud(fpcd1)
            pcd2 = o3d.io.read_point_cloud(fpcd2)

            pcd2.transform(T)

            if not os.path.isdir(os.path.join(root_dir, d, '02_T')):
                os.mkdir(os.path.join(root_dir, d, '02_T'))

            fT = os.path.join(root_dir, d, '02_T', '{}_{}.pkl'.format(pcd1_id, pcd2_id))
            pickle.dump(T, open(fT, 'wb'))

            result = o3d.pipelines.registration.registration_icp(pcd1, pcd2, .02, np.eye(4),
                                                       o3d.pipelines.registration.TransformationEstimationPointToPoint())

            pcd1_overlap_idx = np.asarray(result.correspondence_set)[:, 0]
            pcd2_overlap_idx = np.asarray(result.correspondence_set)[:, 1]

            pcd1_overlap_pts = np.asarray(pcd1.points)[pcd1_overlap_idx]
            pcd2_overlap_pts = np.asarray(pcd2.points)[pcd2_overlap_idx]

            hf.create_dataset('{},{}'.format(pcd1_id, pcd2_id),
                              data=np.asarray(result.correspondence_set),
                              compression='gzip')

            if do_viz:
                pcd1.estimate_normals()
                pcd1.paint_uniform_color([1, 1, 0])
                pcd2.estimate_normals()
                pcd2.paint_uniform_color([0, 1, 1])
                o3d.visualization.draw_geometries([pcd1, pcd2])

                pcd1_overlap = o3d.geometry.PointCloud()
                pcd1_overlap.points = o3d.utility.Vector3dVector(pcd1_overlap_pts)

                pcd2_overlap = o3d.geometry.PointCloud()
                pcd2_overlap.points = o3d.utility.Vector3dVector(pcd2_overlap_pts)

                pcd1_overlap.paint_uniform_color([1, 1, .2])
                pcd1_overlap.estimate_normals()
                pcd2_overlap.paint_uniform_color([0, 1, 1])
                pcd2_overlap.estimate_normals()
                o3d.visualization.draw_geometries([pcd1_overlap, pcd2_overlap])

            frag_counter += 1
            print('{}/{} - {},{}'.format(frag_counter, nfrag, pcd1_id, pcd2_id))

        hf.close()
        print(d + ' -> Done')
