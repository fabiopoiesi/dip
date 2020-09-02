from scipy.spatial import cKDTree
import numpy as np
import open3d as o3d
import torch
from network import PointNetFeature
import h5py
import os
import pickle

do_rotated3DMatch = False

# directory of the checkpoint/model to test
model_root = './model'

dataset_root = 'path-to-dataset-root/3DMatch_test'

root_dirs = ['7-scenes-redkitchen',
             'sun3d-home_at-home_at_scan1_2013_jan_1',
             'sun3d-home_md-home_md_scan9_2012_sep_30',
             'sun3d-hotel_uc-scan3',
             'sun3d-hotel_umd-maryland_hotel1',
             'sun3d-hotel_umd-maryland_hotel3',
             'sun3d-mit_76_studyroom-76-1studyroom2',
             'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika']

# feature-matching recall parameters
tau_1 = .1
tau_2 = .05

pts_to_sample = 5000
batch_size = 1000
patch_size = 256
voxel_size = .01
lrf_kernel = .3 * np.sqrt(3)
dim = 32
perc = 5

rotated_label = ''
if do_rotated3DMatch:
    rotated_label = '_rotated'

l2norm = True # activate/deactivate LRN (if training is done with l2norm=True, here it must be True as well)
tnet = True # activate/deactivate TNet (if training is done with l2norm=True, here it must be True as well)

ckpt_name = 'final_chkpt.pth'

net = PointNetFeature(dim=dim, l2norm=True, tnet=True)

checkpoint = '{}/{}'.format(model_root, ckpt_name)
net.load_state_dict(torch.load(checkpoint))
net.cuda()
net.eval()

recall_tau2 = []
RECALL_tau1 = []
RECALL_tau2 = []

ths_tau1 = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
RECALL_tau1_ths = []

ths_tau2 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21]
RECALL_tau2_ths = []

print('*****************************')
print('testing:' + checkpoint)
print('*****************************')
print('datasets to test:')
for i, d in enumerate(root_dirs):
    hf_patches = h5py.File(os.path.join(dataset_root + '_pre', 'patches_lrf', '{}.hdf5'.format(d)), 'r')
    corrs = np.asarray(list(hf_patches.keys()))
    print('{}. {} -> {} point-cloud pairs'.format(i + 1, d, len(corrs)))

print('*****************************')
print('start:')

for d in root_dirs:

    if do_rotated3DMatch:
        # this requires preprocessing of patches and lrfs
        hf_patches = h5py.File(os.path.join(dataset_root + '_pre', 'patches_lrf_rot', '{}.hdf5'.format(d)), 'r')
        hf_points = h5py.File(os.path.join(dataset_root + '_pre', 'points_lrf_rot', '{}.hdf5'.format(d)), 'r')
        hf_lrfs = h5py.File(os.path.join(dataset_root + '_pre', 'lrfs_rot', '{}.hdf5'.format(d)), 'r')
        hf_rotations = h5py.File(os.path.join(dataset_root + '_pre', 'rotations_lrf_rot', '{}.hdf5'.format(d)), 'r')
    else:
        hf_patches = h5py.File(os.path.join(dataset_root + '_pre', 'patches_lrf', '{}.hdf5'.format(d)), 'r')
        hf_points = h5py.File(os.path.join(dataset_root + '_pre', 'points_lrf', '{}.hdf5'.format(d)), 'r')
        hf_lrfs = h5py.File(os.path.join(dataset_root + '_pre', 'lrfs', '{}.hdf5'.format(d)), 'r')

    corrs = np.asarray(list(hf_patches.keys()))

    recall_tau1 = []

    for j in range(len(corrs)):

        patches = np.asarray(hf_patches[corrs[j]])
        patches1 = patches[0]
        patches2 = patches[1]

        if do_rotated3DMatch:
            rotations = np.asarray(hf_rotations[corrs[j]])
            R1 = rotations[0]
            R2 = rotations[1]

        points = np.asarray(hf_points[corrs[j]])
        pts1 = points[0]
        pts2 = points[1]

        lrfs = np.asarray(hf_lrfs[corrs[j]])
        lrf1 = lrfs[0]
        lrf2 = lrfs[1]

        if do_rotated3DMatch:
            pts1 = np.dot(R1, pts1.transpose((1, 0))).transpose((1, 0))
            pts2 = np.dot(R2, pts2.transpose((1, 0))).transpose((1, 0))

        # COMPUTE DESCRIPTOR
        pcd1_desc = np.empty((patches1.shape[0], dim))
        pcd2_desc = np.empty((patches2.shape[0], dim))

        pcd1_mx = np.empty((patches1.shape[0], 1024))

        pcd2_mx = np.empty((patches2.shape[0], 1024))

        pcd1_amx = np.empty((patches1.shape[0], 1024), dtype=int)
        pcd2_amx = np.empty((patches2.shape[0], 1024), dtype=int)

        for b in range(int(np.ceil(patches1.shape[0] / batch_size))):
            i_start = b * batch_size
            i_end = (b + 1) * batch_size
            if i_end > pts_to_sample:
                i_end = pts_to_sample

            pcd1_batch = torch.Tensor(patches1[i_start:i_end]).cuda()

            with torch.no_grad():
                f, mx1, amx1 = net(pcd1_batch)
            pcd1_desc[i_start:i_end] = f.cpu().detach().numpy()[:i_end - i_start]
            pcd1_mx[i_start:i_end] = mx1.cpu().detach().numpy().squeeze()[:i_end - i_start]
            pcd1_amx[i_start:i_end] = amx1.cpu().detach().numpy().squeeze()[:i_end - i_start]

            pcd2_batch = torch.Tensor(patches2[i_start:i_end]).cuda()

            with torch.no_grad():
                f, mx2, amx2 = net(pcd2_batch)
            pcd2_desc[i_start:i_end] = f.cpu().detach().numpy()[:i_end - i_start]
            pcd2_mx[i_start:i_end] = mx2.cpu().detach().numpy().squeeze()[:i_end - i_start]
            pcd2_amx[i_start:i_end] = amx2.cpu().detach().numpy().squeeze()[:i_end - i_start]

        mag_pcd1_mx = np.linalg.norm(pcd1_mx, axis=1)
        mag_pcd2_mx = np.linalg.norm(pcd2_mx, axis=1)

        perc_th = np.min([np.percentile(mag_pcd1_mx, perc), np.percentile(mag_pcd1_mx, perc)])

        good_pcd1_desc = mag_pcd1_mx > perc_th
        good_pcd2_desc = mag_pcd2_mx > perc_th

        pcd1_desc = pcd1_desc[good_pcd1_desc]
        pcd2_desc = pcd2_desc[good_pcd2_desc]

        # find nearest neighbours
        pcd2_desc_tree = cKDTree(pcd2_desc)
        _, nn2_inds = pcd2_desc_tree.query(pcd1_desc)

        pcd1_desc_tree = cKDTree(pcd1_desc)
        _, nn1_inds = pcd1_desc_tree.query(pcd2_desc)

        mutual_nn = list(range(pcd1_desc.shape[0])) == nn1_inds[nn2_inds]

        # apply ground-truth transformation to points
        pcd_pts1 = o3d.geometry.PointCloud()
        pcd_pts1.points = o3d.utility.Vector3dVector(pts1[good_pcd1_desc])

        pcd_pts2 = o3d.geometry.PointCloud()
        pcd_pts2.points = o3d.utility.Vector3dVector(pts2[good_pcd2_desc])

        fT = os.path.join(dataset_root, d, '02_T', '{}_{}.pkl'.format(corrs[j].split(',')[0], corrs[j].split(',')[1]))
        T = pickle.load(open(fT, 'rb'))

        pcd_pts2.transform(T)

        # compute distances between points that are in nn in the feature space
        dists = np.linalg.norm(np.asarray(pcd_pts1.points) - np.asarray(pcd_pts2.points)[nn2_inds], axis=1)

        recall_tau1_ths = []

        # final score
        recall_tau1.append(np.mean(dists[mutual_nn] < tau_1))

        # sensitivity analysis
        for th in ths_tau1:
            recall_tau1_ths.append(np.mean(dists[mutual_nn] < th))

        print('dip - {} - {}/{} - recall tau1: {:.3f}'.format(d, j, len(corrs), recall_tau1[-1]))

        RECALL_tau1.append(recall_tau1[-1])
        RECALL_tau1_ths.append(recall_tau1_ths)

    # final score
    RECALL_tau2.append(np.mean(np.asarray(recall_tau1) > tau_2))
    # sensitivity analysis
    _recall_tau2_ths = []
    for th in ths_tau2:
        _recall_tau2_ths.append(np.mean(np.asarray(recall_tau1) > th))
    RECALL_tau2_ths.append(_recall_tau2_ths)

    print('**** fmr {}: {:.4f}.  '.format(d, RECALL_tau2[-1]))
    print('**** racall tau1 {}: {:.4f} +/- {:.4f}  '.format(d, np.mean(recall_tau1), np.std(recall_tau1)))
    print('*****************************')

print('end')
print('*****************************')
print('FINAL SCORES')
print('fmr all: {:.4f}, std: {:.4f}'.format(np.mean(RECALL_tau2), np.std(RECALL_tau2)))
print('recall tau1 all: {:.4f} +/- {:.4f}'.format(np.mean(RECALL_tau1), np.std(RECALL_tau1)))
print('*****************************')
print('SENSITIVITY ANALYSIS')
print('fmr distance thresholds: {}'.format(' '.join(map('{:.4f}'.format, np.mean(np.asarray(RECALL_tau1_ths) > tau_2, axis=0)))))
print('fmr inlier thresholds: {}'.format(' '.join(map('{:.4f}'.format, np.mean(RECALL_tau2_ths, axis=0)))))
