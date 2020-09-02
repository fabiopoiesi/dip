from scipy.spatial import cKDTree
import numpy as np
import open3d as o3d
import torch
from network import PointNetFeature
import h5py
import os

# directory of the checkpoint/model to test
model_root = './model'

dataset_root = 'path-to-dataset-root/ETH_test'

root_dirs = ['gazebo_summer',
             'gazebo_winter',
             'wood_autmn',
             'wood_summer']

tau_1 = 0.1
tau_2 = 0.05

pts_to_sample = 5000
batch_size = 1500
dim = 32
perc = 5

ckpt_name = 'final_chkpt.pth'

net = PointNetFeature(dim=dim, l2norm=True, tnet=True)

checkpoint = '{}/{}'.format(model_root, ckpt_name)
net.load_state_dict(torch.load(checkpoint))
net.cuda()
net.eval()

recall_tau2 = []
RECALL_tau1 = []
RECALL_tau2 = []

ths_tau1 = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17,
            0.18, 0.19, 0.2]
RECALL_tau1_ths = []

ths_tau2 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
            0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20,
            0.21]
RECALL_tau2_ths = []

print('*****************************')
print('testing:' + checkpoint)
print('*****************************')
print('datasets to test:')
for i, d in enumerate(root_dirs):
    hf_patches = h5py.File(os.path.join(dataset_root + '_pre', 'patches_lrf', '{}.hdf5'.format(d)), 'r')
    corrs = np.asarray(list(hf_patches.keys()))
    print('{}. {} -> {} pairs'.format(i + 1, d, len(corrs)))

print('*****************************')
print('start:')

for d in root_dirs:

    hf_patches = h5py.File(os.path.join(dataset_root + '_pre', 'patches_lrf', '{}.hdf5'.format(d)), 'r')
    hf_points = h5py.File(os.path.join(dataset_root + '_pre', 'points_lrf', '{}.hdf5'.format(d)), 'r')

    gt_file = open(os.path.join(dataset_root, d, 'gt.log'), 'r')
    gt = gt_file.readlines()
    nfrag = int(len(gt) / 5)

    recall_tau1 = []

    for frag in range(nfrag):

        frag_ptr = frag * 5
        info = gt[frag_ptr].split('\t')
        pcd1_id = int(info[0])
        pcd2_id = int(info[1])

        corrs = '{},{}'.format(pcd1_id, pcd2_id)

        patches = np.asarray(hf_patches[corrs])
        patches1 = patches[0]
        patches2 = patches[1]

        points = np.asarray(hf_points[corrs])
        pts1 = points[0]
        pts2 = points[1]

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

        # read transformation
        T = np.empty((4, 4))
        T[0, :] = np.asarray(gt[frag_ptr + 1].split('\t'), dtype=np.float)
        T[1, :] = np.asarray(gt[frag_ptr + 2].split('\t'), dtype=np.float)
        T[2, :] = np.asarray(gt[frag_ptr + 3].split('\t'), dtype=np.float)
        T[3, :] = np.asarray(gt[frag_ptr + 4].split('\t'), dtype=np.float)

        pcd_pts2.transform(T)

        # compute distances between points that are in nn in the feature space
        dists = np.linalg.norm(np.asarray(pcd_pts1.points) - np.asarray(pcd_pts2.points)[nn2_inds], axis=1)

        # final score
        recall_tau1.append(np.mean(dists[mutual_nn] < tau_1))

        print('dip - {} - {}/{} - recall tau1: {:.3f}'.format(d, frag, nfrag, recall_tau1[-1]))

    # final score
    RECALL_tau2.append(np.mean(np.asarray(recall_tau1) > tau_2))

    print('**** fmr {}: {:.4f}.  '.format(d, RECALL_tau2[-1]))
    print('*****************************')

print('end')
print('*****************************')
print('FINAL SCORE')
print('fmr all: {:.4f}, std: {:.4f}'.format(np.mean(RECALL_tau2), np.std(RECALL_tau2)))
print('*****************************')