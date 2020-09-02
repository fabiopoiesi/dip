import os
import numpy as np
import torch.utils.data as data
import torch
import h5py
import random
from scipy.spatial.transform import Rotation as rotation


class Dataset(data.Dataset):

    def __init__(self, root,
                 dataset,
                 dataset_to_train,
                 ratio_to_eval=None,
                 do_data_aug=True):

        self.root_ptcld = os.path.join(root, dataset)
        self.root_ptcld_overlap = os.path.join(root, dataset + '_pre', 'correspondences')

        self.list_ptcld_overlap_files = os.listdir(self.root_ptcld_overlap)
        self.list_ptcld_overlap_files.sort()

        self.list_pointers_to_train = np.empty((0, 2))
        self.list_pointers_to_eval = np.empty((0, 2))

        self.ratio_to_eval = ratio_to_eval

        self.do_data_aug = do_data_aug

        # if dataset_to_train is different from None only the dataset specified are used for training
        # we used this option for the ablation study in the paper
        if dataset_to_train != None:
            _list_ptcld_overlap_files = []
            for p in dataset_to_train:
                _list_ptcld_overlap_files.append(self.list_ptcld_overlap_files[p])
            self.list_ptcld_overlap_files = _list_ptcld_overlap_files

        for f in self.list_ptcld_overlap_files:
            hf = h5py.File(os.path.join(self.root_ptcld_overlap, f), 'r')
            corrs_to_train = np.asarray(list(hf.keys()))
            np.random.shuffle(corrs_to_train)

            # use some correspondences for validation during training
            if self.ratio_to_eval != None:
                n_to_eval = int(self.ratio_to_eval * len(corrs_to_train))
                idx_to_eval = list(range(n_to_eval))
                corrs_to_eval = corrs_to_train[idx_to_eval]
                for c in corrs_to_eval:
                    self.list_pointers_to_eval = np.vstack((self.list_pointers_to_eval, (f, c)))
                corrs_to_train = np.delete(corrs_to_train, idx_to_eval)

            for c in corrs_to_train:
                self.list_pointers_to_train = np.vstack((self.list_pointers_to_train, (f, c)))

        self.length = len(self.list_pointers_to_train)


    def __getitem__(self, index):

        pointer = self.list_pointers_to_train[index]

        dataset = pointer[0].split('.')[0]
        frags = pointer[1]

        hf_patches = h5py.File(os.path.join(self.root_ptcld + '_pre', 'patches_lrf', '{}.hdf5'.format(dataset)), 'r')
        hf_points = h5py.File(os.path.join(self.root_ptcld + '_pre', 'points_lrf', '{}.hdf5'.format(dataset)), 'r')
        hf_rotations = h5py.File(os.path.join(self.root_ptcld + '_pre', 'rotations_lrf', '{}.hdf5'.format(dataset)), 'r')
        hf_lrfs = h5py.File(os.path.join(self.root_ptcld + '_pre', 'lrfs', '{}.hdf5'.format(dataset)), 'r')

        patches = np.asarray(hf_patches[frags])
        frag1_batch = patches[0]
        frag2_batch = patches[1]

        rotations = np.asarray(hf_rotations[frags])
        R1 = rotations[0]
        R2 = rotations[1]

        if self.do_data_aug:
            rot_int = np.pi / 30
            R1 = rotation.from_euler('zyx', [random.uniform(0, rot_int),
                                             random.uniform(0, rot_int),
                                             random.uniform(0, rot_int)]).as_matrix()

            R2 = rotation.from_euler('zyx', [random.uniform(0, rot_int),
                                             random.uniform(0, rot_int),
                                             random.uniform(0, rot_int)]).as_matrix()

            frag1_batch = np.dot(R1, frag1_batch).transpose((1, 0, 2))
            frag2_batch = np.dot(R2, frag2_batch).transpose((1, 0, 2))

        frag1_batch = torch.Tensor(frag1_batch)
        frag2_batch = torch.Tensor(frag2_batch)

        points = np.asarray(hf_points[frags])
        fps_pcd1_pts = torch.Tensor(points[0])
        fps_pcd2_pts = torch.Tensor(points[1])

        lrfs = np.asarray(hf_lrfs[frags])
        lrf1 = torch.Tensor(lrfs[0])
        lrf2 = torch.Tensor(lrfs[1])

        hf_patches.close()
        hf_points.close()
        hf_rotations.close()
        hf_lrfs.close()

        return frag1_batch, frag2_batch, fps_pcd1_pts, fps_pcd2_pts, torch.Tensor(R1), torch.Tensor(R2), lrf1, lrf2


    def __len__(self):
        return self.length


    def get_eval_length(self):
        return len(self.list_pointers_to_eval)


    def get_eval_item(self, index):

        pointer = self.list_pointers_to_eval[index]

        dataset = pointer[0].split('.')[0]
        frags = pointer[1]

        hf_patches = h5py.File(os.path.join(self.root_ptcld + '_pre', 'patches_lrf', '{}.hdf5'.format(dataset)), 'r')
        hf_points = h5py.File(os.path.join(self.root_ptcld + '_pre', 'points_lrf', '{}.hdf5'.format(dataset)), 'r')
        hf_rotations = h5py.File(os.path.join(self.root_ptcld + '_pre', 'rotations_lrf', '{}.hdf5'.format(dataset)), 'r')
        hf_lrfs = h5py.File(os.path.join(self.root_ptcld + '_pre', 'lrfs', '{}.hdf5'.format(dataset)), 'r')

        patches = np.asarray(hf_patches[frags])
        frag1_batch = patches[0]
        frag2_batch = patches[1]

        rotations = np.asarray(hf_rotations[frags])
        R1 = rotations[0]
        R2 = rotations[1]

        frag1_batch = torch.Tensor(frag1_batch)
        frag2_batch = torch.Tensor(frag2_batch)

        points = np.asarray(hf_points[frags])
        fps_pcd1_pts = torch.Tensor(points[0])
        fps_pcd2_pts = torch.Tensor(points[1])

        lrfs = np.asarray(hf_lrfs[frags])
        lrf1 = torch.Tensor(lrfs[0])
        lrf2 = torch.Tensor(lrfs[1])

        hf_patches.close()
        hf_points.close()
        hf_rotations.close()
        hf_lrfs.close()

        return frag1_batch, frag2_batch, fps_pcd1_pts, fps_pcd2_pts, torch.Tensor(R1), torch.Tensor(R2), lrf1, lrf2