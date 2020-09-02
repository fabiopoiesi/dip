import numpy as np
import os
from dataset import Dataset
import torch
import torch.optim as optim
from network import PointNetFeature
from losses import hardest_contrastive
import torch_nndistance as NND
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

'''
Notes:
1. training assumes that data (patches, LRFs) are preprocessed with preprocess_3dmatch_lrf_train.py
(it would be too slow to process this info during training)
2. batch size, patch size, lrf kernel size are defined in the preprocessing step
'''

chkpt_dir = './chkpts'
if not os.path.isdir(chkpt_dir):
    os.mkdir(chkpt_dir)

dataset_root = 'path-to-dataset-root' # this is directory that contains '3DMatch_train' directory

dataset_name = '3DMatch_train'

do_data_augmentation = False # activate/deactivate data augmentation
l2norm = True # activate/deactivate LRN
tnet = True # activate/deactivate TNet

dataset_to_train = None # [0, 1]
nepochs = 40
ratio_to_eval = .002 # ratio of the training set used for validation during training
dim = 32

model = PointNetFeature(dim=dim, l2norm=l2norm, tnet=tnet)
device_ids = [0,1,2] # change this according to your GPU setup, e.g. if you have only one GPU -> device_ids = [0]
net = nn.DataParallel(model, device_ids=device_ids).cuda()
net.train()

dataset = Dataset(dataset_root, dataset_name, dataset_to_train, ratio_to_eval, do_data_augmentation)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)

optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3, nesterov=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# logger    
log_dir_root = './logs'
if not os.path.isdir(log_dir_root):
    os.mkdir(log_dir_root)

date = datetime.now().timetuple()
log_dir = os.path.join(log_dir_root, '{}.{:02d}.{:02d}.{:02d}.{:02d}.{:02d}'.format(date[0], date[1], date[2], date[3], date[4], date[5]))

writer = SummaryWriter(log_dir=log_dir)

dists_eval = np.empty((dataset.get_eval_length(),))

n_iter = 0
for e in range(nepochs):
    i = 1
    for frag1_batch, frag2_batch, _, _, R1, R2, lrf1, lrf2 in dataloader:

        '''
        TRAINING
        '''
        frag1_batch = frag1_batch.squeeze().cuda()
        frag2_batch = frag2_batch.squeeze().cuda()
        R1 = R1.squeeze().cuda()
        R2 = R2.squeeze().cuda()
        lrf1 = lrf1.squeeze().cuda()
        lrf2 = lrf2.squeeze().cuda()

        optimizer.zero_grad()

        f1, xtrans1, trans1, f2, xtrans2, trans2 = net(frag1_batch, frag2_batch)

        # hardest-contrastive loss
        lcontrastive, a, b, c = hardest_contrastive(f1, f2)
        # chamfer loss
        dist1, dist2 = NND.nnd(xtrans1.transpose(2, 1).contiguous(), xtrans2.transpose(2, 1).contiguous())
        lchamf = .5 * (torch.mean(dist1) + torch.mean(dist2))
        # combination of losses
        loss = lcontrastive + lchamf

        loss.backward()
        optimizer.step()

        writer.add_scalar('loss/train', loss.item(), n_iter)
        writer.add_scalar('hardest_contrastive/positive - train', torch.mean(a).item(), n_iter)
        writer.add_scalar('hardest_contrastive/negative1 - train', torch.mean(b[0]).item(), n_iter)
        writer.add_scalar('hardest_contrastive/negative2 - train', torch.mean(c[0]).item(), n_iter)
        writer.add_scalar('chamfer/train', lchamf.item(), n_iter)
        writer.add_scalar('tnet determinant/train', torch.det(trans1[0]), n_iter)

        '''
        VALIDATION
        '''
        if i % 50 == 0 and i >= 50:
            net.eval()
            with torch.no_grad():
                for j in range(dataset.get_eval_length()):
                    frag1_batch, frag2_batch, _, _, R1, R2, lrf1, lrf2 = dataset.get_eval_item(j)
                    frag1_batch = frag1_batch.cuda()
                    frag2_batch = frag2_batch.cuda()
                    R1 = R1.squeeze().cuda()
                    R2 = R2.squeeze().cuda()
                    lrf1 = lrf1.squeeze().cuda()
                    lrf2 = lrf2.squeeze().cuda()

                    f1, _, trans1, f2, _, trans2 = net(frag1_batch, frag2_batch)

                    # hardest-contrastive loss
                    lcontrastive, a, b, c = hardest_contrastive(f1, f2)
                    # chamfer loss
                    dist1, dist2 = NND.nnd(xtrans1.transpose(2, 1).contiguous(), xtrans2.transpose(2, 1).contiguous())
                    lchamf = .5 * (torch.mean(dist1) + torch.mean(dist2))
                    # combination of losses
                    dists_eval[j] = lcontrastive + lchamf

                writer.add_scalar('loss/validation', np.mean(dists_eval), n_iter)

            net.train()

        if i % 2400 == 0:
            fckpt_name = '{}/ckpt_e{}_i{}_dim{}.pth'.format(chkpt_dir, e, i, dim)
            torch.save(net.state_dict(), fckpt_name)

        if n_iter % 50 == 0 and i >= 50:
            print('iter {:06d}/{} (train loss {:.4f}, valid loss {:.4f})'.format(n_iter, len(dataloader) * nepochs, loss.item(), np.mean(dists_eval)))

        i += 1
        n_iter += 1

    scheduler.step()