import torch
import torch.nn.functional as F

'''
HARDEST-CONTRASTIVE
'''
mp = torch.Tensor([.1]).cuda()
mn = torch.Tensor([1.4]).cuda()
def hardest_contrastive(fxd, fxm):
    big_eye = 1e9 * torch.eye(fxd.shape[0]).cuda()

    fxd_r = torch.stack([fxd] * fxd.shape[0])
    fxm_r = torch.stack([fxm] * fxm.shape[0]).transpose(0, 1)
    fdists_all = torch.norm(fxd_r - fxm_r, dim=2).T + big_eye

    '''
    fdists_all =

    ||fxd[0]-fxm[0] ||fxd[0]-fxm[1]|| ... ||fxd[0]-fxm[N]||
    ||fxd[1]-fxm[0] ||fxd[1]-fxm[1]|| ... ||fxd[1]-fxm[N]|| 
    .                                       .
    .                                       .
    .                                       .
    ||fxd[N]-fxm[0] ||fxd[N]-fxm[1]|| ... ||fxd[N]-fxm[N]||
    '''

    fdm_mins, fdm_argmins = torch.min(fdists_all, dim=1)
    fmd_mins, fmd_argmins = torch.min(fdists_all, dim=0)

    fdists_pos = torch.norm(fxd - fxm, dim=1)

    a = F.relu(fdists_pos - mp).pow(2).sum() / len(fdists_pos)
    b = torch.mean(F.relu(mn - fdm_mins).pow(2))
    c = torch.mean(F.relu(mn - fmd_mins).pow(2))

    l = a + (b + c) / 2

    return l, fdists_pos, torch.median(fdists_all, dim=1), torch.median(fdists_all, dim=0)