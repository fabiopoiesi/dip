import numpy as np
import open3d as o3d

class lrf():
    '''
    This is our re-implementation (+adaptation) of the LRF computed in:
    Z. Gojcic, C. Zhou, J. Wegner, and W. Andreas,
    “The perfect match: 3Dpoint cloud matching with smoothed densities,”
    CVPR, 2019
    '''
    def __init__(self, pcd, pcd_tree, lrf_kernel, patch_size, viz=False):

        self.pcd = pcd
        self.pcd_tree = pcd_tree
        self.do_viz = viz
        self.patch_kernel = lrf_kernel
        self.patch_size = patch_size

    def get(self, pt):

        _, patch_idx, _ = self.pcd_tree.search_radius_vector_3d(pt, self.patch_kernel)

        ptnn = np.asarray(self.pcd.points)[patch_idx[1:], :].T
        ptall = np.asarray(self.pcd.points)[patch_idx, :].T

        # eq. 3
        ptnn_cov = 1 / len(ptnn) * np.dot((ptnn - pt[:, np.newaxis]), (ptnn - pt[:, np.newaxis]).T)

        if len(patch_idx) < self.patch_kernel / 2:
            _, patch_idx, _ = self.pcd_tree.search_knn_vector_3d(pt, self.patch_kernel)

        # The normalized (unit “length”) eigenvectors, s.t. the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
        a, v = np.linalg.eig(ptnn_cov)
        smallest_eigevalue_idx = np.argmin(a)
        np_hat = v[:, smallest_eigevalue_idx]

        # eq. 4
        zp = np_hat if np.sum(np.dot(np_hat, pt[:, np.newaxis] - ptnn)) > 0 else - np_hat

        v = (ptnn - pt[:, np.newaxis]) - (np.dot((ptnn - pt[:, np.newaxis]).T, zp[:, np.newaxis]) * zp).T
        alpha = (self.patch_kernel - np.linalg.norm(pt[:, np.newaxis] - ptnn, axis=0)) ** 2
        beta = np.dot((ptnn - pt[:, np.newaxis]).T, zp[:, np.newaxis]).squeeze() ** 2

        # e.q. 5
        xp = 1 / np.linalg.norm(np.dot(v, (alpha * beta)[:, np.newaxis])) * np.dot(v, (alpha * beta)[:, np.newaxis])
        xp = xp.squeeze()

        yp = np.cross(xp, zp)

        lRg = np.asarray([xp, yp, zp]).T

        # rotate w.r.t local frame and centre in zero using the chosen point
        ptall = (lRg.T @ (ptall - pt[:, np.newaxis])).T

        # this is our normalisation
        ptall /= self.patch_kernel

        T = np.zeros((4, 4))
        T[-1, -1] = 1
        T[:3, :3] = lRg
        T[:3, -1] = pt

        # visualise patch and local reference frame
        if self.do_viz:
            self.pcd.paint_uniform_color([.3, .3, .3])
            self.pcd.estimate_normals()
            np.asarray(self.pcd.colors)[patch_idx[1:]] = [0, 1, 0]
            local_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            local_frame.transform(T)
            o3d.visualization.draw_geometries([self.pcd, local_frame])

        # to make sure that there are at least self.patch_size points, pad with zeros if not
        if ptall.shape[0] < self.patch_size:
            ptall = np.concatenate((ptall, np.zeros((self.patch_size - ptall.shape[0], 3))))

        inds = np.random.choice(ptall.shape[0], self.patch_size, replace=False)

        return ptall[inds], pt, T
