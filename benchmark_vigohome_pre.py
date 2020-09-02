import numpy as np
import open3d as o3d
import os
import pickle

dataset_root = 'path-to-dataset-root/VigoHome'

tau = 0.2
pts_to_sample = 5000 # in the paper this was changed between 15K to 500, see Tab. VI
num_iters = 100

print('*****************************')
print('testing VigoHome registration')
print('*****************************')
print('start:')

error_mean = []

for iter in range(num_iters):

    # GET POINTS AND DESCRIPTORS
    file = open(os.path.join(dataset_root, 'points_and_descriptors.pkl'), 'rb')
    data = pickle.load(file)
    file.close()

    _pcd1 = o3d.geometry.PointCloud()
    _pcd2 = o3d.geometry.PointCloud()
    _pcd3 = o3d.geometry.PointCloud()
    pcd1_dsdv = o3d.registration.Feature()
    pcd2_dsdv = o3d.registration.Feature()
    pcd3_dsdv = o3d.registration.Feature()

    if data[0].shape[0] > pts_to_sample and data[1].shape[0] > pts_to_sample and data[2].shape[0] > pts_to_sample:
        inds1 = np.random.choice(data[0].shape[0], pts_to_sample, replace=False)
        inds2 = np.random.choice(data[1].shape[0], pts_to_sample, replace=False)
        inds3 = np.random.choice(data[2].shape[0], pts_to_sample, replace=False)

        _pcd1.points = o3d.utility.Vector3dVector(data[0][inds1])
        _pcd2.points = o3d.utility.Vector3dVector(data[1][inds2])
        _pcd3.points = o3d.utility.Vector3dVector(data[2][inds3])

        pcd1_dsdv.data = data[3][inds1].T
        pcd2_dsdv.data = data[4][inds2].T
        pcd3_dsdv.data = data[5][inds3].T
    else:
        _pcd1.points = o3d.utility.Vector3dVector(data[0])
        _pcd2.points = o3d.utility.Vector3dVector(data[1])
        _pcd3.points = o3d.utility.Vector3dVector(data[2])

        pcd1_dsdv.data = data[3].T
        pcd2_dsdv.data = data[4].T
        pcd3_dsdv.data = data[5].T

    # GET CORRESPONDENCES
    file = open(os.path.join(dataset_root, 'correspondences.pkl'), 'rb')
    corrs = pickle.load(file)
    file.close()

    corrs12 = corrs[0]['bedroom_upstairs-bathroom_upstairs']
    corrs32 = corrs[0]['livingroom_downstairs-bathroom_upstairs']

    # GET POINT CLOUDS
    file = open(os.path.join(dataset_root, 'point_clouds.pkl'), 'rb')
    point_clouds = pickle.load(file)
    file.close()

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(point_clouds[0]['bedroom_upstairs'])

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(point_clouds[0]['bathroom_upstairs'])

    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(point_clouds[0]['livingroom_downstairs'])

    est_result12 = o3d.registration.registration_ransac_based_on_feature_matching(
        _pcd1,
        _pcd2,
        pcd1_dsdv,
        pcd2_dsdv,
        .05,
        estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(.9),
                  o3d.registration.CorrespondenceCheckerBasedOnDistance(.05)],
        criteria=o3d.registration.RANSACConvergenceCriteria(50000, 500)
    )

    est_result32 = o3d.registration.registration_ransac_based_on_feature_matching(
        _pcd3,
        _pcd2,
        pcd3_dsdv,
        pcd2_dsdv,
        .05,
        estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(.9),
                  o3d.registration.CorrespondenceCheckerBasedOnDistance(.05)],
        criteria=o3d.registration.RANSACConvergenceCriteria(50000, 500)
    )

    pcd1.transform(est_result12.transformation)
    pcd3.transform(est_result32.transformation)

    pcd1_overlap_pts = np.asarray(pcd1.points)[corrs12[:, 0]]
    pcd2_overlap_pts = np.asarray(pcd2.points)[corrs12[:, 1]]
    norm12 = np.linalg.norm(pcd1_overlap_pts - pcd2_overlap_pts, axis=1)

    pcd3_overlap_pts = np.asarray(pcd3.points)[corrs32[:, 0]]
    pcd2_overlap_pts = np.asarray(pcd2.points)[corrs32[:, 1]]
    norm32 = np.linalg.norm(pcd3_overlap_pts - pcd2_overlap_pts, axis=1)

    emean = np.mean(np.hstack((norm12, norm32)))

    error_mean.append(emean)

    print('DIP - VigoHome - {}/{} - mean err: {:.4f}'.format(iter, num_iters, emean))

print('**** registration recall: {:.4f}'.format(np.mean(np.asarray(error_mean) < tau)))
print('*****************************')