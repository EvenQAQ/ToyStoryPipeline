import math
import os
import json
import numpy as np
from modules.input_reader import VideoReader, ImageReader
from modules.draw import Plotter3d, draw_poses
from modules.parse_poses import parse_poses


def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)
    return poses_3d


if __name__ == "__main__":
    poses_3d = np.array([np.array([4.4186062e+01, -4.3892944e+01,  2.0369699e+02,  8.3816779e-01, 3.9134197e+01, -5.3319836e+01,  1.8532994e+02,  6.9642085e-01, 5.0239643e+01, -6.9828620e+00,  2.3182362e+02, -1.0000000e+00, 5.9791008e+01, -4.3372654e+01,  1.9962967e+02,  7.1894360e-01, 6.8975769e+01, -2.3130114e+01,  2.0520830e+02,  6.8528396e-01, 5.6719280e+01, -1.3284249e+01,  1.9486157e+02,  6.4200741e-01, 5.8286469e+01,  1.3217036e+00,  2.2654358e+02,  6.8404305e-01, 6.1502823e+01,  3.2493820e+01,  2.4396898e+02,  7.1215242e-01, 6.2697735e+01,  6.2765778e+01,  2.6432172e+02,  5.9681177e-01, 3.2124569e+01, -
                                   4.4797119e+01,  2.0476782e+02,  7.4792016e-01, 1.4896990e+01, -2.9107800e+01,  2.1247116e+02,  5.5395073e-01, 1.5764141e+01, -2.7828636e+01,  2.0022934e+02,  2.9126814e-01, 3.4911530e+01, -4.5747619e+00,  2.2310397e+02,  6.6975766e-01, 3.2192688e+01,  2.3215109e+01,  2.3480083e+02,  6.5392292e-01, 3.1397924e+01,  5.2953651e+01,  2.5089944e+02,  4.1037539e-01, 3.9963078e+01, -5.3750423e+01,  1.8360481e+02,  6.4856184e-01, 4.4914398e+01, -5.3077751e+01,  1.8983868e+02,  6.3344139e-01, 3.6533707e+01, -5.4417770e+01,  1.8456305e+02,  1.1476542e-01, 3.3946823e+01, -5.6082458e+01,  1.9110008e+02,  6.9746691e-01])])

    file_path = os.path.join('data', 'extrinsics.json')
    with open(file_path, 'r') as f:
        extrinsics = json.load(f)
    R = np.array(extrinsics['R'], dtype=np.float32)
    t = np.array(extrinsics['t'], dtype=np.float32)
    print(poses_3d)
    if len(poses_3d):
        poses_3d = rotate_poses(poses_3d, R, t)
        print(poses_3d)
        poses_3d_copy = poses_3d.copy()
        x = poses_3d_copy[:, 0::4]
        y = poses_3d_copy[:, 1::4]
        z = poses_3d_copy[:, 2::4]
        poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y

        poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
        edges = (Plotter3d.SKELETON_EDGES + 19 *
                 np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
    print(poses_3d)
