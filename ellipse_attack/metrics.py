import numpy as np

def angle(rot_a, rot_b):
    rot = rot_a @ rot_b.T
    eigs = np.linalg.eigvals(rot)
    pos_angles = np.maximum(0, np.angle(eigs))
    angles = np.minimum(pos_angles, np.abs(np.pi - pos_angles))
    return np.sqrt(np.square(angles).sum())
