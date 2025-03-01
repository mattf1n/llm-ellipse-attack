import numpy as np

def angle(rot_a, rot_b):
    rot = rot_a @ rot_b.T
    eigs = np.linalg.eigvals(rot)
    print(eigs)
