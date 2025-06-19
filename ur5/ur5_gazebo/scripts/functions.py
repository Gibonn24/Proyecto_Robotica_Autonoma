import numpy as np


def rotx(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[1, 0, 0, 0],
                     [0, c, -s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 1]])


def roty(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0, s, 0],
                     [0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [0, 0, 0, 1]])


def rotz(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def transl(x, y, z):
    T = np.eye(4)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


def dh(a, alpha, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])


def fkine_ur5(q):
    """Forward kinematics of the UR5 with a Robotiq gripper."""
    d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
    a = [0, -0.42500, -0.39225, 0, 0, 0]
    alpha = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]

    T = np.eye(4)
    for i in range(6):
        T = T @ dh(a[i], alpha[i], d[i], q[i])

    # wrist_3_link to tool0
    T = T @ transl(0, d[5], 0) @ rotx(-np.pi/2)
    # tool0 to gripper base (coupler + base joint)
    T = T @ transl(0, 0, 0.004) @ rotz(-np.pi/2)
    T = T @ transl(0, 0, 0.004) @ roty(-np.pi/2) @ rotz(np.pi)
    return T
