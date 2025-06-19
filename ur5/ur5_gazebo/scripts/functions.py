import numpy as np
from copy import copy
 #import rbdl
import matplotlib.pyplot as plt
import signal
import sys
import matplotlib.ticker as ticker
import subprocess
import tempfile
import os

pi = np.pi
cos = np.cos
sin = np.sin

class Robot(object):
    def __init__(self, q0, dq0, ndof, dt):
        self.q = q0    # numpy array (ndof x 1)
        self.dq = dq0  # numpy array (ndof x 1)
        self.M = np.zeros([ndof, ndof])
        self.b = np.zeros(ndof)
        self.dt = dt
        # Load the URDF model of the UR5 robot with the Robotiq gripper
        urdf_xacro = os.path.join('ur5', 'ur5_description', 'urdf',
                                 'ur5_robotiq85_gripper.urdf.xacro')
        self.robot = self._load_model_from_xacro(urdf_xacro)

    def _load_model_from_xacro(self, xacro_path):
        """Process a xacro file and load it as an RBDL model."""
        try:
            res = subprocess.run(
                ['xacro', '--inorder', xacro_path],
                check=True, capture_output=True, text=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to process {xacro_path}. Ensure xacro is installed") from exc
        with tempfile.NamedTemporaryFile('w', suffix='.urdf', delete=False) as tmp:
            tmp.write(res.stdout)
            tmp_path = tmp.name
        model = rbdl.loadModel(tmp_path)
        os.unlink(tmp_path)
        return model

    def send_command(self, tau):
        rbdl.CompositeRigidBodyAlgorithm(self.robot, self.q, self.M)
        rbdl.NonlinearEffects(self.robot, self.q, self.dq, self.b)
        ddq = np.linalg.inv(self.M).dot(tau-self.b)
        self.q = self.q + self.dt*self.dq
        self.dq = self.dq + self.dt*ddq

    def read_joint_positions(self):
        return self.q

    def read_joint_velocities(self):
        return self.dq


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

    return T


def jacobian_position(q, delta=0.0001):
    """Analytical Jacobian for the end-effector position.

    The optional gripper joint does not affect the Cartesian position so the
    corresponding column will be zeros if present.
    """
    n = len(q)
    J = np.zeros((3, n))
    T = fkine_ur5(q)
    for i in range(n):
        dq = copy(q)
        dq[i] += delta
        T_inc = fkine_ur5(dq)
        J[:, i] = (T_inc[:3, 3] - T[:3, 3]) / delta
    return J


def ikine_ur5(xdes, q0):
    """
    Numerical inverse kinematics for the UR5 using the Newton method.
    xdes -- desired Cartesian position (x, y, z)
    q0   -- initial joint configuration
    """
    epsilon = 0.001
    max_iter = 10000
    delta = 0.00001
    errors = []
    q = copy(q0)
    for _ in range(max_iter):
        T = fkine_ur5(q)
        x = T[0:3, 3]
        error = np.array(xdes - x)
        if np.linalg.norm(error) < epsilon:
            break
        J = jacobian_position(q)
        delta_q = np.linalg.pinv(J) @ error
        q += delta_q
        errors.append(np.linalg.norm(error))
        if np.linalg.norm(delta_q) < delta:
            break
    return q, errors


def ikine_ur5_gradient(xdes, q0, alpha=0.01):
    """
    Numerical inverse kinematics for the UR5 using the gradient method.
    xdes -- desired Cartesian position (x, y, z)
    q0   -- initial joint configuration
    alpha -- learning rate
    """
    epsilon = 0.001
    max_iter = 10000
    delta = 0.00001
    errors = []
    q = copy(q0)
    for _ in range(max_iter):
        T = fkine_ur5(q)
        x = T[0:3, 3]
        error = np.array(xdes - x)
        if np.linalg.norm(error) < epsilon:
            break
        J = jacobian_position(q)
        delta_q = alpha * (J.T @ error)
        q += delta_q
        errors.append(np.linalg.norm(error))
        if np.linalg.norm(delta_q) < delta:
            break
    return q, errors

def jacobian_pose(q, delta=0.0001):
    """Analytical Jacobian for position and orientation (quaternion).

    If a gripper joint is present its contribution is zero as it does not modify
    the tool pose.
    """
    n = len(q)
    J = np.zeros((7, n))
    T = fkine_ur5(q)
    R = TF2xyzquat(T)
    for i in range(n):
        dq = copy(q)
        dq[i] += delta
        T_inc = fkine_ur5(dq)
        R_inc = TF2xyzquat(T_inc)
        J[:, i] = (R_inc - R) / delta
    return J


def rot2quat(R):
    """
    Convert a rotation matrix into a quaternion [ew, ex, ey, ez].
    """
    quat = [0.0, 0.0, 0.0, 0.0]
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        quat[0] = 0.5 * np.sqrt(trace + 1.0)
        s = 1.0 / (4.0 * quat[0])
        quat[1] = (R[2, 1] - R[1, 2]) * s
        quat[2] = (R[0, 2] - R[2, 0]) * s
        quat[3] = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            quat[1] = 0.5 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            s = 1.0 / (4.0 * quat[1])
            quat[0] = (R[2, 1] - R[1, 2]) * s
            quat[2] = (R[1, 0] + R[0, 1]) * s
            quat[3] = (R[0, 2] + R[2, 0]) * s
        elif R[1, 1] > R[2, 2]:
            quat[2] = 0.5 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            s = 1.0 / (4.0 * quat[2])
            quat[0] = (R[0, 2] - R[2, 0]) * s
            quat[1] = (R[1, 0] + R[0, 1]) * s
            quat[3] = (R[2, 1] + R[1, 2]) * s
        else:
            quat[3] = 0.5 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            s = 1.0 / (4.0 * quat[3])
            quat[0] = (R[1, 0] - R[0, 1]) * s
            quat[1] = (R[0, 2] + R[2, 0]) * s
            quat[2] = (R[2, 1] + R[1, 2]) * s
    return np.array(quat)


def TF2xyzquat(T):
    """
    Convert a homogeneous transformation matrix into a pose vector
    [x, y, z, ew, ex, ey, ez].
    """
    quat = rot2quat(T[0:3, 0:3])
    res = [T[0, 3], T[1, 3], T[2, 3], quat[0], quat[1], quat[2], quat[3]]
    return np.array(res)


def skew(w):
    R = np.zeros([3, 3])
    R[0, 1] = -w[2]; R[0, 2] = w[1]
    R[1, 0] = w[2];  R[1, 2] = -w[0]
    R[2, 0] = -w[1]; R[2, 1] = w[0]
    return R



