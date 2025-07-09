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
import random
from mpl_toolkits.mplot3d import Axes3D
import numpy as np






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


def dh(d, theta, a, alpha):
    """
    Matriz de transformacion homogenea asociada a los parametros DH.
    Retorna una matriz 4x4
    """
    sth = np.sin(theta)
    cth = np.cos(theta)
    sa  = np.sin(alpha)
    ca  = np.cos(alpha)
    T = np.array([[cth, -ca*sth,  sa*sth, a*cth],
               [sth,  ca*cth, -sa*cth, a*sth],
               [0.0,      sa,      ca,     d],
               [0.0,     0.0,     0.0,   1.0]])
    return T


def fkine_ur5(q):
    """
    Calcular la cinematica directa del robot UR5 dados sus valores articulares.
    q es un vector numpy de la forma [q1, q2, q3, q4, q5, q6]
    """

    T1 = dh(0.08920,    q[0],      0,  pi/2)
    T2 = dh(      0,    q[1], -0.425,     0)
    T3 = dh(      0,    q[2], -0.392,     0)
    T4 = dh(0.10930, q[3]+pi,      0, -pi/2)
    T5 = dh(0.09475,    q[4],      0,  pi/2)
    T6 = dh(  0.125,    q[5],      0,     0)
 # Efector final con respecto a la base
    T = T1.dot(T2).dot(T3).dot(T4).dot(T5).dot(T6)
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
        # Copiar la configuracion articular inicial (usar este dq para cada
        # incremento en una articulacion)
        dq = copy(q)
        # Incrementar la articulacion i-esima usando un delta
        dq[i] += delta
        # Transformacion homogenea luego del incremento (q+dq)
        T_inc = fkine_ur5(dq)
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
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
    q = np.asarray(copy(q0), dtype=float).reshape(-1)
    for _ in range(max_iter):
        T = fkine_ur5(q)
        x = T[0:3, 3]
        error = np.array(xdes - x)
        if np.linalg.norm(error) < epsilon:
            break
        J = jacobian_position(q)
        delta_q = np.linalg.pinv(J) @ error
        # Update joint angles
        q = np.asarray(q + delta_q, dtype=float).reshape(-1)
        errors.append(np.linalg.norm(error))
        if np.linalg.norm(delta_q) < delta:
            break
    return q, errors

def rotation_error_to_vector(R_err):
    """
    Convierte la matriz de error de rotación en un vector de rotación.
    """
    theta = np.arccos((np.trace(R_err) - 1) / 2.0)

    # Para evitar división por cero
    if abs(theta) < 1e-6:
        return np.zeros(3)

    wx = (R_err[2,1] - R_err[1,2]) / (2*np.sin(theta))
    wy = (R_err[0,2] - R_err[2,0]) / (2*np.sin(theta))
    wz = (R_err[1,0] - R_err[0,1]) / (2*np.sin(theta))

    return theta * np.array([wx, wy, wz])

def ikine_ur5_ori(T_des, q0):
    """
    IK numérica para UR5 considerando posición y orientación.
    """
    epsilon_pos = 0.001
    epsilon_ori = 0.01
    max_iter = 10000
    delta = 0.00001
    errors = []

    q = np.asarray(copy(q0), dtype=float).reshape(-1)

    for _ in range(max_iter):
        T = fkine_ur5(q)
        p = T[0:3, 3]
        R_curr = T[0:3, 0:3]

        p_des = T_des[0:3, 3]
        R_des = T_des[0:3, 0:3]

        e_pos = p_des - p
        R_err = R_des @ R_curr.T
        e_ori = rotation_error_to_vector(R_err)

        error = np.concatenate((e_pos, e_ori))

        if np.linalg.norm(e_pos) < epsilon_pos and np.linalg.norm(e_ori) < epsilon_ori:
            break

        J = jacobian_pose_vector(q)
        delta_q = np.linalg.pinv(J) @ error
        q = q + delta_q

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
    q = np.asarray(copy(q0), dtype=float).reshape(-1)
    for _ in range(max_iter):
        T = fkine_ur5(q)
        x = T[0:3, 3]
        error = np.array(xdes - x)
        if np.linalg.norm(error) < epsilon:
            break
        J = jacobian_position(q)
        delta_q = alpha * (J.T @ error)
        q = np.asarray(q + delta_q, dtype=float).reshape(-1)
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

def jacobian_pose_vector(q, delta=1e-5):
    """
    Jacobiano numérico de posición y orientación (como vector rotacional).
    Retorna un Jacobiano de 6 x n.
    """
    n = len(q)
    J = np.zeros((6, n))
    T = fkine_ur5(q)
    R = T[0:3, 0:3]
    p = T[0:3, 3]

    for i in range(n):
        dq = copy(q)
        dq[i] += delta
        T_inc = fkine_ur5(dq)
        R_inc = T_inc[0:3, 0:3]
        p_inc = T_inc[0:3, 3]

        # Diferencias
        dp = (p_inc - p) / delta
        R_err = R_inc @ R.T
        do = rotation_error_to_vector(R_err) / delta

        J[0:3, i] = dp
        J[3:6, i] = do

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










#######################################################

class RRT3D(object):
    class Node:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
            self.path_x = []
            self.path_y = []
            self.path_z = []
            self.parent = None


    def __init__(self, start, goal, obstacle_list, rand_area=0.05, expand_dis=0.05, 
                 path_resolution=0.001, goal_sample_rate=10, max_iter=1000):
        self.start = self.Node(start[0], start[1], start[2])
        self.end = self.Node(goal[0], goal[1], goal[2])
        self.x_rand = rand_area[0]  # [xmin, xmax]
        self.y_rand = rand_area[1]  # [ymin, ymax]
        self.z_rand = rand_area[2]  # [zmin, zmax]

        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []

    def planning(self, animation=False):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)

            if self.calc_dist_to_goal(self.node_list[-1].x,
                                      self.node_list[-1].y,
                                      self.node_list[-1].z) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list):
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

        return None

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(from_node.x, from_node.y, from_node.z)
        d, theta, phi = self.calc_distance_and_angles(new_node, to_node)
        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]
        new_node.path_z = [new_node.z]

        if extend_length > d:
            extend_length = d
        n_expand = int(np.floor(extend_length / self.path_resolution))

        for _ in range(n_expand):
            new_node.x += self.path_resolution * np.cos(theta) * np.sin(phi)
            new_node.y += self.path_resolution * np.sin(theta) * np.sin(phi)
            new_node.z += self.path_resolution * np.cos(phi)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)
            new_node.path_z.append(new_node.z)

        d, _, _ = self.calc_distance_and_angles(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.path_z.append(to_node.z)
            new_node.x = to_node.x
            new_node.y = to_node.y
            new_node.z = to_node.z

        new_node.parent = from_node
        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y, self.end.z]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y, node.z])
            node = node.parent
        path.append([node.x, node.y, node.z])
        return path[::-1]

    def calc_dist_to_goal(self, x, y, z):
        dx = x - self.end.x
        dy = y - self.end.y
        dz = z - self.end.z
        return np.sqrt(dx**2 + dy**2 + dz**2)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(

                random.uniform(self.x_rand[0], self.x_rand[1]),
                random.uniform(self.y_rand[0], self.y_rand[1]),
                random.uniform(self.z_rand[0], self.z_rand[1]))

        else:
            rnd = self.Node(self.end.x, self.end.y, self.end.z)
        return rnd

    def draw_graph(self, rnd=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if rnd is not None:
            ax.scatter(rnd.x, rnd.y, rnd.z, c='k', marker='^')

        for node in self.node_list:
            if node.parent:
                ax.plot(node.path_x, node.path_y, node.path_z, "-y")

        for (ox, oy, oz, size) in self.obstacle_list:

            self.plot_cube(ax, ox, oy, oz, sizex,sizey,sizez)

        ax.scatter(self.start.x, self.start.y, self.start.z, c='r', label='Start')
        ax.scatter(self.end.x, self.end.y, self.end.z, c='g', label='Goal')
        
        
        ax.set_xlim(self.x_rand[0], self.x_rand[1])
        ax.set_ylim(self.y_rand[0], self.y_rand[1])
        ax.set_zlim(self.z_rand[0], self.z_rand[1])

        
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.pause(0.01)

    @staticmethod

    def plot_cube(ax, x, y, z, sizex, sizey, sizez, color='b', alpha=0.3):
        ax.bar3d(
        x - sizex / 2, y - sizey / 2, z - sizez / 2,  # centro del cubo
        sizex, sizey, sizez,                          # dimensiones del cubo
        color=color, alpha=alpha, edgecolor='k')
    
    @staticmethod

    def plot_sphere(ax, x, y, z, r):
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        xs = x + r * np.cos(u) * np.sin(v)
        ys = y + r * np.sin(u) * np.sin(v)
        zs = z + r * np.cos(v)
        ax.plot_wireframe(xs, ys, zs, color="b", linewidth=0.5)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 +
                 (node.y - rnd_node.y)**2 +
                 (node.z - rnd_node.z)**2 for node in node_list]
        return dlist.index(min(dlist))

    @staticmethod
    def check_collision(node, obstacle_list):
        if node is None:
            return False

        for (ox, oy, oz, sizex, sizey, sizez) in obstacle_list:
            safety_margin = 0.01  # márgen de seguridad
            x_min = ox - sizex / 2 - safety_margin
            x_max = ox + sizex / 2 + safety_margin
            y_min = oy - sizey / 2 - safety_margin
            y_max = oy + sizey / 2 + safety_margin
            z_min = oz - sizez / 2 - safety_margin
            z_max = oz + sizez / 2 + safety_margin
    
            for x, y, z in zip(node.path_x, node.path_y, node.path_z):
                if (x_min <= x <= x_max) and (y_min <= y <= y_max) and (z_min <= z <= z_max):
                    return False  # Hay colisión
        return True  # No hay colisión


    @staticmethod
    def calc_distance_and_angles(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dz = to_node.z - from_node.z
        d = np.sqrt(dx**2 + dy**2 + dz**2)
        theta = np.arctan2(dy, dx)
        phi = np.arccos(dz / d) if d != 0 else 0
        return d, theta, phi
    


def get_path_length_3d(path):
    le = 0
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        dz = path[i + 1][2] - path[i][2]
        d = np.sqrt(dx**2 + dy**2 + dz**2)
        le += d
    return le


def get_target_point_3d(path, targetL):
    le = 0
    ti = 0
    lastPairLen = 0
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        dz = path[i + 1][2] - path[i][2]
        d = np.sqrt(dx**2 + dy**2 + dz**2)
        le += d
        if le >= targetL:
            ti = i - 1
            lastPairLen = d
            break

    partRatio = (le - targetL) / lastPairLen
    x = path[ti][0] + (path[ti + 1][0] - path[ti][0]) * partRatio
    y = path[ti][1] + (path[ti + 1][1] - path[ti][1]) * partRatio
    z = path[ti][2] + (path[ti + 1][2] - path[ti][2]) * partRatio
    return [x, y, z, ti]



def line_collision_check_3d_box(p1, p2, obstacle_list, num_points=10):
    for i in range(num_points + 1):
        t = i / num_points
        x = p1[0] + (p2[0] - p1[0]) * t
        y = p1[1] + (p2[1] - p1[1]) * t
        z = p1[2] + (p2[2] - p1[2]) * t

        for (ox, oy, oz, sizex, sizey, sizez) in obstacle_list:
            in_x = (ox - sizex / 2) <= x <= (ox + sizex / 2)
            in_y = (oy - sizey / 2) <= y <= (oy + sizey / 2)
            in_z = (oz - sizez / 2) <= z <= (oz + sizez / 2)
            if in_x and in_y and in_z:
                return False  # Colisión detectada

    return True  # Sin colisión




def path_smoothing_3d(path, max_iter, obstacle_list):
    le = get_path_length_3d(path)
    for _ in range(max_iter):
        pickPoints = [random.uniform(0, le), random.uniform(0, le)]
        pickPoints.sort()
        first = get_target_point_3d(path, pickPoints[0])
        second = get_target_point_3d(path, pickPoints[1])

        if first[3] <= 0 or second[3] <= 0:
            continue
        if (second[3] + 1) > len(path):
            continue
        if second[3] == first[3]:
            continue

        if not line_collision_check_3d_box(first, second, obstacle_list):
            continue

        newPath = []
        newPath.extend(path[:first[3] + 1])
        newPath.append(first[:3])

        newPath.append(second[:3])
        newPath.extend(path[second[3] + 1:])
        path = newPath
        le = get_path_length_3d(path)

    return path
    
   

def transform(local, model_pose):
    x_rel, y_rel, z_rel = local
    x_model, y_model, z_model, _, _, yaw = model_pose
    # Aplicar rotación yaw (en el plano XY)
    x_rot = np.cos(yaw)*x_rel - np.sin(yaw)*y_rel
    y_rot = np.sin(yaw)*x_rel + np.cos(yaw)*y_rel
    # Sumar traslación
    x_global = x_model + x_rot
    y_global = y_model + y_rot
    z_global = z_model + z_rel
    return (x_global, y_global, z_global)


# Rotar vector según eje y ángulo
def rotate_vector(v, angle_deg, axis):
    angle_rad = np.radians(angle_deg)
    x, y, z = v
    if axis == 'x':
        return (
            x,
            y * np.cos(angle_rad) - z * np.sin(angle_rad),
            y * np.sin(angle_rad) + z * np.cos(angle_rad)
        )
    elif axis == 'y':
        return (
            x * np.cos(angle_rad) + z * np.sin(angle_rad),
            y,
            -x * np.sin(angle_rad) + z * np.cos(angle_rad)
        )
    elif axis == 'z':
        return (
            x * np.cos(angle_rad) - y * np.sin(angle_rad),
            x * np.sin(angle_rad) + y * np.cos(angle_rad),
            z
        )

# Rotar y trasladar un bloque
def rotate_and_translate_block(pos, dims, center, target_pos):
    # 1. Trasladar al origen
    rel_pos = tuple(np.subtract(pos, center))

    # 2. Rotaciones: primero Y, luego Z
    rel_pos = rotate_vector(rel_pos, -90, 'y')
    rel_pos = rotate_vector(rel_pos, 90, 'z')

    # 3. Trasladar al target
    final_pos = tuple(np.add(rel_pos, target_pos))

    # 4. Rotar dimensiones igual
    dx, dy, dz = dims
    dims_rot = rotate_vector((dx, dy, dz), -90, 'y')
    dims_rot = rotate_vector(dims_rot, 90, 'z')
    dims_rot = tuple(map(abs, dims_rot))

    return final_pos + dims_rot
