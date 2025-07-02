#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utiliza un robot UR5 con una cámara de profundidad para escanear estanterías,
detectar latas con YOLO, calcular su posición 3D y transformar sus coordenadas
a un marco de referencia global para su manipulación.
"""

import os
import csv
import rospy
import cv2
import tf2_ros  # Necesario para las transformaciones de coordenadas (TF)
import message_filters  # Necesario para sincronizar tópicos de la cámara

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
from geometry_msgs.msg import PointStamped  # Para trabajar con puntos con TF

# Importación segura de YOLO
try:
    from ultralytics import YOLO
except ImportError:
    rospy.logerr("El paquete 'ultralytics' no fue encontrado. "
                 "Por favor, instálalo con: pip install ultralytics")
    YOLO = None


class ShelfScanner:
    """
    Controla el escaneo, detección, cálculo 3D y transformación de coordenadas.
    """

    def __init__(self):
        """Inicializa todos los componentes del nodo ROS."""
        if YOLO is None:
            raise RuntimeError("Dependencia 'ultralytics' no está disponible. Abortando.")

        self.bridge = CvBridge()

        # --- Parámetros de ROS ---
        # Marco de referencia objetivo para las coordenadas finales (muy importante)
        self.target_frame = rospy.get_param('~target_frame', 'world')
        script_dir = os.path.dirname(os.path.realpath(__file__))
        default_model_path = os.path.join(script_dir, 'best.pt')
        model_path = rospy.get_param('~model_path', default_model_path)
        self.shelf_positions = rospy.get_param('~shelf_positions', self.get_default_positions())
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        # --- Inicialización del Modelo YOLO ---
        rospy.loginfo(f"Cargando modelo YOLO desde: {model_path}")
        self.model = YOLO(model_path)

        # --- Almacenamiento de Datos ---
        self.color_img = None
        self.depth_img = None
        self.camera_intrinsics = None  # Matriz intrínseca K
        self.last_header = None # Cabecera del mensaje para la marca de tiempo de TF
        self.detections = []  # Almacenará (marca, (x, y, z) en target_frame)

        # --- Listener de Transformaciones (TF) ---
        # ARREGLO: Se inicializa el buffer y el listener de TF2 para poder transformar coordenadas.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # --- Subscriptores de ROS ---
        # ARREGLO: Se eliminan los subscriptores individuales y se usa message_filters.
        # Esto garantiza que la imagen de color y de profundidad correspondan al mismo instante.
        self.info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.info_callback)
        color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/depth/image_raw', Image)

        # Sincronizador que llama a image_callback solo cuando tiene mensajes de ambos tópicos
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.image_callback)

        # --- Publicador de ROS ---
        self.joint_pub = rospy.Publisher('/trajectory_controller/command', JointTrajectory, queue_size=10)

        rospy.loginfo(f"ShelfScanner inicializado. Las coordenadas se guardarán en el marco '{self.target_frame}'.")

    def get_default_positions(self) -> list:
        """Define las posiciones de escaneo si no se proporcionan como parámetro."""
        # Limpiado para mayor claridad
        return [
            # Estantería 1
            [1.57, -2.4, 2.5, 3.14, -1.5, 3.14] ,  # bottom 1
            [1.10, -2.4, 2.5, 3.14, -1.5, 3.14],  # middle 1
            [1.57, -2.4, 2.2, 3.14, -1.5, 3.14] ,  # top
            [1.10, -2.4, 2.2, 3.14, -1.5, 3.14] ,  # bottom 1
            [1.57, -2.0, 1.5, 3.14, -1.5, 3.14] ,  # middle 1
            [1.10, -2.0, 1.5, 3.14, -1.5, 3.14] ,  # top
            # Bookshelf 2
            [3.14, -2.4, 2.5, 3.14, -1.5, 3.14],   # bottom
            [2.67, -2.4, 2.5, 3.14, -1.5, 3.14],   # bottom 2
            [3.14, -2.4, 2.2, 3.14, -1.5, 3.14],  # middle1
            [2.67, -2.4, 2.2, 3.14, -1.5, 3.14],  # middle2
            [3.14, -2.0, 1.5, 3.14, -1.5, 3.14],  # top
            [2.67, -2.0, 1.5, 3.14, -1.5, 3.14],  # top2
            # Bookshelf 3
            [-1.57, -2.4, 2.5, 3.14, -1.5, 3.14],  # bottom
            [-1.10, -2.4, 2.5, 3.14, -1.5, 3.14],   # bottom2
            [-1.57, -2.4, 2.2, 3.14, -1.5, 3.14],  # middle
            [-1.10, -2.4, 2.2, 3.14, -1.5, 3.14],  # middle2
            [-1.57, -2.0, 1.5, 3.14, -1.5, 3.14],  # top
            [-1.10, -2.0, 1.5, 3.14, -1.5, 3.14],  # top2
        ]

    def info_callback(self, msg: CameraInfo):
        """Almacena los parámetros intrínsecos de la cámara y se desuscribe."""
        if self.camera_intrinsics is None:
            self.camera_intrinsics = msg.K
            rospy.loginfo("Parámetros intrínsecos de la cámara recibidos.")
            self.info_sub.unregister()

    def image_callback(self, color_msg: Image, depth_msg: Image):
        """Callback sincronizado. Almacena las imágenes y la cabecera."""
        try:
            self.color_img = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            self.depth_img = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            # Guardamos la cabecera para saber el frame_id y timestamp de la cámara
            self.last_header = color_msg.header
        except CvBridgeError as e:
            rospy.logerr(f"Error de CvBridge: {e}")

    def move_arm(self, joint_positions: list):
        """Mueve el brazo a una posición articular de forma más segura."""
        traj = JointTrajectory()
        traj.header = Header(stamp=rospy.Time.now())
        traj.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        # ARREGLO: Aumentado el tiempo para un movimiento más suave y seguro.
        point.time_from_start = rospy.Duration(5.0)
        traj.points.append(point)
        self.joint_pub.publish(traj)
        rospy.loginfo(f"Moviendo brazo a la posición: {joint_positions}")

    def process_frame(self):
        """
        Detecta objetos, calcula sus coordenadas 3D en el marco de la cámara
        y las transforma al marco de referencia global.
        """
        if any(v is None for v in [self.color_img, self.depth_img, self.camera_intrinsics, self.last_header]):
            rospy.logwarn("Esperando datos sincronizados de la cámara (color, profundidad, info).")
            return

        rospy.loginfo("Procesando cuadro...")
        results = self.model(self.color_img, verbose=False)
        
        fx, fy = self.camera_intrinsics[0], self.camera_intrinsics[4]
        cx, cy = self.camera_intrinsics[2], self.camera_intrinsics[5]

        for r in results:
            for box in r.boxes:
                u, v = (int((box.xyxy[0][0] + box.xyxy[0][2]) / 2),
                        int((box.xyxy[0][1] + box.xyxy[0][3]) / 2))
                
                try:
                    depth_mm = self.depth_img[v, u]
                    if depth_mm == 0:  # Lectura inválida
                        continue
                except IndexError:
                    continue # El centroide está fuera de la imagen de profundidad
                
                depth_m = float(depth_mm) / 1000.0
                
                # --- 1. Cálculo 3D en el marco de la cámara ---
                # Este cálculo es correcto según el modelo de cámara pinhole.
                X_cam = (u - cx) * depth_m / fx
                Y_cam = (v - cy) * depth_m / fy
                Z_cam = depth_m

                # --- 2. Transformación de Coordenadas (TF) ---
                # ARREGLO: Transforma el punto del marco de la cámara al marco global.
                camera_frame = self.last_header.frame_id
                
                # Creamos un punto con su cabecera correcta
                point_in_camera_frame = PointStamped()
                point_in_camera_frame.header = self.last_header
                point_in_camera_frame.point.x = X_cam
                point_in_camera_frame.point.y = Y_cam
                point_in_camera_frame.point.z = Z_cam
                
                try:
                    # Usamos el buffer de TF para transformar el punto
                    point_in_target_frame = self.tf_buffer.transform(point_in_camera_frame, self.target_frame, rospy.Duration(1.0))
                    
                    brand = self.model.names[int(box.cls)]
                    x_world, y_world, z_world = point_in_target_frame.point.x, point_in_target_frame.point.y, point_in_target_frame.point.z
                    
                    self.detections.append((brand, (x_world, y_world, z_world)))
                    rospy.loginfo(f"Detectada '{brand}' en {self.target_frame} en ({x_world:.3f}, {y_world:.3f}, {z_world:.3f})")

                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    rospy.logwarn(f"No se pudo transformar la coordenada del frame '{camera_frame}' a '{self.target_frame}': {e}")

    def scan(self):
        """Ejecuta la rutina completa de escaneo."""
        # ARREGLO: Limpiar detecciones de un escaneo anterior.
        self.detections.clear()

        while self.camera_intrinsics is None and not rospy.is_shutdown():
            rospy.loginfo("Esperando los parámetros intrínsecos de la cámara...")
            rospy.sleep(1.0)
        
        if rospy.is_shutdown(): return

        for pos in self.shelf_positions:
            if rospy.is_shutdown(): break
            self.move_arm(pos)
            # ARREGLO: La pausa debe ser mayor que el tiempo del movimiento.
            rospy.sleep(5.5)
            self.process_frame()
            rospy.sleep(0.5)

        self.save_results_to_csv()

    def save_results_to_csv(self):
        """Guarda las detecciones finales en un archivo CSV."""
        if not self.detections:
            rospy.logwarn("No se detectaron latas para guardar.")
            return

        out_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'detected_cans.csv')
        try:
            with open(out_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['brand', f'x_{self.target_frame}', f'y_{self.target_frame}', f'z_{self.target_frame}'])
                for brand, (x, y, z) in self.detections:
                    writer.writerow([brand, f'{x:.3f}', f'{y:.3f}', f'{z:.3f}'])
            rospy.loginfo(f"Resultados guardados en {out_path}")
        except IOError as e:
            rospy.logerr(f"Error al escribir el archivo CSV: {e}")

def main():
    try:
        rospy.init_node('shelf_scanner_node')
        scanner = ShelfScanner()
        rospy.sleep(2.0) # Pausa para que se establezcan las conexiones
        scanner.scan()
        rospy.loginfo("Escaneo completado.")
    except (rospy.ROSInterruptException, RuntimeError) as e:
        rospy.logerr(f"Proceso detenido por error: {e}")

if __name__ == '__main__':
    main()