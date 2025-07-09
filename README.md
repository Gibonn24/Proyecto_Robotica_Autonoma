# Proyecto de Robótica Autónoma

Este repositorio contiene el material utilizado para simular y controlar un robot **UR5** equipado con la pinza **Robotiq 85** y una cámara de profundidad. Incluye modelos para Gazebo, archivos de descripción en URDF y un conjunto de scripts en Python que sirven como nodos de ROS para diferentes tareas de percepción y planificación.

## Requisitos

- **ROS Noetic** o compatible con Python 3.
- Un espacio de trabajo *catkin* configurado (por ejemplo `~/catkin_ws`).
- Dependencias de Python: `ultralytics`, `opencv-python`, `numpy`, `pcl`, `cv_bridge` y `rospy`.
- Paquetes ROS estándar para el control de robots y la simulación en Gazebo.

Para instalar las dependencias de Python puedes usar:

```bash
pip install ultralytics opencv-python numpy pcl
```

Las dependencias de ROS se listan en los archivos `dependencies.rosinstall` de las carpetas `robotiq` y `ur5`.

## Instalación

1. Ve al directorio `src` de tu espacio de trabajo y clona el repositorio:

   ```bash
   cd ~/catkin_ws/src
   git clone <este_repo>
   ```

2. Compila los paquetes:

   ```bash
   cd ~/catkin_ws
   catkin_make
   ```

3. Recarga tu entorno:

   ```bash
   source devel/setup.bash
   ```

## Estructura del Proyecto

- **robotiq** – modelo y plugins de la pinza Robotiq 85.
- **ur5/ur5_description** – archivos URDF y mallas del UR5 con la pinza.
- **ur5/ur5_gazebo** – lanzadores, mundos y scripts para Gazebo.
  - `scripts/functions.py` contiene utilidades de cinemática (directa, inversa, Jacobianos) y funciones auxiliares para planificación de trayectorias.
  - `scripts/markers.py` dibuja marcadores en RViz.
  - `scripts/detect_cans.py` ejecuta un detector YOLO para identificar latas usando la cámara del robot.
  - `scripts/scan_shelves.py` mueve el brazo por varias posiciones predefinidas, detecta latas y guarda las coordenadas transformadas en `detected_cans.csv`.
  - `scripts/send_gripper.py` y `scripts/send_joints.py` permiten enviar comandos simples al actuador y a las articulaciones.
  - `scripts/rrt_generate_path` genera una trayectoria mediante RRT y guarda los puntos suavizados en `smoothed_path.csv`.

## Uso Básico

### Lanzar la simulación

Para cargar el escenario base del UR5 en Gazebo:

```bash
roslaunch ur5_gazebo ur5_cubes.launch
```

El archivo `ur5_gazebo/launch/ur5_project.launch` puede utilizarse como punto de partida para proyectos personalizados.

### Detección de latas con YOLO

Ejecuta el nodo de detección:

```bash
rosrun ur5_gazebo detect_cans.py
```

El modelo YOLO por defecto se carga desde `ur5/ur5_gazebo/scripts/best.pt`, aunque puedes indicar otro con el parámetro `~model_path`.

### Escaneo de estanterías

Para recorrer posiciones predeterminadas y registrar las latas detectadas en un CSV:

```bash
rosrun ur5_gazebo scan_shelves.py```

Las posiciones de escaneo y el marco de referencia objetivo se pueden ajustar mediante parámetros ROS.

### Generación de trayectorias con RRT

El script `rrt_generate_path` calcula un camino en el espacio 3D evitando obstáculos y guarda el resultado suavizado en `smoothed_path.csv`. Puede modificarse para incluir la planificación en tu propio nodo.

### Control del gripper y de articulaciones

- Abrir o cerrar la pinza:

  ```bash
  rosrun ur5_gazebo send_gripper.py --value 0.5
  ```
  El valor varía entre `0.0` (cerrada) y `0.8` (abierta).

- Enviar una configuración articular simple:

  ```bash
  rosrun ur5_gazebo send_joints.py
  ```
  Edita el script para cambiar los valores deseados.

## Documentación de Funciones

El archivo `functions.py` agrupa numerosas utilidades que puedes reutilizar en tus propios nodos:

- **fkine_ur5** y **ikine_ur5** – cinemática directa e inversa del brazo.
- **jacobian_position** – cálculo del Jacobiano para el efector final.
- **path_smoothing_3d** y utilidades relacionadas con RRT para planificación.
- Funciones de transformación y manejo de cuaterniones.

Consulta el código para más detalles y ejemplos de uso.

## Licencia

Este proyecto se distribuye bajo la licencia que figura en cada paquete. Consulta los archivos `LICENSE` y `package.xml` correspondientes para más información.

