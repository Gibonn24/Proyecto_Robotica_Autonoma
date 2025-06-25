# Proyecto_Robotica_Autonoma

Este repositorio contiene los archivos de simulación para un robot UR5 con
pinza Robotiq 85. Ahora el modelo incluye una cámara de profundidad montada
encima del gripper para facilitar la detección de objetos durante las pruebas
en Gazebo.

## Detección de latas en tiempo real

El script `ur5/ur5_gazebo/scripts/detect_cans.py` suscribe la imagen de la cámara
montada en el gripper y aplica un modelo YOLO entrenado para detectar latas de
Coca-Cola, Pepsi y Sprite. El modelo debe guardarse como `best.pt` en la misma
carpeta que el script.

Para ejecutarlo primero inicia la simulación y luego en otra terminal corre:

```bash
rosrun ur5_gazebo detect_cans.py
```

Aparecerá una ventana con las detecciones en tiempo real.
