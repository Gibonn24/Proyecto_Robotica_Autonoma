# Proyecto_Robotica_Autonoma

Este repositorio contiene los archivos de simulación para un robot UR5 con
pinza Robotiq 85. Ahora el modelo incluye una cámara de profundidad montada
encima del gripper para facilitar la detección de objetos durante las pruebas
en Gazebo.

## Detección de latas con YOLO

El script `ur5/ur5_gazebo/scripts/detect_cans.py` utiliza la cámara del UR5 y un
modelo entrenado (`best.pt`) para detectar latas de Coca-Cola, Pepsi y Sprite en
tiempo real. Para ejecutarlo asegúrate de tener instaladas las dependencias de
`ultralytics`, `opencv` y `cv_bridge`.

```bash
rosrun ur5_gazebo detect_cans.py
```

El modelo se carga por defecto desde el mismo directorio del script, pero puede
especificarse otro archivo utilizando el parámetro `model_path`.
