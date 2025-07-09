#!/usr/bin/env python3
# ========================================
# Clustering Euclideano
# ========================================

# Conversión de XYZRGB a XYZ (K-d tree requiere solo información espacial)
white_cloud = XYZRGB_to_XYZ(extracted_outliers)
# K-d tree requiere el formato XYZ
tree = white_cloud.make_kdtree()

# Objeto que se usará para la extracción del cluster
cluster = white_cloud.make_EuclideanClusterExtraction()

# Tolerancias para threshold de distancia, tamaño mín y max del cluster
# Se puede modificar estos parámetros para mejores resultados
cluster.set_ClusterTolerance(0.001)
cluster.set_MinClusterSize(10)
cluster.set_MaxClusterSize(250)
# Buscar clusters en el k-d tree
cluster.set_SearchMethod(tree)

# Extraer los índices para cada cluster descubierto (cada cluster tiene un índice diferente)
cluster_indices = cluster.Extract()
    

# Visualización del cluster: color a cada objeto segmentado
cluster_color = get_color_list(len(cluster_indices))
color_cluster_point_list = []
for j, indices in enumerate(cluster_indices):
    for i, indice in enumerate(indices):
        color_cluster_point_list.append([white_cloud[indice][0],
                                         white_cloud[indice][1],
                                         white_cloud[indice][2],
                                         rgb_to_float(cluster_color[j])])
        
# Crear una nueva nube conteniendo todos los clusters, cada una con un color único
cluster_cloud = pcl.PointCloud_PointXYZRGB()
cluster_cloud.from_list(color_cluster_point_list)
