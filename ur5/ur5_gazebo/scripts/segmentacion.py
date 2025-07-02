#!/usr/bin/env python3
import argparse
import os
import sys
import pcl
def segment_plane(input_path: str, output_path: str, distance_threshold: float, keep_inliers: bool) -> None:
    if not os.path.isfile(input_path):
        sys.stderr.write(f"Input file not found: {input_path}\n")
        sys.exit(1)

    cloud = pcl.load_XYZRGB(input_path)

    seg = cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(distance_threshold)

    indices, model = seg.segment()

    if len(indices) == 0:
        sys.stderr.write("No se encontró un plano — abortando.\n")
        sys.exit(1)

    if keep_inliers:
        extracted = cloud.extract(indices, negative=False)
    else:
        extracted = cloud.extract(indices, negative=True)

    pcl.save(extracted, output_path)
    tipo = "inliers" if keep_inliers else "outliers"
    print(f"Guardado '{output_path}' ({tipo}, dist máx = {distance_threshold:.4f} m)")


def parse_args():
    p = argparse.ArgumentParser(description="Segmentación de plano por RANSAC (PCL)")
    p.add_argument('--input',  default='mesa_passthrough.pcd', help='Archivo PCD de entrada')
    p.add_argument('--output', default='mesa_segmented.pcd',    help='Archivo PCD de salida')
    p.add_argument('--max_distance', type=float, default=0.01, help='Distancia máx al plano (m) [default: 0.01]')
    p.add_argument('--inliers', action='store_true', help='Guardar puntos que pertenecen al plano (inliers)')
    p.add_argument('--outliers', action='store_true', help='Guardar puntos que NO pertenecen al plano (outliers)')
    return p.parse_args()



args = parse_args()

if args.inliers == args.outliers:
    print("[✗] Debe especificar solo --inliers o solo --outliers")
    sys.exit(1)

keep_inliers = args.inliers
segment_plane(args.input, args.output, args.max_distance, keep_inliers)
