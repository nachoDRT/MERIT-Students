#!/bin/bash
set -e

STEP="${PIPELINE_STEP:-students}"

case "$STEP" in
  students)
    python /app/src/students.py
    ;;
  visual_extract_vectors)
    python /app/src/experiments/visual/extract_vectors.py
    ;;
  visual_extract_student_vectors)
    python /app/src/experiments/visual/extract_student_vectors.py
    ;;
  visual_layer_sweep)
    python /app/src/experiments/visual/layer_sweep.py
    ;;
  baseline_sweep)
    python /app/src/experiments/visual/baseline_sweep.py
    ;;
  layer_sweep_students)
    python /app/src/experiments/visual/layer_sweep_students.py
    ;;
  plot_activation_norms)
    python /app/src/experiments/visual/plot_activation_norms.py
    ;;
  *)
    echo "Unknown PIPELINE_STEP: $STEP"
    echo "Valid options: students | visual_extract_vectors | visual_extract_student_vectors | visual_layer_sweep | baseline_sweep | layer_sweep_students | plot_activation_norms"
    exit 1
    ;;
esac
