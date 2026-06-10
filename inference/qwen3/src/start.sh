#!/bin/bash
set -e

STEP="${PIPELINE_STEP:-students}"

case "$STEP" in
  students)
    python /app/src/students.py
    ;;
  plot_bias)
    python /app/src/plot_bias_sweep.py
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
  sae_download)
    python /app/src/experiments/sae_decomposition/download_sae.py
    ;;
  sae_validate)
    python /app/src/experiments/sae_decomposition/validate_sae.py
    ;;
  sae_decompose)
    python /app/src/experiments/sae_decomposition/decompose_vector.py
    ;;
  sae_interpret)
    python /app/src/experiments/sae_decomposition/feature_interpretation.py
    ;;
  sae_ablation)
    python /app/src/experiments/sae_decomposition/ablation_steering.py
    ;;
  cache_activations)
    python /app/src/experiments/sae_decomposition/cache_activations.py
    ;;
  analyze_conditions)
    python /app/src/experiments/sae_decomposition/analyze_conditions.py
    ;;
  *)
    echo "Unknown PIPELINE_STEP: $STEP"
    echo "Valid options: students | plot_bias | visual_extract_vectors | visual_extract_student_vectors | visual_layer_sweep | baseline_sweep | layer_sweep_students | plot_activation_norms | sae_download | sae_validate | sae_decompose | sae_interpret | sae_ablation | cache_activations | analyze_conditions"
    exit 1
    ;;
esac
