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
  classify_gallery)
    python /app/src/classify_gallery.py ${SUBJECTS:-}
    ;;
  scholarship_bias)
    python /app/src/experiments/scholarship/scholarship_bias.py
    ;;
  scholarship_score)
    python /app/src/experiments/scholarship/scholarship_score.py
    ;;
  visual_extract_vectors)
    python /app/src/experiments/visual/extract_vectors.py
    ;;
  visual_extract_student_vectors)
    python /app/src/experiments/visual/extract_student_vectors.py
    ;;
  extract_filtered_vectors)
    python /app/src/experiments/visual/extract_filtered_vectors.py
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
  layer_sweep_students_batch)
    # Run the layer sweep for several subjects in series, each with its own
    # steering vector (subject_<id>_neg_to_subject_8_pos). Other knobs
    # (N_IMAGES, BETA, LAYER_START, LAYER_END, VERDICT_FILTER) come from env.
    for s in ${SUBJECTS:-2 3 7 11 12}; do
      case "$s" in subject_*) sub="$s";; *) sub="subject_$s";; esac
      export SUBJECT="$sub"
      export VECTOR_PAIR="${sub}_neg_to_subject_8_pos"
      echo "==================================================================="
      echo "=== layer sweep: $sub  (vector $VECTOR_PAIR) ==="
      echo "==================================================================="
      python /app/src/experiments/visual/layer_sweep_students.py
    done
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
    echo "Valid options: students | plot_bias | classify_gallery | scholarship_bias | scholarship_score | visual_extract_vectors | visual_extract_student_vectors | visual_layer_sweep | baseline_sweep | layer_sweep_students | layer_sweep_students_batch | extract_filtered_vectors | plot_activation_norms | sae_download | sae_validate | sae_decompose | sae_interpret | sae_ablation | cache_activations | analyze_conditions"
    exit 1
    ;;
esac
