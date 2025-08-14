#!/bin/bash
##############################################################################
# Copyright (C) 2025 Joel Klein                                              #
# All Rights Reserved                                                        #
#                                                                            #
# This work is licensed under the terms described in the LICENSE file        #
# found in the root directory of this source tree.                           #
##############################################################################

# Exit on any error, undefined variable, or pipe failure
set -euo pipefail

# Get the directory of the script and the repository root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# This script handles RL-based quantization search:
# 1. Generate hardware latency lookup table
# 2. Preserve analysis files
# 3. Run RL-based search for mixed precision using INT8 model as starting point
# 4. Generate detailed cost analysis results

# Usage: bash run_rl_quantize.sh [quant_model] [dataset] [dataset_root] [max_accuracy_drop] [min_bit] [max_bit] [train_episodes] [search_finetune_epochs] [force_first_last_layer] [consider_cell_resolution] [output_suffix] [finetune_lr] [uniform_model_file] [wandb_enable] [wandb_project] [gpu_id] [batch_size] [num_workers] [amp_enable] [hardware_config_path] [layer_dims_path] [lookup_table_path]
# Example: bash run_rl_quantize.sh qvgg16 imagenet100 /path/to/dataset 1.0 2 8 600 3 true false accdrop1.0bit28 0.001 /path/to/uniform/model.pth.tar false cim-aq-quantization 1 256 32 true /path/to/hardware.yaml /path/to/layer_dims.yaml /path/to/lookup_table.npy

# Default values
QUANT_MODEL=${1:-"qvgg16"}                                                                                        # Quantized model architecture
DATASET=${2:-"imagenet100"}                                                                                       # Dataset name
DATASET_ROOT=${3:-"${REPO_ROOT}/data/imagenet100"}                                                                # Path to dataset
MAX_ACCURACY_DROP=${4:-"1.0"}                                                                                     # Maximum allowed accuracy drop in percentage points
MIN_BIT=${5:-"1"}                                                                                                 # Minimum bit-width
MAX_BIT=${6:-"8"}                                                                                                 # Maximum bit-width
TRAIN_EPISODES=${7:-"600"}                                                                                        # Number of training episodes for RL search
SEARCH_FINETUNE_EPOCHS=${8:-"3"}                                                                                  # Number of epochs for fine-tuning during search
FORCE_FIRST_LAST_LAYER=${9:-"true"}                                                                               # Force first and last layers to high precision
CONSIDER_CELL_RESOLUTION=${10:-"false"}                                                                           # Whether to consider cell resolution for possible weight bit widths
OUTPUT_SUFFIX=${11:-"accdrop${MAX_ACCURACY_DROP}bit${MIN_BIT}${MAX_BIT}"}                                         # Output suffix for model saving
FINETUNE_LR=${12:-"0.001"}                                                                                        # Learning rate for fine-tuning during RL search
UNIFORM_MODEL_FILE="${13:-"${REPO_ROOT}/checkpoints/${QUANT_MODEL}_per-tensor_uniform_8bit/model_best.pth.tar"}"  # Path to INT8 model file
WANDB_ENABLE=${14:-"false"}                                                                                       # Enable W&B logging
WANDB_PROJECT=${15:-"cim-aq-quantization"}                                                                        # W&B project name
GPU_ID=${16:-"1"}                                                                                                 # GPU ID(s) for CUDA_VISIBLE_DEVICES
BATCH_SIZE=${17:-"128"}                                                                                           # Batch size for RL training
NUM_WORKERS=${18:-"32"}                                                                                           # Number of DataLoader workers
AMP_ENABLE=${19:-"true"}                                                                                          # Enable Automatic Mixed Precision
HARDWARE_CONFIG_PATH=${20:-"${REPO_ROOT}/lib/simulator/hardware_config.yaml"}                                     # Custom hardware config path
BASE_MODEL_NAME=${QUANT_MODEL/q/}
LAYER_DIMS_PATH=${21:-"${REPO_ROOT}/lib/simulator/${BASE_MODEL_NAME}_layer_dimensions.yaml"}                      # Custom layer dimensions path
LOOKUP_TABLE_PATH=${22:-"${REPO_ROOT}/lib/simulator/lookup_tables/${QUANT_MODEL}_batch1_latency_table.npy"}       # Custom lookup table path

# Convert string to arg for Python scripts
if [[ "$FORCE_FIRST_LAST_LAYER" == "true" || "$FORCE_FIRST_LAST_LAYER" == "True" || "$FORCE_FIRST_LAST_LAYER" == "1" ]]; then
  FORCE_FIRST_LAST_CLI_ARG="--force_first_last_layer"
else
  FORCE_FIRST_LAST_CLI_ARG="--no-force_first_last_layer"
fi

if [[ "$CONSIDER_CELL_RESOLUTION" == "true" || "$CONSIDER_CELL_RESOLUTION" == "True" || "$CONSIDER_CELL_RESOLUTION" == "1" ]]; then
  CONSIDER_CELL_RESOLUTION_CLI_ARG="--consider_cell_resolution"
else
  CONSIDER_CELL_RESOLUTION_CLI_ARG="--no-consider_cell_resolution"
fi

if [[ "$WANDB_ENABLE" == "true" || "$WANDB_ENABLE" == "True" || "$WANDB_ENABLE" == "1" ]]; then
  WANDB_CLI_ARG="--wandb_enable"
else
  WANDB_CLI_ARG=""
fi

# Convert AMP enable string to CLI argument
if [[ "$AMP_ENABLE" == "true" || "$AMP_ENABLE" == "True" || "$AMP_ENABLE" == "1" ]]; then
  AMP_CLI_ARG="--amp"
else
  AMP_CLI_ARG=""
fi

echo "========================================================="
echo "Starting RL-based quantization search for $QUANT_MODEL"
echo "Dataset: $DATASET at $DATASET_ROOT"
echo "Maximum accuracy drop: $MAX_ACCURACY_DROP%"
echo "Bit-width range: $MIN_BIT-$MAX_BIT bits"
echo "RL training episodes: $TRAIN_EPISODES"
echo "Search fine-tuning epochs: $SEARCH_FINETUNE_EPOCHS"
echo "Force first/last layer high precision: $FORCE_FIRST_LAST_LAYER"
echo "Consider cell resolution: $CONSIDER_CELL_RESOLUTION"
echo "Output suffix: $OUTPUT_SUFFIX"
echo "Finetune learning rate: $FINETUNE_LR"
echo "GPU ID: $GPU_ID"
echo "AMP (Automatic Mixed Precision): $AMP_ENABLE"
echo "W&B logging: $WANDB_ENABLE"
if [ "$WANDB_ENABLE" = "true" ]; then
  echo "W&B project: $WANDB_PROJECT"
fi
echo "Repository root: $REPO_ROOT"
echo "Hardware config: $HARDWARE_CONFIG_PATH"
echo "Layer dimensions: $LAYER_DIMS_PATH"
echo "Lookup table: $LOOKUP_TABLE_PATH"
echo "========================================================="

# Check if INT8 model exists
if [ ! -f "$UNIFORM_MODEL_FILE" ]; then
  echo "Error: INT8 fine-tuned model not found at $UNIFORM_MODEL_FILE"
  echo "Please run INT8 pretraining first."
  exit 1
fi

# Create symlink to ensure RL-based search uses our 8-bit model
echo ""
echo "Creating symlink to 8-bit model for RL-based search..."
mkdir -p "${REPO_ROOT}/pretrained/imagenet"
ln -sf "$UNIFORM_MODEL_FILE" "${REPO_ROOT}/pretrained/imagenet/${QUANT_MODEL}.pth.tar"

# Step 1: Generate hardware latency lookup table
echo ""
echo "Step 1/4: Generating hardware latency lookup table..."
bash "${SCRIPT_DIR}/run_hardware_config.sh" $QUANT_MODEL $MAX_BIT $LAYER_DIMS_PATH $HARDWARE_CONFIG_PATH $LOOKUP_TABLE_PATH

# Check if hardware config generation succeeded
if [ $? -ne 0 ]; then
  echo "Error: Hardware configuration generation failed."
  exit 1
fi

# Step 2: Preserve analysis files immediately after creation
echo ""
echo "Step 2/4: Preserving analysis files to prevent overwrites during long-running search..."

# Define paths for preservation
OUTPUT_DIR="${REPO_ROOT}/save/${QUANT_MODEL}_${DATASET}_${OUTPUT_SUFFIX}_from_8bit"
mkdir -p "$OUTPUT_DIR"
PRESERVED_HARDWARE_CONFIG="${OUTPUT_DIR}/hardware_config.yaml"
PRESERVED_LAYER_DIMS="${OUTPUT_DIR}/${BASE_MODEL_NAME}_layer_dimensions.yaml"
PRESERVED_LOOKUP_TABLE="${OUTPUT_DIR}/${QUANT_MODEL}_batch1_latency_table.npy"

# Copy analysis files to preserve them from being overwritten during the long RL search
echo "Preserving analysis files in output directory..."

missing_files=()

if [ ! -f "$HARDWARE_CONFIG_PATH" ]; then
  missing_files+=("$HARDWARE_CONFIG_PATH")
fi

if [ ! -f "$LAYER_DIMS_PATH" ]; then
  missing_files+=("$LAYER_DIMS_PATH")
fi

if [ ! -f "$LOOKUP_TABLE_PATH" ]; then
  missing_files+=("$LOOKUP_TABLE_PATH")
fi

if [ -n "${missing_files[*]}" ]; then
  echo "Warning: Some analysis files are missing and cannot be preserved:"
  for file in "${missing_files[@]}"; do
    echo "  - $file"
  done
  echo "Detailed cost analysis may not be available."
else
  # All files exist, proceed with copying
  cp "$HARDWARE_CONFIG_PATH" "$PRESERVED_HARDWARE_CONFIG"
  if [ $? -ne 0 ]; then
    echo "Warning: Could not copy hardware_config.yaml"
  else
    echo "✓ Preserved $HARDWARE_CONFIG_PATH"
  fi

  cp "$LAYER_DIMS_PATH" "$PRESERVED_LAYER_DIMS"
  if [ $? -ne 0 ]; then
    echo "Warning: Could not copy layer dimensions file"
  else
    echo "✓ Preserved $LAYER_DIMS_PATH"
  fi

  cp "$LOOKUP_TABLE_PATH" "$PRESERVED_LOOKUP_TABLE"
  if [ $? -ne 0 ]; then
    echo "Warning: Could not copy lookup table"
  else
    echo "✓ Preserved $LOOKUP_TABLE_PATH"
  fi

  echo "All analysis files preserved successfully!"
fi

# Step 3: Run RL-based quantization search
echo ""
echo "Step 3/4: Running RL-based quantization search from 8-bit model..."
echo "Maximum accuracy drop allowed from 8-bit model: $MAX_ACCURACY_DROP%"

# rl_quantize.py always uses pretrained/imagenet/${MODEL}.pth.tar as the starting point
python "${REPO_ROOT}/rl_quantize.py" \
  --arch $QUANT_MODEL \
  --dataset $DATASET \
  --dataset_root $DATASET_ROOT \
  --suffix "${OUTPUT_SUFFIX}_from_8bit" \
  --orig_bit 8 \
  --max_bit $MAX_BIT \
  --min_bit $MIN_BIT \
  --n_worker $NUM_WORKERS \
  --data_bsize $BATCH_SIZE \
  --train_size 20000 \
  --val_size 10000 \
  --acc_drop $MAX_ACCURACY_DROP \
  $CONSIDER_CELL_RESOLUTION_CLI_ARG \
  $FORCE_FIRST_LAST_CLI_ARG \
  --finetune_epoch $SEARCH_FINETUNE_EPOCHS \
  --finetune_lr $FINETUNE_LR \
  --finetune_gamma 0.9 \
  --warmup 30 \
  --init_delta 0.6 \
  --delta_decay 0.997 \
  --train_episode $TRAIN_EPISODES \
  --lr_a 5e-5 \
  --lr_c 1e-3 \
  --bsize 64 \
  --rmsize 128 \
  --gpu_id $GPU_ID \
  $AMP_CLI_ARG \
  $WANDB_CLI_ARG \
  --wandb_project "$WANDB_PROJECT" \
  --hardware_config_path "$HARDWARE_CONFIG_PATH" \
  --lookup_table_path "$LOOKUP_TABLE_PATH"

# Check if RL search succeeded
if [ $? -ne 0 ]; then
  echo "Error: RL-based quantization search failed."
  exit 1
fi

# Check if best strategy file exists
STRATEGY_FILE="${REPO_ROOT}/save/${QUANT_MODEL}_${DATASET}_${OUTPUT_SUFFIX}_from_8bit/best_policy.npy"
if [ ! -f "$STRATEGY_FILE" ]; then
  echo "Error: Strategy file not found. Search may have failed."
  exit 1
fi

# Step 4: Generate detailed results using preserved files
echo ""
echo "Step 4/4: Generating detailed cost analysis using preserved files..."

# Run detailed cost analysis using the preserved files
echo "Running detailed cost analysis..."

ANALYSIS_OUTPUT_DIR="${OUTPUT_DIR}/analysis_results"
mkdir -p "$ANALYSIS_OUTPUT_DIR"

# Check if we have all required files for analysis
if [ -f "$PRESERVED_HARDWARE_CONFIG" ] && [ -f "$PRESERVED_LAYER_DIMS" ] && [ -f "$PRESERVED_LOOKUP_TABLE" ]; then
  echo "All required files available. Running comprehensive analysis..."

  python "${REPO_ROOT}/lib/simulator/get_cost_from_lookup_table.py" \
    --strategy "$STRATEGY_FILE" \
    --lookup_table "$PRESERVED_LOOKUP_TABLE" \
    --hardware_config_yaml "$PRESERVED_HARDWARE_CONFIG" \
    --layer_dims_yaml "$PRESERVED_LAYER_DIMS" \
    --save_results "$ANALYSIS_OUTPUT_DIR"
elif [ -f "$PRESERVED_LOOKUP_TABLE" ]; then
  echo "Running basic analysis with lookup table only..."

  python "${REPO_ROOT}/lib/simulator/get_cost_from_lookup_table.py" \
    --strategy "$STRATEGY_FILE" \
    --lookup_table "$PRESERVED_LOOKUP_TABLE" \
    --save_results "$ANALYSIS_OUTPUT_DIR"
else
  echo "Warning: Cannot run analysis - lookup table not available."
  echo "Manual analysis may be possible if files exist in lib/simulator/ directory."
  echo "Continuing without detailed cost analysis."
fi

# Check if analysis succeeded
if [ $? -eq 0 ]; then
  echo "✓ Detailed cost analysis completed successfully!"
  echo "Analysis results saved in: $ANALYSIS_OUTPUT_DIR"
  echo "Full results available in: ${ANALYSIS_OUTPUT_DIR}/results.txt"
  echo "CSV data available in: ${ANALYSIS_OUTPUT_DIR}/results.csv"
else
  echo "Warning: Detailed cost analysis failed, but RL search completed successfully."
  echo "You can manually run the analysis later using the preserved files:"
  echo "  Hardware config: $PRESERVED_HARDWARE_CONFIG"
  echo "  Layer dimensions: $PRESERVED_LAYER_DIMS"
  echo "  Lookup table: $PRESERVED_LOOKUP_TABLE"
  echo "  Strategy: $STRATEGY_FILE"
fi

echo ""
echo "========================================================="
echo "RL-based quantization search complete!"
echo "Mixed precision strategy: $STRATEGY_FILE"
echo "Preserved analysis files in: $OUTPUT_DIR"
if [ -d "$ANALYSIS_OUTPUT_DIR" ]; then
  echo "Detailed cost analysis: $ANALYSIS_OUTPUT_DIR"
fi
echo "========================================================="
