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

# This script generates a latency lookup table for a model using the hardware configuration
# Usage: bash run_hardware_config.sh [model_name] [max_bit] [layer_dims_path] [hardware_config_path] [output_path]
# Example: bash run_hardware_config.sh qvgg16 8 /path/to/vgg16_layer_dimensions.yaml /path/to/hardware_config.yaml /path/to/output_table.npy

# Get the directory of the script and the repository root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Default values with command line overrides
MODEL_NAME=${1:-"qvgg16"}
MAX_BIT=${2:-8}
LAYER_DIMS_PATH=${3:-"${REPO_ROOT}/lib/simulator/vgg16_layer_dimensions.yaml"}
HARDWARE_CONFIG_PATH=${4:-"${REPO_ROOT}/lib/simulator/hardware_config.yaml"}
OUTPUT_PATH=${5:-"${REPO_ROOT}/lib/simulator/lookup_tables/${MODEL_NAME}_batch1_latency_table.npy"}

echo "Generating latency lookup table for ${MODEL_NAME}..."
echo "Using layer dimensions: ${LAYER_DIMS_PATH}"
echo "Using hardware config: ${HARDWARE_CONFIG_PATH}"
echo "Maximum bit width: ${MAX_BIT}"

# Check if required files exist
if [ ! -f "$HARDWARE_CONFIG_PATH" ]; then
  echo "Error: Hardware config file not found at $HARDWARE_CONFIG_PATH"
  exit 1
fi

if [ ! -f "$LAYER_DIMS_PATH" ]; then
  echo "Error: Layer dimensions file not found at $LAYER_DIMS_PATH"
  exit 1
fi

# Run the latency table generation
echo "Generating latency lookup table..."
python ${REPO_ROOT}/lib/simulator/create_custom_latency_table.py \
  --model_name $MODEL_NAME \
  --max_bit $MAX_BIT \
  --layer_dims_yaml $LAYER_DIMS_PATH \
  --hardware_config_yaml $HARDWARE_CONFIG_PATH \
  --output_path "$OUTPUT_PATH"

# Check if latency table generation succeeded
if [ $? -ne 0 ]; then
  echo "Error: Latency table generation failed."
  exit 1
fi

echo "Latency table generation complete."
echo "The table is saved at: $OUTPUT_PATH"
