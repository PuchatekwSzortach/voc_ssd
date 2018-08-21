#!/usr/bin/env bash

# Exit on error
set -e

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate voc_ssd_environment

DIRECTORIES_TO_SCAN="net scripts tests"

echo "Running pylint..."
pylint $DIRECTORIES_TO_SCAN

echo "Running pycodestyle..."
pycodestyle $DIRECTORIES_TO_SCAN

echo "Running xenon..."
xenon . --max-absolute B
