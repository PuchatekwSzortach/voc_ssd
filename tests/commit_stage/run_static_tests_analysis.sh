#!/usr/bin/env bash

# Exit on error
set -e

source activate voc_ssd_environment

DIRECTORIES_TO_SCAN="tests"

echo "Running pylint..."
pylint $DIRECTORIES_TO_SCAN

echo "Running pycodestyle..."
pycodestyle $DIRECTORIES_TO_SCAN

echo "Running xenon..."
xenon . --max-absolute B
