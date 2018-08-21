#!/usr/bin/env bash

set -e

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate voc_ssd_environment

py.test
