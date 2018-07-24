#!/usr/bin/env bash

set -e

bash ./tests/commit_stage/run_unit_tests.sh
bash ./tests/commit_stage/run_static_tests_analysis.sh
