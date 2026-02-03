#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
./run_train_v1.7_v1.10_s456.sh
./run_train_v1.7_v1.10_s789.sh
./run_train_v1.7_v1.10_s42.sh
