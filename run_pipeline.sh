#!/usr/bin/env bash
set -e
export PYTHONPATH=src
python -m churn.download
python -m churn.train
python -m churn.monitor
pytest -q
