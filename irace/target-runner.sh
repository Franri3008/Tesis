#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")"

../algorithm/__venv__/bin/python ../algorithm/metaheuristic.py "$@"