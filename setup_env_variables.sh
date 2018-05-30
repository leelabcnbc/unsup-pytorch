#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

TANG_PATH="${DIR}"
export PYTHONPATH="${TANG_PATH}":"${PYTHONPATH}"
