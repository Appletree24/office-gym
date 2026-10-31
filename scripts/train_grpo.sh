#!/usr/bin/env bash
set -euo pipefail
exec python -m officegym.training.launch "$@"
