#!/usr/bin/env bash
set -euo pipefail
MODEL_KEY=llama31_8b
METHOD=medusa
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/launch/train_model.sh" "$@"
