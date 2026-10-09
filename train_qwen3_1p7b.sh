#!/usr/bin/env bash
set -euo pipefail
MODEL_KEY=qwen3_1p7b
METHOD=medusa_reflex
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/launch/train_model.sh" "$@"
