#!/usr/bin/env bash
set -euo pipefail
MODEL_KEY=qwen25_3b
METHOD=medusa_reflex
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/launch/train_model.sh" "$@"
