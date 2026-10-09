#!/usr/bin/env bash
set -euo pipefail
MODEL_KEY=qwen3_1p7b
# Resolve DATA_ROOT/dataset paths in the shared launcher after loading defaults.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/launch/pretrain_model.sh" "$@"
