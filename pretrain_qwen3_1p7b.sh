#!/usr/bin/env bash
set -euo pipefail
MODEL_KEY=qwen3_1p7b
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/launch/pretrain_model.sh" "$@"
