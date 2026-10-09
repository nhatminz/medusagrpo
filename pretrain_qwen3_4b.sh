#!/usr/bin/env bash
set -euo pipefail
MODEL_KEY=qwen3_4b
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/launch/pretrain_model.sh" "$@"
