#!/usr/bin/env bash
set -euo pipefail
MODEL_KEY=qwen25_3b
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/launch/pretrain_model.sh" "$@"
