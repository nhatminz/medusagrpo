#!/usr/bin/env bash
set -euo pipefail
MODEL_KEY=llama31_8b
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/launch/pretrain_model.sh" "$@"
