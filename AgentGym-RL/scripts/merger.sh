#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/mnt/data/kw/jzs/Agent/AgentGym-RL/AgentGym-RL/saves/textcraftv2_qwen3b}"
SCRIPT="/mnt/data/kw/jzs/Agent/AgentGym-RL/AgentGym-RL/scripts/model_merger.py"

# 按 global_step 数字顺序遍历
for step_dir in $(ls -d "${ROOT_DIR}"/global_step_* 2>/dev/null | sort -V); do
  actor_dir="${step_dir}/actor"
  if [ -d "${actor_dir}" ]; then
    echo "[RUN] python ${SCRIPT} --local_dir ${actor_dir}"
    python "${SCRIPT}" --local_dir "${actor_dir}"
  else
    echo "[SKIP] ${step_dir} (missing actor/)"
  fi
done
