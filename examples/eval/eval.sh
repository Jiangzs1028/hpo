#!/usr/bin/env bash
set -euo pipefail
set -x

export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS

task_name="searchqa"
env_server_url="http://127.0.0.1:36001"

sample_num=1
max_rounds=4

cd AgentGym-RL
export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_DIRS=(
Qwen/Qwen2.5-7B-Instruct
)


# eval data
DATA_PATH="../data/test_data.json"

LOG_ROOT="infer_logs/batch_$(date +%Y%m%d_%H%M%S)"


for ckpt_path in "${MODEL_DIRS[@]}"; do
  if [[ -d "${ckpt_path}/huggingface" ]]; then
    model_path="${ckpt_path}/huggingface"
  else
    model_path="${ckpt_path}"
  fi
  step_dir="$(basename "$(dirname "$(dirname "${model_path}")")")"
  exp_dir="$(basename "$(dirname "$(dirname "$(dirname "${model_path}")")")")"

  if [[ ! "${step_dir}" =~ ^global_step_ ]]; then
    echo "[WARN] Failed to parse step_dir from model_path=${model_path}, fallback to full name"
    step_dir="$(basename "${model_path}")"
  fi


  model_tag="${exp_dir}/${step_dir}"

  this_log_dir="${LOG_ROOT}/${model_tag}"
  mkdir -p "${this_log_dir}"

  run_log="${this_log_dir}/run.log"
  metrics_log="${this_log_dir}/metrics.txt"

  echo "============================================================"
  echo "[RUN] model_path=${model_path}"
  echo "[TAG] model_tag=${model_tag}"
  echo "[LOG] log_dir=${this_log_dir}"
  echo "[OUT] run_log=${run_log}"
  echo "[OUT] metrics_log=${metrics_log}"
  echo "============================================================"


  : > "${run_log}"
  : > "${metrics_log}"

  HYDRA_FULL_ERROR=1 python3 -m verl.agent_trainer.main_generation \
    data.path="${DATA_PATH}" \
    data.max_prompt_length=1024 \
    data.max_response_length=8196 \
    data.n_samples="${sample_num}" \
    data.batch_size=1024 \
    agentgym.task_name="${task_name}" \
    agentgym.env_addr="${env_server_url}" \
    agentgym.max_rounds="${max_rounds}" \
    agentgym.timeout=500 \
    model.path="${model_path}" \
    rollout.gpu_memory_utilization=0.85 \
    rollout.temperature=0 \
    rollout.max_model_len=32768 \
    rollout.max_tokens=512 \
    rollout.tensor_model_parallel_size=1 \
    rollout.rollout_log_dir="${this_log_dir}" \
    trainer.n_gpus_per_node=8 \
    2>&1 | tee "${run_log}" | awk '
      BEGIN {capture=0}
      /============Total Task Evaluation============/ {capture=1}
      capture==1 {print}
      /============ End ============/ {if (capture==1) {capture=0}}
    ' | tee "${metrics_log}"

  if [[ ! -s "${metrics_log}" ]]; then
    echo "[WARN] metrics.txt is empty. Pattern may not match log format." | tee -a "${run_log}"
  fi

  echo "[DONE] ${model_tag}"
done

echo "All runs finished. Logs under: ${LOG_ROOT}"
