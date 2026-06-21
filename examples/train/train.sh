set -x
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
task_name="searchqa"

cd AgentGym-RL
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_BASE_URL=https://api.bandw.top

env_server_url="http://127.0.0.1:36001"

# start training
# wandb login xxx
SWANLAB_API_KEY="Your Key"
swanlab login --relogin --api-key $SWANLAB_API_KEY

agent_model_path="Qwen/Qwen2.5-7B-Instruct"
encoder_path="Qwen/Qwen3-Embedding-0.6B"

kl_coef=0.001
policy_learning_rate=1e-6
rollout_sample_num=8
train_batch_size=64
ppo_mini_batch_size=32
ppo_micro_batch_size_per_gpu=1
ppo_inner_epochs=1
seed=100

total_epoches=100

model_save_dir="saves"
mkdir -p ${model_save_dir}
exp_name="searchqa-qwen7b"
model_save_path=${model_save_dir}/${exp_name}

mkdir -p ${model_save_path}

HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m verl.agent_trainer.main_ppo  \
    algorithm.adv_estimator=ot_grpo \
    algorithm.rounds_ctrl.type=fixed \
    algorithm.rounds_ctrl.rounds=4 \
    data.seed=${seed} \
    data.train_file=../data/train_data.jsonl \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    actor_rollout_ref.agentgym.task_name=${task_name} \
    actor_rollout_ref.agentgym.env_addr=${env_server_url} \
    actor_rollout_ref.agentgym.timeout=600 \
    actor_rollout_ref.model.path=${agent_model_path} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${rollout_sample_num} \
    actor_rollout_ref.rollout.max_tokens=512 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=9216 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.encoder.enable_encoder=true \
    actor_rollout_ref.encoder.offload=true \
    actor_rollout_ref.encoder.strategy=off-policy \
    actor_rollout_ref.encoder.max_off_num=8 \
    actor_rollout_ref.encoder.step_adv_coef=0.5 \
    actor_rollout_ref.encoder.step_adv_coef_min=0.5 \
    actor_rollout_ref.encoder.encoder_path=${encoder_path} \
    actor_rollout_ref.actor.ppo_epochs=${ppo_inner_epochs} \
    actor_rollout_ref.actor.optim.lr=${policy_learning_rate} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.rollout_log_dir=${model_save_path}/executer_logs \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    trainer.default_local_dir=${model_save_path} \
    trainer.project_name=agentic_rl \
    trainer.experiment_name=${exp_name} \
    trainer.save_freq=100 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=${total_epoches}
status=$?
exit $status
