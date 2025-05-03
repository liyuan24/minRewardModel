from dataclasses import dataclass

@dataclass
class TrainingConfig:
    device: str = "cuda"
    learning_rate: float = 5e-4
    min_learning_rate: float = 5e-5
    eval_interval: int = 10
    dtype: str = "bfloat16"
    wandb_log: bool = True
    wandb_project: str = "OutputRewardModel"
    wandb_run_name: str = "OutputRewardModel"
    adamw_use_fused: bool = True
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    warmup_iters: int = 500
    lr_decay_iters: int = 1000
    decay_lr: bool = True
    out_dir: str = "output"
    resume: bool = False
    checkpoint_path_prefix: str = "orm_model"
    resume_checkpoint_path: str = None
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.95
    adamw_weight_decay: float = 0.1
    use_eight_bit_optimizer: bool = True
    grad_clip: float = 1.0
    dataset_name: str = "RLHFlow/Mistral-ORM-Data"
    base_model_name: str = "Qwen/Qwen3-1.7B-Base"
    validation_data_size: int = 500
    test_data_size: int = 500
    epochs: int = 10