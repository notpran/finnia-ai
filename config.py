"""Central configuration for the mini-GPT project."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    n_layers: int = 6
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048
    seq_len: int = 512
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    rope_theta: Optional[float] = None  # Placeholder for rotary embeddings if desired later.


@dataclass
class TrainingConfig:
    batch_size_per_gpu: int = 4
    grad_accum_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 5000
    total_steps: int = 20000
    mixed_precision: str = "bf16"  # options: "fp32", "fp16", "bf16"
    max_grad_norm: float = 1.0
    log_interval: int = 50
    eval_interval: int = 500
    sample_interval: int = 1000
    checkpoint_interval: int = 2000
    resume_from: Optional[str] = None


@dataclass
class DataConfig:
    seq_len: int = 512
    tokenizer_path: str = "artifacts/tokenizer.json"
    dataset_cache_dir: str = "artifacts/hf_cache"
    packed_dataset_path: Optional[str] = None
    max_samples_per_source: Optional[int] = 20000
    num_workers: int = 2


@dataclass
class RuntimeConfig:
    device: str = "cuda"
    seed: int = 42
    num_devices: int = 1
    use_compile: bool = False
    use_accelerate: bool = False
    output_dir: str = "artifacts"


@dataclass
class EvalConfig:
    eval_batch_size: int = 4
    max_eval_batches: int = 50


@dataclass
class ProjectConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def override(self, **kwargs) -> "ProjectConfig":
        """Return a copy with updated nested attributes."""
        cfg = ProjectConfig(
            model=ModelConfig(**vars(self.model)),
            training=TrainingConfig(**vars(self.training)),
            data=DataConfig(**vars(self.data)),
            runtime=RuntimeConfig(**vars(self.runtime)),
            eval=EvalConfig(**vars(self.eval)),
        )
        for dotted_key, value in kwargs.items():
            section_name, _, attr = dotted_key.partition(".")
            if not attr:
                raise ValueError(f"Expected dotted key like 'model.d_model', got {dotted_key}")
            section = getattr(cfg, section_name)
            setattr(section, attr, value)
        return cfg


def load_default_config() -> ProjectConfig:
    return ProjectConfig()
