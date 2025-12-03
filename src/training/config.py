"""
Training configuration for GPN-1.

Provides typed configuration dataclass per data-model.md.

Exports:
    - TrainingConfig: Complete training configuration
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import yaml

import torch


@dataclass
class WeaverConfig:
    """Weaver (generator) architecture configuration."""

    hidden_dims: list[int] = field(default_factory=lambda: [256, 512, 1024])
    use_batch_norm: bool = True
    v_pred_hidden: int = 128
    v_pred_dim: int = 16  # Value prediction output dimension


@dataclass
class WitnessConfig:
    """Witness (classifier) architecture configuration."""

    hidden_dims: list[int] = field(default_factory=lambda: [1024, 512, 256])
    use_batch_norm: bool = True
    v_seen_hidden: int = 128
    v_seen_dim: int = 16  # Value estimation output dimension
    dropout: float = 0.3


@dataclass
class JudgeConfig:
    """Judge (frozen classifier) configuration."""

    hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    checkpoint_path: Optional[str] = None


@dataclass
class DataConfig:
    """Data loading configuration."""

    dataset: str = "mnist"
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    type: str = "adam"
    lr: float = 2e-4
    betas: tuple[float, float] = (0.5, 0.999)
    weight_decay: float = 0.0


@dataclass
class LossWeights:
    """Loss weights for a training phase."""

    grounding: float = 1.0
    alignment: float = 0.1
    empowerment: float = 0.0


@dataclass
class EmpowermentConfig:
    """Empowerment (Goldilocks) loss configuration."""

    target_kl: float = 0.5
    tolerance: float = 0.1


@dataclass
class EMAConfig:
    """EMA state tracking configuration."""

    decay: float = 0.99
    variance_threshold: float = 1e-6  # Stagnation detection
    window_size: int = 100  # Stagnation detection window


@dataclass
class CollusionConfig:
    """Collusion detection configuration (T032a)."""

    enabled: bool = True
    alignment_drop_threshold: float = 0.1
    quality_stagnation_threshold: float = 0.01


@dataclass
class LoggingConfig:
    """Logging configuration."""

    log_dir: str = "experiments"
    log_interval: int = 100
    sample_interval: int = 500
    num_samples: int = 64


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""

    checkpoint_dir: str = "checkpoints"
    save_interval: int = 1000
    keep_last_n: int = 3
    save_on_phase_transition: bool = True


@dataclass
class TrainingConfig:
    """
    Complete training configuration for GPN-1.

    Per data-model.md: provides typed configuration for all training parameters.

    Attributes:
        seed: Random seed for reproducibility
        total_steps: Total training steps
        phase1_steps: Steps for Phase 1 (Scaffolding)
        phase2_steps: Steps for Phase 2 end (Relationship)
        device: Training device ("auto", "cpu", or "cuda")

    Example:
        >>> config = TrainingConfig.from_yaml("configs/gpn1_default.yaml")
        >>> print(config.phase1_steps)  # 5000
    """

    # Core settings
    seed: int = 42
    total_steps: int = 15000
    phase1_steps: int = 5000
    phase2_steps: int = 10000  # Phase 2 ends at this step
    device: str = "auto"

    # Model dimensions
    latent_dim: int = 64
    num_classes: int = 10
    image_channels: int = 1
    image_size: int = 28

    # Component configs
    weaver: WeaverConfig = field(default_factory=WeaverConfig)
    witness: WitnessConfig = field(default_factory=WitnessConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # Phase-specific loss weights
    phase1_weights: LossWeights = field(
        default_factory=lambda: LossWeights(grounding=1.0, alignment=0.1, empowerment=0.0)
    )
    phase2_weights: LossWeights = field(
        default_factory=lambda: LossWeights(grounding=1.0, alignment=0.5, empowerment=0.3)
    )
    phase3_weights: LossWeights = field(
        default_factory=lambda: LossWeights(grounding=0.0, alignment=0.0, empowerment=0.0)
    )

    # Additional configs
    empowerment: EmpowermentConfig = field(default_factory=EmpowermentConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    collusion: CollusionConfig = field(default_factory=CollusionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)

    def get_device(self) -> torch.device:
        """Get resolved torch device."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def get_phase(self, step: int) -> int:
        """Get training phase for a given step."""
        if step < self.phase1_steps:
            return 1
        elif step < self.phase2_steps:
            return 2
        else:
            return 3

    def get_loss_weights(self, phase: int) -> LossWeights:
        """Get loss weights for a given phase."""
        if phase == 1:
            return self.phase1_weights
        elif phase == 2:
            return self.phase2_weights
        else:
            return self.phase3_weights

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            TrainingConfig instance
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        """
        Create configuration from dictionary.

        Args:
            data: Configuration dictionary (e.g., from YAML)

        Returns:
            TrainingConfig instance
        """
        # Extract nested configs
        weaver_data = data.get("model", {}).get("weaver", {})
        witness_data = data.get("model", {}).get("witness", {})
        judge_data = data.get("model", {}).get("judge", {})
        data_config = data.get("data", {})
        optimizer_data = data.get("training", {}).get("optimizer", {})
        training_data = data.get("training", {})
        losses_data = data.get("losses", {})
        ema_data = data.get("ema", {})
        collusion_data = data.get("collusion", {})
        logging_data = data.get("logging", {})
        checkpoint_data = data.get("checkpointing", {})

        return cls(
            seed=data.get("seed", 42),
            total_steps=training_data.get("total_steps", 15000),
            phase1_steps=training_data.get("phase1_steps", 5000),
            phase2_steps=training_data.get("phase2_steps", 10000),
            device=data.get("device", "auto"),
            latent_dim=data.get("model", {}).get("latent_dim", 64),
            num_classes=data.get("model", {}).get("num_classes", 10),
            image_channels=data.get("model", {}).get("image_channels", 1),
            image_size=data.get("model", {}).get("image_size", 28),
            weaver=WeaverConfig(
                hidden_dims=weaver_data.get("hidden_dims", [256, 512, 1024]),
                use_batch_norm=weaver_data.get("use_batch_norm", True),
                v_pred_hidden=weaver_data.get("v_pred_hidden", 128),
                v_pred_dim=weaver_data.get("v_pred_dim", 16),
            ),
            witness=WitnessConfig(
                hidden_dims=witness_data.get("hidden_dims", [1024, 512, 256]),
                use_batch_norm=witness_data.get("use_batch_norm", True),
                v_seen_hidden=witness_data.get("v_seen_hidden", 128),
                v_seen_dim=witness_data.get("v_seen_dim", 16),
                dropout=witness_data.get("dropout", 0.3),
            ),
            judge=JudgeConfig(
                hidden_dims=judge_data.get("hidden_dims", [512, 256]),
                checkpoint_path=judge_data.get("checkpoint_path"),
            ),
            data=DataConfig(
                dataset=data_config.get("dataset", "mnist"),
                batch_size=data_config.get("batch_size", 64),
                num_workers=data_config.get("num_workers", 4),
                pin_memory=data_config.get("pin_memory", True),
            ),
            optimizer=OptimizerConfig(
                type=optimizer_data.get("type", "adam"),
                lr=float(optimizer_data.get("lr", 2e-4)),
                betas=tuple(optimizer_data.get("betas", [0.5, 0.999])),
                weight_decay=optimizer_data.get("weight_decay", 0.0),
            ),
            phase1_weights=LossWeights(**losses_data.get("phase1", {})),
            phase2_weights=LossWeights(**losses_data.get("phase2", {})),
            phase3_weights=LossWeights(**losses_data.get("phase3", {})),
            empowerment=EmpowermentConfig(
                target_kl=losses_data.get("empowerment", {}).get("target_kl", 0.5),
                tolerance=losses_data.get("empowerment", {}).get("tolerance", 0.1),
            ),
            ema=EMAConfig(
                decay=float(ema_data.get("decay", 0.99)),
                variance_threshold=float(ema_data.get("stagnation", {}).get("variance_threshold", 1e-6)),
                window_size=int(ema_data.get("stagnation", {}).get("window_size", 100)),
            ),
            collusion=CollusionConfig(
                enabled=collusion_data.get("enabled", True),
                alignment_drop_threshold=float(collusion_data.get("phase2", {}).get(
                    "alignment_drop_threshold", 0.1
                )),
                quality_stagnation_threshold=float(collusion_data.get("phase2", {}).get(
                    "quality_stagnation_threshold", 0.01
                )),
            ),
            logging=LoggingConfig(
                log_dir=logging_data.get("log_dir", "experiments"),
                log_interval=logging_data.get("log_interval", 100),
                sample_interval=logging_data.get("sample_interval", 500),
                num_samples=logging_data.get("num_samples", 64),
            ),
            checkpointing=CheckpointConfig(
                checkpoint_dir=checkpoint_data.get("checkpoint_dir", "checkpoints"),
                save_interval=checkpoint_data.get("save_interval", 1000),
                keep_last_n=checkpoint_data.get("keep_last_n", 3),
                save_on_phase_transition=checkpoint_data.get("save_on_phase_transition", True),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "seed": self.seed,
            "device": self.device,
            "model": {
                "latent_dim": self.latent_dim,
                "num_classes": self.num_classes,
                "image_channels": self.image_channels,
                "image_size": self.image_size,
                "weaver": {
                    "hidden_dims": self.weaver.hidden_dims,
                    "use_batch_norm": self.weaver.use_batch_norm,
                    "v_pred_hidden": self.weaver.v_pred_hidden,
                    "v_pred_dim": self.weaver.v_pred_dim,
                },
                "witness": {
                    "hidden_dims": self.witness.hidden_dims,
                    "use_batch_norm": self.witness.use_batch_norm,
                    "v_seen_hidden": self.witness.v_seen_hidden,
                    "v_seen_dim": self.witness.v_seen_dim,
                    "dropout": self.witness.dropout,
                },
                "judge": {
                    "hidden_dims": self.judge.hidden_dims,
                    "checkpoint_path": self.judge.checkpoint_path,
                },
            },
            "data": {
                "dataset": self.data.dataset,
                "batch_size": self.data.batch_size,
                "num_workers": self.data.num_workers,
                "pin_memory": self.data.pin_memory,
            },
            "training": {
                "total_steps": self.total_steps,
                "phase1_steps": self.phase1_steps,
                "phase2_steps": self.phase2_steps,
                "optimizer": {
                    "type": self.optimizer.type,
                    "lr": self.optimizer.lr,
                    "betas": list(self.optimizer.betas),
                    "weight_decay": self.optimizer.weight_decay,
                },
            },
            "losses": {
                "phase1": {
                    "grounding": self.phase1_weights.grounding,
                    "alignment": self.phase1_weights.alignment,
                    "empowerment": self.phase1_weights.empowerment,
                },
                "phase2": {
                    "grounding": self.phase2_weights.grounding,
                    "alignment": self.phase2_weights.alignment,
                    "empowerment": self.phase2_weights.empowerment,
                },
                "phase3": {
                    "grounding": self.phase3_weights.grounding,
                    "alignment": self.phase3_weights.alignment,
                    "empowerment": self.phase3_weights.empowerment,
                },
                "empowerment": {
                    "target_kl": self.empowerment.target_kl,
                    "tolerance": self.empowerment.tolerance,
                },
            },
            "ema": {
                "decay": self.ema.decay,
                "stagnation": {
                    "variance_threshold": self.ema.variance_threshold,
                    "window_size": self.ema.window_size,
                },
            },
            "collusion": {
                "enabled": self.collusion.enabled,
                "alignment_drop_threshold": self.collusion.alignment_drop_threshold,
                "quality_stagnation_threshold": self.collusion.quality_stagnation_threshold,
            },
            "logging": {
                "log_dir": self.logging.log_dir,
                "log_interval": self.logging.log_interval,
                "sample_interval": self.logging.sample_interval,
                "num_samples": self.logging.num_samples,
            },
            "checkpointing": {
                "checkpoint_dir": self.checkpointing.checkpoint_dir,
                "save_interval": self.checkpointing.save_interval,
                "keep_last_n": self.checkpointing.keep_last_n,
                "save_on_phase_transition": self.checkpointing.save_on_phase_transition,
            },
        }
