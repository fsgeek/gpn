"""
Relational Trainer: Trains RelationalWeaver to generate X > Y displays

Pedagogical training approach:
- Judge validates: P(X > Y) from generated [X][>][Y] images
- Loss: Maximize judge's validity probability + digit classification accuracy
- No adversarial loss (pure pedagogical)

This tests whether pedagogical training can learn context-dependent representations
where position-invariant primitives are insufficient.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

from src.models.relational_weaver import RelationalWeaver, create_relational_weaver
from src.models.relation_judge import RelationJudge, create_relation_judge


class RelationalTrainer:
    """
    Trainer for relational generation task: X > Y

    Uses pure pedagogical signal from RelationJudge.
    """

    def __init__(
        self,
        weaver: RelationalWeaver,
        judge: RelationJudge,
        latent_dim: int = 64,
        num_relations: int = 10,  # Number of valid relations in curriculum
        device: torch.device = torch.device("cpu"),
        lr: float = 0.0002,
    ):
        self.weaver = weaver.to(device)
        self.judge = judge.to(device)
        self.latent_dim = latent_dim
        self.num_relations = num_relations
        self.device = device

        # Optimizer for weaver only (judge is frozen)
        self.optimizer = torch.optim.Adam(
            self.weaver.parameters(),
            lr=lr,
            betas=(0.5, 0.999),
        )

        # Loss weights
        self.validity_weight = 1.0  # P(X > Y) maximization
        self.digit_weight = 0.1     # Digit classification accuracy

    def train_step(self, batch_size: int = 64) -> dict[str, float]:
        """
        Single training step.

        Args:
            batch_size: Batch size

        Returns:
            Dictionary of metrics
        """
        self.weaver.train()
        self.judge.eval()

        # Sample random relation labels
        relation_labels = torch.randint(
            0, self.num_relations, (batch_size,), device=self.device
        )

        # Decompose into X and Y for ground truth
        x_labels, y_labels = self._decode_labels(relation_labels)

        # Generate relational images
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        generated_images = self.weaver(z, relation_labels)

        # Judge validation
        x_logits, y_logits, valid_prob = self.judge(generated_images)

        # Loss: Maximize validity + digit classification
        # Validity loss: want valid_prob → 1 (X > Y should hold)
        validity_loss = -valid_prob.mean()  # Maximize probability

        # Digit classification loss
        x_class_loss = F.cross_entropy(x_logits, x_labels)
        y_class_loss = F.cross_entropy(y_logits, y_labels)
        digit_loss = (x_class_loss + y_class_loss) / 2

        # Total loss
        total_loss = (
            self.validity_weight * validity_loss +
            self.digit_weight * digit_loss
        )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Compute metrics
        with torch.no_grad():
            x_acc = (x_logits.argmax(dim=1) == x_labels).float().mean()
            y_acc = (y_logits.argmax(dim=1) == y_labels).float().mean()
            avg_validity = valid_prob.mean()

            # Ground truth relation accuracy
            valid_gt = (x_labels > y_labels).float()
            valid_pred = (valid_prob > 0.5).float()
            relation_acc = (valid_pred == valid_gt).float().mean()

        return {
            "loss": total_loss.item(),
            "validity_loss": validity_loss.item(),
            "digit_loss": digit_loss.item(),
            "x_accuracy": x_acc.item(),
            "y_accuracy": y_acc.item(),
            "avg_validity": avg_validity.item(),
            "relation_accuracy": relation_acc.item(),
        }

    def evaluate(
        self,
        num_samples: int = 1000,
        curriculum_mode: bool = True,
    ) -> dict[str, float]:
        """
        Evaluate on curriculum or transfer set.

        Args:
            num_samples: Number of samples to evaluate
            curriculum_mode: If True, use curriculum relations [0,4]
                           If False, use transfer relations [5,9]

        Returns:
            Dictionary of metrics
        """
        self.weaver.eval()
        self.judge.eval()

        # Determine relation range
        if curriculum_mode:
            # Curriculum: [0,4] → 10 valid relations
            num_relations = 10
        else:
            # Transfer: [5,9] → 10 valid relations
            # Map to indices 10-19 (need to handle in label decode)
            num_relations = 10

        all_x_acc = []
        all_y_acc = []
        all_validity = []
        all_relation_acc = []

        batch_size = 64
        for i in range(0, num_samples, batch_size):
            batch_sz = min(batch_size, num_samples - i)

            with torch.no_grad():
                # Sample relations
                if curriculum_mode:
                    relation_labels = torch.randint(
                        0, num_relations, (batch_sz,), device=self.device
                    )
                else:
                    # Transfer mode: generate labels for [5,9] range
                    # Use similar encoding but offset
                    relation_labels = torch.randint(
                        0, num_relations, (batch_sz,), device=self.device
                    ) + 10  # Offset for transfer range

                x_labels, y_labels = self._decode_labels(
                    relation_labels, transfer=not curriculum_mode
                )

                z = torch.randn(batch_sz, self.latent_dim, device=self.device)
                generated_images = self.weaver(z, relation_labels)

                x_logits, y_logits, valid_prob = self.judge(generated_images)

                x_acc = (x_logits.argmax(dim=1) == x_labels).float().mean()
                y_acc = (y_logits.argmax(dim=1) == y_labels).float().mean()

                valid_gt = (x_labels > y_labels).float()
                valid_pred = (valid_prob > 0.5).float()
                relation_acc = (valid_pred == valid_gt).float().mean()

                all_x_acc.append(x_acc.item())
                all_y_acc.append(y_acc.item())
                all_validity.append(valid_prob.mean().item())
                all_relation_acc.append(relation_acc.item())

        return {
            "x_accuracy": sum(all_x_acc) / len(all_x_acc),
            "y_accuracy": sum(all_y_acc) / len(all_y_acc),
            "avg_validity": sum(all_validity) / len(all_validity),
            "relation_accuracy": sum(all_relation_acc) / len(all_relation_acc),
        }

    def _decode_labels(
        self,
        labels: torch.Tensor,
        transfer: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode relation labels to (X, Y) digit pairs.

        Args:
            labels: Relation indices
            transfer: If True, decode for transfer range [5,9]

        Returns:
            Tuple of (x_labels, y_labels)
        """
        if transfer:
            # Transfer range [5,9]: offset by 5
            # Relations: (6,5), (7,5), (7,6), (8,5), ..., (9,8)
            offset = 5
        else:
            # Curriculum range [0,4]
            # Relations: (1,0), (2,0), (2,1), ..., (4,3)
            offset = 0

        device = labels.device
        B = labels.size(0)

        x_digit = torch.zeros(B, dtype=torch.long, device=device)
        y_digit = torch.zeros(B, dtype=torch.long, device=device)

        for i in range(B):
            # Decode using same logic as in RelationalWeaver
            idx = labels[i].item()

            # Adjust for transfer offset
            if transfer:
                idx = idx - 10  # Remove transfer offset

            # Find X such that X(X-1)/2 <= idx < X(X+1)/2
            x = 1
            cumsum = 0
            while cumsum + x <= idx:
                cumsum += x
                x += 1

            y = idx - cumsum

            # Apply digit offset
            x_digit[i] = x + offset
            y_digit[i] = y + offset

        return x_digit, y_digit

    def save_checkpoint(
        self,
        path: str,
        step: int,
        metrics: Optional[dict] = None,
    ):
        """Save training checkpoint."""
        checkpoint = {
            "step": step,
            "models": {
                "weaver": self.weaver.state_dict(),
            },
            "optimizer": self.optimizer.state_dict(),
            "metrics": metrics or {},
            "config": {
                "latent_dim": self.latent_dim,
                "num_relations": self.num_relations,
            },
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)


def create_relational_trainer(
    single_digit_checkpoint: str,
    judge_checkpoint: str,
    latent_dim: int = 64,
    num_relations: int = 10,
    device: torch.device | None = None,
    freeze_digits: bool = False,
) -> RelationalTrainer:
    """
    Create RelationalTrainer from checkpoints.

    Args:
        single_digit_checkpoint: Path to single-digit Weaver checkpoint
        judge_checkpoint: Path to RelationJudge checkpoint
        latent_dim: Latent dimension
        num_relations: Number of valid relations in curriculum
        device: Device to use
        freeze_digits: Whether to freeze single-digit weights

    Returns:
        RelationalTrainer instance
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create weaver
    weaver = create_relational_weaver(
        single_digit_checkpoint=single_digit_checkpoint,
        latent_dim=latent_dim,
        device=device,
        freeze_digits=freeze_digits,
    )

    # Create judge (frozen)
    judge = create_relation_judge(
        checkpoint_path=judge_checkpoint,
        device=device,
        freeze=True,
    )

    return RelationalTrainer(
        weaver=weaver,
        judge=judge,
        latent_dim=latent_dim,
        num_relations=num_relations,
        device=device,
    )


class RelationalTrainerHoldout(RelationalTrainer):
    """
    Trainer for Phase 1.6: Hold-out pair testing.

    Trains on all digits [0-9] but excludes specific relation pairs.
    Tests compositional generalization to unseen combinations.
    """

    def __init__(
        self,
        weaver: RelationalWeaver,
        judge: RelationJudge,
        holdout_pairs: list[tuple[int, int]],
        latent_dim: int = 64,
        device: torch.device = torch.device("cpu"),
        lr: float = 0.0002,
    ):
        # Count total valid relations minus hold-outs
        num_relations = 45 - len(holdout_pairs)  # 45 total X>Y pairs for [0-9]

        super().__init__(
            weaver=weaver,
            judge=judge,
            latent_dim=latent_dim,
            num_relations=num_relations,
            device=device,
            lr=lr,
        )

        self.holdout_pairs = holdout_pairs

        # Build training and holdout relation lists
        self.training_relations = []
        self.holdout_relations = []

        for x in range(10):
            for y in range(x):
                if (x, y) in holdout_pairs:
                    self.holdout_relations.append((x, y))
                else:
                    self.training_relations.append((x, y))

    def train_step(self, batch_size: int = 64) -> dict[str, float]:
        """Training step using only training relations."""
        self.weaver.train()
        self.judge.eval()

        # Sample random training relations
        indices = torch.randint(
            0, len(self.training_relations), (batch_size,), device=self.device
        )

        x_labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        y_labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for i, idx in enumerate(indices):
            x, y = self.training_relations[idx.item()]
            x_labels[i] = x
            y_labels[i] = y

        # Generate relational images using X,Y pairs directly
        z = torch.randn(batch_size, self.latent_dim, device=self.device)

        # Use weaver's direct generation (bypass label encoding)
        generated_images = self.weaver.generate_from_digits(z, x_labels, y_labels)

        # Judge validation
        x_logits, y_logits, valid_prob = self.judge(generated_images)

        # Loss: Maximize validity + digit classification
        validity_loss = -valid_prob.mean()
        x_class_loss = F.cross_entropy(x_logits, x_labels)
        y_class_loss = F.cross_entropy(y_logits, y_labels)
        digit_loss = (x_class_loss + y_class_loss) / 2

        total_loss = (
            self.validity_weight * validity_loss +
            self.digit_weight * digit_loss
        )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Compute metrics
        with torch.no_grad():
            x_acc = (x_logits.argmax(dim=1) == x_labels).float().mean()
            y_acc = (y_logits.argmax(dim=1) == y_labels).float().mean()
            avg_validity = valid_prob.mean()

            valid_gt = (x_labels > y_labels).float()
            valid_pred = (valid_prob > 0.5).float()
            relation_acc = (valid_pred == valid_gt).float().mean()

        return {
            "loss": total_loss.item(),
            "validity_loss": validity_loss.item(),
            "digit_loss": digit_loss.item(),
            "x_accuracy": x_acc.item(),
            "y_accuracy": y_acc.item(),
            "avg_validity": avg_validity.item(),
            "relation_accuracy": relation_acc.item(),
        }

    def evaluate(
        self,
        num_samples: int = 1000,
        holdout_mode: bool = False,
    ) -> dict[str, float]:
        """
        Evaluate on training or holdout relations.

        Args:
            num_samples: Number of samples to evaluate
            holdout_mode: If True, test on holdout pairs. If False, test on training pairs.

        Returns:
            Dictionary of metrics
        """
        self.weaver.eval()
        self.judge.eval()

        relations = self.holdout_relations if holdout_mode else self.training_relations

        all_x_acc = []
        all_y_acc = []
        all_validity = []
        all_relation_acc = []

        batch_size = 64
        for i in range(0, num_samples, batch_size):
            batch_sz = min(batch_size, num_samples - i)

            with torch.no_grad():
                # Sample relations
                indices = torch.randint(
                    0, len(relations), (batch_sz,), device=self.device
                )

                x_labels = torch.zeros(batch_sz, dtype=torch.long, device=self.device)
                y_labels = torch.zeros(batch_sz, dtype=torch.long, device=self.device)

                for j, idx in enumerate(indices):
                    x, y = relations[idx.item()]
                    x_labels[j] = x
                    y_labels[j] = y

                z = torch.randn(batch_sz, self.latent_dim, device=self.device)
                generated_images = self.weaver.generate_from_digits(z, x_labels, y_labels)

                x_logits, y_logits, valid_prob = self.judge(generated_images)

                x_acc = (x_logits.argmax(dim=1) == x_labels).float().mean()
                y_acc = (y_logits.argmax(dim=1) == y_labels).float().mean()

                valid_gt = (x_labels > y_labels).float()
                valid_pred = (valid_prob > 0.5).float()
                relation_acc = (valid_pred == valid_gt).float().mean()

                all_x_acc.append(x_acc.item())
                all_y_acc.append(y_acc.item())
                all_validity.append(valid_prob.mean().item())
                all_relation_acc.append(relation_acc.item())

        return {
            "x_accuracy": sum(all_x_acc) / len(all_x_acc),
            "y_accuracy": sum(all_y_acc) / len(all_y_acc),
            "avg_validity": sum(all_validity) / len(all_validity),
            "relation_accuracy": sum(all_relation_acc) / len(all_relation_acc),
        }


def create_relational_trainer_holdout(
    single_digit_checkpoint: str,
    judge_checkpoint: str,
    latent_dim: int = 64,
    holdout_pairs: list[tuple[int, int]] | None = None,
    device: torch.device | None = None,
    freeze_digits: bool = True,
) -> RelationalTrainerHoldout:
    """
    Create RelationalTrainerHoldout for Phase 1.6.

    Args:
        single_digit_checkpoint: Path to single-digit Weaver checkpoint
        judge_checkpoint: Path to RelationJudge checkpoint
        latent_dim: Latent dimension
        holdout_pairs: List of (x, y) pairs to exclude from training
        device: Device to use
        freeze_digits: Whether to freeze single-digit weights

    Returns:
        RelationalTrainerHoldout instance
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    holdout_pairs = holdout_pairs or []

    # Create weaver
    weaver = create_relational_weaver(
        single_digit_checkpoint=single_digit_checkpoint,
        latent_dim=latent_dim,
        device=device,
        freeze_digits=freeze_digits,
    )

    # Create judge (frozen)
    judge = create_relation_judge(
        checkpoint_path=judge_checkpoint,
        device=device,
        freeze=True,
    )

    return RelationalTrainerHoldout(
        weaver=weaver,
        judge=judge,
        holdout_pairs=holdout_pairs,
        latent_dim=latent_dim,
        device=device,
    )
